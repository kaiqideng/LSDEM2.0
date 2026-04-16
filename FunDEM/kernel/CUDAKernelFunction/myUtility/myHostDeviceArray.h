#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>

inline void check_cuda_error(cudaError_t result, const char* func, const char* file, int line)
{
    if (result != cudaSuccess) 
    {
        std::string msg = std::string("CUDA Error at ") + file + ":" + std::to_string(line) +
        " / " + func + ": " + cudaGetErrorString(result);
        // Throw exception to allow destructors (RAII) to clean up
        throw std::runtime_error(msg);
    }
}

#define CUDA_CHECK(val) check_cuda_error((val), #val, __FILE__, __LINE__)

#ifndef NDEBUG
#include <iomanip>
#include <iostream>
template <typename T>
void debug_dump_device_array(const T* d_ptr, std::size_t n, const char* name, cudaStream_t stream = 0)
{
    if (n == 0) return;

    if (stream == 0) 
    {
        CUDA_CHECK(cudaDeviceSynchronize());
    } 
    else 
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<T> h_buf(n);

    CUDA_CHECK(cudaMemcpy(h_buf.data(), d_ptr,
    n * sizeof(T),
    cudaMemcpyDeviceToHost));

    std::cout << "[DEBUG] " << name << " (first " << n << " values):\n";
    for (std::size_t i = 0; i < n; ++i) 
    {
        std::cout << "  [" << i << "] = " << h_buf[i] << "\n";
    }
}

template <>
inline void debug_dump_device_array<double3>(const double3* d_ptr, std::size_t n, const char* name, cudaStream_t stream)
{
    if (n == 0) return;

    if (stream == 0) 
    {
        CUDA_CHECK(cudaDeviceSynchronize());
    } 
    else 
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<double3> h(n);
    cudaMemcpyAsync(h.data(), d_ptr, n * sizeof(double3), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::cout << name << ":\n";

    std::ios old_state(nullptr);
    old_state.copyfmt(std::cout);

    std::cout << std::scientific << std::setprecision(3);

    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& v = h[i];
        std::cout << "  [" << i << "] = ("
        << v.x << ", "
        << v.y << ", "
        << v.z << ")\n";
    }

    std::cout.copyfmt(old_state);
}
#endif

template <typename T>
struct HostDeviceArray1D
{
private:
    std::vector<T> h_data;     // host data
    size_t d_size {0};         // number of elements on device

public:
    T* d_ptr {nullptr};        // device pointer

private:
    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------
    // Best-effort sync free for destructor / move-assignment.
    // Do NOT CUDA_CHECK here because CUDA runtime may already be shutting down.
    void releaseDeviceNoThrow_() noexcept
    {
        if (d_ptr)
        {
            (void)cudaFree(d_ptr); // ignore error on teardown
            d_ptr = nullptr;
        }
        d_size = 0;
    }

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    HostDeviceArray1D() = default;

    ~HostDeviceArray1D()
    {
        releaseDeviceNoThrow_();
    }

    HostDeviceArray1D(const HostDeviceArray1D&) = delete;
    HostDeviceArray1D& operator=(const HostDeviceArray1D&) = delete;

    HostDeviceArray1D(HostDeviceArray1D&& other) noexcept
    {
        *this = std::move(other);
    }

    HostDeviceArray1D& operator=(HostDeviceArray1D&& other) noexcept
    {
        if (this != &other)
        {
            releaseDeviceNoThrow_();

            h_data = std::move(other.h_data);
            d_ptr = std::exchange(other.d_ptr,  nullptr);
            d_size = std::exchange(other.d_size, 0);
        }
        return *this;
    }

public:
    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t hostSize() const { return h_data.size(); }
    size_t deviceSize() const { return d_size; }

public:
    // ---------------------------------------------------------------------
    // Device memory management
    // ---------------------------------------------------------------------
    void releaseDevice(cudaStream_t stream)
    {
        if (d_ptr)
        {
            cudaFreeAsync(d_ptr, stream);
            d_ptr = nullptr;
        }
        d_size = 0;
    }

    void releaseDeviceSync()
    {
        if (d_ptr)
        {
            cudaFree(d_ptr);
            d_ptr = nullptr;
        }
        d_size = 0;
    }

    void allocateDevice(size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        if (d_ptr) releaseDevice(stream);

        d_size = n;
        if (d_size == 0)
        {
            d_ptr = nullptr;
            return;
        }

        cudaMallocAsync((void**)&d_ptr, d_size * sizeof(T), stream);
        if (zeroFill)
        {
            cudaMemsetAsync(d_ptr, 0, d_size * sizeof(T), stream);
        }
    }

public:
    // ---------------------------------------------------------------------
    // Host-side modifications
    // ---------------------------------------------------------------------
    void pushHost(const T& value) { h_data.push_back(value); }

    void insertHost(size_t index, const T& value)
    {
        if (index >= hostSize()) { h_data.push_back(value); return; }
        h_data.insert(h_data.begin() + static_cast<std::ptrdiff_t>(index), value);
    }

    void eraseHost(size_t index)
    {
        if (index >= h_data.size()) return;
        h_data.erase(h_data.begin() + static_cast<std::ptrdiff_t>(index));
    }

    void clearHost() { h_data.clear(); }
    void reserveHost(size_t n) { h_data.reserve(n); }
    void resizeHost(size_t n) { h_data.resize(n); }

    const std::vector<T>& hostRef() const { return h_data; }
    void setHost(const std::vector<T>& newData) { h_data = newData; }

public:
    // ---------------------------------------------------------------------
    // Host <-> Device transfer
    // ---------------------------------------------------------------------
    void copyHostToDevice(cudaStream_t stream)
    {
        const size_t n = hostSize();

        if (n != d_size || d_ptr == nullptr)
        {
            allocateDevice(n, stream, /*zeroFill=*/false);
        }

        if (n > 0)
        {
            cudaMemcpyAsync(d_ptr,
            h_data.data(),
            n * sizeof(T),
            cudaMemcpyHostToDevice,
            stream);
        }
    }

    void copyDeviceToHost(cudaStream_t stream)
    {
        if (d_size == 0 || d_ptr == nullptr) return;

        if (d_size != hostSize())
        {
            h_data.resize(d_size);
        }

        cudaMemcpyAsync(h_data.data(),
        d_ptr,
        d_size * sizeof(T),
        cudaMemcpyDeviceToHost,
        stream);
    }

    std::vector<T> getHostCopy()
    {
        if (d_size > 0 && d_ptr)
        {
            if (d_size != hostSize())
            {
                h_data.resize(d_size);
            }
            cudaMemcpy(h_data.data(),
            d_ptr,
            d_size * sizeof(T),
            cudaMemcpyDeviceToHost);
        }
        return h_data;
    }
};