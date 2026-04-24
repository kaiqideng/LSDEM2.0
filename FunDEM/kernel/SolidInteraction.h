#pragma once
#include "CUDAKernelFunction/myUtility/myHostDeviceArray.h"
#include "CUDAKernelFunction/myUtility/myVec.h"
#include "CUDAKernelFunction/myUtility/buildHashStartEnd.h"
#include "CUDAKernelFunction/myUtility/myFileEdit.h"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>

struct SolidInteraction
{
public:
    SolidInteraction() = default;
    ~SolidInteraction() = default;

    SolidInteraction(const SolidInteraction&) = delete;
    SolidInteraction& operator=(const SolidInteraction&) = delete;

    SolidInteraction(SolidInteraction&&) noexcept = default;
    SolidInteraction& operator=(SolidInteraction&&) noexcept = default;

    void initialize(const size_t numMasterObject, const size_t maxGPUThread, cudaStream_t stream)
    {
        const size_t numMasterObject0 = masterNeighborPrefixSum_.hostSize();
        if (numMasterObject <= numMasterObject0) return;
        
        if (numMasterObject0 == 0)
        {
            masterNeighborCount_.allocateDevice(numMasterObject, stream);
            masterNeighborPrefixSum_.allocateDevice(numMasterObject, stream);
            masterNeighborPrefixSum0_.allocateDevice(numMasterObject, stream);
        }
        else
        {
            masterNeighborPrefixSum_.copyDeviceToHost(stream);
            for (size_t i = 0; i < numMasterObject - numMasterObject0; i++)
            {
                masterNeighborPrefixSum_.pushHost(masterNeighborPrefixSum_.hostRef()[numMasterObject0 - 1]);
            }
            masterNeighborPrefixSum_.copyHostToDevice(stream);
            masterNeighborCount_.allocateDevice(numMasterObject, stream);
            masterNeighborPrefixSum0_.allocateDevice(numMasterObject, stream);
        }
        
        if (maxGPUThread > 0) masterBlockDim_ = maxGPUThread;
        if (numMasterObject < maxGPUThread) masterBlockDim_ = numMasterObject;
        masterGridDim_ = (numMasterObject + masterBlockDim_ - 1) / masterBlockDim_;
    }

    double deviceMemoryGB() const
    {
        double total = 0.0;

        // Contact data
        total += contactPoint_.deviceMemoryGB();
        total += contactNormal_.deviceMemoryGB();
        total += contactOverlap_.deviceMemoryGB();
        total += normalElasticEnergy_.deviceMemoryGB();
        total += slidingElasticEnergy_.deviceMemoryGB();
        total += slidingSpring_.deviceMemoryGB();
        total += masterID_.deviceMemoryGB();
        total += slaveID_.deviceMemoryGB();

        // Previous step
        total += slidingSpring0_.deviceMemoryGB();
        total += slaveID0_.deviceMemoryGB();

        // Neighbor bookkeeping
        total += masterNeighborCount_.deviceMemoryGB();
        total += masterNeighborPrefixSum_.deviceMemoryGB();
        total += masterNeighborPrefixSum0_.deviceMemoryGB();

        return total;
    }

    void setInteractionArraySize(const size_t arraySize, cudaStream_t stream)
    {
        contactPoint_.allocateDevice(arraySize, stream);
        contactNormal_.allocateDevice(arraySize, stream);
        contactOverlap_.allocateDevice(arraySize, stream);
        normalElasticEnergy_.allocateDevice(arraySize, stream);
        slidingElasticEnergy_.allocateDevice(arraySize, stream);
        slidingSpring_.allocateDevice(arraySize, stream);
        masterID_.allocateDevice(arraySize, stream);
        slaveID_.allocateDevice(arraySize, stream);

        slidingSpring0_.allocateDevice(arraySize, stream);
        slaveID0_.allocateDevice(arraySize, stream);
    }

    void updateNeighborPrefixSum(cudaStream_t stream)
    {
        cudaMemcpyAsync(masterNeighborPrefixSum0_.d_ptr, masterNeighborPrefixSum_.d_ptr, 
        masterNeighborPrefixSum_.deviceSize() * sizeof(int), cudaMemcpyDeviceToDevice, stream);

        buildPrefixSum(masterNeighborPrefixSum_.d_ptr,
        masterNeighborCount_.d_ptr,
        masterNeighborCount_.deviceSize(),
        stream);
    }

    void updateNumPair(const size_t maxGPUThread, cudaStream_t stream)
    {
        cudaMemcpyAsync(&numPair_, masterNeighborPrefixSum_.d_ptr + masterNeighborPrefixSum_.deviceSize() - 1, 
        sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (numPair_ > 0)
        {
            pairBlockDim_ = static_cast<size_t>(numPair_);
            if (maxGPUThread > 0 && maxGPUThread < static_cast<size_t>(numPair_)) pairBlockDim_ = maxGPUThread;
            pairGridDim_ = (static_cast<size_t>(numPair_) + pairBlockDim_ - 1) / pairBlockDim_;   
        }
        else
        {
            pairGridDim_ = 1;
            pairBlockDim_ = 1;
            return;
        }
    }

    void savePreviousStep(cudaStream_t stream)
    {
        const size_t arraySize = static_cast<size_t>(numPair_);
        if (arraySize > contactPoint_.deviceSize())
        {
            slidingSpring0_.allocateDevice(arraySize, stream);
            slaveID0_.allocateDevice(arraySize, stream);

            cudaMemcpyAsync(slidingSpring0_.d_ptr, slidingSpring_.d_ptr, 
            slidingSpring_.deviceSize() * sizeof(double3), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(slaveID0_.d_ptr, slaveID_.d_ptr, 
            slaveID_.deviceSize() * sizeof(int), cudaMemcpyDeviceToDevice, stream);
 
            contactPoint_.allocateDevice(arraySize, stream);
            contactNormal_.allocateDevice(arraySize, stream);
            contactOverlap_.allocateDevice(arraySize, stream);
            normalElasticEnergy_.allocateDevice(arraySize, stream);
            slidingElasticEnergy_.allocateDevice(arraySize, stream);
            slidingSpring_.allocateDevice(arraySize, stream);
            masterID_.allocateDevice(arraySize, stream);
            slaveID_.allocateDevice(arraySize, stream);
        }
        else 
        {
            cudaMemcpyAsync(slidingSpring0_.d_ptr, slidingSpring_.d_ptr, 
            slidingSpring_.deviceSize() * sizeof(double3), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(slaveID0_.d_ptr, slaveID_.d_ptr, 
            slaveID_.deviceSize() * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        }
    }

    void outputVTU(const std::string& dir, const size_t iFrame, const size_t iStep, const double time) const
    {
        MKDIR(dir.c_str());

        std::ostringstream fname;
        fname << dir << "/LSInteraction_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";

        size_t N = 0;
        if (numMaster() > 0) N = masterNeighborPrefixSum_.hostRef()[numMaster() - 1];

        const std::vector<double3>& p  = contactPoint_.hostRef();
        const std::vector<double3>& n  = contactNormal_.hostRef();
        const std::vector<double>&  nE = normalElasticEnergy_.hostRef();
        const std::vector<double>&  sE = slidingElasticEnergy_.hostRef();
        const std::vector<double3>& s  = slidingSpring_.hostRef();

        // -------------------------------------------------------------------------
        // Precompute arrays
        // -------------------------------------------------------------------------
        std::vector<float>   points(N * 3);
        std::vector<float>   normal(N * 3);
        std::vector<float>   slidingDirection(N * 3);
        std::vector<float>   normalElasticEnergy(N);
        std::vector<float>   slidingElasticEnergy(N);
        std::vector<int32_t> conn(N);
        std::vector<int32_t> offs(N);
        std::vector<uint8_t> types(N);

        for (size_t i = 0; i < N; ++i)
        {
            points[3 * i + 0] = static_cast<float>(p[i].x);
            points[3 * i + 1] = static_cast<float>(p[i].y);
            points[3 * i + 2] = static_cast<float>(p[i].z);

            normal[3 * i + 0] = static_cast<float>(n[i].x);
            normal[3 * i + 1] = static_cast<float>(n[i].y);
            normal[3 * i + 2] = static_cast<float>(n[i].z);

            const double3 s1 = normalize(s[i]);
            slidingDirection[3 * i + 0] = static_cast<float>(s1.x);
            slidingDirection[3 * i + 1] = static_cast<float>(s1.y);
            slidingDirection[3 * i + 2] = static_cast<float>(s1.z);

            normalElasticEnergy[i]  = static_cast<float>(nE[i]);
            slidingElasticEnergy[i] = static_cast<float>(sE[i]);

            conn[i]  = static_cast<int32_t>(i);
            offs[i]  = static_cast<int32_t>(i + 1);
            types[i] = static_cast<uint8_t>(1);
        }

        // -------------------------------------------------------------------------
        // Open file
        // -------------------------------------------------------------------------
        std::ofstream out(fname.str());
        if (!out)
            throw std::runtime_error("Cannot open for writing: " + fname.str());

        // -------------------------------------------------------------------------
        // base64 encoder
        // -------------------------------------------------------------------------
        static const char kB64Table[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        auto toB64 = [&](const void* data, size_t nBytes)
        {
            const uint64_t sz = static_cast<uint64_t>(nBytes);
            std::vector<uint8_t> buf(sizeof(uint64_t) + nBytes);
            std::memcpy(buf.data(),                    &sz,  sizeof(uint64_t));
            std::memcpy(buf.data() + sizeof(uint64_t), data, nBytes);

            const uint8_t* in  = buf.data();
            const size_t   len = buf.size();

            for (size_t i = 0; i < len; i += 3)
            {
                const uint32_t b0 = in[i];
                const uint32_t b1 = (i + 1 < len) ? in[i + 1] : 0u;
                const uint32_t b2 = (i + 2 < len) ? in[i + 2] : 0u;
                const uint32_t v  = (b0 << 16) | (b1 << 8) | b2;

                out.put(kB64Table[(v >> 18) & 0x3F]);
                out.put(kB64Table[(v >> 12) & 0x3F]);
                out.put((i + 1 < len) ? kB64Table[(v >> 6) & 0x3F] : '=');
                out.put((i + 2 < len) ? kB64Table[(v     ) & 0x3F] : '=');
            }
        };

        // -------------------------------------------------------------------------
        // XML
        // -------------------------------------------------------------------------
        out << "<?xml version=\"1.0\"?>\n"
            << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
            "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
            << "<UnstructuredGrid>\n";

        // FieldData
        out << "<FieldData>\n"
            << "<DataArray type=\"Float32\" Name=\"TIME\" "
            "NumberOfTuples=\"1\" format=\"ascii\">"
            << static_cast<float>(time)
            << "</DataArray>\n"
            << "<DataArray type=\"Int32\" Name=\"STEP\" "
            "NumberOfTuples=\"1\" format=\"ascii\">"
            << static_cast<int32_t>(iStep)
            << "</DataArray>\n"
            << "</FieldData>\n";

        out << "<Piece NumberOfPoints=\"" << N
            << "\" NumberOfCells=\""      << N << "\">\n";

        // Points
        out << "<Points>\n"
            << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"binary\">\n";
        toB64(points.data(), points.size() * sizeof(float));
        out << "\n</DataArray>\n"
            << "</Points>\n";

        // Cells
        out << "<Cells>\n";

        out << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"binary\">\n";
        toB64(conn.data(), conn.size() * sizeof(int32_t));
        out << "\n</DataArray>\n";

        out << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"binary\">\n";
        toB64(offs.data(), offs.size() * sizeof(int32_t));
        out << "\n</DataArray>\n";

        out << "<DataArray type=\"UInt8\" Name=\"types\" format=\"binary\">\n";
        toB64(types.data(), types.size() * sizeof(uint8_t));
        out << "\n</DataArray>\n";

        out << "</Cells>\n";

        // PointData
        out << "<PointData>\n";

        out << "<DataArray type=\"Float32\" Name=\"contactNormal\" "
            "NumberOfComponents=\"3\" format=\"binary\">\n";
        toB64(normal.data(), normal.size() * sizeof(float));
        out << "\n</DataArray>\n";

        out << "<DataArray type=\"Float32\" Name=\"slidingDirection\" "
            "NumberOfComponents=\"3\" format=\"binary\">\n";
        toB64(slidingDirection.data(), slidingDirection.size() * sizeof(float));
        out << "\n</DataArray>\n";

        out << "<DataArray type=\"Float32\" Name=\"normalElasticEnergy\" format=\"binary\">\n";
        toB64(normalElasticEnergy.data(), normalElasticEnergy.size() * sizeof(float));
        out << "\n</DataArray>\n";

        out << "<DataArray type=\"Float32\" Name=\"slidingElasticEnergy\" format=\"binary\">\n";
        toB64(slidingElasticEnergy.data(), slidingElasticEnergy.size() * sizeof(float));
        out << "\n</DataArray>\n";

        out << "</PointData>\n"
            << "</Piece>\n"
            << "</UnstructuredGrid>\n"
            << "</VTKFile>\n";

        if (!out)
            throw std::runtime_error("Write error on: " + fname.str());
    }

    void finalize(cudaStream_t stream)
    {
        copyDeviceToHost(stream);
    }

    double3* contactPoint() { return contactPoint_.d_ptr; }
    double3* contactNormal() { return contactNormal_.d_ptr; }
    double* contactOverlap() { return contactOverlap_.d_ptr; }
    double* normalElasticEnergy() { return normalElasticEnergy_.d_ptr; }
    double* slidingElasticEnergy() { return slidingElasticEnergy_.d_ptr; }
    double3* slidingSpring() { return slidingSpring_.d_ptr; }
    int* masterID() { return masterID_.d_ptr; }
    int* slaveID() { return slaveID_.d_ptr; }
    double3* previousSlidingSpring() { return slidingSpring0_.d_ptr; }
    int* previousSlaveID() { return slaveID0_.d_ptr; }

    size_t numPair_device() const { return static_cast<size_t>(numPair_); }
    const size_t& pairGridDim() const { return pairGridDim_; }
    const size_t& pairBlockDim() const { return pairBlockDim_; }

    int* masterNeighborCount() { return masterNeighborCount_.d_ptr; }
    int* masterNeighborPrefixSum() { return masterNeighborPrefixSum_.d_ptr; }
    int* previousMasterNeighborPrefixSum() { return masterNeighborPrefixSum0_.d_ptr; }

    size_t numMaster() const { return masterNeighborPrefixSum_.hostSize(); }
    size_t numMaster_device() const { return masterNeighborPrefixSum_.deviceSize(); }
    const size_t& masterGridDim() const { return masterGridDim_; }
    const size_t& masterBlockDim() const { return masterBlockDim_; }

    const std::vector<double3> contactPointHostRef() const
    {
        std::vector<double3> v = contactPoint_.hostRef();
        v.resize(numPair_device());
        return v;
    }

    const std::vector<double3> contactNormalHostRef() const
    {
        std::vector<double3> v = contactNormal_.hostRef();
        v.resize(numPair_device());
        return v;
    }

    const std::vector<int> masterIDHostRef() const
    {
        std::vector<int> v = masterID_.hostRef();
        v.resize(numPair_device());
        return v;
    }

    const std::vector<int> slaveIDHostRef() const
    {
        std::vector<int> v = slaveID_.hostRef();
        v.resize(numPair_device());
        return v;
    }

private:
    void copyDeviceToHost(cudaStream_t stream)
    {
        contactPoint_.copyDeviceToHost(stream);
        contactNormal_.copyDeviceToHost(stream);
        contactOverlap_.copyDeviceToHost(stream);
        normalElasticEnergy_.copyDeviceToHost(stream);
        slidingElasticEnergy_.copyDeviceToHost(stream);
        slidingSpring_.copyDeviceToHost(stream);
        masterID_.copyDeviceToHost(stream);
        slaveID_.copyDeviceToHost(stream);

        masterNeighborPrefixSum_.copyDeviceToHost(stream);
    }

    HostDeviceArray1D<double3> contactPoint_;
    HostDeviceArray1D<double3> contactNormal_;
    HostDeviceArray1D<double> contactOverlap_;
    HostDeviceArray1D<double> normalElasticEnergy_;
    HostDeviceArray1D<double> slidingElasticEnergy_;
    HostDeviceArray1D<double3> slidingSpring_;
    HostDeviceArray1D<int> masterID_;
    HostDeviceArray1D<int> slaveID_;
    HostDeviceArray1D<double3> slidingSpring0_;
    HostDeviceArray1D<int> slaveID0_;
    int numPair_{0};
    size_t pairGridDim_{1};
    size_t pairBlockDim_{1};

    HostDeviceArray1D<int> masterNeighborCount_;
    HostDeviceArray1D<int> masterNeighborPrefixSum_;
    HostDeviceArray1D<int> masterNeighborPrefixSum0_;
    size_t masterGridDim_{1};
    size_t masterBlockDim_{1};
};