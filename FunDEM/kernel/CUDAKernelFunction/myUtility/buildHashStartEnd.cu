#include "buildHashStartEnd.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

// -----------------------------------------------------------------------------
// setHashIndex
// -----------------------------------------------------------------------------
// Initialize the permutation array "hashIndex" with identity mapping:
//   hashIndex[i] = i
//
// This is typically used before sorting (hashValue, hashIndex) by hashValue,
// so that after sorting, hashIndex stores the original element indices.
//
// hashIndex      : device array of length hashListSize
// hashListSize   : number of elements in the hash list
// -----------------------------------------------------------------------------
__global__ void setHashIndex(int* hashIndex, 
const size_t hashListSize)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hashListSize) return;
    hashIndex[index] = static_cast<int>(index);
}

// -----------------------------------------------------------------------------
// findStartAndEnd
// -----------------------------------------------------------------------------
// Given a sorted hash array "sortedHashValue" (ascending), compute for each hash
// value h the half-open range [start[h], end[h]) of indices in the sorted list.
//
// The arrays start/end are assumed to be initialized to -1 (0xFF) beforehand.
// This kernel writes:
//   start[h] = first index where sortedHashValue[index] == h
//   end[h]   = (last index where sortedHashValue[index] == h) + 1
//
// start          : device array length startEndSize (per-cell start)
// end            : device array length startEndSize (per-cell end)
// sortedHashValue: device array length hashListSize, must be sorted
// startEndSize   : number of possible hash buckets (e.g., number of grid cells)
// hashListSize   : length of sortedHashValue
//
// Notes:
// - If a hash bucket h has no elements, start[h] and end[h] remain -1.
// - Any hash value outside [0, startEndSize) is ignored.
// -----------------------------------------------------------------------------
__global__ void findStartAndEnd(int* start, 
int* end, 
int* sortedHashValue, 
const size_t startEndSize,
const size_t hashListSize)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hashListSize) return;

    int h = sortedHashValue[index];

    if (h < 0 || h >= startEndSize) return;
    if (index == 0 || sortedHashValue[index - 1] != h)
    {
        start[h] = static_cast<int>(index);
    }
    if (index == hashListSize - 1 || sortedHashValue[index + 1] != h) 
    {
        end[h] = static_cast<int>(index + 1);
    }
}

extern "C" void buildHashStartEnd(int* start, 
int* end, 

int* hashIndex, 
int* hashValue, 

const size_t startEndSize,

const size_t hashListSize,
const size_t gridD_GPU, 
const size_t blockD_GPU,  
cudaStream_t stream_GPU)
{
    cudaMemsetAsync(start, 0xFF, startEndSize * sizeof(int), stream_GPU);
    cudaMemsetAsync(end, 0xFF, startEndSize * sizeof(int), stream_GPU);

#ifndef NDEBUG
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess)
    {
        std::cerr << "[buildHashStartEnd] cudaMemsetAsync start/end, cudaGetLastError = "
        << cudaGetErrorString(err1) << "\n";
    }
#endif

    setHashIndex <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (hashIndex, 
    hashListSize);

    auto exec = thrust::cuda::par.on(stream_GPU);
#ifndef NDEBUG
    try
    {
        cudaError_t err0 = cudaGetLastError();
        if (err0 != cudaSuccess)
        {
            std::cerr << "[buildHashStartEnd] before sort, cudaGetLastError = "
            << cudaGetErrorString(err0) << "\n";
        }

        thrust::sort_by_key(exec,
        hashValue,
        hashValue + hashListSize,
        hashIndex);

        cudaError_t err1 = cudaGetLastError();
        if (err1 != cudaSuccess)
        {
            std::cerr << "[buildHashStartEnd] after sort, cudaGetLastError = "
            << cudaGetErrorString(err1) << "\n";
        }
    }
    catch (thrust::system_error& e)
    {
        std::cerr << "thrust::sort_by_key threw: " << e.what() << "\n";
        throw;
    }
#else
        thrust::sort_by_key(exec,
        hashValue,
        hashValue + hashListSize,
        hashIndex);
#endif

    findStartAndEnd <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (start, 
    end, 
    hashValue, 
    startEndSize, 
    hashListSize);
}

extern "C" void buildPrefixSum(int* prefixSum,
int* count, 
const size_t size, 
cudaStream_t stream)
{
    auto exec = thrust::cuda::par.on(stream);
#ifndef NDEBUG
    try
    {
        cudaError_t err0 = cudaGetLastError();
        if (err0 != cudaSuccess)
        {
            std::cerr << "[buildPrefixSum] before prefixSum, cudaGetLastError = "
            << cudaGetErrorString(err0) << "\n";
        }

        thrust::inclusive_scan(exec,
        thrust::device_pointer_cast(count),
        thrust::device_pointer_cast(count + size),
        thrust::device_pointer_cast(prefixSum));

        cudaError_t err1 = cudaGetLastError();
        if (err1 != cudaSuccess)
        {
            std::cerr << "[buildPrefixSum] after prefixSum, cudaGetLastError = "
            << cudaGetErrorString(err1) << "\n";
        }
    }
    catch (thrust::system_error& e)
    {
        std::cerr << "thrust::sort_by_key threw: " << e.what() << "\n";
        throw;
    }
#else
        thrust::inclusive_scan(exec,
        thrust::device_pointer_cast(count),
        thrust::device_pointer_cast(count + size),
        thrust::device_pointer_cast(prefixSum));
#endif
}