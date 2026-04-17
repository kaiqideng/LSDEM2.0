#pragma once
#include <driver_types.h>

// -----------------------------------------------------------------------------
// buildHashStartEnd
// -----------------------------------------------------------------------------
// Build per-hash-bucket start/end ranges for a hash grid.
//
// Workflow:
// 1) Initialize start/end arrays to -1 (0xFF).
// 2) Initialize hashIndex to identity (0..hashListSize-1).
// 3) Sort (hashValue, hashIndex) by hashValue using thrust::sort_by_key.
//    After sorting:
//      - hashValue is sorted ascending
//      - hashIndex carries the original indices
// 4) Launch findStartAndEnd to compute start/end for each hash bucket.
//
// start          : device array length startEndSize, output (start indices)
// end            : device array length startEndSize, output (end indices)
// hashIndex      : device array length hashListSize, output (permutation indices)
// hashValue      : device array length hashListSize, IN/OUT (sorted in place)
// startEndSize   : number of hash buckets (e.g., number of cells)
// hashListSize   : number of hashed items (e.g., particles)
// gridD_GPU      : kernel grid dimension for setHashIndex/findStartAndEnd
// blockD_GPU     : kernel block dimension for setHashIndex/findStartAndEnd
// stream_GPU     : CUDA stream used for memset, kernels, and thrust algorithms
//
// Debug notes:
// - In debug builds, the function prints cudaGetLastError() around key steps and
//   catches thrust::system_error to surface runtime failures.
// -----------------------------------------------------------------------------
extern "C" void buildHashStartEnd(int* start, 
int* end, 
int* hashIndex, 

int* hashValue, 

const size_t startEndSize,

const size_t hashListSize,
const size_t gridD_GPU, 
const size_t blockD_GPU,  
cudaStream_t stream_GPU);

// -----------------------------------------------------------------------------
// buildPrefixSum
// -----------------------------------------------------------------------------
// Compute inclusive prefix sum (scan) of an integer "count" array:
//   prefixSum[i] = sum_{k=0..i} count[k]
//
// Typically used to convert per-item counts into write offsets, where the last
// prefixSum value equals the total number of pairs/elements.
//
// prefixSum : device array length size, output
// count     : device array length size, input
// size      : number of elements
// stream    : CUDA stream for thrust execution
//
// Debug notes:
// - In debug builds, prints cudaGetLastError() and catches thrust::system_error.
// -----------------------------------------------------------------------------
extern "C" void buildPrefixSum(int* prefixSum,
int* count, 
const size_t size, 
cudaStream_t stream);