#include "mySpatialGrid.h"
#include "buildHashStartEnd.h"

__global__ void calculateHashKernel(int* hashValue, 
const double3* position, 
const double3 minBound, 
const double3 maxBound, 
const double3 inverseCellSize, 
const int3 gridSize3D,
const size_t numObject)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObject) return;
    
    double3 p = position[idx];
    int3 gridPosition = calculateGridPosition(p, minBound, inverseCellSize);
    
    if (gridPosition.x < 0) { hashValue[idx] = -1; return; }
    else if (gridPosition.x >= gridSize3D.x) { hashValue[idx] = -1; return; }
    if (gridPosition.y < 0) { hashValue[idx] = -1; return; }
    else if (gridPosition.y >= gridSize3D.y) { hashValue[idx] = -1; return; }
    if (gridPosition.z < 0) { hashValue[idx] = -1; return; }
    else if (gridPosition.z >= gridSize3D.z) { hashValue[idx] = -1; return; }
    
    hashValue[idx] = linearIndex3D(gridPosition, gridSize3D);
}

extern "C" void launchUpdateSpatialGridHashStartEnd(double3* position, 
int* hashIndex, 
int* hashValue, 

int* gridHashStart,
int* gridHashEnd,

const double3 minBound,
const double3 maxBound,
const double3 inverseCellSize,
const int3 gridSize3D,
const size_t numGrid,

const size_t numObject,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    calculateHashKernel <<< gridD_GPU, blockD_GPU, 0, stream_GPU >>> (hashValue, 
    position, 
    minBound, 
    maxBound, 
    inverseCellSize, 
    gridSize3D,
    numObject);

    buildHashStartEnd(gridHashStart,
    gridHashEnd,
    hashIndex,
    hashValue,
    numGrid,
    numObject,
    gridD_GPU,
    blockD_GPU,
    stream_GPU);
}
