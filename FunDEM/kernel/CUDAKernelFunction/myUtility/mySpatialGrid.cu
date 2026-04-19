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
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObject) return;
    
    const double3 p = position[idx];
    
    if (p.x < minBound.x) { hashValue[idx] = -1; return; }
    else if (p.x >= maxBound.x) { hashValue[idx] = -1; return; }
    if (p.y < minBound.y) { hashValue[idx] = -1; return; }
    else if (p.y >= maxBound.y) { hashValue[idx] = -1; return; }
    if (p.z < minBound.z) { hashValue[idx] = -1; return; }
    else if (p.z >= maxBound.z) { hashValue[idx] = -1; return; }
    
    int3 gridPosition = calculateGridPosition(p, minBound, inverseCellSize);
    hashValue[idx] = linearIndex3D(gridPosition, gridSize3D);
}

__global__ void updatePositionOutOfBoundaryXDKernel(double3* position, 
const double3 minBound, 
const double3 maxBound, 
const size_t numObject)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObject) return;

    double3 p = position[idx];
    const double3 offset_X = make_double3(maxBound.x - minBound.x, 0., 0.);
    if (p.x < minBound.x) p += offset_X;
    else if (p.x >= maxBound.x) p -= offset_X;
    position[idx] = p;
}

__global__ void updatePositionOutOfBoundaryYDKernel(double3* position, 
const double3 minBound, 
const double3 maxBound, 
const size_t numObject)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObject) return;

    double3 p = position[idx];
    const double3 offset_Y = make_double3(0., maxBound.y - minBound.y, 0.);
    if (p.y < minBound.y) p += offset_Y;
    else if (p.y >= maxBound.y) p -= offset_Y;
    position[idx] = p;
}

__global__ void updateVelocityPositionOrientationOutOfSectorKernel(double3* velocity, 
double3* position, 
quaternion* orientation, 
const double3 minBound, 
const size_t numObject)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObject) return;

    double3 v = velocity[idx];
    double3 p = position[idx];
    quaternion q = orientation[idx];
    if (p.x < minBound.x && p.y >= minBound.y)
    {
        v = rotateVector(v, make_double3(0., 0., -0.5 * pi()));
        p = minBound + rotateVector(p - minBound, make_double3(0., 0., -0.5 * pi()));
        q = rotateQuaternion(q, make_double3(0., 0., -0.5 * pi()));
    }
    if (p.y < minBound.y && p.x >= minBound.x)
    {
        v = rotateVector(v, make_double3(0., 0., 0.5 * pi()));
        p = minBound + rotateVector(p - minBound, make_double3(0., 0., 0.5 * pi()));
        q = rotateQuaternion(q, make_double3(0., 0., 0.5 * pi()));
    }
    velocity[idx] = v;
    position[idx] = p;
    orientation[idx] = q;
}

__global__ void calculateGhostPositionXDKernel(double3* ghostPosition_XD, 
const double3* position, 
const double3 minBound, 
const double3 maxBound, 
const double3 inverseCellSize, 
const int3 gridSize3D, 
const size_t numObject)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObject) return;

    const double3 p = position[idx];
    const double3 offset_X = make_double3(maxBound.x - minBound.x, 0., 0.);

    ghostPosition_XD[idx] = p;
    const int3 gridPosition = calculateGridPosition(p, minBound, inverseCellSize);
    if (gridPosition.x == 0) 
    {
        ghostPosition_XD[idx] += offset_X;
    }
}

__global__ void calculateGhostPositionYDKernel(double3* ghostPosition_YD, 
const double3* position, 
const double3 minBound, 
const double3 maxBound, 
const double3 inverseCellSize, 
const int3 gridSize3D, 
const size_t numObject)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObject) return;

    const double3 p = position[idx];
    const double3 offset_Y = make_double3(0., maxBound.y - minBound.y, 0.);

    ghostPosition_YD[idx] = p;
    const int3 gridPosition = calculateGridPosition(p, minBound, inverseCellSize);
    if (gridPosition.y == 0) 
    {
        ghostPosition_YD[idx] += offset_Y;
    }
}

__global__ void calculateGhostPositionXYDKernel(double3* ghostPosition_XYD, 
const double3* position, 
const double3 minBound, 
const double3 maxBound, 
const double3 inverseCellSize, 
const int3 gridSize3D, 
const size_t numObject)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObject) return;

    const double3 p = position[idx];
    const double3 offset_X = make_double3(maxBound.x - minBound.x, 0., 0.);
    const double3 offset_Y = make_double3(0., maxBound.y - minBound.y, 0.);

    ghostPosition_XYD[idx] = p;
    const int3 gridPosition = calculateGridPosition(p, minBound, inverseCellSize);
    if (gridPosition.x == 0 && gridPosition.y == 0) 
    {
        ghostPosition_XYD[idx] += offset_X + offset_Y;
    }
}

__global__ void calculateSectorGhostVelocityPositionOrientationKernel(double3* ghostVelocity_R90, 
double3* ghostVelocity_R180, 
double3* ghostVelocity_R270, 
double3* ghostPosition_R90, 
double3* ghostPosition_R180, 
double3* ghostPosition_R270, 
quaternion* ghostOrientation_R90, 
quaternion* ghostOrientation_R180, 
quaternion* ghostOrientation_R270, 
const double3* velocity, 
const double3* position, 
const quaternion* orientation, 
const double3 minBound, 
const double3 maxBound, 
const double3 inverseCellSize, 
const int3 gridSize3D, 
const size_t numObject)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObject) return;

    const double3 v = velocity[idx];
    const double3 p = position[idx];
    const quaternion q = orientation[idx];

    ghostVelocity_R90[idx] = v;
    ghostVelocity_R180[idx] = v;
    ghostVelocity_R270[idx] = v;
    ghostPosition_R90[idx] = p;
    ghostPosition_R180[idx] = p;
    ghostPosition_R270[idx] = p;
    ghostOrientation_R90[idx] = q;
    ghostOrientation_R180[idx] = q;
    ghostOrientation_R270[idx] = q;
    const int3 gridPosition = calculateGridPosition(p, minBound, inverseCellSize);
    if (gridPosition.y == 0)
    {
        ghostVelocity_R90[idx] = rotateVector(v, make_double3(0., 0., 0.5 * pi()));
        ghostPosition_R90[idx] = minBound + rotateVector(p - minBound, make_double3(0., 0., 0.5 * pi()));
        ghostOrientation_R90[idx] = rotateQuaternion(q, make_double3(0., 0., 0.5 * pi()));

        if (gridPosition.x == 0)
        {
            ghostVelocity_R270[idx] = rotateVector(v, make_double3(0., 0., -0.5 * pi()));
            ghostPosition_R270[idx] = minBound + rotateVector(p - minBound, make_double3(0., 0., -0.5 * pi()));
            ghostOrientation_R270[idx] = rotateQuaternion(q, make_double3(0., 0., -0.5 * pi()));

            ghostVelocity_R180[idx] = rotateVector(v, make_double3(0., 0., pi()));
            ghostPosition_R180[idx] = minBound + rotateVector(p - minBound, make_double3(0., 0., pi()));
            ghostOrientation_R180[idx] = rotateQuaternion(q, make_double3(0., 0., pi()));
        }
    }
    else if (gridPosition.x == 0)
    {
        ghostVelocity_R270[idx] = rotateVector(v, make_double3(0., 0., -0.5 * pi()));
        ghostPosition_R270[idx] = minBound + rotateVector(p - minBound, make_double3(0., 0., -0.5 * pi()));
        ghostOrientation_R270[idx] = rotateQuaternion(q, make_double3(0., 0., -0.5 * pi()));
    }
}

__global__ void calculateGhostHashKernel(int* hashValue, 
const double3* ghostPosition, 
const double3 minBound, 
const double3 maxBound, 
const double3 inverseCellSize, 
const int3 gridSize3D,
const size_t numObject)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObject) return;
    
    const double3 p = ghostPosition[idx];
    int3 gridPosition = calculateGridPosition(p, minBound, inverseCellSize);
    bool outOfBoundary = false;

    if (p.x < minBound.x) { gridPosition.x = 0; outOfBoundary = true; }
    else if (p.x >= maxBound.x) { gridPosition.x = gridSize3D.x - 1; outOfBoundary = true; }

    if (p.y < minBound.y) { gridPosition.y = 0; outOfBoundary = true; }
    else if (p.y >= maxBound.y) { gridPosition.y = gridSize3D.y - 1; outOfBoundary = true; }

    if (p.z < minBound.z) { gridPosition.z = 0; outOfBoundary = true; }
    else if (p.z >= maxBound.y) { gridPosition.z = gridSize3D.z - 1; outOfBoundary = true; }

    if (outOfBoundary) hashValue[idx] = linearIndex3D(gridPosition, gridSize3D);
    else hashValue[idx] = -1;
}

extern "C" void launchUpdateSpatialGridHashStartEnd(int* hashIndex, 
int* hashValue, 
const double3* position, 

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

extern "C" void launchUpdatePositionOutOfBoundaryXD(double3* position,
const double3 minBound,
const double3 maxBound,
const size_t numObject,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    updatePositionOutOfBoundaryXDKernel<<<gridD, blockD, 0, stream>>>(position,
    minBound,
    maxBound,
    numObject);
}

extern "C" void launchUpdatePositionOutOfBoundaryYD(double3* position,
const double3 minBound,
const double3 maxBound,
const size_t numObject,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    updatePositionOutOfBoundaryYDKernel<<<gridD, blockD, 0, stream>>>(position,
    minBound,
    maxBound,
    numObject);
}

extern "C" void launchUpdateVelocityPositionOrientationOutOfSector(double3* velocity,
double3* position,
quaternion* orientation,
const double3 minBound,
const size_t numObject,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    updateVelocityPositionOrientationOutOfSectorKernel<<<gridD, blockD, 0, stream>>>(velocity,
    position,
    orientation,
    minBound,
    numObject);
}

extern "C" void launchCalculateGhostPositionXD(double3* ghostPosition_XD,

const double3* position,

const double3 minBound,
const double3 maxBound,
const double3 inverseCellSize,
const int3 gridSize3D,

const size_t numObject,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    calculateGhostPositionXDKernel<<<gridD, blockD, 0, stream>>>(ghostPosition_XD,

    position,

    minBound,
    maxBound,
    inverseCellSize,
    gridSize3D,

    numObject);
}

extern "C" void launchCalculateGhostPositionYD(double3* ghostPosition_YD,

const double3* position,

const double3 minBound,
const double3 maxBound,
const double3 inverseCellSize,
const int3 gridSize3D,

const size_t numObject,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    calculateGhostPositionYDKernel<<<gridD, blockD, 0, stream>>>(ghostPosition_YD,

    position,

    minBound,
    maxBound,
    inverseCellSize,
    gridSize3D,

    numObject);
}

extern "C" void launchCalculateGhostPositionXYD(double3* ghostPosition_XYD,

const double3* position,

const double3 minBound,
const double3 maxBound,
const double3 inverseCellSize,
const int3 gridSize3D,

const size_t numObject,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    calculateGhostPositionXYDKernel<<<gridD, blockD, 0, stream>>>(ghostPosition_XYD,

    position,

    minBound,
    maxBound,
    inverseCellSize,
    gridSize3D,

    numObject);
}

extern "C" void launchCalculateSectorGhostVelocityPositionOrientation(double3* ghostVelocity_R90,
double3* ghostVelocity_R180,
double3* ghostVelocity_R270,

double3* ghostPosition_R90,
double3* ghostPosition_R180,
double3* ghostPosition_R270,

quaternion* ghostOrientation_R90,
quaternion* ghostOrientation_R180,
quaternion* ghostOrientation_R270,

const double3* velocity,
const double3* position,
const quaternion* orientation,

const double3 minBound,
const double3 maxBound,
const double3 inverseCellSize,
const int3 gridSize3D,

const size_t numObject,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    calculateSectorGhostVelocityPositionOrientationKernel<<<gridD, blockD, 0, stream>>>(ghostVelocity_R90,
    ghostVelocity_R180,
    ghostVelocity_R270,

    ghostPosition_R90,
    ghostPosition_R180,
    ghostPosition_R270,

    ghostOrientation_R90,
    ghostOrientation_R180,
    ghostOrientation_R270,

    velocity,
    position,
    orientation,

    minBound,
    maxBound,
    inverseCellSize,
    gridSize3D,

    numObject);
}

extern "C" void launchUpdateGhostSpatialGridHashStartEnd(int* hashIndex, 
int* hashValue, 
const double3* ghostPosition, 

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
    calculateGhostHashKernel <<< gridD_GPU, blockD_GPU, 0, stream_GPU >>> (hashValue, 
    ghostPosition, 
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