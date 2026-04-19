#pragma once
#include "myHostDeviceArray.h"
#include "myQua.h"

/**
 * @brief Compute the 3D uniform-grid cell index of a point.
 *
 * Maps a world/global position to an integer grid coordinate (cell index) using
 * a grid origin and the inverse cell size:
 *   grid = floor((position - minBoundary) * inverseCellSize)
 *
 * Notes:
 * - inverseCellSize should be (1/cellSize.x, 1/cellSize.y, 1/cellSize.z).
 * - This performs truncation toward zero via int(...) after multiplication, so
 *   it assumes (position - minBoundary) is non-negative in typical usage.
 *   If positions can be below minBoundary, consider using floor() explicitly.
 * - Intended for use inside CUDA kernels / device code.
 *
 * @param position         Point position in world/global coordinates.
 * @param minBoundary      Grid minimum corner (origin) in world/global coordinates.
 * @param inverseCellSize  Component-wise inverse cell size (1/cellSize).
 * @return int3            Grid cell coordinate (ix, iy, iz).
 */
__device__ __forceinline__ int3 calculateGridPosition(double3 position, 
const double3 minBound, 
const double3 inverseCellSize)
{
    return make_int3(int((position.x - minBound.x) * inverseCellSize.x),
    int((position.y - minBound.y) * inverseCellSize.y),
    int((position.z - minBound.z) * inverseCellSize.z));
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
cudaStream_t stream_GPU);

extern "C" void launchUpdatePositionOutOfBoundaryXD(double3* position,
const double3 minBound,
const double3 maxBound,
const size_t numObject,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchUpdatePositionOutOfBoundaryYD(double3* position,
const double3 minBound,
const double3 maxBound,
const size_t numObject,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchUpdateVelocityPositionOrientationOutOfSector(double3* velocity,
double3* position,
quaternion* orientation,
const double3 minBound,
const size_t numObject,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchCalculateGhostPositionXD(double3* ghostPosition_XD,

const double3* position,

const double3 minBound,
const double3 maxBound,
const double3 inverseCellSize,
const int3 gridSize3D,

const size_t numObject,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchCalculateGhostPositionYD(double3* ghostPosition_YD,

const double3* position,

const double3 minBound,
const double3 maxBound,
const double3 inverseCellSize,
const int3 gridSize3D,

const size_t numObject,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchCalculateGhostPositionXYD(double3* ghostPosition_XYD,

const double3* position,

const double3 minBound,
const double3 maxBound,
const double3 inverseCellSize,
const int3 gridSize3D,

const size_t numObject,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

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
cudaStream_t stream);

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
cudaStream_t stream_GPU);

struct spatialGrid
{
private:
    // ---------------------------------------------------------------------
    // Geometry / grid parameters (not SoA)
    // ---------------------------------------------------------------------
    double3 minBoundary_ {0.0, 0.0, 0.0};
    double3 maxBoundary_ {0.0, 0.0, 0.0};
    double3 inverseCellSize_ {0.0, 0.0, 0.0};

    int3 size3D_ {2, 2, 2};
    size_t num_ {8};

    // ---------------------------------------------------------------------
    // Per-cell arrays
    //   hashStart[h] : start index in sorted list for cell h
    //   hashEnd[h]   : end index in sorted list for cell h
    // ---------------------------------------------------------------------
    HostDeviceArray1D<int> hashStart_;
    HostDeviceArray1D<int> hashEnd_;

    // ---------------------------------------------------------------------
    // Device buffer allocation (empty buffer)
    // ---------------------------------------------------------------------
    void allocateDevice(const size_t numGrid,
    cudaStream_t stream)
    {
        if (numGrid > 0) num_ = numGrid;
        hashStart_.allocateDevice(num_, stream, /*zeroFill=*/false);
        hashEnd_.allocateDevice(num_, stream, /*zeroFill=*/false);
        cudaMemsetAsync(hashStart_.d_ptr, 0xFF, num_ * sizeof(int), stream);
        cudaMemsetAsync(hashEnd_.d_ptr, 0xFF, num_ * sizeof(int), stream);
    }

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    spatialGrid() = default;
    ~spatialGrid() = default;

    spatialGrid(const spatialGrid&) = delete;
    spatialGrid& operator=(const spatialGrid&) = delete;

    spatialGrid(spatialGrid&&) noexcept = default;
    spatialGrid& operator=(spatialGrid&&) noexcept = default;

    // ---------------------------------------------------------------------
    // Set / initialize grid
    // ---------------------------------------------------------------------
    void set(double3 minBoundary,
    double3 maxBoundary,
    double cellSizeOneDim,
    cudaStream_t stream)
    {
        if (maxBoundary.x <= minBoundary.x) { maxBoundary.x = minBoundary.x + cellSizeOneDim; }
        if (maxBoundary.y <= minBoundary.y) { maxBoundary.y = minBoundary.y + cellSizeOneDim; }
        if (maxBoundary.z <= minBoundary.z) { maxBoundary.z = minBoundary.z + cellSizeOneDim; }

        minBoundary_ = minBoundary;
        maxBoundary_ = maxBoundary;

        if (isZero(cellSizeOneDim)) 
        { 
            allocateDevice(8, stream); 
            return; 
        }

        const double3 domainSize = maxBoundary - minBoundary;
        size3D_.x = domainSize.x > 2. * cellSizeOneDim ? int(domainSize.x / cellSizeOneDim) : 2;
        size3D_.y = domainSize.y > 2. * cellSizeOneDim ? int(domainSize.y / cellSizeOneDim) : 2;
        size3D_.z = domainSize.z > 2. * cellSizeOneDim ? int(domainSize.z / cellSizeOneDim) : 2;

        inverseCellSize_.x = (domainSize.x > 0.0) ? (double(size3D_.x) / domainSize.x) : 0.0;
        inverseCellSize_.y = (domainSize.y > 0.0) ? (double(size3D_.y) / domainSize.y) : 0.0;
        inverseCellSize_.z = (domainSize.z > 0.0) ? (double(size3D_.z) / domainSize.z) : 0.0;

        allocateDevice(size_t(size3D_.x) * size_t(size3D_.y) * size_t(size3D_.z), stream);
    }

    // ---------------------------------------------------------------------
    // Device pointers (hash)
    // ---------------------------------------------------------------------
    int* hashStart() { return hashStart_.d_ptr; }
    int* hashEnd() { return hashEnd_.d_ptr; }

    // ---------------------------------------------------------------------
    // Getters (host-side parameters)
    // ---------------------------------------------------------------------
    const double3& minimumBoundary() const { return minBoundary_; }
    const double3& maximumBoundary() const { return maxBoundary_; }
    const double3& inverseCellSize() const { return inverseCellSize_; }
    const int3& size3D() const { return size3D_; }
    size_t num() const { return num_; }
    size_t num_device() const { return num_; }
};