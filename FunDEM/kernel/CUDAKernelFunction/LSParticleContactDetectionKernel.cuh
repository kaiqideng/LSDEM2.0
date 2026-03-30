#pragma once
#include "myUtility/myQua.h"

__device__ __forceinline__ double interpolateLevelSetFunctionValue(const double x, 
const double y, 
const double z, 
const double phi000, 
const double phi100,
const double phi010,
const double phi110,
const double phi001,
const double phi101,
const double phi011,
const double phi111)
{
    // Standard trilinear weights
    const double wx0 = 1.0 - x;
    const double wx1 = x;
    const double wy0 = 1.0 - y;
    const double wy1 = y;
    const double wz0 = 1.0 - z;
    const double wz1 = z;

    return phi000 * wx0 * wy0 * wz0 +
    phi100 * wx1 * wy0 * wz0 +
    phi010 * wx0 * wy1 * wz0 +
    phi110 * wx1 * wy1 * wz0 +
    phi001 * wx0 * wy0 * wz1 +
    phi101 * wx1 * wy0 * wz1 +
    phi011 * wx0 * wy1 * wz1 +
    phi111 * wx1 * wy1 * wz1;
}

__device__ __forceinline__ double3 interpolateLevelSetFunctionGradient(const int x, 
const double y, 
const double z, 
const double phi000, 
const double phi100,
const double phi010,
const double phi110,
const double phi001,
const double phi101,
const double phi011,
const double phi111)
{
    const double wx0 = 1.0 - x;
    const double wx1 = x;
    const double wy0 = 1.0 - y;
    const double wy1 = y;
    const double wz0 = 1.0 - z;
    const double wz1 = z;

    // dphi/dx_normalized
    const double dphidx_n =
    (phi100 - phi000) * wy0 * wz0 +
    (phi110 - phi010) * wy1 * wz0 +
    (phi101 - phi001) * wy0 * wz1 +
    (phi111 - phi011) * wy1 * wz1;

    // dphi/dy_normalized
    const double dphidy_n =
    (phi010 - phi000) * wx0 * wz0 +
    (phi110 - phi100) * wx1 * wz0 +
    (phi011 - phi001) * wx0 * wz1 +
    (phi111 - phi101) * wx1 * wz1;

    // dphi/dz_normalized
    const double dphidz_n =
    (phi001 - phi000) * wx0 * wy0 +
    (phi101 - phi100) * wx1 * wy0 +
    (phi011 - phi010) * wx0 * wy1 +
    (phi111 - phi110) * wx1 * wy1;

    return make_double3(dphidx_n,
    dphidy_n,
    dphidz_n);
}

extern "C" void launchBuildLevelSetBoundaryNodeInteractions1st(double3* localPosition_bNode,
int* particleID_bNode,

int* boundaryNodeNeighborCount,

double* LSFV_gNode,

double3* position_p,
quaternion* orientation_p,
double* radii_p,
double* inverseGridNodeSpacing_p,
double3* gridNodeLocalOrigin_p,
int3* gridNodeSize_p,
int* gridNodePrefixSum_p,
int* hashIndex_p,

int* spatialGridHashStart,
int* spatialGridHashEnd,

const double3 minBound,
const double3 inverseCellSize,
const int3 gridSize3D,

const size_t numBoundaryNode,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchBuildLevelSetBoundaryNodeInteractions2nd(double3* slidingSpring,
double3* contactPoint,
double3* contactNormal,
double* contactOverlap,
int* boundaryNodePointed,
int* objectPointing,
double3* slidingSpring_old,
int* objectPointing_old,

double3* localPosition_bNode,
int* particleID_bNode,

int* boundaryNodeNeighborPrefixSum,
int* boundaryNodeNeighborPrefixSum_old,

double* LSFV_gNode,

double3* position_p,
quaternion* orientation_p,
double* radii_p,
double* inverseGridNodeSpacing_p,
double3* gridNodeLocalOrigin_p,
int3* gridNodeSize_p,
int* gridNodePrefixSum_p,
int* hashIndex_p,

int* spatialGridHashStart,
int* spatialGridHashEnd,

const double3 minBound,
const double3 inverseCellSize,
const int3 gridSize3D,

const size_t numBoundaryNode,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchBuildLevelSetBoundaryNodeFixedParticleInteractions1st(double3* localPosition_bNode,
int* particleID_bNode,

int* boundaryNodeNeighborCount,

double* LSFV_gNode_fp,

double3* position_p,
quaternion* orientation_p,

double3* position_fp,
quaternion* orientation_fp,
double* inverseGridNodeSpacing_fp,
double3* gridNodeLocalOrigin_fp,
int3* gridNodeSize_fp,
int* gridNodePrefixSum_fp,
int* hashIndex_fp,

int* spatialGridHashStart_f,
int* spatialGridHashEnd_f,

const double3 minBound_f,
const double3 inverseCellSize_f,
const int3 gridSize3D_f,

const size_t numBoundaryNode,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchBuildLevelSetBoundaryNodeFixedParticleInteractions2nd(double3* slidingSpring,
double3* contactPoint,
double3* contactNormal,
double* contactOverlap,
int* boundaryNodePointed,
int* objectPointing,
double3* slidingSpring_old,
int* objectPointing_old,

double3* localPosition_bNode,
int* particleID_bNode,

int* boundaryNodeNeighborPrefixSum,
int* boundaryNodeNeighborPrefixSum_old,

double* LSFV_gNode_fp,

double3* position_p,
quaternion* orientation_p,

double3* position_fp,
quaternion* orientation_fp,
double* inverseGridNodeSpacing_fp,
double3* gridNodeLocalOrigin_fp,
int3* gridNodeSize_fp,
int* gridNodePrefixSum_fp,
int* hashIndex_fp,

int* spatialGridHashStart_f,
int* spatialGridHashEnd_f,

const double3 minBound_f,
const double3 inverseCellSize_f,
const int3 gridSize3D_f,

const size_t numBoundaryNode,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);