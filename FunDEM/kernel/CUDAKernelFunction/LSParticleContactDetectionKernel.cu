#include "LSParticleContactDetectionKernel.cuh"
#include "myUtility/mySpatialGrid.h"

__global__ void countLevelSetBoundaryNodeInteractionsKernel(int* boundaryNodeNeighborCount,
const double3* localPosition_bNode, 
const int* particleID_bNode, 

const double* LSFV_gNode, 

const double3* position_p,
const quaternion* orientation_p,
const double* radii_p,
const double* inverseGridNodeSpacing_p,
const double3* gridNodeLocalOrigin_p,
const int3* gridNodeSize_p,
const int* gridNodePrefixSum_p,
const int* hashIndex_p,

const int* spatialGridHashStart, 
const int* spatialGridHashEnd,

const double3 minBound, 
const double3 inverseCellSize, 
const int3 gridSize3D, 

const size_t numBoundaryNode)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoundaryNode) return;

    int count = 0;

    const int idxA = particleID_bNode[idx];
    const double3 globalPosition_idx = rotateVectorByQuaternion(orientation_p[idxA], localPosition_bNode[idx]) + position_p[idxA];
    int3 gridPositionA = calculateGridPosition(globalPosition_idx, minBound, inverseCellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= gridSize3D.x - 1) { gridPositionA.x = gridSize3D.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= gridSize3D.y - 1) { gridPositionA.y = gridSize3D.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= gridSize3D.z - 1) { gridPositionA.z = gridSize3D.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                const int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                const int hashB = linearIndex3D(gridPositionB, gridSize3D);
                const int startIndex = spatialGridHashStart[hashB];
                if (startIndex == -1) continue;
                const int endIndex = spatialGridHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    const int idxB = hashIndex_p[i];
                    if (idxA >= idxB) continue;
                    const double3 r_idxidxB = globalPosition_idx - position_p[idxB];
                    const double rad_B = radii_p[idxB];
                    if (lengthSquared(r_idxidxB) > rad_B * rad_B) continue;
                    const quaternion oriB = orientation_p[idxB];
                    const double3 localPositionFrameB_idx = reverseRotateVectorByQuaternion(r_idxidxB, oriB);
                    const double3 gridNodeLocalOriginB = gridNodeLocalOrigin_p[idxB];
                    const double invG = inverseGridNodeSpacing_p[idxB];
                    const int3 gridNodeSizeB = gridNodeSize_p[idxB];

                    const double gx = (localPositionFrameB_idx.x - gridNodeLocalOriginB.x) * invG;
                    const double gy = (localPositionFrameB_idx.y - gridNodeLocalOriginB.y) * invG;
                    const double gz = (localPositionFrameB_idx.z - gridNodeLocalOriginB.z) * invG;

                    int i0 = (int)floor(gx);
                    int j0 = (int)floor(gy);
                    int k0 = (int)floor(gz);

                    if (i0 < 0) continue;
                    if (j0 < 0) continue;
                    if (k0 < 0) continue;

                    if (i0 >= gridNodeSizeB.x - 1) continue;
                    if (j0 >= gridNodeSizeB.y - 1) continue;
                    if (k0 >= gridNodeSizeB.z - 1) continue;

                    const int i1 = i0 + 1;
                    const int j1 = j0 + 1;
                    const int k1 = k0 + 1;

                    const double x = gx - static_cast<double>(i0);
                    const double y = gy - static_cast<double>(j0);
                    const double z = gz - static_cast<double>(k0);

                    int gridNodeStartB = 0;
                    if (idxB > 0) gridNodeStartB = gridNodePrefixSum_p[idxB - 1];
                    const double phi000 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j0, k0), gridNodeSizeB)];
                    const double phi100 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j0, k0), gridNodeSizeB)];
                    const double phi010 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j1, k0), gridNodeSizeB)];
                    const double phi110 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j1, k0), gridNodeSizeB)];
                    const double phi001 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j0, k1), gridNodeSizeB)];
                    const double phi101 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j0, k1), gridNodeSizeB)];
                    const double phi011 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j1, k1), gridNodeSizeB)];
                    const double phi111 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j1, k1), gridNodeSizeB)];

                    const double ovelap = -interpolateLevelSetFunctionValue(x, 
                    y, 
                    z, 
                    phi000,
                    phi100,
                    phi010,
                    phi110,
                    phi001,
                    phi101, 
                    phi011,
                    phi111);

                    if (ovelap > 0.) count++;
                }
            }
        }
    }

    boundaryNodeNeighborCount[idx] = count;
}

__global__ void writeLevelSetBoundaryNodeInteractionsKernel(double3* slidingSpring,
double3* contactPoint,
double3* contactNormal,
double* contactOverlap,
int* boundaryNodePointed,
int* objectPointing,
const double3* slidingSpring_old,
const int* objectPointing_old,

const double3* localPosition_bNode, 
const int* particleID_bNode, 

const int* boundaryNodeNeighborPrefixSum, 
const int* boundaryNodeNeighborPrefixSum_old, 

const double* LSFV_gNode, 

const double3* position_p,
const quaternion* orientation_p,
const double* radii_p,
const double* inverseGridNodeSpacing_p,
const double3* gridNodeLocalOrigin_p,
const int3* gridNodeSize_p,
const int* gridNodePrefixSum_p,
const int* hashIndex_p,

const int* spatialGridHashStart, 
const int* spatialGridHashEnd,

const double3 minBound, 
const double3 inverseCellSize, 
const int3 gridSize3D,

const size_t numBoundaryNode)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoundaryNode) return;

    int base_w = 0, base_w_old = 0;
    if (idx > 0)
    {
        base_w = boundaryNodeNeighborPrefixSum[idx - 1];
        base_w_old = boundaryNodeNeighborPrefixSum_old[idx - 1];
    }
    const int end_w = boundaryNodeNeighborPrefixSum[idx];
    if (end_w - base_w == 0) return;
    int end_old = boundaryNodeNeighborPrefixSum_old[idx];

    int count = 0;

    const int idxA = particleID_bNode[idx];
    const double3 globalPosition_idx = rotateVectorByQuaternion(orientation_p[idxA], localPosition_bNode[idx]) + position_p[idxA];
    int3 gridPositionA = calculateGridPosition(globalPosition_idx, minBound, inverseCellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= gridSize3D.x - 1) { gridPositionA.x = gridSize3D.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= gridSize3D.y - 1) { gridPositionA.y = gridSize3D.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= gridSize3D.z - 1) { gridPositionA.z = gridSize3D.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                const int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                const int hashB = linearIndex3D(gridPositionB, gridSize3D);
                const int startIndex = spatialGridHashStart[hashB];
                if (startIndex == -1) continue;
                const int endIndex = spatialGridHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    const int idxB = hashIndex_p[i];
                    if (idxA >= idxB) continue;
                    const double3 r_idxidxB = globalPosition_idx - position_p[idxB];
                    const double rad_B = radii_p[idxB];
                    if (lengthSquared(r_idxidxB) > rad_B * rad_B) continue;
                    const quaternion oriB = orientation_p[idxB];
                    const double3 localPositionFrameB_idx = reverseRotateVectorByQuaternion(r_idxidxB, oriB);
                    const double3 gridNodeLocalOriginB = gridNodeLocalOrigin_p[idxB];
                    const double invG = inverseGridNodeSpacing_p[idxB];
                    const int3 gridNodeSizeB = gridNodeSize_p[idxB];

                    const double gx = (localPositionFrameB_idx.x - gridNodeLocalOriginB.x) * invG;
                    const double gy = (localPositionFrameB_idx.y - gridNodeLocalOriginB.y) * invG;
                    const double gz = (localPositionFrameB_idx.z - gridNodeLocalOriginB.z) * invG;

                    int i0 = (int)floor(gx);
                    int j0 = (int)floor(gy);
                    int k0 = (int)floor(gz);

                    if (i0 < 0) continue;
                    if (j0 < 0) continue;
                    if (k0 < 0) continue;

                    if (i0 >= gridNodeSizeB.x - 1) continue;
                    if (j0 >= gridNodeSizeB.y - 1) continue;
                    if (k0 >= gridNodeSizeB.z - 1) continue;

                    const int i1 = i0 + 1;
                    const int j1 = j0 + 1;
                    const int k1 = k0 + 1;

                    const double x = gx - static_cast<double>(i0);
                    const double y = gy - static_cast<double>(j0);
                    const double z = gz - static_cast<double>(k0);

                    int gridNodeStartB = 0;
                    if (idxB > 0) gridNodeStartB = gridNodePrefixSum_p[idxB - 1];

                    const double phi000 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j0, k0), gridNodeSizeB)];
                    const double phi100 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j0, k0), gridNodeSizeB)];
                    const double phi010 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j1, k0), gridNodeSizeB)];
                    const double phi110 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j1, k0), gridNodeSizeB)];
                    const double phi001 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j0, k1), gridNodeSizeB)];
                    const double phi101 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j0, k1), gridNodeSizeB)];
                    const double phi011 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j1, k1), gridNodeSizeB)];
                    const double phi111 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j1, k1), gridNodeSizeB)];

                    const double ovelap = -interpolateLevelSetFunctionValue(x,
                    y,
                    z,
                    phi000,
                    phi100,
                    phi010,
                    phi110,
                    phi001,
                    phi101,
                    phi011,
                    phi111);

                    if (ovelap > 0.)
                    {
                        const int index_w = base_w + count;
                        double3 n_c = interpolateLevelSetFunctionGradient(x,
                        y,
                        z,
                        phi000,
                        phi100,
                        phi010,
                        phi110,
                        phi001,
                        phi101,
                        phi011,
                        phi111);

                        n_c = rotateVectorByQuaternion(oriB, n_c);
                        n_c = normalize(n_c);
                        contactPoint[index_w] = globalPosition_idx + 0.5 * ovelap * n_c;
                        contactNormal[index_w] = n_c;
                        contactOverlap[index_w] = ovelap;
                        boundaryNodePointed[index_w] = idx;
                        objectPointing[index_w] = idxB;

                        double3 ss = make_double3(0., 0., 0.);
                        for (int j = base_w_old; j < end_old; j++)
                        {
                            if (idxB == objectPointing_old[j])
                            {
                                ss = slidingSpring_old[j];
                                break;
                            }
                        }
                        slidingSpring[index_w] = ss;

                        count++;
                    }
                }
            }
        }
    }
}

__global__ void countLevelSetBoundaryNodeFixedParticleInteractionsKernel(int* boundaryNodeNeighborCount,
const double3* localPosition_bNode, 
const int* particleID_bNode, 

const double* LSFV_gNode_fp, 

const double3* position_p,
const quaternion* orientation_p,

const double3* position_fp,
const quaternion* orientation_fp,
const double* inverseGridNodeSpacing_fp,
const double3* gridNodeLocalOrigin_fp,
const int3* gridNodeSize_fp,
const int* gridNodePrefixSum_fp,
const int* hashIndex_fp,

const int* spatialGridHashStart_f, 
const int* spatialGridHashEnd_f,

const double3 minBound_f, 
const double3 inverseCellSize_f, 
const int3 gridSize3D_f, 

const size_t numBoundaryNode)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoundaryNode) return;

    int count = 0;

    const int idxA = particleID_bNode[idx];
    const double3 globalPosition_idx = rotateVectorByQuaternion(orientation_p[idxA], localPosition_bNode[idx]) + position_p[idxA];
    int3 gridPositionA = calculateGridPosition(globalPosition_idx, minBound_f, inverseCellSize_f);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= gridSize3D_f.x - 1) { gridPositionA.x = gridSize3D_f.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= gridSize3D_f.y - 1) { gridPositionA.y = gridSize3D_f.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= gridSize3D_f.z - 1) { gridPositionA.z = gridSize3D_f.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                const int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                const int hashB = linearIndex3D(gridPositionB, gridSize3D_f);
                const int startIndex = spatialGridHashStart_f[hashB];
                if (startIndex == -1) continue;
                const int endIndex = spatialGridHashEnd_f[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    const int idxB = hashIndex_fp[i];
                    const double3 r_idxidxB = globalPosition_idx - position_fp[idxB];
                    const quaternion oriB = orientation_fp[idxB];
                    const double3 localPositionFrameB_idx = reverseRotateVectorByQuaternion(r_idxidxB, oriB);
                    const double3 gridNodeLocalOriginB = gridNodeLocalOrigin_fp[idxB];
                    const double invG = inverseGridNodeSpacing_fp[idxB];
                    const int3 gridNodeSizeB = gridNodeSize_fp[idxB];

                    const double gx = (localPositionFrameB_idx.x - gridNodeLocalOriginB.x) * invG;
                    const double gy = (localPositionFrameB_idx.y - gridNodeLocalOriginB.y) * invG;
                    const double gz = (localPositionFrameB_idx.z - gridNodeLocalOriginB.z) * invG;

                    int i0 = (int)floor(gx);
                    int j0 = (int)floor(gy);
                    int k0 = (int)floor(gz);

                    if (i0 < 0) continue;
                    if (j0 < 0) continue;
                    if (k0 < 0) continue;

                    if (i0 >= gridNodeSizeB.x - 1) continue;
                    if (j0 >= gridNodeSizeB.y - 1) continue;
                    if (k0 >= gridNodeSizeB.z - 1) continue;

                    const int i1 = i0 + 1;
                    const int j1 = j0 + 1;
                    const int k1 = k0 + 1;

                    const double x = gx - static_cast<double>(i0);
                    const double y = gy - static_cast<double>(j0);
                    const double z = gz - static_cast<double>(k0);

                    int gridNodeStartB = 0;
                    if (idxB > 0) gridNodeStartB = gridNodePrefixSum_fp[idxB - 1];
                    const double phi000 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i0, j0, k0), gridNodeSizeB)];
                    const double phi100 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i1, j0, k0), gridNodeSizeB)];
                    const double phi010 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i0, j1, k0), gridNodeSizeB)];
                    const double phi110 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i1, j1, k0), gridNodeSizeB)];
                    const double phi001 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i0, j0, k1), gridNodeSizeB)];
                    const double phi101 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i1, j0, k1), gridNodeSizeB)];
                    const double phi011 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i0, j1, k1), gridNodeSizeB)];
                    const double phi111 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i1, j1, k1), gridNodeSizeB)];

                    const double ovelap = -interpolateLevelSetFunctionValue(x, 
                    y, 
                    z, 
                    phi000,
                    phi100,
                    phi010,
                    phi110,
                    phi001,
                    phi101, 
                    phi011,
                    phi111);

                    if (ovelap > 0.) count++;
                }
            }
        }
    }

    boundaryNodeNeighborCount[idx] = count;
}

__global__ void writeLevelSetBoundaryNodeFixedParticleInteractionsKernel(double3* slidingSpring,
double3* contactPoint,
double3* contactNormal,
double* contactOverlap,
int* boundaryNodePointed,
int* objectPointing,
const double3* slidingSpring_old,
const int* objectPointing_old,

const double3* localPosition_bNode, 
const int* particleID_bNode, 

const int* boundaryNodeNeighborPrefixSum, 
const int* boundaryNodeNeighborPrefixSum_old, 

const double* LSFV_gNode_fp, 

const double3* position_p,
const quaternion* orientation_p,

const double3* position_fp,
const quaternion* orientation_fp,
const double* inverseGridNodeSpacing_fp,
const double3* gridNodeLocalOrigin_fp,
const int3* gridNodeSize_fp,
const int* gridNodePrefixSum_fp,
const int* hashIndex_fp,

const int* spatialGridHashStart_f, 
const int* spatialGridHashEnd_f,

const double3 minBound_f, 
const double3 inverseCellSize_f, 
const int3 gridSize3D_f,

const size_t numBoundaryNode)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoundaryNode) return;

    int base_w = 0, base_w_old = 0;
    if (idx > 0)
    {
        base_w = boundaryNodeNeighborPrefixSum[idx - 1];
        base_w_old = boundaryNodeNeighborPrefixSum_old[idx - 1];
    }
    const int end_w = boundaryNodeNeighborPrefixSum[idx];
    if (end_w - base_w == 0) return;
    int end_old = boundaryNodeNeighborPrefixSum_old[idx];

    int count = 0;

    const int idxA = particleID_bNode[idx];
    const double3 globalPosition_idx = rotateVectorByQuaternion(orientation_p[idxA], localPosition_bNode[idx]) + position_p[idxA];
    int3 gridPositionA = calculateGridPosition(globalPosition_idx, minBound_f, inverseCellSize_f);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= gridSize3D_f.x - 1) { gridPositionA.x = gridSize3D_f.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= gridSize3D_f.y - 1) { gridPositionA.y = gridSize3D_f.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= gridSize3D_f.z - 1) { gridPositionA.z = gridSize3D_f.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                const int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                const int hashB = linearIndex3D(gridPositionB, gridSize3D_f);
                const int startIndex = spatialGridHashStart_f[hashB];
                if (startIndex == -1) continue;
                const int endIndex = spatialGridHashEnd_f[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    const int idxB = hashIndex_fp[i];
                    const double3 r_idxidxB = globalPosition_idx - position_fp[idxB];
                    const quaternion oriB = orientation_fp[idxB];
                    const double3 localPositionFrameB_idx = reverseRotateVectorByQuaternion(r_idxidxB, oriB);
                    const double3 gridNodeLocalOriginB = gridNodeLocalOrigin_fp[idxB];
                    const double invG = inverseGridNodeSpacing_fp[idxB];
                    const int3 gridNodeSizeB = gridNodeSize_fp[idxB];

                    const double gx = (localPositionFrameB_idx.x - gridNodeLocalOriginB.x) * invG;
                    const double gy = (localPositionFrameB_idx.y - gridNodeLocalOriginB.y) * invG;
                    const double gz = (localPositionFrameB_idx.z - gridNodeLocalOriginB.z) * invG;

                    int i0 = (int)floor(gx);
                    int j0 = (int)floor(gy);
                    int k0 = (int)floor(gz);

                    if (i0 < 0) continue;
                    if (j0 < 0) continue;
                    if (k0 < 0) continue;

                    if (i0 >= gridNodeSizeB.x - 1) continue;
                    if (j0 >= gridNodeSizeB.y - 1) continue;
                    if (k0 >= gridNodeSizeB.z - 1) continue;

                    const int i1 = i0 + 1;
                    const int j1 = j0 + 1;
                    const int k1 = k0 + 1;

                    const double x = gx - static_cast<double>(i0);
                    const double y = gy - static_cast<double>(j0);
                    const double z = gz - static_cast<double>(k0);

                    int gridNodeStartB = 0;
                    if (idxB > 0) gridNodeStartB = gridNodePrefixSum_fp[idxB - 1];

                    const double phi000 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i0, j0, k0), gridNodeSizeB)];
                    const double phi100 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i1, j0, k0), gridNodeSizeB)];
                    const double phi010 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i0, j1, k0), gridNodeSizeB)];
                    const double phi110 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i1, j1, k0), gridNodeSizeB)];
                    const double phi001 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i0, j0, k1), gridNodeSizeB)];
                    const double phi101 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i1, j0, k1), gridNodeSizeB)];
                    const double phi011 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i0, j1, k1), gridNodeSizeB)];
                    const double phi111 = LSFV_gNode_fp[gridNodeStartB + linearIndex3D(make_int3(i1, j1, k1), gridNodeSizeB)];

                    const double ovelap = -interpolateLevelSetFunctionValue(x,
                    y,
                    z,
                    phi000,
                    phi100,
                    phi010,
                    phi110,
                    phi001,
                    phi101,
                    phi011,
                    phi111);

                    if (ovelap > 0.)
                    {
                        const int index_w = base_w + count;
                        double3 n_c = interpolateLevelSetFunctionGradient(x,
                        y,
                        z,
                        phi000,
                        phi100,
                        phi010,
                        phi110,
                        phi001,
                        phi101,
                        phi011,
                        phi111);

                        n_c = rotateVectorByQuaternion(oriB, n_c);
                        n_c = normalize(n_c);
                        contactPoint[index_w] = globalPosition_idx + 0.5 * ovelap * n_c;
                        contactNormal[index_w] = n_c;
                        contactOverlap[index_w] = ovelap;
                        boundaryNodePointed[index_w] = idx;
                        objectPointing[index_w] = idxB;

                        double3 ss = make_double3(0., 0., 0.);
                        for (int j = base_w_old; j < end_old; j++)
                        {
                            if (idxB == objectPointing_old[j])
                            {
                                ss = slidingSpring_old[j];
                                break;
                            }
                        }
                        slidingSpring[index_w] = ss;

                        count++;
                    }
                }
            }
        }
    }
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
cudaStream_t stream)
{
    countLevelSetBoundaryNodeInteractionsKernel<<<gridD, blockD, 0, stream>>>(boundaryNodeNeighborCount,
    localPosition_bNode,
    particleID_bNode,

    LSFV_gNode,

    position_p,
    orientation_p,
    radii_p,
    inverseGridNodeSpacing_p,
    gridNodeLocalOrigin_p,
    gridNodeSize_p,
    gridNodePrefixSum_p,
    hashIndex_p,

    spatialGridHashStart,
    spatialGridHashEnd,

    minBound,
    inverseCellSize,
    gridSize3D,

    numBoundaryNode);
}

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
cudaStream_t stream)
{
    writeLevelSetBoundaryNodeInteractionsKernel<<<gridD, blockD, 0, stream>>>(slidingSpring,
    contactPoint,
    contactNormal,
    contactOverlap,
    boundaryNodePointed,
    objectPointing,
    slidingSpring_old,
    objectPointing_old, 

    localPosition_bNode,
    particleID_bNode,

    boundaryNodeNeighborPrefixSum,
    boundaryNodeNeighborPrefixSum_old,

    LSFV_gNode,

    position_p,
    orientation_p,
    radii_p,
    inverseGridNodeSpacing_p,
    gridNodeLocalOrigin_p,
    gridNodeSize_p,
    gridNodePrefixSum_p,
    hashIndex_p,

    spatialGridHashStart,
    spatialGridHashEnd,

    minBound,
    inverseCellSize,
    gridSize3D,

    numBoundaryNode);
}

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
cudaStream_t stream)
{
    countLevelSetBoundaryNodeFixedParticleInteractionsKernel<<<gridD, blockD, 0, stream>>>(boundaryNodeNeighborCount,
    localPosition_bNode,
    particleID_bNode,

    LSFV_gNode_fp,

    position_p,
    orientation_p,

    position_fp,
    orientation_fp,
    inverseGridNodeSpacing_fp,
    gridNodeLocalOrigin_fp,
    gridNodeSize_fp,
    gridNodePrefixSum_fp,
    hashIndex_fp,

    spatialGridHashStart_f,
    spatialGridHashEnd_f,

    minBound_f,
    inverseCellSize_f,
    gridSize3D_f,

    numBoundaryNode);
}

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
cudaStream_t stream)
{
    writeLevelSetBoundaryNodeFixedParticleInteractionsKernel<<<gridD, blockD, 0, stream>>>(slidingSpring,
    contactPoint,
    contactNormal,
    contactOverlap,
    boundaryNodePointed,
    objectPointing,
    slidingSpring_old,
    objectPointing_old, 

    localPosition_bNode,
    particleID_bNode,

    boundaryNodeNeighborPrefixSum,
    boundaryNodeNeighborPrefixSum_old,

    LSFV_gNode_fp,

    position_p,
    orientation_p,

    position_fp,
    orientation_fp,
    inverseGridNodeSpacing_fp,
    gridNodeLocalOrigin_fp,
    gridNodeSize_fp,
    gridNodePrefixSum_fp,
    hashIndex_fp,

    spatialGridHashStart_f,
    spatialGridHashEnd_f,

    minBound_f,
    inverseCellSize_f,
    gridSize3D_f,

    numBoundaryNode);
}
