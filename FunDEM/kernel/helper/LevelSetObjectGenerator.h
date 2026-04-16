// NOTE:
// Most of the code on this page was generated with the assistance of AI tools.

#pragma once
#include "CUDAKernelFunction/myUtility/myVec.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>

namespace LevelSetObject
{
    //=========================================================================
    // Utility
    //=========================================================================
    namespace Utility
    {
        inline double safeAbsPow(const double value,
                                 const double exponent)
        {
            return std::pow(std::fabs(value), exponent);
        }

        inline double signNoZero(const double value)
        {
            return (value >= 0.0) ? 1.0 : -1.0;
        }

        inline double clampValue(const double value,
                                 const double lowerBound,
                                 const double upperBound)
        {
            return std::max(lowerBound, std::min(value, upperBound));
        }

        inline void applyLevelSetSign(double& implicitFunctionValue,
                                      double3& implicitFunctionGradient,
                                      const bool levelSetPositiveInside)
        {
            if (levelSetPositiveInside)
            {
                implicitFunctionValue = -implicitFunctionValue;
                implicitFunctionGradient = -implicitFunctionGradient;
            }
        }
    }

    //=========================================================================
    // Grid info
    //=========================================================================
    class GridInfo
    {
    public:
        // =========================
        // Fields
        // =========================
        // Level set sign convention:
        //   value < 0 : inside particle
        //   value > 0 : outside particle
        //   value = 0 : on particle surface
        double3 gridOrigin{0.0, 0.0, 0.0};
        int3 gridNodeSize{0, 0, 0};
        double gridNodeSpacing{0.0};
        std::vector<double> gridNodeLevelSetFunctionValue;
    };

    //=========================================================================
    // Base
    //=========================================================================
    class Base
    {
    protected:
        // =========================
        // Fields
        // =========================
        GridInfo gridInfo_;

    public:
        // =========================
        // Rule of Five
        // =========================
        Base() = default;
        virtual ~Base() = default;
        Base(const Base&) = default;
        Base(Base&&) noexcept = default;
        Base& operator=(const Base&) = default;
        Base& operator=(Base&&) noexcept = default;

        // =========================
        // Host operations
        // =========================
        void clearGrid()
        {
            gridInfo_ = GridInfo{};
        }

        bool isGridBuilt() const
        {
            return !gridInfo_.gridNodeLevelSetFunctionValue.empty();
        }

        // =========================
        // Virtual interfaces
        // =========================
        virtual bool isValid() const = 0;

        virtual void evaluateImplicitFunctionValueAndGradient(double& implicitFunctionValue,
                                                            double3& implicitFunctionGradient,
                                                            const double3& point) const = 0;

        virtual double3 boundingBoxMin() const = 0;
        virtual double3 boundingBoxMax() const = 0;

        // =========================
        // Build
        // =========================
        void buildGrid(const double backgroundGridNodeSpacing,
                    const int backgroundGridPaddingLayers = 2)
        {
            clearGrid();

            if (!isValid()) return;
            if (backgroundGridNodeSpacing <= 0.0) return;

            const double gridSpacing = backgroundGridNodeSpacing;
            gridInfo_.gridNodeSpacing = gridSpacing;

            const double3 particleBoundingBoxMin = boundingBoxMin();
            const double3 particleBoundingBoxMax = boundingBoxMax();

            const double paddingDistance =
                double(std::max(1, backgroundGridPaddingLayers)) * gridSpacing;

            double3 backgroundGridMin = make_double3(particleBoundingBoxMin.x - paddingDistance,
                                                    particleBoundingBoxMin.y - paddingDistance,
                                                    particleBoundingBoxMin.z - paddingDistance);

            double3 backgroundGridMax = make_double3(particleBoundingBoxMax.x + paddingDistance,
                                                    particleBoundingBoxMax.y + paddingDistance,
                                                    particleBoundingBoxMax.z + paddingDistance);

            const auto snapDownToGrid = [&](const double coordinate)
            {
                return gridSpacing * std::floor(coordinate / gridSpacing);
            };

            const auto snapUpToGrid = [&](const double coordinate)
            {
                return gridSpacing * std::ceil(coordinate / gridSpacing);
            };

            backgroundGridMin.x = snapDownToGrid(backgroundGridMin.x);
            backgroundGridMin.y = snapDownToGrid(backgroundGridMin.y);
            backgroundGridMin.z = snapDownToGrid(backgroundGridMin.z);

            backgroundGridMax.x = snapUpToGrid(backgroundGridMax.x);
            backgroundGridMax.y = snapUpToGrid(backgroundGridMax.y);
            backgroundGridMax.z = snapUpToGrid(backgroundGridMax.z);

            const int numGridNodesX =
                int(std::floor((backgroundGridMax.x - backgroundGridMin.x) / gridSpacing + 0.5)) + 1;
            const int numGridNodesY =
                int(std::floor((backgroundGridMax.y - backgroundGridMin.y) / gridSpacing + 0.5)) + 1;
            const int numGridNodesZ =
                int(std::floor((backgroundGridMax.z - backgroundGridMin.z) / gridSpacing + 0.5)) + 1;

            if (numGridNodesX <= 0 || numGridNodesY <= 0 || numGridNodesZ <= 0) return;

            gridInfo_.gridOrigin = backgroundGridMin;
            gridInfo_.gridNodeSize = make_int3(numGridNodesX, numGridNodesY, numGridNodesZ);
            gridInfo_.gridNodeLevelSetFunctionValue.assign(
                size_t(numGridNodesX) * size_t(numGridNodesY) * size_t(numGridNodesZ),
                0.0);

            const double minGradientNormForSignedDistanceApproximation = 1e-12;

            for (int gridNodeK = 0; gridNodeK < numGridNodesZ; ++gridNodeK)
            {
                const double gridNodeZ = backgroundGridMin.z + double(gridNodeK) * gridSpacing;

                for (int gridNodeJ = 0; gridNodeJ < numGridNodesY; ++gridNodeJ)
                {
                    const double gridNodeY = backgroundGridMin.y + double(gridNodeJ) * gridSpacing;

                    for (int gridNodeI = 0; gridNodeI < numGridNodesX; ++gridNodeI)
                    {
                        const double gridNodeX = backgroundGridMin.x + double(gridNodeI) * gridSpacing;

                        const double3 point = make_double3(gridNodeX, gridNodeY, gridNodeZ);

                        double implicitFunctionValue = 0.0;
                        double3 implicitFunctionGradient = make_double3(0.0, 0.0, 0.0);

                        evaluateImplicitFunctionValueAndGradient(implicitFunctionValue,
                                                                implicitFunctionGradient,
                                                                point);

                        const double implicitFunctionGradientNorm = length(implicitFunctionGradient);

                        double signedDistanceLikeLevelSetValue = 0.0;

                        if (implicitFunctionGradientNorm > minGradientNormForSignedDistanceApproximation)
                        {
                            signedDistanceLikeLevelSetValue =
                                implicitFunctionValue / implicitFunctionGradientNorm;
                        }
                        else
                        {
                            signedDistanceLikeLevelSetValue =
                                (implicitFunctionValue <= 0.0 ? -1.0 : 1.0) * 0.5 * gridSpacing;
                        }

                        gridInfo_.gridNodeLevelSetFunctionValue[
                            linearIndex3D(make_int3(gridNodeI, gridNodeJ, gridNodeK),
                                        gridInfo_.gridNodeSize)] = signedDistanceLikeLevelSetValue;
                    }
                }
            }
        }

        void buildGridByResolution(const int backgroundGridNodesPerParticleDiameter = 50,
                                const int backgroundGridPaddingLayers = 2)
        {
            if (!isValid()) return;
            if (backgroundGridNodesPerParticleDiameter <= 0) return;

            const double3 particleBoundingBoxMin = boundingBoxMin();
            const double3 particleBoundingBoxMax = boundingBoxMax();

            const double particleBoundingBoxSizeX =
                particleBoundingBoxMax.x - particleBoundingBoxMin.x;
            const double particleBoundingBoxSizeY =
                particleBoundingBoxMax.y - particleBoundingBoxMin.y;
            const double particleBoundingBoxSizeZ =
                particleBoundingBoxMax.z - particleBoundingBoxMin.z;

            const double particleReferenceDiameter =
                std::max(particleBoundingBoxSizeX,
                        std::max(particleBoundingBoxSizeY, particleBoundingBoxSizeZ));

            if (particleReferenceDiameter <= 0.0) return;

            buildGrid(particleReferenceDiameter /
                        double(backgroundGridNodesPerParticleDiameter),
                    backgroundGridPaddingLayers);
        }

        // =========================
        // Output
        // =========================
        void outputGridVTU(const std::string& fileNamePrefix) const
        {
            if (!isGridBuilt()) return;

            const int nx = gridInfo_.gridNodeSize.x;
            const int ny = gridInfo_.gridNodeSize.y;
            const int nz = gridInfo_.gridNodeSize.z;

            if (nx <= 0 || ny <= 0 || nz <= 0) return;

            const size_t N = size_t(nx) * size_t(ny) * size_t(nz);

            if (gridInfo_.gridNodeLevelSetFunctionValue.size() != N) return;

            const std::string fileName = fileNamePrefix + "LSGrid.vtu";

            std::ofstream out(fileName);
            if (!out)
            {
                throw std::runtime_error("Cannot open " + fileName);
            }

            out << "<?xml version=\"1.0\"?>\n";
            out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
            out << "  <UnstructuredGrid>\n";
            out << "    <Piece NumberOfPoints=\"" << N
                << "\" NumberOfCells=\"" << N << "\">\n";

            out << "      <Points>\n";
            out << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";

            for (int k = 0; k < nz; ++k)
            {
                const double z = gridInfo_.gridOrigin.z + double(k) * gridInfo_.gridNodeSpacing;

                for (int j = 0; j < ny; ++j)
                {
                    const double y = gridInfo_.gridOrigin.y + double(j) * gridInfo_.gridNodeSpacing;

                    for (int i = 0; i < nx; ++i)
                    {
                        const double x = gridInfo_.gridOrigin.x + double(i) * gridInfo_.gridNodeSpacing;

                        out << "          "
                            << static_cast<float>(x) << " "
                            << static_cast<float>(y) << " "
                            << static_cast<float>(z) << "\n";
                    }
                }
            }

            out << "        </DataArray>\n";
            out << "      </Points>\n";

            out << "      <Cells>\n";

            out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
            for (size_t id = 0; id < N; ++id)
            {
                out << "          " << static_cast<int>(id) << "\n";
            }
            out << "        </DataArray>\n";

            out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
            for (size_t id = 0; id < N; ++id)
            {
                out << "          " << static_cast<int>(id + 1) << "\n";
            }
            out << "        </DataArray>\n";

            out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
            for (size_t id = 0; id < N; ++id)
            {
                out << "          1\n";
            }
            out << "        </DataArray>\n";

            out << "      </Cells>\n";

            out << "      <PointData Scalars=\"levelSetValue\">\n";

            out << "        <DataArray type=\"Float32\" Name=\"levelSetValue\" format=\"ascii\">\n";
            for (size_t id = 0; id < N; ++id)
            {
                out << "          "
                    << static_cast<float>(gridInfo_.gridNodeLevelSetFunctionValue[id]) << "\n";
            }
            out << "        </DataArray>\n";

            out << "        <DataArray type=\"Int32\" Name=\"gridIJK\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            for (int k = 0; k < nz; ++k)
            {
                for (int j = 0; j < ny; ++j)
                {
                    for (int i = 0; i < nx; ++i)
                    {
                        out << "          " << i << " " << j << " " << k << "\n";
                    }
                }
            }
            out << "        </DataArray>\n";

            out << "      </PointData>\n";

            out << "    </Piece>\n";
            out << "  </UnstructuredGrid>\n";
            out << "</VTKFile>\n";
        }

        // =========================
        // Getters
        // =========================
        const GridInfo& gridInfo() const
        {
            return gridInfo_;
        }

        GridInfo& gridInfo()
        {
            return gridInfo_;
        }
    };

    //=========================================================================
    // Triangle mesh particle (SAH BVH - SAFE NAMING VERSION)
    //=========================================================================
    class TriangleMeshParticle : public Base
    {
    protected:

        // =========================
        // Fields
        // =========================
        std::vector<double3> vertexPosition_;
        std::vector<int3> triangleVertexIndex_;

        bool levelSetPositiveInside_{false};

        double3 boundingBoxMin_{0.0, 0.0, 0.0};
        double3 boundingBoxMax_{0.0, 0.0, 0.0};

        // =========================
        // BVH
        // =========================
        struct BVHNode
        {
            double3 bmin;
            double3 bmax;

            int left = -1;
            int right = -1;

            int start = 0;
            int count = 0;

            bool isLeaf = false;
        };

        std::vector<BVHNode> bvhNodes_;
        std::vector<int> bvhPrimitiveIndices_;
        std::vector<double3> triangleCentroid_;

        static constexpr int NUM_BINS = 12;
        static constexpr int MAX_LEAF_SIZE = 4;

    protected:

        // =========================
        // SAFE vector ops (renamed)
        // =========================
        static inline double3 vecMin(const double3& a, const double3& b)
        {
            return make_double3(
                std::min(a.x, b.x),
                std::min(a.y, b.y),
                std::min(a.z, b.z)
            );
        }

        static inline double3 vecMax(const double3& a, const double3& b)
        {
            return make_double3(
                std::max(a.x, b.x),
                std::max(a.y, b.y),
                std::max(a.z, b.z)
            );
        }

        static inline double3 triCentroid(const double3& a,
                                        const double3& b,
                                        const double3& c)
        {
            return (a + b + c) * (1.0 / 3.0);
        }

        // =========================
        // closest point
        // =========================
        static inline double3 closestPointOnTriangle(const double3& p,
                                                    const double3& a,
                                                    const double3& b,
                                                    const double3& c)
        {
            const double3 ab = b - a;
            const double3 ac = c - a;
            const double3 ap = p - a;

            double d1 = dot(ab, ap);
            double d2 = dot(ac, ap);
            if (d1 <= 0.0 && d2 <= 0.0) return a;

            double3 bp = p - b;
            double d3 = dot(ab, bp);
            double d4 = dot(ac, bp);
            if (d3 >= 0.0 && d4 <= d3) return b;

            double vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
                return a + (d1 / (d1 - d3)) * ab;

            double3 cp = p - c;
            double d5 = dot(ab, cp);
            double d6 = dot(ac, cp);
            if (d6 >= 0.0 && d5 <= d6) return c;

            double vb = d5 * d2 - d1 * d6;
            if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
                return a + (d2 / (d2 - d6)) * ac;

            double va = d3 * d6 - d5 * d4;
            if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
            {
                double3 bc = c - b;
                return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * bc;
            }

            double denom = 1.0 / (va + vb + vc);
            double v = vb * denom;
            double w = vc * denom;

            return a + ab * v + ac * w;
        }

        // =========================
        // triangle bbox
        // =========================
        inline double3 computeTriangleMin(int t) const
        {
            const int3 tri = triangleVertexIndex_[t];
            const double3& a = vertexPosition_[tri.x];
            const double3& b = vertexPosition_[tri.y];
            const double3& c = vertexPosition_[tri.z];
            return vecMin(vecMin(a, b), c);
        }

        inline double3 computeTriangleMax(int t) const
        {
            const int3 tri = triangleVertexIndex_[t];
            const double3& a = vertexPosition_[tri.x];
            const double3& b = vertexPosition_[tri.y];
            const double3& c = vertexPosition_[tri.z];
            return vecMax(vecMax(a, b), c);
        }

        // =========================
        // bbox range
        // =========================
        double3 bboxMinFromRange(int l, int r) const
        {
            double3 mn = computeTriangleMin(bvhPrimitiveIndices_[l]);

            for (int i = l + 1; i < r; ++i)
                mn = vecMin(mn, computeTriangleMin(bvhPrimitiveIndices_[i]));

            return mn;
        }

        double3 bboxMaxFromRange(int l, int r) const
        {
            double3 mx = computeTriangleMax(bvhPrimitiveIndices_[l]);

            for (int i = l + 1; i < r; ++i)
                mx = vecMax(mx, computeTriangleMax(bvhPrimitiveIndices_[i]));

            return mx;
        }

        // =========================
        // SAH split
        // =========================
        int sahSplit(int l, int r, int axis)
        {
            struct Bin
            {
                double3 bmin = make_double3(1e30,1e30,1e30);
                double3 bmax = make_double3(-1e30,-1e30,-1e30);
                int count = 0;
            };

            Bin bins[NUM_BINS];

            double3 sceneMin = bboxMinFromRange(l, r);
            double3 sceneMax = bboxMaxFromRange(l, r);

            double extent =
                (axis == 0) ? sceneMax.x - sceneMin.x :
                (axis == 1) ? sceneMax.y - sceneMin.y :
                            sceneMax.z - sceneMin.z;

            if (extent < 1e-12) return (l + r) / 2;

            for (int i = l; i < r; ++i)
            {
                int tri = bvhPrimitiveIndices_[i];
                double3 c = triangleCentroid_[tri];

                double pos =
                    (axis == 0) ? c.x :
                    (axis == 1) ? c.y :
                                c.z;

                int b = std::min(NUM_BINS - 1,
                        (int)(NUM_BINS * (pos -
                        ((axis==0)?sceneMin.x:(axis==1)?sceneMin.y:sceneMin.z)) / extent));

                b = std::max(0, b);

                double3 mn = computeTriangleMin(tri);
                double3 mx = computeTriangleMax(tri);

                bins[b].bmin = vecMin(bins[b].bmin, mn);
                bins[b].bmax = vecMax(bins[b].bmax, mx);
                bins[b].count++;
            }

            auto SA = [](const double3& mn, const double3& mx)
            {
                double3 e = mx - mn;
                return 2.0 * (e.x*e.y + e.y*e.z + e.z*e.x);
            };

            double bestCost = 1e30;
            int bestSplit = (l + r) / 2;

            for (int i = 1; i < NUM_BINS; ++i)
            {
                double3 lmin = make_double3(1e30,1e30,1e30);
                double3 lmax = make_double3(-1e30,-1e30,-1e30);
                double3 rmin = make_double3(1e30,1e30,1e30);
                double3 rmax = make_double3(-1e30,-1e30,-1e30);

                int lc = 0, rc = 0;

                for (int j = 0; j < i; ++j)
                {
                    if (!bins[j].count) continue;
                    lmin = vecMin(lmin, bins[j].bmin);
                    lmax = vecMax(lmax, bins[j].bmax);
                    lc += bins[j].count;
                }

                for (int j = i; j < NUM_BINS; ++j)
                {
                    if (!bins[j].count) continue;
                    rmin = vecMin(rmin, bins[j].bmin);
                    rmax = vecMax(rmax, bins[j].bmax);
                    rc += bins[j].count;
                }

                double cost = lc * SA(lmin,lmax) + rc * SA(rmin,rmax);

                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestSplit = i;
                }
            }

            return bestSplit;
        }

        // =========================
        // build BVH
        // =========================
        int buildBVH(int l, int r)
        {
            BVHNode node;
            node.bmin = bboxMinFromRange(l, r);
            node.bmax = bboxMaxFromRange(l, r);

            size_t idx = bvhNodes_.size();
            bvhNodes_.push_back(node);

            if (r - l <= MAX_LEAF_SIZE)
            {
                bvhNodes_[idx].isLeaf = true;
                bvhNodes_[idx].start = l;
                bvhNodes_[idx].count = r - l;
                return idx;
            }

            double3 e = node.bmax - node.bmin;

            int axis =
                (e.x > e.y && e.x > e.z) ? 0 :
                (e.y > e.z ? 1 : 2);

            int split = sahSplit(l, r, axis);

            if (split <= (int)l || split >= (int)r)
                split = (l + r) / 2;

            int left = buildBVH(l, split);
            int right = buildBVH(split, r);

            bvhNodes_[idx].left = left;
            bvhNodes_[idx].right = right;

            return idx;
        }

        void rebuildBVH()
        {
            bvhNodes_.clear();
            bvhPrimitiveIndices_.resize(triangleVertexIndex_.size());
            triangleCentroid_.resize(triangleVertexIndex_.size());

            for (size_t i = 0; i < triangleVertexIndex_.size(); ++i)
            {
                bvhPrimitiveIndices_[i] = i;

                const int3 t = triangleVertexIndex_[i];
                triangleCentroid_[i] =
                    triCentroid(vertexPosition_[t.x],
                                vertexPosition_[t.y],
                                vertexPosition_[t.z]);
            }

            if (!bvhPrimitiveIndices_.empty())
                buildBVH(0, bvhPrimitiveIndices_.size());
        }

        // =========================
        // traversal
        // =========================
        struct Hit
        {
            double dist2 = std::numeric_limits<double>::max();
            double3 cp;
        };

        void traverse(int nodeId, const double3& p, Hit& hit) const
        {
            const BVHNode& n = bvhNodes_[nodeId];

            auto boxDist2 = [&](const double3& mn, const double3& mx)
            {
                double dx = std::max({mn.x - p.x, 0.0, p.x - mx.x});
                double dy = std::max({mn.y - p.y, 0.0, p.y - mx.y});
                double dz = std::max({mn.z - p.z, 0.0, p.z - mx.z});
                return dx * dx + dy * dy + dz * dz;
            };

            if (boxDist2(n.bmin, n.bmax) > hit.dist2) return;

            if (n.isLeaf)
            {
                for (int i = 0; i < n.count; ++i)
                {
                    int tri = bvhPrimitiveIndices_[n.start + i];
                    const int3 t = triangleVertexIndex_[tri];

                    const double3& a = vertexPosition_[t.x];
                    const double3& b = vertexPosition_[t.y];
                    const double3& c = vertexPosition_[t.z];

                    double3 cp = closestPointOnTriangle(p,a,b,c);
                    double d2 = lengthSquared(p - cp);

                    if (d2 < hit.dist2)
                    {
                        hit.dist2 = d2;
                        hit.cp = cp;
                    }
                }
                return;
            }

            traverse(n.left, p, hit);
            traverse(n.right, p, hit);
        }

        double3 closestPointBVH(const double3& p) const
        {
            Hit h;
            traverse(0, p, h);
            return h.cp;
        }

        // =========================
        // inside test (unchanged)
        // =========================
        static inline double solidAngleFromPointToTriangle(const double3& p,
                                                            const double3& a,
                                                            const double3& b,
                                                            const double3& c)
        {
            const double3 r0 = a - p;
            const double3 r1 = b - p;
            const double3 r2 = c - p;

            const double l0 = length(r0);
            const double l1 = length(r1);
            const double l2 = length(r2);

            if (l0 < 1e-30 || l1 < 1e-30 || l2 < 1e-30)
                return 0.0;

            const double numerator = dot(r0, cross(r1, r2));
            const double denominator =
                l0 * l1 * l2 +
                dot(r0, r1) * l2 +
                dot(r1, r2) * l0 +
                dot(r2, r0) * l1;

            return 2.0 * std::atan2(numerator, denominator);
        }

        bool isPointInsideClosedTriangleMesh(const double3& point) const
        {
            if (!isValid()) return false;

            double sum = 0.0;

            for (size_t i = 0; i < triangleVertexIndex_.size(); ++i)
            {
                const int3 t = triangleVertexIndex_[i];

                const double3& a = vertexPosition_[t.x];
                const double3& b = vertexPosition_[t.y];
                const double3& c = vertexPosition_[t.z];

                sum += solidAngleFromPointToTriangle(point,a,b,c);
            }

            return std::fabs(sum / (4.0 * pi())) > 0.5;
        }

        void updateBoundingBoxFromMesh()
        {
            if (vertexPosition_.empty())
            {
                boundingBoxMin_ = make_double3(0,0,0);
                boundingBoxMax_ = make_double3(0,0,0);
                return;
            }

            boundingBoxMin_ = vertexPosition_[0];
            boundingBoxMax_ = vertexPosition_[0];

            for (size_t i = 1; i < vertexPosition_.size(); ++i)
            {
                boundingBoxMin_ = vecMin(boundingBoxMin_, vertexPosition_[i]);
                boundingBoxMax_ = vecMax(boundingBoxMax_, vertexPosition_[i]);
            }
        }

        void setMeshInternal(const std::vector<double3>& v,
                            const std::vector<int3>& f)
        {
            vertexPosition_ = v;
            triangleVertexIndex_ = f;

            updateBoundingBoxFromMesh();
            rebuildBVH();
            clearGrid();
        }

    public:

        TriangleMeshParticle() = default;

        TriangleMeshParticle(const std::vector<double3>& v,
                            const std::vector<int3>& f)
            : vertexPosition_(v),
            triangleVertexIndex_(f)
        {
            updateBoundingBoxFromMesh();
            rebuildBVH();
        }

        ~TriangleMeshParticle() override = default;

        void setMesh(const std::vector<double3>& v,
                    const std::vector<int3>& f)
        {
            setMeshInternal(v,f);
        }

        void setLevelSetPositiveInside(bool v)
        {
            levelSetPositiveInside_ = v;
        }

        void clearMesh()
        {
            vertexPosition_.clear();
            triangleVertexIndex_.clear();
            bvhNodes_.clear();
            bvhPrimitiveIndices_.clear();
            triangleCentroid_.clear();

            updateBoundingBoxFromMesh();
            clearGrid();
        }

        bool isValid() const override
        {
            if (vertexPosition_.empty()) return false;
            if (triangleVertexIndex_.empty()) return false;

            int n = (int)vertexPosition_.size();

            for (auto& t : triangleVertexIndex_)
            {
                if (t.x < 0 || t.y < 0 || t.z < 0) return false;
                if (t.x >= n || t.y >= n || t.z >= n) return false;
                if (t.x == t.y || t.y == t.z || t.z == t.x) return false;
            }
            return true;
        }

        void evaluateImplicitFunctionValueAndGradient(double& value,
                                                    double3& grad,
                                                    const double3& p) const override
        {
            if (!isValid())
            {
                value = 1.0;
                grad = make_double3(0,0,0);
                return;
            }

            double3 cp = closestPointBVH(p);
            double d = length(p - cp);

            bool inside = isPointInsideClosedTriangleMesh(p);

            value = inside ? -d : d;

            if (d > 1e-30)
            {
                double3 g = (p - cp) / d;
                grad = inside ? -g : g;
            }
            else grad = make_double3(0,0,0);

            Utility::applyLevelSetSign(value, grad, levelSetPositiveInside_);
        }

        double3 boundingBoxMin() const override { return boundingBoxMin_; }
        double3 boundingBoxMax() const override { return boundingBoxMax_; }

        // =========================
        // Getters
        // =========================
        const std::vector<double3>& vertexPosition() const
        {
            return vertexPosition_;
        }

        std::vector<double3>& vertexPosition()
        {
            return vertexPosition_;
        }

        const std::vector<int3>& triangleVertexIndex() const
        {
            return triangleVertexIndex_;
        }

        std::vector<int3>& triangleVertexIndex()
        {
            return triangleVertexIndex_;
        }

        const double3& meshBoundingBoxMin() const
        {
            return boundingBoxMin_;
        }

        const double3& meshBoundingBoxMax() const
        {
            return boundingBoxMax_;
        }
    };

    //=========================================================================
    // Superellipsoid particle based on triangle mesh (icosphere-style remeshing)
    //=========================================================================
    class Superellipsoid : public TriangleMeshParticle
    {
    protected:
        // =========================
        // Fields
        // =========================
        double rx_{1.0};
        double ry_{1.0};
        double rz_{1.0};
        double ee_{1.0};
        double en_{1.0};

        int subdivisionLevel_{4}; // recommended: 0 ~ 6

    protected:
        // =========================
        // Helpers
        // =========================
        static int clampSubdivisionLevel(const int subdivisionLevel)
        {
            return std::max(0, std::min(subdivisionLevel, 8));
        }

        bool parametersAreValid() const
        {
            return (rx_ > 0.0 &&
                    ry_ > 0.0 &&
                    rz_ > 0.0 &&
                    ee_ > 0.0 &&
                    en_ > 0.0 &&
                    subdivisionLevel_ >= 0);
        }

        double implicitFunctionValueOnly(const double3& point) const
        {
            const double inverseRadiusX = 1.0 / rx_;
            const double inverseRadiusY = 1.0 / ry_;
            const double inverseRadiusZ = 1.0 / rz_;

            const double normalizedAbsoluteX = std::fabs(point.x * inverseRadiusX);
            const double normalizedAbsoluteY = std::fabs(point.y * inverseRadiusY);
            const double normalizedAbsoluteZ = std::fabs(point.z * inverseRadiusZ);

            const double exponentXY = 2.0 / ee_;
            const double exponentZ = 2.0 / en_;
            const double outerExponent = ee_ / en_;

            const double xTerm = Utility::safeAbsPow(normalizedAbsoluteX, exponentXY);
            const double yTerm = Utility::safeAbsPow(normalizedAbsoluteY, exponentXY);
            const double xyTermSum = xTerm + yTerm;

            const double xyBlockValue = Utility::safeAbsPow(xyTermSum, outerExponent);
            const double zBlockValue = Utility::safeAbsPow(normalizedAbsoluteZ, exponentZ);

            return xyBlockValue + zBlockValue - 1.0;
        }

        double3 projectUnitDirectionToSuperellipsoid(const double3& direction) const
        {
            const double directionLengthSquared = lengthSquared(direction);
            if (directionLengthSquared <= 1e-30)
            {
                return make_double3(0.0, 0.0, 0.0);
            }

            const double3 dir = direction / std::sqrt(directionLengthSquared);

            const double exponentXY = 2.0 / ee_;
            const double exponentZ = 2.0 / en_;
            const double outerExponent = ee_ / en_;

            const double xTerm =
                Utility::safeAbsPow(std::fabs(dir.x / rx_), exponentXY);
            const double yTerm =
                Utility::safeAbsPow(std::fabs(dir.y / ry_), exponentXY);
            const double zTerm =
                Utility::safeAbsPow(std::fabs(dir.z / rz_), exponentZ);

            const double scaleMeasure =
                Utility::safeAbsPow(xTerm + yTerm, outerExponent) + zTerm;

            if (scaleMeasure <= 1e-30)
            {
                return make_double3(0.0, 0.0, 0.0);
            }

            const double radialScale = std::pow(scaleMeasure, -0.5 * en_);
            return dir * radialScale;
        }

        static int getMidpointIndex(std::vector<double3>& vertexPosition,
                                    std::unordered_map<unsigned long long, int>& midpointCache,
                                    const int index0,
                                    const int index1)
        {
            const int a = std::min(index0, index1);
            const int b = std::max(index0, index1);

            const unsigned long long key =
                (static_cast<unsigned long long>(static_cast<unsigned int>(a)) << 32) |
                static_cast<unsigned int>(b);

            const auto it = midpointCache.find(key);
            if (it != midpointCache.end()) return it->second;

            const double3 midpoint = normalize(0.5 * (vertexPosition[a] + vertexPosition[b]));

            const int newIndex = static_cast<int>(vertexPosition.size());
            vertexPosition.push_back(midpoint);
            midpointCache.emplace(key, newIndex);
            return newIndex;
        }

        void buildUnitIcosahedron(std::vector<double3>& vertexPosition,
                                std::vector<int3>& triangleVertexIndex) const
        {
            vertexPosition.clear();
            triangleVertexIndex.clear();

            const double goldenRatio = 0.5 * (1.0 + std::sqrt(5.0));

            vertexPosition.reserve(12);
            triangleVertexIndex.reserve(20);

            vertexPosition.push_back(normalize(make_double3(-1.0,  goldenRatio, 0.0)));
            vertexPosition.push_back(normalize(make_double3( 1.0,  goldenRatio, 0.0)));
            vertexPosition.push_back(normalize(make_double3(-1.0, -goldenRatio, 0.0)));
            vertexPosition.push_back(normalize(make_double3( 1.0, -goldenRatio, 0.0)));

            vertexPosition.push_back(normalize(make_double3(0.0, -1.0,  goldenRatio)));
            vertexPosition.push_back(normalize(make_double3(0.0,  1.0,  goldenRatio)));
            vertexPosition.push_back(normalize(make_double3(0.0, -1.0, -goldenRatio)));
            vertexPosition.push_back(normalize(make_double3(0.0,  1.0, -goldenRatio)));

            vertexPosition.push_back(normalize(make_double3( goldenRatio, 0.0, -1.0)));
            vertexPosition.push_back(normalize(make_double3( goldenRatio, 0.0,  1.0)));
            vertexPosition.push_back(normalize(make_double3(-goldenRatio, 0.0, -1.0)));
            vertexPosition.push_back(normalize(make_double3(-goldenRatio, 0.0,  1.0)));

            triangleVertexIndex.push_back(make_int3(0, 11, 5));
            triangleVertexIndex.push_back(make_int3(0, 5, 1));
            triangleVertexIndex.push_back(make_int3(0, 1, 7));
            triangleVertexIndex.push_back(make_int3(0, 7, 10));
            triangleVertexIndex.push_back(make_int3(0, 10, 11));

            triangleVertexIndex.push_back(make_int3(1, 5, 9));
            triangleVertexIndex.push_back(make_int3(5, 11, 4));
            triangleVertexIndex.push_back(make_int3(11, 10, 2));
            triangleVertexIndex.push_back(make_int3(10, 7, 6));
            triangleVertexIndex.push_back(make_int3(7, 1, 8));

            triangleVertexIndex.push_back(make_int3(3, 9, 4));
            triangleVertexIndex.push_back(make_int3(3, 4, 2));
            triangleVertexIndex.push_back(make_int3(3, 2, 6));
            triangleVertexIndex.push_back(make_int3(3, 6, 8));
            triangleVertexIndex.push_back(make_int3(3, 8, 9));

            triangleVertexIndex.push_back(make_int3(4, 9, 5));
            triangleVertexIndex.push_back(make_int3(2, 4, 11));
            triangleVertexIndex.push_back(make_int3(6, 2, 10));
            triangleVertexIndex.push_back(make_int3(8, 6, 7));
            triangleVertexIndex.push_back(make_int3(9, 8, 1));
        }

        void orientTrianglesOutward(const std::vector<double3>& vertexPosition,
                                    std::vector<int3>& triangleVertexIndex) const
        {
            for (size_t triangleIndex = 0; triangleIndex < triangleVertexIndex.size(); ++triangleIndex)
            {
                int3& triangle = triangleVertexIndex[triangleIndex];

                const double3& v0 = vertexPosition[triangle.x];
                const double3& v1 = vertexPosition[triangle.y];
                const double3& v2 = vertexPosition[triangle.z];

                const double3 faceCenter = (v0 + v1 + v2) / 3.0;
                const double3 faceNormal = cross(v1 - v0, v2 - v0);

                if (dot(faceNormal, faceCenter) < 0.0)
                {
                    std::swap(triangle.y, triangle.z);
                }
            }
        }

        void subdivideUnitSphereMesh(std::vector<double3>& vertexPosition,
                                    std::vector<int3>& triangleVertexIndex) const
        {
            std::unordered_map<unsigned long long, int> midpointCache;
            midpointCache.reserve(triangleVertexIndex.size() * 3);

            std::vector<int3> refinedTriangleVertexIndex;
            refinedTriangleVertexIndex.reserve(triangleVertexIndex.size() * 4);

            for (size_t triangleIndex = 0; triangleIndex < triangleVertexIndex.size(); ++triangleIndex)
            {
                const int3 triangle = triangleVertexIndex[triangleIndex];

                const int i0 = triangle.x;
                const int i1 = triangle.y;
                const int i2 = triangle.z;

                const int i01 = getMidpointIndex(vertexPosition, midpointCache, i0, i1);
                const int i12 = getMidpointIndex(vertexPosition, midpointCache, i1, i2);
                const int i20 = getMidpointIndex(vertexPosition, midpointCache, i2, i0);

                refinedTriangleVertexIndex.push_back(make_int3(i0,  i01, i20));
                refinedTriangleVertexIndex.push_back(make_int3(i1,  i12, i01));
                refinedTriangleVertexIndex.push_back(make_int3(i2,  i20, i12));
                refinedTriangleVertexIndex.push_back(make_int3(i01, i12, i20));
            }

            triangleVertexIndex.swap(refinedTriangleVertexIndex);
        }

        void projectSphereMeshToSuperellipsoid(std::vector<double3>& vertexPosition) const
        {
            for (size_t vertexIndex = 0; vertexIndex < vertexPosition.size(); ++vertexIndex)
            {
                vertexPosition[vertexIndex] = projectUnitDirectionToSuperellipsoid(vertexPosition[vertexIndex]);
            }
        }

        void rebuildMesh()
        {
            clearMesh();

            if (!parametersAreValid()) return;

            std::vector<double3> vertexPosition;
            std::vector<int3> triangleVertexIndex;

            buildUnitIcosahedron(vertexPosition, triangleVertexIndex);
            orientTrianglesOutward(vertexPosition, triangleVertexIndex);

            for (int level = 0; level < subdivisionLevel_; ++level)
            {
                subdivideUnitSphereMesh(vertexPosition, triangleVertexIndex);
                orientTrianglesOutward(vertexPosition, triangleVertexIndex);
            }

            projectSphereMeshToSuperellipsoid(vertexPosition);
            orientTrianglesOutward(vertexPosition, triangleVertexIndex);

            setMeshInternal(vertexPosition, triangleVertexIndex);
        }

    public:
        // =========================
        // Rule of Five
        // =========================
        Superellipsoid() = default;

        Superellipsoid(const double rx,
                            const double ry,
                            const double rz,
                            const double ee,
                            const double en,
                            const int subdivisionLevel = 4)
            : rx_(rx),
            ry_(ry),
            rz_(rz),
            ee_(ee),
            en_(en),
            subdivisionLevel_(clampSubdivisionLevel(subdivisionLevel))
        {
            rebuildMesh();
        }

        ~Superellipsoid() override = default;
        Superellipsoid(const Superellipsoid&) = default;
        Superellipsoid(Superellipsoid&&) noexcept = default;
        Superellipsoid& operator=(const Superellipsoid&) = default;
        Superellipsoid& operator=(Superellipsoid&&) noexcept = default;

        // =========================
        // Host operations
        // =========================
        void setParams(const double rx,
                    const double ry,
                    const double rz,
                    const double ee,
                    const double en,
                    const int subdivisionLevel = 4)
        {
            rx_ = rx;
            ry_ = ry;
            rz_ = rz;
            ee_ = ee;
            en_ = en;
            subdivisionLevel_ = clampSubdivisionLevel(subdivisionLevel);
            rebuildMesh();
        }

        void setSubdivisionLevel(const int subdivisionLevel)
        {
            subdivisionLevel_ = clampSubdivisionLevel(subdivisionLevel);
            rebuildMesh();
        }

        // =========================
        // Virtual interfaces
        // =========================
        bool isValid() const override
        {
            return parametersAreValid() && TriangleMeshParticle::isValid();
        }

        void evaluateImplicitFunctionValueAndGradient(double& implicitFunctionValue,
                                                    double3& implicitFunctionGradient,
                                                    const double3& point) const override
        {
            if (!parametersAreValid())
            {
                implicitFunctionValue = 1.0;
                implicitFunctionGradient = make_double3(0.0, 0.0, 0.0);
                return;
            }

            const double inverseRadiusX = 1.0 / rx_;
            const double inverseRadiusY = 1.0 / ry_;
            const double inverseRadiusZ = 1.0 / rz_;

            const double normalizedAbsoluteX = std::fabs(point.x * inverseRadiusX);
            const double normalizedAbsoluteY = std::fabs(point.y * inverseRadiusY);
            const double normalizedAbsoluteZ = std::fabs(point.z * inverseRadiusZ);

            const double exponentXY = 2.0 / ee_;
            const double exponentZ = 2.0 / en_;
            const double outerExponent = ee_ / en_;

            const double xTerm = Utility::safeAbsPow(normalizedAbsoluteX, exponentXY);
            const double yTerm = Utility::safeAbsPow(normalizedAbsoluteY, exponentXY);
            const double xyTermSum = xTerm + yTerm;

            const double xyBlockValue = Utility::safeAbsPow(xyTermSum, outerExponent);
            const double zBlockValue = Utility::safeAbsPow(normalizedAbsoluteZ, exponentZ);

            implicitFunctionValue = xyBlockValue + zBlockValue - 1.0;

            const double numericalEpsilon = 1e-30;

            const double safeXYTermSum = std::max(xyTermSum, numericalEpsilon);
            const double safeNormalizedAbsoluteX = std::max(normalizedAbsoluteX, numericalEpsilon);
            const double safeNormalizedAbsoluteY = std::max(normalizedAbsoluteY, numericalEpsilon);
            const double safeNormalizedAbsoluteZ = std::max(normalizedAbsoluteZ, numericalEpsilon);

            const double xyTermSumToOuterExponentMinusOne =
                Utility::safeAbsPow(safeXYTermSum, outerExponent - 1.0);

            const double derivativeOfXYTermSumWithRespectToX =
                exponentXY *
                Utility::safeAbsPow(safeNormalizedAbsoluteX, exponentXY - 1.0) *
                Utility::signNoZero(point.x) *
                inverseRadiusX;

            const double derivativeOfXYTermSumWithRespectToY =
                exponentXY *
                Utility::safeAbsPow(safeNormalizedAbsoluteY, exponentXY - 1.0) *
                Utility::signNoZero(point.y) *
                inverseRadiusY;

            const double derivativeOfXYBlockWithRespectToX =
                outerExponent *
                xyTermSumToOuterExponentMinusOne *
                derivativeOfXYTermSumWithRespectToX;

            const double derivativeOfXYBlockWithRespectToY =
                outerExponent *
                xyTermSumToOuterExponentMinusOne *
                derivativeOfXYTermSumWithRespectToY;

            const double derivativeOfZBlockWithRespectToZ =
                exponentZ *
                Utility::safeAbsPow(safeNormalizedAbsoluteZ, exponentZ - 1.0) *
                Utility::signNoZero(point.z) *
                inverseRadiusZ;

            implicitFunctionGradient = make_double3(derivativeOfXYBlockWithRespectToX,
                                                    derivativeOfXYBlockWithRespectToY,
                                                    derivativeOfZBlockWithRespectToZ);

            Utility::applyLevelSetSign(implicitFunctionValue,
            implicitFunctionGradient,
            levelSetPositiveInside_);
        }

        double3 boundingBoxMin() const override
        {
            return make_double3(-rx_, -ry_, -rz_);
        }

        double3 boundingBoxMax() const override
        {
            return make_double3(rx_, ry_, rz_);
        }

        // =========================
        // Getters
        // =========================
        double rx() const { return rx_; }
        double ry() const { return ry_; }
        double rz() const { return rz_; }
        double ee() const { return ee_; }
        double en() const { return en_; }
    };

    class Sphere : public TriangleMeshParticle
    {
    protected:
        // =========================
        // Fields
        // =========================
        double radius_ {0.0};

        size_t targetSurfacePointCount_ {0};

    protected:
        // =========================
        // Helpers
        // =========================
        static inline uint64_t makeEdgeKey(const int vertexIndex0,
                                        const int vertexIndex1)
        {
            const uint32_t minimumIndex =
                static_cast<uint32_t>(std::min(vertexIndex0, vertexIndex1));
            const uint32_t maximumIndex =
                static_cast<uint32_t>(std::max(vertexIndex0, vertexIndex1));

            return (static_cast<uint64_t>(minimumIndex) << 32) |
                static_cast<uint64_t>(maximumIndex);
        }

        static inline double3 projectPointToSphere(const double3& point,
                                                const double radius)
        {
            const double pointLength = length(point);

            if (pointLength < 1e-30)
            {
                return make_double3(0.0, 0.0, 0.0);
            }

            return (radius / pointLength) * point;
        }

        static inline int estimateSubdivisionLevelAtLeast(const size_t targetSurfacePointCount)
        {
            if (targetSurfacePointCount <= 12)
            {
                return 0;
            }

            for (int subdivisionLevel = 0; subdivisionLevel <= 12; ++subdivisionLevel)
            {
                const size_t vertexCount =
                    static_cast<size_t>(10.0 * std::pow(4.0, subdivisionLevel) + 2.0);

                if (vertexCount >= targetSurfacePointCount)
                {
                    return subdivisionLevel;
                }
            }

            return 12;
        }

        static inline int getOrCreateMidpointVertex(const int vertexIndex0,
                                                    const int vertexIndex1,
                                                    const double radius,
                                                    std::vector<double3>& vertexPosition,
                                                    std::unordered_map<uint64_t, int>& midpointCache)
        {
            const uint64_t edgeKey = makeEdgeKey(vertexIndex0, vertexIndex1);

            const auto iterator = midpointCache.find(edgeKey);
            if (iterator != midpointCache.end())
            {
                return iterator->second;
            }

            const double3 midpoint =
                0.5 * (vertexPosition[vertexIndex0] + vertexPosition[vertexIndex1]);

            const double3 projectedMidpoint = projectPointToSphere(midpoint, radius);

            const int newVertexIndex = static_cast<int>(vertexPosition.size());
            vertexPosition.push_back(projectedMidpoint);
            midpointCache[edgeKey] = newVertexIndex;

            return newVertexIndex;
        }

        static void createIcosahedron(const double radius,
                                    std::vector<double3>& vertexPosition,
                                    std::vector<int3>& triangleVertexIndex)
        {
            const double goldenRatio = 0.5 * (1.0 + std::sqrt(5.0));

            vertexPosition =
            {
                make_double3(-1.0,  goldenRatio,  0.0),
                make_double3( 1.0,  goldenRatio,  0.0),
                make_double3(-1.0, -goldenRatio,  0.0),
                make_double3( 1.0, -goldenRatio,  0.0),

                make_double3( 0.0, -1.0,  goldenRatio),
                make_double3( 0.0,  1.0,  goldenRatio),
                make_double3( 0.0, -1.0, -goldenRatio),
                make_double3( 0.0,  1.0, -goldenRatio),

                make_double3( goldenRatio,  0.0, -1.0),
                make_double3( goldenRatio,  0.0,  1.0),
                make_double3(-goldenRatio,  0.0, -1.0),
                make_double3(-goldenRatio,  0.0,  1.0)
            };

            for (size_t vertexIndex = 0; vertexIndex < vertexPosition.size(); ++vertexIndex)
            {
                vertexPosition[vertexIndex] = projectPointToSphere(vertexPosition[vertexIndex], radius);
            }

            triangleVertexIndex =
            {
                make_int3(0, 11, 5),   make_int3(0, 5, 1),    make_int3(0, 1, 7),    make_int3(0, 7, 10),   make_int3(0, 10, 11),
                make_int3(1, 5, 9),    make_int3(5, 11, 4),   make_int3(11, 10, 2),  make_int3(10, 7, 6),   make_int3(7, 1, 8),
                make_int3(3, 9, 4),    make_int3(3, 4, 2),    make_int3(3, 2, 6),    make_int3(3, 6, 8),    make_int3(3, 8, 9),
                make_int3(4, 9, 5),    make_int3(2, 4, 11),   make_int3(6, 2, 10),   make_int3(8, 6, 7),    make_int3(9, 8, 1)
            };
        }

        static void subdivideMesh(const double radius,
                                std::vector<double3>& vertexPosition,
                                std::vector<int3>& triangleVertexIndex)
        {
            std::unordered_map<uint64_t, int> midpointCache;
            midpointCache.reserve(triangleVertexIndex.size() * 3);

            std::vector<int3> newTriangleVertexIndex;
            newTriangleVertexIndex.reserve(triangleVertexIndex.size() * 4);

            for (size_t triangleIndex = 0; triangleIndex < triangleVertexIndex.size(); ++triangleIndex)
            {
                const int3 triangle = triangleVertexIndex[triangleIndex];

                const int vertexIndex0 = triangle.x;
                const int vertexIndex1 = triangle.y;
                const int vertexIndex2 = triangle.z;

                const int midpointIndex01 =
                    getOrCreateMidpointVertex(vertexIndex0,
                                            vertexIndex1,
                                            radius,
                                            vertexPosition,
                                            midpointCache);

                const int midpointIndex12 =
                    getOrCreateMidpointVertex(vertexIndex1,
                                            vertexIndex2,
                                            radius,
                                            vertexPosition,
                                            midpointCache);

                const int midpointIndex20 =
                    getOrCreateMidpointVertex(vertexIndex2,
                                            vertexIndex0,
                                            radius,
                                            vertexPosition,
                                            midpointCache);

                newTriangleVertexIndex.push_back(make_int3(vertexIndex0, midpointIndex01, midpointIndex20));
                newTriangleVertexIndex.push_back(make_int3(vertexIndex1, midpointIndex12, midpointIndex01));
                newTriangleVertexIndex.push_back(make_int3(vertexIndex2, midpointIndex20, midpointIndex12));
                newTriangleVertexIndex.push_back(make_int3(midpointIndex01, midpointIndex12, midpointIndex20));
            }

            triangleVertexIndex = std::move(newTriangleVertexIndex);
        }

        static void generateSphereMesh(const double radius,
                                    const size_t targetSurfacePointCount,
                                    std::vector<double3>& vertexPosition,
                                    std::vector<int3>& triangleVertexIndex)
        {
            if (radius <= 0.0)
            {
                throw std::invalid_argument("Sphere::generateSphereMesh: radius must be > 0.");
            }

            if (targetSurfacePointCount < 4)
            {
                throw std::invalid_argument("Sphere::generateSphereMesh: targetSurfacePointCount must be >= 4.");
            }

            int subdivisionLevel = estimateSubdivisionLevelAtLeast(targetSurfacePointCount);

            createIcosahedron(radius, vertexPosition, triangleVertexIndex);

            for (int level = 0; level < subdivisionLevel; ++level)
            {
                subdivideMesh(radius, vertexPosition, triangleVertexIndex);
            }
        }

        void rebuildMesh()
        {
            if (radius_ <= 0.0 || targetSurfacePointCount_ < 4)
            {
                clearMesh();
                return;
            }

            std::vector<double3> vertexPosition;
            std::vector<int3> triangleVertexIndex;

            generateSphereMesh(radius_,
                            targetSurfacePointCount_,
                            vertexPosition,
                            triangleVertexIndex);

            setMeshInternal(vertexPosition, triangleVertexIndex);
        }

    public:
        // =========================
        // Rule of Five
        // =========================
        Sphere() = default;

        Sphere(const double radius,
            const size_t targetSurfacePointCount)
            :radius_(radius),
            targetSurfacePointCount_(targetSurfacePointCount)
        {
            rebuildMesh();
        }

        ~Sphere() override = default;
        Sphere(const Sphere&) = default;
        Sphere(Sphere&&) noexcept = default;
        Sphere& operator=(const Sphere&) = default;
        Sphere& operator=(Sphere&&) noexcept = default;

        // =========================
        // Host operations
        // =========================
        void setParams(const double radius,
                    const size_t targetSurfacePointCount)
        {
            radius_ = radius;
            targetSurfacePointCount_ = targetSurfacePointCount;
            rebuildMesh();
        }

        void clearSphere()
        {
            radius_ = 0.0;
            targetSurfacePointCount_ = 0;
            levelSetPositiveInside_ = false;
            clearMesh();
        }

        // =========================
        // Virtual interfaces
        // =========================
        bool isValid() const override
        {
            if (radius_ <= 0.0) return false;
            if (targetSurfacePointCount_ < 4) return false;
            return TriangleMeshParticle::isValid();
        }

        void evaluateImplicitFunctionValueAndGradient(double& implicitFunctionValue,
                                                    double3& implicitFunctionGradient,
                                                    const double3& point) const override
        {
            if (radius_ <= 0.0)
            {
                implicitFunctionValue = 1.0;
                implicitFunctionGradient = make_double3(0.0, 0.0, 0.0);
                return;
            }

            const double distanceToCenter = length(point);

            implicitFunctionValue = distanceToCenter - radius_;

            if (distanceToCenter > 1e-30) implicitFunctionGradient = point / distanceToCenter;

            Utility::applyLevelSetSign(implicitFunctionValue,
                                       implicitFunctionGradient,
                                       levelSetPositiveInside_);
        }

        double3 boundingBoxMin() const override
        {
           return -make_double3(radius_, radius_, radius_);
        }

        double3 boundingBoxMax() const override
        {
           return make_double3(radius_, radius_, radius_);
        }

        // =========================
        // Getters
        // =========================
        double radius() const
        {
            return radius_;
        }
    };

    //=========================================================================
    // Box wall
    //=========================================================================
    class BoxWall : public TriangleMeshParticle
    {
    protected:
        // =========================
        // Fields
        // =========================
        double lx_ {1.0};
        double ly_ {1.0};
        double lz_ {1.0};

    protected:
        // =========================
        // Helpers
        // =========================
        void rebuildMesh()
        {
            std::vector<double3> vertices;
            std::vector<int3> faces;

            if (lx_ <= 0.0 || ly_ <= 0.0 || lz_ <= 0.0)
            {
                setMeshInternal(vertices, faces);
                return;
            }

            const double hx = 0.5 * lx_;
            const double hy = 0.5 * ly_;
            const double hz = 0.5 * lz_;

            vertices.reserve(8);
            faces.reserve(12);

            vertices.push_back(make_double3(-hx, -hy, -hz)); // 0
            vertices.push_back(make_double3( hx, -hy, -hz)); // 1
            vertices.push_back(make_double3( hx,  hy, -hz)); // 2
            vertices.push_back(make_double3(-hx,  hy, -hz)); // 3
            vertices.push_back(make_double3(-hx, -hy,  hz)); // 4
            vertices.push_back(make_double3( hx, -hy,  hz)); // 5
            vertices.push_back(make_double3( hx,  hy,  hz)); // 6
            vertices.push_back(make_double3(-hx,  hy,  hz)); // 7

            faces.push_back(make_int3(0, 2, 1));
            faces.push_back(make_int3(0, 3, 2));

            faces.push_back(make_int3(4, 5, 6));
            faces.push_back(make_int3(4, 6, 7));

            faces.push_back(make_int3(0, 7, 3));
            faces.push_back(make_int3(0, 4, 7));

            faces.push_back(make_int3(1, 2, 6));
            faces.push_back(make_int3(1, 6, 5));

            faces.push_back(make_int3(0, 1, 5));
            faces.push_back(make_int3(0, 5, 4));

            faces.push_back(make_int3(3, 7, 6));
            faces.push_back(make_int3(3, 6, 2));

            setMeshInternal(vertices, faces);
        }

    public:
        // =========================
        // Rule of Five
        // =========================
        BoxWall() { levelSetPositiveInside_ = true; }

        BoxWall(const double lx,
                const double ly,
                const double lz)
            : lx_(lx),
              ly_(ly),
              lz_(lz)
        {
            levelSetPositiveInside_ = true;
            rebuildMesh();
        }

        ~BoxWall() override = default;
        BoxWall(const BoxWall&) = default;
        BoxWall(BoxWall&&) noexcept = default;
        BoxWall& operator=(const BoxWall&) = default;
        BoxWall& operator=(BoxWall&&) noexcept = default;

        // =========================
        // Host operations
        // =========================
        void setParams(const double lx,
                       const double ly,
                       const double lz)
        {
            lx_ = lx;
            ly_ = ly;
            lz_ = lz;
            rebuildMesh();
            clearGrid();
        }

        // =========================
        // Virtual interfaces
        // =========================
        bool isValid() const override
        {
            return (lx_ > 0.0 &&
                    ly_ > 0.0 &&
                    lz_ > 0.0);
        }

        void evaluateImplicitFunctionValueAndGradient(double& implicitFunctionValue,
                                                      double3& implicitFunctionGradient,
                                                      const double3& point) const override
        {
            if (!isValid())
            {
                implicitFunctionValue = 1.0;
                implicitFunctionGradient = make_double3(0.0, 0.0, 0.0);
                return;
            }

            const double3 halfBoxSize = make_double3(0.5 * lx_,
                                                     0.5 * ly_,
                                                     0.5 * lz_);

            const double3 q = make_double3(std::fabs(point.x) - halfBoxSize.x,
                                           std::fabs(point.y) - halfBoxSize.y,
                                           std::fabs(point.z) - halfBoxSize.z);

            const double3 qOutside = make_double3(std::max(q.x, 0.0),
                                                  std::max(q.y, 0.0),
                                                  std::max(q.z, 0.0));

            const double outsideDistance = length(qOutside);
            const double insideDistance = std::min(std::max(q.x, std::max(q.y, q.z)), 0.0);

            double signedDistanceValue = outsideDistance + insideDistance;
            double3 signedDistanceGradient = make_double3(0.0, 0.0, 0.0);

            if (outsideDistance > 1e-30)
            {
                signedDistanceGradient = qOutside / outsideDistance;

                if (point.x < 0.0 && q.x > 0.0) signedDistanceGradient.x *= -1.0;
                if (point.y < 0.0 && q.y > 0.0) signedDistanceGradient.y *= -1.0;
                if (point.z < 0.0 && q.z > 0.0) signedDistanceGradient.z *= -1.0;
            }
            else
            {
                const double distanceToFaceX = halfBoxSize.x - std::fabs(point.x);
                const double distanceToFaceY = halfBoxSize.y - std::fabs(point.y);
                const double distanceToFaceZ = halfBoxSize.z - std::fabs(point.z);

                if (distanceToFaceX <= distanceToFaceY && distanceToFaceX <= distanceToFaceZ)
                {
                    signedDistanceGradient = make_double3(point.x >= 0.0 ? 1.0 : -1.0,
                                                          0.0,
                                                          0.0);
                }
                else if (distanceToFaceY <= distanceToFaceX && distanceToFaceY <= distanceToFaceZ)
                {
                    signedDistanceGradient = make_double3(0.0,
                                                          point.y >= 0.0 ? 1.0 : -1.0,
                                                          0.0);
                }
                else
                {
                    signedDistanceGradient = make_double3(0.0,
                                                          0.0,
                                                          point.z >= 0.0 ? 1.0 : -1.0);
                }
            }

            implicitFunctionValue = signedDistanceValue;
            implicitFunctionGradient = signedDistanceGradient;

            Utility::applyLevelSetSign(implicitFunctionValue,
                                       implicitFunctionGradient,
                                       levelSetPositiveInside_);
        }

        double3 boundingBoxMin() const override
        {
            return make_double3(-0.5 * lx_,
                                -0.5 * ly_,
                                -0.5 * lz_);
        }

        double3 boundingBoxMax() const override
        {
            return make_double3(0.5 * lx_,
                                0.5 * ly_,
                                0.5 * lz_);
        }

        // =========================
        // Getters
        // =========================
        double lx() const { return lx_; }
        double ly() const { return ly_; }
        double lz() const { return lz_; }
    };

    //=========================================================================
    // Cylinder wall
    //=========================================================================
    class CylinderWall : public TriangleMeshParticle
    {
    protected:
        // =========================
        // Fields
        // =========================
        double3 pointA_ {0.0, 0.0, -0.5};
        double3 pointB_ {0.0, 0.0,  0.5};
        double radius_ {1.0};

        int numSegmentsCircumference_ {256};
        int numSegmentsAxial_ {1};

    protected:
        // =========================
        // Helpers
        // =========================
        void rebuildMesh()
        {
            std::vector<double3> vertices;
            std::vector<int3> faces;

            if (!isValid())
            {
                setMeshInternal(vertices, faces);
                return;
            }

            const double3 axisVector = pointB_ - pointA_;
            const double axisLength = length(axisVector);
            const double3 axisDirection = axisVector / axisLength;

            double3 referenceDirection =
                (std::fabs(axisDirection.x) < 0.9)
                ? make_double3(1.0, 0.0, 0.0)
                : make_double3(0.0, 1.0, 0.0);

            const double3 radialDirectionU = normalize(cross(axisDirection, referenceDirection));
            const double3 radialDirectionV = normalize(cross(axisDirection, radialDirectionU));

            const int numRingVertices = numSegmentsCircumference_;
            const int numAxialLayers = numSegmentsAxial_ + 1;

            vertices.reserve(size_t(numAxialLayers) * size_t(numRingVertices) + 2);

            for (int axialIndex = 0; axialIndex < numAxialLayers; ++axialIndex)
            {
                const double t = double(axialIndex) / double(numSegmentsAxial_);
                const double3 center = pointA_ + t * axisVector;

                for (int segmentIndex = 0; segmentIndex < numSegmentsCircumference_; ++segmentIndex)
                {
                    const double theta =
                        2.0 * M_PI * double(segmentIndex) / double(numSegmentsCircumference_);

                    const double3 radialDirection =
                        std::cos(theta) * radialDirectionU +
                        std::sin(theta) * radialDirectionV;

                    vertices.push_back(center + radius_ * radialDirection);
                }
            }

            auto ringVertexID = [&](const int axialIndex, const int segmentIndex) -> int
            {
                const int wrappedSegmentIndex =
                    (segmentIndex % numSegmentsCircumference_ + numSegmentsCircumference_) %
                    numSegmentsCircumference_;

                return axialIndex * numRingVertices + wrappedSegmentIndex;
            };

            for (int axialIndex = 0; axialIndex < numSegmentsAxial_; ++axialIndex)
            {
                for (int segmentIndex = 0; segmentIndex < numSegmentsCircumference_; ++segmentIndex)
                {
                    const int v00 = ringVertexID(axialIndex,     segmentIndex);
                    const int v01 = ringVertexID(axialIndex,     segmentIndex + 1);
                    const int v10 = ringVertexID(axialIndex + 1, segmentIndex);
                    const int v11 = ringVertexID(axialIndex + 1, segmentIndex + 1);

                    faces.push_back(make_int3(v00, v10, v11));
                    faces.push_back(make_int3(v00, v11, v01));
                }
            }

            const int capCenterAID = static_cast<int>(vertices.size());
            vertices.push_back(pointA_);

            const int capCenterBID = static_cast<int>(vertices.size());
            vertices.push_back(pointB_);

            for (int segmentIndex = 0; segmentIndex < numSegmentsCircumference_; ++segmentIndex)
            {
                const int v0 = ringVertexID(0, segmentIndex);
                const int v1 = ringVertexID(0, segmentIndex + 1);

                faces.push_back(make_int3(capCenterAID, v1, v0));
            }

            for (int segmentIndex = 0; segmentIndex < numSegmentsCircumference_; ++segmentIndex)
            {
                const int v0 = ringVertexID(numAxialLayers - 1, segmentIndex);
                const int v1 = ringVertexID(numAxialLayers - 1, segmentIndex + 1);

                faces.push_back(make_int3(capCenterBID, v0, v1));
            }

            setMeshInternal(vertices, faces);
        }

    public:
        // =========================
        // Rule of Five
        // =========================
        CylinderWall() { levelSetPositiveInside_ = true; }

        CylinderWall(const double3& pointA,
                     const double3& pointB,
                     const double radius,
                     const int numSegmentsCircumference = 256,
                     const int numSegmentsAxial = 1)
            : pointA_(pointA),
              pointB_(pointB),
              radius_(radius),
              numSegmentsCircumference_(std::max(3, numSegmentsCircumference)),
              numSegmentsAxial_(std::max(1, numSegmentsAxial))
        {
            levelSetPositiveInside_ = true;
            rebuildMesh();
        }

        ~CylinderWall() override = default;
        CylinderWall(const CylinderWall&) = default;
        CylinderWall(CylinderWall&&) noexcept = default;
        CylinderWall& operator=(const CylinderWall&) = default;
        CylinderWall& operator=(CylinderWall&&) noexcept = default;

        // =========================
        // Host operations
        // =========================
        void setParams(const double3& pointA,
                         const double3& pointB,
                         const double radius,
                         const int numSegmentsCircumference = 256,
                         const int numSegmentsAxial = 1)
        {
            pointA_ = pointA;
            pointB_ = pointB;
            radius_ = radius;
            numSegmentsCircumference_ = std::max(3, numSegmentsCircumference);
            numSegmentsAxial_ = std::max(1, numSegmentsAxial);
            rebuildMesh();
            clearGrid();
        }

        void setMeshResolution(const int numSegmentsCircumference,
                               const int numSegmentsAxial)
        {
            numSegmentsCircumference_ = std::max(3, numSegmentsCircumference);
            numSegmentsAxial_ = std::max(1, numSegmentsAxial);
            rebuildMesh();
            clearGrid();
        }

        // =========================
        // Virtual interfaces
        // =========================
        bool isValid() const override
        {
            return (radius_ > 0.0 && length(pointB_ - pointA_) > 1e-30);
        }

        void evaluateImplicitFunctionValueAndGradient(double& implicitFunctionValue,
                                                    double3& implicitFunctionGradient,
                                                    const double3& point) const override
        {
            if (!isValid())
            {
                implicitFunctionValue = 1.0;
                implicitFunctionGradient = make_double3(0.0, 0.0, 0.0);
                return;
            }

            const double3 axisVector = pointB_ - pointA_;
            const double axisLength = length(axisVector);
            const double3 axisDirection = axisVector / axisLength;
            const double3 pointRelativeToA = point - pointA_;
            const double axialCoordinate = dot(pointRelativeToA, axisDirection);
            const double clampedAxialCoordinate = Utility::clampValue(axialCoordinate, 0.0, axisLength);
            const double3 closestPointOnAxisSegment = pointA_ + clampedAxialCoordinate * axisDirection;

            bool inRadialRange = (length(cross(pointRelativeToA, axisDirection)) <= radius_);
            bool inAxialRange = (axialCoordinate >= 0. && axialCoordinate <= axisLength);

            if (inRadialRange && inAxialRange)
            {
                const double3 radialVector = point - closestPointOnAxisSegment;
                const double radialDistance = length(radialVector);
                const double radialSignedDistance = radialDistance - radius_;
                const double axialSignedDistance = axialCoordinate < 0.5 * axisLength ? -axialCoordinate : axialCoordinate - axisLength;
                const double radialAbsoluteDistance = std::abs(radialSignedDistance);
                const double axialAbsoluteDistance = std::abs(axialSignedDistance);
                if (radialAbsoluteDistance <= axialAbsoluteDistance) 
                {
                    implicitFunctionValue = radialSignedDistance;
                    implicitFunctionGradient = normalize(radialVector);
                }
                else 
                {
                    implicitFunctionValue = axialSignedDistance;
                    implicitFunctionGradient = axialCoordinate < 0.5 * axisLength ? -axisVector : axisVector;
                }
            }
            else if (!inRadialRange && inAxialRange)
            {
                const double3 radialVector = point - closestPointOnAxisSegment;
                const double radialDistance = length(radialVector);
                implicitFunctionValue = radialDistance - radius_;
                implicitFunctionGradient = normalize(radialVector);
            }
            else if (!inRadialRange && !inAxialRange)
            {
                const double3 closestPointOnSurface = closestPointOnAxisSegment + 
                normalize(pointRelativeToA - axialCoordinate * axisDirection) * radius_;
                implicitFunctionValue = length(point - closestPointOnSurface);
                implicitFunctionGradient = normalize(point - closestPointOnSurface);
            }
            else
            {
                implicitFunctionValue = axialCoordinate < 0.0 ? -axialCoordinate : axialCoordinate - axisLength;
                implicitFunctionGradient = axialCoordinate < 0.0 ? -axisVector : axisVector;
            }
            
            Utility::applyLevelSetSign(implicitFunctionValue,
                                       implicitFunctionGradient,
                                       levelSetPositiveInside_);
        }

        double3 boundingBoxMin() const override
        {
            return make_double3(std::min(pointA_.x, pointB_.x) - radius_,
                                std::min(pointA_.y, pointB_.y) - radius_,
                                std::min(pointA_.z, pointB_.z) - radius_);
        }

        double3 boundingBoxMax() const override
        {
            return make_double3(std::max(pointA_.x, pointB_.x) + radius_,
                                std::max(pointA_.y, pointB_.y) + radius_,
                                std::max(pointA_.z, pointB_.z) + radius_);
        }

        // =========================
        // Getters
        // =========================
        const double3& pointA() const { return pointA_; }
        const double3& pointB() const { return pointB_; }
        double radius() const { return radius_; }
    };

} // namespace LevelSetObject