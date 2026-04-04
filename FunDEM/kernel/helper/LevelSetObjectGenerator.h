#pragma once
#include "CUDAKernelFunction/myUtility/myVec.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
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
        double3 gridOrigin {0.0, 0.0, 0.0};
        int3 gridNodeSize {0, 0, 0};
        double gridNodeSpacing {0.0};
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
            gridInfo_ = GridInfo {};
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
        // Query functions
        // =========================
        double evaluateImplicitFunctionValue(const double3& point) const
        {
            double implicitFunctionValue = 0.0;
            double3 implicitFunctionGradient = make_double3(0.0, 0.0, 0.0);

            evaluateImplicitFunctionValueAndGradient(implicitFunctionValue,
                                                     implicitFunctionGradient,
                                                     point);

            return implicitFunctionValue;
        }

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

            auto snapDownToGrid = [&](const double coordinate)
            {
                return gridSpacing * std::floor(coordinate / gridSpacing);
            };

            auto snapUpToGrid = [&](const double coordinate)
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

            const double minGradientNormForSignedDistanceApproximation = 1e-14;

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
                                          gridInfo_.gridNodeSize)] =
                            signedDistanceLikeLevelSetValue;
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


            const std::string fileName = fileNamePrefix + ".vtu";

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

            // ---------------------------------------------------------------------
            // Points
            // ---------------------------------------------------------------------
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

            // ---------------------------------------------------------------------
            // Cells: one VTK_VERTEX per point
            // ---------------------------------------------------------------------
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
                out << "          1\n"; // VTK_VERTEX
            }
            out << "        </DataArray>\n";

            out << "      </Cells>\n";

            // ---------------------------------------------------------------------
            // Point data
            // ---------------------------------------------------------------------
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
    // Generic surface point generation for star-shaped geometry
    //=========================================================================
    inline std::vector<double3> generateSurfacePointsUniform_StarShaped(const Base& object,
                                                                        const int numSurfacePoints = 10000)
    {
        std::vector<double3> surfacePoints;
        if (numSurfacePoints <= 0 || !object.isValid()) return surfacePoints;

        const double3 particleBoundingBoxMin = object.boundingBoxMin();
        const double3 particleBoundingBoxMax = object.boundingBoxMax();

        const double particleBoundingBoxSizeX = particleBoundingBoxMax.x - particleBoundingBoxMin.x;
        const double particleBoundingBoxSizeY = particleBoundingBoxMax.y - particleBoundingBoxMin.y;
        const double particleBoundingBoxSizeZ = particleBoundingBoxMax.z - particleBoundingBoxMin.z;

        const double maximumSearchRadius =
            0.5 * std::max(particleBoundingBoxSizeX,
                           std::max(particleBoundingBoxSizeY, particleBoundingBoxSizeZ));

        if (maximumSearchRadius <= 0.0) return surfacePoints;

        auto solveSurfaceRadiusAlongUnitDirection =
            [&](const double3& unitDirection) -> double
        {
            double lowerBoundRadius = 0.0;
            double upperBoundRadius = 2.0 * maximumSearchRadius;

            double upperBoundImplicitFunctionValue = 0.0;

            {
                double implicitFunctionValue = 0.0;
                double3 implicitFunctionGradient = make_double3(0.0, 0.0, 0.0);

                object.evaluateImplicitFunctionValueAndGradient(
                    implicitFunctionValue,
                    implicitFunctionGradient,
                    make_double3(upperBoundRadius * unitDirection.x,
                                 upperBoundRadius * unitDirection.y,
                                 upperBoundRadius * unitDirection.z));

                upperBoundImplicitFunctionValue = implicitFunctionValue;

                int numBracketExpansion = 0;
                while (upperBoundImplicitFunctionValue < 0.0 && numBracketExpansion < 30)
                {
                    upperBoundRadius *= 2.0;

                    object.evaluateImplicitFunctionValueAndGradient(
                        implicitFunctionValue,
                        implicitFunctionGradient,
                        make_double3(upperBoundRadius * unitDirection.x,
                                     upperBoundRadius * unitDirection.y,
                                     upperBoundRadius * unitDirection.z));

                    upperBoundImplicitFunctionValue = implicitFunctionValue;
                    numBracketExpansion++;
                }

                if (upperBoundImplicitFunctionValue < 0.0) return upperBoundRadius;
            }

            double currentRadius = 0.5 * (lowerBoundRadius + upperBoundRadius);

            for (int iteration = 0; iteration < 25; ++iteration)
            {
                const double3 point = make_double3(currentRadius * unitDirection.x,
                                                   currentRadius * unitDirection.y,
                                                   currentRadius * unitDirection.z);

                double implicitFunctionValue = 0.0;
                double3 implicitFunctionGradient = make_double3(0.0, 0.0, 0.0);

                object.evaluateImplicitFunctionValueAndGradient(
                    implicitFunctionValue,
                    implicitFunctionGradient,
                    point);

                if (implicitFunctionValue > 0.0) upperBoundRadius = currentRadius;
                else lowerBoundRadius = currentRadius;

                const double derivativeOfImplicitFunctionAlongRadius =
                    implicitFunctionGradient.x * unitDirection.x +
                    implicitFunctionGradient.y * unitDirection.y +
                    implicitFunctionGradient.z * unitDirection.z;

                double updatedRadius = currentRadius;

                if (std::fabs(derivativeOfImplicitFunctionAlongRadius) > 1e-14)
                {
                    updatedRadius =
                        currentRadius -
                        implicitFunctionValue / derivativeOfImplicitFunctionAlongRadius;
                }

                if (updatedRadius <= lowerBoundRadius || updatedRadius >= upperBoundRadius)
                {
                    updatedRadius = 0.5 * (lowerBoundRadius + upperBoundRadius);
                }

                if (std::fabs(upperBoundRadius - lowerBoundRadius) <
                    1e-12 * std::max(1.0, upperBoundRadius))
                {
                    currentRadius = updatedRadius;
                    break;
                }

                currentRadius = updatedRadius;
            }

            return currentRadius;
        };

        surfacePoints.reserve(size_t(numSurfacePoints));

        const double goldenRatio = 0.5 * (1.0 + std::sqrt(5.0));
        const double goldenAngle = 2.0 * M_PI * (1.0 - 1.0 / goldenRatio);

        for (int surfacePointIndex = 0; surfacePointIndex < numSurfacePoints; ++surfacePointIndex)
        {
            const double normalizedIndex =
                (surfacePointIndex + 0.5) / double(numSurfacePoints);

            const double unitDirectionY = 1.0 - 2.0 * normalizedIndex;
            const double unitDirectionRadiusOnXZPlane =
                std::sqrt(std::max(0.0, 1.0 - unitDirectionY * unitDirectionY));

            const double azimuthAngle = goldenAngle * double(surfacePointIndex);

            const double unitDirectionX = unitDirectionRadiusOnXZPlane * std::cos(azimuthAngle);
            const double unitDirectionZ = unitDirectionRadiusOnXZPlane * std::sin(azimuthAngle);

            const double3 unitDirection = make_double3(unitDirectionX,
                                                       unitDirectionY,
                                                       unitDirectionZ);

            const double surfaceRadius =
                solveSurfaceRadiusAlongUnitDirection(unitDirection);

            surfacePoints.push_back(make_double3(surfaceRadius * unitDirection.x,
                                                 surfaceRadius * unitDirection.y,
                                                 surfaceRadius * unitDirection.z));
        }

        return surfacePoints;
    }

    //=========================================================================
    // Superellipsoid particle
    //=========================================================================
    class SuperellipsoidParticle : public Base
    {
    protected:
        // =========================
        // Fields
        // =========================
        double rx_ {1.0};
        double ry_ {1.0};
        double rz_ {1.0};
        double ee_ {1.0};
        double en_ {1.0};

    public:
        // =========================
        // Rule of Five
        // =========================
        SuperellipsoidParticle() = default;

        SuperellipsoidParticle(const double rx,
                               const double ry,
                               const double rz,
                               const double ee,
                               const double en)
            : rx_(rx),
              ry_(ry),
              rz_(rz),
              ee_(ee),
              en_(en)
        {
        }

        ~SuperellipsoidParticle() override = default;
        SuperellipsoidParticle(const SuperellipsoidParticle&) = default;
        SuperellipsoidParticle(SuperellipsoidParticle&&) noexcept = default;
        SuperellipsoidParticle& operator=(const SuperellipsoidParticle&) = default;
        SuperellipsoidParticle& operator=(SuperellipsoidParticle&&) noexcept = default;

        // =========================
        // Host operations
        // =========================
        void setParams(const double rx,
                       const double ry,
                       const double rz,
                       const double ee,
                       const double en)
        {
            rx_ = rx;
            ry_ = ry;
            rz_ = rz;
            ee_ = ee;
            en_ = en;
            clearGrid();
        }

        // =========================
        // Virtual interfaces
        // =========================
        bool isValid() const override
        {
            return (rx_ > 0.0 &&
                    ry_ > 0.0 &&
                    rz_ > 0.0 &&
                    ee_ > 0.0 &&
                    en_ > 0.0);
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
        // Surface point generation
        // =========================
        std::vector<double3> generateSurfacePointsUniform(const int numSurfacePoints = 10000) const
        {
            return generateSurfacePointsUniform_StarShaped(*this, numSurfacePoints);
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

    //=========================================================================
    // Triangle mesh particle
    //=========================================================================
    class TriangleMeshParticle : public Base
    {
    protected:
        // =========================
        // Fields
        // =========================
        std::vector<double3> vertexPosition_;
        std::vector<int3> triangleVertexIndex_;

        double3 boundingBoxMin_ {0.0, 0.0, 0.0};
        double3 boundingBoxMax_ {0.0, 0.0, 0.0};

    protected:
        // =========================
        // Helpers
        // =========================
        static inline double solidAngleFromPointToTriangle(const double3& queryPoint,
                                                           const double3& triangleVertex0,
                                                           const double3& triangleVertex1,
                                                           const double3& triangleVertex2)
        {
            const double3 r0 = triangleVertex0 - queryPoint;
            const double3 r1 = triangleVertex1 - queryPoint;
            const double3 r2 = triangleVertex2 - queryPoint;

            const double l0 = length(r0);
            const double l1 = length(r1);
            const double l2 = length(r2);

            if (l0 < 1e-30 || l1 < 1e-30 || l2 < 1e-30) return 0.0;

            const double numerator = dot(r0, cross(r1, r2));
            const double denominator =
                l0 * l1 * l2 +
                dot(r0, r1) * l2 +
                dot(r1, r2) * l0 +
                dot(r2, r0) * l1;

            return 2.0 * std::atan2(numerator, denominator);
        }

        static inline double3 closestPointOnTriangle(const double3& queryPoint,
                                                     const double3& triangleVertex0,
                                                     const double3& triangleVertex1,
                                                     const double3& triangleVertex2)
        {
            const double3 edge0 = triangleVertex1 - triangleVertex0;
            const double3 edge1 = triangleVertex2 - triangleVertex0;
            const double3 vertex0ToPoint = queryPoint - triangleVertex0;

            const double d1 = dot(edge0, vertex0ToPoint);
            const double d2 = dot(edge1, vertex0ToPoint);
            if (d1 <= 0.0 && d2 <= 0.0) return triangleVertex0;

            const double3 vertex1ToPoint = queryPoint - triangleVertex1;
            const double d3 = dot(edge0, vertex1ToPoint);
            const double d4 = dot(edge1, vertex1ToPoint);
            if (d3 >= 0.0 && d4 <= d3) return triangleVertex1;

            const double edgeRegion01Determinant = d1 * d4 - d3 * d2;
            if (edgeRegion01Determinant <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
            {
                const double barycentricCoordinate = d1 / (d1 - d3);
                return triangleVertex0 + barycentricCoordinate * edge0;
            }

            const double3 vertex2ToPoint = queryPoint - triangleVertex2;
            const double d5 = dot(edge0, vertex2ToPoint);
            const double d6 = dot(edge1, vertex2ToPoint);
            if (d6 >= 0.0 && d5 <= d6) return triangleVertex2;

            const double edgeRegion02Determinant = d5 * d2 - d1 * d6;
            if (edgeRegion02Determinant <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
            {
                const double barycentricCoordinate = d2 / (d2 - d6);
                return triangleVertex0 + barycentricCoordinate * edge1;
            }

            const double edgeRegion12Determinant = d3 * d6 - d5 * d4;
            if (edgeRegion12Determinant <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
            {
                const double3 edge12 = triangleVertex2 - triangleVertex1;
                const double barycentricCoordinate =
                    (d4 - d3) / ((d4 - d3) + (d5 - d6));
                return triangleVertex1 + barycentricCoordinate * edge12;
            }

            const double denominator =
                1.0 / (edgeRegion01Determinant + edgeRegion02Determinant + edgeRegion12Determinant);

            const double barycentricCoordinateV = edgeRegion02Determinant * denominator;
            const double barycentricCoordinateW = edgeRegion12Determinant * denominator;

            return triangleVertex0 + barycentricCoordinateV * edge0 + barycentricCoordinateW * edge1;
        }

        bool isPointInsideClosedTriangleMesh(const double3& point) const
        {
            double totalSolidAngle = 0.0;

            for (size_t triangleIndex = 0; triangleIndex < triangleVertexIndex_.size(); ++triangleIndex)
            {
                const int3 triangle = triangleVertexIndex_[triangleIndex];

                const double3& triangleVertex0 = vertexPosition_[triangle.x];
                const double3& triangleVertex1 = vertexPosition_[triangle.y];
                const double3& triangleVertex2 = vertexPosition_[triangle.z];

                totalSolidAngle += solidAngleFromPointToTriangle(point,
                                                                 triangleVertex0,
                                                                 triangleVertex1,
                                                                 triangleVertex2);
            }

            const double windingNumber = totalSolidAngle / (4.0 * M_PI);
            return std::fabs(windingNumber) > 0.5;
        }

        void updateBoundingBoxFromMesh()
        {
            if (vertexPosition_.empty())
            {
                boundingBoxMin_ = make_double3(0.0, 0.0, 0.0);
                boundingBoxMax_ = make_double3(0.0, 0.0, 0.0);
                return;
            }

            boundingBoxMin_ = vertexPosition_.front();
            boundingBoxMax_ = vertexPosition_.front();

            for (size_t vertexIndex = 1; vertexIndex < vertexPosition_.size(); ++vertexIndex)
            {
                const double3& vertex = vertexPosition_[vertexIndex];

                boundingBoxMin_.x = std::min(boundingBoxMin_.x, vertex.x);
                boundingBoxMin_.y = std::min(boundingBoxMin_.y, vertex.y);
                boundingBoxMin_.z = std::min(boundingBoxMin_.z, vertex.z);

                boundingBoxMax_.x = std::max(boundingBoxMax_.x, vertex.x);
                boundingBoxMax_.y = std::max(boundingBoxMax_.y, vertex.y);
                boundingBoxMax_.z = std::max(boundingBoxMax_.z, vertex.z);
            }
        }

        void setMeshInternal(const std::vector<double3>& vertexPosition,
                             const std::vector<int3>& triangleVertexIndex)
        {
            vertexPosition_ = vertexPosition;
            triangleVertexIndex_ = triangleVertexIndex;
            updateBoundingBoxFromMesh();
            clearGrid();
        }

    public:
        // =========================
        // Rule of Five
        // =========================
        TriangleMeshParticle() = default;

        TriangleMeshParticle(const std::vector<double3>& vertexPosition,
                             const std::vector<int3>& triangleVertexIndex)
            : vertexPosition_(vertexPosition),
              triangleVertexIndex_(triangleVertexIndex)
        {
            updateBoundingBoxFromMesh();
        }

        ~TriangleMeshParticle() override = default;
        TriangleMeshParticle(const TriangleMeshParticle&) = default;
        TriangleMeshParticle(TriangleMeshParticle&&) noexcept = default;
        TriangleMeshParticle& operator=(const TriangleMeshParticle&) = default;
        TriangleMeshParticle& operator=(TriangleMeshParticle&&) noexcept = default;

        // =========================
        // Host operations
        // =========================
        void setMesh(const std::vector<double3>& vertexPosition,
                     const std::vector<int3>& triangleVertexIndex)
        {
            setMeshInternal(vertexPosition, triangleVertexIndex);
        }

        void clearMesh()
        {
            vertexPosition_.clear();
            triangleVertexIndex_.clear();
            updateBoundingBoxFromMesh();
            clearGrid();
        }

        // =========================
        // Virtual interfaces
        // =========================
        bool isValid() const override
        {
            if (vertexPosition_.empty()) return false;
            if (triangleVertexIndex_.empty()) return false;

            for (size_t triangleIndex = 0; triangleIndex < triangleVertexIndex_.size(); ++triangleIndex)
            {
                const int3 triangle = triangleVertexIndex_[triangleIndex];

                if (triangle.x < 0 || triangle.y < 0 || triangle.z < 0) return false;
                if (triangle.x >= int(vertexPosition_.size())) return false;
                if (triangle.y >= int(vertexPosition_.size())) return false;
                if (triangle.z >= int(vertexPosition_.size())) return false;

                if (triangle.x == triangle.y || triangle.y == triangle.z || triangle.z == triangle.x)
                    return false;
            }

            return true;
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

            double minimumDistanceSquared = std::numeric_limits<double>::max();
            double3 closestPointOnSurface = make_double3(0.0, 0.0, 0.0);

            for (size_t triangleIndex = 0; triangleIndex < triangleVertexIndex_.size(); ++triangleIndex)
            {
                const int3 triangle = triangleVertexIndex_[triangleIndex];

                const double3& triangleVertex0 = vertexPosition_[triangle.x];
                const double3& triangleVertex1 = vertexPosition_[triangle.y];
                const double3& triangleVertex2 = vertexPosition_[triangle.z];

                const double3 candidateClosestPoint =
                    closestPointOnTriangle(point,
                                           triangleVertex0,
                                           triangleVertex1,
                                           triangleVertex2);

                const double candidateDistanceSquared =
                    lengthSquared(point - candidateClosestPoint);

                if (candidateDistanceSquared < minimumDistanceSquared)
                {
                    minimumDistanceSquared = candidateDistanceSquared;
                    closestPointOnSurface = candidateClosestPoint;
                }
            }

            const double unsignedDistance = std::sqrt(std::max(0.0, minimumDistanceSquared));
            const bool isInsideParticle = isPointInsideClosedTriangleMesh(point);

            implicitFunctionValue = isInsideParticle ? -unsignedDistance : unsignedDistance;

            if (unsignedDistance > 1e-30)
            {
                const double3 surfaceToPointDirection =
                    (point - closestPointOnSurface) / unsignedDistance;

                implicitFunctionGradient =
                    isInsideParticle ? (-surfaceToPointDirection) : surfaceToPointDirection;
            }
            else
            {
                implicitFunctionGradient = make_double3(0.0, 0.0, 0.0);
            }
        }

        double3 boundingBoxMin() const override
        {
            return boundingBoxMin_;
        }

        double3 boundingBoxMax() const override
        {
            return boundingBoxMax_;
        }

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
        bool levelSetPositiveInside_ {true};

    protected:
        // =========================
        // Helpers
        // =========================
        void updateMesh()
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
        BoxWall() = default;

        BoxWall(const double lx,
                const double ly,
                const double lz,
                const bool levelSetPositiveInside = true)
            : lx_(lx),
              ly_(ly),
              lz_(lz),
              levelSetPositiveInside_(levelSetPositiveInside)
        {
            updateMesh();
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
                       const double lz,
                       const bool levelSetPositiveInside = true)
        {
            lx_ = lx;
            ly_ = ly;
            lz_ = lz;
            levelSetPositiveInside_ = levelSetPositiveInside;
            updateMesh();
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
        bool levelSetPositiveInside() const { return levelSetPositiveInside_; }
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
        bool levelSetPositiveInside_ {true};

        int numSegmentsCircumference_ {32};
        int numSegmentsAxial_ {1};

    protected:
        // =========================
        // Helpers
        // =========================
        void updateMesh()
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
        CylinderWall() = default;

        CylinderWall(const double3& pointA,
                     const double3& pointB,
                     const double radius,
                     const bool levelSetPositiveInside = true,
                     const int numSegmentsCircumference = 32,
                     const int numSegmentsAxial = 1)
            : pointA_(pointA),
              pointB_(pointB),
              radius_(radius),
              levelSetPositiveInside_(levelSetPositiveInside),
              numSegmentsCircumference_(std::max(3, numSegmentsCircumference)),
              numSegmentsAxial_(std::max(1, numSegmentsAxial))
        {
            updateMesh();
        }

        ~CylinderWall() override = default;
        CylinderWall(const CylinderWall&) = default;
        CylinderWall(CylinderWall&&) noexcept = default;
        CylinderWall& operator=(const CylinderWall&) = default;
        CylinderWall& operator=(CylinderWall&&) noexcept = default;

        // =========================
        // Host operations
        // =========================
        void setGeometry(const double3& pointA,
                         const double3& pointB,
                         const double radius)
        {
            pointA_ = pointA;
            pointB_ = pointB;
            radius_ = radius;
            updateMesh();
            clearGrid();
        }

        void setMeshResolution(const int numSegmentsCircumference,
                               const int numSegmentsAxial)
        {
            numSegmentsCircumference_ = std::max(3, numSegmentsCircumference);
            numSegmentsAxial_ = std::max(1, numSegmentsAxial);
            updateMesh();
            clearGrid();
        }

        void setLevelSetSign(const bool levelSetPositiveInside)
        {
            levelSetPositiveInside_ = levelSetPositiveInside;
            clearGrid();
        }

        // =========================
        // Virtual interfaces
        // =========================
        bool isValid() const override
        {
            return (radius_ > 0.0 &&
                    length(pointB_ - pointA_) > 1e-30);
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

            const double clampedAxialCoordinate =
                Utility::clampValue(axialCoordinate, 0.0, axisLength);
            const double3 closestPointOnAxisSegment =
                pointA_ + clampedAxialCoordinate * axisDirection;

            const double3 radialVector = point - closestPointOnAxisSegment;
            const double radialDistance = length(radialVector);

            const double radialSignedDistance = radialDistance - radius_;
            const double axialSignedDistance =
                (axialCoordinate < 0.0) ? (-axialCoordinate) :
                (axialCoordinate > axisLength ? (axialCoordinate - axisLength) : 0.0);

            const double outsideRadial = std::max(radialSignedDistance, 0.0);
            const double outsideAxial = std::max(axialSignedDistance, 0.0);

            const double outsideDistance =
                std::sqrt(outsideRadial * outsideRadial + outsideAxial * outsideAxial);

            const double insideDistance =
                std::min(std::max(radialSignedDistance, axialSignedDistance), 0.0);

            double signedDistanceValue = outsideDistance + insideDistance;
            double3 signedDistanceGradient = make_double3(0.0, 0.0, 0.0);

            if (outsideDistance > 1e-30)
            {
                double3 radialDirection = make_double3(0.0, 0.0, 0.0);
                if (radialDistance > 1e-30)
                {
                    radialDirection = radialVector / radialDistance;
                }

                double3 axialDirection = make_double3(0.0, 0.0, 0.0);
                if (axialCoordinate < 0.0) axialDirection = -axisDirection;
                else if (axialCoordinate > axisLength) axialDirection = axisDirection;

                signedDistanceGradient =
                    (outsideRadial / outsideDistance) * radialDirection +
                    (outsideAxial / outsideDistance) * axialDirection;
            }
            else
            {
                const double distanceToSideWall = radius_ - radialDistance;
                const double distanceToCapA = axialCoordinate;
                const double distanceToCapB = axisLength - axialCoordinate;

                if (distanceToSideWall <= distanceToCapA &&
                    distanceToSideWall <= distanceToCapB)
                {
                    if (radialDistance > 1e-30)
                    {
                        signedDistanceGradient = radialVector / radialDistance;
                    }
                    else
                    {
                        double3 referenceDirection =
                            (std::fabs(axisDirection.x) < 0.9)
                            ? make_double3(1.0, 0.0, 0.0)
                            : make_double3(0.0, 1.0, 0.0);

                        signedDistanceGradient =
                            normalize(cross(axisDirection, referenceDirection));
                    }
                }
                else if (distanceToCapA <= distanceToCapB)
                {
                    signedDistanceGradient = -axisDirection;
                }
                else
                {
                    signedDistanceGradient = axisDirection;
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
        bool levelSetPositiveInside() const { return levelSetPositiveInside_; }
        int numSegmentsCircumference() const { return numSegmentsCircumference_; }
        int numSegmentsAxial() const { return numSegmentsAxial_; }
    };

} // namespace LevelSetObject