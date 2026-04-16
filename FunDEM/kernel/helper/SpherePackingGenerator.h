// NOTE:
// Most of the code on this page was generated with the assistance of AI tools.

#pragma once
#include "kernel/CUDAKernelFunction/myUtility/myVec.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace SpherePacking
{
    //=========================================================================
    // Pack
    //=========================================================================
    struct Pack
    {
        std::vector<double3> centers;
        std::vector<double> radii;
    };

    //=========================================================================
    // Utility
    //=========================================================================
    namespace Utility
    {
        inline double3 sortedMin(const double3& a,
                                 const double3& b)
        {
            return make_double3(std::min(a.x, b.x),
                                std::min(a.y, b.y),
                                std::min(a.z, b.z));
        }

        inline double3 sortedMax(const double3& a,
                                 const double3& b)
        {
            return make_double3(std::max(a.x, b.x),
                                std::max(a.y, b.y),
                                std::max(a.z, b.z));
        }

        inline bool isSphereInsideBox(const double3& center,
                                      const double radius,
                                      const double3& boxMin,
                                      const double3& boxMax)
        {
            return (center.x - radius >= boxMin.x &&
                    center.x + radius <= boxMax.x &&
                    center.y - radius >= boxMin.y &&
                    center.y + radius <= boxMax.y &&
                    center.z - radius >= boxMin.z &&
                    center.z + radius <= boxMax.z);
        }

        inline int64_t packSpatialHashKey(const int cx,
                                          const int cy,
                                          const int cz)
        {
            const int64_t B = (1LL << 20);
            const int64_t x = (int64_t)cx + B;
            const int64_t y = (int64_t)cy + B;
            const int64_t z = (int64_t)cz + B;
            return (x << 42) ^ (y << 21) ^ z;
        }
    }

    //=========================================================================
    // Build non-overlapping sphere pack in box, large first
    //=========================================================================
    inline Pack buildNonOverlappingInBox_LargeFirst(const double3 boxMin,
                                                    const double3 boxMax,
                                                    const int nSpheres,
                                                    const double rMin,
                                                    const double rMax,
                                                    const uint32_t seed = 12345u,
                                                    const int maxAttemptsPerSphere = 2000)
    {
        Pack out;

        if (nSpheres <= 0) return out;
        if (rMin <= 0.0) return out;
        if (rMax < rMin) return out;

        const double3 bmin = Utility::sortedMin(boxMin, boxMax);
        const double3 bmax = Utility::sortedMax(boxMin, boxMax);

        if ((bmax.x - bmin.x) < 2.0 * rMin) return out;
        if ((bmax.y - bmin.y) < 2.0 * rMin) return out;
        if ((bmax.z - bmin.z) < 2.0 * rMin) return out;

        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> ur01(0.0, 1.0);
        std::uniform_real_distribution<double> urR(rMin, rMax);

        // ---------------------------------------------------------------------
        // 1) Pre-generate radii and sort from large to small
        // ---------------------------------------------------------------------
        std::vector<double> targetRadii;
        targetRadii.reserve((size_t)nSpheres);

        for (int i = 0; i < nSpheres; ++i)
        {
            targetRadii.push_back(urR(rng));
        }

        std::sort(targetRadii.begin(), targetRadii.end(), std::greater<double>());

        // ---------------------------------------------------------------------
        // 2) Spatial hash grid
        // ---------------------------------------------------------------------
        const double cellSize = 2.0 * rMax;

        auto cellCoord = [&](const double3& p) -> std::array<int, 3>
        {
            const double fx = (p.x - bmin.x) / cellSize;
            const double fy = (p.y - bmin.y) / cellSize;
            const double fz = (p.z - bmin.z) / cellSize;

            return {(int)std::floor(fx),
                    (int)std::floor(fy),
                    (int)std::floor(fz)};
        };

        std::unordered_map<int64_t, std::vector<int>> buckets;
        buckets.reserve((size_t)nSpheres * 2);

        auto overlapsAny = [&](const double3& center,
                               const double radius) -> bool
        {
            const auto cc = cellCoord(center);

            for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int nx = cc[0] + dx;
                const int ny = cc[1] + dy;
                const int nz = cc[2] + dz;

                const int64_t key = Utility::packSpatialHashKey(nx, ny, nz);
                auto it = buckets.find(key);
                if (it == buckets.end()) continue;

                for (const int id : it->second)
                {
                    const double3 placedCenter = out.centers[(size_t)id];
                    const double placedRadius = out.radii[(size_t)id];

                    const double dx = center.x - placedCenter.x;
                    const double dy = center.y - placedCenter.y;
                    const double dz = center.z - placedCenter.z;
                    const double rr = radius + placedRadius;

                    if (dx * dx + dy * dy + dz * dz < rr * rr)
                    {
                        return true;
                    }
                }
            }

        return false;
        };

        auto sampleCenter = [&](const double radius) -> double3
        {
            const double xmin = bmin.x + radius;
            const double ymin = bmin.y + radius;
            const double zmin = bmin.z + radius;

            const double xmax = bmax.x - radius;
            const double ymax = bmax.y - radius;
            const double zmax = bmax.z - radius;

            return make_double3(xmin + ur01(rng) * (xmax - xmin),
                                ymin + ur01(rng) * (ymax - ymin),
                                zmin + ur01(rng) * (zmax - zmin));
        };

        // ---------------------------------------------------------------------
        // 3) Place spheres from large to small
        // ---------------------------------------------------------------------
        out.centers.reserve((size_t)nSpheres);
        out.radii.reserve((size_t)nSpheres);

        for (int s = 0; s < nSpheres; ++s)
        {
            const double radius = targetRadii[(size_t)s];

            if ((bmax.x - bmin.x) < 2.0 * radius) continue;
            if ((bmax.y - bmin.y) < 2.0 * radius) continue;
            if ((bmax.z - bmin.z) < 2.0 * radius) continue;

            for (int attempt = 0; attempt < maxAttemptsPerSphere; ++attempt)
            {
                const double3 center = sampleCenter(radius);
                if (overlapsAny(center, radius)) continue;

                const int id = (int)out.centers.size();
                out.centers.push_back(center);
                out.radii.push_back(radius);

                const auto cc = cellCoord(center);
                buckets[Utility::packSpatialHashKey(cc[0], cc[1], cc[2])].push_back(id);
                break;
            }
        }

        return out;
    }

    //=========================================================================
    // Build non-overlapping sphere pack in cylinder, large first
    //=========================================================================
    inline Pack buildNonOverlappingInCylinder_LargeFirst(const double3 cylinderPointA,
                                                         const double3 cylinderPointB,
                                                         const double cylinderRadius,
                                                         const int nSpheres,
                                                         const double rMin,
                                                         const double rMax,
                                                         const uint32_t seed = 12345u,
                                                         const int maxAttemptsPerSphere = 2000)
    {
        Pack out;

        if (nSpheres <= 0) return out;
        if (rMin <= 0.0) return out;
        if (rMax < rMin) return out;
        if (cylinderRadius <= 0.0) return out;

        const double3 axisVector = cylinderPointB - cylinderPointA;
        const double cylinderLength = length(axisVector);

        if (cylinderLength <= 0.0) return out;
        if (cylinderLength < 2.0 * rMin) return out;
        if (cylinderRadius < rMin) return out;

        const double3 axisDirection = axisVector / cylinderLength;

        double3 auxiliaryDirection;
        if (std::fabs(axisDirection.x) < 0.9)
        {
            auxiliaryDirection = make_double3(1.0, 0.0, 0.0);
        }
        else
        {
            auxiliaryDirection = make_double3(0.0, 1.0, 0.0);
        }

        const double3 radialDirectionU = normalize(cross(axisDirection, auxiliaryDirection));
        const double3 radialDirectionV = normalize(cross(axisDirection, radialDirectionU));

        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> ur01(0.0, 1.0);
        std::uniform_real_distribution<double> urR(rMin, rMax);

        // ---------------------------------------------------------------------
        // 1) Pre-generate radii and sort from large to small
        // ---------------------------------------------------------------------
        std::vector<double> targetRadii;
        targetRadii.reserve((size_t)nSpheres);

        for (int i = 0; i < nSpheres; ++i)
        {
            targetRadii.push_back(urR(rng));
        }

        std::sort(targetRadii.begin(), targetRadii.end(), std::greater<double>());

        // ---------------------------------------------------------------------
        // 2) Spatial hash grid
        // ---------------------------------------------------------------------
        const double cellSize = 2.0 * rMax;

        const double3 cylinderBoundingBoxMin =
            make_double3(std::min(cylinderPointA.x, cylinderPointB.x) - cylinderRadius,
                         std::min(cylinderPointA.y, cylinderPointB.y) - cylinderRadius,
                         std::min(cylinderPointA.z, cylinderPointB.z) - cylinderRadius);

        auto cellCoord = [&](const double3& p) -> std::array<int, 3>
        {
            const double fx = (p.x - cylinderBoundingBoxMin.x) / cellSize;
            const double fy = (p.y - cylinderBoundingBoxMin.y) / cellSize;
            const double fz = (p.z - cylinderBoundingBoxMin.z) / cellSize;

            return {(int)std::floor(fx),
                    (int)std::floor(fy),
                    (int)std::floor(fz)};
        };

        std::unordered_map<int64_t, std::vector<int>> buckets;
        buckets.reserve((size_t)nSpheres * 2);

        auto overlapsAny = [&](const double3& center,
                               const double radius) -> bool
        {
            const auto cc = cellCoord(center);

            for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int nx = cc[0] + dx;
                const int ny = cc[1] + dy;
                const int nz = cc[2] + dz;

                const int64_t key = Utility::packSpatialHashKey(nx, ny, nz);
                auto it = buckets.find(key);
                if (it == buckets.end()) continue;

                for (const int id : it->second)
                {
                    const double3 placedCenter = out.centers[(size_t)id];
                    const double placedRadius = out.radii[(size_t)id];
                    const double3 delta = center - placedCenter;
                    const double rr = radius + placedRadius;

                    if (dot(delta, delta) < rr * rr)
                    {
                        return true;
                    }
                }
            }

        return false;
        };

        auto isInsideCylinder = [&](const double3& center,
                                    const double radius) -> bool
        {
            const double3 fromA = center - cylinderPointA;
            const double axialCoordinate = dot(fromA, axisDirection);

            if (axialCoordinate < radius) return false;
            if (axialCoordinate > cylinderLength - radius) return false;

            const double3 projectedPointOnAxis =
                cylinderPointA + axialCoordinate * axisDirection;
            const double3 radialVector = center - projectedPointOnAxis;
            const double radialDistanceSquared = dot(radialVector, radialVector);

            const double allowedRadialDistance = cylinderRadius - radius;
            if (allowedRadialDistance < 0.0) return false;

            return radialDistanceSquared <= allowedRadialDistance * allowedRadialDistance;
        };

        auto sampleCenter = [&](const double radius) -> double3
        {
            const double effectiveCylinderRadius = cylinderRadius - radius;
            const double effectiveAxialMin = radius;
            const double effectiveAxialMax = cylinderLength - radius;

            const double u = ur01(rng);
            const double v = ur01(rng);

            const double radialDistance = effectiveCylinderRadius * std::sqrt(u);
            const double azimuthAngle = 2.0 * pi() * v;

            const double localU = radialDistance * std::cos(azimuthAngle);
            const double localV = radialDistance * std::sin(azimuthAngle);
            const double axialCoordinate =
                effectiveAxialMin + ur01(rng) * (effectiveAxialMax - effectiveAxialMin);

            return cylinderPointA +
                   axialCoordinate * axisDirection +
                   localU * radialDirectionU +
                   localV * radialDirectionV;
        };

        // ---------------------------------------------------------------------
        // 3) Place spheres from large to small
        // ---------------------------------------------------------------------
        out.centers.reserve((size_t)nSpheres);
        out.radii.reserve((size_t)nSpheres);

        for (int s = 0; s < nSpheres; ++s)
        {
            const double radius = targetRadii[(size_t)s];

            if (cylinderLength < 2.0 * radius) continue;
            if (cylinderRadius < radius) continue;

            for (int attempt = 0; attempt < maxAttemptsPerSphere; ++attempt)
            {
                const double3 center = sampleCenter(radius);

                if (!isInsideCylinder(center, radius)) continue;
                if (overlapsAny(center, radius)) continue;

                const int id = (int)out.centers.size();
                out.centers.push_back(center);
                out.radii.push_back(radius);

                const auto cc = cellCoord(center);
                buckets[Utility::packSpatialHashKey(cc[0], cc[1], cc[2])].push_back(id);
                break;
            }
        }

        return out;
    }

    //=========================================================================
    // Build regular sphere pack in box
    //=========================================================================
    // Packing type:
    //   Simple cubic packing
    //
    // Rule:
    //   Only spheres fully inside the box are kept.
    inline Pack buildRegularInBox(const double3 boxMin,
                                  const double3 boxMax,
                                  const double sphereRadius)
    {
        if (sphereRadius <= 0.0)
        {
            throw std::runtime_error("sphereRadius must be positive.");
        }

        if (boxMax.x <= boxMin.x || boxMax.y <= boxMin.y || boxMax.z <= boxMin.z)
        {
            throw std::runtime_error("Invalid box: boxMax must be larger than boxMin.");
        }

        Pack pack;

        const double r = sphereRadius;
        const double eps = 1.0e-12;

        const double xMin = boxMin.x + r;
        const double yMin = boxMin.y + r;
        const double zMin = boxMin.z + r;

        const double xMax = boxMax.x - r;
        const double yMax = boxMax.y - r;
        const double zMax = boxMax.z - r;

        if (xMin > xMax || yMin > yMax || zMin > zMax)
        {
            return pack;
        }

        const double dx = 2.0 * r;
        const double dy = 2.0 * r;
        const double dz = 2.0 * r;

        for (double z = zMin; z <= zMax + eps; z += dz)
        {
            for (double y = yMin; y <= yMax + eps; y += dy)
            {
                for (double x = xMin; x <= xMax + eps; x += dx)
                {
                    pack.centers.push_back(make_double3(x, y, z));
                    pack.radii.push_back(r);
                }
            }
        }

        return pack;
    }

    //=========================================================================
    // Build hex sphere pack in box
    //=========================================================================
    // Packing type:
    //   3D HCP-style staggered packing
    //
    // In-plane spacing:
    //   dx = 2r
    //   dy = sqrt(3) r
    //
    // Layer spacing:
    //   dz = sqrt(8/3) r
    //
    // Shift rule:
    //   odd row   -> x + r
    //   odd layer -> x + r, y + sqrt(3) r / 3
    //
    // Rule:
    //   Only spheres fully inside the box are kept.
    inline Pack buildHexInBox(const double3 boxMin,
                              const double3 boxMax,
                              const double sphereRadius)
    {
        if (sphereRadius <= 0.0)
        {
            throw std::runtime_error("sphereRadius must be positive.");
        }

        if (boxMax.x <= boxMin.x || boxMax.y <= boxMin.y || boxMax.z <= boxMin.z)
        {
            throw std::runtime_error("Invalid box: boxMax must be larger than boxMin.");
        }

        Pack pack;

        const double r = sphereRadius;
        const double eps = 1.0e-12;

        const double xMin = boxMin.x + r;
        const double yMin = boxMin.y + r;
        const double zMin = boxMin.z + r;

        const double xMax = boxMax.x - r;
        const double yMax = boxMax.y - r;
        const double zMax = boxMax.z - r;

        if (xMin > xMax || yMin > yMax || zMin > zMax)
        {
            return pack;
        }

        const double dx = 2.0 * r;
        const double dy = std::sqrt(3.0) * r;
        const double dz = std::sqrt(8.0 / 3.0) * r;

        const double layerShiftX = r;
        const double layerShiftY = std::sqrt(3.0) * r / 3.0;

        int k = 0;
        for (double zBase = zMin; zBase <= zMax + eps; zBase += dz, ++k)
        {
            const bool oddLayer = (k % 2 == 1);
            const double yStart = yMin + (oddLayer ? layerShiftY : 0.0);

            int j = 0;
            for (double yBase = yStart; yBase <= yMax + eps; yBase += dy, ++j)
            {
                const bool oddRow = (j % 2 == 1);

                double xStart = xMin;
                if (oddRow)   xStart += r;
                if (oddLayer) xStart += layerShiftX;

                for (double x = xStart; x <= xMax + eps; x += dx)
                {
                    const double3 center = make_double3(x, yBase, zBase);

                    if (Utility::isSphereInsideBox(center, r, boxMin, boxMax))
                    {
                        pack.centers.push_back(center);
                        pack.radii.push_back(r);
                    }
                }
            }
        }

        return pack;
    }

} // namespace SpherePacking