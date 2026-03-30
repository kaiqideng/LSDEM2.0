#include "CUDAKernelFunction/myUtility/myQua.h"
#include <algorithm>
#include <random>
#include <unordered_map>
#include <array>

inline int rand_deterministic(int min, int max)
{
    if (max <= min) return min;
    static std::mt19937 rng(123456); // fixed seed
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

inline quaternion randomQuaternionUniform_deterministic()
{
    static std::mt19937 rng(123456);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    const double u1 = U(rng), u2 = U(rng), u3 = U(rng);
    const double s1 = std::sqrt(1.0 - u1);
    const double s2 = std::sqrt(u1);
    const double a = 2.0 * M_PI * u2;
    const double b = 2.0 * M_PI * u3;

    return quaternion{
        s2 * std::cos(b),
        s1 * std::sin(a),
        s1 * std::cos(a),
        s2 * std::sin(b)
    };
}

struct SpherePack
{
    std::vector<double3> centers;
    std::vector<double> radii;
};

inline SpherePack generateNonOverlappingSpheresInBox_LargeFirst(const double3 boxMin,
const double3 boxMax,
const int nSpheres,
const double rMin,
const double rMax,
const uint32_t seed = 12345u,
const int maxAttemptsPerSphere = 2000)
{
    SpherePack out;

    if (nSpheres <= 0) return out;
    if (rMin <= 0.0) return out;
    if (rMax < rMin) return out;

    const double3 bmin = make_double3(std::min(boxMin.x, boxMax.x),
    std::min(boxMin.y, boxMax.y),
    std::min(boxMin.z, boxMax.z));

    const double3 bmax = make_double3(std::max(boxMin.x, boxMax.x),
    std::max(boxMin.y, boxMax.y),
    std::max(boxMin.z, boxMax.z));

    if ((bmax.x - bmin.x) < 2.0 * rMin) return out;
    if ((bmax.y - bmin.y) < 2.0 * rMin) return out;
    if ((bmax.z - bmin.z) < 2.0 * rMin) return out;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ur01(0.0, 1.0);
    std::uniform_real_distribution<double> urR(rMin, rMax);

    // ------------------------------------------------------------
    // 1) Pre-generate radii then sort large -> small
    // ------------------------------------------------------------
    std::vector<double> radiiTarget;
    radiiTarget.reserve((size_t)nSpheres);
    for (int i = 0; i < nSpheres; ++i) radiiTarget.push_back(urR(rng));
    std::sort(radiiTarget.begin(), radiiTarget.end(), std::greater<double>());

    // ------------------------------------------------------------
    // 2) Spatial hash grid (cell size based on max radius)
    // ------------------------------------------------------------
    const double cellSize = 2.0 * rMax;

    auto cellCoord = [&](const double3& p) -> std::array<int, 3>
    {
        const double fx = (p.x - bmin.x) / cellSize;
        const double fy = (p.y - bmin.y) / cellSize;
        const double fz = (p.z - bmin.z) / cellSize;
        return { (int)std::floor(fx), (int)std::floor(fy), (int)std::floor(fz) };
    };

    auto packKey = [&](int cx, int cy, int cz) -> int64_t
    {
        const int64_t B = (1LL << 20);
        const int64_t x = (int64_t)cx + B;
        const int64_t y = (int64_t)cy + B;
        const int64_t z = (int64_t)cz + B;
        return (x << 42) ^ (y << 21) ^ z;
    };

    std::unordered_map<int64_t, std::vector<int>> buckets;
    buckets.reserve((size_t)nSpheres * 2);

    auto overlapsAny = [&](const double3& c, const double r) -> bool
    {
        const auto cc = cellCoord(c);

        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int nx = cc[0] + dx;
            const int ny = cc[1] + dy;
            const int nz = cc[2] + dz;

            const int64_t key = packKey(nx, ny, nz);
            auto it = buckets.find(key);
            if (it == buckets.end()) continue;

            for (int id : it->second)
            {
                const double3 cj = out.centers[(size_t)id];
                const double rj = out.radii[(size_t)id];

                const double dx = c.x - cj.x;
                const double dy = c.y - cj.y;
                const double dz = c.z - cj.z;
                const double rr = r + rj;

                if (dx*dx + dy*dy + dz*dz < rr * rr) return true;
            }
        }
    return false;
    };

    auto sampleCenter = [&](const double r) -> double3
    {
        const double xmin = bmin.x + r;
        const double ymin = bmin.y + r;
        const double zmin = bmin.z + r;

        const double xmax = bmax.x - r;
        const double ymax = bmax.y - r;
        const double zmax = bmax.z - r;

        return make_double3(
            xmin + ur01(rng) * (xmax - xmin),
            ymin + ur01(rng) * (ymax - ymin),
            zmin + ur01(rng) * (zmax - zmin)
        );
    };

    // ------------------------------------------------------------
    // 3) Place spheres: large -> small
    // ------------------------------------------------------------
    out.centers.reserve((size_t)nSpheres);
    out.radii.reserve((size_t)nSpheres);

    for (int s = 0; s < nSpheres; ++s)
    {
        const double r = radiiTarget[(size_t)s];

        // If this radius cannot fit at all, skip it (continue to smaller ones)
        if ((bmax.x - bmin.x) < 2.0 * r) continue;
        if ((bmax.y - bmin.y) < 2.0 * r) continue;
        if ((bmax.z - bmin.z) < 2.0 * r) continue;

        bool placed = false;
        for (int attempt = 0; attempt < maxAttemptsPerSphere; ++attempt)
        {
            const double3 c = sampleCenter(r);
            if (overlapsAny(c, r)) continue;

            const int id = (int)out.centers.size();
            out.centers.push_back(c);
            out.radii.push_back(r);

            const auto cc = cellCoord(c);
            buckets[packKey(cc[0], cc[1], cc[2])].push_back(id);

            placed = true;
            break;
        }

        (void)placed;
    }

    return out;
}

inline SpherePack generateNonOverlappingSpheresInCylinder_LargeFirst(const double3 cylinderPointA,
const double3 cylinderPointB,
const double cylinderRadius,
const int nSpheres,
const double rMin,
const double rMax,
const uint32_t seed = 12345u,
const int maxAttemptsPerSphere = 2000)
{
    SpherePack out;

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

    // Build an orthonormal basis (u, v, w), where w = axisDirection
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

    // ------------------------------------------------------------
    // 1) Pre-generate radii then sort large -> small
    // ------------------------------------------------------------
    std::vector<double> radiiTarget;
    radiiTarget.reserve((size_t)nSpheres);
    for (int i = 0; i < nSpheres; ++i) radiiTarget.push_back(urR(rng));
    std::sort(radiiTarget.begin(), radiiTarget.end(), std::greater<double>());

    // ------------------------------------------------------------
    // 2) Spatial hash grid
    // ------------------------------------------------------------
    const double cellSize = 2.0 * rMax;

    const double3 cylinderBoundingBoxMin = make_double3(std::min(cylinderPointA.x, cylinderPointB.x) - cylinderRadius,
                                                        std::min(cylinderPointA.y, cylinderPointB.y) - cylinderRadius,
                                                        std::min(cylinderPointA.z, cylinderPointB.z) - cylinderRadius);

    auto cellCoord = [&](const double3& p) -> std::array<int, 3>
    {
        const double fx = (p.x - cylinderBoundingBoxMin.x) / cellSize;
        const double fy = (p.y - cylinderBoundingBoxMin.y) / cellSize;
        const double fz = (p.z - cylinderBoundingBoxMin.z) / cellSize;
        return { (int)std::floor(fx), (int)std::floor(fy), (int)std::floor(fz) };
    };

    auto packKey = [&](int cx, int cy, int cz) -> int64_t
    {
        const int64_t B = (1LL << 20);
        const int64_t x = (int64_t)cx + B;
        const int64_t y = (int64_t)cy + B;
        const int64_t z = (int64_t)cz + B;
        return (x << 42) ^ (y << 21) ^ z;
    };

    std::unordered_map<int64_t, std::vector<int>> buckets;
    buckets.reserve((size_t)nSpheres * 2);

    auto overlapsAny = [&](const double3& center, const double radius) -> bool
    {
        const auto cc = cellCoord(center);

        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int nx = cc[0] + dx;
            const int ny = cc[1] + dy;
            const int nz = cc[2] + dz;

            const int64_t key = packKey(nx, ny, nz);
            auto it = buckets.find(key);
            if (it == buckets.end()) continue;

            for (int id : it->second)
            {
                const double3 placedCenter = out.centers[(size_t)id];
                const double placedRadius = out.radii[(size_t)id];

                const double3 delta = center - placedCenter;
                const double rr = radius + placedRadius;

                if (dot(delta, delta) < rr * rr) return true;
            }
        }

    return false;
    };

    auto isInsideCylinder = [&](const double3& center, const double radius) -> bool
    {
        const double3 fromA = center - cylinderPointA;
        const double axialCoordinate = dot(fromA, axisDirection);

        if (axialCoordinate < radius) return false;
        if (axialCoordinate > cylinderLength - radius) return false;

        const double3 projectedPointOnAxis = cylinderPointA + axialCoordinate * axisDirection;
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
        const double azimuthAngle = 2.0 * M_PI * v;

        const double localU = radialDistance * std::cos(azimuthAngle);
        const double localV = radialDistance * std::sin(azimuthAngle);
        const double axialCoordinate = effectiveAxialMin + ur01(rng) * (effectiveAxialMax - effectiveAxialMin);

        return cylinderPointA +
               axialCoordinate * axisDirection +
               localU * radialDirectionU +
               localV * radialDirectionV;
    };

    // ------------------------------------------------------------
    // 3) Place spheres: large -> small
    // ------------------------------------------------------------
    out.centers.reserve((size_t)nSpheres);
    out.radii.reserve((size_t)nSpheres);

    for (int s = 0; s < nSpheres; ++s)
    {
        const double radius = radiiTarget[(size_t)s];

        if (cylinderLength < 2.0 * radius) continue;
        if (cylinderRadius < radius) continue;

        bool placed = false;

        for (int attempt = 0; attempt < maxAttemptsPerSphere; ++attempt)
        {
            const double3 center = sampleCenter(radius);

            if (!isInsideCylinder(center, radius)) continue;
            if (overlapsAny(center, radius)) continue;

            const int id = (int)out.centers.size();
            out.centers.push_back(center);
            out.radii.push_back(radius);

            const auto cc = cellCoord(center);
            buckets[packKey(cc[0], cc[1], cc[2])].push_back(id);

            placed = true;
            break;
        }

        (void)placed;
    }

    return out;
}