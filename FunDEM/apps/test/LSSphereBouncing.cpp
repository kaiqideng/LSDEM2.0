#include "kernel/LSSolver.h"

int main(const int argc, char** argv)
{
    LSSolver solver_("LSSphereBouncing");

    const double radius = 0.01;
    const double density = 2000.;
    const double height = 1.;

    const double3 planeMin = make_double3(-0.5, -0.5, 0.);
    const double3 planeMax = make_double3(0.5, 0.5, 0.);
    const double3 gravity = make_double3(0., 0., -9.81);

    const double collideTime = 0.005;

    const double sphereMass = (4. / 3.) * pi() * radius * radius * radius * density;
    const symMatrix sphereInertia = make_symMatrix(0.4 * sphereMass * radius * radius, 
        0.4 * sphereMass * radius * radius, 
        0.4 * sphereMass * radius * radius, 
        0., 
        0., 
        0.);
    const double normalStiffness = 0.5 * sphereMass * pi() * pi() / (collideTime * collideTime);

    const double sphereGridNodeSpacing = radius / 5.;
    const int3 sphereGridNodeSize = make_int3(13, 13, 13);
    const double3 sphereGridNodeLocalOrigin = make_double3(-0.5 * sphereGridNodeSpacing * (sphereGridNodeSize.x - 1), 
        -0.5 * sphereGridNodeSpacing * (sphereGridNodeSize.y - 1), 
        -0.5 * sphereGridNodeSpacing * (sphereGridNodeSize.z - 1));
    std::vector<double> sphereGridNodeSignedDistance(sphereGridNodeSize.x * sphereGridNodeSize.y * sphereGridNodeSize.z);
    for (int x = 0; x < sphereGridNodeSize.x; x++)
    {
        for (int y = 0; y < sphereGridNodeSize.y; y++)
        {
            for (int z = 0; z < sphereGridNodeSize.z; z++)
            {
                const int index = linearIndex3D(make_int3(x, y, z), sphereGridNodeSize);
                const double3 nodePosition = sphereGridNodeLocalOrigin + make_double3(double(x), double(y), double(z)) * sphereGridNodeSpacing;
                sphereGridNodeSignedDistance[index] = length(nodePosition) - radius;
            }
        }
    } 
    std::vector<double3> sphereBoundaryNodeLocalPosition;
    std::vector<int3> sphereBoundaryNodeConnectivity;
    const size_t numSegment = 6;
    for (size_t i = 0; i < numSegment; i++)
    {
        for (size_t j = 0; j < numSegment; j++)
        {
            const double theta = double(i) * pi() / (numSegment - 1);
            const double phi = double(j) * 2. * pi() / numSegment;
            const double x = radius * std::sin(theta) * std::cos(phi);
            const double y = radius * std::sin(theta) * std::sin(phi);
            const double z = radius * std::cos(theta);
            sphereBoundaryNodeLocalPosition.push_back(make_double3(x, y, z));
        }
    }
    for (size_t i = 0; i < numSegment - 1; i++)
    {
        for (size_t j = 0; j < numSegment; j++)
        {
            const int3 tri1 = make_int3(i * numSegment + j, (i + 1) * numSegment + j, (i + 1) * numSegment + (j + 1) % numSegment);
            const int3 tri2 = make_int3(i * numSegment + j, (i + 1) * numSegment + (j + 1) % numSegment, i * numSegment + (j + 1) % numSegment);
            sphereBoundaryNodeConnectivity.push_back(tri1);
            sphereBoundaryNodeConnectivity.push_back(tri2);
        }
    }

    solver_.addLSParticle(sphereBoundaryNodeLocalPosition, 
    sphereGridNodeSignedDistance, 
    sphereGridNodeLocalOrigin, 
    sphereGridNodeSize, 
    sphereGridNodeSpacing, 
    make_double3(0., 0., 1.), 
    make_double3(0., 0., 0.), 
    make_double3(0., 0., 0.),
    make_quaternion(1., 0., 0., 0.),
    sphereMass,
    sphereInertia,
    normalStiffness,
    0.,
    0.,
    1.,
    sphereBoundaryNodeConnectivity);

    const double planeGridNodeSpacing = std::max((planeMax - planeMin).x, (planeMax - planeMin).y);
    const int3 planeGridNodeSize = make_int3(2, 2, 2);
    const double3 planeGridNodeLocalOrigin = 0.5 * (planeMin + planeMax) - 0.5 * make_double3(
        (planeGridNodeSize.x - 1) * planeGridNodeSpacing, 
        (planeGridNodeSize.y - 1) * planeGridNodeSpacing, 
        (planeGridNodeSize.z - 1) * planeGridNodeSpacing);
    std::vector<double> planeGridNodeSignedDistance(planeGridNodeSize.x * planeGridNodeSize.y * planeGridNodeSize.z);
    for (int x = 0; x < planeGridNodeSize.x; x++)
    {
        for (int y = 0; y < planeGridNodeSize.y; y++)
        {
            for (int z = 0; z < planeGridNodeSize.z; z++)
            {
                const int index = linearIndex3D(make_int3(x, y, z), planeGridNodeSize);
                const double3 nodePosition = planeGridNodeLocalOrigin + make_double3(double(x), double(y), double(z)) * planeGridNodeSpacing;
                planeGridNodeSignedDistance[index] = nodePosition.z;
            }
        }
    }

    std::vector<double3> planeBoundaryNodeLocalPosition = {
        make_double3(planeMin.x, planeMin.y, 0.),
        make_double3(planeMax.x, planeMin.y, 0.),
        make_double3(planeMax.x, planeMax.y, 0.),
        make_double3(planeMin.x, planeMax.y, 0.)
    };
    std::vector<int3> planeBoundaryNodeConnectivity = {
        make_int3(0, 1, 2),
        make_int3(0, 2, 3)
    };

    solver_.addWall(planeBoundaryNodeLocalPosition,
    planeBoundaryNodeConnectivity, 
    planeGridNodeSignedDistance,
    planeGridNodeLocalOrigin,
    planeGridNodeSize, 
    planeGridNodeSpacing,
    0.5 * (planeMin + planeMax),
    make_quaternion(1., 0., 0., 0.),
    0.);

    solver_.solve(planeMin - make_double3(0., 0., height), 
    planeMax + make_double3(0., 0., height), 
    gravity, 
    collideTime / 50., 
    2., 
    200, 
    argc, 
    argv);
}