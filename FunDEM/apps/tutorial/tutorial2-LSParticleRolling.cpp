#include "kernel/LSSolver.h"
#include "kernel/LSParticleGenerator.h"
#include "kernel/spherePackGenerator.h"
#include "globalDamping.cuh"

struct SuperellipsoidParams 
{
    double rx{1.};
    double ry{1.};
    double rz{1.};
    double ee{1.};
    double en{1.};
};

class solver:
    public LSSolver
{
public:
    solver(): LSSolver(0) {}

    double dampingCoefficient = 0.2;

    void addExternalForceTorque(const double time) override
    {
        launchAddGlobalDampingForceTorque(getLSParticle().force(),
        getLSParticle().torque(),
        getLSParticle().velocity(),
        getLSParticle().angularVelocity(),
        dampingCoefficient,
        getLSParticle().num_device(),
        getLSParticle().gridDim(),
        getLSParticle().blockDim(),
        0);
    }
};

int main()
{
    solver solver_;

    const double l = 1.;
    const double3 boxMin = make_double3(-l, -l, -l);
    const double3 boxMax = make_double3(l, l, l);
    const double rMin = 0.1;
    const double rMax = 0.3;
    const double density = 1000.;
    const int nSpheres_max = 3000;

    SpherePack SpherePack_ = generateNonOverlappingSpheresInBox_LargeFirst(boxMin, 
    boxMax, 
    nSpheres_max, 
    0.5 * (rMin + rMax), 
    rMax);

    for (size_t i = 0; i < SpherePack_.centers.size(); i++)
    {
        SuperellipsoidParams s;
        const int shapeIndex = rand_deterministic(0, 3);
        if (shapeIndex == 0)
        {
            s.rx = 0.4, s.ry = 1., s.rz = 0.8, s.ee = 0.4, s.en = 1.6;
        }
        else if (shapeIndex == 1)
        {
            s.rx = 0.42, s.ry = 1., s.rz = 0.83, s.ee = 0.1, s.en = 1.;
        }
        else if (shapeIndex == 2)
        {
            s.rx = 1., s.ry = 1., s.rz = 1., s.ee = 1., s.en = 0.5;
        }
        else if (shapeIndex == 3)
        {
            s.rx = 0.5, s.ry = 0.7, s.rz = 1., s.ee = 1.4, s.en = 1.2;
        }
        s.rx *= 0.7 * SpherePack_.radii[i];
        s.ry *= 0.7 * SpherePack_.radii[i];
        s.rz *= 0.7 * SpherePack_.radii[i];
        quaternion q = randomQuaternionUniform_deterministic();

        SuperellipsoidParticle SP;
        SP.setParams(s.rx, s.ry, s.rz, s.ee, s.en);
        SP.buildGridByResolution();
        solver_.addLSParticle(SP.generateSurfacePointsUniform(int(10000 * SpherePack_.radii[i])), 
        SP.gridInfo().gridNodeLevelSetFunctionValue, 
        SP.gridInfo().gridOrigin, 
        SP.gridInfo().gridNodeSize, 
        SP.gridInfo().gridNodeSpacing, 
        SpherePack_.centers[i], 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        q, 
        6.e5, 
        1.8e5, 
        0.577,
        density);
    }

    CylinderWall CW;
    CW.setGeometry(make_double3(- l, 0, 0.), 
    make_double3(l, 0., 0.), 
    2. * l);
    CW.buildGridByResolution(100);
    solver_.addWall(CW.generateSurfacePointsUniform(1000), 
    CW.gridInfo().gridNodeLevelSetFunctionValue, 
    CW.gridInfo().gridOrigin, 
    CW.gridInfo().gridNodeSize, 
    CW.gridInfo().gridNodeSpacing, 
    make_double3(0., 0., 0.), 
    make_quaternion(1., 0., 0., 0.), 
    6.e5, 
    1.8e5, 
    0.577);

    solver_.solve(make_double3(-l, -2. * l, -2. * l), 
    make_double3(l, 2. * l, 2. * l), 
    make_double3(0., 0., -9.81), 
    1.e-4, 
    2.5, 
    25, 
    "tutorial2-period1");

    SpherePack_ = generateNonOverlappingSpheresInBox_LargeFirst(boxMin, 
    boxMax, 
    nSpheres_max, 
    rMin, 
    0.5 * (rMin + rMax));

    for (size_t i = 0; i < SpherePack_.centers.size(); i++)
    {
        SuperellipsoidParams s;
        const int shapeIndex = rand_deterministic(0, 3);
        if (shapeIndex == 0)
        {
            s.rx = 0.4, s.ry = 1., s.rz = 0.8, s.ee = 0.4, s.en = 1.6;
        }
        else if (shapeIndex == 1)
        {
            s.rx = 0.42, s.ry = 1., s.rz = 0.83, s.ee = 0.1, s.en = 1.;
        }
        else if (shapeIndex == 2)
        {
            s.rx = 1., s.ry = 1., s.rz = 1., s.ee = 1., s.en = 0.5;
        }
        else if (shapeIndex == 3)
        {
            s.rx = 0.5, s.ry = 0.7, s.rz = 1., s.ee = 1.4, s.en = 1.2;
        }
        s.rx *= 0.7 * SpherePack_.radii[i];
        s.ry *= 0.7 * SpherePack_.radii[i];
        s.rz *= 0.7 * SpherePack_.radii[i];
        quaternion q = randomQuaternionUniform_deterministic();

        SuperellipsoidParticle SP;
        SP.setParams(s.rx, s.ry, s.rz, s.ee, s.en);
        SP.buildGridByResolution();
        solver_.addLSParticle(SP.generateSurfacePointsUniform(int(10000 * SpherePack_.radii[i])), 
        SP.gridInfo().gridNodeLevelSetFunctionValue, 
        SP.gridInfo().gridOrigin, 
        SP.gridInfo().gridNodeSize, 
        SP.gridInfo().gridNodeSpacing, 
        SpherePack_.centers[i], 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        q, 
        6.e5, 
        1.8e5, 
        0.577,
        density);
    }

    solver_.solve(make_double3(-l, -2. * l, -2. * l), 
    make_double3(l, 2. * l, 2. * l), 
    make_double3(0., 0., -9.81), 
    1.e-4, 
    2.5, 
    25, 
    "tutorial2-period2");

    solver_.setFixedAngularVelocityToWall(0, make_double3(M_PI, 0., 0.));
    solver_.solve(make_double3(-l, -2. * l, -2. * l), 
    make_double3(l, 2. * l, 2. * l), 
    make_double3(0., 0., -9.81), 
    1.e-4, 
    10., 
    100, 
    "tutorial2-period3");
}