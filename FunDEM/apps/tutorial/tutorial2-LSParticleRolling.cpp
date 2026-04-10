#include "kernel/LSSolver.h"
#include "kernel/helper/LevelSetObjectGenerator.h"
#include "kernel/helper/SpherePackingGenerator.h"
#include "globalDamping.cuh"

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
    solver(): LSSolver("tutorial2") {}

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

int main(const int argc, char** argv)
{
    solver solver_;

    const double l = 1.;
    
    const double rMin = 0.1;
    const double rMax = 0.3;
    const double density = 1000.;
    const int nSpheres_max = 3000;

    SpherePacking::Pack Pack_ = SpherePacking::buildNonOverlappingInCylinder_LargeFirst(make_double3(- l, 0, 0.), 
    make_double3(l, 0., 0.), 
    2 * l, 
    nSpheres_max, 
    rMin, 
    rMax);

    for (size_t i = 0; i < Pack_.centers.size(); i++)
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
        s.rx *= 0.7 * Pack_.radii[i];
        s.ry *= 0.7 * Pack_.radii[i];
        s.rz *= 0.7 * Pack_.radii[i];
        quaternion q = randomQuaternionUniform_deterministic();

        LevelSetObject::SuperellipsoidParticle SP;
        SP.setParams(s.rx, s.ry, s.rz, s.ee, s.en);
        SP.buildGridByResolution();
        solver_.addLSParticle(SP.generateSurfacePointsUniform(int(10000 * Pack_.radii[i])), 
        SP.gridInfo().gridNodeLevelSetFunctionValue, 
        SP.gridInfo().gridOrigin, 
        SP.gridInfo().gridNodeSize, 
        SP.gridInfo().gridNodeSpacing, 
        Pack_.centers[i], 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        q, 
        6.e5, 
        1.8e5, 
        0.577,
        1.,
        density);
    }

    LevelSetObject::CylinderWall CW;
    CW.setGeometry(make_double3(- l, 0, 0.), 
    make_double3(l, 0., 0.), 
    2. * l);
    CW.buildGridByResolution(100);
    solver_.addWall(CW.vertexPosition(), 
    CW.triangleVertexIndex(), 
    CW.gridInfo().gridNodeLevelSetFunctionValue, 
    CW.gridInfo().gridOrigin, 
    CW.gridInfo().gridNodeSize, 
    CW.gridInfo().gridNodeSpacing, 
    make_double3(0., 0., 0.), 
    make_quaternion(1., 0., 0., 0.), 
    0.577);

    solver_.solve(make_double3(-l, -2. * l, -2. * l), 
    make_double3(l, 2. * l, 2. * l), 
    make_double3(0., 0., -9.81), 
    1.e-4, 
    5., 
    50, 
    argc,
    argv);

    solver_.setFixedAngularVelocityToWall(0, make_double3(M_PI, 0., 0.));
    solver_.solve(make_double3(-l, -2. * l, -2. * l), 
    make_double3(l, 2. * l, 2. * l), 
    make_double3(0., 0., -9.81), 
    1.e-4, 
    5., 
    50, 
    argc,
    argv);

    solver_.setFixedAngularVelocityToWall(0, make_double3(0., 0., 0.));
    solver_.solve(make_double3(-l, -2. * l, -2. * l), 
    make_double3(l, 2. * l, 2. * l), 
    make_double3(0., 0., -9.81), 
    1.e-4, 
    5., 
    50, 
    argc,
    argv);
}