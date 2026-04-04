#include "kernel/LSSolver.h"
#include "kernel/helper/LevelSetObjectGenerator.h"
#include "kernel/helper/SpherePackingGenerator.h"
#include "kernel/helper/OBJLoader.h"
#include "globalDamping.cuh"

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

int main(const int argc, char** argv)
{
    const double l = 1.5;
    const double3 boxMin = make_double3(0., 0., 0.);
    const double3 boxMax = make_double3(l, l, 3. * l);

    const double density = 1000.;
    std::vector<double3> vertexPosition; 
    std::vector<int3> triangleVertexIndex;
    OBJLoader::loadOBJMesh("bunny.obj", vertexPosition, triangleVertexIndex, argc, argv);
    for (auto& p:vertexPosition) p.y -= 0.1;
    LevelSetObject::TriangleMeshParticle TMP;
    TMP.setMesh(vertexPosition, triangleVertexIndex);
    TMP.buildGridByResolution();
    TMP.outputGridVTU("build/bunny");

    SpherePacking::Pack Pack_ = SpherePacking::buildRegularInBox(boxMin, 
    boxMax,
    0.5 * length(TMP.boundingBoxMax() - TMP.boundingBoxMin()));

    solver solver_;

    for (size_t i = 0; i < Pack_.centers.size(); i++)
    {
        quaternion q = randomQuaternionUniform_deterministic();
        
        solver_.addLSParticle(TMP.vertexPosition(), 
        TMP.gridInfo().gridNodeLevelSetFunctionValue, 
        TMP.gridInfo().gridOrigin, 
        TMP.gridInfo().gridNodeSize, 
        TMP.gridInfo().gridNodeSpacing, 
        Pack_.centers[i], 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        q, 
        6.e5, 
        1.8e5, 
        0.577,
        density,
        TMP.triangleVertexIndex());
    }

    LevelSetObject::BoxWall BW;
    BW.setParams(boxMax.x - boxMin.x, boxMax.y - boxMin.y, boxMax.z - boxMin.z);
    BW.buildGridByResolution();
    solver_.addWall(BW.vertexPosition(), 
    BW.triangleVertexIndex(),
    BW.gridInfo().gridNodeLevelSetFunctionValue, 
    BW.gridInfo().gridOrigin, 
    BW.gridInfo().gridNodeSize, 
    BW.gridInfo().gridNodeSpacing, 
    0.5 * (boxMin + boxMax), 
    make_quaternion(1., 0., 0., 0.), 
    6.e5, 
    1.8e5, 
    0.577);

    solver_.solve(boxMin, 
    boxMax, 
    make_double3(0., 0., -9.81), 
    1.e-4, 
    5., 
    50, 
    "tutorial1",
    argc,
    argv);
}