#include "kernel/LSSolver.h"
#include "kernel/helper/LevelSetObjectGenerator.h"
#include "kernel/helper/OBJLoader.h"
#include "globalDamping.cuh"

class solver:
    public LSSolver
{
public:
    solver(): LSSolver("tutorial3") {}

    double dampingCoefficient = 0.1;
    double3 globalForce = make_double3(0., 0., 100e3);
    double3 globalTorque = make_double3(0., 0., 0.);

    void addExternalForceTorque(const double time) override
    {
        launchAddGlobalConstantForceTorque(getLSParticle().force(),
        getLSParticle().torque(),
        globalForce * (time < 1. ? time : 1.),
        globalTorque * (time < 1. ? time : 1.),
        numParticle - 1, 
        0);

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
    const double beamLength = 4.;
    const size_t numParticle = 11;
    const double density = 7800.;

    std::vector<double3> vertexPosition; 
    std::vector<int3> triangleVertexIndex;
    OBJLoader::loadOBJMesh("bunny.obj", vertexPosition, triangleVertexIndex, argc, argv);
    for (auto& p:vertexPosition) p.y -= 0.1;
    LevelSetObject::TriangleMeshParticle TMP;
    TMP.setMesh(vertexPosition, triangleVertexIndex);
    TMP.buildGridByResolution();

    solver solver_;

    const double particleRadii = beamLength / double(numParticle - 1) / 2.;
    for (size_t i = 0; i < numParticle; i++)
    {
        const double3 particlePosition = make_double3(i * 2. * particleRadii, 0., 0.);

        solver_.addLSParticle(TMP.vertexPosition(), 
        TMP.gridInfo().gridNodeLevelSetFunctionValue, 
        TMP.gridInfo().gridOrigin, 
        TMP.gridInfo().gridNodeSize, 
        TMP.gridInfo().gridNodeSpacing, 
        particlePosition, 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        make_quaternion(1., 0., 0., 0.), 
        0., 
        0., 
        0.,
        1.,
        density * (i == 0 ? 0. : 1.),
        TMP.triangleVertexIndex());

        if (i > 0)
        {
            solver_.addBondedInteraction(i - 1, 
            i, 
            particlePosition - make_double3(particleRadii, 0., 0.), 
            make_double3(-1., 0., 0.), 
            particleRadii, 
            2. * particleRadii, 
            200.e9, 
            0.3);
        }
    }

    solver_.solve(make_double3(-particleRadii, -particleRadii, -particleRadii), 
    make_double3(beamLength + particleRadii, particleRadii, particleRadii), 
    make_double3(0., 0., 0.), 
    1.e-5, 
    1.e-3, 
    1, 
    argc,
    argv);
}