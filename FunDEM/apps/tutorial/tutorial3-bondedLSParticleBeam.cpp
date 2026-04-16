#include "kernel/LSSolver.h"
#include "kernel/helper/LevelSetObjectGenerator.h"
#include "globalDamping.cuh"

class solver:
    public LSSolver
{
public:
    solver(): LSSolver("tutorial3") {}

    size_t numBond{0};
    double dampingCoefficient = 0.1;
    double3 globalForce = make_double3(0., 0., 1.e7);
    double3 globalTorque = make_double3(0., 0., 0.);

    void addExternalForceTorque(const double time) override
    {
        launchAddGlobalConstantForceTorque(getLSParticle().force(),
        getLSParticle().torque(),
        globalForce * (time < 1. ? time : 1.),
        globalTorque * (time < 1. ? time : 1.),
        numBond, 
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

    solver solver_;
    solver_.numBond = numParticle - 1;
    const double particleRadii = beamLength / double(numParticle - 1) / 2.;
    const double mass = 4. / 3. * pi() * particleRadii * particleRadii * particleRadii * density;
    const double inertia = 0.4 * mass * particleRadii * particleRadii;
    const symMatrix inertiaTensor = make_symMatrix(inertia, inertia, inertia, 0., 0., 0.);

    LevelSetObject::Sphere S;
    S.setParams(particleRadii, 1000);
    S.buildGridByResolution();

    for (size_t i = 0; i < numParticle; i++)
    {
        const double3 particlePosition = make_double3(i * 2. * particleRadii, 0., 0.);

        solver_.addLSParticle(S.vertexPosition(), 
        S.gridInfo().gridNodeLevelSetFunctionValue, 
        S.gridInfo().gridOrigin, 
        S.gridInfo().gridNodeSize, 
        S.gridInfo().gridNodeSpacing, 
        particlePosition, 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        make_quaternion(1., 0., 0., 0.), 
        mass * (i > 0), 
        inertiaTensor, 
        0., 
        0., 
        0.,
        1.,
        S.triangleVertexIndex());

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
    5., 
    50, 
    argc,
    argv);
}