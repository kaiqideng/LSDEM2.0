#pragma once
#include "LSParticle.h"
#include "SolidInteraction.h"
#include "VBondedInteraction.h"
#include "PeriodicBoundary.h"
#include "CUDAKernelFunction/LSParticleContactDetectionKernel.cuh"
#include "CUDAKernelFunction/contactKernel.cuh"
#include "CUDAKernelFunction/particleIntegrationKernel.cuh"
#include <filesystem>

class LSSolver
{
public:
    LSSolver(const std::string dir = "Problem", cudaStream_t stream = 0, const size_t maxGPUThread = 256, const int device = 0)
    {
        dir_ = dir;
        phase_ = 0;
        stream_ = stream;
        maxGPUThread_ = maxGPUThread;
        device_ = device;
    }

    ~LSSolver() = default;

    /**
     * @brief Add one free level-set particle.
     *
     * @param boundaryNodeLocalPosition Local boundary node positions of the particle.
     * @param gridNodeLevelSetFunctionValue Level-set values stored on the background grid nodes.
     * @param gridNodeLocalOrigin Local origin of the level-set grid.
     * @param gridNodeSize Grid resolution in x-, y-, and z-directions.
     * @param gridNodeSpacing Uniform grid spacing of the level-set grid.
     * @param position Initial world position of the particle.
     * @param velocity Initial translational velocity of the particle.
     * @param angularVelocity Initial angular velocity of the particle.
     * @param orientation Initial particle orientation.
     * @param normalStiffness Normal contact stiffness.
     * @param shearStiffness Shear contact stiffness.
     * @param frictionCoefficient Friction coefficient.
     * @param restitutionCoefficient Restitution coefficient.
     * @param density Particle density.
     * @param boundaryNodeConnectivity Optional boundary mesh connectivity.
     * Notes / assumptions:
     * It computes basic rigid-body properties from the grid:
     * - mass via a smoothed Heaviside integration of the level-set field
     * - center of mass (centroidLocalPosition) from the same Heaviside weights
     * - inertia tensor around the center of mass, then inverse inertia.
     *
     * Here the Heaviside assumes "inside is negative" (phi < 0 means inside),
     * because smoothHeaviside(phi/gridNodeSpacing, ...) returns ~1 for negative phi.
     * If your convention is opposite, you must flip the sign passed into smoothHeaviside.
     * This function modifies host arrays. If previous data has been uploaded (upload_==true),
     * it first downloads device -> host to keep host-side buffers consistent.
     */
    void addLSParticle(const std::vector<double3>& boundaryNodeLocalPosition,
    const std::vector<double>& gridNodeLevelSetFunctionValue,
    const double3 gridNodeLocalOrigin,
    const int3 gridNodeSize,
    const double gridNodeSpacing,
    const double3 position,
    const double3 velocity,
    const double3 angularVelocity,
    const quaternion orientation,
    const double normalStiffness,
    const double shearStiffness,
    const double frictionCoefficient,
    const double restitutionCoefficient, 
    const double density,
    const std::vector<int3>& boundaryNodeConnectivity = {})
    {
        double mass = 0.;
        auto smoothHeaviside = [&](const double phi_dimensionless, const double smoothParameter) -> double
        {
            if (smoothParameter <= 0.0) return (phi_dimensionless > 0.0) ? 0.0 : 1.0;

            if (phi_dimensionless < -smoothParameter) return 1.0;
            if (phi_dimensionless > smoothParameter) return 0.0;

            const double x = -phi_dimensionless / smoothParameter;
            return 0.5 * (1.0 + x + std::sin(pi() * x) / pi());
        };
        const double m_gridNode = density * gridNodeSpacing * gridNodeSpacing * gridNodeSpacing;
        for (int x = 0; x < gridNodeSize.x; x++)
        {
            for (int y = 0; y < gridNodeSize.y; y++)
            {
                for (int z = 0; z < gridNodeSize.z; z++)
                {
                    const int index = linearIndex3D(make_int3(x, y, z), gridNodeSize);
                    const double H = smoothHeaviside(gridNodeLevelSetFunctionValue[index] / gridNodeSpacing, 1.5);
                    mass += H;
                }
            }
        }
        mass *= m_gridNode;

        double inverseMass = 0.;
        double3 centroidLocalPosition = make_double3(0., 0., 0.);
        symMatrix I = make_symMatrix(0., 0., 0., 0., 0., 0.);
        symMatrix inverseInertiaTensor = make_symMatrix(0., 0., 0., 0., 0., 0.);
        if (mass > 0.)
        {
            inverseMass = 1. / mass;
            for (int x = 0; x < gridNodeSize.x; x++)
            {
                for (int y = 0; y < gridNodeSize.y; y++)
                {
                    for (int z = 0; z < gridNodeSize.z; z++)
                    {
                        const int index = linearIndex3D(make_int3(x, y, z), gridNodeSize);
                        const double H = smoothHeaviside(gridNodeLevelSetFunctionValue[index] / gridNodeSpacing, 1.5);
                        centroidLocalPosition.x += H * (gridNodeLocalOrigin.x + double(x) * gridNodeSpacing);
                        centroidLocalPosition.y += H * (gridNodeLocalOrigin.y + double(y) * gridNodeSpacing);
                        centroidLocalPosition.z += H * (gridNodeLocalOrigin.z + double(z) * gridNodeSpacing);
                    }
                }
            }
            centroidLocalPosition *= m_gridNode / mass;

            for (int x = 0; x < gridNodeSize.x; x++)
            {
                for (int y = 0; y < gridNodeSize.y; y++)
                {
                    for (int z = 0; z < gridNodeSize.z; z++)
                    {
                        const int index = linearIndex3D(make_int3(x, y, z), gridNodeSize);
                        const double H = smoothHeaviside(gridNodeLevelSetFunctionValue[index] / gridNodeSpacing, 1.5);
                        double3 r = gridNodeLocalOrigin + gridNodeSpacing * make_double3(double(x), double(y), double(z)) - centroidLocalPosition;
                        I.xx += H * (r.y * r.y + r.z * r.z) * m_gridNode;
                        I.yy += H * (r.x * r.x + r.z * r.z) * m_gridNode;
                        I.zz += H * (r.y * r.y + r.x * r.x) * m_gridNode;
                        I.xy -= H * r.x * r.y * m_gridNode;
                        I.xz -= H * r.x * r.z * m_gridNode;
                        I.yz -= H * r.y * r.z * m_gridNode;
                    }
                }
            }
            inverseInertiaTensor = inverse(I);
        }

        const quaternion orientation_new = normalize(orientation);
        const double3 position_new = position + rotateVectorByQuaternion(orientation_new, centroidLocalPosition);
        std::vector<double3> boundaryNodeLocalPosition_new;
        for (const auto& p : boundaryNodeLocalPosition)
        {
            boundaryNodeLocalPosition_new.push_back(p - centroidLocalPosition);
        }
        const double3 gridNodeLocalOrigin_new = gridNodeLocalOrigin - centroidLocalPosition;

        LSParticle_.add(boundaryNodeLocalPosition_new, 
        boundaryNodeConnectivity, 
        gridNodeLevelSetFunctionValue, 
        gridNodeLocalOrigin_new, 
        gridNodeSize, 
        gridNodeSpacing, 
        position_new, 
        velocity, 
        angularVelocity, 
        orientation_new, 
        inverseMass, 
        inverseInertiaTensor, 
        normalStiffness,
        shearStiffness,
        frictionCoefficient, 
        restitutionCoefficient);
    }

    void addLSParticle(const std::vector<double3>& boundaryNodeLocalPosition,
    const std::vector<double>& gridNodeLevelSetFunctionValue,
    const double3 gridNodeLocalOrigin,
    const int3 gridNodeSize,
    const double gridNodeSpacing,
    const double3 position,
    const double3 velocity,
    const double3 angularVelocity,
    const quaternion orientation,
    const double mass,
    const symMatrix inertiaTensor,
    const double normalStiffness,
    const double shearStiffness,
    const double frictionCoefficient,
    const double restitutionCoefficient,
    const std::vector<int3>& boundaryNodeConnectivity = {})
    {
        LSParticle_.add(boundaryNodeLocalPosition, 
        boundaryNodeConnectivity, 
        gridNodeLevelSetFunctionValue, 
        gridNodeLocalOrigin, 
        gridNodeSize, 
        gridNodeSpacing, 
        position, 
        velocity, 
        angularVelocity, 
        orientation, 
        mass > 0. ? 1. / mass : 0., 
        mass > 0. ? inverse(inertiaTensor) : make_symMatrix(0., 0., 0., 0., 0., 0.), 
        normalStiffness, 
        shearStiffness, 
        frictionCoefficient, 
        restitutionCoefficient);
    }

    /**
     * @brief Add one wall object represented by a level-set particle.
     *
     * @param objVertexLocalPosition Local mesh vertex positions of the wall object.
     * @param objTriangleVertexID Triangle connectivity of the wall mesh.
     * @param gridNodeLevelSetFunctionValue Level-set values stored on the background grid nodes.
     * @param gridNodeLocalOrigin Local origin of the level-set grid.
     * @param gridNodeSize Grid resolution in x-, y-, and z-directions.
     * @param gridNodeSpacing Uniform grid spacing of the level-set grid.
     * @param position Initial world position of the wall object.
     * @param orientation Initial wall orientation.
     * @param frictionCoefficient Friction coefficient.
     * @param restitutionCoefficient Restitution coefficient.
     */
    void addWall(const std::vector<double3>& objVertexLocalPosition,
    const std::vector<int3>& objTriangleVertexID, 
    const std::vector<double>& gridNodeLevelSetFunctionValue,
    const double3 gridNodeLocalOrigin,
    const int3 gridNodeSize,
    const double gridNodeSpacing,
    const double3 position,
    const quaternion orientation,
    const double frictionCoefficient, 
    const double restitutionCoefficient = 1.)
    {
        Wall_.add(objVertexLocalPosition, 
        objTriangleVertexID, 
        gridNodeLevelSetFunctionValue, 
        gridNodeLocalOrigin, 
        gridNodeSize, 
        gridNodeSpacing, 
        position, 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        orientation, 
        0., 
        make_symMatrix(0., 0., 0., 0., 0., 0.), 
        0., 
        0., 
        frictionCoefficient, 
        restitutionCoefficient);
    }

    void moveLSParticle(const size_t index, const double3 offSet) 
    { 
        LSParticle_.move(index, offSet); 
    }

    void moveWall(const size_t index, const double3 offSet) 
    { 
        Wall_.move(index, offSet); 
    }

    void setFixedVelocityToWall(const size_t index, const double3 velocity)
    {
        Wall_.setVelocity(index, velocity);
    }

    void setFixedAngularVelocityToWall(const size_t index, const double3 angularVelocity)
    {
        Wall_.setAngularVelocity(index, angularVelocity);
    }

    /**
     * @brief Create bonded interactions between free level-set particles.
     *
     * @param YoungsModulus Bond Young's modulus.
     * @param poissonRatio Bond Poisson ratio.
     * @param radius Bond creation radius.
     * @param initialLength Bond creation length.
     * @param tensileStrength Bond tensile strength.
     * @param cohesion Bond cohesion.
     * @param frictionCoefficient Bond friction coefficient.
     */
    void addBondedInteraction(const double radius, 
    const double initialLength, 
    const double YoungsModulus, 
    const double poissonRatio, 
    const double tensileStrength = 0., 
    const double cohesion= 0., 
    const double frictionCoefficient= 0.)
    {
        const std::vector<int> masterBoundaryNodeID = LSParticleInteraction_.masterIDHostRef();
        const std::vector<int> slaveParticleID = LSParticleInteraction_.slaveIDHostRef();
        const std::vector<double3> point = LSParticleInteraction_.contactPointHostRef();
        const std::vector<double3> normal = LSParticleInteraction_.contactNormalHostRef();

        for(size_t k = 0; k < point.size(); k++)
        {
            const int i = LSParticle_.LSBoundaryNode_.particleIDHostRef()[masterBoundaryNodeID[k]];
            const int j = slaveParticleID[k];
            VBondedInteraction_.add(i, 
            j, 
            LSParticle_.positionHostRef()[i], 
            LSParticle_.positionHostRef()[j], 
            LSParticle_.orientationHostRef()[i], 
            LSParticle_.orientationHostRef()[j], 
            point[k], 
            normal[k], 
            radius, 
            initialLength, 
            YoungsModulus, 
            poissonRatio, 
            tensileStrength, 
            cohesion, 
            frictionCoefficient);
        }
    }

    void addBondedInteraction(const int masterObjectID, 
    const int slaveObjectID, 
    const double3 bondPoint, 
    const double3 bondNormal, 
    const double radius, 
    const double initialLength, 
    const double YoungsModulus, 
    const double poissonRatio, 
    const double tensileStrength = 0., 
    const double cohesion= 0., 
    const double frictionCoefficient= 0.)
    {
        if (masterObjectID >= LSParticle_.num() || slaveObjectID >= LSParticle_.num()) return;
        VBondedInteraction_.add(masterObjectID, 
        slaveObjectID, 
        LSParticle_.positionHostRef()[masterObjectID], 
        LSParticle_.positionHostRef()[slaveObjectID], 
        LSParticle_.orientationHostRef()[masterObjectID], 
        LSParticle_.orientationHostRef()[slaveObjectID], 
        bondPoint, 
        bondNormal, 
        radius, 
        initialLength, 
        YoungsModulus, 
        poissonRatio, 
        tensileStrength, 
        cohesion, 
        frictionCoefficient);
    }

    void addPeriodicBoundaryXY2D()
    {
        PeriodicBoundaryXY2D_.turnOn();
        PeriodicBoundarySector_.turnOff();
    }

    void addPeriodicBoundarySector()
    {
        PeriodicBoundaryXY2D_.turnOff();
        PeriodicBoundarySector_.turnOn();
    }

    /**
     * @brief Run the solver time integration loop.
     *
     * @param minDomain Minimum simulation domain corner.
     * @param maxDomain Maximum simulation domain corner.
     * @param gravity Gravity acceleration.
     * @param timeStep Time step size.
     * @param maximumTime Maximum physical simulation time.
     * @param numFrame Requested number of output frames.
     * @param argc Argument count from main().
     * @param argv Argument array from main().
     */
    void solve(const double3 minDomain, 
    const double3 maxDomain, 
    const double3 gravity, 
    const double timeStep, 
    const double maximumTime, 
    const size_t numFrame, 
    const int argc,
    char** argv)
    {
        if (!checkSolverInput(minDomain, maxDomain, timeStep, maximumTime)) return;
        if (!activateGPUDevice()) return;
        phase_ += 1;
        updateDir(argc, argv);
        removeFiles();
        upload(minDomain, maxDomain);
        
        const size_t numStep = size_t(maximumTime / timeStep) + 1;
        size_t frameInterval = numStep;
        if (numFrame > 0) frameInterval = numStep / numFrame;
        if (frameInterval < 1) frameInterval = 1;
        
        bool addBondFlag = VBondedInteraction_.numPair_device() > 0;
        bool addPeriodicBoundaryFlag = (PeriodicBoundaryXY2D_.isActived() || PeriodicBoundarySector_.isActived());
        if ((!addBondFlag) && (!addPeriodicBoundaryFlag)) compute(gravity, timeStep, numStep, frameInterval);
        else if (addBondFlag && (!addPeriodicBoundaryFlag)) compute_addBond(gravity, timeStep, numStep, frameInterval);
        else if ((!addBondFlag) && addPeriodicBoundaryFlag) compute_addPeriodicBoundary(gravity, timeStep, numStep, frameInterval);
        else compute_addBondAndPeriodicBoundary(gravity, timeStep, numStep, frameInterval);
    }

protected:
    virtual void addExternalForceTorque(const double time) {}

    LSParticle& getLSParticle() { return LSParticle_; }

    LSParticle& getWall() { return Wall_; }

    SolidInteraction& getLSParticleInteraction() { return LSParticleInteraction_; }

    SolidInteraction& getLSParticleWallInteraction() { return WallLSParticleInteraction_; }

    VBondedInteraction& getLSParticleBondedInteraction() { return VBondedInteraction_; }

    void compute(const double3 gravity, const double timeStep, const size_t numStep, const size_t frameInterval, 
    size_t iStep = 0, size_t iFrame = 0, double time = 0.)
    {
        updateLSParticleSpatialGrid();
        buildLSParticleInteraction();
        output(iFrame, iStep, time);
        const double halfTimeStep = 0.5 * timeStep;
        while (iStep <= numStep)
        {
            time += halfTimeStep;
            clearLSParticleForceTorque();
            addLSParticleContactForceTorque(halfTimeStep);
            addExternalForceTorque(time);
            integration1stHalf(gravity, halfTimeStep);

            updateLSParticleSpatialGrid();
            buildLSParticleInteraction();

            time += halfTimeStep;
            clearLSParticleForceTorque();
            addLSParticleContactForceTorque(halfTimeStep);
            addExternalForceTorque(time);
            integration2ndHalf(gravity, halfTimeStep);

            iStep += 1;
            if (iStep % frameInterval == 0)
            {
                iFrame++;
                output(iFrame, iStep, time);
            }
        }
        iFrame++;
        output(iFrame, iStep, time);
        std::cout << "[Solver] Computation Completed" << std::endl;
    }

    void compute_addBond(const double3 gravity, const double timeStep, const size_t numStep, const size_t frameInterval, 
    size_t iStep = 0, size_t iFrame = 0, double time = 0.)
    {
        updateLSParticleSpatialGrid();
        buildLSParticleInteraction();
        output(iFrame, iStep, time);
        const double halfTimeStep = 0.5 * timeStep;
        while (iStep <= numStep)
        {
            time += halfTimeStep;
            clearLSParticleForceTorque();
            addLSParticleContactForceTorque(halfTimeStep);
            addBondedForceTorque();
            addExternalForceTorque(time);
            integration1stHalf(gravity, halfTimeStep);

            updateLSParticleSpatialGrid();
            buildLSParticleInteraction();

            time += halfTimeStep;
            clearLSParticleForceTorque();
            addLSParticleContactForceTorque(halfTimeStep);
            addBondedForceTorque();
            addExternalForceTorque(time);
            integration2ndHalf(gravity, halfTimeStep);

            iStep += 1;
            if (iStep % frameInterval == 0)
            {
                iFrame++;
                output(iFrame, iStep, time);
            }
        }
        iFrame++;
        output(iFrame, iStep, time);
        std::cout << "[Solver] Computation Completed" << std::endl;
    }

    void compute_addPeriodicBoundary(const double3 gravity, const double timeStep, const size_t numStep, const size_t frameInterval, 
    size_t iStep = 0, size_t iFrame = 0, double time = 0.)
    {
        updateLSParticleSpatialGrid();
        buildLSParticleInteraction();
        updateGhostSpatialGrid();
        buildGhostInteraction();
        output(iFrame, iStep, time);
        const double halfTimeStep = 0.5 * timeStep;
        while (iStep <= numStep)
        {
            time += halfTimeStep;
            clearLSParticleForceTorque();
            addLSParticleContactForceTorque(halfTimeStep);
            addGhostForceTorque(halfTimeStep);
            addExternalForceTorque(time);
            integration1stHalf(gravity, halfTimeStep);

            updateLSParticleSpatialGrid();
            buildLSParticleInteraction();
            updateGhostSpatialGrid();
            buildGhostInteraction();

            time += halfTimeStep;
            clearLSParticleForceTorque();
            addLSParticleContactForceTorque(halfTimeStep);
            addGhostForceTorque(halfTimeStep);
            addExternalForceTorque(time);
            integration2ndHalf(gravity, halfTimeStep);

            iStep += 1;
            if (iStep % frameInterval == 0)
            {
                iFrame++;
                output(iFrame, iStep, time);
            }
        }
        iFrame++;
        output(iFrame, iStep, time);
        std::cout << "[Solver] Computation Completed" << std::endl;
    }

    void compute_addBondAndPeriodicBoundary(const double3 gravity, const double timeStep, const size_t numStep, const size_t frameInterval, 
    size_t iStep = 0, size_t iFrame = 0, double time = 0.)
    {
        updateLSParticleSpatialGrid();
        buildLSParticleInteraction();
        updateGhostSpatialGrid();
        buildGhostInteraction();
        output(iFrame, iStep, time);
        const double halfTimeStep = 0.5 * timeStep;
        while (iStep <= numStep)
        {
            time += halfTimeStep;
            clearLSParticleForceTorque();
            addLSParticleContactForceTorque(halfTimeStep);
            addBondedForceTorque();
            addGhostForceTorque(halfTimeStep);
            addExternalForceTorque(time);
            integration1stHalf(gravity, halfTimeStep);

            updateLSParticleSpatialGrid();
            buildLSParticleInteraction();
            updateGhostSpatialGrid();
            buildGhostInteraction();

            time += halfTimeStep;
            clearLSParticleForceTorque();
            addLSParticleContactForceTorque(halfTimeStep);
            addBondedForceTorque();
            addGhostForceTorque(halfTimeStep);
            addExternalForceTorque(time);
            integration2ndHalf(gravity, halfTimeStep);

            iStep += 1;
            if (iStep % frameInterval == 0)
            {
                iFrame++;
                output(iFrame, iStep, time);
            }
        }
        iFrame++;
        output(iFrame, iStep, time);
        std::cout << "[Solver] Computation Completed" << std::endl;
    }

    void download()
    {
        std::cout << "[Solver] Downloading..." << std::endl;
        LSParticle_.finalize(stream_);
        LSParticleInteraction_.finalize(stream_);
        Wall_.finalize(stream_);
        WallLSParticleInteraction_.finalize(stream_);
        VBondedInteraction_.finalize(stream_);
        std::cout << "[Solver] Download Completed" << std::endl;
    }

    void output(const size_t iFrame, const size_t iStep, const double time)
    {
        std::cout << "[Solver] ------ Frame " << iFrame << " at Time " << time << " ------ " << std::endl;

        download();
        
        std::cout << "[Solver] Outputting... " << std::endl;
        const std::string dir = dir_ + "_phase" + std::to_string(phase_);
        const std::string dir1 = dir + "/LSParticle";
        const std::string dir2 = dir + "/LSParticleInteraction";
        const std::string dir3 = dir + "/Wall";
        const std::string dir4 = dir + "/LSParticle-WallInteraction";
        MKDIR(dir.c_str());
        LSParticle_.outputVTU(dir1, iFrame, iStep, time);
        LSParticleInteraction_.outputVTU(dir2, iFrame, iStep, time);
        VBondedInteraction_.outputVTU(dir2, iFrame, iStep, time);
        Wall_.outputVTU(dir3, iFrame, iStep, time);
        WallLSParticleInteraction_.outputVTU(dir4, iFrame, iStep, time);
        std::cout << "[Solver] Output Completed" << std::endl;
    }

private:
    bool activateGPUDevice()
    {
        cudaError_t cudaStatus = cudaSetDevice(device_);
        if (cudaStatus != cudaSuccess) 
        {
            std::cout << "cudaSetDevice( " << device_ 
            << " ) failed! Do you have a CUDA-capable GPU installed?" 
            << std::endl; 
            exit(1);
            return false;
        }
        return true;
    }

    inline std::filesystem::path getBuildDirectoryFromExecutable(const char* argv0)
    {
        if (argv0 == nullptr) return std::filesystem::current_path();
        std::filesystem::path exePath(argv0);
        if (exePath.is_relative())
        {
            exePath = std::filesystem::absolute(exePath);
        }
        exePath = std::filesystem::weakly_canonical(exePath);
        std::filesystem::path outputDir = exePath.parent_path();
        while (!outputDir.empty() && outputDir.filename() != "build")
        {
            const std::filesystem::path parent = outputDir.parent_path();
            if (parent == outputDir) break;
            outputDir = parent;
        }
        if (!outputDir.empty() && outputDir.filename() == "build")
        {
            return outputDir;
        }
        return exePath.parent_path();
    }

    void updateDir(const int argc, char** argv)
    {
        const char* argv0 = (argc > 0) ? argv[0] : nullptr;
        const std::filesystem::path p(dir_);
        const std::filesystem::path resolvedPath = p.is_absolute() ? p : getBuildDirectoryFromExecutable(argv0) / p;
        dir_ = resolvedPath.string();
    }

    bool checkSolverInput(const double3 minDomain, const double3 maxDomain, const double timeStep, const double maximumTime)
    {
        const double3 domainSize = maxDomain - minDomain;
        if (domainSize.x <= 0.0 || domainSize.y <= 0.0 || domainSize.z <= 0.0)
        {
            std::cerr << "[Solver] Invalid simulation domain size: ("
                    << domainSize.x << ", "
                    << domainSize.y << ", "
                    << domainSize.z << ")."
                    << std::endl;
            return false;
        }

        if (timeStep <= 0.0)
        {
            std::cerr << "[Solver] Invalid timeStep: "
                    << timeStep << "."
                    << std::endl;
            return false;
        }

        if (maximumTime < 0.0)
        {
            std::cerr << "[Solver] Invalid maximumTime: "
                    << maximumTime << "."
                    << std::endl;
            return false;
        }

        return true;
    }

    void removeFiles()
    {
        const std::string dir = dir_ + "_phase" + std::to_string(phase_);
        const std::string dir1 = dir + "/LSParticle";
        const std::string dir2 = dir + "/LSParticleInteraction";
        const std::string dir3 = dir + "/Wall";
        const std::string dir4 = dir + "/LSParticle-WallInteraction";
        MKDIR(dir.c_str());
        removeVtuFiles(dir1);
        removeVtuFiles(dir2);
        removeVtuFiles(dir3);
        removeVtuFiles(dir4);
    }

    void upload(const double3 minDomain, const double3 maxDomain)
    {
        std::cout << "[Solver] Uploading..." << std::endl;
        LSParticle_.initialize(minDomain, maxDomain, maxGPUThread_, stream_);
        LSParticleInteraction_.initialize(LSParticle_.LSBoundaryNode_.num(), maxGPUThread_, stream_);
        Wall_.initialize(minDomain, maxDomain, maxGPUThread_, stream_);
        WallLSParticleInteraction_.initialize(LSParticle_.LSBoundaryNode_.num(), maxGPUThread_, stream_);
        VBondedInteraction_.initialize(maxGPUThread_, stream_);
        PeriodicBoundaryXY2D_.initialize(LSParticle_, maxGPUThread_, stream_);
        PeriodicBoundarySector_.initialize(LSParticle_, maxGPUThread_, stream_);
        std::cout << "[Solver] Upload Completed" << std::endl;
    }

    void updateLSParticleSpatialGrid()
    {
        launchUpdateSpatialGridHashStartEnd(LSParticle_.hashIndex(), 
        LSParticle_.hashValue(), 
        LSParticle_.position(), 
        LSParticle_.spatialGrid_.hashStart(), 
        LSParticle_.spatialGrid_.hashEnd(), 
        LSParticle_.spatialGrid_.minimumBoundary(), 
        LSParticle_.spatialGrid_.maximumBoundary(), 
        LSParticle_.spatialGrid_.inverseCellSize(), 
        LSParticle_.spatialGrid_.size3D(), 
        LSParticle_.spatialGrid_.num_device(), 
        LSParticle_.num_device(), 
        LSParticle_.gridDim(), 
        LSParticle_.blockDim(), 
        stream_);

        launchUpdateSpatialGridHashStartEnd(Wall_.hashIndex(), 
        Wall_.hashValue(), 
        Wall_.position(), 
        Wall_.spatialGrid_.hashStart(), 
        Wall_.spatialGrid_.hashEnd(), 
        Wall_.spatialGrid_.minimumBoundary(), 
        Wall_.spatialGrid_.maximumBoundary(), 
        Wall_.spatialGrid_.inverseCellSize(), 
        Wall_.spatialGrid_.size3D(), 
        Wall_.spatialGrid_.num_device(), 
        Wall_.num_device(), 
        Wall_.gridDim(), 
        Wall_.blockDim(), 
        stream_);
    }

    void updateGhostSpatialGrid()
    {
        PeriodicBoundaryXY2D_.updateGhostSpatialGrid(LSParticle_, stream_);
        PeriodicBoundarySector_.updateGhostSpatialGrid(LSParticle_, stream_);
    }

    void buildLSParticleInteraction()
    {
        launchBuildLevelSetBoundaryNodeInteractions1st(LSParticleInteraction_.masterNeighborCount(), 
        LSParticle_.LSBoundaryNode_.localPosition(), 
        LSParticle_.LSBoundaryNode_.particleID(), 
        LSParticle_.LSGridNode_.levelSetFunctionValue(), 
        LSParticle_.position(), 
        LSParticle_.orientation(), 
        LSParticle_.radius(),
        LSParticle_.inverseGridNodeSpacing(), 
        LSParticle_.gridNodeLocalOrigin(), 
        LSParticle_.gridNodeSize(), 
        LSParticle_.gridNodePrefixSum(), 
        LSParticle_.hashIndex(),
        LSParticle_.spatialGrid_.hashStart(), 
        LSParticle_.spatialGrid_.hashEnd(), 
        LSParticle_.spatialGrid_.minimumBoundary(), 
        LSParticle_.spatialGrid_.inverseCellSize(), 
        LSParticle_.spatialGrid_.size3D(), 
        LSParticleInteraction_.numMaster_device(), 
        LSParticleInteraction_.masterGridDim(), 
        LSParticleInteraction_.masterBlockDim(), 
        stream_);

        LSParticleInteraction_.updateNeighborPrefixSum(stream_);
        LSParticleInteraction_.updateNumPair(maxGPUThread_, stream_);
        LSParticleInteraction_.savePreviousStep(stream_);

        launchBuildLevelSetBoundaryNodeInteractions2nd(LSParticleInteraction_.slidingSpring(), 
        LSParticleInteraction_.contactPoint(), 
        LSParticleInteraction_.contactNormal(), 
        LSParticleInteraction_.contactOverlap(), 
        LSParticleInteraction_.masterID(), 
        LSParticleInteraction_.slaveID(), 
        LSParticleInteraction_.previousSlidingSpring(), 
        LSParticleInteraction_.previousSlaveID(), 
        LSParticle_.LSBoundaryNode_.localPosition(), 
        LSParticle_.LSBoundaryNode_.particleID(), 
        LSParticleInteraction_.masterNeighborPrefixSum(), 
        LSParticleInteraction_.previousMasterNeighborPrefixSum(), 
        LSParticle_.LSGridNode_.levelSetFunctionValue(), 
        LSParticle_.position(), 
        LSParticle_.orientation(), 
        LSParticle_.radius(), 
        LSParticle_.inverseGridNodeSpacing(), 
        LSParticle_.gridNodeLocalOrigin(), 
        LSParticle_.gridNodeSize(), 
        LSParticle_.gridNodePrefixSum(), 
        LSParticle_.hashIndex(),
        LSParticle_.spatialGrid_.hashStart(), 
        LSParticle_.spatialGrid_.hashEnd(), 
        LSParticle_.spatialGrid_.minimumBoundary(), 
        LSParticle_.spatialGrid_.inverseCellSize(), 
        LSParticle_.spatialGrid_.size3D(), 
        LSParticleInteraction_.numMaster_device(), 
        LSParticleInteraction_.masterGridDim(), 
        LSParticleInteraction_.masterBlockDim(), 
        stream_);

        launchBuildLevelSetBoundaryNodeFixedParticleInteractions1st(WallLSParticleInteraction_.masterNeighborCount(), 
        LSParticle_.LSBoundaryNode_.localPosition(), 
        LSParticle_.LSBoundaryNode_.particleID(), 
        Wall_.LSGridNode_.levelSetFunctionValue(), 
        LSParticle_.position(), 
        LSParticle_.orientation(), 
        Wall_.position(), 
        Wall_.orientation(), 
        Wall_.inverseGridNodeSpacing(), 
        Wall_.gridNodeLocalOrigin(), 
        Wall_.gridNodeSize(), 
        Wall_.gridNodePrefixSum(), 
        Wall_.hashIndex(),
        Wall_.spatialGrid_.hashStart(), 
        Wall_.spatialGrid_.hashEnd(), 
        Wall_.spatialGrid_.minimumBoundary(), 
        Wall_.spatialGrid_.inverseCellSize(), 
        Wall_.spatialGrid_.size3D(), 
        WallLSParticleInteraction_.numMaster_device(), 
        WallLSParticleInteraction_.masterGridDim(), 
        WallLSParticleInteraction_.masterBlockDim(), 
        stream_);

        WallLSParticleInteraction_.updateNeighborPrefixSum(stream_);
        WallLSParticleInteraction_.updateNumPair(maxGPUThread_, stream_);
        WallLSParticleInteraction_.savePreviousStep(stream_);

        launchBuildLevelSetBoundaryNodeFixedParticleInteractions2nd(WallLSParticleInteraction_.slidingSpring(), 
        WallLSParticleInteraction_.contactPoint(), 
        WallLSParticleInteraction_.contactNormal(), 
        WallLSParticleInteraction_.contactOverlap(), 
        WallLSParticleInteraction_.masterID(), 
        WallLSParticleInteraction_.slaveID(), 
        WallLSParticleInteraction_.previousSlidingSpring(), 
        WallLSParticleInteraction_.previousSlaveID(), 
        LSParticle_.LSBoundaryNode_.localPosition(), 
        LSParticle_.LSBoundaryNode_.particleID(), 
        WallLSParticleInteraction_.masterNeighborPrefixSum(), 
        WallLSParticleInteraction_.previousMasterNeighborPrefixSum(), 
        Wall_.LSGridNode_.levelSetFunctionValue(), 
        LSParticle_.position(), 
        LSParticle_.orientation(), 
        Wall_.position(), 
        Wall_.orientation(), 
        Wall_.inverseGridNodeSpacing(), 
        Wall_.gridNodeLocalOrigin(), 
        Wall_.gridNodeSize(), 
        Wall_.gridNodePrefixSum(), 
        Wall_.hashIndex(),
        Wall_.spatialGrid_.hashStart(), 
        Wall_.spatialGrid_.hashEnd(), 
        Wall_.spatialGrid_.minimumBoundary(), 
        Wall_.spatialGrid_.inverseCellSize(), 
        Wall_.spatialGrid_.size3D(), 
        WallLSParticleInteraction_.numMaster_device(), 
        WallLSParticleInteraction_.masterGridDim(), 
        WallLSParticleInteraction_.masterBlockDim(), 
        stream_);
    }

    void buildGhostInteraction()
    {
        PeriodicBoundaryXY2D_.buildGhostInteraction(LSParticle_, maxGPUThread_, stream_);
        PeriodicBoundarySector_.buildGhostInteraction(LSParticle_, maxGPUThread_, stream_);
    }

    void clearLSParticleForceTorque()
    {
        cudaMemsetAsync(LSParticle_.force(), 0, LSParticle_.num_device() * sizeof(double3), stream_);
        cudaMemsetAsync(LSParticle_.torque(), 0, LSParticle_.num_device() * sizeof(double3), stream_);
    }

    void addLSParticleContactForceTorque(const double halfTimeStep)
    {
        launchAddLevelSetParticleContactForceTorque(LSParticleInteraction_.slidingSpring(), 
        LSParticleInteraction_.normalElasticEnergy(),
        LSParticleInteraction_.slidingElasticEnergy(),
        LSParticleInteraction_.contactPoint(),
        LSParticleInteraction_.contactNormal(), 
        LSParticleInteraction_.contactOverlap(), 
        LSParticleInteraction_.masterID(), 
        LSParticleInteraction_.slaveID(), 
        LSParticle_.LSBoundaryNode_.localPosition(),
        LSParticle_.LSBoundaryNode_.particleID(), 
        LSParticle_.force(), 
        LSParticle_.torque(), 
        LSParticle_.position(),
        LSParticle_.velocity(),
        LSParticle_.angularVelocity(),
        LSParticle_.inverseMass(),
        LSParticle_.normalStiffness(),
        LSParticle_.shearStiffness(),
        LSParticle_.frictionCoefficient(),
        LSParticle_.restitutionCoefficient(), 
        halfTimeStep, 
        LSParticleInteraction_.numPair_device(), 
        LSParticleInteraction_.pairGridDim(), 
        LSParticleInteraction_.pairBlockDim(), 
        stream_);

        launchAddFixedLevelSetParticleContactForceTorque(WallLSParticleInteraction_.slidingSpring(), 
        WallLSParticleInteraction_.normalElasticEnergy(),
        WallLSParticleInteraction_.slidingElasticEnergy(),
        WallLSParticleInteraction_.contactPoint(),
        WallLSParticleInteraction_.contactNormal(), 
        WallLSParticleInteraction_.contactOverlap(),
        WallLSParticleInteraction_.masterID(),
        WallLSParticleInteraction_.slaveID(), 
        LSParticle_.LSBoundaryNode_.localPosition(),
        LSParticle_.LSBoundaryNode_.particleID(), 
        LSParticle_.force(), 
        LSParticle_.torque(), 
        LSParticle_.position(),
        LSParticle_.velocity(),
        LSParticle_.angularVelocity(),
        LSParticle_.inverseMass(),
        LSParticle_.normalStiffness(),
        LSParticle_.shearStiffness(),
        LSParticle_.frictionCoefficient(),
        LSParticle_.restitutionCoefficient(),
        Wall_.position(),
        Wall_.velocity(),
        Wall_.angularVelocity(),
        Wall_.frictionCoefficient(),
        Wall_.restitutionCoefficient(),
        halfTimeStep, 
        WallLSParticleInteraction_.numPair_device(), 
        WallLSParticleInteraction_.pairGridDim(), 
        WallLSParticleInteraction_.pairBlockDim(), 
        stream_);
    }

    void addBondedForceTorque()
    {
        launchAddBondedForceTorque(VBondedInteraction_.centerPoint(), 
        VBondedInteraction_.maxNormalStress(), 
        VBondedInteraction_.maxShearStress(), 
        VBondedInteraction_.Un(), 
        VBondedInteraction_.Us(), 
        VBondedInteraction_.Ub(), 
        VBondedInteraction_.Ut(), 
        VBondedInteraction_.activated(), 
        VBondedInteraction_.B1(), 
        VBondedInteraction_.B2(), 
        VBondedInteraction_.B3(), 
        VBondedInteraction_.B4(), 
        VBondedInteraction_.radius(), 
        VBondedInteraction_.initialLength(), 
        VBondedInteraction_.tensileStrength(), 
        VBondedInteraction_.cohesion(), 
        VBondedInteraction_.frictionCoefficient(), 
        VBondedInteraction_.masterVBondPointLocalVectorN1(), 
        VBondedInteraction_.masterVBondPointLocalVectorN2(), 
        VBondedInteraction_.masterVBondPointLocalVectorN3(), 
        VBondedInteraction_.masterVBondPointLocalPosition(), 
        VBondedInteraction_.slaveVBondPointLocalVectorN1(), 
        VBondedInteraction_.slaveVBondPointLocalVectorN2(), 
        VBondedInteraction_.slaveVBondPointLocalVectorN3(), 
        VBondedInteraction_.slaveVBondPointLocalPosition(), 
        VBondedInteraction_.masterObjectID(), 
        VBondedInteraction_.slaveObjectID(), 
        LSParticle_.force(), 
        LSParticle_.torque(), 
        LSParticle_.position(), 
        LSParticle_.orientation(), 
        VBondedInteraction_.numPair_device(), 
        VBondedInteraction_.gridDim(), 
        VBondedInteraction_.blockDim(), 
        stream_);
    }

    void addGhostForceTorque(const double halfTimeStep)
    {
        PeriodicBoundaryXY2D_.addGhostForceTorque(LSParticle_, halfTimeStep, stream_); 
        PeriodicBoundarySector_.addGhostForceTorque(LSParticle_, halfTimeStep, stream_); 
    }

    void integration1stHalf(const double3 gravity, const double halfTimeStep)
    {
        launchParticleVelocityAngularVelocityIntegration(LSParticle_.velocity(),
        LSParticle_.angularVelocity(),
        LSParticle_.force(), 
        LSParticle_.torque(), 
        LSParticle_.inverseMass(), 
        LSParticle_.orientation(), 
        LSParticle_.inverseInertiaTensor(), 
        gravity, 
        halfTimeStep, 
        LSParticle_.num_device(), 
        LSParticle_.gridDim(), 
        LSParticle_.blockDim(),
        stream_);

        launchParticlePositionOrientationIntegration(LSParticle_.position(), 
        LSParticle_.orientation(), 
        LSParticle_.velocity(),
        LSParticle_.angularVelocity(), 
        2. * halfTimeStep, 
        LSParticle_.num_device(), 
        LSParticle_.gridDim(), 
        LSParticle_.blockDim(),
        stream_);

        launchParticlePositionOrientationIntegration(Wall_.position(), 
        Wall_.orientation(), 
        Wall_.velocity(),
        Wall_.angularVelocity(), 
        2. * halfTimeStep, 
        Wall_.num_device(), 
        Wall_.gridDim(), 
        Wall_.blockDim(),
        stream_);
    }

    void integration2ndHalf(const double3 gravity, const double halfTimeStep)
    {
        launchParticleVelocityAngularVelocityIntegration(LSParticle_.velocity(),
        LSParticle_.angularVelocity(),
        LSParticle_.force(), 
        LSParticle_.torque(), 
        LSParticle_.inverseMass(), 
        LSParticle_.orientation(), 
        LSParticle_.inverseInertiaTensor(), 
        gravity, 
        halfTimeStep, 
        LSParticle_.num_device(), 
        LSParticle_.gridDim(), 
        LSParticle_.blockDim(),
        stream_);
    }

    LSParticle LSParticle_;
    LSParticle Wall_;
    SolidInteraction LSParticleInteraction_;
    SolidInteraction WallLSParticleInteraction_;
    VBondedInteraction VBondedInteraction_;
    PeriodicBoundaryXY2D PeriodicBoundaryXY2D_;
    PeriodicBoundarySector PeriodicBoundarySector_;

    std::string dir_;
    size_t phase_;
    cudaStream_t stream_;
    size_t maxGPUThread_;
    size_t device_;
};