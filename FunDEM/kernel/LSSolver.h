#pragma once
#include "LSParticle.h"
#include "SolidInteraction.h"
#include "VBondedInteraction.h"
#include "CUDAKernelFunction/LSParticleContactDetectionKernel.cuh"
#include "CUDAKernelFunction/contactKernel.cuh"
#include "CUDAKernelFunction/particleIntegrationKernel.cuh"
#include "myMat.h"
#include <filesystem>

class LSSolver
{
public:
    LSSolver(const std::string dir, cudaStream_t stream = 0, const size_t maxGPUThread = 256, const int device = 0)
    {
        dir_ = dir;
        phase_ = 0;
        stream_ = stream;
        maxGPUThread_ = maxGPUThread;

        activateGPUDevice(device);
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
        restitutionCoefficient, 
        stream_);
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
        restitutionCoefficient, 
        stream_);
    }

    /**
     * @brief Add one fixed wall object represented by a level-set particle.
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
        fixedLSParticle_.add(objVertexLocalPosition, 
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
        restitutionCoefficient, 
        stream_);
    }

    void moveLSParticle(const size_t index, const double3 offSet) 
    { 
        LSParticle_.move(index, offSet, stream_); 
    }

    void moveWall(const size_t index, const double3 offSet) 
    { 
        fixedLSParticle_.move(index, offSet, stream_); 
    }

    void setFixedVelocityToWall(const size_t index, const double3 velocity)
    {
        fixedLSParticle_.setVelocity(index, velocity, stream_);
    }

    void setFixedAngularVelocityToWall(const size_t index, const double3 angularVelocity)
    {
        fixedLSParticle_.setAngularVelocity(index, angularVelocity, stream_);
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
        std::vector<double3> point = LSParticleInteraction_.contactPointHostCopy();
        std::vector<double3> normal = LSParticleInteraction_.contactNormalHostCopy();
        std::vector<int> masterBoundaryNodeID = LSParticleInteraction_.masterIDHostCopy();
        std::vector<int> slaveParticleID = LSParticleInteraction_.slaveIDHostCopy();
        std::vector<double3> position = LSParticle_.positionHostCopy();
        std::vector<quaternion> orientation = LSParticle_.orientationHostCopy();

        for(size_t k = 0; k < point.size(); k++)
        {
            const int i = LSParticle_.LSBoundaryNode_.particleIDHostRef()[masterBoundaryNodeID[k]];
            const int j = slaveParticleID[k];
            VBondedInteraction_.add(i, 
            j, 
            position[i], 
            position[j], 
            orientation[i], 
            orientation[j], 
            point[k], 
            normal[k], 
            radius, 
            initialLength, 
            YoungsModulus, 
            poissonRatio, 
            tensileStrength, 
            cohesion, 
            frictionCoefficient, 
            stream_);
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
        frictionCoefficient, 
        stream_);
    }

    void activateGPUDevice(const int device)
    {
        cudaError_t cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) 
        {
            std::cout << "cudaSetDevice( " << device 
            << " ) failed! Do you have a CUDA-capable GPU installed?" 
            << std::endl; 
            exit(1); 
        }
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
        phase_ += 1;
        std::string outputDir = dir_ + "_phase" + std::to_string(phase_);
        const char* argv0 = (argc > 0) ? argv[0] : nullptr;
        outputDir = resolveOutputDirFromBuild(outputDir, argv0);
        std::filesystem::create_directories(outputDir);

        std::cout << "[Solver] Uploading..." << std::endl;
        const double3 domainSize = maxDomain - minDomain;
        if (domainSize.x <= 0.0 || domainSize.y <= 0.0 || domainSize.z <= 0.0)
        {
            std::cerr << "[Solver] Invalid simulation domain size: ("
                    << domainSize.x << ", "
                    << domainSize.y << ", "
                    << domainSize.z << ")."
                    << std::endl;
            return;
        }

        if (timeStep <= 0.0)
        {
            std::cerr << "[Solver] Invalid timeStep: "
                    << timeStep << "."
                    << std::endl;
            return;
        }

        if (maximumTime < 0.0)
        {
            std::cerr << "[Solver] Invalid maximumTime: "
                    << maximumTime << "."
                    << std::endl;
            return;
        }

        removeFiles(outputDir);
        upload(minDomain, maxDomain);
        updateSpatialGrid();
        buildLSParticleInteraction();
        std::cout << "[Solver] Upload Completed" << std::endl;

        const size_t numStep = size_t(maximumTime / timeStep) + 1;
        size_t frameInterval = numStep;
        if (numFrame > 0) frameInterval = numStep / numFrame;
        if (frameInterval < 1) frameInterval = 1;
        size_t iStep = 0, iFrame = 0;
        double time = 0.;
        const double halfTimeStep = 0.5 * timeStep;
        output(outputDir, iFrame, iStep, time);
        while (iStep <= numStep)
        {
            simulateOneStep(time, gravity, halfTimeStep);
            iStep += 1;
            if (iStep % frameInterval == 0)
            {
                iFrame++;
                output(outputDir, iFrame, iStep, time);
            }
        }
        iFrame++;
        output(outputDir, iFrame, iStep, time);
        std::cout << "[Solver] Simulation Completed" << std::endl;
    }

    void copyFromHost(const LSSolver& other)
    {
        LSParticle_.copyFromHost(other.LSParticle_);
        fixedLSParticle_.copyFromHost(other.fixedLSParticle_);
        LSParticleInteraction_.copyFromHost(other.LSParticleInteraction_);
        fixedLSParticleInteraction_.copyFromHost(other.fixedLSParticleInteraction_);
        VBondedInteraction_.copyFromHost(other.VBondedInteraction_);

        dir_ = other.dir_;
        phase_ = other.phase_;
        stream_ = other.stream_;
        maxGPUThread_ = other.maxGPUThread_;
    }

protected:
    virtual void addExternalForceTorque(const double time) {}

    LSParticle& getLSParticle() { return LSParticle_; }

    LSParticle& getWall() { return fixedLSParticle_; }

    SolidInteraction& getLSParticleInteraction() { return LSParticleInteraction_; }

    SolidInteraction& getLSParticleWallInteraction() { return fixedLSParticleInteraction_; }

    VBondedInteraction& getLSParticleBondedInteraction() { return VBondedInteraction_; }

    void removeFiles(const std::string outputDir)
    {
        const std::string dir1 = outputDir + "/LSParticle";
        const std::string dir2 = outputDir + "/LSParticleInteraction";
        const std::string dir3 = outputDir + "/Wall";
        const std::string dir4 = outputDir + "/LSParticle-WallInteraction";
        MKDIR(outputDir.c_str());
        removeVtuFiles(dir1);
        removeVtuFiles(dir2);
        removeVtuFiles(dir3);
        removeVtuFiles(dir4);
    }

    void upload(const double3 minDomain, const double3 maxDomain)
    {
        LSParticle_.initialize(minDomain, maxDomain, maxGPUThread_, stream_);
        LSParticleInteraction_.initialize(LSParticle_.LSBoundaryNode_.num(), maxGPUThread_, stream_);
        fixedLSParticle_.initialize(minDomain, maxDomain, maxGPUThread_, stream_);
        fixedLSParticleInteraction_.initialize(LSParticle_.LSBoundaryNode_.num(), maxGPUThread_, stream_);
        VBondedInteraction_.initialize(maxGPUThread_, stream_);
    }

    void simulateOneStep(double& time, const double3 gravity, const double halfTimeStep)
    {
        updateSpatialGrid();
        buildLSParticleInteraction();
        calLSParticleContactForceTorque(halfTimeStep);
        time += halfTimeStep;
        addExternalForceTorque(time);
        integration1stHalf(gravity, halfTimeStep);
        buildLSParticleInteraction();
        calLSParticleContactForceTorque(halfTimeStep);
        time += halfTimeStep;
        addExternalForceTorque(time);
        integration2ndHalf(gravity, halfTimeStep);
    }

    void download()
    {
        LSParticle_.finalize(stream_);
        LSParticleInteraction_.finalize(stream_);
        fixedLSParticle_.finalize(stream_);
        fixedLSParticleInteraction_.finalize(stream_);
        VBondedInteraction_.finalize(stream_);
    }

    void output(const std::string &outputDir, const size_t iFrame, const size_t iStep, const double time)
    {
        std::cout << "[Solver] ------ Frame " << iFrame << " at Time " << time << " ------ " << std::endl;

        std::cout << "[Solver] Downloading..." << std::endl;
        download();
        std::cout << "[Solver] Download Completed" << std::endl;

        std::cout << "[Solver] Outputting... " << std::endl;
        const std::string dir1 = outputDir + "/LSParticle";
        const std::string dir2 = outputDir + "/LSParticleInteraction";
        const std::string dir3 = outputDir + "/Wall";
        const std::string dir4 = outputDir + "/LSParticle-WallInteraction";
        MKDIR(outputDir.c_str());

        LSParticle_.outputVTU(dir1, iFrame, iStep, time);
        LSParticleInteraction_.outputVTU(dir2, iFrame, iStep, time);
        VBondedInteraction_.outputVTU(dir2, iFrame, iStep, time);
        fixedLSParticle_.outputVTU(dir3, iFrame, iStep, time);
        fixedLSParticleInteraction_.outputVTU(dir4, iFrame, iStep, time);
        std::cout << "[Solver] Output Completed" << std::endl;
    }

private:
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

    inline std::string resolveOutputDirFromBuild(const std::string& outputDirName, const char* argv0)
    {
        const std::filesystem::path p(outputDirName);

        const std::filesystem::path resolvedPath = p.is_absolute() ? p : getBuildDirectoryFromExecutable(argv0) / p;

        return resolvedPath.string();
    }

    void updateSpatialGrid()
    {
        launchUpdateSpatialGridHashStartEnd(LSParticle_.position(), 
        LSParticle_.hashIndex(), 
        LSParticle_.hashValue(), 
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

        launchUpdateSpatialGridHashStartEnd(fixedLSParticle_.position(), 
        fixedLSParticle_.hashIndex(), 
        fixedLSParticle_.hashValue(), 
        fixedLSParticle_.spatialGrid_.hashStart(), 
        fixedLSParticle_.spatialGrid_.hashEnd(), 
        fixedLSParticle_.spatialGrid_.minimumBoundary(), 
        fixedLSParticle_.spatialGrid_.maximumBoundary(), 
        fixedLSParticle_.spatialGrid_.inverseCellSize(), 
        fixedLSParticle_.spatialGrid_.size3D(), 
        fixedLSParticle_.spatialGrid_.num_device(), 
        fixedLSParticle_.num_device(), 
        fixedLSParticle_.gridDim(), 
        fixedLSParticle_.blockDim(), 
        stream_);
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
        LSParticleInteraction_.updateNumPairFromNeighborPrefixSum(maxGPUThread_, stream_);
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

        launchBuildLevelSetBoundaryNodeFixedParticleInteractions1st(fixedLSParticleInteraction_.masterNeighborCount(), 
        LSParticle_.LSBoundaryNode_.localPosition(), 
        LSParticle_.LSBoundaryNode_.particleID(), 
        fixedLSParticle_.LSGridNode_.levelSetFunctionValue(), 
        LSParticle_.position(), 
        LSParticle_.orientation(), 
        fixedLSParticle_.position(), 
        fixedLSParticle_.orientation(), 
        fixedLSParticle_.inverseGridNodeSpacing(), 
        fixedLSParticle_.gridNodeLocalOrigin(), 
        fixedLSParticle_.gridNodeSize(), 
        fixedLSParticle_.gridNodePrefixSum(), 
        fixedLSParticle_.hashIndex(),
        fixedLSParticle_.spatialGrid_.hashStart(), 
        fixedLSParticle_.spatialGrid_.hashEnd(), 
        fixedLSParticle_.spatialGrid_.minimumBoundary(), 
        fixedLSParticle_.spatialGrid_.inverseCellSize(), 
        fixedLSParticle_.spatialGrid_.size3D(), 
        fixedLSParticleInteraction_.numMaster_device(), 
        fixedLSParticleInteraction_.masterGridDim(), 
        fixedLSParticleInteraction_.masterBlockDim(), 
        stream_);

        fixedLSParticleInteraction_.updateNeighborPrefixSum(stream_);
        fixedLSParticleInteraction_.updateNumPairFromNeighborPrefixSum(maxGPUThread_, stream_);
        fixedLSParticleInteraction_.savePreviousStep(stream_);

        launchBuildLevelSetBoundaryNodeFixedParticleInteractions2nd(fixedLSParticleInteraction_.slidingSpring(), 
        fixedLSParticleInteraction_.contactPoint(), 
        fixedLSParticleInteraction_.contactNormal(), 
        fixedLSParticleInteraction_.contactOverlap(), 
        fixedLSParticleInteraction_.masterID(), 
        fixedLSParticleInteraction_.slaveID(), 
        fixedLSParticleInteraction_.previousSlidingSpring(), 
        fixedLSParticleInteraction_.previousSlaveID(), 
        LSParticle_.LSBoundaryNode_.localPosition(), 
        LSParticle_.LSBoundaryNode_.particleID(), 
        fixedLSParticleInteraction_.masterNeighborPrefixSum(), 
        fixedLSParticleInteraction_.previousMasterNeighborPrefixSum(), 
        fixedLSParticle_.LSGridNode_.levelSetFunctionValue(), 
        LSParticle_.position(), 
        LSParticle_.orientation(), 
        fixedLSParticle_.position(), 
        fixedLSParticle_.orientation(), 
        fixedLSParticle_.inverseGridNodeSpacing(), 
        fixedLSParticle_.gridNodeLocalOrigin(), 
        fixedLSParticle_.gridNodeSize(), 
        fixedLSParticle_.gridNodePrefixSum(), 
        fixedLSParticle_.hashIndex(),
        fixedLSParticle_.spatialGrid_.hashStart(), 
        fixedLSParticle_.spatialGrid_.hashEnd(), 
        fixedLSParticle_.spatialGrid_.minimumBoundary(), 
        fixedLSParticle_.spatialGrid_.inverseCellSize(), 
        fixedLSParticle_.spatialGrid_.size3D(), 
        fixedLSParticleInteraction_.numMaster_device(), 
        fixedLSParticleInteraction_.masterGridDim(), 
        fixedLSParticleInteraction_.masterBlockDim(), 
        stream_);
    }

    void calLSParticleContactForceTorque(const double halfTimeStep)
    {
        cudaMemsetAsync(LSParticle_.force(), 0, LSParticle_.num_device() * sizeof(double3), stream_);
        cudaMemsetAsync(LSParticle_.torque(), 0, LSParticle_.num_device() * sizeof(double3), stream_);

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

        launchAddFixedLevelSetParticleContactForceTorque(fixedLSParticleInteraction_.slidingSpring(), 
        fixedLSParticleInteraction_.normalElasticEnergy(),
        fixedLSParticleInteraction_.slidingElasticEnergy(),
        fixedLSParticleInteraction_.contactPoint(),
        fixedLSParticleInteraction_.contactNormal(), 
        fixedLSParticleInteraction_.contactOverlap(),
        fixedLSParticleInteraction_.masterID(),
        fixedLSParticleInteraction_.slaveID(), 
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
        fixedLSParticle_.position(),
        fixedLSParticle_.velocity(),
        fixedLSParticle_.angularVelocity(),
        fixedLSParticle_.frictionCoefficient(),
        fixedLSParticle_.restitutionCoefficient(),
        halfTimeStep, 
        fixedLSParticleInteraction_.numPair_device(), 
        fixedLSParticleInteraction_.pairGridDim(), 
        fixedLSParticleInteraction_.pairBlockDim(), 
        stream_);

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

        launchParticlePositionOrientationIntegration(fixedLSParticle_.position(), 
        fixedLSParticle_.orientation(), 
        fixedLSParticle_.velocity(),
        fixedLSParticle_.angularVelocity(), 
        2. * halfTimeStep, 
        fixedLSParticle_.num_device(), 
        fixedLSParticle_.gridDim(), 
        fixedLSParticle_.blockDim(),
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
    LSParticle fixedLSParticle_;
    SolidInteraction LSParticleInteraction_;
    SolidInteraction fixedLSParticleInteraction_;
    VBondedInteraction VBondedInteraction_;

    std::string dir_;
    size_t phase_{0};
    cudaStream_t stream_;
    size_t maxGPUThread_;
};