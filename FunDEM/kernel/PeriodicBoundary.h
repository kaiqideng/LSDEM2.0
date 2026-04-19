#pragma once
#include "LSParticle.h"
#include "SolidInteraction.h"
#include "CUDAKernelFunction/LSParticleContactDetectionKernel.cuh"
#include "CUDAKernelFunction/contactKernel.cuh"

struct Ghost
{
private:
    HostDeviceArray1D<double3> ghostVelocity_;
    HostDeviceArray1D<double3> ghostPosition_;
    HostDeviceArray1D<quaternion> ghostOrientation_;

    HostDeviceArray1D<int> ghostHashValue_;
    HostDeviceArray1D<int> ghostHashIndex_;

    HostDeviceArray1D<int> ghostGridHashStart_;
    HostDeviceArray1D<int> ghostGridHashEndt_;

    SolidInteraction ghostSolidInteraction_;
    
public:
    void initialize(const LSParticle& LS, const size_t maxGPUThread, cudaStream_t stream)
    {
        const size_t numParticle = LS.num();
        const size_t numGrid = LS.spatialGrid_.num();
        const size_t numBoundaryNode = LS.LSBoundaryNode_.num();

        ghostVelocity_.allocateDevice(numParticle, stream);
        ghostPosition_.allocateDevice(numParticle, stream);
        ghostOrientation_.allocateDevice(numParticle, stream);
        ghostHashValue_.allocateDevice(numParticle, stream);
        ghostHashIndex_.allocateDevice(numParticle, stream);

        ghostGridHashStart_.allocateDevice(numGrid, stream);
        ghostGridHashEndt_.allocateDevice(numGrid, stream);

        ghostSolidInteraction_.initialize(numBoundaryNode, maxGPUThread, stream);
    }

    void updateSpatialGrid(LSParticle& LS, cudaStream_t stream)
    {
        launchUpdateGhostSpatialGridHashStartEnd(ghostHashIndex_.d_ptr, 
        ghostHashValue_.d_ptr, 
        ghostPosition_.d_ptr, 
        ghostGridHashStart_.d_ptr, 
        ghostGridHashEndt_.d_ptr, 
        LS.spatialGrid_.minimumBoundary(), 
        LS.spatialGrid_.maximumBoundary(), 
        LS.spatialGrid_.inverseCellSize(), 
        LS.spatialGrid_.size3D(), 
        LS.spatialGrid_.num_device(), 
        LS.num_device(), 
        LS.gridDim(), 
        LS.blockDim(), 
        stream);
    }

    void buildInteraction(LSParticle& LS, const size_t maxGPUThread, cudaStream_t stream)
    {
        launchBuildLevelSetBoundaryNodeFixedParticleInteractions1st(ghostSolidInteraction_.masterNeighborCount(), 
        LS.LSBoundaryNode_.localPosition(), 
        LS.LSBoundaryNode_.particleID(), 
        LS.LSGridNode_.signedDistanceField(), 
        LS.position(), 
        LS.orientation(), 
        ghostPosition_.d_ptr, 
        ghostOrientation_.d_ptr, 
        LS.inverseGridNodeSpacing(), 
        LS.gridNodeLocalOrigin(), 
        LS.gridNodeSize(), 
        LS.gridNodePrefixSum(), 
        ghostHashIndex_.d_ptr,
        ghostGridHashStart_.d_ptr, 
        ghostGridHashEndt_.d_ptr, 
        LS.spatialGrid_.minimumBoundary(), 
        LS.spatialGrid_.inverseCellSize(), 
        LS.spatialGrid_.size3D(), 
        ghostSolidInteraction_.numMaster_device(), 
        ghostSolidInteraction_.masterGridDim(), 
        ghostSolidInteraction_.masterBlockDim(), 
        stream);

        ghostSolidInteraction_.updateNeighborPrefixSum(stream);
        ghostSolidInteraction_.updateNumPair(maxGPUThread, stream);
        ghostSolidInteraction_.savePreviousStep(stream);

        launchBuildLevelSetBoundaryNodeFixedParticleInteractions2nd(ghostSolidInteraction_.slidingSpring(), 
        ghostSolidInteraction_.contactPoint(), 
        ghostSolidInteraction_.contactNormal(), 
        ghostSolidInteraction_.contactOverlap(), 
        ghostSolidInteraction_.masterID(), 
        ghostSolidInteraction_.slaveID(), 
        ghostSolidInteraction_.previousSlidingSpring(), 
        ghostSolidInteraction_.previousSlaveID(), 
        LS.LSBoundaryNode_.localPosition(), 
        LS.LSBoundaryNode_.particleID(), 
        ghostSolidInteraction_.masterNeighborPrefixSum(), 
        ghostSolidInteraction_.previousMasterNeighborPrefixSum(), 
        LS.LSGridNode_.signedDistanceField(), 
        LS.position(), 
        LS.orientation(), 
        ghostPosition_.d_ptr, 
        ghostOrientation_.d_ptr, 
        LS.inverseGridNodeSpacing(), 
        LS.gridNodeLocalOrigin(), 
        LS.gridNodeSize(), 
        LS.gridNodePrefixSum(), 
        ghostHashIndex_.d_ptr,
        ghostGridHashStart_.d_ptr, 
        ghostGridHashEndt_.d_ptr, 
        LS.spatialGrid_.minimumBoundary(), 
        LS.spatialGrid_.inverseCellSize(), 
        LS.spatialGrid_.size3D(), 
        ghostSolidInteraction_.numMaster_device(), 
        ghostSolidInteraction_.masterGridDim(), 
        ghostSolidInteraction_.masterBlockDim(), 
        stream);
    }

    void addForceTorque(LSParticle& LS, const double timeStep, cudaStream_t stream)
    {
        launchAddGhostLevelSetParticleContactForceTorque(ghostSolidInteraction_.slidingSpring(), 
        ghostSolidInteraction_.normalElasticEnergy(),
        ghostSolidInteraction_.slidingElasticEnergy(),
        ghostSolidInteraction_.contactPoint(),
        ghostSolidInteraction_.contactNormal(), 
        ghostSolidInteraction_.contactOverlap(), 
        ghostSolidInteraction_.masterID(), 
        ghostSolidInteraction_.slaveID(), 
        LS.LSBoundaryNode_.particleID(), 
        LS.force(), 
        LS.torque(), 
        LS.position(),
        LS.velocity(),
        LS.angularVelocity(),
        LS.inverseMass(),
        LS.normalStiffness(),
        LS.shearStiffness(),
        LS.frictionCoefficient(),
        LS.restitutionCoefficient(), 
        ghostPosition_.d_ptr, 
        ghostVelocity_.d_ptr, 
        timeStep, 
        ghostSolidInteraction_.numPair_device(), 
        ghostSolidInteraction_.pairGridDim(), 
        ghostSolidInteraction_.pairBlockDim(), 
        stream);
    }

    double3* ghostVelocity() { return ghostVelocity_.d_ptr; }
    double3* ghostPosition() { return ghostPosition_.d_ptr; }
    quaternion* ghostOrientation() { return ghostOrientation_.d_ptr; }
};

struct PeriodicBoundaryXY2D
{
    PeriodicBoundaryXY2D() = default;
    ~PeriodicBoundaryXY2D() = default;

    PeriodicBoundaryXY2D(const PeriodicBoundaryXY2D&) = delete;
    PeriodicBoundaryXY2D& operator=(const PeriodicBoundaryXY2D&) = delete;

    PeriodicBoundaryXY2D(PeriodicBoundaryXY2D&&) noexcept = default;
    PeriodicBoundaryXY2D& operator=(PeriodicBoundaryXY2D&&) noexcept = default;

public:
    void turnXOn() { ativateXFlag_ = true; }

    void turnYOn() { ativateYFlag_ = true; }

    void turnXOff() { ativateXFlag_ = false; }

    void turnYOff() { ativateYFlag_ = false; }

    bool isActived()
    {
        return (ativateXFlag_ || ativateYFlag_);
    }

    void initialize(const LSParticle& LS, const size_t maxGPUThread, cudaStream_t stream)
    {
        if (ativateXFlag_) X_.initialize(LS, maxGPUThread, stream);
        if (ativateYFlag_) Y_.initialize(LS, maxGPUThread, stream);
        if (ativateXFlag_ && ativateYFlag_) XY_.initialize(LS, maxGPUThread, stream);
    }

    void updateGhostSpatialGrid(LSParticle& LS, cudaStream_t stream)
    {
        if (ativateXFlag_)
        {
            launchUpdatePositionOutOfBoundaryXD(LS.position(), 
            LS.spatialGrid_.minimumBoundary(), 
            LS.spatialGrid_.maximumBoundary(), 
            LS.num_device(), 
            LS.gridDim(), 
            LS.blockDim(), 
            stream);

            launchCalculateGhostPositionXD(X_.ghostPosition(), 
            LS.position(), 
            LS.spatialGrid_.minimumBoundary(), 
            LS.spatialGrid_.maximumBoundary(), 
            LS.spatialGrid_.inverseCellSize(), 
            LS.spatialGrid_.size3D(), 
            LS.num_device(), 
            LS.gridDim(), 
            LS.blockDim(), 
            stream);

            X_.updateSpatialGrid(LS, stream);

            cudaMemcpyAsync(X_.ghostOrientation(), LS.orientation(), LS.num_device() * sizeof(quaternion), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(X_.ghostVelocity(), LS.velocity(), LS.num_device() * sizeof(double3), cudaMemcpyDeviceToDevice, stream);
        }

        if (ativateYFlag_)
        {
            launchUpdatePositionOutOfBoundaryYD(LS.position(), 
            LS.spatialGrid_.minimumBoundary(), 
            LS.spatialGrid_.maximumBoundary(), 
            LS.num_device(), 
            LS.gridDim(), 
            LS.blockDim(), 
            stream);

            launchCalculateGhostPositionYD(Y_.ghostPosition(), 
            LS.position(), 
            LS.spatialGrid_.minimumBoundary(), 
            LS.spatialGrid_.maximumBoundary(), 
            LS.spatialGrid_.inverseCellSize(), 
            LS.spatialGrid_.size3D(), 
            LS.num_device(), 
            LS.gridDim(), 
            LS.blockDim(), 
            stream);

            Y_.updateSpatialGrid(LS, stream);

            cudaMemcpyAsync(Y_.ghostOrientation(), LS.orientation(), LS.num_device() * sizeof(quaternion), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(Y_.ghostVelocity(), LS.velocity(), LS.num_device() * sizeof(double3), cudaMemcpyDeviceToDevice, stream);
        }

        if (ativateXFlag_ && ativateYFlag_)
        {
            launchCalculateGhostPositionXYD(XY_.ghostPosition(), 
            LS.position(), 
            LS.spatialGrid_.minimumBoundary(), 
            LS.spatialGrid_.maximumBoundary(), 
            LS.spatialGrid_.inverseCellSize(), 
            LS.spatialGrid_.size3D(), 
            LS.num_device(), 
            LS.gridDim(), 
            LS.blockDim(), 
            stream);

            XY_.updateSpatialGrid(LS, stream);

            cudaMemcpyAsync(XY_.ghostOrientation(), LS.orientation(), LS.num_device() * sizeof(quaternion), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(XY_.ghostVelocity(), LS.velocity(), LS.num_device() * sizeof(double3), cudaMemcpyDeviceToDevice, stream);
        }
    }

    void buildGhostInteraction(LSParticle& LS, const size_t maxGPUThread, cudaStream_t stream)
    {
        if (ativateXFlag_) X_.buildInteraction(LS, maxGPUThread, stream);
        if (ativateYFlag_) Y_.buildInteraction(LS, maxGPUThread, stream);
        if (ativateXFlag_ && ativateYFlag_) XY_.buildInteraction(LS, maxGPUThread, stream);
    }

    void addGhostForceTorque(LSParticle& LS, const double timeStep, cudaStream_t stream)
    {
        if (ativateXFlag_) X_.addForceTorque(LS, timeStep, stream);
        if (ativateYFlag_) Y_.addForceTorque(LS, timeStep, stream);
        if (ativateXFlag_ && ativateYFlag_) XY_.addForceTorque(LS, timeStep, stream);
    }

private:
    Ghost X_;
    Ghost Y_;
    Ghost XY_;
    bool ativateXFlag_{false};
    bool ativateYFlag_{false};
};

struct PeriodicBoundarySector
{
    PeriodicBoundarySector() = default;
    ~PeriodicBoundarySector() = default;

    PeriodicBoundarySector(const PeriodicBoundarySector&) = delete;
    PeriodicBoundarySector& operator=(const PeriodicBoundarySector&) = delete;

    PeriodicBoundarySector(PeriodicBoundarySector&&) noexcept = default;
    PeriodicBoundarySector& operator=(PeriodicBoundarySector&&) noexcept = default;

public:
    void turnOn() { ativateFlag_ = true; }

    void turnOff() { ativateFlag_ = false; }

    bool isActived()
    {
        return ativateFlag_;
    }

    void initialize(const LSParticle& LS, const size_t maxGPUThread, cudaStream_t stream)
    {
        if (ativateFlag_)
        {
            R90_.initialize(LS, maxGPUThread, stream);
            R180_.initialize(LS, maxGPUThread, stream);
            R270_.initialize(LS, maxGPUThread, stream);
        }
    }

    void updateGhostSpatialGrid(LSParticle& LS, cudaStream_t stream)
    {
        if (ativateFlag_)
        {
            launchUpdateVelocityPositionOrientationOutOfSector(LS.velocity(), 
            LS.position(), 
            LS.orientation(), 
            LS.spatialGrid_.minimumBoundary(),
            LS.num_device(), 
            LS.gridDim(), 
            LS.blockDim(), 
            stream);

            launchCalculateSectorGhostVelocityPositionOrientation(R90_.ghostVelocity(), 
            R180_.ghostVelocity(), 
            R270_.ghostVelocity(), 
            R90_.ghostPosition(), 
            R180_.ghostPosition(),
            R270_.ghostPosition(),  
            R90_.ghostOrientation(), 
            R180_.ghostOrientation(), 
            R270_.ghostOrientation(), 
            LS.velocity(), 
            LS.position(), 
            LS.orientation(), 
            LS.spatialGrid_.minimumBoundary(), 
            LS.spatialGrid_.maximumBoundary(), 
            LS.spatialGrid_.inverseCellSize(), 
            LS.spatialGrid_.size3D(), 
            LS.num_device(), 
            LS.gridDim(), 
            LS.blockDim(), 
            stream);

            R90_.updateSpatialGrid(LS, stream);
            R180_.updateSpatialGrid(LS, stream);
            R270_.updateSpatialGrid(LS, stream);  
        }     
    }

    void buildGhostInteraction(LSParticle& LS, const size_t maxGPUThread, cudaStream_t stream)
    {
        if (ativateFlag_)
        {
            R90_.buildInteraction(LS, maxGPUThread, stream);
            R180_.buildInteraction(LS, maxGPUThread, stream);
            R270_.buildInteraction(LS, maxGPUThread, stream);
        }
    }

    void addGhostForceTorque(LSParticle& LS, const double timeStep, cudaStream_t stream)
    {
        if (ativateFlag_)
        {
            R90_.addForceTorque(LS, timeStep, stream);
            R180_.addForceTorque(LS, timeStep, stream);
            R270_.addForceTorque(LS, timeStep, stream);
        }
    }

private:
    Ghost R90_;
    Ghost R180_;
    Ghost R270_;
    bool ativateFlag_{false};
};