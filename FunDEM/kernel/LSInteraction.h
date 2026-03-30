#pragma once
#include "LSParticle.h"
#include "CUDAKernelFunction/myUtility/buildHashStartEnd.h"


struct LSParticleInteraction
{
public:
    LSParticleInteraction() = default;
    ~LSParticleInteraction() = default;

    LSParticleInteraction(const LSParticleInteraction&) = delete;
    LSParticleInteraction& operator=(const LSParticleInteraction&) = delete;

    LSParticleInteraction(LSParticleInteraction&&) noexcept = default;
    LSParticleInteraction& operator=(LSParticleInteraction&&) noexcept = default;

    void initialize(const LSParticle& LSP, const size_t maxGPUThread, cudaStream_t stream)
    {
        const size_t numBoundaryNode = LSP.LSBoundaryNode_.num();
        if (numBoundaryNode < boundaryNodeNeighborPrefixSum_.hostSize()) return;
        boundaryNodeNeighborCount_.allocateDevice(numBoundaryNode, stream);
        boundaryNodeNeighborPrefixSum_.allocateDevice(numBoundaryNode, stream);
        boundaryNodeNeighborPrefixSum0_.allocateDevice(numBoundaryNode, stream);
        if (maxGPUThread > 0) boundaryNodeBlockDim_ = maxGPUThread;
        if (numBoundaryNode < maxGPUThread) boundaryNodeBlockDim_ = numBoundaryNode;
        boundaryNodeGridDim_ = (numBoundaryNode + boundaryNodeBlockDim_ - 1) / boundaryNodeBlockDim_;
    }

    void save(const size_t maxGPUThread, cudaStream_t stream)
    {
        cudaMemcpyAsync(boundaryNodeNeighborPrefixSum0_.d_ptr, boundaryNodeNeighborPrefixSum_.d_ptr, 
        boundaryNodeNeighborPrefixSum_.deviceSize() * sizeof(int), cudaMemcpyDeviceToDevice, stream);

        buildPrefixSum(boundaryNodeNeighborPrefixSum_.d_ptr,
        boundaryNodeNeighborCount_.d_ptr,
        boundaryNodeNeighborCount_.deviceSize(),
        stream);

        cudaStreamSynchronize(stream);
        cudaMemcpyAsync(&numPair_, boundaryNodeNeighborPrefixSum_.d_ptr + boundaryNodeNeighborPrefixSum_.deviceSize() - 1, 
        sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (numPair_ == 0)
        {
            pairGridDim_ = 1;
            pairBlockDim_ = 1;
            return;
        }
        else
        {
            pairBlockDim_ = static_cast<size_t>(numPair_);
            if (maxGPUThread > 0 && maxGPUThread < static_cast<size_t>(numPair_)) pairBlockDim_ = maxGPUThread;
            pairGridDim_ = (static_cast<size_t>(numPair_) + pairBlockDim_ - 1) / pairBlockDim_;
        }

        if (static_cast<size_t>(numPair_) > slidingSpring0_.deviceSize())
        {
            slidingSpring0_.allocateDevice(static_cast<size_t>(numPair_), stream);
            slaveParticleID0_.allocateDevice(static_cast<size_t>(numPair_), stream);

            cudaMemcpyAsync(slidingSpring0_.d_ptr, slidingSpring_.d_ptr, 
            slidingSpring_.deviceSize() * sizeof(double3), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(slaveParticleID0_.d_ptr, slaveParticleID_.d_ptr, 
            slaveParticleID_.deviceSize() * sizeof(int), cudaMemcpyDeviceToDevice, stream);
 
            contactPoint_.allocateDevice(static_cast<size_t>(numPair_), stream);
            contactNormal_.allocateDevice(static_cast<size_t>(numPair_), stream);
            contactOverlap_.allocateDevice(static_cast<size_t>(numPair_), stream);
            slidingSpring_.allocateDevice(static_cast<size_t>(numPair_), stream);
            masterBoundaryNodeID_.allocateDevice(static_cast<size_t>(numPair_), stream);
            slaveParticleID_.allocateDevice(static_cast<size_t>(numPair_), stream);
        }
        else 
        {
            cudaMemcpyAsync(slidingSpring0_.d_ptr, slidingSpring_.d_ptr, 
            slidingSpring_.deviceSize() * sizeof(double3), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(slaveParticleID0_.d_ptr, slaveParticleID_.d_ptr, 
            slaveParticleID_.deviceSize() * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        }
    }

    void outputVTU(const std::vector<int>& masterBoundaryNodeParticleID,
    const std::vector<double>& masterParticleNormalStiffness,
    const std::vector<double>& masterParticleShearStiffness,
    const std::vector<double>& slaveParticleNormalStiffness,
    const std::vector<double>& slaveParticleShearStiffness,
    const std::string& dir,
    const size_t iFrame,
    const size_t iStep,
    const double time)
    {
        MKDIR(dir.c_str());

        std::ostringstream fname;
        fname << dir << "/LSInteraction_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";

        const size_t N = numPair_;

        std::vector<int> pairMasterBoundaryID = masterBoundaryNodeID_.getHostCopy();
        std::vector<int> pairSlaveParticleID = slaveParticleID_.getHostCopy();
        std::vector<double3> p = contactPoint_.getHostCopy();
        std::vector<double> o = contactOverlap_.getHostCopy();
        std::vector<double3> n = contactNormal_.getHostCopy();
        std::vector<double3> s = slidingSpring_.getHostCopy();

        pairMasterBoundaryID.resize(N);
        pairSlaveParticleID.resize(N);
        p.resize(N);
        o.resize(N);
        n.resize(N);
        s.resize(N);

        std::vector<float> points;
        std::vector<float> normal;
        std::vector<float> slidingForce;

        std::vector<float> normalElasticEnergy;
        std::vector<float> shearElasticEnergy;
        std::vector<float> totalElasticEnergy;

        std::vector<int32_t> conn;
        std::vector<int32_t> offs;
        std::vector<uint8_t> types;

        points.resize(N * 3);
        normal.resize(N * 3);
        slidingForce.resize(N * 3);

        normalElasticEnergy.resize(N);
        shearElasticEnergy.resize(N);
        totalElasticEnergy.resize(N);

        conn.resize(N);
        offs.resize(N);
        types.resize(N);

        for (size_t i = 0; i < N; ++i)
        {
            const int masterBoundaryNodeID = pairMasterBoundaryID[i];
            const int masterParticleID = masterBoundaryNodeParticleID[masterBoundaryNodeID];
            const int slaveParticleID = pairSlaveParticleID[i];

            const double kn_i = masterParticleNormalStiffness[masterParticleID];
            const double ks_i = masterParticleShearStiffness[masterParticleID];
            const double kn_j = slaveParticleNormalStiffness[slaveParticleID];
            const double ks_j = slaveParticleShearStiffness[slaveParticleID];

            const double knEff = (kn_i > 0.0 && kn_j > 0.0) ? kn_i * kn_j / (kn_i + kn_j) : 0.0;
            const double ksEff = (ks_i > 0.0 && ks_j > 0.0) ? ks_i * ks_j / (ks_i + ks_j) : 0.0;

            const double overlapValue = o[i];
            const double spring2 = dot(s[i], s[i]);

            const double En = 0.5 * knEff * overlapValue * overlapValue;
            const double Es = 0.5 * ksEff * spring2;
            const double Etotal = En + Es;

            const double3 fs = -ksEff * s[i];

            points[3 * i + 0] = static_cast<float>(p[i].x);
            points[3 * i + 1] = static_cast<float>(p[i].y);
            points[3 * i + 2] = static_cast<float>(p[i].z);

            normal[3 * i + 0] = static_cast<float>(n[i].x);
            normal[3 * i + 1] = static_cast<float>(n[i].y);
            normal[3 * i + 2] = static_cast<float>(n[i].z);

            slidingForce[3 * i + 0] = static_cast<float>(fs.x);
            slidingForce[3 * i + 1] = static_cast<float>(fs.y);
            slidingForce[3 * i + 2] = static_cast<float>(fs.z);

            normalElasticEnergy[i] = static_cast<float>(En);
            shearElasticEnergy[i] = static_cast<float>(Es);
            totalElasticEnergy[i] = static_cast<float>(Etotal);

            conn[i] = static_cast<int32_t>(i);
            offs[i] = static_cast<int32_t>(i + 1);
            types[i] = static_cast<uint8_t>(1);
        }

        auto pad8 = [](size_t n) -> size_t
        {
            const size_t a = 8;
            return (n + (a - 1)) & ~(a - 1);
        };

        auto blockBytes = [&](size_t n) -> size_t
        {
            return sizeof(uint64_t) + pad8(n);
        };

        size_t off_points = 0;
        size_t off_conn = off_points + blockBytes(points.size() * sizeof(float));
        size_t off_offs = off_conn + blockBytes(conn.size() * sizeof(int32_t));
        size_t off_types = off_offs + blockBytes(offs.size() * sizeof(int32_t));

        size_t off_normal = off_types + blockBytes(types.size() * sizeof(uint8_t));
        size_t off_slidingForce = off_normal + blockBytes(normal.size() * sizeof(float));

        size_t off_normalElasticEnergy = off_slidingForce + blockBytes(slidingForce.size() * sizeof(float));
        size_t off_shearElasticEnergy = off_normalElasticEnergy + blockBytes(normalElasticEnergy.size() * sizeof(float));
        size_t off_totalElasticEnergy = off_shearElasticEnergy + blockBytes(shearElasticEnergy.size() * sizeof(float));

        std::ofstream out(fname.str(), std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open " + fname.str());

        out
            << "<?xml version=\"1.0\"?>\n"
            << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
            << "  <UnstructuredGrid>\n"
            << "    <FieldData>\n"
            << "      <DataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\"> " << static_cast<float>(time) << " </DataArray>\n"
            << "      <DataArray type=\"Int32\" Name=\"STEP\" NumberOfTuples=\"1\" format=\"ascii\"> " << static_cast<int32_t>(iStep) << " </DataArray>\n"
            << "    </FieldData>\n"
            << "    <Piece NumberOfPoints=\"" << N << "\" NumberOfCells=\"" << N << "\">\n"
            << "      <Points>\n"
            << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_points << "\"/>\n"
            << "      </Points>\n"
            << "      <Cells>\n"
            << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\"" << off_conn << "\"/>\n"
            << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\"" << off_offs << "\"/>\n"
            << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\"" << off_types << "\"/>\n"
            << "      </Cells>\n"
            << "      <PointData>\n"
            << "        <DataArray type=\"Float32\" Name=\"contactNormal\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_normal << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"slidingForce\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_slidingForce << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"normalElasticEnergy\" format=\"appended\" offset=\"" << off_normalElasticEnergy << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"shearElasticEnergy\" format=\"appended\" offset=\"" << off_shearElasticEnergy << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"totalElasticEnergy\" format=\"appended\" offset=\"" << off_totalElasticEnergy << "\"/>\n"
            << "      </PointData>\n"
            << "    </Piece>\n"
            << "  </UnstructuredGrid>\n"
            << "  <AppendedData encoding=\"raw\">\n"
            << "    _";

        auto writeBlock = [&](const void* data, size_t nbytes)
        {
            const uint64_t sz = static_cast<uint64_t>(nbytes);
            out.write(reinterpret_cast<const char*>(&sz), sizeof(uint64_t));
            if (nbytes) out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(nbytes));

            const size_t pad = pad8(nbytes) - nbytes;
            if (pad)
            {
                static const char zeros[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                out.write(zeros, static_cast<std::streamsize>(pad));
            }
        };

        writeBlock(points.data(), points.size() * sizeof(float));
        writeBlock(conn.data(), conn.size() * sizeof(int32_t));
        writeBlock(offs.data(), offs.size() * sizeof(int32_t));
        writeBlock(types.data(), types.size() * sizeof(uint8_t));

        writeBlock(normal.data(), normal.size() * sizeof(float));
        writeBlock(slidingForce.data(), slidingForce.size() * sizeof(float));

        writeBlock(normalElasticEnergy.data(), normalElasticEnergy.size() * sizeof(float));
        writeBlock(shearElasticEnergy.data(), shearElasticEnergy.size() * sizeof(float));
        writeBlock(totalElasticEnergy.data(), totalElasticEnergy.size() * sizeof(float));

        out << "\n  </AppendedData>\n</VTKFile>\n";
    }

    void finalize(cudaStream_t stream)
    {
        copyDeviceToHost(stream);
    }

    void copyFromHost(const LSParticleInteraction& other)
    {
        contactPoint_.setHost(other.contactPoint_.hostRef());
        contactNormal_.setHost(other.contactNormal_.hostRef());
        contactOverlap_.setHost(other.contactOverlap_.hostRef());
        slidingSpring_.setHost(other.slidingSpring_.hostRef());
        masterBoundaryNodeID_.setHost(other.masterBoundaryNodeID_.hostRef());
        slaveParticleID_.setHost(other.slaveParticleID_.hostRef());

        numPair_ = other.numPair_;

        boundaryNodeNeighborPrefixSum_.setHost(other.boundaryNodeNeighborPrefixSum_.hostRef());
    }

    double3* contactPoint() { return contactPoint_.d_ptr; }
    double3* contactNormal() { return contactNormal_.d_ptr; }
    double* contactOverlap() { return contactOverlap_.d_ptr; }
    double3* slidingSpring() { return slidingSpring_.d_ptr; }
    int* masterBoundaryNodeID() { return masterBoundaryNodeID_.d_ptr; }
    int* slaveParticleID() { return slaveParticleID_.d_ptr; }
    double3* previousSlidingSpring() { return slidingSpring0_.d_ptr; }
    int* previousSlaveParticleID() { return slaveParticleID0_.d_ptr; }

    size_t numPair_device() const { return static_cast<size_t>(numPair_); }
    size_t pairGridDim() const { return pairGridDim_; }
    size_t pairBlockDim() const { return pairBlockDim_; }

    int* boundaryNodeNeighborCount() { return boundaryNodeNeighborCount_.d_ptr; }
    int* boundaryNodeNeighborPrefixSum() { return boundaryNodeNeighborPrefixSum_.d_ptr; }
    int* previousBundaryNodeNeighborPrefixSum() { return boundaryNodeNeighborPrefixSum0_.d_ptr; }
    size_t numBoundaryNode_device() const { return boundaryNodeNeighborPrefixSum_.deviceSize(); }
    size_t boundaryNodeGridDim() const { return boundaryNodeGridDim_; }
    size_t boundaryNodeBlockDim() const { return boundaryNodeBlockDim_; }

    std::vector<double3> contactPointHostCopy()
    {
        std::vector<double3> v = contactPoint_.getHostCopy();
        v.resize(numPair_device());
        return v;
    }

    std::vector<double3> contactNormalHostCopy()
    {
        std::vector<double3> v = contactNormal_.getHostCopy();
        v.resize(numPair_device());
        return v;
    }

    std::vector<int> masterBoundaryNodeIDHostCopy() 
    {
        std::vector<int> v = masterBoundaryNodeID_.getHostCopy();
        v.resize(numPair_device());
        return v;
    }

    std::vector<int> slaveParticleIDHostCopy() 
    {
        std::vector<int> v = slaveParticleID_.getHostCopy();
        v.resize(numPair_device());
        return v;
    }

private:
    void copyHostToDevice(cudaStream_t stream)
    {
        contactPoint_.copyHostToDevice(stream);
        contactNormal_.copyHostToDevice(stream);
        contactOverlap_.copyHostToDevice(stream);
        slidingSpring_.copyHostToDevice(stream);
        masterBoundaryNodeID_.copyHostToDevice(stream);
        slaveParticleID_.copyHostToDevice(stream);

        boundaryNodeNeighborPrefixSum_.copyHostToDevice(stream);
    }

    void copyDeviceToHost(cudaStream_t stream)
    {
        contactPoint_.copyDeviceToHost(stream);
        contactNormal_.copyDeviceToHost(stream);
        contactOverlap_.copyDeviceToHost(stream);
        slidingSpring_.copyDeviceToHost(stream);
        masterBoundaryNodeID_.copyDeviceToHost(stream);
        slaveParticleID_.copyDeviceToHost(stream);

        boundaryNodeNeighborPrefixSum_.copyDeviceToHost(stream);
    }

    HostDeviceArray1D<double3> contactPoint_;
    HostDeviceArray1D<double3> contactNormal_;
    HostDeviceArray1D<double> contactOverlap_;
    HostDeviceArray1D<double3> slidingSpring_;
    HostDeviceArray1D<int> masterBoundaryNodeID_;
    HostDeviceArray1D<int> slaveParticleID_;
    HostDeviceArray1D<double3> slidingSpring0_;
    HostDeviceArray1D<int> slaveParticleID0_;
    int numPair_{0};
    size_t pairGridDim_{1};
    size_t pairBlockDim_{1};

    HostDeviceArray1D<int> boundaryNodeNeighborCount_;
    HostDeviceArray1D<int> boundaryNodeNeighborPrefixSum_;
    HostDeviceArray1D<int> boundaryNodeNeighborPrefixSum0_;
    size_t boundaryNodeGridDim_{1};
    size_t boundaryNodeBlockDim_{1};
};

struct BondedParticleInteraction
{
public:
    BondedParticleInteraction() = default;
    ~BondedParticleInteraction() = default;

    BondedParticleInteraction(const BondedParticleInteraction&) = delete;
    BondedParticleInteraction& operator=(const BondedParticleInteraction&) = delete;

    BondedParticleInteraction(BondedParticleInteraction&&) noexcept = default;
    BondedParticleInteraction& operator=(BondedParticleInteraction&&) noexcept = default;

public:
    void addBond(const std::vector<int>& masterParticleID, 
    const std::vector<int>& slaveParticleID, 
    const std::vector<double3>& contactPoint, 
    const std::vector<double3>& contactNormal, 
    const double YoungsModulus, 
    const double poissonRatio, 
    const double radius, 
    const double tensileStrength, 
    const double cohesion, 
    const double frictionCoefficient, 

    const std::vector<double3>& particlePosition, 
    const std::vector<quaternion>& particleOrientation, 

    const size_t maxGPUThread, 
    cudaStream_t stream)
    {
        if (YoungsModulus <= 0. || poissonRatio <= -1. || radius <= 0.) return;

        if (upload_) 
        {
            copyDeviceToHost(stream);
            upload_ = false;
        }

        const size_t n = std::min({masterParticleID.size(),
        slaveParticleID.size(),
        contactPoint.size(),
        contactNormal.size()});

        const double Rb = radius;
        const double Lb = 2 * Rb;
        const double Eb = YoungsModulus;
        const double Gb = Eb / (2. * (1. + poissonRatio));
        const double Ab = M_PI * Rb * Rb;
        const double Ib = Ab * Rb * Rb / 4.;
        const double Jb = 2. * Ib;
        const double Kn = Eb * Ab / Lb;
        const double Kt = Gb * Jb / Lb;
        const double Ks = 12. * Eb * Ib / (Lb * Lb * Lb);
        const double Kb = Eb * Ib / Lb;

        for (size_t k = 0; k < n; ++k)
        {
            const int idxA = masterParticleID[k];
            const int idxB = slaveParticleID[k];
            if (idxA >= particlePosition.size() || idxB >= particlePosition.size()) continue;
            if (idxA >= particleOrientation.size() || idxB >= particleOrientation.size()) continue;
            const double3 rA = particlePosition[idxA];
            const double3 rB = particlePosition[idxB];
            const quaternion qA = particleOrientation[idxA];
            const quaternion qB = particleOrientation[idxB];

            const double3 pc_in = contactPoint[k];
            double3 n_in = contactNormal[k];
            const double nlen = length(n_in);
            if (isZero(nlen)) continue;
            n_in = n_in / nlen;

            // Bond endpoints in GLOBAL frame (along bond normal)
            const double3 pA_global = pc_in + 0.5 * Lb * n_in;
            const double3 pB_global = pc_in - 0.5 * Lb * n_in;

            // Convert endpoints to each particle local frame
            const double3 pA_local = reverseRotateVectorByQuaternion(pA_global - rA, qA);
            const double3 pB_local = reverseRotateVectorByQuaternion(pB_global - rB, qB);

            // Pair mapping
            masterParticleID_.pushHost(idxA);
            slaveParticleID_.pushHost(idxB);

            // Bond state
            isBonded_.pushHost(1);
            point_.pushHost(pc_in);
            normal_.pushHost(n_in);
            normalForce_.pushHost(0.0);
            torsionTorque_.pushHost(0.0);
            shearForce_.pushHost(make_double3(0.0, 0.0, 0.0));
            bendingTorque_.pushHost(make_double3(0.0, 0.0, 0.0));
            maxNormalStress_.pushHost(0.0);
            maxShearStress_.pushHost(0.0);

            // Bond parameters
            normalStiffness_.pushHost(Kn);
            torsionStiffness_.pushHost(Kt);
            shearStiffness_.pushHost(Ks);
            bendingStiffness_.pushHost(Kb);
            radius_.pushHost(radius);
            tensileStrength_.pushHost(tensileStrength);
            cohesion_.pushHost(cohesion);
            frictionCoefficient_.pushHost(frictionCoefficient);
            endPointALocalPosition_.pushHost(pA_local);
            endPointBLocalPosition_.pushHost(pB_local);
        }
    }

    void addLSBond(LSParticle& LSP, 
    LSParticleInteraction& LSPI, 
    const double YoungsModulus, 
    const double poissonRatio, 
    const double radius, 
    const double tensileStrength, 
    const double cohesion, 
    const double frictionCoefficient, 
    const size_t maxGPUThread, 
    cudaStream_t stream)
    {
        const size_t numPair = LSPI.numPair_device();
        if (numPair == 0) return;

        std::vector<int> masterParticleID;
        std::vector<int> masterBoundaryNodeID = LSPI.masterBoundaryNodeIDHostCopy();
        for (const auto& p : masterBoundaryNodeID)
        {
            masterParticleID.push_back(LSP.LSBoundaryNode_.particleIDHostRef()[p]);
        }
        std::vector<int> slaveParticleID = LSPI.slaveParticleIDHostCopy();
        std::vector<double3> contactPoint = LSPI.contactPointHostCopy();
        std::vector<double3> contactNormal = LSPI.contactNormalHostCopy();

        std::vector<double3> particlePosition = LSP.positionHostCopy();
        std::vector<quaternion> particleOrientation = LSP.orientationHostCopy();

        addBond(masterParticleID, 
        slaveParticleID, 
        contactPoint, 
        contactNormal, 
        YoungsModulus, 
        poissonRatio, 
        radius, 
        tensileStrength, 
        cohesion, 
        frictionCoefficient,
        particlePosition, 
        particleOrientation, 
        maxGPUThread, 
        stream);
    }

    void initialize(const size_t maxGPUThread, cudaStream_t stream)
    {
        if (numPair() == 0 || upload_) return;
        copyHostToDevice(stream);
        if (maxGPUThread > 0) blockDim_ = maxGPUThread;
        if (numPair() < maxGPUThread) blockDim_ = numPair();
        gridDim_ = (numPair() + blockDim_ - 1) / blockDim_;
        upload_ = true;
    }

    void outputVTU(const std::string& dir,
    const size_t iFrame,
    const size_t iStep,
    const double time)
    {
        MKDIR(dir.c_str());

        std::ostringstream fname;
        fname << dir << "/LSBondState_"
            << std::setw(4) << std::setfill('0') << iFrame
            << ".vtu";

        const std::vector<int> isBonded = isBondedHostCopy();
        const std::vector<double3> bondPoint = pointHostCopy();
        const std::vector<double3> bondNormal = normalHostCopy();

        const std::vector<double> normalForce = normalForceHostCopy();
        const std::vector<double> torsionTorque = torsionTorqueHostCopy();
        const std::vector<double3> shearForce = shearForceHostCopy();
        const std::vector<double3> bendingTorque = bendingTorqueHostCopy();
        const std::vector<double> maxNormalStress = maxNormalStressHostCopy();
        const std::vector<double> maxShearStress = maxShearStressHostCopy();

        const std::vector<double> normalStiffness = normalStiffnessHostRef();
        const std::vector<double> shearStiffness = shearStiffnessHostRef();
        const std::vector<double> bendingStiffness = bendingStiffnessHostRef();
        const std::vector<double> torsionStiffness = torsionStiffnessHostRef();

        const size_t N = bondPoint.size();

        std::vector<float> points(3 * N);
        std::vector<int32_t> conn(N);
        std::vector<int32_t> offs(N);
        std::vector<uint8_t> types(N);

        std::vector<int32_t> bonded(N);
        std::vector<float> bNormal(3 * N);

        std::vector<float> sN(N);
        std::vector<float> sS(N);

        std::vector<float> En(N);
        std::vector<float> Es(N);
        std::vector<float> Eb(N);
        std::vector<float> Et(N);
        std::vector<float> Etotal(N);

        for (size_t i = 0; i < N; ++i)
        {
            const double3 pc = bondPoint[i];
            const double3 nn = bondNormal[i];

            const double fn = normalForce[i];
            const double tt = torsionTorque[i];
            const double3 fs = shearForce[i];
            const double3 tb = bendingTorque[i];

            const double kn = normalStiffness[i];
            const double ks = shearStiffness[i];
            const double kb = bendingStiffness[i];
            const double kt = torsionStiffness[i];

            points[3 * i + 0] = static_cast<float>(pc.x);
            points[3 * i + 1] = static_cast<float>(pc.y);
            points[3 * i + 2] = static_cast<float>(pc.z);

            conn[i] = static_cast<int32_t>(i);
            offs[i] = static_cast<int32_t>(i + 1);
            types[i] = static_cast<uint8_t>(1);

            bonded[i] = static_cast<int32_t>(isBonded[i]);

            bNormal[3 * i + 0] = static_cast<float>(nn.x);
            bNormal[3 * i + 1] = static_cast<float>(nn.y);
            bNormal[3 * i + 2] = static_cast<float>(nn.z);

            sN[i] = static_cast<float>(maxNormalStress[i]);
            sS[i] = static_cast<float>(maxShearStress[i]);

            const double fs2 = fs.x * fs.x + fs.y * fs.y + fs.z * fs.z;
            const double tb2 = tb.x * tb.x + tb.y * tb.y + tb.z * tb.z;

            const double en = (kn > 0.0) ? 0.5 * fn * fn / kn : 0.0;
            const double es = (ks > 0.0) ? 0.5 * fs2 / ks : 0.0;
            const double eb = (kb > 0.0) ? 0.5 * tb2 / kb : 0.0;
            const double et = (kt > 0.0) ? 0.5 * tt * tt / kt : 0.0;

            En[i] = static_cast<float>(en);
            Es[i] = static_cast<float>(es);
            Eb[i] = static_cast<float>(eb);
            Et[i] = static_cast<float>(et);
            Etotal[i] = static_cast<float>(en + es + eb + et);
        }

        auto align8 = [](const size_t n) -> size_t
        {
            const size_t a = 8;
            return (n + (a - 1)) & ~(a - 1);
        };

        auto blockBytes = [&](const size_t payloadBytes) -> size_t
        {
            return sizeof(uint64_t) + align8(payloadBytes);
        };

        const size_t off_points = 0;
        const size_t off_conn = off_points + blockBytes(points.size() * sizeof(float));
        const size_t off_offs = off_conn + blockBytes(conn.size() * sizeof(int32_t));
        const size_t off_types = off_offs + blockBytes(offs.size() * sizeof(int32_t));

        const size_t off_bonded = off_types + blockBytes(types.size() * sizeof(uint8_t));
        const size_t off_bNormal = off_bonded + blockBytes(bonded.size() * sizeof(int32_t));

        const size_t off_sN = off_bNormal + blockBytes(bNormal.size() * sizeof(float));
        const size_t off_sS = off_sN + blockBytes(sN.size() * sizeof(float));

        const size_t off_En = off_sS + blockBytes(sS.size() * sizeof(float));
        const size_t off_Es = off_En + blockBytes(En.size() * sizeof(float));
        const size_t off_Eb = off_Es + blockBytes(Es.size() * sizeof(float));
        const size_t off_Et = off_Eb + blockBytes(Eb.size() * sizeof(float));
        const size_t off_Etotal = off_Et + blockBytes(Et.size() * sizeof(float));

        std::ofstream out(fname.str(), std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open " + fname.str());

        out
            << "<?xml version=\"1.0\"?>\n"
            << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
            << "  <UnstructuredGrid>\n"
            << "    <FieldData>\n"
            << "      <DataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\"> "
            << static_cast<float>(time) << " </DataArray>\n"
            << "      <DataArray type=\"Int32\" Name=\"STEP\" NumberOfTuples=\"1\" format=\"ascii\"> "
            << static_cast<int32_t>(iStep) << " </DataArray>\n"
            << "    </FieldData>\n"
            << "    <Piece NumberOfPoints=\"" << N << "\" NumberOfCells=\"" << N << "\">\n"
            << "      <Points>\n"
            << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_points << "\"/>\n"
            << "      </Points>\n"
            << "      <Cells>\n"
            << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\"" << off_conn << "\"/>\n"
            << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\"" << off_offs << "\"/>\n"
            << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\"" << off_types << "\"/>\n"
            << "      </Cells>\n"
            << "      <PointData>\n"
            << "        <DataArray type=\"Int32\" Name=\"isBonded\" format=\"appended\" offset=\"" << off_bonded << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"bondNormal\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_bNormal << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"maxNormalStress\" format=\"appended\" offset=\"" << off_sN << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"maxShearStress\" format=\"appended\" offset=\"" << off_sS << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"normalElasticEnergy\" format=\"appended\" offset=\"" << off_En << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"shearElasticEnergy\" format=\"appended\" offset=\"" << off_Es << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"bendingElasticEnergy\" format=\"appended\" offset=\"" << off_Eb << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"torsionElasticEnergy\" format=\"appended\" offset=\"" << off_Et << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"totalElasticEnergy\" format=\"appended\" offset=\"" << off_Etotal << "\"/>\n"
            << "      </PointData>\n"
            << "    </Piece>\n"
            << "  </UnstructuredGrid>\n"
            << "  <AppendedData encoding=\"raw\">\n"
            << "    _";

        auto writeBlock = [&](const void* data, const size_t nbytes)
        {
            const uint64_t sz = static_cast<uint64_t>(nbytes);
            out.write(reinterpret_cast<const char*>(&sz), sizeof(uint64_t));

            if (nbytes)
            {
                out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(nbytes));
            }

            const size_t pad = align8(nbytes) - nbytes;
            if (pad)
            {
                static const char zeros[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                out.write(zeros, static_cast<std::streamsize>(pad));
            }
        };

        writeBlock(points.data(), points.size() * sizeof(float));
        writeBlock(conn.data(), conn.size() * sizeof(int32_t));
        writeBlock(offs.data(), offs.size() * sizeof(int32_t));
        writeBlock(types.data(), types.size() * sizeof(uint8_t));

        writeBlock(bonded.data(), bonded.size() * sizeof(int32_t));
        writeBlock(bNormal.data(), bNormal.size() * sizeof(float));

        writeBlock(sN.data(), sN.size() * sizeof(float));
        writeBlock(sS.data(), sS.size() * sizeof(float));

        writeBlock(En.data(), En.size() * sizeof(float));
        writeBlock(Es.data(), Es.size() * sizeof(float));
        writeBlock(Eb.data(), Eb.size() * sizeof(float));
        writeBlock(Et.data(), Et.size() * sizeof(float));
        writeBlock(Etotal.data(), Etotal.size() * sizeof(float));

        out
            << "\n"
            << "  </AppendedData>\n"
            << "</VTKFile>\n";
    }

    void finalize(cudaStream_t stream)
    {
        copyDeviceToHost(stream);
    }

    void copyFromHost(const BondedParticleInteraction& other)
    {
        masterParticleID_.setHost(other.masterParticleID_.hostRef());
        slaveParticleID_.setHost(other.slaveParticleID_.hostRef());

        isBonded_.setHost(other.isBonded_.hostRef());
        point_.setHost(other.point_.hostRef());
        normal_.setHost(other.normal_.hostRef());
        normalForce_.setHost(other.normalForce_.hostRef());
        torsionTorque_.setHost(other.torsionTorque_.hostRef());
        shearForce_.setHost(other.shearForce_.hostRef());
        bendingTorque_.setHost(other.bendingTorque_.hostRef());
        maxNormalStress_.setHost(other.maxNormalStress_.hostRef());
        maxShearStress_.setHost(other.maxShearStress_.hostRef());

        normalStiffness_.setHost(other.normalStiffness_.hostRef());
        torsionStiffness_.setHost(other.torsionStiffness_.hostRef());
        shearStiffness_.setHost(other.shearStiffness_.hostRef());
        bendingStiffness_.setHost(other.bendingStiffness_.hostRef());
        radius_.setHost(other.radius_.hostRef());
        tensileStrength_.setHost(other.tensileStrength_.hostRef());
        cohesion_.setHost(other.cohesion_.hostRef());
        frictionCoefficient_.setHost(other.frictionCoefficient_.hostRef());
        endPointALocalPosition_.setHost(other.endPointALocalPosition_.hostRef());
        endPointBLocalPosition_.setHost(other.endPointBLocalPosition_.hostRef());

        upload_ = false;
    }

    int* masterParticleID() { return masterParticleID_.d_ptr; }
    int* slaveParticleID() { return slaveParticleID_.d_ptr; }

    int* isBonded() { return isBonded_.d_ptr; }
    double3* point() { return point_.d_ptr; }
    double3* normal() { return normal_.d_ptr; }
    double* normalForce() { return normalForce_.d_ptr; }
    double* torsionTorque() { return torsionTorque_.d_ptr; }
    double3* shearForce() { return shearForce_.d_ptr; }
    double3* bendingTorque() { return bendingTorque_.d_ptr; }
    double* maxNormalStress() { return maxNormalStress_.d_ptr; }
    double* maxShearStress() { return maxShearStress_.d_ptr; }

    double* normalStiffness() { return normalStiffness_.d_ptr; }
    double* torsionStiffness() { return torsionStiffness_.d_ptr; }
    double* shearStiffness() { return shearStiffness_.d_ptr; }
    double* bendingStiffness() { return bendingStiffness_.d_ptr; }
    double* radius() { return radius_.d_ptr; }
    double* tensileStrength() { return tensileStrength_.d_ptr; }
    double* cohesion() { return cohesion_.d_ptr; }
    double* frictionCoefficient() { return frictionCoefficient_.d_ptr; }
    double3* endPointALocalPosition() { return endPointALocalPosition_.d_ptr; }
    double3* endPointBLocalPosition() { return endPointBLocalPosition_.d_ptr; }

    std::vector<int> isBondedHostCopy() { return isBonded_.getHostCopy(); }
    std::vector<double3> pointHostCopy() { return point_.getHostCopy(); }
    std::vector<double3> normalHostCopy() { return normal_.getHostCopy(); }
    std::vector<double> normalForceHostCopy() { return normalForce_.getHostCopy(); }
    std::vector<double> torsionTorqueHostCopy() { return torsionTorque_.getHostCopy(); }
    std::vector<double3> shearForceHostCopy() { return shearForce_.getHostCopy(); }
    std::vector<double3> bendingTorqueHostCopy() { return bendingTorque_.getHostCopy(); }
    std::vector<double> maxNormalStressHostCopy() { return maxNormalStress_.getHostCopy(); }
    std::vector<double> maxShearStressHostCopy() { return maxShearStress_.getHostCopy(); }

    const std::vector<double>& normalStiffnessHostRef() const { return normalStiffness_.hostRef(); }
    const std::vector<double>& torsionStiffnessHostRef() const { return torsionStiffness_.hostRef(); }
    const std::vector<double>& shearStiffnessHostRef() const { return shearStiffness_.hostRef(); }
    const std::vector<double>& bendingStiffnessHostRef() const { return bendingStiffness_.hostRef(); }

    const size_t numPair() const { return point_.hostSize(); }
    const size_t numPair_device() const { return point_.deviceSize(); }
    const size_t& gridDim() const { return gridDim_; }
    const size_t& blockDim() const { return blockDim_; }

private:
    void copyHostToDevice(cudaStream_t stream)
    {
        // pairs
        masterParticleID_.copyHostToDevice(stream);
        slaveParticleID_.copyHostToDevice(stream);

        // bond state
        isBonded_.copyHostToDevice(stream);
        point_.copyHostToDevice(stream);
        normal_.copyHostToDevice(stream);
        normalForce_.copyHostToDevice(stream);
        torsionTorque_.copyHostToDevice(stream);
        shearForce_.copyHostToDevice(stream);
        bendingTorque_.copyHostToDevice(stream);
        maxNormalStress_.copyHostToDevice(stream);
        maxShearStress_.copyHostToDevice(stream);

        normalStiffness_.copyHostToDevice(stream);
        torsionStiffness_.copyHostToDevice(stream);
        shearStiffness_.copyHostToDevice(stream);
        bendingStiffness_.copyHostToDevice(stream);
        radius_.copyHostToDevice(stream);
        tensileStrength_.copyHostToDevice(stream);
        cohesion_.copyHostToDevice(stream);
        frictionCoefficient_.copyHostToDevice(stream);
        endPointALocalPosition_.copyHostToDevice(stream);
        endPointBLocalPosition_.copyHostToDevice(stream);
    }

    void copyDeviceToHost(cudaStream_t stream)
    {
        // pairs
        masterParticleID_.copyDeviceToHost(stream);
        slaveParticleID_.copyDeviceToHost(stream);

        // bond state
        isBonded_.copyDeviceToHost(stream);
        point_.copyDeviceToHost(stream);
        normal_.copyDeviceToHost(stream);
        normalForce_.copyDeviceToHost(stream);
        torsionTorque_.copyDeviceToHost(stream);
        shearForce_.copyDeviceToHost(stream);
        bendingTorque_.copyDeviceToHost(stream);
        maxNormalStress_.copyDeviceToHost(stream);
        maxShearStress_.copyDeviceToHost(stream);
    }

private:
    HostDeviceArray1D<int> masterParticleID_;
    HostDeviceArray1D<int> slaveParticleID_;

    HostDeviceArray1D<int> isBonded_;
    HostDeviceArray1D<double3> point_;
    HostDeviceArray1D<double3> normal_;
    HostDeviceArray1D<double> normalForce_;
    HostDeviceArray1D<double> torsionTorque_;
    HostDeviceArray1D<double3> shearForce_;
    HostDeviceArray1D<double3> bendingTorque_;
    HostDeviceArray1D<double> maxNormalStress_;
    HostDeviceArray1D<double> maxShearStress_;

    HostDeviceArray1D<double> normalStiffness_;
    HostDeviceArray1D<double> torsionStiffness_;
    HostDeviceArray1D<double> shearStiffness_;
    HostDeviceArray1D<double> bendingStiffness_;
    HostDeviceArray1D<double> radius_;
    HostDeviceArray1D<double> tensileStrength_;
    HostDeviceArray1D<double> cohesion_;
    HostDeviceArray1D<double> frictionCoefficient_;
    HostDeviceArray1D<double3> endPointALocalPosition_;
    HostDeviceArray1D<double3> endPointBLocalPosition_;

    size_t gridDim_{1};
    size_t blockDim_{1};

    bool upload_{false};
};