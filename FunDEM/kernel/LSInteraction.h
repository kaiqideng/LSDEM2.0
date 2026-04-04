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
    const double time) const
    {
        MKDIR(dir.c_str());

        std::ostringstream fname;
        fname << dir << "/LSInteraction_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";

        const size_t N = numPair_;

        std::vector<int> pairMasterBoundaryID = masterBoundaryNodeID_.hostRef();
        std::vector<int> pairSlaveParticleID = slaveParticleID_.hostRef();
        std::vector<double3> p = contactPoint_.hostRef();
        std::vector<double> o = contactOverlap_.hostRef();
        std::vector<double3> n = contactNormal_.hostRef();
        std::vector<double3> s = slidingSpring_.hostRef();

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