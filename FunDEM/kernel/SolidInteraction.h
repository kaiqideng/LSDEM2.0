#pragma once
#include "CUDAKernelFunction/myUtility/myHostDeviceArray.h"
#include "CUDAKernelFunction/myUtility/myVec.h"
#include "CUDAKernelFunction/myUtility/buildHashStartEnd.h"
#include "CUDAKernelFunction/myUtility/myFileEdit.h"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>

struct SolidInteraction
{
public:
    SolidInteraction() = default;
    ~SolidInteraction() = default;

    SolidInteraction(const SolidInteraction&) = delete;
    SolidInteraction& operator=(const SolidInteraction&) = delete;

    SolidInteraction(SolidInteraction&&) noexcept = default;
    SolidInteraction& operator=(SolidInteraction&&) noexcept = default;

    void initialize(const size_t numMasterObject, const size_t maxGPUThread, cudaStream_t stream)
    {
        const size_t numMasterObject0 = masterNeighborPrefixSum_.hostSize();
        if (numMasterObject <= numMasterObject0) return;
        
        if (numMasterObject0 == 0)
        {
            masterNeighborCount_.allocateDevice(numMasterObject, stream);
            masterNeighborPrefixSum_.allocateDevice(numMasterObject, stream);
            masterNeighborPrefixSum0_.allocateDevice(numMasterObject, stream);
        }
        else
        {
            masterNeighborPrefixSum_.copyDeviceToHost(stream);
            for (size_t i = 0; i < numMasterObject - numMasterObject0; i++)
            {
                masterNeighborPrefixSum_.pushHost(masterNeighborPrefixSum_.hostRef()[numMasterObject0 - 1]);
            }
            masterNeighborPrefixSum_.copyHostToDevice(stream);
            masterNeighborCount_.allocateDevice(numMasterObject, stream);
            masterNeighborPrefixSum0_.allocateDevice(numMasterObject, stream);
        }
        
        if (maxGPUThread > 0) masterBlockDim_ = maxGPUThread;
        if (numMasterObject < maxGPUThread) masterBlockDim_ = numMasterObject;
        masterGridDim_ = (numMasterObject + masterBlockDim_ - 1) / masterBlockDim_;
    }

    void setInteractionArraySize(const size_t arraySize, cudaStream_t stream)
    {
        contactPoint_.allocateDevice(arraySize, stream);
        contactNormal_.allocateDevice(arraySize, stream);
        contactOverlap_.allocateDevice(arraySize, stream);
        normalElasticEnergy_.allocateDevice(arraySize, stream);
        slidingElasticEnergy_.allocateDevice(arraySize, stream);
        slidingSpring_.allocateDevice(arraySize, stream);
        masterID_.allocateDevice(arraySize, stream);
        slaveID_.allocateDevice(arraySize, stream);

        slidingSpring0_.allocateDevice(arraySize, stream);
        slaveID0_.allocateDevice(arraySize, stream);
    }

    void updateNeighborPrefixSum(cudaStream_t stream)
    {
        cudaMemcpyAsync(masterNeighborPrefixSum0_.d_ptr, masterNeighborPrefixSum_.d_ptr, 
        masterNeighborPrefixSum_.deviceSize() * sizeof(int), cudaMemcpyDeviceToDevice, stream);

        buildPrefixSum(masterNeighborPrefixSum_.d_ptr,
        masterNeighborCount_.d_ptr,
        masterNeighborCount_.deviceSize(),
        stream);
    }

    void updateNumPair(const size_t maxGPUThread, cudaStream_t stream)
    {
        cudaStreamSynchronize(stream);
        cudaMemcpyAsync(&numPair_, masterNeighborPrefixSum_.d_ptr + masterNeighborPrefixSum_.deviceSize() - 1, 
        sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (numPair_ > 0)
        {
            pairBlockDim_ = static_cast<size_t>(numPair_);
            if (maxGPUThread > 0 && maxGPUThread < static_cast<size_t>(numPair_)) pairBlockDim_ = maxGPUThread;
            pairGridDim_ = (static_cast<size_t>(numPair_) + pairBlockDim_ - 1) / pairBlockDim_;   
        }
        else
        {
            pairGridDim_ = 1;
            pairBlockDim_ = 1;
            return;
        }
    }

    void savePreviousStep(cudaStream_t stream)
    {
        if (static_cast<size_t>(numPair_) > contactPoint_.deviceSize())
        {
            slidingSpring0_.allocateDevice(static_cast<size_t>(numPair_), stream);
            slaveID0_.allocateDevice(static_cast<size_t>(numPair_), stream);

            cudaMemcpyAsync(slidingSpring0_.d_ptr, slidingSpring_.d_ptr, 
            slidingSpring_.deviceSize() * sizeof(double3), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(slaveID0_.d_ptr, slaveID_.d_ptr, 
            slaveID_.deviceSize() * sizeof(int), cudaMemcpyDeviceToDevice, stream);
 
            contactPoint_.allocateDevice(static_cast<size_t>(numPair_), stream);
            contactNormal_.allocateDevice(static_cast<size_t>(numPair_), stream);
            contactOverlap_.allocateDevice(static_cast<size_t>(numPair_), stream);
            normalElasticEnergy_.allocateDevice(static_cast<size_t>(numPair_), stream);
            slidingElasticEnergy_.allocateDevice(static_cast<size_t>(numPair_), stream);
            slidingSpring_.allocateDevice(static_cast<size_t>(numPair_), stream);
            masterID_.allocateDevice(static_cast<size_t>(numPair_), stream);
            slaveID_.allocateDevice(static_cast<size_t>(numPair_), stream);
        }
        else 
        {
            cudaMemcpyAsync(slidingSpring0_.d_ptr, slidingSpring_.d_ptr, 
            slidingSpring_.deviceSize() * sizeof(double3), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(slaveID0_.d_ptr, slaveID_.d_ptr, 
            slaveID_.deviceSize() * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        }
    }

    void outputVTU(const std::string& dir, const size_t iFrame, const size_t iStep, const double time) const
    {
        MKDIR(dir.c_str());

        std::ostringstream fname;
        fname << dir << "/LSInteraction_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";

        size_t N = 0;
        if (numMaster() > 0) N = masterNeighborPrefixSum_.hostRef()[numMaster() - 1];

        const std::vector<double3>& p = contactPoint_.hostRef();
        const std::vector<double3>& n = contactNormal_.hostRef();
        const std::vector<double>& nE = normalElasticEnergy_.hostRef();
        const std::vector<double>& sE = slidingElasticEnergy_.hostRef();
        const std::vector<double3>& s = slidingSpring_.hostRef();

        std::vector<float> points;
        std::vector<float> normal;
        std::vector<float> slidingDirection;

        std::vector<float> normalElasticEnergy;
        std::vector<float> slidingElasticEnergy;

        std::vector<int32_t> conn;
        std::vector<int32_t> offs;
        std::vector<uint8_t> types;

        points.resize(N * 3);
        normal.resize(N * 3);
        slidingDirection.resize(N * 3);

        normalElasticEnergy.resize(N);
        slidingElasticEnergy.resize(N);

        conn.resize(N);
        offs.resize(N);
        types.resize(N);

        for (size_t i = 0; i < N; ++i)
        {
            points[3 * i + 0] = static_cast<float>(p[i].x);
            points[3 * i + 1] = static_cast<float>(p[i].y);
            points[3 * i + 2] = static_cast<float>(p[i].z);

            normal[3 * i + 0] = static_cast<float>(n[i].x);
            normal[3 * i + 1] = static_cast<float>(n[i].y);
            normal[3 * i + 2] = static_cast<float>(n[i].z);

            const double3 s1 = normalize(s[i]);
            slidingDirection[3 * i + 0] = static_cast<float>(s1.x);
            slidingDirection[3 * i + 1] = static_cast<float>(s1.y);
            slidingDirection[3 * i + 2] = static_cast<float>(s1.z);

            normalElasticEnergy[i] = static_cast<float>(nE[i]);
            slidingElasticEnergy[i] = static_cast<float>(sE[i]);

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
        size_t off_slidingDirection = off_normal + blockBytes(normal.size() * sizeof(float));

        size_t off_normalElasticEnergy = off_slidingDirection + blockBytes(slidingDirection.size() * sizeof(float));
        size_t off_slidingElasticEnergy = off_normalElasticEnergy + blockBytes(normalElasticEnergy.size() * sizeof(float));

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
            << "        <DataArray type=\"Float32\" Name=\"slidingDirection\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_slidingDirection << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"normalElasticEnergy\" format=\"appended\" offset=\"" << off_normalElasticEnergy << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"slidingElasticEnergy\" format=\"appended\" offset=\"" << off_slidingElasticEnergy << "\"/>\n"
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
        writeBlock(slidingDirection.data(), slidingDirection.size() * sizeof(float));

        writeBlock(normalElasticEnergy.data(), normalElasticEnergy.size() * sizeof(float));
        writeBlock(slidingElasticEnergy.data(), slidingElasticEnergy.size() * sizeof(float));

        out << "\n  </AppendedData>\n</VTKFile>\n";
    }

    void finalize(cudaStream_t stream)
    {
        copyDeviceToHost(stream);
    }

    double3* contactPoint() { return contactPoint_.d_ptr; }
    double3* contactNormal() { return contactNormal_.d_ptr; }
    double* contactOverlap() { return contactOverlap_.d_ptr; }
    double* normalElasticEnergy() { return normalElasticEnergy_.d_ptr; }
    double* slidingElasticEnergy() { return slidingElasticEnergy_.d_ptr; }
    double3* slidingSpring() { return slidingSpring_.d_ptr; }
    int* masterID() { return masterID_.d_ptr; }
    int* slaveID() { return slaveID_.d_ptr; }
    double3* previousSlidingSpring() { return slidingSpring0_.d_ptr; }
    int* previousSlaveID() { return slaveID0_.d_ptr; }

    size_t numPair_device() const { return static_cast<size_t>(numPair_); }
    const size_t& pairGridDim() const { return pairGridDim_; }
    const size_t& pairBlockDim() const { return pairBlockDim_; }

    int* masterNeighborCount() { return masterNeighborCount_.d_ptr; }
    int* masterNeighborPrefixSum() { return masterNeighborPrefixSum_.d_ptr; }
    int* previousMasterNeighborPrefixSum() { return masterNeighborPrefixSum0_.d_ptr; }

    size_t numMaster() const { return masterNeighborPrefixSum_.hostSize(); }
    size_t numMaster_device() const { return masterNeighborPrefixSum_.deviceSize(); }
    const size_t& masterGridDim() const { return masterGridDim_; }
    const size_t& masterBlockDim() const { return masterBlockDim_; }

    const std::vector<double3> contactPointHostRef() const
    {
        std::vector<double3> v = contactPoint_.hostRef();
        v.resize(numPair_device());
        return v;
    }

    const std::vector<double3> contactNormalHostRef() const
    {
        std::vector<double3> v = contactNormal_.hostRef();
        v.resize(numPair_device());
        return v;
    }

    const std::vector<int> masterIDHostRef() const
    {
        std::vector<int> v = masterID_.hostRef();
        v.resize(numPair_device());
        return v;
    }

    const std::vector<int> slaveIDHostRef() const
    {
        std::vector<int> v = slaveID_.hostRef();
        v.resize(numPair_device());
        return v;
    }

private:
    void copyDeviceToHost(cudaStream_t stream)
    {
        contactPoint_.copyDeviceToHost(stream);
        contactNormal_.copyDeviceToHost(stream);
        contactOverlap_.copyDeviceToHost(stream);
        normalElasticEnergy_.copyDeviceToHost(stream);
        slidingElasticEnergy_.copyDeviceToHost(stream);
        slidingSpring_.copyDeviceToHost(stream);
        masterID_.copyDeviceToHost(stream);
        slaveID_.copyDeviceToHost(stream);

        masterNeighborPrefixSum_.copyDeviceToHost(stream);
    }

    HostDeviceArray1D<double3> contactPoint_;
    HostDeviceArray1D<double3> contactNormal_;
    HostDeviceArray1D<double> contactOverlap_;
    HostDeviceArray1D<double> normalElasticEnergy_;
    HostDeviceArray1D<double> slidingElasticEnergy_;
    HostDeviceArray1D<double3> slidingSpring_;
    HostDeviceArray1D<int> masterID_;
    HostDeviceArray1D<int> slaveID_;
    HostDeviceArray1D<double3> slidingSpring0_;
    HostDeviceArray1D<int> slaveID0_;
    int numPair_{0};
    size_t pairGridDim_{1};
    size_t pairBlockDim_{1};

    HostDeviceArray1D<int> masterNeighborCount_;
    HostDeviceArray1D<int> masterNeighborPrefixSum_;
    HostDeviceArray1D<int> masterNeighborPrefixSum0_;
    size_t masterGridDim_{1};
    size_t masterBlockDim_{1};
};