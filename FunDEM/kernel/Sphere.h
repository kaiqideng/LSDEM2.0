#pragma once
#include "CUDAKernelFunction/myUtility/myHostDeviceArray.h"
#include "CUDAKernelFunction/myUtility/myMat.h"
#include "CUDAKernelFunction/myUtility/mySpatialGrid.h"
#include "CUDAKernelFunction/myUtility/myFileEdit.h"
#include <algorithm>
#include <cstdint>
#include <fstream>

struct Sphere
{
public:
    Sphere() = default;
    ~Sphere() = default;

    Sphere(const Sphere&) = delete;
    Sphere& operator=(const Sphere&) = delete;

    Sphere(Sphere&&) noexcept = default;
    Sphere& operator=(Sphere&&) noexcept = default;

    /**
    * @brief Append one sphere particle to host-side buffers.
    *
    * @param[in] position                 Particle center position in WORLD frame.
    * @param[in] velocity                 Particle linear velocity in WORLD frame.
    * @param[in] angularVelocity          Particle angular velocity in WORLD frame.
    * @param[in] orientation              Particle orientation.
    * @param[in] radius                   Sphere radius.
    * @param[in] normalStiffness          Normal stiffness for this particle.
    * @param[in] shearStiffness           Shear stiffness for this particle.
    * @param[in] frictionCoefficient      Friction coefficient for this particle.
    * @param[in] restitutionCoefficient   Restitution coefficient for this particle, valid in (0, 1].
    *                                     Invalid input falls back to 1.0.
    * @param[in] density                  Material density. Zero is allowed.
    * @param[in] stream                   CUDA stream (used only if we must download device -> host).
    */
    void add(const double3 position,
    const double3 velocity,
    const double3 angularVelocity,
    const quaternion orientation,
    const double radius,
    const double normalStiffness,
    const double shearStiffness,
    const double frictionCoefficient,
    const double restitutionCoefficient = 1.0,
    const double density = 0.0,
    cudaStream_t stream = 0)
    {
        if (radius <= 0.0)
        {
            std::cerr << "[Sphere] Invalid sphere radius: "
                    << radius << "."
                    << std::endl;
            return;
        }

        if (density < 0.0)
        {
            std::cerr << "[Sphere] Invalid density: "
                    << density << "."
                    << std::endl;
            return;
        }

        if (upload_)
        {
            copyDeviceToHost(stream);
            upload_ = false;
        }

        const double volume = (4.0 / 3.0) * pi() * radius * radius * radius;
        const double mass = density * volume;
        const double invM = (mass > 0.0) ? (1.0 / mass) : 0.0;

        symMatrix invI = make_symMatrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        if (mass > 0.0)
        {
            const double inertia = (2.0 / 5.0) * mass * radius * radius;
            const double invI_scalar = (inertia > 0.0) ? (1.0 / inertia) : 0.0;
            invI = make_symMatrix(invI_scalar, invI_scalar, invI_scalar,
                                  0.0, 0.0, 0.0);
        }

        double e = restitutionCoefficient;
        if (e <= 0.0 || e > 1.0) e = 1.0;

        position_.pushHost(position);
        velocity_.pushHost(velocity);
        angularVelocity_.pushHost(angularVelocity);
        force_.pushHost(make_double3(0., 0., 0.));
        torque_.pushHost(make_double3(0., 0., 0.));
        orientation_.pushHost(orientation);
        radius_.pushHost(radius);
        inverseMass_.pushHost(invM);
        inverseInertiaTensor_.pushHost(invI);
        normalStiffness_.pushHost(normalStiffness > 0. ? normalStiffness : 0.);
        shearStiffness_.pushHost(shearStiffness > 0. ? shearStiffness : 0.);
        frictionCoefficient_.pushHost(frictionCoefficient > 0. ? frictionCoefficient : 0.);
        restitutionCoefficient_.pushHost(e);
    }
    
    void move(const size_t index, const double3 offset, cudaStream_t stream)
    {
        if (index >= num()) return;

        if (upload_)
        {
            copyDeviceToHost(stream);
            upload_ = false;
        }

        std::vector<double3> pos = position_.hostRef();
        pos[index] += offset;
        position_.setHost(pos);
    }

    void setVelocity(const size_t index, const double3 velocity, cudaStream_t stream)
    {
        if (index >= num()) return;

        if (upload_)
        {
            copyDeviceToHost(stream);
            upload_ = false;
        }

        std::vector<double3> vel = velocity_.hostRef();
        vel[index] = velocity;
        velocity_.setHost(vel);
    }

    void setAngularVelocity(const size_t index, const double3 angularVelocity, cudaStream_t stream)
    {
        if (index >= num()) return;

        if (upload_)
        {
            copyDeviceToHost(stream);
            upload_ = false;
        }

        std::vector<double3> ang = angularVelocity_.hostRef();
        ang[index] = angularVelocity;
        angularVelocity_.setHost(ang);
    }

    void initialize(const double3 minDomain, const double3 maxDomain, const size_t maxGPUThread, cudaStream_t stream)
    {
        if (upload_) return;

        double cellSizeOneDim = 0.;
        if (num() > 0) 
        {
            copyHostToDevice(stream);

            hashValue_.allocateDevice(num_device(), stream);
            hashIndex_.allocateDevice(num_device(), stream);
            if (maxGPUThread > 0) blockDim_ = maxGPUThread;
            if (num_device() < maxGPUThread) blockDim_ = num_device();
            gridDim_ = (num_device() + blockDim_ - 1) / blockDim_;

            cellSizeOneDim = *std::max_element(radius_.hostRef().begin(), radius_.hostRef().end()) * 2.0;
        }
        spatialGrid_.set(minDomain, maxDomain, cellSizeOneDim, stream);
        upload_ = true;
    }

    void outputVTU(const std::string& dir, const size_t iFrame, const size_t iStep, const double time) const
    {
        if (num() == 0) return;
        
        MKDIR(dir.c_str());

        std::ostringstream fname;
        fname << dir << "/Sphere_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";

        const size_t N = num();

        const std::vector<double3>& p = position_.hostRef();
        const std::vector<double3>& v = velocity_.hostRef();
        const std::vector<double>& r = radius_.hostRef();

        std::vector<float> points;
        std::vector<float> vel;
        std::vector<float> rad;
        std::vector<int32_t> conn;
        std::vector<int32_t> offs;
        std::vector<uint8_t> types;

        points.resize(N * 3);
        vel.resize(N * 3);
        rad.resize(N);
        conn.resize(N);
        offs.resize(N);
        types.resize(N);

        for (size_t i = 0; i < N; ++i)
        {
            points[3 * i + 0] = static_cast<float>(p[i].x);
            points[3 * i + 1] = static_cast<float>(p[i].y);
            points[3 * i + 2] = static_cast<float>(p[i].z);

            vel[3 * i + 0] = static_cast<float>(v[i].x);
            vel[3 * i + 1] = static_cast<float>(v[i].y);
            vel[3 * i + 2] = static_cast<float>(v[i].z);

            rad[i] = static_cast<float>(r[i]);

            conn[i] = static_cast<int32_t>(i);
            offs[i] = static_cast<int32_t>(i + 1);
            types[i] = static_cast<uint8_t>(1); // VTK_VERTEX
        }

        auto blockBytes = [](size_t n) -> size_t
        {
            const size_t a = 8;
            const size_t padded = (n + (a - 1)) & ~(a - 1);
            return sizeof(uint64_t) + padded;
        };

        size_t off_points = 0;
        size_t off_conn = off_points + blockBytes(points.size() * sizeof(float));
        size_t off_offs = off_conn + blockBytes(conn.size() * sizeof(int32_t));
        size_t off_types = off_offs + blockBytes(offs.size() * sizeof(int32_t));

        size_t off_vel = off_types + blockBytes(types.size() * sizeof(uint8_t));
        size_t off_rad = off_vel + blockBytes(vel.size() * sizeof(float));

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
            << "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_vel << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"radius\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" << off_rad << "\"/>\n"
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

            const size_t a = 8;
            const size_t padded = (nbytes + (a - 1)) & ~(a - 1);
            const size_t pad = padded - nbytes;
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
        writeBlock(vel.data(), vel.size() * sizeof(float));
        writeBlock(rad.data(), rad.size() * sizeof(float));

        out << "\n  </AppendedData>\n</VTKFile>\n";
    }

    void finalize(cudaStream_t stream)
    {
        copyDeviceToHost(stream);
    }

    void copyFromHost(const Sphere& other)
    {
        position_.setHost(other.position_.hostRef());
        velocity_.setHost(other.velocity_.hostRef());
        angularVelocity_.setHost(other.angularVelocity_.hostRef());
        force_.setHost(other.force_.hostRef());
        torque_.setHost(other.torque_.hostRef());
        orientation_.setHost(other.orientation_.hostRef());
        radius_.setHost(other.radius_.hostRef());
        inverseMass_.setHost(other.inverseMass_.hostRef());
        inverseInertiaTensor_.setHost(other.inverseInertiaTensor_.hostRef());
        normalStiffness_.setHost(other.normalStiffness_.hostRef());
        shearStiffness_.setHost(other.shearStiffness_.hostRef());
        frictionCoefficient_.setHost(other.frictionCoefficient_.hostRef());
        restitutionCoefficient_.setHost(other.restitutionCoefficient_.hostRef());

        upload_ = false;
    }

    const size_t num() const { return position_.hostSize(); }
    const size_t num_device() const { return position_.deviceSize(); }
    const size_t& gridDim() const { return gridDim_; }
    const size_t& blockDim() const { return blockDim_; }

    double3* position() { return position_.d_ptr; }
    double3* velocity() { return velocity_.d_ptr; }
    double3* angularVelocity() { return angularVelocity_.d_ptr; }
    double3* force() { return force_.d_ptr; }
    double3* torque() { return torque_.d_ptr; }
    quaternion* orientation() { return orientation_.d_ptr; }
    double* radius() { return radius_.d_ptr; }
    double* inverseMass() { return inverseMass_.d_ptr; }
    symMatrix* inverseInertiaTensor() { return inverseInertiaTensor_.d_ptr; }
    double* normalStiffness() { return normalStiffness_.d_ptr; }
    double* shearStiffness() { return shearStiffness_.d_ptr; }
    double* frictionCoefficient() { return frictionCoefficient_.d_ptr; }
    double* restitutionCoefficient() { return restitutionCoefficient_.d_ptr; }

    int* hashValue() { return hashValue_.d_ptr; }
    int* hashIndex() { return hashIndex_.d_ptr; }

    std::vector<double3> positionHostCopy() { return position_.getHostCopy(); }
    std::vector<double3> velocityHostCopy() { return velocity_.getHostCopy(); }
    std::vector<double3> angularVelocityHostCopy() { return angularVelocity_.getHostCopy(); }
    std::vector<double3> forceHostCopy() { return force_.getHostCopy(); }
    std::vector<double3> torqueHostCopy() { return torque_.getHostCopy(); }
    std::vector<quaternion> orientationHostCopy() { return orientation_.getHostCopy(); }

    const std::vector<double>& radiusHostRef() const { return radius_.hostRef(); }
    const std::vector<double>& inverseMassHostRef() const { return inverseMass_.hostRef(); }
    const std::vector<symMatrix>& inverseInertiaTensorHostRef() const { return inverseInertiaTensor_.hostRef(); }
    const std::vector<double>& normalStiffnessHostRef() const { return normalStiffness_.hostRef(); }
    const std::vector<double>& shearStiffnessHostRef() const { return shearStiffness_.hostRef(); }
    const std::vector<double>& frictionCoefficientHostRef() const { return frictionCoefficient_.hostRef(); }
    const std::vector<double>& restitutionCoefficientHostRef() const { return restitutionCoefficient_.hostRef(); }

    spatialGrid spatialGrid_;

private:
    void copyHostToDevice(cudaStream_t stream)
    {
        position_.copyHostToDevice(stream);
        velocity_.copyHostToDevice(stream);
        angularVelocity_.copyHostToDevice(stream);
        force_.copyHostToDevice(stream);
        torque_.copyHostToDevice(stream);
        orientation_.copyHostToDevice(stream);
        radius_.copyHostToDevice(stream);
        inverseMass_.copyHostToDevice(stream);
        inverseInertiaTensor_.copyHostToDevice(stream);
        normalStiffness_.copyHostToDevice(stream);
        shearStiffness_.copyHostToDevice(stream);
        frictionCoefficient_.copyHostToDevice(stream);
        restitutionCoefficient_.copyHostToDevice(stream);
    }

    void copyDeviceToHost(cudaStream_t stream)
    {
        position_.copyDeviceToHost(stream);
        velocity_.copyDeviceToHost(stream);
        angularVelocity_.copyDeviceToHost(stream);
        force_.copyDeviceToHost(stream);
        torque_.copyDeviceToHost(stream);
        orientation_.copyDeviceToHost(stream);
    }

    HostDeviceArray1D<double3> position_;
    HostDeviceArray1D<double3> velocity_;
    HostDeviceArray1D<double3> angularVelocity_;
    HostDeviceArray1D<double3> force_;
    HostDeviceArray1D<double3> torque_;
    HostDeviceArray1D<quaternion> orientation_;
    HostDeviceArray1D<double> radius_;
    HostDeviceArray1D<double> inverseMass_;
    HostDeviceArray1D<symMatrix> inverseInertiaTensor_;
    HostDeviceArray1D<double> normalStiffness_;
    HostDeviceArray1D<double> shearStiffness_;
    HostDeviceArray1D<double> frictionCoefficient_;
    HostDeviceArray1D<double> restitutionCoefficient_;

    HostDeviceArray1D<int> hashValue_;
    HostDeviceArray1D<int> hashIndex_;

    size_t gridDim_{1};
    size_t blockDim_{1};

    bool upload_{false};
};