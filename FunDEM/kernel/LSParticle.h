#pragma once
#include "CUDAKernelFunction/myUtility/myHostDeviceArray.h"
#include "CUDAKernelFunction/myUtility/myMat.h"
#include "CUDAKernelFunction/myUtility/mySpatialGrid.h"
#include "CUDAKernelFunction/myUtility/myFileEdit.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>

struct LSGridNode
{
private:
    HostDeviceArray1D<double> signedDistanceField_;

public:
    void pushHost(const double signedDistanceField) { signedDistanceField_.pushHost(signedDistanceField); }

    void copyHostToDevice(cudaStream_t stream) { signedDistanceField_.copyHostToDevice(stream); }

    const size_t num() const { return signedDistanceField_.hostSize(); }
    const size_t num_device() const { return signedDistanceField_.deviceSize(); }
    double deviceMemoryGB() const
    {
        return signedDistanceField_.deviceMemoryGB();
    }

    double* signedDistanceField() { return signedDistanceField_.d_ptr; }

    const std::vector<double>& signedDistanceFieldHostRef() const { return signedDistanceField_.hostRef(); }

    void setHost(const std::vector<double>& signedDistanceFields) { signedDistanceField_.setHost(signedDistanceFields); }
};

struct LSBoundaryNode
{
private:
    HostDeviceArray1D<double3> localPosition_;
    HostDeviceArray1D<int> particleID_;

public:
    // p -- local position, id -- particle id
    void pushHost(const double3 localPosition, const int particleID) 
    { 
        localPosition_.pushHost(localPosition); 
        particleID_.pushHost(particleID);  
    }

    void copyHostToDevice(cudaStream_t stream) 
    { 
        localPosition_.copyHostToDevice(stream); 
        particleID_.copyHostToDevice(stream); 
    }

    const size_t num() const { return localPosition_.hostSize(); }
    const size_t num_device() const { return localPosition_.deviceSize(); }
    double deviceMemoryGB() const
    {
        return localPosition_.deviceMemoryGB() + particleID_.deviceMemoryGB();
    }
    double3* localPosition() { return localPosition_.d_ptr; }
    int* particleID() { return particleID_.d_ptr; }

    const std::vector<double3>& localPositionHostRef() const { return localPosition_.hostRef(); }
    const std::vector<int>& particleIDHostRef() const { return particleID_.hostRef(); }

    void setHost(const std::vector<double3>& localPositions, const std::vector<int>& particleIDs) 
    {
        localPosition_.setHost(localPositions); 
        particleID_.setHost(particleIDs); 
    }
};

struct LSParticle
{
public:
    LSParticle() = default;
    ~LSParticle() = default;

    LSParticle(const LSParticle&) = delete;
    LSParticle& operator=(const LSParticle&) = delete;

    LSParticle(LSParticle&&) noexcept = default;
    LSParticle& operator=(LSParticle&&) noexcept = default;

    /**
    * @brief Append one level-set particle to host-side buffers.
    *
    * This function takes a level-set particle representation defined in its LOCAL frame:
    * - boundary nodes (typically phi=0 surface samples) in local coordinates
    * - a regular background grid with per-node level-set values (Signed Distance Field, SDF)
    * - grid metadata: local origin, node counts, spacing
    *
    * @param[in] boundaryNodeLocalPosition      Boundary node positions in the particle LOCAL frame.
    * @param[in] boundaryNodeConnectivity       Optional: particle faces (triangles).

    * @param[in] gridNodeSignedDistance         Signed distance on the particle background grid.
    *                                           Indexing: ix + nx*(iy + ny*iz).
    * @param[in] gridNodeLocalOrigin            LOCAL coordinate of grid node (0,0,0).
    * @param[in] gridNodeSize                   Grid resolution (nx, ny, nz). Must be >= (2,2,2).
    * @param[in] gridNodeSpacing                Grid spacing (must be > 0).
    *
    * @param[in] position                       Particle center position in WORLD frame (will be shifted by rotated localCenter).
    * @param[in] velocity                       Particle linear velocity in WORLD frame.
    * @param[in] angularVelocity                Particle angular velocity in WORLD frame.
    * @param[in] orientation                    Particle orientation (LOCAL -> WORLD).
    * @param[in] inverseMass                    Particle inverse mass.
    * @param[in] inverseInertiaTensor           Particle inverse inertia tensor (LOCAL).
    * @param[in] normalStiffness                Normal stiffness for this particle.
    * @param[in] shearStiffness                 Shear stiffness for this particle.
    * @param[in] frictionCoefficient            Friction coefficient for this particle.
    * @param[in] restitutionCoefficient         Restitution coefficient for this particle.
    */
    void add(const std::vector<double3>& boundaryNodeLocalPosition,
    const std::vector<int3>& boundaryNodeConnectivity,
    
    const std::vector<double>& gridNodeSignedDistance,
    const double3 gridNodeLocalOrigin,
    const int3 gridNodeSize,
    const double gridNodeSpacing,

    const double3 position,
    const double3 velocity,
    const double3 angularVelocity,
    const quaternion orientation,
    const double inverseMass, 
    const symMatrix inverseInertiaTensor,  
    const double normalStiffness,
    const double shearStiffness,
    const double frictionCoefficient,
    const double restitutionCoefficient)
    {
        if (gridNodeSize.x < 2 || gridNodeSize.y < 2 || gridNodeSize.z < 2)
        {
            std::cerr << "[LSParticle] Invalid level-set host grid size: ("
                    << gridNodeSize.x << ", "
                    << gridNodeSize.y << ", "
                    << gridNodeSize.z << ")."
                    << std::endl;
            return;
        }

        if (gridNodeSpacing <= 0.0)
        {
            std::cerr << "[LSParticle] Invalid level-set host grid spacing: "
                    << gridNodeSpacing << "."
                    << std::endl;
            return;
        }

        const size_t expectedNumGridNodes = size_t(gridNodeSize.x) * size_t(gridNodeSize.y) * size_t(gridNodeSize.z);
        if (expectedNumGridNodes != gridNodeSignedDistance.size())
        {
            std::cerr << "[LSParticle] Inconsistent level-set host data size. "
                    << "Expected "
                    << expectedNumGridNodes
                    << ", got "
                    << gridNodeSignedDistance.size()
                    << "."
                    << std::endl;
            return;
        }

        if (boundaryNodeConnectivity.size() > 0)
        {
            const int n = static_cast<int>(LSBoundaryNode_.num());
            for (const auto& p:boundaryNodeConnectivity) 
            {
                const int3 p1 = make_int3(p.x + n, p.y + n, p.z + n);
                LSBoundaryNodeConnectivity_.push_back(p1);
            }
        }

        double radius = 0.;
        const int particleID = static_cast<int>(num());
        for (const auto& p : boundaryNodeLocalPosition)
        {
            LSBoundaryNode_.pushHost(p, particleID);
            radius = std::max(length(p), radius);
        }

        for (const auto& v : gridNodeSignedDistance)
        {
            LSGridNode_.pushHost(v);
        }
        
        position_.pushHost(position);
        velocity_.pushHost(velocity);
        angularVelocity_.pushHost(angularVelocity);
        force_.pushHost(make_double3(0., 0., 0.));
        torque_.pushHost(make_double3(0., 0., 0.));
        orientation_.pushHost(normalize(orientation));
        radius_.pushHost(radius);
        inverseMass_.pushHost(inverseMass);
        inverseInertiaTensor_.pushHost(inverseInertiaTensor);
        normalStiffness_.pushHost(normalStiffness > 0. ? normalStiffness : 0.);
        shearStiffness_.pushHost(shearStiffness > 0. ? shearStiffness : 0.);
        frictionCoefficient_.pushHost(frictionCoefficient > 0. ? frictionCoefficient : 0.);
        double e = restitutionCoefficient;
        if (e <= 0.0 || e > 1.0) e = 1.0;
        restitutionCoefficient_.pushHost(e);

        gridNodeLocalOrigin_.pushHost(gridNodeLocalOrigin);
        inverseGridNodeSpacing_.pushHost(1. / gridNodeSpacing);
        gridNodeSize_.pushHost(gridNodeSize);
        const int gridNodePrefixSum = static_cast<int>(LSGridNode_.num());
        gridNodePrefixSum_.pushHost(gridNodePrefixSum);

        hashValue_.pushHost(-1);
        hashIndex_.pushHost(-1);
    }
    
    void move(const size_t index, const double3 offset)
    {
        if (index >= num()) return;

        std::vector<double3> pos = position_.hostRef();
        pos[index] += offset;
        position_.setHost(pos);
    }

    void setVelocity(const size_t index, const double3 velocity)
    {
        if (index >= num()) return;

        std::vector<double3> vel = velocity_.hostRef();
        vel[index] = velocity;
        velocity_.setHost(vel);
    }

    void setAngularVelocity(const size_t index, const double3 angularVelocity)
    {
        if (index >= num()) return;

        std::vector<double3> ang = angularVelocity_.hostRef();
        ang[index] = angularVelocity;
        angularVelocity_.setHost(ang);
    }

    void initialize(const double3 minDomain, const double3 maxDomain, const size_t maxGPUThread, cudaStream_t stream)
    {
        double cellSizeOneDim = 0.;
        const size_t numParticle = position_.hostSize();
        if (numParticle > 0) 
        {
            copyHostToDevice(stream);

            if (maxGPUThread > 0) blockDim_ = maxGPUThread;
            if (numParticle < maxGPUThread) blockDim_ = numParticle;
            gridDim_ = (numParticle + blockDim_ - 1) / blockDim_;

            cellSizeOneDim = *std::max_element(radius_.hostRef().begin(), radius_.hostRef().end()) * 2.0;
        }
        spatialGrid_.set(minDomain, maxDomain, cellSizeOneDim, stream);
    }

    double deviceMemoryGB() const
    {
        double total = 0.0;

        // Particle state
        total += position_.deviceMemoryGB();
        total += velocity_.deviceMemoryGB();
        total += angularVelocity_.deviceMemoryGB();
        total += force_.deviceMemoryGB();
        total += torque_.deviceMemoryGB();
        total += orientation_.deviceMemoryGB();
        total += radius_.deviceMemoryGB();
        total += inverseMass_.deviceMemoryGB();
        total += inverseInertiaTensor_.deviceMemoryGB();
        total += normalStiffness_.deviceMemoryGB();
        total += shearStiffness_.deviceMemoryGB();
        total += frictionCoefficient_.deviceMemoryGB();
        total += restitutionCoefficient_.deviceMemoryGB();

        // Grid metadata
        total += gridNodeLocalOrigin_.deviceMemoryGB();
        total += inverseGridNodeSpacing_.deviceMemoryGB();
        total += gridNodeSize_.deviceMemoryGB();
        total += gridNodePrefixSum_.deviceMemoryGB();

        // LS grid node SDF values
        total += LSGridNode_.deviceMemoryGB();

        // LS boundary nodes
        total += LSBoundaryNode_.deviceMemoryGB();

        // Hash
        total += hashValue_.deviceMemoryGB();
        total += hashIndex_.deviceMemoryGB();

        return total;
    }

    /**
    * @brief Export Level Set boundary mesh to VTU (inline base64 binary format)
    *
    * Output:
    * - Points (world position)
    * - Cells (triangle connectivity)
    * - PointData: particleID, velocity
    *
    * @param dir     Output directory
    * @param iFrame  Frame index
    * @param iStep   Simulation step
    * @param time    Simulation time
    *
    * @note Inline base64 binary VTU format (UInt64 header, LittleEndian)
    */
    void outputVTU(const std::string& dir, const size_t iFrame, const size_t iStep, const double time) const
    {
        if (num() == 0) return;

        MKDIR(dir.c_str());

        std::ostringstream fname;
        fname << dir << "/LSObjectMesh_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";

        const size_t N = LSBoundaryNode_.num();
        const size_t M = LSBoundaryNodeConnectivity_.size();

        const std::vector<int>&        pID    = LSBoundaryNode_.particleIDHostRef();
        const std::vector<double3>&    pLocal = LSBoundaryNode_.localPositionHostRef();

        const std::vector<double3>&    p_p = position_.hostRef();
        const std::vector<double3>&    v_p = velocity_.hostRef();
        const std::vector<double3>&    w_p = angularVelocity_.hostRef();
        const std::vector<quaternion>& q_p = orientation_.hostRef();

        // -------------------------------------------------------------------------
        // Precompute arrays
        // -------------------------------------------------------------------------
        std::vector<float>   points(3 * N);
        std::vector<int32_t> particleID(N);
        std::vector<float>   velocity(3 * N);

        for (size_t i = 0; i < N; ++i)
        {
            const int p = pID[i];

            double3 pw = make_double3(0.0, 0.0, 0.0);
            double3 vw = make_double3(0.0, 0.0, 0.0);

            if (p >= 0 && static_cast<size_t>(p) < p_p.size())
            {
                pw = p_p[p] + rotateVectorByQuaternion(q_p[p], pLocal[i]);

                if (static_cast<size_t>(p) < v_p.size() &&
                    static_cast<size_t>(p) < w_p.size())
                {
                    vw = v_p[p] + cross(w_p[p], pw - p_p[p]);
                }
            }

            points[3 * i + 0] = static_cast<float>(pw.x);
            points[3 * i + 1] = static_cast<float>(pw.y);
            points[3 * i + 2] = static_cast<float>(pw.z);

            velocity[3 * i + 0] = static_cast<float>(vw.x);
            velocity[3 * i + 1] = static_cast<float>(vw.y);
            velocity[3 * i + 2] = static_cast<float>(vw.z);

            particleID[i] = static_cast<int32_t>(p);
        }

        // -------------------------------------------------------------------------
        // Connectivity
        // -------------------------------------------------------------------------
        std::vector<int32_t> connectivity(3 * M);
        std::vector<int32_t> offsets(M);
        std::vector<uint8_t> types(M, 5); // VTK_TRIANGLE = 5

        for (size_t i = 0; i < M; ++i)
        {
            const int3 tri = LSBoundaryNodeConnectivity_[i];

            connectivity[3 * i + 0] = tri.x;
            connectivity[3 * i + 1] = tri.y;
            connectivity[3 * i + 2] = tri.z;

            offsets[i] = static_cast<int32_t>(3 * (i + 1));
        }

        // -------------------------------------------------------------------------
        // Open file (text mode — XML only, no raw binary)
        // -------------------------------------------------------------------------
        std::ofstream out(fname.str());
        if (!out)
            throw std::runtime_error("Cannot open for writing: " + fname.str());

        // -------------------------------------------------------------------------
        // base64 encoder
        // Inline binary format: each DataArray = base64( [uint64_t byteCount][data] )
        // -------------------------------------------------------------------------
        static const char kB64Table[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        auto toB64 = [&](const void* data, size_t nBytes)
        {
            const uint64_t sz = static_cast<uint64_t>(nBytes);
            std::vector<uint8_t> buf(sizeof(uint64_t) + nBytes);
            std::memcpy(buf.data(),                    &sz,  sizeof(uint64_t));
            std::memcpy(buf.data() + sizeof(uint64_t), data, nBytes);

            const uint8_t* in  = buf.data();
            const size_t   len = buf.size();

            for (size_t i = 0; i < len; i += 3)
            {
                const uint32_t b0 = in[i];
                const uint32_t b1 = (i + 1 < len) ? in[i + 1] : 0u;
                const uint32_t b2 = (i + 2 < len) ? in[i + 2] : 0u;
                const uint32_t v  = (b0 << 16) | (b1 << 8) | b2;

                out.put(kB64Table[(v >> 18) & 0x3F]);
                out.put(kB64Table[(v >> 12) & 0x3F]);
                out.put((i + 1 < len) ? kB64Table[(v >> 6) & 0x3F] : '=');
                out.put((i + 2 < len) ? kB64Table[(v     ) & 0x3F] : '=');
            }
        };

        // -------------------------------------------------------------------------
        // XML
        // -------------------------------------------------------------------------
        out << "<?xml version=\"1.0\"?>\n"
            << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
            "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
            << "<UnstructuredGrid>\n";

        // FieldData
        out << "<FieldData>\n"
            << "<DataArray type=\"Float32\" Name=\"TIME\" "
            "NumberOfTuples=\"1\" format=\"ascii\">"
            << static_cast<float>(time)
            << "</DataArray>\n"
            << "<DataArray type=\"Int32\" Name=\"STEP\" "
            "NumberOfTuples=\"1\" format=\"ascii\">"
            << static_cast<int32_t>(iStep)
            << "</DataArray>\n"
            << "</FieldData>\n";

        out << "<Piece NumberOfPoints=\"" << N
            << "\" NumberOfCells=\""      << M << "\">\n";

        // Points
        out << "<Points>\n"
            << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"binary\">\n";
        toB64(points.data(), points.size() * sizeof(float));
        out << "\n</DataArray>\n"
            << "</Points>\n";

        // Cells
        out << "<Cells>\n";

        out << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"binary\">\n";
        toB64(connectivity.data(), connectivity.size() * sizeof(int32_t));
        out << "\n</DataArray>\n";

        out << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"binary\">\n";
        toB64(offsets.data(), offsets.size() * sizeof(int32_t));
        out << "\n</DataArray>\n";

        out << "<DataArray type=\"UInt8\" Name=\"types\" format=\"binary\">\n";
        toB64(types.data(), types.size() * sizeof(uint8_t));
        out << "\n</DataArray>\n";

        out << "</Cells>\n";

        // PointData
        out << "<PointData Vectors=\"velocity\">\n";

        out << "<DataArray type=\"Int32\" Name=\"particleID\" format=\"binary\">\n";
        toB64(particleID.data(), particleID.size() * sizeof(int32_t));
        out << "\n</DataArray>\n";

        out << "<DataArray type=\"Float32\" Name=\"velocity\" "
            "NumberOfComponents=\"3\" format=\"binary\">\n";
        toB64(velocity.data(), velocity.size() * sizeof(float));
        out << "\n</DataArray>\n";

        out << "</PointData>\n"
            << "</Piece>\n"
            << "</UnstructuredGrid>\n"
            << "</VTKFile>\n";

        if (!out)
            throw std::runtime_error("Write error on: " + fname.str());
    }

    /**
    * @brief Export particle state to VTU (inline base64 binary format)
    *
    * Output:
    * - Points: particle position
    * - PointData:
    *   - velocity
    *   - angularVelocity
    *   - orientation (quaternion as vec4)
    *
    * @param dir     Output directory
    * @param iFrame  Frame index
    * @param iStep   Simulation step
    * @param time    Simulation time
    *
    * @note Inline base64 binary VTU format (UInt64 header, LittleEndian)
    */
    void outputParticleVTU(const std::string& dir, const size_t iFrame, const size_t iStep, const double time) const
    {
        const size_t N = num();
        if (N == 0) return;

        MKDIR(dir.c_str());

        std::ostringstream fname;
        fname << dir << "/Particle_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";

        const std::vector<double3>&    p = position_.hostRef();
        const std::vector<double3>&    v = velocity_.hostRef();
        const std::vector<double3>&    w = angularVelocity_.hostRef();
        const std::vector<quaternion>& q = orientation_.hostRef();
        const std::vector<double>&     r = radius_.hostRef();

        // -------------------------------------------------------------------------
        // Precompute arrays
        // -------------------------------------------------------------------------
        std::vector<float> points(3 * N);
        std::vector<float> vel(3 * N);
        std::vector<float> angVel(3 * N);
        std::vector<float> ori(4 * N);
        std::vector<float> radius(N);

        for (size_t i = 0; i < N; ++i)
        {
            points[3 * i + 0] = static_cast<float>(p[i].x);
            points[3 * i + 1] = static_cast<float>(p[i].y);
            points[3 * i + 2] = static_cast<float>(p[i].z);

            vel[3 * i + 0] = static_cast<float>(v[i].x);
            vel[3 * i + 1] = static_cast<float>(v[i].y);
            vel[3 * i + 2] = static_cast<float>(v[i].z);

            angVel[3 * i + 0] = static_cast<float>(w[i].x);
            angVel[3 * i + 1] = static_cast<float>(w[i].y);
            angVel[3 * i + 2] = static_cast<float>(w[i].z);

            ori[4 * i + 0] = static_cast<float>(q[i].q0);
            ori[4 * i + 1] = static_cast<float>(q[i].q1);
            ori[4 * i + 2] = static_cast<float>(q[i].q2);
            ori[4 * i + 3] = static_cast<float>(q[i].q3);

            radius[i] = static_cast<float>(r[i]);
        }

        // -------------------------------------------------------------------------
        // Open file
        // -------------------------------------------------------------------------
        std::ofstream out(fname.str());
        if (!out)
            throw std::runtime_error("Cannot open for writing: " + fname.str());

        // -------------------------------------------------------------------------
        // base64 encoder
        // -------------------------------------------------------------------------
        static const char kB64Table[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        auto toB64 = [&](const void* data, size_t nBytes)
        {
            const uint64_t sz = static_cast<uint64_t>(nBytes);
            std::vector<uint8_t> buf(sizeof(uint64_t) + nBytes);
            std::memcpy(buf.data(),                    &sz,  sizeof(uint64_t));
            std::memcpy(buf.data() + sizeof(uint64_t), data, nBytes);

            const uint8_t* in  = buf.data();
            const size_t   len = buf.size();

            for (size_t i = 0; i < len; i += 3)
            {
                const uint32_t b0 = in[i];
                const uint32_t b1 = (i + 1 < len) ? in[i + 1] : 0u;
                const uint32_t b2 = (i + 2 < len) ? in[i + 2] : 0u;
                const uint32_t v  = (b0 << 16) | (b1 << 8) | b2;

                out.put(kB64Table[(v >> 18) & 0x3F]);
                out.put(kB64Table[(v >> 12) & 0x3F]);
                out.put((i + 1 < len) ? kB64Table[(v >> 6) & 0x3F] : '=');
                out.put((i + 2 < len) ? kB64Table[(v     ) & 0x3F] : '=');
            }
        };

        // -------------------------------------------------------------------------
        // XML
        // -------------------------------------------------------------------------
        out << "<?xml version=\"1.0\"?>\n"
            << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
            "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
            << "<UnstructuredGrid>\n";

        // FieldData
        out << "<FieldData>\n"
            << "<DataArray type=\"Float32\" Name=\"TIME\" "
            "NumberOfTuples=\"1\" format=\"ascii\">"
            << static_cast<float>(time)
            << "</DataArray>\n"
            << "<DataArray type=\"Int32\" Name=\"STEP\" "
            "NumberOfTuples=\"1\" format=\"ascii\">"
            << static_cast<int32_t>(iStep)
            << "</DataArray>\n"
            << "</FieldData>\n";

        out << "<Piece NumberOfPoints=\"" << N << "\" NumberOfCells=\"0\">\n";

        // Points
        out << "<Points>\n"
            << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"binary\">\n";
        toB64(points.data(), points.size() * sizeof(float));
        out << "\n</DataArray>\n"
            << "</Points>\n";

        // Cells (empty)
        out << "<Cells/>\n";

        // PointData
        out << "<PointData Vectors=\"velocity\">\n";

        out << "<DataArray type=\"Float32\" Name=\"velocity\" "
            "NumberOfComponents=\"3\" format=\"binary\">\n";
        toB64(vel.data(), vel.size() * sizeof(float));
        out << "\n</DataArray>\n";

        out << "<DataArray type=\"Float32\" Name=\"angularVelocity\" "
            "NumberOfComponents=\"3\" format=\"binary\">\n";
        toB64(angVel.data(), angVel.size() * sizeof(float));
        out << "\n</DataArray>\n";

        out << "<DataArray type=\"Float32\" Name=\"orientation\" "
            "NumberOfComponents=\"4\" format=\"binary\">\n";
        toB64(ori.data(), ori.size() * sizeof(float));
        out << "\n</DataArray>\n";

        out << "<DataArray type=\"Float32\" Name=\"radius\" "
        "NumberOfComponents=\"1\" format=\"binary\">\n";
        toB64(radius.data(), radius.size() * sizeof(float));
        out << "\n</DataArray>\n";

        out << "</PointData>\n"
            << "</Piece>\n"
            << "</UnstructuredGrid>\n"
            << "</VTKFile>\n";

        if (!out)
            throw std::runtime_error("Write error on: " + fname.str());
    }

    void finalize(cudaStream_t stream)
    {
        copyDeviceToHost(stream);
    }

    size_t num() const { return position_.hostSize(); }
    size_t num_device() const { return position_.deviceSize(); }
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
    double3* gridNodeLocalOrigin() { return gridNodeLocalOrigin_.d_ptr; }
    double* inverseGridNodeSpacing() { return inverseGridNodeSpacing_.d_ptr; }
    int3* gridNodeSize() { return gridNodeSize_.d_ptr; }
    int* gridNodePrefixSum() { return gridNodePrefixSum_.d_ptr; }
    
    int* hashValue() { return hashValue_.d_ptr; }
    int* hashIndex() { return hashIndex_.d_ptr; }

    const std::vector<double3>& positionHostRef() const { return position_.hostRef(); }
    const std::vector<double3>& velocityHostRef() { return velocity_.hostRef(); }
    const std::vector<double3>& angularVelocityHostRef() { return angularVelocity_.hostRef(); }
    const std::vector<double3>& forceHostRef() { return force_.hostRef(); }
    const std::vector<double3>& torqueHostRef() { return torque_.hostRef(); }
    const std::vector<quaternion>& orientationHostRef() const { return orientation_.hostRef(); }

    LSGridNode LSGridNode_;
    LSBoundaryNode LSBoundaryNode_;
    spatialGrid spatialGrid_;

private:
    void copyHostToDevice(cudaStream_t stream)
    {
        // Particles
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

        gridNodeLocalOrigin_.copyHostToDevice(stream);
        inverseGridNodeSpacing_.copyHostToDevice(stream);
        gridNodeSize_.copyHostToDevice(stream);
        gridNodePrefixSum_.copyHostToDevice(stream);

        // Grid node field values (all particles packed)
        LSGridNode_.copyHostToDevice(stream);

        // Boundary nodes
        LSBoundaryNode_.copyHostToDevice(stream);

        // Hash
        hashValue_.copyHostToDevice(stream);
        hashIndex_.copyHostToDevice(stream);
    }

    void copyDeviceToHost(cudaStream_t stream)
    {
        position_.copyDeviceToHost(stream);
        velocity_.copyDeviceToHost(stream);
        angularVelocity_.copyDeviceToHost(stream);
        force_.copyDeviceToHost(stream);
        torque_.copyDeviceToHost(stream);
        orientation_.copyDeviceToHost(stream);
        
        hashValue_.copyDeviceToHost(stream);
        hashIndex_.copyDeviceToHost(stream);
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
    HostDeviceArray1D<double3> gridNodeLocalOrigin_;
    HostDeviceArray1D<double> inverseGridNodeSpacing_;
    HostDeviceArray1D<int3> gridNodeSize_;
    HostDeviceArray1D<int> gridNodePrefixSum_;

    HostDeviceArray1D<int> hashValue_;
    HostDeviceArray1D<int> hashIndex_;

    size_t gridDim_{1};
    size_t blockDim_{1};

    std::vector<int3> LSBoundaryNodeConnectivity_;
};