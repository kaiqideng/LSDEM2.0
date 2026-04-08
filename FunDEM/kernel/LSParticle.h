#pragma once
#include "CUDAKernelFunction/myUtility/myHostDeviceArray.h"
#include "CUDAKernelFunction/myUtility/myMat.h"
#include "CUDAKernelFunction/myUtility/mySpatialGrid.h"
#include "CUDAKernelFunction/myUtility/myFileEdit.h"
#include <algorithm>
#include <cstdint>
#include <fstream>

struct LSGridNode
{
private:
    HostDeviceArray1D<double> levelSetFunctionValue_;

public:
    void pushHost(const double levelSetFunctionValue) { levelSetFunctionValue_.pushHost(levelSetFunctionValue); }

    void copyHostToDevice(cudaStream_t stream) { levelSetFunctionValue_.copyHostToDevice(stream); }

    const size_t num() const { return levelSetFunctionValue_.hostSize(); }

    double* levelSetFunctionValue() { return levelSetFunctionValue_.d_ptr; }

    const std::vector<double>& levelSetFunctionValueHostRef() const { return levelSetFunctionValue_.hostRef(); }

    void setHost(const std::vector<double>& levelSetFunctionValues) { levelSetFunctionValue_.setHost(levelSetFunctionValues); }
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
    * - a regular background grid with per-node level-set values (LSF)
    * - grid metadata: local origin, node counts, spacing
    *
    * Then it computes basic rigid-body properties from the grid:
    * - mass via a smoothed Heaviside integration of the level-set field
    * - center of mass (localCenter) from the same Heaviside weights
    * - inertia tensor around the center of mass, then inverse inertia
    *
    * Finally it appends:
    * - boundary nodes shifted by -localCenter (so boundary nodes become COM-centered)
    * - grid node LSF values (flat array, size = gridNodeSize.x*gridNodeSize.y*gridNodeSize.z)
    * - particle state (position/velocity/omega/orientation/material, etc.)
    *
    * Notes / assumptions:
    * - Here the Heaviside assumes "inside is negative" (phi < 0 means inside),
    *   because smoothHeaviside(phi/gridNodeSpacing, ...) returns ~1 for negative phi.
    * - If your convention is opposite, you must flip the sign passed into smoothHeaviside.
    * - This function modifies host arrays. If previous data has been uploaded (upload_==true),
    *   it first downloads device -> host to keep host-side buffers consistent.
    *
    * @param[in] boundaryNodeLocalPosition      Boundary node positions in the particle LOCAL frame.
    * @param[in] boundaryNodeConnectivity       Optional: particle faces (triangles).

    * @param[in] gridNodeLevelSetFunctionValue  Flattened LSF values on the particle background grid.
    *                                           Indexing: ix + nx*(iy + ny*iz).
    * @param[in] gridNodeLocalOrigin            LOCAL coordinate of grid node (0,0,0).
    * @param[in] gridNodeSize                   Grid resolution (nx, ny, nz). Must be >= (2,2,2).
    * @param[in] gridNodeSpacing                Grid spacing (must be > 0).
    *
    * @param[in] position                       Particle center position in WORLD frame (will be shifted by rotated localCenter).
    * @param[in] velocity                       Particle linear velocity in WORLD frame.
    * @param[in] angularVelocity                Particle angular velocity in WORLD frame.
    * @param[in] orientation                    Particle orientation (LOCAL -> WORLD).
    * @param[in] normalStiffness                Normal stiffness for this particle.
    * @param[in] shearStiffness                 Shear stiffness for this particle.
    * @param[in] frictionCoefficient            Friction coefficient for this particle.
    *
    * @param[in] density                        Material density used for grid integration.
    * @param[in] stream                         CUDA stream (used only if we must download device -> host).
    */
    void add(const std::vector<double3>& boundaryNodeLocalPosition,
    const std::vector<int3>& boundaryNodeConnectivity,
    
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

    const double density,
    cudaStream_t stream)
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
        if (expectedNumGridNodes != gridNodeLevelSetFunctionValue.size())
        {
            std::cerr << "[LSParticle] Inconsistent level-set host data size. "
                    << "Expected "
                    << expectedNumGridNodes
                    << ", got "
                    << gridNodeLevelSetFunctionValue.size()
                    << "."
                    << std::endl;
            return;
        }

        if (upload_)
        {
            copyDeviceToHost(stream);
            upload_ = false;
        }

        double mass = 0.;

        auto smoothHeaviside = [&](const double phi_dimensionless, const double smoothParameter) -> double
        {
            if (smoothParameter <= 0.0) return (phi_dimensionless > 0.0) ? 0.0 : 1.0;

            if (phi_dimensionless < -smoothParameter) return 1.0;
            if (phi_dimensionless > smoothParameter) return 0.0;

            const double x = -phi_dimensionless / smoothParameter;
            return 0.5 * (1.0 + x + std::sin(M_PI * x) / M_PI);
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

        double invM = 0.;
        double3 localCenter = make_double3(0., 0., 0.);
        symMatrix I = make_symMatrix(0., 0., 0., 0., 0., 0.);
        symMatrix invI = make_symMatrix(0., 0., 0., 0., 0., 0.);
        if (mass > 0.)
        {
            invM = 1. / mass;
            for (int x = 0; x < gridNodeSize.x; x++)
            {
                for (int y = 0; y < gridNodeSize.y; y++)
                {
                    for (int z = 0; z < gridNodeSize.z; z++)
                    {
                        const int index = linearIndex3D(make_int3(x, y, z), gridNodeSize);
                        const double H = smoothHeaviside(gridNodeLevelSetFunctionValue[index] / gridNodeSpacing, 1.5);
                        localCenter.x += H * (gridNodeLocalOrigin.x + double(x) * gridNodeSpacing);
                        localCenter.y += H * (gridNodeLocalOrigin.y + double(y) * gridNodeSpacing);
                        localCenter.z += H * (gridNodeLocalOrigin.z + double(z) * gridNodeSpacing);
                    }
                }
            }
            localCenter *= m_gridNode / mass;

            for (int x = 0; x < gridNodeSize.x; x++)
            {
                for (int y = 0; y < gridNodeSize.y; y++)
                {
                    for (int z = 0; z < gridNodeSize.z; z++)
                    {
                        const int index = linearIndex3D(make_int3(x, y, z), gridNodeSize);
                        const double H = smoothHeaviside(gridNodeLevelSetFunctionValue[index] / gridNodeSpacing, 1.5);
                        double3 r = gridNodeLocalOrigin + gridNodeSpacing * make_double3(double(x), double(y), double(z)) - localCenter;
                        I.xx += H * (r.y * r.y + r.z * r.z) * m_gridNode;
                        I.yy += H * (r.x * r.x + r.z * r.z) * m_gridNode;
                        I.zz += H * (r.y * r.y + r.x * r.x) * m_gridNode;
                        I.xy -= H * r.x * r.y * m_gridNode;
                        I.xz -= H * r.x * r.z * m_gridNode;
                        I.yz -= H * r.y * r.z * m_gridNode;
                    }
                }
            }
            invI = inverse(I);
        }

        if (boundaryNodeConnectivity.size() > 0)
        {
            const int n = int(LSBoundaryNode_.num());
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
            LSBoundaryNode_.pushHost(p - localCenter, particleID);
            radius = std::max(length(p - localCenter), radius);
        }

        for (const auto& v : gridNodeLevelSetFunctionValue)
        {
            LSGridNode_.pushHost(v);
        }

        position_.pushHost(position + rotateVectorByQuaternion(orientation, localCenter));
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

        gridNodeLocalOrigin_.pushHost(gridNodeLocalOrigin - localCenter);
        inverseGridNodeSpacing_.pushHost(1. / gridNodeSpacing);
        gridNodeSize_.pushHost(gridNodeSize);
        const int gridNodePrefixSum = static_cast<int>(LSGridNode_.num());
        gridNodePrefixSum_.pushHost(gridNodePrefixSum);
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
        fname << dir << "/LSObject_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";

        const size_t N = LSBoundaryNode_.num();

        const std::vector<int> &pID = LSBoundaryNode_.particleIDHostRef();
        const std::vector<double3> &pLocal = LSBoundaryNode_.localPositionHostRef();

        const std::vector<double3> &p_p = position_.hostRef();
        const std::vector<double3> &v_p = velocity_.hostRef();
        const std::vector<double3> &w_p = angularVelocity_.hostRef();
        const std::vector<quaternion> &q_p = orientation_.hostRef();

        std::vector<float> points;
        std::vector<float> vel;
        std::vector<int32_t> pid;
        std::vector<int32_t> conn;
        std::vector<int32_t> offs;
        std::vector<uint8_t> types;

        points.resize(N * 3);
        vel.resize(N * 3);
        pid.resize(N);
        conn.resize(N);
        offs.resize(N);
        types.resize(N);

        for (size_t i = 0; i < N; ++i)
        {
            const int p = pID[i];
            if (p < 0 || static_cast<size_t>(p) >= p_p.size())
            {
                points[3 * i + 0] = 0.f;
                points[3 * i + 1] = 0.f;
                points[3 * i + 2] = 0.f;

                vel[3 * i + 0] = 0.f;
                vel[3 * i + 1] = 0.f;
                vel[3 * i + 2] = 0.f;

                pid[i] = -1;
            }
            else
            {
                const double3 pw = p_p[p] + rotateVectorByQuaternion(q_p[p], pLocal[i]);
                const double3 vw = v_p[p] + cross(w_p[p], pw - p_p[p]);

                points[3 * i + 0] = static_cast<float>(pw.x);
                points[3 * i + 1] = static_cast<float>(pw.y);
                points[3 * i + 2] = static_cast<float>(pw.z);

                vel[3 * i + 0] = static_cast<float>(vw.x);
                vel[3 * i + 1] = static_cast<float>(vw.y);
                vel[3 * i + 2] = static_cast<float>(vw.z);

                pid[i] = static_cast<int32_t>(p);
            }

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

        size_t off_pid = off_types + blockBytes(types.size() * sizeof(uint8_t));
        size_t off_vel = off_pid + blockBytes(pid.size() * sizeof(int32_t));

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
            << "        <DataArray type=\"Int32\" Name=\"particleID\" format=\"appended\" offset=\"" << off_pid << "\"/>\n"
            << "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_vel << "\"/>\n"
            << "      </PointData>\n"
            << "    </Piece>\n"
            << "  </UnstructuredGrid>\n"
            << "  <AppendedData encoding=\"raw\">\n"
            << "    _";

        auto writeBlock = [&](const void *data, size_t nbytes)
        {
            const uint64_t sz = static_cast<uint64_t>(nbytes);
            out.write(reinterpret_cast<const char *>(&sz), sizeof(uint64_t));
            if (nbytes) out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(nbytes));

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
        writeBlock(pid.data(), pid.size() * sizeof(int32_t));
        writeBlock(vel.data(), vel.size() * sizeof(float));

        out << "\n  </AppendedData>\n</VTKFile>\n";
    }

    void outputVTU_connectivity(const std::string& dir, const size_t iFrame, const size_t iStep, const double time) const
    {
        if (num() == 0 || LSBoundaryNodeConnectivity_.empty()) return;

        MKDIR(dir.c_str());

        std::ostringstream fname;
        fname << dir << "/LSObjectMesh_"
            << std::setw(4) << std::setfill('0') << iFrame
            << ".vtu";

        const size_t N = LSBoundaryNode_.num();
        const size_t M = LSBoundaryNodeConnectivity_.size();

        const std::vector<int>& pID = LSBoundaryNode_.particleIDHostRef();
        const std::vector<double3>& pLocal = LSBoundaryNode_.localPositionHostRef();

        const std::vector<double3>& p_p = position_.hostRef();
        const std::vector<double3>& v_p = velocity_.hostRef();
        const std::vector<double3>& w_p = angularVelocity_.hostRef();
        const std::vector<quaternion>& q_p = orientation_.hostRef();

        std::ofstream out(fname.str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());

        out << "<?xml version=\"1.0\"?>\n";
        out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        out << "  <UnstructuredGrid>\n";
        out << "    <FieldData>\n";
        out << "      <DataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\"> "
            << static_cast<float>(time) << " </DataArray>\n";
        out << "      <DataArray type=\"Int32\" Name=\"STEP\" NumberOfTuples=\"1\" format=\"ascii\"> "
            << static_cast<int32_t>(iStep) << " </DataArray>\n";
        out << "    </FieldData>\n";
        out << "    <Piece NumberOfPoints=\"" << N
            << "\" NumberOfCells=\"" << M << "\">\n";

        // -------------------------------------------------------------------------
        // Points
        // -------------------------------------------------------------------------
        out << "      <Points>\n";
        out << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";

        for (size_t i = 0; i < N; ++i)
        {
            const int p = pID[i];

            double3 pw = make_double3(0.0, 0.0, 0.0);

            if (p >= 0 && static_cast<size_t>(p) < p_p.size())
            {
                pw = p_p[p] + rotateVectorByQuaternion(q_p[p], pLocal[i]);
            }

            out << "          "
                << static_cast<float>(pw.x) << " "
                << static_cast<float>(pw.y) << " "
                << static_cast<float>(pw.z) << "\n";
        }

        out << "        </DataArray>\n";
        out << "      </Points>\n";

        // -------------------------------------------------------------------------
        // Cells
        // -------------------------------------------------------------------------
        out << "      <Cells>\n";

        out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (size_t i = 0; i < M; ++i)
        {
            const int3 tri = LSBoundaryNodeConnectivity_[i];

            if (tri.x < 0 || tri.y < 0 || tri.z < 0 ||
                static_cast<size_t>(tri.x) >= N ||
                static_cast<size_t>(tri.y) >= N ||
                static_cast<size_t>(tri.z) >= N)
            {
                std::cerr << "[LSParticle] Invalid triangle index at triangle "
                        << i << std::endl;
                out << "          0 0 0\n";
                continue;
            }

            out << "          "
                << tri.x << " "
                << tri.y << " "
                << tri.z << "\n";
        }
        out << "        </DataArray>\n";

        out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (size_t i = 0; i < M; ++i)
        {
            out << "          " << static_cast<int32_t>(3 * (i + 1)) << "\n";
        }
        out << "        </DataArray>\n";

        out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (size_t i = 0; i < M; ++i)
        {
            out << "          5\n"; // VTK_TRIANGLE
        }
        out << "        </DataArray>\n";

        out << "      </Cells>\n";

        // -------------------------------------------------------------------------
        // Point data
        // -------------------------------------------------------------------------
        out << "      <PointData Vectors=\"velocity\">\n";

        out << "        <DataArray type=\"Int32\" Name=\"particleID\" format=\"ascii\">\n";
        for (size_t i = 0; i < N; ++i)
        {
            out << "          " << static_cast<int32_t>(pID[i]) << "\n";
        }
        out << "        </DataArray>\n";

        out << "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (size_t i = 0; i < N; ++i)
        {
            const int p = pID[i];

            double3 vw = make_double3(0.0, 0.0, 0.0);

            if (p >= 0 &&
                static_cast<size_t>(p) < p_p.size() &&
                static_cast<size_t>(p) < v_p.size() &&
                static_cast<size_t>(p) < w_p.size() &&
                static_cast<size_t>(p) < q_p.size())
            {
                const double3 pw = p_p[p] + rotateVectorByQuaternion(q_p[p], pLocal[i]);
                vw = v_p[p] + cross(w_p[p], pw - p_p[p]);
            }

            out << "          "
                << static_cast<float>(vw.x) << " "
                << static_cast<float>(vw.y) << " "
                << static_cast<float>(vw.z) << "\n";
        }
        out << "        </DataArray>\n";

        out << "      </PointData>\n";

        out << "    </Piece>\n";
        out << "  </UnstructuredGrid>\n";
        out << "</VTKFile>\n";
    }

    void finalize(cudaStream_t stream)
    {
        copyDeviceToHost(stream);
    }

    void copyFromHost(const LSParticle& other)
    {
        // ---- particle state ----
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

        // ---- per-particle level-set grid meta ----
        gridNodeLocalOrigin_.setHost(other.gridNodeLocalOrigin_.hostRef());
        inverseGridNodeSpacing_.setHost(other.inverseGridNodeSpacing_.hostRef());
        gridNodeSize_.setHost(other.gridNodeSize_.hostRef());
        gridNodePrefixSum_.setHost(other.gridNodePrefixSum_.hostRef());

        // ---- LSGridNode (phi packed) ----
        std::vector<double> phi = other.LSGridNode_.levelSetFunctionValueHostRef();
        LSGridNode_.setHost(phi); // requires friendship or a setter

        // ---- LSBoundaryNode (surface nodes + pid) ----
        std::vector<double3> lp = other.LSBoundaryNode_.localPositionHostRef();
        std::vector<int> pid = other.LSBoundaryNode_.particleIDHostRef();
        LSBoundaryNode_.setHost(lp, pid);

        upload_ = false; // host copy only; device pointers are not valid until you initialize()
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
    double3* gridNodeLocalOrigin() { return gridNodeLocalOrigin_.d_ptr; }
    double* inverseGridNodeSpacing() { return inverseGridNodeSpacing_.d_ptr; }
    int3* gridNodeSize() { return gridNodeSize_.d_ptr; }
    int* gridNodePrefixSum() { return gridNodePrefixSum_.d_ptr; }
    
    int* hashValue() { return hashValue_.d_ptr; }
    int* hashIndex() { return hashIndex_.d_ptr; }

    std::vector<double3> positionHostCopy() { return position_.getHostCopy(); }
    std::vector<double3> velocityHostCopy() { return velocity_.getHostCopy(); }
    std::vector<double3> angularVelocityHostCopy() { return angularVelocity_.getHostCopy(); }
    std::vector<double3> forceHostCopy() { return force_.getHostCopy(); }
    std::vector<double3> torqueHostCopy() { return torque_.getHostCopy(); }
    std::vector<quaternion> orientationHostCopy() { return orientation_.getHostCopy(); }

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

        gridNodeLocalOrigin_.copyHostToDevice(stream);
        inverseGridNodeSpacing_.copyHostToDevice(stream);
        gridNodeSize_.copyHostToDevice(stream);
        gridNodePrefixSum_.copyHostToDevice(stream);

        // Grid node field values (all particles packed)
        LSGridNode_.copyHostToDevice(stream);

        // Boundary nodes
        LSBoundaryNode_.copyHostToDevice(stream);
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
    HostDeviceArray1D<double3> gridNodeLocalOrigin_;
    HostDeviceArray1D<double> inverseGridNodeSpacing_;
    HostDeviceArray1D<int3> gridNodeSize_;
    HostDeviceArray1D<int> gridNodePrefixSum_;

    HostDeviceArray1D<int> hashValue_;
    HostDeviceArray1D<int> hashIndex_;

    size_t gridDim_{1};
    size_t blockDim_{1};

    bool upload_{false};

    std::vector<int3> LSBoundaryNodeConnectivity_;
};