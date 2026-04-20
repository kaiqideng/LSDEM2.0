#pragma once
#include "CUDAKernelFunction/myUtility/myHostDeviceArray.h"
#include "CUDAKernelFunction/myUtility/myQua.h"
#include "CUDAKernelFunction/myUtility/myFileEdit.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>

inline double3 getInitialOrthogonalUnitVectorsN2(const double3 n1)
{
    const double3 ref = (fabs(n1.x) < 0.9) ?
    make_double3(1., 0., 0.) : 
    make_double3(0., 1., 0.);

    return normalize(cross(ref, n1));
}

inline double3 getInitialOrthogonalUnitVectorsN3(const double3 n1, const double3 n2)
{
    return normalize(cross(n1, n2));
}

struct VBondPoint
{
    HostDeviceArray1D<double3> localVectorN1_;
    HostDeviceArray1D<double3> localVectorN2_;
    HostDeviceArray1D<double3> localVectorN3_;
    HostDeviceArray1D<double3> localPosition_;

    void pushHost(const double3 localN1, const double3 localN2, const double3 localN3, const double3 localPosition)
    {
        localVectorN1_.pushHost(normalize(localN1));
        localVectorN2_.pushHost(normalize(localN2));
        localVectorN3_.pushHost(normalize(localN3));
        localPosition_.pushHost(localPosition);
    }

    void copyHostToDevice(cudaStream_t stream)
    {
        localVectorN1_.copyHostToDevice(stream);
        localVectorN2_.copyHostToDevice(stream);
        localVectorN3_.copyHostToDevice(stream);
        localPosition_.copyHostToDevice(stream);
    }

    void setHost(const VBondPoint& other)
    {
        localVectorN1_.setHost(other.localVectorN1_.hostRef());
        localVectorN2_.setHost(other.localVectorN2_.hostRef());
        localVectorN3_.setHost(other.localVectorN3_.hostRef());
        localPosition_.setHost(other.localPosition_.hostRef());
    }
};

struct VBondedInteraction
{
public:
    VBondedInteraction() = default;
    ~VBondedInteraction() = default;

    VBondedInteraction(const VBondedInteraction&) = delete;
    VBondedInteraction& operator=(const VBondedInteraction&) = delete;

    VBondedInteraction(VBondedInteraction&&) noexcept = default;
    VBondedInteraction& operator=(VBondedInteraction&&) noexcept = default;

    void add(const int masterObjectID, 
    const int slaveObjectID, 
    const double3 masterObjectPosition, 
    const double3 slaveObjectPosition, 
    const quaternion masterObjectOrientation, 
    const quaternion slaveObjectOrientation, 
    const double3 globalCenterPoint, 
    const double3 globalNormal, 

    const double radius, 
    const double initialLength, 

    const double YoungsModulus, 
    const double poissonRatio, 
    const double tensileStrength, 
    const double cohesion, 
    const double frictionCoefficient)
    {
        if (isZero(lengthSquared(globalNormal)))
        {
            std::cerr << "[VBondedInteraction] Invalid bond normal: ("
            << globalNormal.x << ", "
            << globalNormal.y << ", "
            << globalNormal.z << ")."
            << std::endl;
            return;
        }

        if (radius <= 0.)
        {
            std::cerr << "[VBondedInteraction] Invalid bond radius: "
            << radius << "."
            << std::endl;
            return;
        }

        if (initialLength <= 0.)
        {
            std::cerr << "[VBondedInteraction] Invalid bond length: "
            << initialLength << "."
            << std::endl;
            return;
        }

        if (YoungsModulus <= 0.)
        {
            std::cerr << "[VBondedInteraction] Invalid Young's modulus: "
            << YoungsModulus << "." 
            << std::endl;
            return;
        }

        if (poissonRatio <= -1.0 || poissonRatio >= 0.5)
        {
            std::cerr << "[VBondedInteraction] Invalid Poisson ratio: "
                    << poissonRatio << "." << std::endl;
            return;
        }

        const double Rb = radius;
        const double Lb = initialLength;
        const double Eb = YoungsModulus;
        const double Gb = Eb / (2. * (1. + poissonRatio));
        const double Ab = pi() * Rb * Rb;
        const double Ib = Ab * Rb * Rb / 4.;
        const double Jb = 2. * Ib;
        const double Kn = Eb * Ab / Lb;
        const double Kt = Gb * Jb / Lb;
        const double Ks = 12. * Eb * Ib / (Lb * Lb * Lb);
        const double Kb = Eb * Ib / Lb;

        const double B1 = Kn;
        const double B2 = Lb * Lb * Ks;
        const double B4 = Kt;
        const double B3 = Kb - B2 / 4. - B4 / 2.;

        const double3 n = globalNormal / length(globalNormal);
        const double3 globalPosition_i = globalCenterPoint + 0.5 * Lb * n;
        const double3 globalPosition_j = globalCenterPoint - 0.5 * Lb * n;
        const double3 localPosition_i = reverseRotateVectorByQuaternion(globalPosition_i - masterObjectPosition, masterObjectOrientation);
        const double3 localPosition_j = reverseRotateVectorByQuaternion(globalPosition_j - slaveObjectPosition, slaveObjectOrientation);

        const double3 globalN1_i = -n;
        const double3 globalN2_i = getInitialOrthogonalUnitVectorsN2(globalN1_i);
        const double3 globalN3_i = getInitialOrthogonalUnitVectorsN3(globalN1_i, globalN2_i);
        const double3 globalN1_j = -globalN1_i;
        const double3 globalN2_j = globalN2_i;
        const double3 globalN3_j = globalN3_i;
        const double3 localN1_i = reverseRotateVectorByQuaternion(globalN1_i, masterObjectOrientation);
        const double3 localN2_i = reverseRotateVectorByQuaternion(globalN2_i, masterObjectOrientation);
        const double3 localN3_i = reverseRotateVectorByQuaternion(globalN3_i, masterObjectOrientation);
        const double3 localN1_j = reverseRotateVectorByQuaternion(globalN1_j, slaveObjectOrientation);
        const double3 localN2_j = reverseRotateVectorByQuaternion(globalN2_j, slaveObjectOrientation);
        const double3 localN3_j = reverseRotateVectorByQuaternion(globalN3_j, slaveObjectOrientation);

        masterObjectID_.pushHost(masterObjectID);
        slaveObjectID_.pushHost(slaveObjectID);

        activated_.pushHost(1);
        centerPoint_.pushHost(globalCenterPoint);

        radius_.pushHost(radius);
        initialLength_.pushHost(initialLength);
        B1_.pushHost(B1);
        B2_.pushHost(B2);
        B3_.pushHost(B3);
        B4_.pushHost(B4);

        Un_.pushHost(0.);
        Us_.pushHost(0.);
        Ub_.pushHost(0.);
        Ut_.pushHost(0.);
        maxNormalStress_.pushHost(0.);
        maxShearStress_.pushHost(0.);

        tensileStrength_.pushHost(tensileStrength);
        cohesion_.pushHost(cohesion);
        frictionCoefficient_.pushHost(frictionCoefficient);

        masterVBondPoint_.pushHost(localN1_i, localN2_i, localN3_i, localPosition_i);
        slaveVBondPoint_.pushHost(localN1_j, localN2_j, localN3_j, localPosition_j);
    }

    void initialize(const size_t maxGPUThread, cudaStream_t stream)
    {
        const size_t numBondedPair = centerPoint_.hostSize();
        if (numBondedPair == 0) return;

        copyHostToDevice(stream);

        if (maxGPUThread > 0) blockDim_ = maxGPUThread;
        if (numBondedPair < maxGPUThread) blockDim_ = numBondedPair;
        gridDim_ = (numBondedPair + blockDim_ - 1) / blockDim_;
    }

    /**
    * @brief Output bonded interaction state to VTU (binary appended format).
    *
    * Each bond is stored as VTK_VERTEX at centerPoint_.
    * PointData stores bond state variables.
    *
    * @param dir     Output directory
    * @param iFrame  Frame index
    * @param iStep   Step index
    * @param time    Physical time
    */
    void outputVTU(const std::string& dir, const size_t iFrame, const size_t iStep, const double time) const
    {
        if (numPair() == 0) return;

        MKDIR(dir.c_str());

        std::ostringstream fname;
        fname << dir << "/VBondedInteraction_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";

        const size_t N = numPair();

        const std::vector<int>& masterObjectID = masterObjectID_.hostRef();
        const std::vector<int>& slaveObjectID = slaveObjectID_.hostRef();
        const std::vector<int>& activated = activated_.hostRef();
        const std::vector<double3>& point = centerPoint_.hostRef();

        const std::vector<double>& Un = Un_.hostRef();
        const std::vector<double>& Us = Us_.hostRef();
        const std::vector<double>& Ub = Ub_.hostRef();
        const std::vector<double>& Ut = Ut_.hostRef();
        const std::vector<double>& maxNormalStress = maxNormalStress_.hostRef();
        const std::vector<double>& maxShearStress = maxShearStress_.hostRef();

        // -------------------------------------------------------------------------
        std::vector<float> points(3 * N);
        std::vector<int32_t> connectivity(N), offsets(N);
        std::vector<uint8_t> types(N, 1);

        std::vector<int32_t> mid(masterObjectID.begin(), masterObjectID.end());
        std::vector<int32_t> sid(slaveObjectID.begin(), slaveObjectID.end());
        std::vector<int32_t> act(activated.begin(), activated.end());

        std::vector<float> un(N), us(N), ub(N), ut(N);
        std::vector<float> sn(N), ss(N);

        for (size_t i = 0; i < N; ++i)
        {
            points[3*i+0] = (float)point[i].x;
            points[3*i+1] = (float)point[i].y;
            points[3*i+2] = (float)point[i].z;

            connectivity[i] = (int32_t)i;
            offsets[i] = (int32_t)(i + 1);

            un[i] = (float)Un[i];
            us[i] = (float)Us[i];
            ub[i] = (float)Ub[i];
            ut[i] = (float)Ut[i];

            sn[i] = (float)maxNormalStress[i];
            ss[i] = (float)maxShearStress[i];
        }

        // -------------------------------------------------------------------------
        size_t offset = 0;
        auto B = [&](size_t n) { return sizeof(uint64_t) + n; };

        size_t off_p = offset; offset += B(points.size()*sizeof(float));
        size_t off_c = offset; offset += B(connectivity.size()*sizeof(int32_t));
        size_t off_o = offset; offset += B(offsets.size()*sizeof(int32_t));
        size_t off_t = offset; offset += B(types.size()*sizeof(uint8_t));

        size_t off_mid = offset; offset += B(mid.size()*sizeof(int32_t));
        size_t off_sid = offset; offset += B(sid.size()*sizeof(int32_t));
        size_t off_act = offset; offset += B(act.size()*sizeof(int32_t));

        size_t off_un = offset; offset += B(un.size()*sizeof(float));
        size_t off_us = offset; offset += B(us.size()*sizeof(float));
        size_t off_ub = offset; offset += B(ub.size()*sizeof(float));
        size_t off_ut = offset; offset += B(ut.size()*sizeof(float));

        size_t off_sn = offset; offset += B(sn.size()*sizeof(float));
        size_t off_ss = offset; offset += B(ss.size()*sizeof(float));

        std::ofstream out(fname.str(), std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open " + fname.str());

        // -------------------------------------------------------------------------
        out << "<?xml version=\"1.0\"?>\n";
        out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
            "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
        out << "<UnstructuredGrid>\n";

        out << "<FieldData>\n";
        out << "<DataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">" << (float)time << "</DataArray>\n";
        out << "<DataArray type=\"Int32\" Name=\"STEP\" NumberOfTuples=\"1\" format=\"ascii\">" << (int32_t)iStep << "</DataArray>\n";
        out << "</FieldData>\n";

        out << "<Piece NumberOfPoints=\"" << N << "\" NumberOfCells=\"" << N << "\">\n";

        out << "<Points>\n";
        out << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_p << "\"/>\n";
        out << "</Points>\n";

        out << "<Cells>\n";
        out << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\"" << off_c << "\"/>\n";
        out << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\"" << off_o << "\"/>\n";
        out << "<DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\"" << off_t << "\"/>\n";
        out << "</Cells>\n";

        out << "<PointData Scalars=\"activated\">\n";
        out << "<DataArray type=\"Int32\" Name=\"masterObjectID\" format=\"appended\" offset=\"" << off_mid << "\"/>\n";
        out << "<DataArray type=\"Int32\" Name=\"slaveObjectID\" format=\"appended\" offset=\"" << off_sid << "\"/>\n";
        out << "<DataArray type=\"Int32\" Name=\"activated\" format=\"appended\" offset=\"" << off_act << "\"/>\n";

        out << "<DataArray type=\"Float32\" Name=\"normalElasticEnergy\" format=\"appended\" offset=\"" << off_un << "\"/>\n";
        out << "<DataArray type=\"Float32\" Name=\"shearElasticEnergy\" format=\"appended\" offset=\"" << off_us << "\"/>\n";
        out << "<DataArray type=\"Float32\" Name=\"bendingElasticEnergy\" format=\"appended\" offset=\"" << off_ub << "\"/>\n";
        out << "<DataArray type=\"Float32\" Name=\"torsionElasticEnergy\" format=\"appended\" offset=\"" << off_ut << "\"/>\n";

        out << "<DataArray type=\"Float32\" Name=\"maxNormalStress\" format=\"appended\" offset=\"" << off_sn << "\"/>\n";
        out << "<DataArray type=\"Float32\" Name=\"maxShearStress\" format=\"appended\" offset=\"" << off_ss << "\"/>\n";
        out << "</PointData>\n";

        out << "</Piece>\n</UnstructuredGrid>\n";
        out << "<AppendedData encoding=\"raw\">\n_";

        auto W = [&](const void* d, size_t n)
        {
            uint64_t s = n;
            out.write((char*)&s, sizeof(uint64_t));
            out.write((char*)d, n);
        };

        W(points.data(), points.size()*sizeof(float));
        W(connectivity.data(), connectivity.size()*sizeof(int32_t));
        W(offsets.data(), offsets.size()*sizeof(int32_t));
        W(types.data(), types.size()*sizeof(uint8_t));

        W(mid.data(), mid.size()*sizeof(int32_t));
        W(sid.data(), sid.size()*sizeof(int32_t));
        W(act.data(), act.size()*sizeof(int32_t));

        W(un.data(), un.size()*sizeof(float));
        W(us.data(), us.size()*sizeof(float));
        W(ub.data(), ub.size()*sizeof(float));
        W(ut.data(), ut.size()*sizeof(float));

        W(sn.data(), sn.size()*sizeof(float));
        W(ss.data(), ss.size()*sizeof(float));

        out << "\n</AppendedData>\n</VTKFile>\n";
    }

    void finalize(cudaStream_t stream)
    {
        copyDeviceToHost(stream);
    }

    int* masterObjectID() { return masterObjectID_.d_ptr; }
    int* slaveObjectID() { return slaveObjectID_.d_ptr; }

    int* activated() { return activated_.d_ptr; }
    double3* centerPoint() { return centerPoint_.d_ptr; }

    double* radius() { return radius_.d_ptr; }
    double* initialLength() { return initialLength_.d_ptr; }
    double* B1() { return B1_.d_ptr; }
    double* B2() { return B2_.d_ptr; }
    double* B3() { return B3_.d_ptr; }
    double* B4() { return B4_.d_ptr; }

    double* Un() { return Un_.d_ptr; }
    double* Us() { return Us_.d_ptr; }
    double* Ub() { return Ub_.d_ptr; }
    double* Ut() { return Ut_.d_ptr; }
    double* maxNormalStress() { return maxNormalStress_.d_ptr; }
    double* maxShearStress() { return maxShearStress_.d_ptr; }

    double* tensileStrength() { return tensileStrength_.d_ptr; }
    double* cohesion() { return cohesion_.d_ptr; }
    double* frictionCoefficient() { return frictionCoefficient_.d_ptr; }

    double3* masterVBondPointLocalVectorN1() { return masterVBondPoint_.localVectorN1_.d_ptr; }
    double3* masterVBondPointLocalVectorN2() { return masterVBondPoint_.localVectorN2_.d_ptr; }
    double3* masterVBondPointLocalVectorN3() { return masterVBondPoint_.localVectorN3_.d_ptr; }
    double3* masterVBondPointLocalPosition() { return masterVBondPoint_.localPosition_.d_ptr; }

    double3* slaveVBondPointLocalVectorN1() { return slaveVBondPoint_.localVectorN1_.d_ptr; }
    double3* slaveVBondPointLocalVectorN2() { return slaveVBondPoint_.localVectorN2_.d_ptr; }
    double3* slaveVBondPointLocalVectorN3() { return slaveVBondPoint_.localVectorN3_.d_ptr; }
    double3* slaveVBondPointLocalPosition() { return slaveVBondPoint_.localPosition_.d_ptr; }

    size_t numPair() const { return centerPoint_.hostSize(); }
    size_t numPair_device() const { return centerPoint_.deviceSize(); }
    const size_t& gridDim() const { return gridDim_; }
    const size_t& blockDim() const { return blockDim_; }

private:
    void copyHostToDevice(cudaStream_t stream)
    {
        masterObjectID_.copyHostToDevice(stream);
        slaveObjectID_.copyHostToDevice(stream);

        activated_.copyHostToDevice(stream);
        centerPoint_.copyHostToDevice(stream);

        radius_.copyHostToDevice(stream);
        initialLength_.copyHostToDevice(stream);
        B1_.copyHostToDevice(stream);
        B2_.copyHostToDevice(stream);
        B3_.copyHostToDevice(stream);
        B4_.copyHostToDevice(stream);

        Un_.copyHostToDevice(stream);
        Us_.copyHostToDevice(stream);
        Ub_.copyHostToDevice(stream);
        Ut_.copyHostToDevice(stream);
        maxNormalStress_.copyHostToDevice(stream);
        maxShearStress_.copyHostToDevice(stream);

        tensileStrength_.copyHostToDevice(stream);
        cohesion_.copyHostToDevice(stream);
        frictionCoefficient_.copyHostToDevice(stream);

        masterVBondPoint_.copyHostToDevice(stream);
        slaveVBondPoint_.copyHostToDevice(stream);
    }

    void copyDeviceToHost(cudaStream_t stream)
    {
        activated_.copyDeviceToHost(stream);
        centerPoint_.copyDeviceToHost(stream);

        Un_.copyDeviceToHost(stream);
        Us_.copyDeviceToHost(stream);
        Ub_.copyDeviceToHost(stream);
        Ut_.copyDeviceToHost(stream);
        maxNormalStress_.copyDeviceToHost(stream);
        maxShearStress_.copyDeviceToHost(stream);
    }

    HostDeviceArray1D<int> masterObjectID_;
    HostDeviceArray1D<int> slaveObjectID_;

    HostDeviceArray1D<int> activated_;
    HostDeviceArray1D<double3> centerPoint_;

    HostDeviceArray1D<double> radius_;
    HostDeviceArray1D<double> initialLength_;
    HostDeviceArray1D<double> B1_;
    HostDeviceArray1D<double> B2_;
    HostDeviceArray1D<double> B3_;
    HostDeviceArray1D<double> B4_;

    HostDeviceArray1D<double> Un_;
    HostDeviceArray1D<double> Us_;
    HostDeviceArray1D<double> Ub_;
    HostDeviceArray1D<double> Ut_;
    HostDeviceArray1D<double> maxNormalStress_;
    HostDeviceArray1D<double> maxShearStress_;

    HostDeviceArray1D<double> tensileStrength_;
    HostDeviceArray1D<double> cohesion_;
    HostDeviceArray1D<double> frictionCoefficient_;

    VBondPoint masterVBondPoint_;
    VBondPoint slaveVBondPoint_;

    size_t gridDim_{1};
    size_t blockDim_{1};
};