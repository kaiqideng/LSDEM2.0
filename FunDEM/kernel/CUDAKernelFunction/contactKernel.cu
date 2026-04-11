#include "contactKernel.cuh"
#include "myVec.h"

/**
 * @brief Atomic add for double on device. Uses native atomicAdd on sm_60+; CAS loop otherwise.
 *
 * @param[in,out] addr   Address to add into.
 * @param[in]     val    Value to add.
 * @return The old value stored at *addr before the add (CUDA atomicAdd semantics).
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)       // sm 6.0+
__device__ __forceinline__ double atomicAddDouble(double* addr, double val)
{
	return atomicAdd(addr, val);
}
#else                                                   
__device__ __forceinline__ double atomicAddDouble(double* addr, double val)
{
	auto  addr_ull = reinterpret_cast<unsigned long long*>(addr);
	unsigned long long old = *addr_ull, assumed;

	do {
		assumed = old;
		double  old_d = __longlong_as_double(assumed);
		double  new_d = old_d + val;
		old = atomicCAS(addr_ull, assumed, __double_as_longlong(new_d));
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

/**
 * @brief Atomic add a double3 vector into arr[idx] component-wise.
 *
 * @param[in,out] arr   Target array of double3.
 * @param[in]     idx   Index into arr.
 * @param[in]     v     Value to add to arr[idx].
 */
__device__ void atomicAddDouble3(double3* arr, size_t idx, const double3& v)
{
    atomicAddDouble(&(arr[idx].x), v.x);
	atomicAddDouble(&(arr[idx].y), v.y);
	atomicAddDouble(&(arr[idx].z), v.z);
}

__global__ void addLevelSetParticleContactForceTorqueKernel(double3* slidingSpring, 
double* normalElasticEnergy,
double* slidingElasticEnergy,
double3* force_p,
double3* torque_p,

const double3* contactPoint,
const double3* contactNormal,
const double* overlap, 
const int* boundaryNodePointed, 
const int* objectPointing, 

const double3* localPosition_bNode,
const int* particleID_bNode,

const double3* position_p, 
const double3* velocity_p, 
const double3* angularVelocity_p, 
const double* inverseMass_p, 
const double* normalStiffness_p, 
const double* shearStiffness_p, 
const double* frictionCoefficient_p, 
const double* restitutionCoefficient_p, 

const double dt,
const size_t numPair)
{
    const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= numPair) return;

	const int idx_i = particleID_bNode[boundaryNodePointed[k]];
	const double3 r_i = position_p[idx_i];
	const double3 v_i = velocity_p[idx_i];
	const double3 w_i = angularVelocity_p[idx_i];
	const double invM_i = inverseMass_p[idx_i];
    const double kn_i = normalStiffness_p[idx_i];
	const double ks_i = shearStiffness_p[idx_i];
	const double mu_i = frictionCoefficient_p[idx_i];
	const double res_i = restitutionCoefficient_p[idx_i];

	const int idx_j = objectPointing[k];
	const double3 r_j = position_p[idx_j];
	const double3 v_j = velocity_p[idx_j];
	const double3 w_j = angularVelocity_p[idx_j];
	const double invM_j = inverseMass_p[idx_j];
	const double kn_j = normalStiffness_p[idx_j];
	const double ks_j = shearStiffness_p[idx_j];
	const double mu_j = frictionCoefficient_p[idx_j];
	const double res_j = restitutionCoefficient_p[idx_j];

    const double3 r_c = contactPoint[k];
	const double3 n_ij = contactNormal[k];
	const double delta = overlap[k];
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));
	double kn = 0., ks = 0., effM = 0.;
	if (kn_i > 0. && kn_j > 0.) kn = kn_i * kn_j / (kn_i + kn_j);
	if (ks_i > 0. && ks_j > 0.) ks = ks_i * ks_j / (ks_i + ks_j);
	if (invM_i > 0. || invM_j > 0.) effM = 1. / (invM_i + invM_j);
	const double res = fmin(res_i, res_j);
	const double mu = fmin(mu_i, mu_j);
	double3 F_c = make_double3(0., 0., 0.);
	double3 epsilon_s = slidingSpring[k];
	LinearContact(F_c, epsilon_s, 
	v_c_ij, n_ij, delta, dt, 
	kn, ks, mu, res, effM);

	slidingSpring[k] = epsilon_s;
	normalElasticEnergy[k] = 0.5 * delta * delta * kn;
	slidingElasticEnergy[k] = 0.5 * lengthSquared(epsilon_s) * ks;
	atomicAddDouble3(force_p, idx_i, F_c);
	atomicAddDouble3(torque_p, idx_i, cross(r_c - r_i, F_c));
	atomicAddDouble3(force_p, idx_j, -F_c);
	atomicAddDouble3(torque_p, idx_j, cross(r_c - r_j, -F_c));
}

__global__ void addFixedLevelSetParticleContactForceTorqueKernel(double3* slidingSpring, 
double* normalElasticEnergy,
double* slidingElasticEnergy,
double3* force_p,
double3* torque_p,

const double3* contactPoint,
const double3* contactNormal,
const double* overlap, 
const int* boundaryNodePointed,
const int* objectPointing, 

const double3* localPosition_bNode,
const int* particleID_bNode,

const double3* position_p, 
const double3* velocity_p, 
const double3* angularVelocity_p, 
const double* inverseMass_p, 
const double* normalStiffness_p, 
const double* shearStiffness_p, 
const double* frictionCoefficient_p, 
const double* restitutionCoefficient_p, 

const double3* position_fp, 
const double3* velocity_fp, 
const double3* angularVelocity_fp, 
const double* frictionCoefficient_fp, 
const double* restitutionCoefficient_fp, 

const double dt,
const size_t numPair)
{
    const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= numPair) return;

	const int idx_i = particleID_bNode[boundaryNodePointed[k]];
	const double3 r_i = position_p[idx_i];
	const double3 v_i = velocity_p[idx_i];
	const double3 w_i = angularVelocity_p[idx_i];
	const double invM_i = inverseMass_p[idx_i];
    const double kn_i = normalStiffness_p[idx_i];
	const double ks_i = shearStiffness_p[idx_i];
	const double mu_i = frictionCoefficient_p[idx_i];
	const double res_i = restitutionCoefficient_p[idx_i];

	const int idx_j = objectPointing[k];
	const double3 r_j = position_fp[idx_j];
	const double3 v_j = velocity_fp[idx_j];
	const double3 w_j = angularVelocity_fp[idx_j];
	const double mu_j = frictionCoefficient_fp[idx_i];
	const double res_j = restitutionCoefficient_fp[idx_j];

	const double3 r_c = contactPoint[k];
	const double3 n_ij = contactNormal[k];
	const double delta = overlap[k];
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));
	double effM = 0.;
	if (invM_i > 0.) effM = 1. / invM_i;
	const double mu = fmin(mu_i, mu_j);
	const double res = fmin(res_i, res_j);
	double3 F_c = make_double3(0., 0., 0.);
	double3 epsilon_s = slidingSpring[k];
	LinearContact(F_c, epsilon_s, 
	v_c_ij, n_ij, delta, dt, 
	kn_i, ks_i, mu, res, effM);
	
	slidingSpring[k] = epsilon_s;
	normalElasticEnergy[k] = 0.5 * delta * delta * kn_i;
	slidingElasticEnergy[k] = 0.5 * lengthSquared(epsilon_s) * ks_i;
	atomicAddDouble3(force_p, idx_i, F_c);
	atomicAddDouble3(torque_p, idx_i, cross(r_c - r_i, F_c));
}

__global__ void addBondedForceTorqueKernel(double3* bondPoint,
double* maxNormalStress, 
double* maxShearStress,
double* Un, 
double* Us, 
double* Ub,
double* Ut,
int* isBonded, 

double3* force_p, 
double3* torque_p, 

const double* B1, 
const double* B2, 
const double* B3, 
const double* B4, 
const double* bondRadius,
const double* bondInitialLength,
const double* tensileStrength, 
const double* cohesion, 
const double* frictionCoefficient, 
const double3* masterVBondPointLocalVectorN1,
const double3* masterVBondPointLocalVectorN2,
const double3* masterVBondPointLocalVectorN3,
const double3* masterVBondPointLocalPosition,
const double3* slaveVBondPointLocalVectorN1,
const double3* slaveVBondPointLocalVectorN2,
const double3* slaveVBondPointLocalVectorN3,
const double3* slaveVBondPointLocalPosition,
const int* masterObjectID,
const int* slaveObjectID,

const double3* position_p,
const quaternion* orientation_p,

const size_t numPair)
{
	const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= numPair) return;

	if (isBonded[k] == 0)
	{
		Un[k] = 0.;
		Us[k] = 0.;
		Ub[k] = 0.;
		Ut[k] = 0.;
		maxNormalStress[k] = 0.;
		maxShearStress[k] = 0.;
		return;
	}

	const int idx_i = masterObjectID[k];
	const int idx_j = slaveObjectID[k];
	const quaternion q_i = orientation_p[idx_i];
	const quaternion q_j = orientation_p[idx_j];
	const double3 n1_i = rotateVectorByQuaternion(q_i, masterVBondPointLocalVectorN1[k]);
	const double3 n2_i = rotateVectorByQuaternion(q_i, masterVBondPointLocalVectorN2[k]);
	const double3 n3_i = rotateVectorByQuaternion(q_i, masterVBondPointLocalVectorN3[k]);
	const double3 n1_j = rotateVectorByQuaternion(q_j, slaveVBondPointLocalVectorN1[k]);
	const double3 n2_j = rotateVectorByQuaternion(q_j, slaveVBondPointLocalVectorN2[k]);
	const double3 n3_j = rotateVectorByQuaternion(q_j, slaveVBondPointLocalVectorN3[k]);
    const double3 r_i = position_p[idx_i];
	const double3 r_j = position_p[idx_j];   
	const double3 rb_i = rotateVectorByQuaternion(q_i, masterVBondPointLocalPosition[k]) + r_i;
	const double3 rb_j = rotateVectorByQuaternion(q_j, slaveVBondPointLocalPosition[k]) + r_j;
	bondPoint[k] = 0.5 * (rb_i +rb_j);

	double3 F_ij = make_double3(0., 0., 0.);
	double3 M_ij = make_double3(0., 0., 0.);
	double3 M_ji = make_double3(0., 0., 0.);
	isBonded[k] = VBond(F_ij, M_ij, M_ji, Un[k], Us[k], Ub[k], Ut[k], maxNormalStress[k],maxShearStress[k],
	rb_i, rb_j, n1_i, n2_i, n3_i, n1_j, n2_j, n3_j, B1[k], B2[k], B3[k], B4[k], bondRadius[k], bondInitialLength[k], 
	tensileStrength[k], cohesion[k], frictionCoefficient[k]);

	atomicAddDouble3(force_p, idx_i, F_ij);
	atomicAddDouble3(torque_p, idx_i, M_ij + cross(rb_i - r_i, F_ij));
	atomicAddDouble3(force_p, idx_j, -F_ij);
	atomicAddDouble3(torque_p, idx_j, M_ji + cross(rb_j - r_j, -F_ij));
}

extern "C" void launchAddLevelSetParticleContactForceTorque(double3* slidingSpring, 
double* normalElasticEnergy, 
double* slidingElasticEnergy, 
const double3* contactPoint, 
const double3* contactNormal,
const double* overlap,
const int* boundaryNodePointed,
const int* objectPointing,

const double3* localPosition_bNode,
const int* particleID_bNode,

double3* force_p,
double3* torque_p,
const double3* position_p,
const double3* velocity_p,
const double3* angularVelocity_p,
const double* inverseMass_p,
const double* normalStiffness_p,
const double* shearStiffness_p,
const double* frictionCoefficient_p,
const double* restitutionCoefficient_p,

const double timeStep,

const size_t numPair,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    addLevelSetParticleContactForceTorqueKernel<<<gridD, blockD, 0, stream>>>(slidingSpring,
	normalElasticEnergy, 
	slidingElasticEnergy, 
    force_p,
    torque_p,

    contactPoint,
    contactNormal,
    overlap,
	boundaryNodePointed,
    objectPointing,

    localPosition_bNode,
    particleID_bNode,

    position_p,
    velocity_p,
    angularVelocity_p,
	inverseMass_p,
    normalStiffness_p,
    shearStiffness_p,
    frictionCoefficient_p,
	restitutionCoefficient_p,

    timeStep,

    numPair);
}

extern "C" void launchAddFixedLevelSetParticleContactForceTorque(double3* slidingSpring, 
double* normalElasticEnergy, 
double* slidingElasticEnergy, 
const double3* contactPoint, 
const double3* contactNormal,
const double* overlap,
const int* boundaryNodePointed,
const int* objectPointing,

const double3* localPosition_bNode,
const int* particleID_bNode,

double3* force_p,
double3* torque_p,
const double3* position_p,
const double3* velocity_p,
const double3* angularVelocity_p,
const double* inverseMass_p,
const double* normalStiffness_p,
const double* shearStiffness_p,
const double* frictionCoefficient_p,
const double* restitutionCoefficient_p,

const double3* position_fp,
const double3* velocity_fp,
const double3* angularVelocity_fp,
const double* frictionCoefficient_fp,
const double* restitutionCoefficient_fp,

const double timeStep,

const size_t numPair,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    addFixedLevelSetParticleContactForceTorqueKernel<<<gridD, blockD, 0, stream>>>(slidingSpring,
	normalElasticEnergy, 
	slidingElasticEnergy, 
    force_p,
    torque_p,

    contactPoint,
    contactNormal,
    overlap,
	boundaryNodePointed,
    objectPointing,

    localPosition_bNode,
    particleID_bNode,

    position_p,
    velocity_p,
    angularVelocity_p,
	inverseMass_p,
    normalStiffness_p,
    shearStiffness_p,
    frictionCoefficient_p,
	restitutionCoefficient_p,

	position_fp,
	velocity_fp,
	angularVelocity_fp,
	frictionCoefficient_fp,
	restitutionCoefficient_fp,

    timeStep,

    numPair);
}

extern "C" void launchAddBondedForceTorque(
double3* bondPoint,
double* maxNormalStress,
double* maxShearStress,
double* Un,
double* Us,
double* Ub,
double* Ut,
int* isBonded,
const double* B1,
const double* B2,
const double* B3,
const double* B4,
const double* bondRadius,
const double* bondInitialLength,
const double* tensileStrength,
const double* cohesion,
const double* frictionCoefficient,
const double3* masterVBondPointLocalVectorN1,
const double3* masterVBondPointLocalVectorN2,
const double3* masterVBondPointLocalVectorN3,
const double3* masterVBondPointLocalPosition,
const double3* slaveVBondPointLocalVectorN1,
const double3* slaveVBondPointLocalVectorN2,
const double3* slaveVBondPointLocalVectorN3,
const double3* slaveVBondPointLocalPosition,
const int* masterObjectID,
const int* slaveObjectID,

double3* force_p,
double3* torque_p,
const double3* position_p,
const quaternion* orientation_p,

const size_t numPair,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    addBondedForceTorqueKernel<<<gridD, blockD, 0, stream>>>(bondPoint,
    maxNormalStress,
    maxShearStress,
    Un,
    Us,
    Ub,
    Ut,
    isBonded,

    force_p,
    torque_p,

    B1,
    B2,
    B3,
    B4,
    bondRadius,
    bondInitialLength,
    tensileStrength,
    cohesion,
    frictionCoefficient,
    masterVBondPointLocalVectorN1,
    masterVBondPointLocalVectorN2,
    masterVBondPointLocalVectorN3,
    masterVBondPointLocalPosition,
    slaveVBondPointLocalVectorN1,
    slaveVBondPointLocalVectorN2,
    slaveVBondPointLocalVectorN3,
    slaveVBondPointLocalPosition,
    masterObjectID,
    slaveObjectID,

    position_p,
    orientation_p,

    numPair);
}