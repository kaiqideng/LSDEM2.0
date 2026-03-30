#include "contactKernel.cuh"
#include "myUtility/myVec.h"

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
const double* normalStiffness_p, 
const double* shearStiffness_p, 
const double* frictionCoefficient_p, 

const double dt,
const size_t numPair)
{
    const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= numPair) return;

    const size_t idx = boundaryNodePointed[k];
	const int idx_i = particleID_bNode[idx];
	const double3 r_i = position_p[idx_i];
	const double3 v_i = velocity_p[idx_i];
	const double3 w_i = angularVelocity_p[idx_i];
    const double kn_i = normalStiffness_p[idx_i];
	const double ks_i = shearStiffness_p[idx_i];
	const double mu_i = frictionCoefficient_p[idx_i];

	const int idx_j = objectPointing[k];
	const double3 r_j = position_p[idx_j];
	const double3 v_j = velocity_p[idx_j];
	const double3 w_j = angularVelocity_p[idx_j];
	const double kn_j = normalStiffness_p[idx_j];
	const double ks_j = shearStiffness_p[idx_j];
	const double mu_j = frictionCoefficient_p[idx_j];

    const double3 r_c = contactPoint[k];
	const double3 n_ij = contactNormal[k];
	const double delta = overlap[k];
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));
	const double3 w_ij = w_i - w_j;
	double kn = 0., ks = 0.;
	if (kn_i > 0. && kn_j > 0.) kn = kn_i * kn_j / (kn_i + kn_j);
	if (ks_i > 0. && ks_j > 0.) ks = ks_i * ks_j / (ks_i + ks_j);
	const double mu = fmin(mu_i, mu_j);
	double3 F_c = make_double3(0., 0., 0.);
	double3 epsilon_s = slidingSpring[k];
	LinearContactForLevelSetParticle(F_c, epsilon_s, 
	v_c_ij, w_ij, n_ij, delta, dt, 
	kn, ks, mu);

	slidingSpring[k] = epsilon_s;
	atomicAddDouble3(force_p, idx_i, F_c);
	atomicAddDouble3(torque_p, idx_i, cross(r_c - r_i, F_c));
	atomicAddDouble3(force_p, idx_j, -F_c);
	atomicAddDouble3(torque_p, idx_j, cross(r_c - r_j, -F_c));
}

__global__ void addFixedLevelSetParticleContactForceTorqueKernel(double3* slidingSpring, 
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
const double* normalStiffness_p, 
const double* shearStiffness_p, 
const double* frictionCoefficient_p, 

const double3* position_fp, 
const double3* velocity_fp, 
const double3* angularVelocity_fp, 
const double* normalStiffness_fp, 
const double* shearStiffness_fp, 
const double* frictionCoefficient_fp,

const double dt,
const size_t numPair)
{
    const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= numPair) return;

    const size_t idx = boundaryNodePointed[k];
	const int idx_i = particleID_bNode[idx];
	const double3 r_i = position_p[idx_i];
	const double3 v_i = velocity_p[idx_i];
	const double3 w_i = angularVelocity_p[idx_i];
    const double kn_i = normalStiffness_p[idx_i];
	const double ks_i = shearStiffness_p[idx_i];
	const double mu_i = frictionCoefficient_p[idx_i];

	const int idx_j = objectPointing[k];
	const double3 r_j = position_fp[idx_j];
	const double3 v_j = velocity_fp[idx_j];
	const double3 w_j = angularVelocity_fp[idx_j];
	const double kn_j = normalStiffness_fp[idx_j];
	const double ks_j = shearStiffness_fp[idx_j];
	const double mu_j = frictionCoefficient_fp[idx_j];

	const double3 r_c = contactPoint[k];
	const double3 n_ij = contactNormal[k];
	const double delta = overlap[k];
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));
	const double3 w_ij = w_i - w_j;
	double kn = 0., ks = 0.;
	if (kn_i > 0. && kn_j > 0.) kn = kn_i * kn_j / (kn_i + kn_j);
	if (ks_i > 0. && ks_j > 0.) ks = ks_i * ks_j / (ks_i + ks_j);
	const double mu = fmin(mu_i, mu_j);
	double3 F_c = make_double3(0., 0., 0.);
	double3 epsilon_s = slidingSpring[k];
	LinearContactForLevelSetParticle(F_c, epsilon_s, 
	v_c_ij, w_ij, n_ij, delta, dt, 
	kn, ks, mu);
	
	slidingSpring[k] = epsilon_s;
	atomicAddDouble3(force_p, idx_i, F_c);
	atomicAddDouble3(torque_p, idx_i, cross(r_c - r_i, F_c));
}

__global__ void addLevelSetParticleBondedForceTorqueKernel(double3* bondPoint,
double3* bondNormal,
double3* shearForce, 
double3* bendingTorque,
double* normalForce, 
double* torsionTorque, 
double* maxNormalStress,
double* maxShearStress,
int* isBonded, 

double3* force_p, 
double3* torque_p, 

const double* normalStiffness, 
const double* torsionStiffness, 
const double* shearStiffness, 
const double* bendingStiffness, 
const double* bondRadius,
const double* tensileStrength, 
const double* cohesion, 
const double* frictionCoefficient, 
const double3* bondEndPointALocalPosition,
const double3* bondEndPointBLocalPosition,
const int* objectPointed_b,
const int* objectPointing_b,

const double3* position_p, 
const double3* velocity_p, 
const double3* angularVelocity_p, 
const quaternion* orientation_p,

const double dt,
const size_t numPair)
{
	const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= numPair) return;

	if (isBonded[k] == 0)
	{
		normalForce[k] = 0;
		torsionTorque[k] = 0;
		shearForce[k] = make_double3(0, 0, 0);
		bendingTorque[k] = make_double3(0, 0, 0);
		return;
	}

	const int idx_i = objectPointed_b[k];
	const int idx_j = objectPointing_b[k];
	const double3 n_ij0 = bondNormal[k];

    const double3 r_i = position_p[idx_i];
	const double3 r_j = position_p[idx_j];
	const double3 rb_i = rotateVectorByQuaternion(orientation_p[idx_i], bondEndPointALocalPosition[k]) + r_i;
	const double3 rb_j = rotateVectorByQuaternion(orientation_p[idx_j], bondEndPointBLocalPosition[k]) + r_j;
	const double3 r_c = 0.5 * (rb_i + rb_j);
	const double3 n_ij = normalize(rb_i - rb_j);
	const double3 v_i = velocity_p[idx_i];
	const double3 v_j = velocity_p[idx_j];
	const double3 w_i = angularVelocity_p[idx_i];
	const double3 w_j = angularVelocity_p[idx_j];
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));

	bondPoint[k] = r_c;
    if (!isZero(length(n_ij))) bondNormal[k] = n_ij;

	const double k_n = normalStiffness[k];
    const double k_s = shearStiffness[k];
    const double k_b = bendingStiffness[k];
    const double k_t = torsionStiffness[k];
	const double rad_b = bondRadius[k];
	const double sigma_s = tensileStrength[k];
	const double C = cohesion[k];
	const double mu = frictionCoefficient[k];

	double F_n = normalForce[k];
	double3 F_s = shearForce[k];
	double T_t = torsionTorque[k];
	double3 T_b = bendingTorque[k];
	double sigma_max = maxNormalStress[k];
	double tau_max = maxShearStress[k];

	isBonded[k] = ParallelBondedContactForLevelSetParticle(F_n, T_t, F_s, T_b, 
	sigma_max, tau_max,
	n_ij0, n_ij, 
	v_c_ij, w_i, w_j, 
	dt, 
	rad_b, k_n, k_s, k_b, k_t, 
	sigma_s, C, mu);

	normalForce[k] = F_n;
	shearForce[k] = F_s;
	torsionTorque[k] = T_t;
	bendingTorque[k] = T_b;
	maxNormalStress[k] = sigma_max;
	maxShearStress[k] = tau_max;

    double3 F_c = F_n * n_ij + F_s;
	double3 T_c = T_t * n_ij + T_b;
	atomicAddDouble3(force_p, idx_i, F_c);
	atomicAddDouble3(torque_p, idx_i, T_c + cross(r_c - r_i, F_c));
	atomicAddDouble3(force_p, idx_j, -F_c);
	atomicAddDouble3(torque_p, idx_j, -T_c + cross(r_c - r_j, -F_c));
}

extern "C" void launchAddLevelSetParticleContactForceTorque(double3* slidingSpring, 
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
const double* normalStiffness_p,
const double* shearStiffness_p,
const double* frictionCoefficient_p,

const double timeStep,

const size_t numPair,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    addLevelSetParticleContactForceTorqueKernel<<<gridD, blockD, 0, stream>>>(slidingSpring,
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
    normalStiffness_p,
    shearStiffness_p,
    frictionCoefficient_p,

    timeStep,

    numPair);
}

extern "C" void launchAddFixedLevelSetParticleContactForceTorque(double3* slidingSpring, 
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
const double* normalStiffness_p,
const double* shearStiffness_p,
const double* frictionCoefficient_p,

const double3* position_fp,
const double3* velocity_fp,
const double3* angularVelocity_fp,
const double* normalStiffness_fp,
const double* shearStiffness_fp,
const double* frictionCoefficient_fp,

const double timeStep,

const size_t numPair,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    addFixedLevelSetParticleContactForceTorqueKernel<<<gridD, blockD, 0, stream>>>(slidingSpring,
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
    normalStiffness_p,
    shearStiffness_p,
    frictionCoefficient_p,

	position_fp,
	velocity_fp,
	angularVelocity_fp,
	normalStiffness_fp,
	shearStiffness_fp,
	frictionCoefficient_fp,

    timeStep,

    numPair);
}

extern "C" void launchAddLevelSetParticleBondedForceTorqueKernel(double3* bondPoint,
double3* bondNormal,
double3* shearForce, 
double3* bendingTorque,
double* normalForce, 
double* torsionTorque, 
double* maxNormalStress,
double* maxShearStress,
int* isBonded, 
const double* normalStiffness, 
const double* torsionStiffness, 
const double* shearStiffness, 
const double* bendingStiffness, 
const double* bondRadius,
const double* tensileStrength, 
const double* cohesion, 
const double* frictionCoefficient, 
const double3* bondEndPointALocalPosition,
const double3* bondEndPointBLocalPosition,
const int* objectPointed_b,
const int* objectPointing_b,

double3* force_p, 
double3* torque_p, 
const double3* position_p, 
const double3* velocity_p, 
const double3* angularVelocity_p, 
const quaternion* orientation_p,

const double timeStep,

const size_t numPair,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
	addLevelSetParticleBondedForceTorqueKernel<<<gridD, blockD, 0, stream>>>(bondPoint,
	bondNormal,
	shearForce, 
	bendingTorque,
	normalForce, 
	torsionTorque, 
	maxNormalStress,
	maxShearStress,
	isBonded, 

	force_p, 
	torque_p, 

	normalStiffness, 
	torsionStiffness, 
	shearStiffness, 
	bendingStiffness, 
	bondRadius,
	tensileStrength, 
	cohesion, 
	frictionCoefficient, 
	bondEndPointALocalPosition,
	bondEndPointBLocalPosition,
	objectPointed_b,
	objectPointing_b,

	position_p, 
	velocity_p, 
	angularVelocity_p, 
	orientation_p,

	timeStep,
	numPair);
}