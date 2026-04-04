#pragma once
#include "myUtility/myQua.h"

static __device__ __forceinline__ double3 integrateSlidingOrRollingSpring(const double3 springPrev, 
const double3 springVelocity, 
const double3 contactNormal, 
const double3 normalContactForce, 
const double frictionCoefficient, 
const double stiffness, 
const double dampingCoefficient, 
const double timeStep)
{
	double3 spring = make_double3(0., 0., 0.);
	if (frictionCoefficient > 0. && stiffness > 0.)
	{
		double3 springPrev1 = springPrev - dot(springPrev, contactNormal) * contactNormal;
		double absoluteSpringPrev1 = length(springPrev1);
		if (!isZero(absoluteSpringPrev1))
		{
			springPrev1 *= length(springPrev) / absoluteSpringPrev1;
		}
		spring = springPrev1 + springVelocity * timeStep;
		double3 springForce = -stiffness * spring - dampingCoefficient * springVelocity;
		double absoluteSpringForce = length(springForce);
		double absoluteNormalContactForce = length(normalContactForce);
		if (absoluteSpringForce > frictionCoefficient * absoluteNormalContactForce)
		{
			double ratio = frictionCoefficient * absoluteNormalContactForce / absoluteSpringForce;
			springForce *= ratio;
			spring = -(springForce + dampingCoefficient * springVelocity) / stiffness;
		}
	}
	return spring;
}

__device__ __forceinline__ void LinearContact(double3& contactForce, 
double3& slidingSpring, 
const double3 relativeVelocityAtContact,
const double3 relativeAngularVelocityAtContact,
const double3 contactNormal,
const double normalOverlap,
const double timeStep,
const double normalStiffness,
const double slidingStiffness,
const double slidingFrictionCoefficient)
{
	if (normalOverlap > 0.)
	{
		const double3 normalRelativeVelocityAtContact = dot(relativeVelocityAtContact, contactNormal) * contactNormal;
		const double3 normalContactForce = normalStiffness * normalOverlap * contactNormal;

		const double3 slidingRelativeVelocity = relativeVelocityAtContact - normalRelativeVelocityAtContact;
		slidingSpring = integrateSlidingOrRollingSpring(slidingSpring, 
		slidingRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		slidingFrictionCoefficient, 
		slidingStiffness, 
		0.0, 
		timeStep);
		const double3 slidingForce = -slidingStiffness * slidingSpring;

		contactForce = normalContactForce + slidingForce;
	}
	else
	{
		slidingSpring = make_double3(0., 0., 0.);
	}
}

__device__ __forceinline__ int ParallelBond(double& bondNormalForce, 
double& bondTorsionalTorque, 
double3& bondShearForce, 
double3& bondBendingTorque,
double& maxNormalStress,
double& maxShearStress,
const double3 previousContactNormal,
const double3 contactNormal,
const double3 relativeVelocityAtContact,
const double3 angularVelocityA,
const double3 angularVelocityB,
const double timeStep,
const double bondRadius,
const double normalStiffness,
const double shearStiffness,
const double bendingStiffness,
const double torsionStiffness,
const double bondTensileStrength,
const double bondCohesion,
const double bondFrictionCoefficient)
{
	const double3 nn = cross(previousContactNormal, contactNormal);
	const double3 axis1 = normalize(nn);
	const double sinTheta1 = length(nn);
	bondShearForce = rotateVectorAxisSin(bondShearForce, axis1, sinTheta1);
	bondBendingTorque = rotateVectorAxisSin(bondBendingTorque, axis1, sinTheta1);
	const double3 theta2 = dot(0.5 * (angularVelocityA + angularVelocityB) * timeStep, contactNormal) * contactNormal;
	bondShearForce = rotateVector(bondShearForce, theta2);
	bondBendingTorque = rotateVector(bondBendingTorque, theta2);

	const double bondArea = bondRadius * bondRadius * pi();// cross-section area of beam of the bond
	const double bondInertiaMoment = bondRadius * bondRadius * bondRadius * bondRadius * pi() / 4.;// inertia moment
	const double bondPolarInertiaMoment = 2 * bondInertiaMoment;// polar inertia moment

	const double3 normalTranslationIncrement = dot(relativeVelocityAtContact, contactNormal) * contactNormal * timeStep;
	const double3 tangentialTranslationIncrement = relativeVelocityAtContact * timeStep - normalTranslationIncrement;
	bondNormalForce -= dot(normalTranslationIncrement * normalStiffness, contactNormal);
	bondShearForce -= tangentialTranslationIncrement * shearStiffness;
	const double3 relativeAngularVelocity = angularVelocityA - angularVelocityB;
	const double3 normalRotationIncrement = dot(relativeAngularVelocity, contactNormal) * contactNormal * timeStep;
	const double3 tangentialRotationIncrement = relativeAngularVelocity * timeStep - normalRotationIncrement;
	bondTorsionalTorque -= dot(normalRotationIncrement * torsionStiffness, contactNormal);
	bondBendingTorque -= tangentialRotationIncrement * bendingStiffness;

	maxNormalStress = -bondNormalForce / bondArea + length(bondBendingTorque) / bondInertiaMoment * bondRadius;// maximum tension stress
	maxShearStress = length(bondShearForce) / bondArea + fabs(bondTorsionalTorque) / bondPolarInertiaMoment * bondRadius;// maximum shear stress

	int isBonded = 1;
	if (bondTensileStrength > 0 && maxNormalStress > bondTensileStrength)
	{
		isBonded = 0;
	}
	else if (bondCohesion > 0 && maxShearStress > bondCohesion - bondFrictionCoefficient * maxNormalStress)
	{
		isBonded = 0;
	}
	return isBonded;
}

__device__ __forceinline__ int VBond(double3& F_ij, 
double3& M_ij, 
double3& M_ji, 
double& Un,
double& Us,
double& Ub,
double& Ut,
double& maxNormalStress,
double& maxShearStress,
const double3 rb_i,
const double3 rb_j,
const double3 n1_i,
const double3 n2_i,
const double3 n3_i,
const double3 n1_j,
const double3 n2_j,
const double3 n3_j,
const double B1,
const double B2,
const double B3,
const double B4,
const double bondRadius, 
const double bondInitialLength, 
const double bondTensileStrength,
const double bondCohesion,
const double bondFrictionCoefficient)
{
	const double3 rb_ij = rb_i - rb_j;
	const double D = length(rb_ij);
	if (isZero(D)) return 0;
	const double3 n_ij = normalize(rb_ij);
	const double3 e_ij = -n_ij;
	const double D0 = bondInitialLength;
	Un = 0.5 * B1 * (D - D0) * (D - D0);
	Us = B2 * (0.5 * dot((n1_j - n1_i), e_ij) - 0.25 * dot(n1_i, n1_j) + 0.75);
	Ub = (0.25 * B2 + B3 + 0.5 * B4) * (dot(n1_i, n1_j) + 1.);
	Ut = -0.5 * B4 * (dot(n1_i, n1_j) + dot(n2_i, n2_j) + dot(n3_i, n3_j) - 1);
	F_ij = B1 * (D - D0) * e_ij + B2 / (2 * D) * ((n1_j - n1_i) - dot((n1_j - n1_i), e_ij) * e_ij);
	const double3 M_TB = B3 * cross(n1_j, n1_i) - 0.5 * B4 * (cross(n2_j, n2_i) + cross(n3_j, n3_i));
	M_ij = -0.5 * B2 * cross(e_ij, n1_i) + M_TB;
	M_ji =  0.5 * B2 * cross(e_ij, n1_j) - M_TB;

	const double bondNormalForce = dot(F_ij, n_ij);
	const double3 bondShearForce = F_ij - bondNormalForce * n_ij;
	const double bondTorsionalTorque = dot(M_TB, n_ij);
	const double3 bondBendingTorque = M_TB - bondTorsionalTorque * n_ij;
	const double bondArea = bondRadius * bondRadius * pi();// cross-section area of beam of the bond
	const double bondInertiaMoment = bondRadius * bondRadius * bondRadius * bondRadius * pi() / 4.;// inertia moment
	const double bondPolarInertiaMoment = 2 * bondInertiaMoment;// polar inertia moment
	maxNormalStress = -bondNormalForce / bondArea + length(bondBendingTorque) / bondInertiaMoment * bondRadius;// maximum tension stress
	maxShearStress = length(bondShearForce) / bondArea + fabs(bondTorsionalTorque) / bondPolarInertiaMoment * bondRadius;// maximum shear stress
	if (bondTensileStrength > 0. && maxNormalStress > bondTensileStrength)
	{
		return 0;
	}
	else if (bondCohesion > 0. && maxShearStress > bondCohesion - bondFrictionCoefficient * maxNormalStress)
	{
		return 0;
	}
	return 1;
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
cudaStream_t stream);

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
cudaStream_t stream);

extern "C" void launchAddLevelSetParticleBondedForceTorque(
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
cudaStream_t stream);