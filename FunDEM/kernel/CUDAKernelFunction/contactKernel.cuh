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

__device__ __forceinline__ void LinearContactForLevelSetParticle(double3& contactForce, 
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

__device__ __forceinline__ int ParallelBondedContactForLevelSetParticle(double& bondNormalForce, 
double& bondTorsionalTorque, 
double3& bondShearForce, 
double3& bondBendingTorque,
double& maxNormalStress,
double& maxShearStress,
const double3 contactNormalPrev,
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
	const double3 nn = cross(contactNormalPrev, contactNormal);
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
cudaStream_t stream);