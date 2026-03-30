#include "particleIntegrationKernel.cuh"

__global__ void particleVelocityAngularVelocityIntegrationKernel(double3* velocity, 
double3* angularVelocity, 
const double3* force, 
const double3* torque, 
const double* invMass, 
const quaternion* orientation, 
const symMatrix* inverseInertiaTensor, 

const double3 g,
const double dt, 

const size_t num)
{
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

    double invM = invMass[idx];
	if (isZero(invM)) return;

	velocity[idx] += (force[idx] * invM + g) * dt;
	angularVelocity[idx] += (rotateInverseInertiaTensor(orientation[idx], inverseInertiaTensor[idx]) * torque[idx]) * dt;
}

__global__ void particlePositionOrientationIntegrationKernel(double3* position, 
quaternion* orientation, 
const double3* velocity, 
const double3* angularVelocity, 

const double dt,

const size_t num)
{
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

	position[idx] += dt * velocity[idx];
	orientation[idx] = quaternionRotate(orientation[idx], angularVelocity[idx], dt);
}

extern "C" void launchParticle1stHalfIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* invMass, 
quaternion* orientation, 
symMatrix* inverseInertiaTensor, 

const double3 gravity, 
const double halfTimeStep,

const size_t num,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	particleVelocityAngularVelocityIntegrationKernel <<<gridD, blockD, 0, stream>>> (velocity, 
	angularVelocity, 
	force, 
	torque, 
	invMass, 
    orientation, 
    inverseInertiaTensor, 
	gravity,
	halfTimeStep,
	num);

	particlePositionOrientationIntegrationKernel <<<gridD, blockD, 0, stream>>> (position, 
	orientation, 
	velocity, 
	angularVelocity, 
	2.0 * halfTimeStep,
	num);
}

extern "C" void launchParticle2ndHalfIntegration(double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* invMass, 
quaternion* orientation, 
symMatrix* inverseInertiaTensor, 

const double3 gravity, 
const double halfTimeStep,

const size_t num,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	particleVelocityAngularVelocityIntegrationKernel <<<gridD, blockD, 0, stream>>> (velocity, 
	angularVelocity, 
	force, 
	torque, 
	invMass, 
    orientation, 
    inverseInertiaTensor,
	gravity,
	halfTimeStep,
	num);
}

