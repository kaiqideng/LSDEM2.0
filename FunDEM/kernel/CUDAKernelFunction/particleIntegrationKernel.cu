#include "particleIntegrationKernel.cuh"

__global__ void particleVelocityAngularVelocityIntegrationKernel(double3* velocity, 
double3* angularVelocity, 
const double3* force, 
const double3* torque, 
const double* invMass, 
const quaternion* orientation, 
const symMatrix* inverseInertiaTensor, 

const double3 gravity,
const double timeStep, 

const size_t num)
{
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

    double invM = invMass[idx];
	if (isZero(invM)) return;

	velocity[idx] += (force[idx] * invM + gravity) * timeStep;
	angularVelocity[idx] += (rotateInverseInertiaTensorByQuaternion(orientation[idx], inverseInertiaTensor[idx]) * torque[idx]) * timeStep;
}

__global__ void particlePositionOrientationIntegrationKernel(double3* position, 
quaternion* orientation, 
const double3* velocity, 
const double3* angularVelocity, 

const double timeStep,

const size_t num)
{
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

	position[idx] += timeStep * velocity[idx];
	orientation[idx] = quaternionIntegration(orientation[idx], angularVelocity[idx], timeStep);
}

extern "C" void launchParticleVelocityAngularVelocityIntegration(double3* velocity,
double3* angularVelocity,
const double3* force,
const double3* torque,
const double* invMass,
const quaternion* orientation,
const symMatrix* inverseInertiaTensor,

const double3 gravity,
const double timeStep,

const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    particleVelocityAngularVelocityIntegrationKernel<<<gridD, blockD, 0, stream>>>(velocity,
	angularVelocity,
	force,
	torque,
	invMass,
	orientation,
	inverseInertiaTensor,

	gravity,
	timeStep,

	num);
}

extern "C" void launchParticlePositionOrientationIntegration(double3* position,
quaternion* orientation,
const double3* velocity,
const double3* angularVelocity,

const double timeStep,

const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    particlePositionOrientationIntegrationKernel<<<gridD, blockD, 0, stream>>>(position,
	orientation,
	velocity,
	angularVelocity,

	timeStep,

	num);
}

