#include "globalDamping.cuh"
#include "kernel/CUDAKernelFunction/myUtility/myVec.h"

__global__ void addGlobalDampingForceTorque(double3* force,
double3* torque,
const double3* velocity,
const double3* angularVelocity,
const double dampingCoefficient,
const size_t num)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

    double3 f = force[idx], t = torque[idx];
    force[idx] -= dampingCoefficient * length(f) * normalize(velocity[idx]);
    torque[idx] -= dampingCoefficient * length(t) * normalize(angularVelocity[idx]);
}

extern "C" void launchAddGlobalDampingForceTorque(double3* force,
double3* torque,
const double3* velocity,
const double3* angularVelocity,
const double dampingCoefficient,
const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    addGlobalDampingForceTorque <<<gridD, blockD, 0, stream>>> (force, 
    torque, 
    velocity, 
    angularVelocity, 
    dampingCoefficient, 
    num);
}

__global__ void addGlobalConstantForceTorque(double3* force_p,
double3* torque_p,
const double3 constantForce, 
const double3 constantTorque, 
const size_t index)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1) return;

    force_p[index] += constantForce;
    force_p[index] += constantTorque;
}

extern "C" void launchAddGlobalConstantForceTorque(double3* force_p,
double3* torque_p,
const double3 constantForce, 
const double3 constantTorque, 
const size_t index, 
cudaStream_t stream)
{
    addGlobalConstantForceTorque <<<1, 1, 0, stream>>> (force_p, 
    torque_p, 
    constantForce, 
    constantTorque, 
    index);
}
