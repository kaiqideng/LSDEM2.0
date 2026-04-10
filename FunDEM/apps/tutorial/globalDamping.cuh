#pragma once
#include <cstddef>
#include <driver_types.h>
#include <vector_types.h>

extern "C" void launchAddGlobalDampingForceTorque(double3* force,
double3* torque,
const double3* velocity,
const double3* angularVelocity,
const double dampingCoefficient,
const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchAddGlobalConstantForceTorque(double3* force_p,
double3* torque_p,
const double3 constantForce, 
const double3 constantTorque, 
const size_t index, 
cudaStream_t stream);