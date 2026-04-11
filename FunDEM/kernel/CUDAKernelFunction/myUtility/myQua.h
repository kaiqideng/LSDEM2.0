#pragma once
#include "myVec.h" // Assumes double3, dot, cross, etc. are defined here

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

struct quaternion {
    double q0; // Scalar part (w)
    double q1; // Vector part x
    double q2; // Vector part y
    double q3; // Vector part z
};

// Constructor helper
HOST_DEVICE inline quaternion make_quaternion(double q0, double q1, double q2, double q3) {
    quaternion q;
    q.q0 = q0; q.q1 = q1; q.q2 = q2; q.q3 = q3;
    return q;
}

// --------------------------------------------------------
// Basic Operators
// OPTIMIZATION: Use Pass-by-Value (registers) instead of Reference (memory)
// --------------------------------------------------------

HOST_DEVICE inline quaternion operator+(quaternion a, quaternion b) {
    return make_quaternion(a.q0 + b.q0, a.q1 + b.q1, a.q2 + b.q2, a.q3 + b.q3);
}

HOST_DEVICE inline quaternion operator*(quaternion q, double c) {
    return make_quaternion(q.q0 * c, q.q1 * c, q.q2 * c, q.q3 * c);
}

HOST_DEVICE inline quaternion operator*(double c, quaternion q) {
    return make_quaternion(q.q0 * c, q.q1 * c, q.q2 * c, q.q3 * c);
}

// Optimization: Multiply by inverse instead of division
HOST_DEVICE inline quaternion operator/(quaternion q, double c) {
    double inv = 1.0 / c;
    return make_quaternion(q.q0 * inv, q.q1 * inv, q.q2 * inv, q.q3 * inv);
}

// --------------------------------------------------------
// Math Helper Functions
// --------------------------------------------------------

HOST_DEVICE inline quaternion normalize(quaternion q) {
    double lenSq = q.q0 * q.q0 + q.q1 * q.q1 + q.q2 * q.q2 + q.q3 * q.q3;
    if (isZero(lenSq)) return q; // Prevent NaN

#ifdef __CUDA_ARCH__
    double invLen = rsqrt(lenSq); // Use hardware intrinsic on GPU
#else
    double invLen = 1.0 / sqrt(lenSq);
#endif

    return make_quaternion(q.q0 * invLen, q.q1 * invLen, q.q2 * invLen, q.q3 * invLen);
}

// Returns the conjugate of the quaternion (q0, -q1, -q2, -q3)
HOST_DEVICE inline quaternion conjugate(quaternion q) {
    return make_quaternion(q.q0, -q.q1, -q.q2, -q.q3);
}

// --------------------------------------------------------
// Physics Integration
// --------------------------------------------------------

/**
 * Updates quaternion orientation based on angular velocity.
 * Formula: q_new = q_old + 0.5 * q_old * omega * dt
 */
HOST_DEVICE inline quaternion quaternionIntegration(quaternion q, double3 angularVelocity, double timeStep)
{
    // ω is in WORLD frame.
    // Explicit Euler on: q_dot = 0.5 * (0, ω) ⊗ q
    // Pre-multiply angular velocity by timestep and 0.5
    double3 v = angularVelocity * (0.5 * timeStep);

    // This is explicitly expanded to avoid creating temporary quaternion objects
    quaternion deltaQ;
    deltaQ.q0 = -q.q1 * v.x - q.q2 * v.y - q.q3 * v.z;
    deltaQ.q1 =  q.q0 * v.x + q.q3 * v.y - q.q2 * v.z;
    deltaQ.q2 = -q.q3 * v.x + q.q0 * v.y + q.q1 * v.z;
    deltaQ.q3 =  q.q2 * v.x - q.q1 * v.y + q.q0 * v.z;

    // Add and normalize
    return normalize(q + deltaQ);
}

// --------------------------------------------------------
// Vector Rotation
// --------------------------------------------------------

/**
 * Rotates a vector v by quaternion q.
 * Optimization: Uses vector algebra v' = v + 2*cross(q_vec, cross(q_vec, v) + q_scalar*v)
 * This is faster than building the full 3x3 rotation matrix.
 */
HOST_DEVICE inline double3 rotateVectorByQuaternion(quaternion q, double3 v)
{
    // Extract vector part of quaternion
    double3 q_vec = make_double3(q.q1, q.q2, q.q3);
    
    // t = 2 * cross(q_vec, v)
    double3 t = cross(q_vec, v) * 2.0;

    // v' = v + q0 * t + cross(q_vec, t)
    return v + (t * q.q0) + cross(q_vec, t);
}

/**
 * Reverse rotates a vector (rotates by the inverse of q).
 * Mathematically equivalent to rotating by the conjugate of q.
 */
HOST_DEVICE inline double3 reverseRotateVectorByQuaternion(double3 v, quaternion q)
{
    // Use the conjugate (negate vector part) to reverse rotation
    // We inline the logic of rotateVectorByQuaternion here but with -q_vec
    
    double3 q_vec = make_double3(-q.q1, -q.q2, -q.q3); // Conjugate vector part
    
    double3 t = cross(q_vec, v) * 2.0;
    return v + (t * q.q0) + cross(q_vec, t);
}