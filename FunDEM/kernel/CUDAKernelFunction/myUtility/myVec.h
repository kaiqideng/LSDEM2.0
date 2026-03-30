#pragma once
#include <cuda_runtime.h> // Includes vector_types.h and CUDA definitions
#include <cmath>

// Ensure HOST_DEVICE is correctly defined for both NVCC and standard compilers
#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

// Use constexpr for better compile-time optimization
HOST_DEVICE constexpr double pi()
{
    return 3.14159265358979323846;
}

// ---------------------------------------------------------------------
// Global "physical zero" threshold
//   Any |x| < kEps is treated as 0
// ---------------------------------------------------------------------
HOST_DEVICE __inline__ bool isZero(const double x)
{
    return fabs(x) <= 1.0e-20;
}

// --------------------------------------------------------
// Basic Operator Optimizations
// STRATEGY: Use Pass-by-Value instead of Reference.
// double3 is small (24 bytes) and fits in GPU registers.
// References cause expensive global/local memory access on GPUs.
// --------------------------------------------------------

HOST_DEVICE inline double3 operator+(double3 a, double3 b) 
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

HOST_DEVICE inline double3 operator-(double3 a, double3 b) 
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

HOST_DEVICE inline double3 operator*(double3 a, double s) 
{
    return make_double3(a.x * s, a.y * s, a.z * s);
}

HOST_DEVICE inline double3 operator*(double s, double3 a) 
{
    return make_double3(a.x * s, a.y * s, a.z * s);
}

// Optimization: Convert expensive division to multiplication by reciprocal
HOST_DEVICE inline double3 operator/(double3 a, double s) 
{
    double inv = 1.0 / s;
    return make_double3(a.x * inv, a.y * inv, a.z * inv);
}

HOST_DEVICE inline double3& operator+=(double3& a, double3 b) 
{
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

HOST_DEVICE inline double3& operator-=(double3& a, double3 b) 
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}

HOST_DEVICE inline double3& operator*=(double3& a, double s) 
{
    a.x *= s; a.y *= s; a.z *= s;
    return a;
}

HOST_DEVICE inline double3& operator/=(double3& a, double s) 
{
    double inv = 1.0 / s;
    a.x *= inv; a.y *= inv; a.z *= inv;
    return a;
}

HOST_DEVICE inline double3 operator-(double3 a) 
{
    return make_double3(-a.x, -a.y, -a.z);
}

// --------------------------------------------------------
// Geometric Function Optimizations
// --------------------------------------------------------

HOST_DEVICE inline double dot(double3 a, double3 b) 
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

HOST_DEVICE inline double3 cross(double3 a, double3 b) 
{
    return make_double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

HOST_DEVICE inline double lengthSquared(double3 v) 
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

HOST_DEVICE inline double length(double3 v) 
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Optimization: Use rsqrt (Reciprocal Square Root)
// This avoids one expensive division and one square root.
HOST_DEVICE inline double3 normalize(double3 v) 
{
    double lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
    
    // Avoid division by zero
    if (isZero(lenSq)) return make_double3(0.0, 0.0, 0.0);
    
#ifdef __CUDA_ARCH__
    // Use hardware intrinsic rsqrt on GPU
    double invLen = rsqrt(lenSq);
#else
    // CPU fallback
    double invLen = 1.0 / sqrt(lenSq);
#endif
    
    return make_double3(v.x * invLen, v.y * invLen, v.z * invLen);
}

// --------------------------------------------------------
// Rodrigues' Rotation Formula Optimization
// --------------------------------------------------------

HOST_DEVICE inline double3 rotateVector(const double3 v, const double3 omega) 
{
    const double theta2 = dot(omega, omega);

    // Small-angle threshold (tune if you want)
    if (theta2 < 1e-20)
    {
        // 2nd-order expansion: exp([omega]x) v
        const double3 oxv  = cross(omega, v);
        const double3 oxoxv = cross(omega, oxv);
        return v + oxv + 0.5 * oxoxv;
    }

    // theta = sqrt(theta2) but avoid sqrt if possible:
    double invTheta;
#ifdef __CUDA_ARCH__
    // rsqrt is fast; for double, use rsqrt() if available in your toolchain,
    // otherwise fall back to 1.0/sqrt(theta2).
    invTheta = rsqrt(theta2);              // theta^-1
#else
    invTheta = 1.0 / std::sqrt(theta2);
#endif
    const double theta = theta2 * invTheta; // sqrt(theta2)

    double s, c;
#ifdef __CUDA_ARCH__
    sincos(theta, &s, &c);
#else
    s = std::sin(theta);
    c = std::cos(theta);
#endif

    const double A = s * invTheta; // sin(theta)/theta
    const double B = (1.0 - c) * (invTheta * invTheta); // (1-cos)/theta^2

    const double3 oxv   = cross(omega, v);
    const double3 oxoxv = cross(omega, oxv);

    return v + A * oxv + B * oxoxv;
}

// Rotate vector v around axis by angle theta,
// given sin(theta). Assumes cos(theta) >= 0 (i.e., |theta| <= pi/2 typically).
HOST_DEVICE inline double3 rotateVectorAxisSin(const double3 v, const double3 axis, const double sinTheta)
{
    // Normalize axis (safe even if axis not unit)
    const double axis2 = dot(axis, axis);
    if (isZero(axis2)) return v;

#ifdef __CUDA_ARCH__
    const double invAxis = rsqrt(axis2);
#else
    const double invAxis = 1.0 / std::sqrt(axis2);
#endif
    const double3 k = axis * invAxis;

    // cos(theta) from sin(theta) with non-negative assumption
    const double s = sinTheta;
    const double c = std::sqrt(fmax(0.0, 1.0 - s * s));
    const double oneMinusC = 1.0 - c;

    // Rodrigues: v' = v*c + (k x v)*s + k*(k·v)*(1-c)
    const double3 kxv = cross(k, v);
    const double kDotV = dot(k, v);

    return v * c + kxv * s + k * (kDotV * oneMinusC);
}

// Symmetric pair -> packed upper-triangular index (with fallback to last slot).
// - (a,b) and (b,a) map to the same index.
// - If a/b out of range, returns (capacity-1) as fallback.
HOST_DEVICE inline int upperTriangularIndex(int row, int col, int n, int capacity)
{
    if (n <= 0 || capacity <= 0) return 0;
    if (row < 0 || col < 0 || row >= n || col >= n) return capacity - 1;

    int i = row;
    int j = col;
    if (i > j) { int t = i; i = j; j = t; }

    long long idx = (static_cast<long long>(i) * (2LL * n - i + 1LL)) / 2LL + static_cast<long long>(j - i);

    if (idx < 0) idx = 0;
    if (idx >= capacity) idx = capacity - 1;

    return static_cast<int>(idx);
}

// 3D index (ix,iy,iz) -> 1D linear index (row-major)
HOST_DEVICE inline int linearIndex3D(const int3 ijk, const int3 dims)
{
    return ijk.z * dims.y * dims.x + ijk.y * dims.x + ijk.x;
}