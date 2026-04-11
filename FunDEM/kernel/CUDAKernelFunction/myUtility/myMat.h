#pragma once
#include "myQua.h" // Assumed to contain quaternion definition

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

/**
 * @brief Symmetric 3x3 Matrix structure.
 * Stores only the unique 6 elements (xx, yy, zz, xy, xz, yz).
 * Used primarily for Inertia Tensors and Stress/Strain tensors.
 */
struct symMatrix
{
    double xx, yy, zz;
    double xy, xz, yz;

    // Default constructor
    HOST_DEVICE symMatrix() 
        : xx(0.0), yy(0.0), zz(0.0), xy(0.0), xz(0.0), yz(0.0) {}

    // Value constructor
    HOST_DEVICE symMatrix(double _xx, double _yy, double _zz, double _xy, double _xz, double _yz)
        : xx(_xx), yy(_yy), zz(_zz), xy(_xy), xz(_xz), yz(_yz) {}
};

// Helper for construction (kept for compatibility, but constructor is preferred)
HOST_DEVICE inline symMatrix make_symMatrix(double xx, double yy, double zz, double xy, double xz, double yz)
{
    return symMatrix(xx, yy, zz, xy, xz, yz);
}

// --------------------------------------------------------
// Basic Math Operations
// OPTIMIZATION: Pass-by-Value to use registers
// --------------------------------------------------------

HOST_DEVICE inline double norm(symMatrix m)
{
    // Frobenius norm for symmetric matrix: sqrt(sum(diag^2) + 2*sum(off_diag^2))
    return sqrt(m.xx * m.xx + m.yy * m.yy + m.zz * m.zz + 
                2.0 * (m.xy * m.xy + m.xz * m.xz + m.yz * m.yz));
}

HOST_DEVICE inline symMatrix deviatoric(symMatrix m)
{
    // Optimization: Multiply by inverse of 3 instead of dividing
    double tr = (m.xx + m.yy + m.zz) * 0.3333333333333333;
    return symMatrix(m.xx - tr, m.yy - tr, m.zz - tr, m.xy, m.xz, m.yz);
}

HOST_DEVICE inline symMatrix operator+(symMatrix m1, symMatrix m2)
{
    return symMatrix(m1.xx + m2.xx, m1.yy + m2.yy, m1.zz + m2.zz,
                     m1.xy + m2.xy, m1.xz + m2.xz, m1.yz + m2.yz);
}

HOST_DEVICE inline symMatrix operator-(symMatrix m1, symMatrix m2)
{
    return symMatrix(m1.xx - m2.xx, m1.yy - m2.yy, m1.zz - m2.zz,
                     m1.xy - m2.xy, m1.xz - m2.xz, m1.yz - m2.yz);
}

HOST_DEVICE inline symMatrix operator*(symMatrix m, double a)
{
    return symMatrix(m.xx * a, m.yy * a, m.zz * a, m.xy * a, m.xz * a, m.yz * a);
}

HOST_DEVICE inline symMatrix operator*(double a, symMatrix m)
{
    return symMatrix(m.xx * a, m.yy * a, m.zz * a, m.xy * a, m.xz * a, m.yz * a);
}

// --------------------------------------------------------
// Matrix-Vector Multiplication
// --------------------------------------------------------

HOST_DEVICE inline double3 operator*(symMatrix m, double3 v)
{
    return make_double3(
        m.xx * v.x + m.xy * v.y + m.xz * v.z,
        m.xy * v.x + m.yy * v.y + m.yz * v.z,
        m.xz * v.x + m.yz * v.y + m.zz * v.z
    );
}

HOST_DEVICE inline double3 operator*(double3 v, symMatrix m)
{
    // Symmetric matrix multiplication is commutative with respect to the vector dot product structure
    return make_double3(
        m.xx * v.x + m.xy * v.y + m.xz * v.z,
        m.xy * v.x + m.yy * v.y + m.yz * v.z,
        m.xz * v.x + m.yz * v.y + m.zz * v.z
    );
}

// --------------------------------------------------------
// Tensor Transformations
// --------------------------------------------------------

/**
 * Rotates an inverse inertia tensor (or any symmetric matrix) by a quaternion.
 * Operation: I_world = R * I_local * R^T
 */
HOST_DEVICE inline symMatrix rotateInverseInertiaTensorByQuaternion(quaternion q, symMatrix invI)
{
    // 1. Construct Rotation Matrix elements from Quaternion
    // R = [ a b c ]
    //     [ d e f ]
    //     [ g h i ]
    
    double q0 = q.q0, q1 = q.q1, q2 = q.q2, q3 = q.q3;
    
    // Optimization: Pre-calculate common squares
    double q11 = 2 * q1 * q1;
    double q22 = 2 * q2 * q2;
    double q33 = 2 * q3 * q3;
    
    double a = 1.0 - q22 - q33;
    double b = 2 * (q1 * q2 - q0 * q3);
    double c = 2 * (q1 * q3 + q0 * q2);
    
    double d = 2 * (q1 * q2 + q0 * q3);
    double e = 1.0 - q11 - q33;
    double f = 2 * (q2 * q3 - q0 * q1);
    
    double g = 2 * (q1 * q3 - q0 * q2);
    double h = 2 * (q2 * q3 + q0 * q1);
    double i = 1.0 - q11 - q22;

    // 2. Perform Similarity Transformation: M = R * invI * R^T
    // Explicit expansion is preferred here for compiler optimization over loops
    
    // Helper terms for row-matrix multiplication
    double Row0_x = invI.xx * a + invI.xy * b + invI.xz * c;
    double Row0_y = invI.xy * a + invI.yy * b + invI.yz * c;
    double Row0_z = invI.xz * a + invI.yz * b + invI.zz * c;

    double Row1_x = invI.xx * d + invI.xy * e + invI.xz * f;
    double Row1_y = invI.xy * d + invI.yy * e + invI.yz * f;
    double Row1_z = invI.xz * d + invI.yz * e + invI.zz * f;

    double Row2_x = invI.xx * g + invI.xy * h + invI.xz * i;
    double Row2_y = invI.xy * g + invI.yy * h + invI.yz * i;
    double Row2_z = invI.xz * g + invI.yz * h + invI.zz * i;

    // Final result calculation
    return symMatrix(
        Row0_x * a + Row0_y * b + Row0_z * c, // xx
        Row1_x * d + Row1_y * e + Row1_z * f, // yy
        Row2_x * g + Row2_y * h + Row2_z * i, // zz
        Row0_x * d + Row0_y * e + Row0_z * f, // xy
        Row0_x * g + Row0_y * h + Row0_z * i, // xz
        Row1_x * g + Row1_y * h + Row1_z * i  // yz
    );
}

/**
 * Computes the inverse of a 3x3 symmetric matrix.
 */
HOST_DEVICE inline symMatrix inverse(symMatrix A)
{
    // Compute cofactors (optimized for symmetry)
    // determinant
    const double det =
          A.xx * (A.yy * A.zz - A.yz * A.yz)
        + A.xy * (A.yz * A.xz - A.xy * A.zz)
        + A.xz * (A.xy * A.yz - A.yy * A.xz);

    // Protection: near-singular matrix
    // Pick a tolerance that matches your scale. 1e-30 is ok for "normalized-ish" matrices.
    // If your matrix entries are ~1e6, you need a larger threshold.
    const double detTol = 1e-30;
    if (fabs(det) < detTol)
    {
        // Option 1: return zero => "no response" (often good for fixed/locked objects)
        return symMatrix(0,0,0, 0,0,0);

        // Option 2: return identity => safer for generic math, but physically questionable for inertia
        // return symMatrix(1,1,1, 0,0,0);
    }

    const double det_inv = 1.0 / det;

    return symMatrix(
         (A.yy * A.zz - A.yz * A.yz) * det_inv, // xx
         (A.xx * A.zz - A.xz * A.xz) * det_inv, // yy
         (A.xx * A.yy - A.xy * A.xy) * det_inv, // zz
        -(A.xy * A.zz - A.xz * A.yz) * det_inv, // xy
         (A.xy * A.yz - A.xz * A.yy) * det_inv, // xz
        -(A.xx * A.yz - A.xz * A.xy) * det_inv  // yz
    );
}