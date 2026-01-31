/**
 * Metal Compute Shaders for FAST-LIO GPU Acceleration
 * ===================================================
 *
 * All embarrassingly-parallel kernels run here on the GPU.
 * Data layout matches the ComputeBackend interface (flat float arrays).
 *
 * Key design:
 *   - float precision for per-point ops (transform, plane fit, residual, jacobian)
 *   - Plane fitting via explicit 3x3 normal equations (A^T*A * x = A^T*b)
 *     with Cramer's rule — no QR decomposition needed for 5x3
 *   - HTH/HTh use threadgroup reduction with float accumulation
 */

#include <metal_stdlib>
using namespace metal;

// ─── Shared types ────────────────────────────────────────────────────

struct TransformParams {
    float R_body[9];     // 3x3 column-major body-to-world rotation
    float t_body[3];     // translation
    float R_ext[9];      // 3x3 column-major LiDAR-to-IMU rotation
    float t_ext[3];      // translation
    uint  n;             // number of points
};

struct PlaneFitParams {
    uint  n;             // number of points
    uint  k;             // neighbors per point (typically 5)
    float threshold;     // max residual for valid plane
};

struct PlaneCoeffsGPU {
    float a, b, c, d;
    uint  valid;         // 0 or 1 (bool is problematic across GPU/CPU)
};

struct ResidualParams {
    uint n;
};

struct JacobianParams {
    float R_body[9];     // 3x3 column-major
    float R_ext[9];      // 3x3 column-major
    float t_ext[3];
    uint  m;             // number of valid points
    uint  extrinsic_est_en;
};

struct HTHParams {
    uint m;              // number of rows in H
};

// ─── Helper: 3x3 matrix-vector multiply (column-major) ──────────────
// Overloads for different Metal address spaces

inline float3 mat3x3_mul_impl(float m0, float m1, float m2,
                                float m3, float m4, float m5,
                                float m6, float m7, float m8, float3 v) {
    return float3(
        m0*v.x + m3*v.y + m6*v.z,
        m1*v.x + m4*v.y + m7*v.z,
        m2*v.x + m5*v.y + m8*v.z
    );
}

inline float3 mat3x3_tmul_impl(float m0, float m1, float m2,
                                 float m3, float m4, float m5,
                                 float m6, float m7, float m8, float3 v) {
    return float3(
        m0*v.x + m1*v.y + m2*v.z,
        m3*v.x + m4*v.y + m5*v.z,
        m6*v.x + m7*v.y + m8*v.z
    );
}

// Device buffer versions
inline float3 mat3x3_mul(device const float* M, float3 v) {
    return mat3x3_mul_impl(M[0],M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8], v);
}
inline float3 mat3x3_tmul(device const float* M, float3 v) {
    return mat3x3_tmul_impl(M[0],M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8], v);
}

// Constant buffer versions (for params struct members)
inline float3 mat3x3_mul(const constant float* M, float3 v) {
    return mat3x3_mul_impl(M[0],M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8], v);
}
inline float3 mat3x3_tmul(const constant float* M, float3 v) {
    return mat3x3_tmul_impl(M[0],M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8], v);
}

inline float3 cross_mul(float3 a, float3 v) {
    // skew(a) * v = a × v
    return float3(
        a.y*v.z - a.z*v.y,
        a.z*v.x - a.x*v.z,
        a.x*v.y - a.y*v.x
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 1: Batch point transformation
//   p_world = R_body * (R_ext * p_body + t_ext) + t_body
// ═══════════════════════════════════════════════════════════════════════

kernel void transform_points(
    device const float*     points_body   [[buffer(0)]],
    device       float*     points_world  [[buffer(1)]],
    constant TransformParams& params      [[buffer(2)]],
    uint tid                              [[thread_position_in_grid]])
{
    if (tid >= params.n) return;

    float3 pb = float3(points_body[tid*3], points_body[tid*3+1], points_body[tid*3+2]);

    // Two-stage transform
    float3 p_imu = mat3x3_mul(params.R_ext, pb)
                 + float3(params.t_ext[0], params.t_ext[1], params.t_ext[2]);
    float3 pw = mat3x3_mul(params.R_body, p_imu)
              + float3(params.t_body[0], params.t_body[1], params.t_body[2]);

    points_world[tid*3+0] = pw.x;
    points_world[tid*3+1] = pw.y;
    points_world[tid*3+2] = pw.z;
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 2: Batch plane fitting
//   Solve Ax = b where A is kx3, b = [-1,...,-1]
//   Using normal equations: (A^T A) x = A^T b
//   Then solve 3x3 system with Cramer's rule
// ═══════════════════════════════════════════════════════════════════════

kernel void plane_fit(
    device const float*          neighbors  [[buffer(0)]],
    device       PlaneCoeffsGPU* planes     [[buffer(1)]],
    constant PlaneFitParams&     params     [[buffer(2)]],
    uint tid                                [[thread_position_in_grid]])
{
    if (tid >= params.n) return;

    uint k = params.k;
    device const float* pts = neighbors + tid * k * 3;

    // ── Step 1: Compute scale factor for numerical stability ──
    float max_abs = 0;
    for (uint j = 0; j < k; j++) {
        max_abs = max(max_abs, abs(pts[j*3+0]));
        max_abs = max(max_abs, abs(pts[j*3+1]));
        max_abs = max(max_abs, abs(pts[j*3+2]));
    }
    float scale = (max_abs > 1e-6f) ? (1.0f / max_abs) : 1.0f;

    // ── Step 2: Build A^T*A (3x3 SPD) and A^T*b on scaled coordinates ──
    float ata00 = 0, ata01 = 0, ata02 = 0;
    float ata11 = 0, ata12 = 0, ata22 = 0;
    float atb0 = 0, atb1 = 0, atb2 = 0;

    for (uint j = 0; j < k; j++) {
        float x = pts[j*3+0] * scale;
        float y = pts[j*3+1] * scale;
        float z = pts[j*3+2] * scale;

        ata00 += x*x; ata01 += x*y; ata02 += x*z;
                       ata11 += y*y; ata12 += y*z;
                                      ata22 += z*z;
        atb0 -= x;
        atb1 -= y;
        atb2 -= z;
    }

    PlaneCoeffsGPU result;

    // ── Step 3: Cholesky decomposition of A^T*A = L*L^T ──
    // More numerically stable than Cramer's rule for SPD matrices.
    // L is lower triangular: L = [[l00,0,0],[l10,l11,0],[l20,l21,l22]]

    float l00 = sqrt(ata00);
    if (l00 < 1e-7f) {
        result.a = 0; result.b = 0; result.c = 0; result.d = 0;
        result.valid = 0;
        planes[tid] = result;
        return;
    }
    float l00_inv = 1.0f / l00;
    float l10 = ata01 * l00_inv;
    float l20 = ata02 * l00_inv;

    float diag1 = ata11 - l10 * l10;
    if (diag1 < 1e-14f) {
        result.a = 0; result.b = 0; result.c = 0; result.d = 0;
        result.valid = 0;
        planes[tid] = result;
        return;
    }
    float l11 = sqrt(diag1);
    float l11_inv = 1.0f / l11;
    float l21 = (ata12 - l20 * l10) * l11_inv;

    float diag2 = ata22 - l20 * l20 - l21 * l21;
    if (diag2 < 1e-14f) {
        result.a = 0; result.b = 0; result.c = 0; result.d = 0;
        result.valid = 0;
        planes[tid] = result;
        return;
    }
    float l22 = sqrt(diag2);
    float l22_inv = 1.0f / l22;

    // ── Step 4: Forward substitution: L * y = atb ──
    float y0 = atb0 * l00_inv;
    float y1 = (atb1 - l10 * y0) * l11_inv;
    float y2 = (atb2 - l20 * y0 - l21 * y1) * l22_inv;

    // ── Step 5: Back substitution: L^T * x = y ──
    float x2 = y2 * l22_inv;
    float x1 = (y1 - l21 * x2) * l11_inv;
    float x0 = (y0 - l10 * x1 - l20 * x2) * l00_inv;

    // ── Step 6: Convert back to original coordinates ──
    // Scaled: x0*(s*xi) + x1*(s*yi) + x2*(s*zi) + 1 = 0
    // Original: (s*x0)*xi + (s*x1)*yi + (s*x2)*zi + 1 = 0
    // normvec_orig = s*(x0,x1,x2), d_orig = 1
    // After normalizing by ||normvec_orig|| = s*||x||:
    //   a = x0/||x||, b = x1/||x||, c = x2/||x||, d = 1/(s*||x||)

    float norm = sqrt(x0*x0 + x1*x1 + x2*x2);

    if (norm < 1e-10f) {
        result.a = 0; result.b = 0; result.c = 0; result.d = 0;
        result.valid = 0;
        planes[tid] = result;
        return;
    }

    result.a = x0 / norm;
    result.b = x1 / norm;
    result.c = x2 / norm;
    result.d = 1.0f / (norm * scale);

    // ── Step 7: Check residuals ──
    uint valid = 1;
    for (uint j = 0; j < k; j++) {
        float res = result.a * pts[j*3+0]
                  + result.b * pts[j*3+1]
                  + result.c * pts[j*3+2]
                  + result.d;
        if (abs(res) > params.threshold) {
            valid = 0;
            break;
        }
    }
    result.valid = valid;
    planes[tid] = result;
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 3: Batch residual computation + validity filter
// ═══════════════════════════════════════════════════════════════════════

kernel void compute_residuals(
    device const float*          points_world [[buffer(0)]],
    device const float*          points_body  [[buffer(1)]],
    device const PlaneCoeffsGPU* planes       [[buffer(2)]],
    device       float*          residuals    [[buffer(3)]],
    device       uint8_t*        valid_mask   [[buffer(4)]],
    constant ResidualParams&     params       [[buffer(5)]],
    uint tid                                  [[thread_position_in_grid]])
{
    if (tid >= params.n) return;

    if (planes[tid].valid == 0) {
        residuals[tid] = 0.0f;
        valid_mask[tid] = 0;
        return;
    }

    float3 pw = float3(points_world[tid*3], points_world[tid*3+1], points_world[tid*3+2]);
    float3 pb = float3(points_body[tid*3], points_body[tid*3+1], points_body[tid*3+2]);

    float pd2 = planes[tid].a * pw.x + planes[tid].b * pw.y
              + planes[tid].c * pw.z + planes[tid].d;

    float body_norm = sqrt(pb.x*pb.x + pb.y*pb.y + pb.z*pb.z);
    float s = 1.0f - 0.9f * abs(pd2) / sqrt(body_norm);

    if (s > 0.9f) {
        residuals[tid] = pd2;
        valid_mask[tid] = 1;
    } else {
        residuals[tid] = 0.0f;
        valid_mask[tid] = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 4: Batch Jacobian construction
//   Output: M x 12 floats (NOT doubles — Metal doesn't support double well)
//   We'll convert to double on the CPU side when reading back
// ═══════════════════════════════════════════════════════════════════════

kernel void build_jacobian(
    device const float*       points_body [[buffer(0)]],
    device const float*       normals     [[buffer(1)]],
    device const float*       plane_dists [[buffer(2)]],
    device       float*       H           [[buffer(3)]],  // M x 12 float
    device       float*       h           [[buffer(4)]],  // M float
    constant JacobianParams&  params      [[buffer(5)]],
    uint tid                              [[thread_position_in_grid]])
{
    if (tid >= params.m) return;

    float3 pb = float3(points_body[tid*3], points_body[tid*3+1], points_body[tid*3+2]);
    float3 norm = float3(normals[tid*3], normals[tid*3+1], normals[tid*3+2]);

    // point_imu = R_ext * p_body + t_ext
    float3 point_imu = mat3x3_mul(params.R_ext, pb)
                     + float3(params.t_ext[0], params.t_ext[1], params.t_ext[2]);

    // C = R_body^T * normal
    float3 C = mat3x3_tmul(params.R_body, norm);

    // A = skew(point_imu) * C = point_imu × C
    float3 A = cross_mul(point_imu, C);

    device float* row = H + tid * 12;
    row[0] = norm.x;
    row[1] = norm.y;
    row[2] = norm.z;
    row[3] = A.x;
    row[4] = A.y;
    row[5] = A.z;

    if (params.extrinsic_est_en) {
        // B = skew(p_body) * R_ext^T * C
        float3 R_ext_T_C = mat3x3_tmul(params.R_ext, C);
        float3 B = cross_mul(pb, R_ext_T_C);
        row[6]  = B.x;
        row[7]  = B.y;
        row[8]  = B.z;
        row[9]  = C.x;
        row[10] = C.y;
        row[11] = C.z;
    } else {
        row[6] = 0; row[7] = 0; row[8] = 0;
        row[9] = 0; row[10] = 0; row[11] = 0;
    }

    h[tid] = -plane_dists[tid];
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 5: H^T * H partial reduction
//   Each threadgroup reduces a chunk of rows into a 12x12 partial sum.
//   The CPU does final reduction across threadgroups.
//
//   H is M x 12 float (row-major). 
//   Output: num_groups x 78 float (upper triangle of 12x12 symmetric)
// ═══════════════════════════════════════════════════════════════════════

#define HTH_GROUP_SIZE 256

// Store upper triangle: 12*13/2 = 78 elements
#define HTH_UPPER_SIZE 78

kernel void hth_partial(
    device const float*   H           [[buffer(0)]],
    device       float*   partials    [[buffer(1)]],  // num_groups x 78
    constant HTHParams&   params      [[buffer(2)]],
    uint tid                          [[thread_position_in_grid]],
    uint lid                          [[thread_position_in_threadgroup]],
    uint gid                          [[threadgroup_position_in_grid]])
{
    // Each thread accumulates one row's contribution
    threadgroup float shared[HTH_GROUP_SIZE][12];

    // Load this thread's H row into shared memory
    float row[12];
    if (tid < params.m) {
        for (uint c = 0; c < 12; c++) {
            row[c] = H[tid * 12 + c];
            shared[lid][c] = row[c];
        }
    } else {
        for (uint c = 0; c < 12; c++) {
            row[c] = 0;
            shared[lid][c] = 0;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes partial sums for some elements of the upper triangle
    // We have 78 elements and up to 256 threads — each thread handles ~1 element
    // But for simplicity, thread 0-77 each compute one element by summing over all
    // rows in this threadgroup.

    threadgroup float partial_results[HTH_UPPER_SIZE];

    if (lid < HTH_UPPER_SIZE) {
        // Map linear index to (i, j) in upper triangle
        uint idx = lid;
        uint i = 0, j = 0;
        uint cumsum = 0;
        for (uint row_idx = 0; row_idx < 12; row_idx++) {
            uint row_len = 12 - row_idx;
            if (idx < cumsum + row_len) {
                i = row_idx;
                j = row_idx + (idx - cumsum);
                break;
            }
            cumsum += row_len;
        }

        float sum = 0;
        for (uint t = 0; t < HTH_GROUP_SIZE; t++) {
            sum += shared[t][i] * shared[t][j];
        }
        partial_results[lid] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write out
    if (lid < HTH_UPPER_SIZE) {
        partials[gid * HTH_UPPER_SIZE + lid] = partial_results[lid];
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 6: H^T * h partial reduction
//   Each threadgroup reduces a chunk into a 12-element partial sum.
// ═══════════════════════════════════════════════════════════════════════

#define HTH_GROUP_SIZE_HTh 256

kernel void hth_partial_vec(
    device const float*   H           [[buffer(0)]],
    device const float*   h_vec       [[buffer(1)]],
    device       float*   partials    [[buffer(2)]],  // num_groups x 12
    constant HTHParams&   params      [[buffer(3)]],
    uint tid                          [[thread_position_in_grid]],
    uint lid                          [[thread_position_in_threadgroup]],
    uint gid                          [[threadgroup_position_in_grid]])
{
    threadgroup float shared_h[HTH_GROUP_SIZE_HTh];
    threadgroup float shared_H[HTH_GROUP_SIZE_HTh][12];

    if (tid < params.m) {
        shared_h[lid] = h_vec[tid];
        for (uint c = 0; c < 12; c++) {
            shared_H[lid][c] = H[tid * 12 + c];
        }
    } else {
        shared_h[lid] = 0;
        for (uint c = 0; c < 12; c++) {
            shared_H[lid][c] = 0;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threads 0-11 each reduce one element
    if (lid < 12) {
        float sum = 0;
        for (uint t = 0; t < HTH_GROUP_SIZE_HTh; t++) {
            sum += shared_H[t][lid] * shared_h[t];
        }
        partials[gid * 12 + lid] = sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 7: Batch point undistortion
// ═══════════════════════════════════════════════════════════════════════

struct UndistortParams {
    float R_end[9];       // end-of-scan IMU rotation
    float t_end[3];       // end-of-scan IMU position (as translation in RigidTransform)
    float R_ext[9];       // LiDAR-to-IMU rotation
    float t_ext[3];       // LiDAR-to-IMU translation
    uint  n;
    uint  num_segments;
};

// Rodrigues formula for small rotation: exp(w)
inline float3x3 exp_so3(float3 w) {
    float theta = length(w);
    if (theta < 1e-6f) {
        return float3x3(1,0,0, 0,1,0, 0,0,1);
    }
    float3 axis = w / theta;
    float s = sin(theta);
    float c = cos(theta);
    float t = 1.0f - c;

    return float3x3(
        t*axis.x*axis.x + c,        t*axis.x*axis.y + s*axis.z,  t*axis.x*axis.z - s*axis.y,
        t*axis.x*axis.y - s*axis.z, t*axis.y*axis.y + c,         t*axis.y*axis.z + s*axis.x,
        t*axis.x*axis.z + s*axis.y, t*axis.y*axis.z - s*axis.x,  t*axis.z*axis.z + c
    );
}

// Load a 3x3 column-major matrix from flat float array into float3x3
// Metal float3x3 is column-major by construction: float3x3(col0, col1, col2)
inline float3x3 load_mat3(device const float* m) {
    return float3x3(
        float3(m[0], m[1], m[2]),    // column 0
        float3(m[3], m[4], m[5]),    // column 1
        float3(m[6], m[7], m[8])     // column 2
    );
}

inline float3x3 load_mat3_const(const constant float* m) {
    return float3x3(
        float3(m[0], m[1], m[2]),
        float3(m[3], m[4], m[5]),
        float3(m[6], m[7], m[8])
    );
}

kernel void undistort_points(
    device       float*          points      [[buffer(0)]],
    device const float*          timestamps  [[buffer(1)]],
    device const float*          seg_R       [[buffer(2)]],  // S x 9
    device const float*          seg_vel     [[buffer(3)]],  // S x 3
    device const float*          seg_pos     [[buffer(4)]],  // S x 3
    device const float*          seg_acc     [[buffer(5)]],  // S x 3
    device const float*          seg_angvel  [[buffer(6)]],  // S x 3
    device const float*          seg_t_start [[buffer(7)]],  // S
    constant UndistortParams&    params      [[buffer(8)]],
    uint tid                                 [[thread_position_in_grid]])
{
    if (tid >= params.n) return;

    float t = timestamps[tid];

    // Find segment (linear scan from end — typically < 20 segments)
    int seg = (int)params.num_segments - 1;
    for (int s = (int)params.num_segments - 1; s >= 0; s--) {
        if (t >= seg_t_start[s]) {
            seg = s;
            break;
        }
    }

    float dt = t - seg_t_start[seg];

    float3x3 R_seg = load_mat3(seg_R + seg * 9);
    float3 vel = float3(seg_vel[seg*3], seg_vel[seg*3+1], seg_vel[seg*3+2]);
    float3 pos = float3(seg_pos[seg*3], seg_pos[seg*3+1], seg_pos[seg*3+2]);
    float3 acc = float3(seg_acc[seg*3], seg_acc[seg*3+1], seg_acc[seg*3+2]);
    float3 angvel = float3(seg_angvel[seg*3], seg_angvel[seg*3+1], seg_angvel[seg*3+2]);

    float3x3 R_i = R_seg * exp_so3(angvel * dt);

    float3 P_i = float3(points[tid*3], points[tid*3+1], points[tid*3+2]);

    float3 pos_end = float3(params.t_end[0], params.t_end[1], params.t_end[2]);
    float3 T_ei = pos + vel * dt + 0.5f * acc * dt * dt - pos_end;

    float3x3 R_end = load_mat3_const(params.R_end);
    float3x3 R_ext = load_mat3_const(params.R_ext);
    float3 t_ext = float3(params.t_ext[0], params.t_ext[1], params.t_ext[2]);

    float3x3 R_end_T = transpose(R_end);
    float3x3 R_ext_T = transpose(R_ext);

    float3 P_comp = R_ext_T * (R_end_T * (R_i * (R_ext * P_i + t_ext) + T_ei) - t_ext);

    points[tid*3+0] = P_comp.x;
    points[tid*3+1] = P_comp.y;
    points[tid*3+2] = P_comp.z;
}
