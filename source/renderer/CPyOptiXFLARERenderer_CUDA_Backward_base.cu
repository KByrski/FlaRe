#define _USE_MATH_DEFINES

#include "CPyOptiXFLARERenderer_CUDA.h"
#include "Header.cuh"

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// *** *** *** *** ***

// !!! !!! !!!
__device__ int counter;
// !!! !!! !!!

// *** *** *** *** ***

const int threads_per_block = 128;

// *** *** *** *** ***

// Paper: analytic backward pass for generalized-Gaussian compositing, including
// gradients of primitive geometry, opacity, and shape.
static __global__ void __launch_bounds__(threads_per_block, 1) Backward_base_CUDA_kernel(
	float3 *O_ptr,
	float3 *v_ptr,
	int *ray_indices,
	float *t_hit_ptr,
	int *indices_hit_ptr,
	float *t_min_ptr,
	float *T_ptr,
	int *is_active_ptr,

	int number_of_rays,

	float3 bg_color,

	float3 *m_ptr,
	float2 *s_ptr,
	float4 *q_ptr,
	float4 *RGBA_ptr,
	float *kappa_ptr,

	float3 *I_ptr,
	float3 *dL_dI_ptr,
	float *dL_dRGB_ptr,
	float *dL_dA_ptr,
	float *dL_d_kappa_ptr,
	float *dL_dm_ptr,
	float *dL_ds_ptr,
	float *dL_dq_ptr,

	float T_threshold,

	float4 *depth_reg_accums_ptr,
	float4 *depth_normal_reg_prefix_sums_ptr,
	float lambda_depth,
	float reg_a,
	float reg_b,

	float2 *depth_and_index_ptr,
	float3 *surface_normal_ptr,
	float4 *normal_reg_accums_ptr,
	float lambda_normal
) {
	while (true) {
		int tid_start;
		if ((threadIdx.x & 31) == 0)
			tid_start = atomicAdd(&counter, 32);
		tid_start = __shfl_sync(-1, tid_start, 0);
		if (tid_start >= number_of_rays) return;

		int tid = tid_start + (threadIdx.x & 31);
		int tid_clamped = (tid >= number_of_rays) ? number_of_rays - 1 : tid;

		// *****************************************************************************************

		int pixel_ind = ray_indices[tid_clamped];

		float3 O = O_ptr[pixel_ind];
		float3 v = v_ptr[pixel_ind];

		// *****************************************************************************************

		float T = T_ptr[pixel_ind]; // !!! !!! !!!
		float3 RGB_aggregated = I_ptr[pixel_ind]; // !!! !!! !!!

		// *****************************************************************************************

		// Depth regularization
		float4 depth_reg_accums = depth_reg_accums_ptr[pixel_ind]; // !!! !!! !!!

		// Depth and normal regularization
		float4 depth_normal_reg_prefix_sums = depth_normal_reg_prefix_sums_ptr[pixel_ind]; // !!! !!! !!!

		// Normal regularization
		float4 normal_reg_accums = normal_reg_accums_ptr[pixel_ind];
		float3 N_median = surface_normal_ptr[pixel_ind];
		bool normal_target_valid = __fmaf_rn(N_median.x, N_median.x,
			__fmaf_rn(N_median.y, N_median.y, N_median.z * N_median.z)) > 0.0f;

		// *****************************************************************************************

		int ind; // !!! !!! !!!
		float t_hit; // !!! !!! !!!

		// *****************************************************************************************

		// **************************************** GRADIENT ***************************************
		float3 dL_dI = dL_dI_ptr[pixel_ind];
		float3 dL_dA_RGB = make_float3(-RGB_aggregated.x, -RGB_aggregated.y, -RGB_aggregated.z);
		// **************************************** GRADIENT ***************************************

		#pragma unroll
		for (int i = 0; i < 16; ++i) {
			ind = __ldcs(indices_hit_ptr + tid_clamped + (i * number_of_rays));
			t_hit = __ldcs(t_hit_ptr + tid_clamped + (i * number_of_rays));

			int ind_clamped = (ind == -1) ? 0 : ind;

			// *************************************************************************************

			float3 m_param = m_ptr[ind_clamped];
			float2 s_param = s_ptr[ind_clamped];
			float4 q_param = q_ptr[ind_clamped];

			// *************************************************************************************

			s_param = make_float2(__expf(-s_param.x), __expf(-s_param.y));

			// *************************************************************************************

			float aa = q_param.x * q_param.x;
			float bb = q_param.y * q_param.y;
			float cc = q_param.z * q_param.z;
			float dd = q_param.w * q_param.w;
			float s = __fdividef(2.0f, aa + bb + cc + dd);

			float bs = q_param.y * s;  float cs = q_param.z * s;  float ds = q_param.w * s;
			float ab = q_param.x * bs; float ac = q_param.x * cs; float ad = q_param.x * ds;
			bb = bb * s;			   float bc = q_param.y * cs; float bd = q_param.y * ds;
			cc = cc * s;			   float cd = q_param.z * ds;       dd = dd * s;

			float Q11 = 1.0f - cc - dd;
			float Q12 = bc + ad;
			float Q13 = bd - ac;

			float Q21 = bc - ad;
			float Q22 = 1.0f - bb - dd;
			float Q23 = cd + ab;

			float Q31 = bd + ac;
			float Q32 = cd - ab;
			float Q33 = 1.0f - bb - cc;

			// *************************************************************************************

			float kappa_raw = __ldg(kappa_ptr + ind_clamped);
			float kappa = 1.0f + __logf(1.0f + __expf(kappa_raw));

			float kappa_inv;
			asm volatile ("rcp.approx.ftz.f32 %0, %1;" : "=f"(kappa_inv) : "f"(kappa));

			// *************************************************************************************

			float3 P_hit = make_float3(
				__fmaf_rn(v.x, t_hit, O.x),
				__fmaf_rn(v.y, t_hit, O.y),
				__fmaf_rn(v.z, t_hit, O.z)
			);

			float3 P_hit_prim = make_float3(
				P_hit.x - m_param.x,
				P_hit.y - m_param.y,
				P_hit.z - m_param.z
			);

			float2 uv = make_float2(
				s_param.x * __fmaf_rn(Q11, P_hit_prim.x, __fmaf_rn(Q12, P_hit_prim.y, Q13 * P_hit_prim.z)),
				s_param.y * __fmaf_rn(Q21, P_hit_prim.x, __fmaf_rn(Q22, P_hit_prim.y, Q23 * P_hit_prim.z))
			);

			// *************************************************************************************

			float N_dot_v_inv;
			asm volatile ("rcp.approx.ftz.f32 %0, %1;" : "=f"(N_dot_v_inv) : "f"(__fmaf_rn(Q31, v.x, __fmaf_rn(Q32, v.y, Q33 * v.z))));
			float3 dt_dm = make_float3(
				Q31 * N_dot_v_inv,
				Q32 * N_dot_v_inv,
				Q33 * N_dot_v_inv
			);

			float U_dot_v = __fmaf_rn(Q11, v.x, __fmaf_rn(Q12, v.y, Q13 * v.z));
			float V_dot_v = __fmaf_rn(Q21, v.x, __fmaf_rn(Q22, v.y, Q23 * v.z));

			float3 du_dm = make_float3(
				__fmaf_rn(U_dot_v, dt_dm.x, -Q11) * s_param.x,
				__fmaf_rn(U_dot_v, dt_dm.y, -Q12) * s_param.x,
				__fmaf_rn(U_dot_v, dt_dm.z, -Q13) * s_param.x
			);

			float3 dv_dm = make_float3(
				__fmaf_rn(V_dot_v, dt_dm.x, -Q21) * s_param.y,
				__fmaf_rn(V_dot_v, dt_dm.y, -Q22) * s_param.y,
				__fmaf_rn(V_dot_v, dt_dm.z, -Q23) * s_param.y
			);

			// *************************************************************************************

			float4 RGBA_param = __ldg(RGBA_ptr + ind_clamped);
			float opacity;
			asm volatile ("rcp.approx.ftz.f32 %0, %1;" : "=f"(opacity) : "f"(1.0f + __expf(-RGBA_param.w)));

			// *************************************************************************************

			float r_squared = __fmaf_rn(uv.x, uv.x, uv.y * uv.y);

			float r_squared_pow_kappa_minus_one = __powf(r_squared, kappa - 1.0f);
			// !!! !!! !!!
			if (isnan(r_squared_pow_kappa_minus_one))
				r_squared_pow_kappa_minus_one = 0.0f;
			// !!! !!! !!!

			float r_squared_pow_kappa = r_squared * r_squared_pow_kappa_minus_one;

			// *************************************************************************************

			// !!! !!! !!!
			float dL_dA_delta1 = opacity * __expf(-0.5f * kappa_inv * r_squared_pow_kappa);
			float dL_dA_msq = dL_dA_delta1;
			float dL_dA = dL_dA_msq * (1.0f - opacity);
			// !!! !!! !!!

			// *************************************************************************************

			float opacity_total = dL_dA_msq;
			float T_prev = T;

			T = T - (T * opacity_total);

			// *************************************************************************************

			// Depth regularization
			float t_unclamped = __fdividef(reg_a, t_hit) + reg_b;
			float t = __saturatef(t_unclamped);

			// !!! !!! !!!
			opacity = T_prev * opacity_total;
			// !!! !!! !!!

			float o_accum = depth_normal_reg_prefix_sums.y;
			float t_o_accum = depth_normal_reg_prefix_sums.z;

			//float dL_do = 2.0 * ((t * ((2.0 * o_cum) - o_total_chunk.unsqueeze(0))) - (2.0 * t_o_cum) + t_o_total_chunk.unsqueeze(0));
			//float dL_dt = 2.0 * ((2.0 * o * o_cum) - (o * (o_total_chunk.unsqueeze(0) - o)));

			float dL_do = 2.0f * (__fmaf_rn(t, __fmaf_rn(2.0f, o_accum, -depth_reg_accums.y), -2.0f * t_o_accum) + depth_reg_accums.z);

			float dL_dt = 2.0f * opacity * __fmaf_rn(2.0f, o_accum, opacity - depth_reg_accums.y);
			dL_dt *= __fdividef(-reg_a, t_hit * t_hit);
			if (t != t_unclamped) dL_dt = 0.0f;

			// *************************************************************************************

			// Normal regularization
			float normal_sign = (__fmaf_rn(Q31, v.x, __fmaf_rn(Q32, v.y, Q33 * v.z)) > 0.0f) ? -1.0f : 1.0f;
			float N_median_dot_N = normal_sign * __fmaf_rn(N_median.x, Q31, __fmaf_rn(N_median.y, Q32, N_median.z * Q33));
			float dLn_do = normal_target_valid ? 1.0f - N_median_dot_N : 0.0f;
			float tmp_reg_normal = normal_target_valid ? -opacity * normal_sign * s : 0.0f;

			// *************************************************************************************

			// Depth and normal regularization
			depth_normal_reg_prefix_sums = make_float4(
				__fmaf_rn(opacity, dL_do, depth_normal_reg_prefix_sums.x), // !!! !!! !!! o_dL_do_prefix_sum (inclusive) !!! !!! !!!
				o_accum + opacity,
				__fmaf_rn(t, opacity, t_o_accum),
				__fmaf_rn(opacity, dLn_do, depth_normal_reg_prefix_sums.w) // !!! !!! !!! o_dLn_do_prefix_sum (inclusive) !!! !!! !!!
			);

			// *************************************************************************************

			// Depth regularization
			/*dL_dalpha = torch.where(
				(T_total >= T_min)[:,:,0],
				-((dL_do_total_chunk.unsqueeze(0) - o_dL_do_cum) / (1.0 - opacity[:,:,0])), # !!! !!! !!!
				zeros.reshape(16, rgb.shape[1], -1)[:,:,0]
			);
			dL_dalpha = dL_dalpha + (dL_do * T_total[:,:,0]);
			dL_dalpha = dL_dalpha * (123.456 / (100.0 * number_of_rays_initial));

			dL_dt = dL_dt * (123.456 / (100.0 * number_of_rays_initial));*/

			// lambda = lambda * (123.456 / (100.0 * number_of_rays_initial))
			float dL_dalpha = (T >= T_threshold) ?
				__fdividef(depth_normal_reg_prefix_sums.x - depth_reg_accums.x, 1.0f - opacity_total) :
				0.0f;
			dL_dalpha = lambda_depth * __fmaf_rn(dL_do, T_prev, dL_dalpha);

			dL_dt *= lambda_depth;

			// *************************************************************************************

			// Normal regularization
			// Keep normal consistency orientation-only. Letting this term update
			// opacity/footprint admits the degenerate solution of hiding splats.
			float dLn_dalpha = 0.0f;

			// *************************************************************************************

			bool active = (tid < number_of_rays) && (ind != -1) && (T_prev >= T_threshold);

			// *************************************************************************************

			// dL/dRGB, dL/dA
			if (T < T_threshold) {
				// !!! !!! !!!
				dL_dA_RGB = make_float3(
					-T_prev * bg_color.x,
					-T_prev * bg_color.y,
					-T_prev * bg_color.z
				);
				// !!! !!! !!!
				opacity_total = 0.0f;
			}
			float dL_dA_partial = __fdividef(
				__fmaf_rn(
					dL_dI.x,
					__fmaf_rn(RGBA_param.x, T_prev, dL_dA_RGB.x),
					__fmaf_rn(
						dL_dI.y,
						__fmaf_rn(RGBA_param.y, T_prev, dL_dA_RGB.y),
						dL_dI.z * __fmaf_rn(RGBA_param.z, T_prev, dL_dA_RGB.z)
					)
				),
				1.0f - opacity_total
			);
			//dL_dA *= dL_dA_partial;
			dL_dA *= (dL_dA_partial + dL_dalpha + dLn_dalpha); // Depth and normal regularization

			opacity_total *= T_prev;

			float3 dL_dRGB = make_float3(
				opacity_total,
				opacity_total,
				opacity_total
			);

			dL_dA_RGB = make_float3(
				__fmaf_rn(dL_dRGB.x, RGBA_param.x, dL_dA_RGB.x),
				__fmaf_rn(dL_dRGB.y, RGBA_param.y, dL_dA_RGB.y),
				__fmaf_rn(dL_dRGB.z, RGBA_param.z, dL_dA_RGB.z)
			);

			// !!! !!! !!!
			dL_dRGB.x *= dL_dI.x;
			dL_dRGB.y *= dL_dI.y;
			dL_dRGB.z *= dL_dI.z;
			// !!! !!! !!!

			if (active) {
				atomicAdd(dL_dRGB_ptr + (ind * 3) + 0, dL_dRGB.x);
				atomicAdd(dL_dRGB_ptr + (ind * 3) + 1, dL_dRGB.y);
				atomicAdd(dL_dRGB_ptr + (ind * 3) + 2, dL_dRGB.z);

				atomicAdd(dL_dA_ptr + ind, dL_dA);
			}

			// *************************************************************************************
			// Paper: differentiate kappa = 1 + softplus(k_raw) and the generalized kernel.

			// dL/d_kappa
			float kappa_sigmoid;
			asm volatile ("rcp.approx.ftz.f32 %0, %1;" : "=f"(kappa_sigmoid) : "f"(1.0f + __expf(-kappa_raw)));

			float log_r_squared = __logf(r_squared);
			//float dL_d_kappa = dL_dA_msq * dL_dA_partial * 0.5f * r_squared_pow_kappa * kappa_sigmoid * kappa_inv * (kappa_inv - log_r_squared);
			float dL_d_kappa = dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha) * 0.5f * r_squared_pow_kappa * kappa_sigmoid * kappa_inv * (kappa_inv - log_r_squared); // Depth and normal regularization
			if (isnan(dL_d_kappa))
				dL_d_kappa = 0.0f;

			if (active)
				atomicAdd(dL_d_kappa_ptr + ind, dL_d_kappa);

			// *************************************************************************************

			// dL/dm
			float3 dL_dm2 = make_float3(
				//dL_dA_msq * dL_dA_partial * r_squared_pow_kappa_minus_one * __fmaf_rn(uv.x, du_dm.x, uv.y * dv_dm.x),
				//dL_dA_msq * dL_dA_partial * r_squared_pow_kappa_minus_one * __fmaf_rn(uv.x, du_dm.y, uv.y * dv_dm.y),
				//dL_dA_msq * dL_dA_partial * r_squared_pow_kappa_minus_one * __fmaf_rn(uv.x, du_dm.z, uv.y * dv_dm.z)
				dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha) * r_squared_pow_kappa_minus_one * __fmaf_rn(uv.x, du_dm.x, uv.y * dv_dm.x), // depth and normal regularization
				dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha) * r_squared_pow_kappa_minus_one * __fmaf_rn(uv.x, du_dm.y, uv.y * dv_dm.y), // depth and normal regularization
				dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha) * r_squared_pow_kappa_minus_one * __fmaf_rn(uv.x, du_dm.z, uv.y * dv_dm.z)  // depth and normal regularization
			);

			// depth regularization
			float3 dL_dm3 = make_float3(
				dt_dm.x * dL_dt,
				dt_dm.y * dL_dt,
				dt_dm.z * dL_dt
			);

			if (active) {
				// atomicAdd(dL_dm_ptr + (ind * 3) + 0, dL_dm1.x - dL_dm2.x);
				// atomicAdd(dL_dm_ptr + (ind * 3) + 1, dL_dm1.y - dL_dm2.y);
				// atomicAdd(dL_dm_ptr + (ind * 3) + 2, dL_dm1.z - dL_dm2.z);

				// depth regularization
				atomicAdd(dL_dm_ptr + (ind * 3) + 0, -dL_dm2.x + dL_dm3.x);
				atomicAdd(dL_dm_ptr + (ind * 3) + 1, -dL_dm2.y + dL_dm3.y);
				atomicAdd(dL_dm_ptr + (ind * 3) + 2, -dL_dm2.z + dL_dm3.z);
			}

			// *************************************************************************************

			// dL/ds
			float2 dL_ds = make_float2(
				//(-uv.x * d_x.x) + (dL_dA_msq * dL_dA_partial * r_squared_pow_kappa_minus_one * uv.x * uv.x),
				//(-uv.y * d_x.y) + (dL_dA_msq * dL_dA_partial * r_squared_pow_kappa_minus_one * uv.y * uv.y)
				(dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha) * r_squared_pow_kappa_minus_one * uv.x * uv.x), // depth and normal regularization
				(dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha) * r_squared_pow_kappa_minus_one * uv.y * uv.y)  // depth and normal regularization
			);

			if (active) {
				atomicAdd(dL_ds_ptr + (ind * 2) + 0, dL_ds.x);
				atomicAdd(dL_ds_ptr + (ind * 2) + 1, dL_ds.y);
			}

			// *************************************************************************************

			// dL/dq

			// !!! !!! !!!
			aa *= s;
			// !!! !!! !!!

			// *************************************************************************************

			float tmp7 = 1.0f - aa;
			float tmp8 = 1.0f - bb;
			float tmp9 = 1.0f - cc;
			float tmp10 = 1.0f - dd;

			cd = -cd;
			ab = -ab;
			float a_inv = q_param.y * cd;
			float b_inv = q_param.x * cd;
			float c_inv = ab * q_param.w;
			float d_inv = ab * q_param.z;

			// *************************************************************************************

			float3 m_minus_O = make_float3(m_param.x - O.x, m_param.y - O.y, m_param.z - O.z);
			float N_dot_m_minus_O = __fmaf_rn(Q31, m_minus_O.x, __fmaf_rn(Q32, m_minus_O.y, Q33 * m_minus_O.z));

			//uv.x *= r_squared_pow_kappa_minus_one * dL_dA_msq * dL_dA_partial;
			//uv.y *= r_squared_pow_kappa_minus_one * dL_dA_msq * dL_dA_partial;
			uv.x *= r_squared_pow_kappa_minus_one * dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha); // depth and normal regularization
			uv.y *= r_squared_pow_kappa_minus_one * dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha); // depth and normal regularization

			float2 d_x = make_float2(
				-uv.x,
				-uv.y
			);

			s_param.x *= s;
			s_param.y *= s;

			float tmp11 = -N_dot_v_inv * N_dot_m_minus_O;

			// *************************************************************************************

			float dLn_dq_partial;

			float3 dU_d_coord, dV_d_coord, dN_d_coord;
			float dt_dq, du_dq, dv_dq;
			float4 dL_dq;

			// dUVN_da
			dU_d_coord = make_float3(
				q_param.x * (tmp7 + tmp8),
				__fmaf_rn(q_param.w, tmp7, d_inv),
				__fmaf_rn(-q_param.z, tmp7, c_inv)
			);
			dV_d_coord = make_float3(
				__fmaf_rn(-q_param.w, tmp7, d_inv),
				q_param.x * (tmp7 + tmp9),
				__fmaf_rn(q_param.y, tmp7, b_inv)
			);
			dN_d_coord = make_float3(
				__fmaf_rn(q_param.z, tmp7, c_inv),
				__fmaf_rn(-q_param.y, tmp7, b_inv),
				q_param.x * (tmp7 + tmp10)
			);

			dt_dq = N_dot_v_inv *__fmaf_rn(
				tmp11,
				__fmaf_rn(dN_d_coord.x, v.x, __fmaf_rn(dN_d_coord.y, v.y, dN_d_coord.z * v.z)),
				__fmaf_rn(dN_d_coord.x, m_minus_O.x, __fmaf_rn(dN_d_coord.y, m_minus_O.y, dN_d_coord.z * m_minus_O.z))
			);

			du_dq =	__fmaf_rn(dt_dq, U_dot_v, __fmaf_rn(dU_d_coord.x, P_hit_prim.x, __fmaf_rn(dU_d_coord.y, P_hit_prim.y, dU_d_coord.z * P_hit_prim.z))) * s_param.x;
			dv_dq = __fmaf_rn(dt_dq, V_dot_v, __fmaf_rn(dV_d_coord.x, P_hit_prim.x, __fmaf_rn(dV_d_coord.y, P_hit_prim.y, dV_d_coord.z * P_hit_prim.z))) * s_param.y;

			// Normal regularization
			//dLn_dqa_partial = -o_cropped * (dN_da * N_median_tmp_cropped).sum(1, keepdim=True) * torch.sgn((N * N_median_tmp_cropped).sum(1, keepdim=True));
			dLn_dq_partial = tmp_reg_normal * __fmaf_rn(dN_d_coord.x, N_median.x, __fmaf_rn(dN_d_coord.y, N_median.y, dN_d_coord.z * N_median.z));

			// dL_dq.x = __fmaf_rn(du_dq, d_x.x, dv_dq * d_x.y);
			dL_dq.x = __fmaf_rn(dLn_dq_partial, lambda_normal, __fmaf_rn(dt_dq, dL_dt, __fmaf_rn(du_dq, d_x.x, dv_dq * d_x.y))); // depth and normal regularization

			// dUVN_db
			dU_d_coord = make_float3(
				q_param.y * (tmp8 + tmp7),
				__fmaf_rn(q_param.z, tmp8, c_inv),
				__fmaf_rn(q_param.w, tmp8, -d_inv)
			);
			dV_d_coord = make_float3(
				__fmaf_rn(q_param.z, tmp8, -c_inv),
				-q_param.y * (tmp8 + tmp10),
				__fmaf_rn(q_param.x, tmp8, a_inv)
			);
			dN_d_coord = make_float3(
				__fmaf_rn(q_param.w, tmp8, d_inv),
				__fmaf_rn(-q_param.x, tmp8, a_inv),
				-q_param.y * (tmp8 + tmp9)
			);

			dt_dq = N_dot_v_inv *__fmaf_rn(
				tmp11,
				__fmaf_rn(dN_d_coord.x, v.x, __fmaf_rn(dN_d_coord.y, v.y, dN_d_coord.z * v.z)),
				__fmaf_rn(dN_d_coord.x, m_minus_O.x, __fmaf_rn(dN_d_coord.y, m_minus_O.y, dN_d_coord.z * m_minus_O.z))
			);

			du_dq =	__fmaf_rn(dt_dq, U_dot_v, __fmaf_rn(dU_d_coord.x, P_hit_prim.x, __fmaf_rn(dU_d_coord.y, P_hit_prim.y, dU_d_coord.z * P_hit_prim.z))) * s_param.x;
			dv_dq = __fmaf_rn(dt_dq, V_dot_v, __fmaf_rn(dV_d_coord.x, P_hit_prim.x, __fmaf_rn(dV_d_coord.y, P_hit_prim.y, dV_d_coord.z * P_hit_prim.z))) * s_param.y;

			// Normal regularization
			//dLn_dqb_partial = -o_cropped * (dN_db * N_median_tmp_cropped).sum(1, keepdim=True) * torch.sgn((N * N_median_tmp_cropped).sum(1, keepdim=True));
			dLn_dq_partial = tmp_reg_normal * __fmaf_rn(dN_d_coord.x, N_median.x, __fmaf_rn(dN_d_coord.y, N_median.y, dN_d_coord.z * N_median.z));

			//dL_dq.y = __fmaf_rn(du_dq, d_x.x, dv_dq * d_x.y);
			dL_dq.y = __fmaf_rn(dLn_dq_partial, lambda_normal, __fmaf_rn(dt_dq, dL_dt, __fmaf_rn(du_dq, d_x.x, dv_dq * d_x.y))); // depth and normal regularization

			// dUVN_dc
			dU_d_coord = make_float3(
				-q_param.z * (tmp9 + tmp10),
				__fmaf_rn(q_param.y, tmp9, b_inv),
				__fmaf_rn(-q_param.x, tmp9, a_inv)
			);
			dV_d_coord = make_float3(
				__fmaf_rn(q_param.y, tmp9, -b_inv),
				q_param.z * (tmp9 + tmp7),
				__fmaf_rn(q_param.w, tmp9, d_inv)
			);
			dN_d_coord = make_float3(
				__fmaf_rn(q_param.x, tmp9, a_inv),
				__fmaf_rn(q_param.w, tmp9, -d_inv),
				-q_param.z * (tmp9 + tmp8)
			);

			dt_dq = N_dot_v_inv *__fmaf_rn(
				tmp11,
				__fmaf_rn(dN_d_coord.x, v.x, __fmaf_rn(dN_d_coord.y, v.y, dN_d_coord.z * v.z)),
				__fmaf_rn(dN_d_coord.x, m_minus_O.x, __fmaf_rn(dN_d_coord.y, m_minus_O.y, dN_d_coord.z * m_minus_O.z))
			);

			du_dq =	__fmaf_rn(dt_dq, U_dot_v, __fmaf_rn(dU_d_coord.x, P_hit_prim.x, __fmaf_rn(dU_d_coord.y, P_hit_prim.y, dU_d_coord.z * P_hit_prim.z))) * s_param.x;
			dv_dq = __fmaf_rn(dt_dq, V_dot_v, __fmaf_rn(dV_d_coord.x, P_hit_prim.x, __fmaf_rn(dV_d_coord.y, P_hit_prim.y, dV_d_coord.z * P_hit_prim.z))) * s_param.y;

			// Normal regularization
			//dLn_dqc_partial = -o_cropped * (dN_dc * N_median_tmp_cropped).sum(1, keepdim=True) * torch.sgn((N * N_median_tmp_cropped).sum(1, keepdim=True));
			dLn_dq_partial = tmp_reg_normal * __fmaf_rn(dN_d_coord.x, N_median.x, __fmaf_rn(dN_d_coord.y, N_median.y, dN_d_coord.z * N_median.z));

			//dL_dq.z = __fmaf_rn(du_dq, d_x.x, dv_dq * d_x.y);
			dL_dq.z = __fmaf_rn(dLn_dq_partial, lambda_normal, __fmaf_rn(dt_dq, dL_dt, __fmaf_rn(du_dq, d_x.x, dv_dq * d_x.y))); // depth and normal regularization

			// dUVN_dd
			dU_d_coord = make_float3(
				-q_param.w * (tmp10 + tmp9),
				__fmaf_rn(q_param.x, tmp10, a_inv),
				__fmaf_rn(q_param.y, tmp10, -b_inv)
			);
			dV_d_coord = make_float3(
				__fmaf_rn(-q_param.x, tmp10, a_inv),
				-q_param.w * (tmp10 + tmp8),
				__fmaf_rn(q_param.z, tmp10, c_inv)
			);
			dN_d_coord = make_float3(
				__fmaf_rn(q_param.y, tmp10, b_inv),
				__fmaf_rn(q_param.z, tmp10, -c_inv),
				q_param.w * (tmp10 + tmp7)
			);

			dt_dq = N_dot_v_inv *__fmaf_rn(
				tmp11,
				__fmaf_rn(dN_d_coord.x, v.x, __fmaf_rn(dN_d_coord.y, v.y, dN_d_coord.z * v.z)),
				__fmaf_rn(dN_d_coord.x, m_minus_O.x, __fmaf_rn(dN_d_coord.y, m_minus_O.y, dN_d_coord.z * m_minus_O.z))
			);

			du_dq =	__fmaf_rn(dt_dq, U_dot_v, __fmaf_rn(dU_d_coord.x, P_hit_prim.x, __fmaf_rn(dU_d_coord.y, P_hit_prim.y, dU_d_coord.z * P_hit_prim.z))) * s_param.x;
			dv_dq = __fmaf_rn(dt_dq, V_dot_v, __fmaf_rn(dV_d_coord.x, P_hit_prim.x, __fmaf_rn(dV_d_coord.y, P_hit_prim.y, dV_d_coord.z * P_hit_prim.z))) * s_param.y;

			// Normal regularization
			//dLn_dqd_partial = -o_cropped * (dN_dd * N_median_tmp_cropped).sum(1, keepdim=True) * torch.sgn((N * N_median_tmp_cropped).sum(1, keepdim=True));
			dLn_dq_partial = tmp_reg_normal * __fmaf_rn(dN_d_coord.x, N_median.x, __fmaf_rn(dN_d_coord.y, N_median.y, dN_d_coord.z * N_median.z));

			//dL_dq.w = __fmaf_rn(du_dq, d_x.x, dv_dq * d_x.y);
			dL_dq.w = __fmaf_rn(dLn_dq_partial, lambda_normal, __fmaf_rn(dt_dq, dL_dt, __fmaf_rn(du_dq, d_x.x, dv_dq * d_x.y))); // depth and normal regularization

			if (active) {
				atomicAdd(dL_dq_ptr + (ind * 4) + 0, dL_dq.x);
				atomicAdd(dL_dq_ptr + (ind * 4) + 1, dL_dq.y);
				atomicAdd(dL_dq_ptr + (ind * 4) + 2, dL_dq.z);
				atomicAdd(dL_dq_ptr + (ind * 4) + 3, dL_dq.w);
			}
		}

		// *****************************************************************************************

		if (tid < number_of_rays) {
			I_ptr[pixel_ind]     = make_float3(-dL_dA_RGB.x, -dL_dA_RGB.y, -dL_dA_RGB.z); // !!! !!! !!!
			t_min_ptr[pixel_ind] = (ind != -1) ? nextafter(t_hit, INFINITY) : INFINITY;   // !!! !!! !!!
			T_ptr[pixel_ind]     = T;                                                     // !!! !!! !!!

			// Depth and normal regularization
			depth_normal_reg_prefix_sums_ptr[pixel_ind] = depth_normal_reg_prefix_sums; // !!! !!! !!!

			// !!! !!! !!!
			is_active_ptr[tid] = ((ind != -1) && (T >= T_threshold));
			// !!! !!! !!!
		}
	}
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer_CUDA::Backward_base(
	float3 *O,
	float3 *v,

	int number_of_rays,

	float3 bg_color,

	float3 *m,
	float2 *s,
	float4 *q,
	float4 *RGBA,
	float *kappa,

	float3 *I,
	float3 *dL_dI,
	float *dL_dRGB,
	float *dL_dA,
	float *dL_d_kappa,
	float *dL_dm,
	float *dL_ds,
	float *dL_dq,

	float T_threshold,

	float4 *depth_reg_accums,
	float4 *depth_normal_reg_prefix_sums,
	float lambda_depth,
	float reg_a,
	float reg_b,

	float2 *depth_and_index,
	float3 *surface_normal,
	float4 *normal_reg_accums,
	float lambda_normal
) {
	cudaError_t error_CUDA;
	CUresult error_CUDA_Driver_API;
	OptixResult error_OptiX;

	// *********************************************************************************************

	error_CUDA_Driver_API = cuMemsetD32Async((CUdeviceptr)t_min, 0, number_of_rays, stream);
	if (error_CUDA_Driver_API != CUDA_SUCCESS) throw 0;

	float value = 1.0f;
	error_CUDA_Driver_API = cuMemsetD32Async((CUdeviceptr)T, (unsigned &)value, number_of_rays, stream);
	if (error_CUDA_Driver_API != CUDA_SUCCESS) throw 0;

	try {
		thrust::sequence(
			thrust::cuda::par.on(stream),
			thrust::device_pointer_cast(ray_indices[0]),
			thrust::device_pointer_cast(ray_indices[0]) + number_of_rays
		);
	} catch (...) {
		throw 0;
	}

	// *********************************************************************************************

	SLaunchParams launchParams;

	launchParams.O = (float3 *)O;
	launchParams.v = (float3 *)v;
	launchParams.t_min = t_min;
	launchParams.AS = IAS;
	launchParams.t = t_hit;
	launchParams.indices = indices_hit;

	// *********************************************************************************************

	// !!! !!! !!!
	int pass_num = 0;
	// !!! !!! !!!

	while (number_of_rays > 0) {
		launchParams.ray_indices = ray_indices[pass_num & 1]; // !!! !!! !!!

		error_CUDA = cudaMemcpyAsync(launchParamsBuffer, &launchParams, sizeof(SLaunchParams) * 1, cudaMemcpyHostToDevice, stream);
		if (error_CUDA != cudaSuccess) throw 0;

		int counter_host = 0;
		error_CUDA = cudaMemcpyToSymbolAsync(counter, &counter_host, sizeof(int) * 1, 0, cudaMemcpyHostToDevice, stream);
		if (error_CUDA != cudaSuccess) throw 0;

		// *****************************************************************************************

		error_OptiX = optixLaunch(
			pipeline,
			stream,
			(CUdeviceptr)launchParamsBuffer,
			sizeof(SLaunchParams) * 1,
			sbt,
			number_of_rays,
			1,
			1
		);
		if (error_OptiX != OPTIX_SUCCESS) throw 0;

		// *****************************************************************************************

		Backward_base_CUDA_kernel<<<min((number_of_rays + (threads_per_block - 1)) / threads_per_block, SM_count), threads_per_block, 0, stream>>>(
			O, v,
			ray_indices[pass_num & 1], // !!! !!! !!!
			t_hit, indices_hit, t_min, T, is_active,

			number_of_rays,

			bg_color,

			m, s, q, RGBA, kappa,

			I, dL_dI, dL_dRGB, dL_dA, dL_d_kappa,
			dL_dm, dL_ds, dL_dq,

			T_threshold,

			depth_reg_accums,
			depth_normal_reg_prefix_sums,
			lambda_depth,
			reg_a,
			reg_b,

			depth_and_index,
			surface_normal,
			normal_reg_accums,
			lambda_normal
		);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		// *****************************************************************************************

		try {
			thrust::exclusive_scan(
				thrust::cuda::par.on(stream),
				thrust::device_pointer_cast(is_active),
				thrust::device_pointer_cast(is_active) + number_of_rays,
				thrust::device_pointer_cast(indices_hit) // !!! !!! !!!
			);

			thrust::scatter_if(
				thrust::cuda::par.on(stream),
				thrust::device_pointer_cast(ray_indices[pass_num & 1]),
				thrust::device_pointer_cast(ray_indices[pass_num & 1]) + number_of_rays,
				thrust::device_pointer_cast(indices_hit),
				thrust::device_pointer_cast(is_active),
				thrust::device_pointer_cast(ray_indices[(pass_num + 1) & 1])
			);

			// !!! !!! !!!
			number_of_rays = thrust::reduce(
				thrust::cuda::par.on(stream),
				thrust::device_pointer_cast(is_active),
				thrust::device_pointer_cast(is_active) + number_of_rays
			);
			// !!! !!! !!!
		} catch (...) {
			throw 0;
		}

		// *****************************************************************************************

		++pass_num;
	}
}
