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

const int threads_per_block = 512;

// *** *** *** *** ***

static __global__ void __launch_bounds__(threads_per_block, 1) Forward_training_base_CUDA_kernel(
	float3 *O_ptr,
	float3 *v_ptr,
	int *ray_indices,
	float *t_hit_ptr,
	int *indices_hit_ptr,
	float *t_min_ptr,
	float *T_ptr,
	int *is_active_ptr,
	float3 *bitmap_ptr,

	int number_of_rays,

	float3 bg_color,

	float3 *m_ptr,
	float2 *s_ptr,
	float4 *q_ptr,
	float4 *RGBA_ptr,
	float *kappa_ptr,

	float T_threshold,

	float4 *depth_reg_accums_ptr,
	float reg_a,
	float reg_b,

	float2 *depth_and_index_ptr,
	float3 *surface_normal_ptr,
	float4 *normal_reg_accums_ptr
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
		float3 RGB_aggregated = bitmap_ptr[pixel_ind]; // !!! !!! !!!

		// *****************************************************************************************

		// Depth regularization
		float4 depth_reg_accums = depth_reg_accums_ptr[pixel_ind]; // !!! !!! !!!

		// *****************************************************************************************

		// Normal regularization
		float4 normal_reg_accums = normal_reg_accums_ptr[pixel_ind];
		float3 N_median = surface_normal_ptr[pixel_ind];
		bool normal_target_valid = __fmaf_rn(N_median.x, N_median.x,
			__fmaf_rn(N_median.y, N_median.y, N_median.z * N_median.z)) > 0.0f;

		// *****************************************************************************************

		int ind; // !!! !!! !!!
		float t_hit; // !!! !!! !!!

		// *****************************************************************************************

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

			float Q11 = s_param.x * (1.0f - cc - dd);
			float Q12 = s_param.x * (bc + ad);
			float Q13 = s_param.x * (bd - ac);

			float Q21 = s_param.y * (bc - ad);
			float Q22 = s_param.y * (1.0f - bb - dd);
			float Q23 = s_param.y * (cd + ab);

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
				__fmaf_rn(Q11, P_hit_prim.x, __fmaf_rn(Q12, P_hit_prim.y, Q13 * P_hit_prim.z)),
				__fmaf_rn(Q21, P_hit_prim.x, __fmaf_rn(Q22, P_hit_prim.y, Q23 * P_hit_prim.z))
			);

			// *************************************************************************************

			float4 RGBA_param = __ldg(RGBA_ptr + ind_clamped);
			float opacity1;
			asm volatile ("rcp.approx.ftz.f32 %0, %1;" : "=f"(opacity1) : "f"(1.0f + __expf(-RGBA_param.w)));

			float r_squared = __fmaf_rn(uv.x, uv.x, uv.y * uv.y);

			// to maintain consistency of T computation between Forward and Backward method
			// !!! !!! !!!
			float r_squared_pow_kappa_minus_one = __powf(r_squared, kappa - 1.0f);
			// !!! !!! !!!
			if (isnan(r_squared_pow_kappa_minus_one))
				r_squared_pow_kappa_minus_one = 0.0f;
			// !!! !!! !!!

			float r_squared_pow_kappa = r_squared * r_squared_pow_kappa_minus_one;
			// !!! !!! !!!

			float opacity = T * (opacity1 * __expf(-0.5f * kappa_inv * r_squared_pow_kappa)); // !!! !!! !!!

			if ((ind != -1) && (T >= T_threshold)) {
				// !!! !!! !!!
				RGB_aggregated.x = __fmaf_rn(RGBA_param.x, opacity, RGB_aggregated.x);
				RGB_aggregated.y = __fmaf_rn(RGBA_param.y, opacity, RGB_aggregated.y);
				RGB_aggregated.z = __fmaf_rn(RGBA_param.z, opacity, RGB_aggregated.z);
				// !!! !!! !!!

				T = T - opacity;

				// Depth regularization
				float t = __saturatef(__fdividef(reg_a, t_hit) + reg_b);

				float o_accum = depth_reg_accums.y;
				float t_o_accum = depth_reg_accums.z;
				float L_d_accum = opacity * __fmaf_rn(t, o_accum, -t_o_accum);

				depth_reg_accums = make_float4(
					__fmaf_rn(4.0f, L_d_accum, depth_reg_accums.x),
					o_accum + opacity,
					__fmaf_rn(t, opacity, t_o_accum),
					__fmaf_rn(t_hit, opacity, depth_reg_accums.w)
				);

				// Normal regularization
				float normal_sign = (__fmaf_rn(Q31, v.x, __fmaf_rn(Q32, v.y, Q33 * v.z)) > 0.0f) ? -1.0f : 1.0f;
				float N_median_dot_N = normal_sign * __fmaf_rn(N_median.x, Q31, __fmaf_rn(N_median.y, Q32, N_median.z * Q33));
				float L_n_accum = normal_target_valid ? opacity * (1.0f - N_median_dot_N) : 0.0f;

				normal_reg_accums = make_float4(
					normal_reg_accums.x + L_n_accum,
					normal_reg_accums.y,
					normal_reg_accums.z,
					normal_reg_accums.w
				);
			}
		}

		// *****************************************************************************************

		if (tid < number_of_rays) {
			bool is_active = ((ind != -1) && (T >= T_threshold));

			// background color
			if (!is_active) {
				RGB_aggregated.x = __fmaf_rn(bg_color.x, T, RGB_aggregated.x);
				RGB_aggregated.y = __fmaf_rn(bg_color.y, T, RGB_aggregated.y);
				RGB_aggregated.z = __fmaf_rn(bg_color.z, T, RGB_aggregated.z);
			}

			bitmap_ptr[pixel_ind] = make_float3(
				RGB_aggregated.x,
				RGB_aggregated.y,
				RGB_aggregated.z
			);
			t_min_ptr[pixel_ind] = (ind != -1) ? nextafter(t_hit, INFINITY) : INFINITY; // !!! !!! !!!
			T_ptr[pixel_ind] = T; // !!! !!! !!!

			// Depth regularization
			depth_reg_accums_ptr[pixel_ind] = depth_reg_accums; // !!! !!! !!!

			// Normal regularization
			normal_reg_accums_ptr[pixel_ind] = normal_reg_accums; // !!! !!! !!!

			// !!! !!! !!!
			is_active_ptr[tid] = is_active;
			// !!! !!! !!!
		}

		// *****************************************************************************************
	}
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer_CUDA::Forward_training_base(
	float3 *O,
	float3 *v,
	float3 *bitmap,

	int number_of_rays,

	float3 bg_color,

	float3 *m,
	float2 *s,
	float4 *q,
	float4 *RGBA,
	float *kappa,

	float T_threshold,

	float4 *depth_reg_accums,
	float reg_a,
	float reg_b,

	float2 *depth_and_index,
	float3 *surface_normal,
	float4 *normal_reg_accums
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

		Forward_training_base_CUDA_kernel<<<min((number_of_rays + (threads_per_block - 1)) / threads_per_block, SM_count), threads_per_block, 0, stream>>>(
			O, v,
			ray_indices[pass_num & 1], // !!! !!! !!!
			t_hit, indices_hit, t_min, T, is_active, bitmap,

			number_of_rays,

			bg_color,

			m, s, q, RGBA, kappa,

			T_threshold,

			depth_reg_accums,
			reg_a,
			reg_b,

			depth_and_index,
			surface_normal,
			normal_reg_accums
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
