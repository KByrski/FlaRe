#define _USE_MATH_DEFINES

#include "CPyOptiXFLARERenderer_CUDA.h"
#include "Header.cuh"
#include "Utils.cuh"

#include <cuda_pipeline_primitives.h>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// *** *** *** *** ***

// !!! !!! !!!
__device__ int counter;
// !!! !!! !!!

// *** *** *** *** ***

const int threads_per_block = 512;

// *** *** *** *** ***

constexpr int L_P_hit_object = 4;
constexpr int T_P_hit_object = 16384;
constexpr int F_P_hit_object = 2;
constexpr int N_min_P_hit_object = 16;
constexpr int N_max_P_hit_object = 64;

constexpr int number_of_frequencies_v_world = 4;

constexpr int conditioning_variable_size = 96;

// *** *** *** *** ***

const unsigned prime2 = 2654435761U;

// *** *** *** *** ***

template <
	int L_start, int L
> static __device__ __forceinline__ void HashEncoding2D_CUDA_kernel(
	float2 x_frag,

	int lane_id,
	int res_int,
	int mask,
	float *features,

	half2 &b0_b1
) {
	// HASH ENCODING
	/*float2 x_times_res = make_float2(
		x_frag.x * res_int,
		x_frag.y * res_int
	);

	int2 x_min = make_int2(__float2int_ru(x_times_res.x), __float2int_ru(x_times_res.y)); // !!! !!! !!!
	int2 x_max = make_int2(x_min.x - 1, x_min.y - 1); // !!! !!! !!!
	float2 x_coords = make_float2(x_min.x - x_times_res.x, x_min.y - x_times_res.y); // !!! !!! !!!

	// MapPositive
	x_min = make_int2(
		((x_min.x << 1)) ^ (x_min.x >> 31),
		((x_min.y << 1)) ^ (x_min.y >> 31)
	);
	x_max = make_int2(
		((x_max.x << 1)) ^ (x_max.x >> 31),
		((x_max.y << 1)) ^ (x_max.y >> 31)
	);

	// *********************************************************************************************

	unsigned x_min_times_prime = x_min.y * prime2;
	unsigned x_max_times_prime = x_max.y * prime2;

	// *********************************************************************************************

	int ind00;
	asm("lop3.b32 %0, %1, %2, %3, 40;"
		: "=r"(ind00)
		:  "r"(x_min.x), "r"(x_min_times_prime), "r"(mask)
	);

	int ind10;
	asm("lop3.b32 %0, %1, %2, %3, 40;"
		: "=r"(ind10)
		:  "r"(x_max.x), "r"(x_min_times_prime), "r"(mask)
	);

	int ind01;
	asm("lop3.b32 %0, %1, %2, %3, 40;"
		: "=r"(ind01)
		:  "r"(x_min.x), "r"(x_max_times_prime), "r"(mask)
	);

	int ind11;
	asm("lop3.b32 %0, %1, %2, %3, 40;"
		: "=r"(ind11)
		:  "r"(x_max.x), "r"(x_max_times_prime), "r"(mask)
	);

	// *********************************************************************************************

	int L_num = L_start + (lane_id & 3);

	float2 f00 = ((float2 *)features)[(ind00 * L) + L_num];
	float2 f10 = ((float2 *)features)[(ind10 * L) + L_num];
	float2 f01 = ((float2 *)features)[(ind01 * L) + L_num];
	float2 f11 = ((float2 *)features)[(ind11 * L) + L_num];*/



	// LUT ENCODING

	// Computing offsets using the exclusive prefix sum of the resolutions
	int tmp;
	int layer_size = res_int * res_int;
	int offset = layer_size; // !!! !!! !!!

	tmp = __shfl_up_sync(-1, offset, 1, 4);
	offset += (((lane_id & 3) >= 1) ? tmp : 0);
	tmp = __shfl_up_sync(-1, offset, 2, 4);
	offset += (((lane_id & 3) >= 2) ? tmp : 0);

	offset -= layer_size; // !!! !!! !!!

	// *********************************************************************************************

	x_frag = make_float2(
		__saturatef((x_frag.x + 1.0f) * 0.5f),
		__saturatef((x_frag.y + 1.0f) * 0.5f)
	);

	float scale = res_int - 1.0f;
	float2 x_times_res = make_float2(
		x_frag.x * scale,
		x_frag.y * scale
	);

	int max_ind = res_int - 2;
	int2 x_min = make_int2(
		min(__float2int_rd(x_times_res.x), max_ind),
		min(__float2int_rd(x_times_res.y), max_ind)
	);
	int2 x_max = make_int2(x_min.x + 1, x_min.y + 1);
	float2 x_coords = make_float2(x_times_res.x - x_min.x, x_times_res.y - x_min.y);

	// *********************************************************************************************

	int ind00 = offset + ((x_min.y * res_int) + x_min.x);
	int ind10 = offset + ((x_min.y * res_int) + x_max.x);
	int ind01 = offset + ((x_max.y * res_int) + x_min.x);
	int ind11 = offset + ((x_max.y * res_int) + x_max.x);

	// *********************************************************************************************

	float2 f00 = ((float2 *)features)[ind00];
	float2 f10 = ((float2 *)features)[ind10];
	float2 f01 = ((float2 *)features)[ind01];
	float2 f11 = ((float2 *)features)[ind11];

	// *********************************************************************************************

	float u, v;

	// *********************************************************************************************

	u = x_coords.x;
	v = 1.0f - u;

	float2 f0 = make_float2(
		__fmaf_rn(f00.x, v, f10.x * u),
		__fmaf_rn(f00.y, v, f10.y * u)
	);
	float2 f1 = make_float2(
		__fmaf_rn(f01.x, v, f11.x * u),
		__fmaf_rn(f01.y, v, f11.y * u)
	);

	// *********************************************************************************************

	u = x_coords.y;
	v = 1.0f - u;

	float2 f = make_float2(
		__fmaf_rn(f0.x, v, f1.x * u),
		__fmaf_rn(f0.y, v, f1.y * u)
	);

	// *********************************************************************************************

	b0_b1 = __float22half2_rn(f);
}

// *** *** *** *** ***

static __device__ __forceinline__ void PositionalEncoding3D(
	float x_frag,

	int warp_quarter_id, int lane_id,

	half2 &b0_b1
) {
	float2 sine_cosine;

	__sincosf(((float)M_PI) * (1 << (lane_id & 3)) * x_frag, &sine_cosine.x, &sine_cosine.y);
	b0_b1 = __float22half2_rn(sine_cosine);
}

// *** *** *** *** ***

static __device__ __forceinline__ void GenerateInputTensor(
	float2 P_hit_object,
	float3 v_world,

	half2 *conditioning_variable_ptr,
	int Gauss_ind,

	float *features_P_hit_object_ptr,

	int k,
	half2 (&B)[16]
) {
	// !!! !!! !!!
	int lane_id = threadIdx.x & 31;
	// !!! !!! !!!

	float b;
	float res_float;
	int mask;
	int res_int;

	int srcLane;

	// *********************************************************************************************
	// * HASH ENCODING: P_hit_object                                                               *
	// *********************************************************************************************

	b = 1.5874010519681994747517056392723;

	//res_float = -N_min_P_hit_object; // !!! !!! !!! HASH ENCODING !!! !!! !!!
	res_float = N_min_P_hit_object; // LUT ENCODING
	if ((lane_id & 1) == 1) res_float *= b;
	b *= b;
	if ((lane_id & 2) == 2) res_float *= b;
	// b *= b;

	mask = T_P_hit_object - 1;

	// *********************************************************************************************

	srcLane = (k << 3) + (lane_id >> 2);
	float2 uv = make_float2(
		__shfl_sync(-1, P_hit_object.x, srcLane),
		__shfl_sync(-1, P_hit_object.y, srcLane)
	);

	// *********************************************************************************************

	res_int = __float2int_rn(res_float);

	HashEncoding2D_CUDA_kernel<0 * 4, 4>(uv, lane_id, res_int, mask, features_P_hit_object_ptr, B[0]);

	// *********************************************************************************************
	// * POSITIONAL ENCODING: v_world                                                              *
	// *********************************************************************************************

	float v_world_coord;

	srcLane = (k << 3) + (lane_id >> 2);

	v_world_coord = __shfl_sync(-1, v_world.x, srcLane);
	PositionalEncoding3D(v_world_coord, 0, lane_id, B[1]);

	v_world_coord = __shfl_sync(-1, v_world.y, srcLane);
	PositionalEncoding3D(v_world_coord, 0, lane_id, B[2]);

	v_world_coord = __shfl_sync(-1, v_world.z, srcLane);
	PositionalEncoding3D(v_world_coord, 0, lane_id, B[3]);

	// *********************************************************************************************
	// * CONDITIONING VARIABLE                                                                     *
	// *********************************************************************************************

	srcLane = (k << 3) + (lane_id >> 2);
	int Gauss_ind_shuffle = __shfl_sync(-1, Gauss_ind, srcLane);

	#pragma unroll
	for (int i = 0; i < 12; ++i)
		B[4 + i] = conditioning_variable_ptr[(Gauss_ind_shuffle * 48) + (i << 2) + (lane_id & 3)];
}

// *** *** *** *** ***

static __device__ __forceinline__ void Forward(
	int A1_shared,
	float *b1_shared,
	int A2_shared,
	float *b2_shared,
	int A3_shared,
	float *b3_shared,

	int k,
	half2 (&B)[16],
	float2 (&C)[8]
) {
	// !!! !!! !!!
	int lane_id = threadIdx.x & 31;
	// !!! !!! !!!

	int lane_mask_lo = lane_id & 15;
	int lane_mask_hi = lane_id & 16;
	int mask_swizzle = (lane_mask_lo & 7) << 4;
	int base;

	// *********************************************************************************************

	uint32_t A11, A21, A12, A22;

	// *********************************************************************************************
	// * LAYER 1                                                                                   *
	// *********************************************************************************************

	#pragma unroll
	for (int i = 0; i < 8; ++i) {
		C[i].x = b1_shared[i << 3];
		C[i].y = b1_shared[i << 3];
	}

	__syncwarp();

	// *********************************************************************************************

	#pragma unroll
	for (int j = 0; j < 16; j += 2) {
		int col = lane_mask_hi + (j << 4);
		int j_swizzled = (col & -128) | ((col ^ mask_swizzle) & 112);
		int addr = A1_shared + j_swizzled;

		#pragma unroll
		for (int i = 0; i < 8; i += 2) {
			asm volatile(
				"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
				: "=r"(A11),
				"=r"(A21),
				"=r"(A12),
				"=r"(A22)
				: "r"(addr + (i * 8 * (128 * 2)))
			);

			asm volatile(
				"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
				"{ %0, %1, %2, %3 }, "
				"{ %4, %5, %6, %7 }, "
				"{ %8, %9 }, "
				"{ %10, %11, %12, %13 };"
				: "=f"(C[i].x), "=f"(C[i].y), "=f"(C[i + 1].x), "=f"(C[i + 1].y)
				: "r"(A11), "r"(A21), "r"(A12), "r"(A22),
				"r"((uint32_t &)B[j]), "r"((uint32_t &)B[j + 1]),
				"f"(C[i].x), "f"(C[i].y), "f"(C[i + 1].x), "f"(C[i + 1].y)
			);
		}
	}

	// Hardswish
	#pragma unroll
	for (int i = 0; i < 8; i += 2) {
		C[i].x = (C[i].x * fmaxf(fminf(C[i].x + 3.0f, 6.0f), 0.0f)) * (1.0f / 6.0f);
		C[i].y = (C[i].y * fmaxf(fminf(C[i].y + 3.0f, 6.0f), 0.0f)) * (1.0f / 6.0f);
		C[i + 1].x = (C[i + 1].x * fmaxf(fminf(C[i + 1].x + 3.0f, 6.0f), 0.0f)) * (1.0f / 6.0f);
		C[i + 1].y = (C[i + 1].y * fmaxf(fminf(C[i + 1].y + 3.0f, 6.0f), 0.0f)) * (1.0f / 6.0f);
	}

	// *********************************************************************************************
	// * LAYER 2                                                                                   *
	// *********************************************************************************************

	#pragma unroll
	for (int j = 0; j < 8; ++j)
		asm volatile (
			"movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
			: "=r"((uint32_t &)B[j])
			: "r" (__half22uint32_t(__float22half2_rn(C[j])))
		);

	// *********************************************************************************************

	#pragma unroll
	for (int i = 0; i < 8; ++i) {
		C[i].x = b2_shared[i << 3];
		C[i].y = b2_shared[i << 3];
	}

	__syncwarp();

	// *********************************************************************************************

	#pragma unroll
	for (int j = 0; j < 8; j += 2) {
		int col = lane_mask_hi + (j << 4);
		int j_swizzled = (col & -128) | ((col ^ mask_swizzle) & 112);
		int addr = A2_shared + j_swizzled;

		#pragma unroll
		for (int i = 0; i < 8; i += 2) {
			asm volatile(
				"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
				: "=r"(A11),
				"=r"(A21),
				"=r"(A12),
				"=r"(A22)
				: "r"(addr + (i * 8 * (64 * 2)))
			);

			asm volatile(
				"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
				"{ %0, %1, %2, %3 }, "
				"{ %4, %5, %6, %7 }, "
				"{ %8, %9 }, "
				"{ %10, %11, %12, %13 };"
				: "=f"(C[i].x), "=f"(C[i].y), "=f"(C[i + 1].x), "=f"(C[i + 1].y)
				: "r"(A11), "r"(A21), "r"(A12), "r"(A22),
				"r"((uint32_t &)B[j]), "r"((uint32_t &)B[j + 1]),
				"f"(C[i].x), "f"(C[i].y), "f"(C[i + 1].x), "f"(C[i + 1].y)
			);
		}
	}

	// Hardswish
	#pragma unroll
	for (int i = 0; i < 8; i += 2) {
		C[i].x = (C[i].x * fmaxf(fminf(C[i].x + 3.0f, 6.0f), 0.0f)) * (1.0f / 6.0f);
		C[i].y = (C[i].y * fmaxf(fminf(C[i].y + 3.0f, 6.0f), 0.0f)) * (1.0f / 6.0f);
		C[i + 1].x = (C[i + 1].x * fmaxf(fminf(C[i + 1].x + 3.0f, 6.0f), 0.0f)) * (1.0f / 6.0f);
		C[i + 1].y = (C[i + 1].y * fmaxf(fminf(C[i + 1].y + 3.0f, 6.0f), 0.0f)) * (1.0f / 6.0f);
	}

	// *********************************************************************************************
	// * LAYER 3                                                                                   *
	// *********************************************************************************************

	#pragma unroll
	for (int j = 0; j < 8; ++j)
		asm volatile (
			"movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
			: "=r"((uint32_t &)B[j])
			: "r" (__half22uint32_t(__float22half2_rn(C[j])))
		);

	// *********************************************************************************************

	#pragma unroll
	for (int i = 0; i < 2; ++i) {
		C[i].x = b3_shared[i << 3];
		C[i].y = b3_shared[i << 3];
	}

	__syncwarp();

	// *********************************************************************************************

	#pragma unroll
	for (int j = 0; j < 8; j += 2) {
		int col = lane_mask_hi + (j << 4);
		int j_swizzled = (col & -128) | ((col ^ mask_swizzle) & 112);
		int addr = A3_shared + j_swizzled;

		#pragma unroll
		for (int i = 0; i < 2; i += 2) {
			asm volatile(
				"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
				: "=r"(A11),
				"=r"(A21),
				"=r"(A12),
				"=r"(A22)
				: "r"(addr + (i * 8 * (64 * 2)))
			);

			asm volatile(
				"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
				"{ %0, %1, %2, %3 }, "
				"{ %4, %5, %6, %7 }, "
				"{ %8, %9 }, "
				"{ %10, %11, %12, %13 };"
				: "=f"(C[i].x), "=f"(C[i].y), "=f"(C[i + 1].x), "=f"(C[i + 1].y)
				: "r"(A11), "r"(A21), "r"(A12), "r"(A22),
				"r"((uint32_t &)B[j]), "r"((uint32_t &)B[j + 1]),
				"f"(C[i].x), "f"(C[i].y), "f"(C[i + 1].x), "f"(C[i + 1].y)
			);
		}
	}
}

// *** *** *** *** ***

static __global__ void __launch_bounds__(threads_per_block, 1) Forward_training_CUDA_kernel(
	half2 *conditioning_variable_ptr,
	float *features_P_hit_object_ptr,

	half *A1_ptr,
	float *b1_ptr,
	half *A2_ptr,
	float *b2_ptr,
	half *A3_ptr,
	float *b3_ptr,

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
	extern __shared__ char scratchpad[];

	half *A1_shared = (half *)scratchpad;
	float *b1_shared = (float *)(A1_shared + (64 * 128));
	half *A2_shared = (half *)(b1_shared + 64);
	float *b2_shared = (float *)(A2_shared + (64 * 64));
	half *A3_shared = (half *)(b2_shared + 64);
	float *b3_shared = (float *)(A3_shared + (16 * 64));
	float *features_shared = (float *)(b3_shared + 16);

	// *********************************************************************************************

	// !!! !!! !!!
	int lane_id = threadIdx.x & 31;
	// !!! !!! !!!

	int lane_mask_lo = lane_id & 15;
	int lane_mask_hi = lane_id & 16;
	int mask_swizzle = (lane_mask_lo & 7) << 4;

	// *********************************************************************************************

	/*__shared__ half A1_shared[64 * 128];
	__shared__ float b1_shared[64];
	__shared__ half A2_shared[64 * 64];
	__shared__ float b2_shared[64];
	__shared__ half A3_shared[16 * 64];
	__shared__ float b3_shared[16];*/

	// *********************************************************************************************

	int size = 64 * 128;

	#pragma unroll
	for (int i = threadIdx.x << 3; i < size; i += (threads_per_block * 8)) {
		int row = i / 128;
		int col = i - (row * 128);
		int col_swizzled = (col & -64) | ((((col & 63) >> 3) ^ (row & 7)) << 3);

		__pipeline_memcpy_async(&A1_shared[(row * 128) + col_swizzled], &A1_ptr[i], 16);
	}

	__pipeline_commit();

	// *********************************************************************************************

	size = 64;

	#pragma unroll
	for (int i = threadIdx.x << 2; i < size; i += (threads_per_block * 4))
		__pipeline_memcpy_async(&b1_shared[i], &b1_ptr[i], 16);

	__pipeline_commit();

	// *********************************************************************************************

	size = 64 * 64;

	#pragma unroll
	for (int i = threadIdx.x << 3; i < size; i += (threads_per_block * 8)) {
		int row = i / 64;
		int col = i - (row * 64);
		int col_swizzled = ((col >> 3) ^ (row & 7)) << 3;

		__pipeline_memcpy_async(&A2_shared[(row * 64) + col_swizzled], &A2_ptr[i], 16);
	}

	__pipeline_commit();

	// *********************************************************************************************

	size = 64;

	#pragma unroll
	for (int i = threadIdx.x << 2; i < size; i += (threads_per_block * 4))
		__pipeline_memcpy_async(&b2_shared[i], &b2_ptr[i], 16);

	__pipeline_commit();

	// *********************************************************************************************

	size = 16 * 64;

	#pragma unroll
	for (int i = threadIdx.x << 3; i < size; i += (threads_per_block * 8)) {
		int row = i / 64;
		int col = i - (row * 64);
		int col_swizzled = ((col >> 3) ^ (row & 7)) << 3;

		__pipeline_memcpy_async(&A3_shared[(row * 64) + col_swizzled], &A3_ptr[i], 16);
	}

	__pipeline_commit();

	// *********************************************************************************************

	size = 16;

	#pragma unroll
	for (int i = threadIdx.x << 2; i < size; i += (threads_per_block * 4))
		__pipeline_memcpy_async(&b3_shared[i], &b3_ptr[i], 16);

	__pipeline_commit();

	// *********************************************************************************************

	size = ((16 * 16) + (25 * 25) + (40 * 40) + (64 * 64)) * 2;

	#pragma unroll
	for (int i = threadIdx.x << 2; i < size; i += (threads_per_block * 4))
		__pipeline_memcpy_async(&features_shared[i], &features_P_hit_object_ptr[i], 16);

	__pipeline_commit();

	// *********************************************************************************************

	int A1_shared_int = (int)__cvta_generic_to_shared(A1_shared) + (lane_mask_lo * (128 * 2));
	b1_shared += (lane_id >> 2);
	int A2_shared_int = (int)__cvta_generic_to_shared(A2_shared) + (lane_mask_lo * (64 * 2));
	b2_shared += (lane_id >> 2);
	int A3_shared_int = (int)__cvta_generic_to_shared(A3_shared) + (lane_mask_lo * (64 * 2));
	b3_shared += (lane_id >> 2);

	// *********************************************************************************************

	__pipeline_wait_prior(0);
	__syncthreads();

	// *********************************************************************************************

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

			// LUT ENCODING
			float2 uv_normalized = make_float2(
				uv.x * 0.29689279887990428378374198030042f,
				uv.y * 0.29689279887990428378374198030042f
			);

			// *************************************************************************************

			float4 RGBA;

			// !!! !!! !!!
			int lane_id = threadIdx.x & 31;
			// !!! !!! !!!

			// No buffering
			#pragma unroll
			for (int k = 0; k < 4; ++k) {
				half2 B[16];

				GenerateInputTensor(
					//uv, // HASH ENCODING
					uv_normalized, // LUT ENCODING

					v,

					conditioning_variable_ptr,
					ind_clamped,

					//features_P_hit_object_ptr, // HASH ENCODING
					features_shared, // LUT ENCODING

					k,
					B
				);

				float2 C[8] = {};

				Forward(
					A1_shared_int,
					b1_shared,
					A2_shared_int,
					b2_shared,
					A3_shared_int,
					b3_shared,

					k,
					B,
					C
				);

				float x, y;
				float coord;

				x = __shfl_sync(-1, C[0].x, lane_id >> 1);
				y = __shfl_sync(-1, C[0].y, lane_id >> 1);
				x = (lane_id & 1) ? y : x;
				x = __shfl_sync(-1, x, ((lane_id & 3) << 3) + (lane_id >> 2));
				coord = __shfl_sync(-1, x, ((lane_id & 7) << 2));
				if ((lane_id & 24) == (k << 3)) RGBA.x = coord;
				coord = __shfl_sync(-1, x, ((lane_id & 7) << 2) + 1);
				if ((lane_id & 24) == (k << 3)) RGBA.y = coord;
				coord = __shfl_sync(-1, x, ((lane_id & 7) << 2) + 2);
				if ((lane_id & 24) == (k << 3)) RGBA.z = coord;
				coord = __shfl_sync(-1, x, ((lane_id & 7) << 2) + 3);
				if ((lane_id & 24) == (k << 3)) RGBA.w = coord;
			}

			// 3 x ReLU + Sigmoid
			RGBA.x = fmaxf(RGBA.x, 0.0f);
			RGBA.y = fmaxf(RGBA.y, 0.0f);
			RGBA.z = fmaxf(RGBA.z, 0.0f);
			asm volatile ("rcp.approx.ftz.f32 %0, %1;" : "=f"(RGBA.w) : "f"(1.0f + __expf(-RGBA.w)));

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

			float opacity = T * (opacity1 * __expf(-0.5f * kappa_inv * r_squared_pow_kappa) * RGBA.w); // !!! !!! !!!

			if ((ind != -1) && (T >= T_threshold)) {
				// !!! !!! !!!
				// Paper: accumulate c_const * c_MLP with standard alpha compositing.
				RGB_aggregated.x = __fmaf_rn(RGBA_param.x * RGBA.x, opacity, RGB_aggregated.x);
				RGB_aggregated.y = __fmaf_rn(RGBA_param.y * RGBA.y, opacity, RGB_aggregated.y);
				RGB_aggregated.z = __fmaf_rn(RGBA_param.z * RGBA.z, opacity, RGB_aggregated.z);
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
				// L_n_partial = (o * (1.0 - torch.abs((N_median_tmp * N).sum(2)))).sum(0);
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

void CPyOptiXFLARERenderer_CUDA::Forward_training(
	half2 *conditioning_variable,
	float *features_P_hit_object,

	half *A1,
	float *b1,
	half *A2,
	float *b2,
	half *A3,
	float *b3,

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

	const int size =
		(sizeof(half) * 64 * 128) +
		(sizeof(float) * 64) +
		(sizeof(half) * 64 * 64) +
		(sizeof(float) * 64) +
		(sizeof(half) * 16 * 64) +
		(sizeof(float) * 16) +
		(sizeof(float) * (((16 * 16) + (25 * 25) + (40 * 40) + (64 * 64)) * 2));

	error_CUDA = cudaFuncSetAttribute(Forward_training_CUDA_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, size);
	if (error_CUDA != CUDA_SUCCESS) throw 0;

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

		Forward_training_CUDA_kernel<<<min((number_of_rays + (threads_per_block - 1)) / threads_per_block, SM_count), threads_per_block, size, stream>>>(
			conditioning_variable,
			features_P_hit_object,

			A1,
			b1,
			A2,
			b2,
			A3,
			b3,

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
