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

const int threads_per_block = 128;

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

template <
	int L_start, int L
> static __device__ __forceinline__ void d_HashEncoding2D_CUDA_kernel(
	float2 x_frag,

	int lane_id,
	int res_int,
	int mask,
	float *features,
	float2 (&delta4)[16],
	bool active,

	float2 &d_x,
	float *dL_d_features_ptr
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
	
	int L_num = L_start + (lane_id & 3);

	float2 f00 = ((float2 *)features)[ind00];
	float2 f10 = ((float2 *)features)[ind10];
	float2 f01 = ((float2 *)features)[ind01];
	float2 f11 = ((float2 *)features)[ind11];

	// *********************************************************************************************

	float2 u = x_coords;
	float2 v = make_float2(1.0f - u.x, 1.0f - u.y);

	// *********************************************************************************************

	float2 dfx = make_float2(
		__fmaf_rn(f10.x - f00.x, v.y, (f11.x - f01.x) * u.y),
		__fmaf_rn(f10.y - f00.y, v.y, (f11.y - f01.y) * u.y)
	);

	// *********************************************************************************************

	float2 dfy = make_float2(
		__fmaf_rn(f01.x - f00.x, v.x, (f11.x - f10.x) * u.x),
		__fmaf_rn(f01.y - f00.y, v.x, (f11.y - f10.y) * u.x)
	);

	// *********************************************************************************************

	float delta1 = __shfl_sync(-1, delta4[0].x, ((lane_id & 3) << 3) + (lane_id >> 3));
	float delta2 = __shfl_sync(-1, delta4[0].y, ((lane_id & 3) << 3) + (lane_id >> 3));
	float delta_x = ((lane_id >> 2) & 1) ? delta2 : delta1;

	delta1 = __shfl_sync(-1, delta4[0].x, ((lane_id & 3) << 3) + 4 + (lane_id >> 3));
	delta2 = __shfl_sync(-1, delta4[0].y, ((lane_id & 3) << 3) + 4 + (lane_id >> 3));
	float delta_y = ((lane_id >> 2) & 1) ? delta2 : delta1;

	// *********************************************************************************************

	// HASH ENCODING
	/*d_x.x = -res_int * __fmaf_rn(delta_x, dfx.x, delta_y * dfx.y); // !!! !!! !!!
	d_x.y = -res_int * __fmaf_rn(delta_x, dfy.x, delta_y * dfy.y); // !!! !!! !!!*/

	// LUT ENCODING
	d_x.x = 0.5f * 0.29689279887990428378374198030042f * (res_int - 1.0f) * __fmaf_rn(delta_x, dfx.x, delta_y * dfx.y); // !!! !!! !!!
	d_x.y = 0.5f * 0.29689279887990428378374198030042f * (res_int - 1.0f) * __fmaf_rn(delta_x, dfy.x, delta_y * dfy.y); // !!! !!! !!!

	// *********************************************************************************************

	int ind;

	// *********************************************************************************************

	if (active) {
		float tmp;

		tmp = v.x * v.y;
		atomicAdd(&((float2 *)dL_d_features_ptr)[ind00].x, delta_x * tmp);
		atomicAdd(&((float2 *)dL_d_features_ptr)[ind00].y, delta_y * tmp);

		tmp = u.x * v.y;
		atomicAdd(&((float2 *)dL_d_features_ptr)[ind10].x, delta_x * tmp);
		atomicAdd(&((float2 *)dL_d_features_ptr)[ind10].y, delta_y * tmp);

		tmp = v.x * u.y;
		atomicAdd(&((float2 *)dL_d_features_ptr)[ind01].x, delta_x * tmp);
		atomicAdd(&((float2 *)dL_d_features_ptr)[ind01].y, delta_y * tmp);

		tmp = u.x * u.y;
		atomicAdd(&((float2 *)dL_d_features_ptr)[ind11].x, delta_x * tmp);
		atomicAdd(&((float2 *)dL_d_features_ptr)[ind11].y, delta_y * tmp);
	}
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
	half *A1_shared,
	float *b1_shared,
	half *A2_shared,
	float *b2_shared,
	half *A3_shared,
	float *b3_shared,

	int k,
	half2 (&B)[16],

	__nv_bfloat162 (&z1)[8],
	float2       (&d_z1)[8],

	__nv_bfloat162 (&z2)[8],
	float2       (&d_z2)[8],

	float2 (&y3)[2] // !!! !!! !!!
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
		d_z1[i].x = b1_shared[(i << 3) + (lane_id >> 2)];
		d_z1[i].y = b1_shared[(i << 3) + (lane_id >> 2)];
	}

	__syncwarp();

	// *********************************************************************************************

	base = (int)__cvta_generic_to_shared(A1_shared) + (lane_mask_lo * (128 * 2));

	#pragma unroll
	for (int j = 0; j < 16; j += 2) {

		#pragma unroll
		for (int i = 0; i < 8; i += 2) {
			int col = lane_mask_hi + (j << 4);
			int j_swizzled = (col & -128) | ((col ^ mask_swizzle) & 112);
			int addr = base + (i * 8 * (128 * 2)) + j_swizzled;

			asm volatile(
				"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
				: "=r"(A11),
				"=r"(A21),
				"=r"(A12),
				"=r"(A22)
				: "r"(addr)
			);

			asm volatile(
				"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
				"{ %0, %1, %2, %3 }, "
				"{ %4, %5, %6, %7 }, "
				"{ %8, %9 }, "
				"{ %10, %11, %12, %13 };"
				: "=f"(d_z1[i].x), "=f"(d_z1[i].y), "=f"(d_z1[i + 1].x), "=f"(d_z1[i + 1].y)
				: "r"(A11), "r"(A21), "r"(A12), "r"(A22),
				"r"((uint32_t &)B[j]), "r"((uint32_t &)B[j + 1]),
				"f"(d_z1[i].x), "f"(d_z1[i].y), "f"(d_z1[i + 1].x), "f"(d_z1[i + 1].y)
			);
		}
	}

	// Hardswish
	#pragma unroll
	for (int i = 0; i < 8; ++i) {
		float2 y, z;
		float2 x_unclamped, x_clamped, d_z;

		y = d_z1[i];

		// z1
		z = make_float2(
			(y.x * fmaxf(fminf(y.x + 3.0f, 6.0f), 0.0f)) * (1.0f / 6.0f),
			(y.y * fmaxf(fminf(y.y + 3.0f, 6.0f), 0.0f)) * (1.0f / 6.0f)
		);
		z1[i] = __float22bfloat162_rn(z);

		// d_z1
		x_unclamped = make_float2(y.x + 3.0f, y.y + 3.0f);
		x_clamped = make_float2(
			fmaxf(fminf(x_unclamped.x, 6.0f), 0.0f),
			fmaxf(fminf(x_unclamped.y, 6.0f), 0.0f)
		);
		d_z = make_float2(
			(x_clamped.x + ((x_unclamped.x == x_clamped.x) ? y.x : 0.0f)) * (1.0f / 6.0f),
			(x_clamped.y + ((x_unclamped.y == x_clamped.y) ? y.y : 0.0f)) * (1.0f / 6.0f)
		);
		d_z1[i] = d_z;

		// B
		asm volatile (
			"movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
			: "=r"((uint32_t &)B[i]) 
			: "r" (__half22uint32_t(__float22half2_rn(z)))
		);
	}

	// *********************************************************************************************
	// * LAYER 2                                                                                   *
	// *********************************************************************************************

	/*#pragma unroll
	for (int i = 0; i < 8; ++i)
		asm volatile (
			"movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
			: "=r"((uint32_t &)B[i]) 
			: "r" ((uint32_t &)z1[i])
		);*/

	// *********************************************************************************************

	#pragma unroll
	for (int i = 0; i < 8; ++i) {
		d_z2[i].x = b2_shared[(i << 3) + (lane_id >> 2)];
		d_z2[i].y = b2_shared[(i << 3) + (lane_id >> 2)];
	}

	__syncwarp();

	// *********************************************************************************************

	base = (int)__cvta_generic_to_shared(A2_shared) + (lane_mask_lo * (64 * 2));

	#pragma unroll
	for (int j = 0; j < 8; j += 2) {

		#pragma unroll
		for (int i = 0; i < 8; i += 2) {
			int col = lane_mask_hi + (j << 4);
			int j_swizzled = (col & -128) | ((col ^ mask_swizzle) & 112);
			int addr = base + (i * 8 * (64 * 2)) + j_swizzled;

			asm volatile(
				"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
				: "=r"(A11),
				"=r"(A21),
				"=r"(A12),
				"=r"(A22)
				: "r"(addr)
			);

			asm volatile(
				"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
				"{ %0, %1, %2, %3 }, "
				"{ %4, %5, %6, %7 }, "
				"{ %8, %9 }, "
				"{ %10, %11, %12, %13 };"
				: "=f"(d_z2[i].x), "=f"(d_z2[i].y), "=f"(d_z2[i + 1].x), "=f"(d_z2[i + 1].y)
				: "r"(A11), "r"(A21), "r"(A12), "r"(A22),
				"r"((uint32_t &)B[j]), "r"((uint32_t &)B[j + 1]),
				"f"(d_z2[i].x), "f"(d_z2[i].y), "f"(d_z2[i + 1].x), "f"(d_z2[i + 1].y)
			);
		}
	}

	// Hardswish
	#pragma unroll
	for (int i = 0; i < 8; ++i) {
		float2 y, z;
		float2 x_unclamped, x_clamped, d_z;

		y = d_z2[i];

		// z2
		z = make_float2(
			(y.x * fmaxf(fminf(y.x + 3.0f, 6.0f), 0.0f)) * (1.0f / 6.0f),
			(y.y * fmaxf(fminf(y.y + 3.0f, 6.0f), 0.0f)) * (1.0f / 6.0f)
		);
		z2[i] = __float22bfloat162_rn(z);

		// d_z2
		x_unclamped = make_float2(y.x + 3.0f, y.y + 3.0f);
		x_clamped = make_float2(
			fmaxf(fminf(x_unclamped.x, 6.0f), 0.0f),
			fmaxf(fminf(x_unclamped.y, 6.0f), 0.0f)
		);
		d_z = make_float2(
			(x_clamped.x + ((x_unclamped.x == x_clamped.x) ? y.x : 0.0f)) * (1.0f / 6.0f),
			(x_clamped.y + ((x_unclamped.y == x_clamped.y) ? y.y : 0.0f)) * (1.0f / 6.0f)
		);
		d_z2[i] = d_z;

		// B
		asm volatile (
			"movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
			: "=r"((uint32_t &)B[i]) 
			: "r" (__half22uint32_t(__float22half2_rn(z)))
		);
	}

	// *********************************************************************************************
	// * LAYER 3                                                                                   *
	// *********************************************************************************************

	/*#pragma unroll
	for (int i = 0; i < 8; ++i)
		asm volatile (
			"movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
			: "=r"((uint32_t &)B[i]) 
			: "r" ((uint32_t &)z2[i])
		);*/

	// *********************************************************************************************

	#pragma unroll
	for (int i = 0; i < 2; ++i) {
		y3[i].x = b3_shared[(i << 3) + (lane_id >> 2)];
		y3[i].y = b3_shared[(i << 3) + (lane_id >> 2)];
	}

	__syncwarp();

	// *********************************************************************************************

	base = (int)__cvta_generic_to_shared(A3_shared) + (lane_mask_lo * (64 * 2));

	#pragma unroll
	for (int j = 0; j < 8; j += 2) {

		#pragma unroll
		for (int i = 0; i < 2; i += 2) {
			int col = lane_mask_hi + (j << 4);
			int j_swizzled = (col & -128) | ((col ^ mask_swizzle) & 112);
			int addr = base + (i * 8 * (64 * 2)) + j_swizzled;

			asm volatile(
				"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
				: "=r"(A11),
				"=r"(A21),
				"=r"(A12),
				"=r"(A22)
				: "r"(addr)
			);

			asm volatile(
				"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
				"{ %0, %1, %2, %3 }, "
				"{ %4, %5, %6, %7 }, "
				"{ %8, %9 }, "
				"{ %10, %11, %12, %13 };"
				: "=f"(y3[i].x), "=f"(y3[i].y), "=f"(y3[i + 1].x), "=f"(y3[i + 1].y)
				: "r"(A11), "r"(A21), "r"(A12), "r"(A22),
				"r"((uint32_t &)B[j]), "r"((uint32_t &)B[j + 1]),
				"f"(y3[i].x), "f"(y3[i].y), "f"(y3[i + 1].x), "f"(y3[i + 1].y)
			);
		}
	}
}

// *** *** *** *** ***

static __device__ __forceinline__ void Backward(
	float *features,

	half *A1_shared,
	float *b1_shared,
	half *A2_shared,
	float *b2_shared,
	half *A3_shared,
	float *b3_shared,

	int Gaussian_ind,
	bool Gaussian_active,
	float2 x_frag,
	float2 &d_hash,

	__nv_bfloat162 (&z0)[4][16],

	__nv_bfloat162 (&z1)[4][8],
	float2 (&d_z1)[4][8],

	__nv_bfloat162 (&z2)[4][8],
	float2 (&d_z2)[4][8],

	float2 (&delta1)[4][2],

	float *dL_dw3_ptr,
	float *dL_db3_ptr,
	float *dL_dw2_ptr,
	float *dL_db2_ptr,
	float *dL_dw1_ptr,
	float *dL_db1_ptr,
	float *dL_d_conditioning_ptr,
	float *dL_d_features_ptr
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

	int wid = (threadIdx.x + (blockIdx.x * blockDim.x)) >> 5;

	// *********************************************************************************************
	// * LAYER 3                                                                                   *
	// *********************************************************************************************

	#pragma unroll
	for (int j = 0; j < 1; ++j) {
		float2 C = {};

		#pragma unroll
		for (int k = 0; k < 4; ++k) {
			C.x += delta1[k][j].x;
			C.y += delta1[k][j].y;
		}

		// dL_db3
		atomicAdd(dL_db3_ptr + (wid << 6) + (j << (5 + 1)) + (0 << 5) + (threadIdx.x & 31), C.x);
		atomicAdd(dL_db3_ptr + (wid << 6) + (j << (5 + 1)) + (1 << 5) + (threadIdx.x & 31), C.y);
	}

	// *********************************************************************************************

	#pragma unroll
	for (int j = 0; j < 8; ++j) {
		float2 C[2] = {};

		#pragma unroll
		for (int k = 0; k < 4; k += 2) {
			// !!! !!! !!!
			z2[k][j].x = (__hisnan(z2[k][j].x)) ? __float2bfloat16(0.0f) : z2[k][j].x;
			z2[k][j].y = (__hisnan(z2[k][j].y)) ? __float2bfloat16(0.0f) : z2[k][j].y;
			z2[k + 1][j].x = (__hisnan(z2[k + 1][j].x)) ? __float2bfloat16(0.0f) : z2[k + 1][j].x;
			z2[k + 1][j].y = (__hisnan(z2[k + 1][j].y)) ? __float2bfloat16(0.0f) : z2[k + 1][j].y;
			// !!! !!! !!!

			asm volatile(
				"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
				"{ %0, %1, %2, %3 },"
				"{ %4, %5, %6, %7 },"
				"{ %8, %9 },"
				"{ %10, %11, %12, %13 };"
				: "=f"(C[0].x), "=f"(C[0].y), "=f"(C[1].x), "=f"(C[1].y)
				: "r"(__bfloat1622uint32_t(__float22bfloat162_rn(delta1[k][0]))),
				      "r"(__bfloat1622uint32_t(__float22bfloat162_rn(delta1[k][1]))),
				      "r"(__bfloat1622uint32_t(__float22bfloat162_rn(delta1[k + 1][0]))),
				      "r"(__bfloat1622uint32_t(__float22bfloat162_rn(delta1[k + 1][1]))),
				  "r"((uint32_t &)z2[k][j]),
				      "r"((uint32_t &)z2[k + 1][j]),
				  "f"(C[0].x),
				      "f"(C[0].y),
				      "f"(C[1].x),
				      "f"(C[1].y)
			);
		}

		atomicAdd(dL_dw3_ptr + (wid << (3 + 6)) + (j << (5 + 1)) + (0 << 5) + (threadIdx.x & 31), C[0].x);
		atomicAdd(dL_dw3_ptr + (wid << (3 + 6)) + (j << (5 + 1)) + (1 << 5) + (threadIdx.x & 31), C[0].y);
	}

	// *********************************************************************************************
	// * LAYER 2                                                                                   *
	// *********************************************************************************************

	__nv_bfloat162 delta1_T_b16[4][8];
	float2 delta2[4][8] = {};

	// *********************************************************************************************

	#pragma unroll
	for (int k = 0; k < 4; ++k) {

		#pragma unroll
		for (int i = 0; i < 2; ++i)
			asm volatile (
				"movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
				: "=r"((uint32_t &)delta1_T_b16[k][i]) 
				: "r" (__bfloat1622uint32_t(__float22bfloat162_rn(delta1[k][i])))
			);
	}

	// *********************************************************************************************

	base = (int)__cvta_generic_to_shared(A3_shared) + (lane_mask_lo * (64 * 2));

	#pragma unroll
	for (int j = 0; j < 8; j += 2) {

		#pragma unroll
		for (int i = 0; i < 2; i += 2) {
			int col = lane_mask_hi + (j << 4);
			int j_swizzled = (col & -128) | ((col ^ mask_swizzle) & 112);
			int addr = base + (i * 8 * (64 * 2)) + j_swizzled;

			asm volatile(
				"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
				: "=r"(A11),
				  "=r"(A21),
				  "=r"(A12),
				  "=r"(A22)
				: "r"(addr)
			);

			A11 = __bfloat1622uint32_t(__float22bfloat162_rn(__half22float2((half2 &) A11)));
			A21 = __bfloat1622uint32_t(__float22bfloat162_rn(__half22float2((half2 &) A21)));
			A12 = __bfloat1622uint32_t(__float22bfloat162_rn(__half22float2((half2 &) A12)));
			A22 = __bfloat1622uint32_t(__float22bfloat162_rn(__half22float2((half2 &) A22)));

			#pragma unroll
			for (int k = 0; k < 4; ++k)
				asm volatile(
					"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
					"{ %0, %1, %2, %3 }, "
					"{ %4, %5, %6, %7 }, "
					"{ %8, %9 }, "
					"{ %10, %11, %12, %13 };"
					: "=f"(delta2[k][j].x), "=f"(delta2[k][j].y), "=f"(delta2[k][j + 1].x), "=f"(delta2[k][j + 1].y)
					: "r"(A11), "r"(A12), "r"(A21), "r"(A22), // !!! !!! !!!
					  "r"((uint32_t &)delta1_T_b16[k][i]), "r"((uint32_t &)delta1_T_b16[k][i + 1]),
					  "f"(delta2[k][j].x), "f"(delta2[k][j].y), "f"(delta2[k][j + 1].x), "f"(delta2[k][j + 1].y)
				);
		}
	}

	// *********************************************************************************************

	#pragma unroll
	for (int j = 0; j < 8; ++j) {
		float2 C = {};

		#pragma unroll
		for (int k = 0; k < 4; ++k) {
			delta2[k][j].x *= d_z2[k][j].x;
			delta2[k][j].y *= d_z2[k][j].y;

			C.x += delta2[k][j].x;
			C.y += delta2[k][j].y;
		}

		// dL_db2
		atomicAdd(dL_db2_ptr + (wid << 9) + (j << (5 + 1)) + (0 << 5) + (threadIdx.x & 31), C.x);
		atomicAdd(dL_db2_ptr + (wid << 9) + (j << (5 + 1)) + (1 << 5) + (threadIdx.x & 31), C.y);
	}

	// *********************************************************************************************

	#pragma unroll
	for (int k = 0; k < 4; ++k) {

		#pragma unroll
		for (int j = 0; j < 8; ++j) {
			// !!! !!! !!!
			z1[k][j].x = (__hisnan(z1[k][j].x)) ? __float2bfloat16(0.0f) : z1[k][j].x;
			z1[k][j].y = (__hisnan(z1[k][j].y)) ? __float2bfloat16(0.0f) : z1[k][j].y;
			// !!! !!! !!!
		}
	}

	#pragma unroll
	for (int i = 0; i < 8; i += 2) {

		#pragma unroll
		for (int j = 0; j < 8; ++j) {
			float2 C[2] = {};

			#pragma unroll
			for (int k = 0; k < 4; k += 2)
				asm volatile(
					"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
					"{ %0, %1, %2, %3 },"
					"{ %4, %5, %6, %7 },"
					"{ %8, %9 },"
					"{ %10, %11, %12, %13 };"
					: "=f"(C[0].x), "=f"(C[0].y), "=f"(C[1].x), "=f"(C[1].y)
					: "r"(__bfloat1622uint32_t(__float22bfloat162_rn(delta2[k][i]))),
					      "r"(__bfloat1622uint32_t(__float22bfloat162_rn(delta2[k][i + 1]))),
						  "r"(__bfloat1622uint32_t(__float22bfloat162_rn(delta2[k + 1][i]))),
					      "r"(__bfloat1622uint32_t(__float22bfloat162_rn(delta2[k + 1][i + 1]))),
					  "r"((uint32_t &)z1[k][j]),
					      "r"((uint32_t &)z1[k + 1][j]),
					  "f"(C[0].x),
					      "f"(C[0].y),
					      "f"(C[1].x),
					      "f"(C[1].y)
				);

			atomicAdd(dL_dw2_ptr + (wid << 12) + (((i << 3) + j) << 6) + (0 << 5) + (threadIdx.x & 31), C[0].x);
			atomicAdd(dL_dw2_ptr + (wid << 12) + (((i << 3) + j) << 6) + (1 << 5) + (threadIdx.x & 31), C[0].y);
			atomicAdd(dL_dw2_ptr + (wid << 12) + ((((i + 1) << 3) + j) << 6) + (0 << 5) + (threadIdx.x & 31), C[1].x);
			atomicAdd(dL_dw2_ptr + (wid << 12) + ((((i + 1) << 3) + j) << 6) + (1 << 5) + (threadIdx.x & 31), C[1].y);
		}
	}

	// *********************************************************************************************
	// * LAYER 1                                                                                   *
	// *********************************************************************************************

	__nv_bfloat162 delta2_T_b16[4][8];
	float2         delta3      [4][8] = {};

	// *********************************************************************************************

	#pragma unroll
	for (int k = 0; k < 4; ++k) {

		#pragma unroll
		for (int i = 0; i < 8; ++i)
			asm volatile (
				"movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
				: "=r"((uint32_t &)delta2_T_b16[k][i]) 
				: "r" (__bfloat1622uint32_t(__float22bfloat162_rn(delta2[k][i])))
			);
	}

	// *********************************************************************************************

	base = (int)__cvta_generic_to_shared(A2_shared) + (lane_mask_lo * (64 * 2));

	#pragma unroll
	for (int j = 0; j < 8; j += 2) {

		#pragma unroll
		for (int i = 0; i < 8; i += 2) {
			int col = lane_mask_hi + (j << 4);
			int j_swizzled = (col & -128) | ((col ^ mask_swizzle) & 112);
			int addr = base + (i * 8 * (64 * 2)) + j_swizzled;

			asm volatile(
				"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
				: "=r"(A11),
				  "=r"(A21),
				  "=r"(A12),
				  "=r"(A22)
				: "r"(addr)
			);

			A11 = __bfloat1622uint32_t(__float22bfloat162_rn(__half22float2((half2 &)A11)));
			A21 = __bfloat1622uint32_t(__float22bfloat162_rn(__half22float2((half2 &)A21)));
			A12 = __bfloat1622uint32_t(__float22bfloat162_rn(__half22float2((half2 &)A12)));
			A22 = __bfloat1622uint32_t(__float22bfloat162_rn(__half22float2((half2 &)A22)));

			#pragma unroll
			for (int k = 0; k < 4; ++k)
				asm volatile(
					"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
					"{ %0, %1, %2, %3 }, "
					"{ %4, %5, %6, %7 }, "
					"{ %8, %9 }, "
					"{ %10, %11, %12, %13 };"
					: "=f"(delta3[k][j].x), "=f"(delta3[k][j].y), "=f"(delta3[k][j + 1].x), "=f"(delta3[k][j + 1].y)
					: "r"(A11), "r"(A12), "r"(A21), "r"(A22), // !!! !!! !!!
					  "r"((uint32_t &)delta2_T_b16[k][i]), "r"((uint32_t &)delta2_T_b16[k][i + 1]),
					  "f"(delta3[k][j].x), "f"(delta3[k][j].y), "f"(delta3[k][j + 1].x), "f"(delta3[k][j + 1].y)
				);
		}
	}

	// *********************************************************************************************

	#pragma unroll
	for (int j = 0; j < 8; ++j) {
		float2 C = {};

		#pragma unroll
		for (int k = 0; k < 4; ++k) {
			delta3[k][j].x *= d_z1[k][j].x;
			delta3[k][j].y *= d_z1[k][j].y;

			C.x += delta3[k][j].x;
			C.y += delta3[k][j].y;
		}

		// dL_db1
		atomicAdd(dL_db1_ptr + (wid << 9) + (j << (5 + 1)) + (0 << 5) + (threadIdx.x & 31), C.x);
		atomicAdd(dL_db1_ptr + (wid << 9) + (j << (5 + 1)) + (1 << 5) + (threadIdx.x & 31), C.y);
	}

	// *********************************************************************************************

	#pragma unroll
	for (int k = 0; k < 4; ++k) {

		#pragma unroll
		for (int j = 0; j < 16; ++j) {
			// !!! !!! !!!
			z0[k][j].x = (__hisnan(z0[k][j].x)) ? __float2bfloat16(0.0f) : z0[k][j].x;
			z0[k][j].y = (__hisnan(z0[k][j].y)) ? __float2bfloat16(0.0f) : z0[k][j].y;
			// !!! !!! !!!

			asm volatile (
				"movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
				: "=r"((uint32_t &)z0[k][j]) 
				: "r" ((uint32_t &)z0[k][j])
			);
		}
	}

	#pragma unroll
	for (int i = 0; i < 8; i += 2) {

		#pragma unroll
		for (int j = 0; j < 16; ++j) {
			float2 C[2] = {};

			#pragma unroll
			for (int k = 0; k < 4; k += 2)
				asm volatile(
					"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
					"{ %0, %1, %2, %3 },"
					"{ %4, %5, %6, %7 },"
					"{ %8, %9 },"
					"{ %10, %11, %12, %13 };"
					: "=f"(C[0].x), "=f"(C[0].y), "=f"(C[1].x), "=f"(C[1].y)
					: "r"(__bfloat1622uint32_t(__float22bfloat162_rn(delta3[k][i]))),
					      "r"(__bfloat1622uint32_t(__float22bfloat162_rn(delta3[k][i + 1]))),
					      "r"(__bfloat1622uint32_t(__float22bfloat162_rn(delta3[k + 1][i]))),
					      "r"(__bfloat1622uint32_t(__float22bfloat162_rn(delta3[k + 1][i + 1]))),
					  "r"((uint32_t &)z0[k][j]),
					      "r"((uint32_t &)z0[k + 1][j]),
					      "f"(C[0].x),
					      "f"(C[0].y),
					      "f"(C[1].x),
					      "f"(C[1].y)
					);


			atomicAdd(dL_dw1_ptr + (wid << 13) + (((i << 4) + j) << 6) + (0 << 5) + (threadIdx.x & 31), C[0].x);
			atomicAdd(dL_dw1_ptr + (wid << 13) + (((i << 4) + j) << 6) + (1 << 5) + (threadIdx.x & 31), C[0].y);
			atomicAdd(dL_dw1_ptr + (wid << 13) + ((((i + 1) << 4) + j) << 6) + (0 << 5) + (threadIdx.x & 31), C[1].x);
			atomicAdd(dL_dw1_ptr + (wid << 13) + ((((i + 1) << 4) + j) << 6) + (1 << 5) + (threadIdx.x & 31), C[1].y);
		}
	}

	// *********************************************************************************************
	// * LAYER 0                                                                                   *
	// *********************************************************************************************

	__nv_bfloat162 delta3_T_b16[4][8];
	float2         delta4      [4][16] = {};

	// *********************************************************************************************

	#pragma unroll
	for (int k = 0; k < 4; ++k) {
		#pragma unroll

		for (int i = 0; i < 8; ++i)
			asm volatile (
				"movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
				: "=r"((uint32_t &)delta3_T_b16[k][i]) 
				: "r" (__bfloat1622uint32_t(__float22bfloat162_rn(delta3[k][i])))
			);
	}

	// *********************************************************************************************

	base = (int)__cvta_generic_to_shared(A1_shared) + (lane_mask_lo * (128 * 2));

	#pragma unroll
	for (int j = 0; j < 16; j += 2) {

		#pragma unroll
		for (int i = 0; i < 8; i += 2) {
			int col = lane_mask_hi + (j << 4);
			int j_swizzled = (col & -128) | ((col ^ mask_swizzle) & 112);
			int addr = base + (i * 8 * (128 * 2)) + j_swizzled;

			asm volatile(
				"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
				: "=r"(A11),
				"=r"(A21),
				"=r"(A12),
				"=r"(A22)
				: "r"(addr)
				);

			A11 = __bfloat1622uint32_t(__float22bfloat162_rn(__half22float2((half2 &)A11)));
			A21 = __bfloat1622uint32_t(__float22bfloat162_rn(__half22float2((half2 &)A21)));
			A12 = __bfloat1622uint32_t(__float22bfloat162_rn(__half22float2((half2 &)A12)));
			A22 = __bfloat1622uint32_t(__float22bfloat162_rn(__half22float2((half2 &)A22)));

			#pragma unroll
			for (int k = 0; k < 4; ++k)
				asm volatile(
					"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
					"{ %0, %1, %2, %3 }, "
					"{ %4, %5, %6, %7 }, "
					"{ %8, %9 }, "
					"{ %10, %11, %12, %13 };"
					: "=f"(delta4[k][j].x), "=f"(delta4[k][j].y), "=f"(delta4[k][j + 1].x), "=f"(delta4[k][j + 1].y)
					: "r"(A11), "r"(A12), "r"(A21), "r"(A22), // !!! !!! !!!
					  "r"((uint32_t &)delta3_T_b16[k][i]), "r"((uint32_t &)delta3_T_b16[k][i + 1]),
					  "f"(delta4[k][j].x), "f"(delta4[k][j].y), "f"(delta4[k][j + 1].x), "f"(delta4[k][j + 1].y)
				);
		}
	}

	// *********************************************************************************************

	#pragma unroll
	for (int k = 0; k < 4; ++k) {
	
		#pragma unroll
		for (int j = 0; j < 12; ++j) {
			float coord;
			int ind;
			bool active;

			coord = __shfl_sync(-1, delta4[k][4 + j].x, ((lane_id & 7) << 2) + (lane_id >> 3));
			ind = __shfl_sync(-1, Gaussian_ind, (k << 3) + ((lane_id >> 3) << 1));
			active = __shfl_sync(-1, Gaussian_active, (k << 3) + ((lane_id >> 3) << 1));

			if (active)
				atomicAdd(dL_d_conditioning_ptr + (ind * 96) + (j << 3) + (lane_id & 7), coord);

			coord = __shfl_sync(-1, delta4[k][4 + j].y, ((lane_id & 7) << 2) + (lane_id >> 3));
			ind = __shfl_sync(-1, Gaussian_ind, (k << 3) + ((lane_id >> 3) << 1) + 1);
			active = __shfl_sync(-1, Gaussian_active, (k << 3) + ((lane_id >> 3) << 1) + 1);

			if (active)
				atomicAdd(dL_d_conditioning_ptr + (ind * 96) + (j << 3) + (lane_id & 7), coord);
		}

		// *****************************************************************************************
		// * dL/d_hash                                                                             *
		// *****************************************************************************************

		float b;
		float res_float;
		int mask;
		int res_int;

		int srcLane;

		// *****************************************************************************************
		// * HASH ENCODING: P_hit_object                                                           *
		// *****************************************************************************************

		b = 1.5874010519681994747517056392723;

		//res_float = -N_min_P_hit_object; // !!! !!! !!! HASH ENCODING !!! !!! !!!
		res_float = N_min_P_hit_object; // LUT ENCODING
		if ((lane_id & 1) == 1) res_float *= b;
		b *= b;
		if ((lane_id & 2) == 2) res_float *= b;
		// b *= b;

		mask = T_P_hit_object - 1;

		// *****************************************************************************************

		srcLane = (k << 3) + (lane_id >> 2);
		float2 uv = make_float2(
			__shfl_sync(-1, x_frag.x, srcLane),
			__shfl_sync(-1, x_frag.y, srcLane)
		);

		// *****************************************************************************************

		res_int = __float2int_rn(res_float);

		// *****************************************************************************************

		bool active = __shfl_sync(-1, Gaussian_active, (k << 3) + (lane_id >> 2));

		float2 d_x;

		d_HashEncoding2D_CUDA_kernel<0 * 4, 4>(uv, lane_id, res_int, mask, features, delta4[k], active, d_x, dL_d_features_ptr);

		// *****************************************************************************************

		d_x.x += __shfl_xor_sync(-1, d_x.x, 1);
		d_x.x += __shfl_xor_sync(-1, d_x.x, 2);

		d_x.y += __shfl_xor_sync(-1, d_x.y, 1);
		d_x.y += __shfl_xor_sync(-1, d_x.y, 2);

		// *****************************************************************************************

		float coord = __shfl_sync(-1, d_x.x, (lane_id & 7) << 2);
		if ((lane_id & 24) == (k << 3)) d_hash.x = coord;
		coord = __shfl_sync(-1, d_x.y, (lane_id & 7) << 2);
		if ((lane_id & 24) == (k << 3)) d_hash.y = coord;
	}
}

// *** *** *** *** ***

// Paper: analytic backward pass for generalized-Gaussian compositing, including
// gradients of primitive geometry, opacity, shape, and decoder inputs.
static __global__ void __launch_bounds__(threads_per_block, 1) Backward_CUDA_kernel(
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
	float *dL_dw3_ptr,
	float *dL_db3_ptr,
	float *dL_dw2_ptr,
	float *dL_db2_ptr,
	float *dL_dw1_ptr,
	float *dL_db1_ptr,
	float *dL_d_conditioning_ptr,
	float *dL_d_deatures_ptr,
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
	extern __shared__ char scratchpad[];

	half *A1_shared = (half *)scratchpad;
	float *b1_shared = (float *)(A1_shared + (64 * 128));
	half *A2_shared = (half *)(b1_shared + 64);
	float *b2_shared = (float *)(A2_shared + (64 * 64));
	half *A3_shared = (half *)(b2_shared + 64);
	float *b3_shared = (float *)(A3_shared + (16 * 64));
	float *features_shared = (float *)(b3_shared + 16);

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

			// LUT ENCODING
			float2 uv_normalized = make_float2(
				uv.x * 0.29689279887990428378374198030042f,
				uv.y * 0.29689279887990428378374198030042f
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

			float4 y3_unpacked;

			// !!! !!! !!!
			int lane_id = threadIdx.x & 31;
			// !!! !!! !!!

			// *************************************************************************************

			half2    z0[4][16];

			__nv_bfloat162 z1[4][8];
			float2       d_z1[4][8];

			__nv_bfloat162 z2[4][8];
			float2       d_z2[4][8];

			// *************************************************************************************

			// FORWARD

			#pragma unroll
			for (int k = 0; k < 4; ++k) {
				GenerateInputTensor(
					//uv, // HASH ENCODING
					uv_normalized, // LUT ENCODING

					v,

					conditioning_variable_ptr,
					ind_clamped,

					//features_P_hit_object_ptr, // HASH ENCODING
					features_shared, // LUT ENCODING

					k,
					z0[k]
				);

				// *********************************************************************************

				half2 B[16];

				#pragma unroll
				for (int j = 0; j < 16; ++j)
					B[j] = z0[k][j];

				// *********************************************************************************

				float2 y3[2] = {};

				Forward(
					A1_shared,
					b1_shared,
					A2_shared,
					b2_shared,
					A3_shared,
					b3_shared,

					k,
					B,
					
					z1[k], d_z1[k],
					z2[k], d_z2[k],
					y3
				);

				// *********************************************************************************

				float first, second;
				float coord;

				first  = __shfl_sync(-1, y3[0].x, lane_id >> 1);
				second = __shfl_sync(-1, y3[0].y, lane_id >> 1);
				first = (lane_id & 1) ? second : first;
				first = __shfl_sync(-1, first, ((lane_id & 3) << 3) + (lane_id >> 2));

				// *********************************************************************************

				coord = __shfl_sync(-1, first, ((lane_id & 7) << 2));
				if ((lane_id & 24) == (k << 3)) y3_unpacked.x = coord;
				coord = __shfl_sync(-1, first, ((lane_id & 7) << 2) + 1);
				if ((lane_id & 24) == (k << 3)) y3_unpacked.y = coord;
				coord = __shfl_sync(-1, first, ((lane_id & 7) << 2) + 2);
				if ((lane_id & 24) == (k << 3)) y3_unpacked.z = coord;
				coord = __shfl_sync(-1, first, ((lane_id & 7) << 2) + 3);
				if ((lane_id & 24) == (k << 3)) y3_unpacked.w = coord;
			}

			// *************************************************************************************

			float4 z3;
			float4 d_z3;

			// 3 x ReLU + Sigmoid
			z3.x = fmaxf(y3_unpacked.x, 0.0f);
			z3.y = fmaxf(y3_unpacked.y, 0.0f);
			z3.z = fmaxf(y3_unpacked.z, 0.0f);
			asm volatile ("rcp.approx.ftz.f32 %0, %1;" : "=f"(z3.w) : "f"(1.0f + __expf(-y3_unpacked.w)));

			d_z3 = make_float4(
				y3_unpacked.x >= 0.0f,
				y3_unpacked.y >= 0.0f,
				y3_unpacked.z >= 0.0f,
				z3.w * (1.0f - z3.w)
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
			float dL_dA_msq = dL_dA_delta1 * z3.w;
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
					__fmaf_rn(z3.x * RGBA_param.x, T_prev, dL_dA_RGB.x),
					__fmaf_rn(
						dL_dI.y,
						__fmaf_rn(z3.y * RGBA_param.y, T_prev, dL_dA_RGB.y),
						dL_dI.z * __fmaf_rn(z3.z * RGBA_param.z, T_prev, dL_dA_RGB.z)
					)
				),
				1.0f - opacity_total
			);
			//dL_dA *= dL_dA_partial;
			dL_dA *= (dL_dA_partial + dL_dalpha + dLn_dalpha); // Depth and normal regularization

			opacity_total *= T_prev;

			float3 dL_dRGB = make_float3(
				z3.x * opacity_total,
				z3.y * opacity_total,
				z3.z * opacity_total
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

			// delta1
			float4 delta1 = make_float4(
				active ? ((dL_dI.x * RGBA_param.x * dL_dA_msq * T_prev) * d_z3.x) : 0.0f,
				active ? ((dL_dI.y * RGBA_param.y * dL_dA_msq * T_prev) * d_z3.y) : 0.0f,
				active ? ((dL_dI.z * RGBA_param.z * dL_dA_msq * T_prev) * d_z3.z) : 0.0f,
				//active ? ((dL_dA_delta1 * dL_dA_partial) * d_z3.w) : 0.0f
				active ? ((dL_dA_delta1 * (dL_dA_partial + dL_dalpha + dLn_dalpha)) * d_z3.w) : 0.0f // Depth and normal regularization
			);

			// *************************************************************************************

			// BACKWARD

			float2 delta1_packed[4][2] = {};
			__nv_bfloat162 z0_bf16[4][16] = {};

			// !!! !!! !!!
			float2 d_x = make_float2(0.0f, 0.0f);
			// !!! !!! !!!

			#pragma unroll
			for (int k = 0; k < 4; ++k) {
				float coord;
				float tmp;

				coord = __shfl_sync(-1, delta1.x, (k << 3) + (lane_id & 7));
				if ((lane_id & 24) == (0 << 3)) tmp = coord;
				coord = __shfl_sync(-1, delta1.y, (k << 3) + (lane_id & 7));
				if ((lane_id & 24) == (1 << 3)) tmp = coord;
				coord = __shfl_sync(-1, delta1.z, (k << 3) + (lane_id & 7));
				if ((lane_id & 24) == (2 << 3)) tmp = coord;
				coord = __shfl_sync(-1, delta1.w, (k << 3) + (lane_id & 7));
				if ((lane_id & 24) == (3 << 3)) tmp = coord;

				delta1_packed[k][0] = make_float2(
					__shfl_sync(-1, tmp, (lane_id << 1) + 0),
					__shfl_sync(-1, tmp, (lane_id << 1) + 1)
				);
				if (lane_id >= 16)
					delta1_packed[k][0] = make_float2(0.0f, 0.0f);

				// *********************************************************************************

				#pragma unroll
				for (int j = 0; j < 16; ++j)
					z0_bf16[k][j] = __float22bfloat162_rn(__half22float2(z0[k][j]));
			}
				
			Backward(
				//features_P_hit_object_ptr, // HASH ENCODING
				features_shared, // LUT ENCODING

				A1_shared,
				b1_shared,
				A2_shared,
				b2_shared,
				A3_shared,
				b3_shared,

				ind,
				active,

				//uv, // HASH ENCODING
				uv_normalized, // LUT ENCODING

				d_x,

				z0_bf16,
				z1, d_z1,
				z2, d_z2,

				delta1_packed,

				dL_dw3_ptr,
				dL_db3_ptr,
				dL_dw2_ptr,
				dL_db2_ptr,
				dL_dw1_ptr,
				dL_db1_ptr,
				dL_d_conditioning_ptr,
				dL_d_deatures_ptr
			);

			// *************************************************************************************

			// dL/dm
			float3 dL_dm1 = make_float3(
				__fmaf_rn(du_dm.x, d_x.x, dv_dm.x * d_x.y),
				__fmaf_rn(du_dm.y, d_x.x, dv_dm.y * d_x.y),
				__fmaf_rn(du_dm.z, d_x.x, dv_dm.z * d_x.y)
			);

			float3 dL_dm2 = make_float3(
				//dL_dA_msq * dL_dA_partial * r_squared_pow_kappa_minus_one * __fmaf_rn(uv.x, du_dm.x, uv.y * dv_dm.x),
				//dL_dA_msq * dL_dA_partial * r_squared_pow_kappa_minus_one * __fmaf_rn(uv.x, du_dm.y, uv.y * dv_dm.y),
				//dL_dA_msq * dL_dA_partial * r_squared_pow_kappa_minus_one * __fmaf_rn(uv.x, du_dm.z, uv.y * dv_dm.z)
				dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha) * r_squared_pow_kappa_minus_one * __fmaf_rn(uv.x, du_dm.x, uv.y * dv_dm.x), // Depth and normal regularization
				dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha) * r_squared_pow_kappa_minus_one * __fmaf_rn(uv.x, du_dm.y, uv.y * dv_dm.y), // Depth and normal regularization
				dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha) * r_squared_pow_kappa_minus_one * __fmaf_rn(uv.x, du_dm.z, uv.y * dv_dm.z)  // Depth and normal regularization
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
				atomicAdd(dL_dm_ptr + (ind * 3) + 0, dL_dm1.x - dL_dm2.x + dL_dm3.x);
				atomicAdd(dL_dm_ptr + (ind * 3) + 1, dL_dm1.y - dL_dm2.y + dL_dm3.y);
				atomicAdd(dL_dm_ptr + (ind * 3) + 2, dL_dm1.z - dL_dm2.z + dL_dm3.z);
			}

			// *************************************************************************************

			// dL/ds
			float2 dL_ds = make_float2(
				//(-uv.x * d_x.x) + (dL_dA_msq * dL_dA_partial * r_squared_pow_kappa_minus_one * uv.x * uv.x),
				//(-uv.y * d_x.y) + (dL_dA_msq * dL_dA_partial * r_squared_pow_kappa_minus_one * uv.y * uv.y)
				(-uv.x * d_x.x) + (dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha) * r_squared_pow_kappa_minus_one * uv.x * uv.x), // Depth and normal regularization
				(-uv.y * d_x.y) + (dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha) * r_squared_pow_kappa_minus_one * uv.y * uv.y)  // Depth and normal regularization
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
			uv.x *= r_squared_pow_kappa_minus_one * dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha); // Depth and normal regularization
			uv.y *= r_squared_pow_kappa_minus_one * dL_dA_msq * (dL_dA_partial + dL_dalpha + dLn_dalpha); // Depth and normal regularization

			d_x.x -= uv.x;
			d_x.y -= uv.y;

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

void CPyOptiXFLARERenderer_CUDA::Backward(
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
	float *dL_dw3,
	float *dL_db3,
	float *dL_dw2,
	float *dL_db2,
	float *dL_dw1,
	float *dL_db1,
	float *dL_d_conditioning,
	float *dL_d_deatures,
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

	const int size =
		(sizeof(half) * 64 * 128) +
		(sizeof(float) * 64) +
		(sizeof(half) * 64 * 64) +
		(sizeof(float) * 64) + 
		(sizeof(half) * 16 * 64) +
		(sizeof(float) * 16) +
		(sizeof(float) * (((16 * 16) + (25 * 25) + (40 * 40) + (64 * 64)) * 2));

	error_CUDA = cudaFuncSetAttribute(Backward_CUDA_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, size);
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

		Backward_CUDA_kernel<<<min((number_of_rays + (threads_per_block - 1)) / threads_per_block, SM_count), threads_per_block, size, stream>>>(
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
			t_hit, indices_hit, t_min, T, is_active,

			number_of_rays,

			bg_color,

			m, s, q, RGBA, kappa,

			I, dL_dI, dL_dRGB, dL_dA, dL_d_kappa,
			dL_dw3, dL_db3,
			dL_dw2, dL_db2,
			dL_dw1, dL_db1,
			dL_d_conditioning,
			dL_d_deatures,
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

		//return;

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
