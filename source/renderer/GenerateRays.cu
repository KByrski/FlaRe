#include "GenerateRays.h"
#include "Header.cuh"

// *** *** *** *** ***

__global__ void GenerateRays_CUDA_kernel(
	float *R_ptr, float *D_ptr, float *F_ptr,
	int width, int height,
	float double_tan_half_fov_X, float double_tan_half_fov_Y,
	float *result_ptr
) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int size = width * height;

	if (tid < size) {
		float3 R = *((float3 *)R_ptr);
		float3 D = *((float3 *)D_ptr);
		float3 F = *((float3 *)F_ptr);

		int x = tid % width;
		int y = tid / width;

		float a_x = double_tan_half_fov_X / width;
		float b_x = 0.5f * double_tan_half_fov_X * ((1.0f / width) - 1.0f);
		float a_y = double_tan_half_fov_Y / height;
		float b_y = 0.5f * double_tan_half_fov_Y * ((1.0f / height) - 1.0f);
		
		float2 d = make_float2(
			__fmaf_rn(a_x, x, b_x),
			__fmaf_rn(a_y, y, b_y)
		);

		float3 v = make_float3(
			__fmaf_rn(R.x, d.x, __fmaf_rn(D.x, d.y, F.x)),
			__fmaf_rn(R.y, d.x, __fmaf_rn(D.y, d.y, F.y)),
			__fmaf_rn(R.z, d.x, __fmaf_rn(D.z, d.y, F.z))
		);

		((float3 *)result_ptr)[tid] = v;
	}
}

// *** *** *** *** ***

void GenerateRays_CUDA(
	float *R_ptr, float *D_ptr, float *F_ptr,
	int width, int height,
	float fov_X, float fov_Y,
	float *result_ptr
) {
	cudaError_t error_CUDA;

	// *********************************************************************************************

	float double_tan_half_fov_X = 2.0f * tanf(0.5f * fov_X);
	float double_tan_half_fov_Y = 2.0f * tanf(0.5f * fov_Y);

	int size = width * height;

	// *********************************************************************************************

	GenerateRays_CUDA_kernel<<<((width * height) + 63) >> 6, 64>>>(
		R_ptr, D_ptr, F_ptr,
		width, height,
		double_tan_half_fov_X, double_tan_half_fov_Y,
		result_ptr
		);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) throw 0;
}