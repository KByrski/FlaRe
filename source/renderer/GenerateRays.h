#pragma once

// *** *** *** *** ***

#ifndef __CUDACC__
	#include <torch/extension.h>
#endif

// *** *** *** *** ***

void GenerateRays_CUDA(
	float *R_ptr, float *D_ptr, float *F_ptr,
	int width, int height,
	float fov_X, float fov_Y,
	float *result_ptr
);

// *** *** *** *** ***

#ifndef __CUDACC__
	torch::Tensor GenerateRays(
		torch::Tensor &R, torch::Tensor &D, torch::Tensor &F,
		int width, int height,
		float fov_X, float fov_Y
	);
#endif