#include "GenerateRays.h"

// *** *** *** *** ***

torch::Tensor GenerateRays(
	torch::Tensor &R, torch::Tensor &D, torch::Tensor &F,
	int width, int height,
	float fov_X, float fov_Y
) {
	float *R_ptr = (float *)R.data_ptr();
	float *D_ptr = (float *)D.data_ptr();
	float *F_ptr = (float *)F.data_ptr();

	const int64_t size[] = {height, width, 3};
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	torch::Tensor result = torch::empty(size, options);
	float *result_ptr = (float *)result.data_ptr();

	GenerateRays_CUDA(
		R_ptr, D_ptr, F_ptr,
		width, height,
		fov_X, fov_Y,
		result_ptr
	);

	return result;
}