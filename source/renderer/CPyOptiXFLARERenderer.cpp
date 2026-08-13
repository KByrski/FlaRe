#ifdef _WIN32
	#define NOMINMAX
#endif

#include "constants.h"
#include "CPyOptiXFLARERenderer.h"

// *** *** *** *** ***

CPyOptiXFLARERenderer::CPyOptiXFLARERenderer(
	int number_of_sides, float chi_square_squared_radius, int max_batch_size
) :
	renderer(number_of_sides, chi_square_squared_radius, max_batch_size)
{
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer::SetGeometry(torch::Tensor &m, torch::Tensor &s, torch::Tensor &q, torch::Tensor &opacity, torch::Tensor &kappa) {
	float *m_ptr = (float *)m.data_ptr();
	float *s_ptr = (float *)s.data_ptr();
	float *q_ptr = (float *)q.data_ptr();
	float *opacity_ptr = (float *)opacity.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();
	c10::IntArrayRef sizes = m.sizes();
	renderer.SetGeometry(
		m_ptr, s_ptr, q_ptr, opacity_ptr, kappa_ptr,
		sizes[0]
	);
}

// *** *** *** *** ***

std::tuple<torch::Tensor, torch::Tensor> CPyOptiXFLARERenderer::Render(
	torch::Tensor &O, torch::Tensor &v, torch::Tensor &t_min,
	int size
) {
	float *O_ptr = (float *)O.data_ptr();
	float *v_ptr = (float *)v.data_ptr();
	float *t_min_ptr = (float *)t_min.data_ptr();

	const int64_t size1[] = {HIT_BUFFER_SIZE, size};
	torch::TensorOptions options;

	options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	torch::Tensor t = torch::empty(size1, options);

	options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	torch::Tensor indices = torch::empty(size1, options);

	float *t_ptr = (float *)t.data_ptr();
	int *indices_ptr = (int *)indices.data_ptr();

	renderer.Render(
		O_ptr, v_ptr, t_min_ptr,
		size,
		t_ptr, indices_ptr
	);

	return std::make_tuple(t, indices);
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer::GetMedianDepth_base(
	torch::Tensor &O,
	torch::Tensor &v,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold,

	torch::Tensor &depth_and_index
) {
	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	float2 *depth_and_index_ptr = (float2 *)depth_and_index.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.GetMedianDepth_base(
		O_ptr,
		v_ptr,

		size1[0], // number of rays

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		T_threshold,

		depth_and_index_ptr
	);
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer::Forward_training_base(
	torch::Tensor &O,
	torch::Tensor &v,
	torch::Tensor &bitmap,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold,

	torch::Tensor &depth_reg_accums,
	float reg_a,
	float reg_b,

	torch::Tensor &depth_and_index,
	torch::Tensor &surface_normal,
	torch::Tensor &normal_reg_accums
) {
	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();
	float3 *bitmap_ptr = (float3 *)bitmap.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	float4 *depth_reg_accums_ptr = (float4 *)depth_reg_accums.data_ptr();

	float2 *depth_and_index_ptr = (float2 *)depth_and_index.data_ptr();
	float3 *surface_normal_ptr = (float3 *)surface_normal.data_ptr();
	float4 *normal_reg_accums_ptr = (float4 *)normal_reg_accums.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.Forward_training_base(
		O_ptr,
		v_ptr,
		bitmap_ptr,

		size1[0], // number of rays

		make_float3(bg_color_R, bg_color_G, bg_color_B),

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		T_threshold,

		depth_reg_accums_ptr,
		reg_a,
		reg_b,

		depth_and_index_ptr,
		surface_normal_ptr,
		normal_reg_accums_ptr
	);
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer::Forward_inference_base(
	torch::Tensor &O,
	torch::Tensor &v,
	torch::Tensor &bitmap,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold
) {
	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();
	float3 *bitmap_ptr = (float3 *)bitmap.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.Forward_inference_base(
		O_ptr,
		v_ptr,
		bitmap_ptr,
		nullptr,
		nullptr,
		nullptr,

		size1[0], // number of rays

		make_float3(bg_color_R, bg_color_G, bg_color_B),

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		T_threshold
	);
}

void CPyOptiXFLARERenderer::Forward_inference_base_with_geometry(
	torch::Tensor &O,
	torch::Tensor &v,
	torch::Tensor &bitmap,
	torch::Tensor &normal,
	torch::Tensor &alpha,
	torch::Tensor &expected_depth_numerator,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold
) {
	c10::IntArrayRef size1 = O.sizes();
	renderer.Forward_inference_base(
		(float3 *)O.data_ptr(),
		(float3 *)v.data_ptr(),
		(float3 *)bitmap.data_ptr(),
		(float3 *)normal.data_ptr(),
		(float *)alpha.data_ptr(),
		(float *)expected_depth_numerator.data_ptr(),

		size1[0],
		make_float3(bg_color_R, bg_color_G, bg_color_B),

		(float3 *)m.data_ptr(),
		(float2 *)s.data_ptr(),
		(float4 *)q.data_ptr(),
		(float4 *)RGBA.data_ptr(),
		(float *)kappa.data_ptr(),

		T_threshold
	);
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer::Backward_base(
	torch::Tensor &O,
	torch::Tensor &v,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	torch::Tensor &I,
	torch::Tensor &dL_dI,
	torch::Tensor &dL_dRGB,
	torch::Tensor &dL_dA,
	torch::Tensor &dL_d_kappa,
	torch::Tensor &dL_dm,
	torch::Tensor &dL_ds,
	torch::Tensor &dL_dq,

	float T_threshold,

	torch::Tensor &depth_reg_accums,
	torch::Tensor &depth_normal_reg_prefix_sums,
	float lambda_depth,
	float reg_a,
	float reg_b,

	torch::Tensor &depth_and_index,
	torch::Tensor &surface_normal,
	torch::Tensor &normal_reg_accums,
	float lambda_normal
) {
	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	float3 *I_ptr = (float3 *)I.data_ptr();
	float3 *dL_dI_ptr = (float3 *)dL_dI.data_ptr();
	float *dL_dRGB_ptr = (float *)dL_dRGB.data_ptr();
	float *dL_dA_ptr = (float *)dL_dA.data_ptr();
	float *dL_d_kappa_ptr = (float *)dL_d_kappa.data_ptr();
	float *dL_dm_ptr = (float *)dL_dm.data_ptr();
	float *dL_ds_ptr = (float *)dL_ds.data_ptr();
	float *dL_dq_ptr = (float *)dL_dq.data_ptr();

	float4 *depth_reg_accums_ptr = (float4 *)depth_reg_accums.data_ptr();
	float4 *depth_normal_reg_prefix_sums_ptr = (float4 *)depth_normal_reg_prefix_sums.data_ptr();

	float2 *depth_and_index_ptr = (float2 *)depth_and_index.data_ptr();
	float3 *surface_normal_ptr = (float3 *)surface_normal.data_ptr();
	float4 *normal_reg_accums_ptr = (float4 *)normal_reg_accums.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.Backward_base(
		O_ptr,
		v_ptr,

		size1[0], // number of rays

		make_float3(bg_color_R, bg_color_G, bg_color_B),

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		I_ptr,
		dL_dI_ptr,
		dL_dRGB_ptr,
		dL_dA_ptr,
		dL_d_kappa_ptr,
		dL_dm_ptr,
		dL_ds_ptr,
		dL_dq_ptr,

		T_threshold,

		depth_reg_accums_ptr,
		depth_normal_reg_prefix_sums_ptr,
		lambda_depth,
		reg_a,
		reg_b,

		depth_and_index_ptr,
		surface_normal_ptr,
		normal_reg_accums_ptr,
		lambda_normal
	);
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer::GetMedianDepth(
	torch::Tensor &conditioning_variable,
	torch::Tensor &features_P_hit_object,

	torch::Tensor &A1,
	torch::Tensor &b1,
	torch::Tensor &A2,
	torch::Tensor &b2,
	torch::Tensor &A3,
	torch::Tensor &b3,

	torch::Tensor &O,
	torch::Tensor &v,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold,

	torch::Tensor &depth_and_index
) {
	half2 *conditioning_variable_ptr = (half2 *)conditioning_variable.data_ptr();
	float *features_P_hit_object_ptr = (float *)features_P_hit_object.data_ptr();

	half *A1_ptr = (half *)A1.data_ptr();
	float *b1_ptr = (float *)b1.data_ptr();
	half *A2_ptr = (half *)A2.data_ptr();
	float *b2_ptr = (float *)b2.data_ptr();
	half *A3_ptr = (half *)A3.data_ptr();
	float *b3_ptr = (float *)b3.data_ptr();

	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	float2 *depth_and_index_ptr = (float2 *)depth_and_index.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.GetMedianDepth(
		conditioning_variable_ptr,
		features_P_hit_object_ptr,

		A1_ptr,
		b1_ptr,
		A2_ptr,
		b2_ptr,
		A3_ptr,
		b3_ptr,

		O_ptr,
		v_ptr,

		size1[0], // number of rays

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		T_threshold,

		depth_and_index_ptr
	);
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer::Forward_training(
	torch::Tensor &conditioning_variable,
	torch::Tensor &features_P_hit_object,

	torch::Tensor &A1,
	torch::Tensor &b1,
	torch::Tensor &A2,
	torch::Tensor &b2,
	torch::Tensor &A3,
	torch::Tensor &b3,

	torch::Tensor &O,
	torch::Tensor &v,
	torch::Tensor &bitmap,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold,

	torch::Tensor &depth_reg_accums,
	float reg_a,
	float reg_b,

	torch::Tensor &depth_and_index,
	torch::Tensor &surface_normal,
	torch::Tensor &normal_reg_accums
) {
	half2 *conditioning_variable_ptr = (half2 *)conditioning_variable.data_ptr();
	float *features_P_hit_object_ptr = (float *)features_P_hit_object.data_ptr();

	half *A1_ptr = (half *)A1.data_ptr();
	float *b1_ptr = (float *)b1.data_ptr();
	half *A2_ptr = (half *)A2.data_ptr();
	float *b2_ptr = (float *)b2.data_ptr();
	half *A3_ptr = (half *)A3.data_ptr();
	float *b3_ptr = (float *)b3.data_ptr();

	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();
	float3 *bitmap_ptr = (float3 *)bitmap.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	float4 *depth_reg_accums_ptr = (float4 *)depth_reg_accums.data_ptr();

	float2 *depth_and_index_ptr = (float2 *)depth_and_index.data_ptr();
	float3 *surface_normal_ptr = (float3 *)surface_normal.data_ptr();
	float4 *normal_reg_accums_ptr = (float4 *)normal_reg_accums.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.Forward_training(
		conditioning_variable_ptr,
		features_P_hit_object_ptr,

		A1_ptr,
		b1_ptr,
		A2_ptr,
		b2_ptr,
		A3_ptr,
		b3_ptr,

		O_ptr,
		v_ptr,
		bitmap_ptr,

		size1[0], // number of rays

		make_float3(bg_color_R, bg_color_G, bg_color_B),

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		T_threshold,

		depth_reg_accums_ptr,
		reg_a,
		reg_b,

		depth_and_index_ptr,
		surface_normal_ptr,
		normal_reg_accums_ptr
	);
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer::Forward_inference(
	torch::Tensor &conditioning_variable,
	torch::Tensor &features_P_hit_object,

	torch::Tensor &A1,
	torch::Tensor &b1,
	torch::Tensor &A2,
	torch::Tensor &b2,
	torch::Tensor &A3,
	torch::Tensor &b3,

	torch::Tensor &O,
	torch::Tensor &v,
	torch::Tensor &bitmap,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold
) {
	half2 *conditioning_variable_ptr = (half2 *)conditioning_variable.data_ptr();
	float *features_P_hit_object_ptr = (float *)features_P_hit_object.data_ptr();

	half *A1_ptr = (half *)A1.data_ptr();
	float *b1_ptr = (float *)b1.data_ptr();
	half *A2_ptr = (half *)A2.data_ptr();
	float *b2_ptr = (float *)b2.data_ptr();
	half *A3_ptr = (half *)A3.data_ptr();
	float *b3_ptr = (float *)b3.data_ptr();

	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();
	float3 *bitmap_ptr = (float3 *)bitmap.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.Forward_inference(
		conditioning_variable_ptr,
		features_P_hit_object_ptr,

		A1_ptr,
		b1_ptr,
		A2_ptr,
		b2_ptr,
		A3_ptr,
		b3_ptr,

		O_ptr,
		v_ptr,
		bitmap_ptr,
		nullptr,
		nullptr,
		nullptr,

		size1[0], // number of rays

		make_float3(bg_color_R, bg_color_G, bg_color_B),

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		T_threshold,
		nullptr
	);
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer::Forward_pca(
	torch::Tensor &conditioning_variable,
	torch::Tensor &features_P_hit_object,
	torch::Tensor &A1, torch::Tensor &b1, torch::Tensor &A2, torch::Tensor &b2,
	torch::Tensor &A3, torch::Tensor &b3, torch::Tensor &override_colors,
	torch::Tensor &O, torch::Tensor &v, torch::Tensor &bitmap,
	float bg_color_R, float bg_color_G, float bg_color_B,
	torch::Tensor &m, torch::Tensor &s, torch::Tensor &q,
	torch::Tensor &RGBA, torch::Tensor &kappa, float T_threshold
) {
	renderer.Forward_inference(
		(half2 *)conditioning_variable.data_ptr(), (float *)features_P_hit_object.data_ptr(),
		(half *)A1.data_ptr(), (float *)b1.data_ptr(),
		(half *)A2.data_ptr(), (float *)b2.data_ptr(),
		(half *)A3.data_ptr(), (float *)b3.data_ptr(),
		(float3 *)O.data_ptr(), (float3 *)v.data_ptr(), (float3 *)bitmap.data_ptr(),
		nullptr, nullptr, nullptr, O.sizes()[0],
		make_float3(bg_color_R, bg_color_G, bg_color_B),
		(float3 *)m.data_ptr(), (float2 *)s.data_ptr(), (float4 *)q.data_ptr(),
		(float4 *)RGBA.data_ptr(), (float *)kappa.data_ptr(), T_threshold,
		(float3 *)override_colors.data_ptr()
	);
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer::Forward_inference_with_geometry(
	torch::Tensor &conditioning_variable,
	torch::Tensor &features_P_hit_object,

	torch::Tensor &A1,
	torch::Tensor &b1,
	torch::Tensor &A2,
	torch::Tensor &b2,
	torch::Tensor &A3,
	torch::Tensor &b3,

	torch::Tensor &O,
	torch::Tensor &v,
	torch::Tensor &bitmap,
	torch::Tensor &normal,
	torch::Tensor &alpha,
	torch::Tensor &expected_depth_numerator,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold
) {
	c10::IntArrayRef size1 = O.sizes();
	renderer.Forward_inference(
		(half2 *)conditioning_variable.data_ptr(),
		(float *)features_P_hit_object.data_ptr(),

		(half *)A1.data_ptr(),
		(float *)b1.data_ptr(),
		(half *)A2.data_ptr(),
		(float *)b2.data_ptr(),
		(half *)A3.data_ptr(),
		(float *)b3.data_ptr(),

		(float3 *)O.data_ptr(),
		(float3 *)v.data_ptr(),
		(float3 *)bitmap.data_ptr(),
		(float3 *)normal.data_ptr(),
		(float *)alpha.data_ptr(),
		(float *)expected_depth_numerator.data_ptr(),

		size1[0],
		make_float3(bg_color_R, bg_color_G, bg_color_B),

		(float3 *)m.data_ptr(),
		(float2 *)s.data_ptr(),
		(float4 *)q.data_ptr(),
		(float4 *)RGBA.data_ptr(),
		(float *)kappa.data_ptr(),

		T_threshold,
		nullptr
	);
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer::Backward(
	torch::Tensor &conditioning_variable,
	torch::Tensor &features_P_hit_object,

	torch::Tensor &A1,
	torch::Tensor &b1,
	torch::Tensor &A2,
	torch::Tensor &b2,
	torch::Tensor &A3,
	torch::Tensor &b3,

	torch::Tensor &O,
	torch::Tensor &v,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	torch::Tensor &I,
	torch::Tensor &dL_dI,
	torch::Tensor &dL_dRGB,
	torch::Tensor &dL_dA,
	torch::Tensor &dL_d_kappa,
	torch::Tensor &dL_dw3,
	torch::Tensor &dL_db3,
	torch::Tensor &dL_dw2,
	torch::Tensor &dL_db2,
	torch::Tensor &dL_dw1,
	torch::Tensor &dL_db1,
	torch::Tensor &dL_d_conditioning,
	torch::Tensor &dL_d_deatures,
	torch::Tensor &dL_dm,
	torch::Tensor &dL_ds,
	torch::Tensor &dL_dq,

	float T_threshold,

	torch::Tensor &depth_reg_accums,
	torch::Tensor &depth_normal_reg_prefix_sums,
	float lambda_depth,
	float reg_a,
	float reg_b,

	torch::Tensor &depth_and_index,
	torch::Tensor &surface_normal,
	torch::Tensor &normal_reg_accums,
	float lambda_normal
) {
	half2 *conditioning_variable_ptr = (half2 *)conditioning_variable.data_ptr();
	float *features_P_hit_object_ptr = (float *)features_P_hit_object.data_ptr();

	half *A1_ptr = (half *)A1.data_ptr();
	float *b1_ptr = (float *)b1.data_ptr();
	half *A2_ptr = (half *)A2.data_ptr();
	float *b2_ptr = (float *)b2.data_ptr();
	half *A3_ptr = (half *)A3.data_ptr();
	float *b3_ptr = (float *)b3.data_ptr();

	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	float3 *I_ptr = (float3 *)I.data_ptr();
	float3 *dL_dI_ptr = (float3 *)dL_dI.data_ptr();
	float *dL_dRGB_ptr = (float *)dL_dRGB.data_ptr();
	float *dL_dA_ptr = (float *)dL_dA.data_ptr();
	float *dL_d_kappa_ptr = (float *)dL_d_kappa.data_ptr();
	float *dL_dw3_ptr = (float *)dL_dw3.data_ptr();
	float *dL_db3_ptr = (float *)dL_db3.data_ptr();
	float *dL_dw2_ptr = (float *)dL_dw2.data_ptr();
	float *dL_db2_ptr = (float *)dL_db2.data_ptr();
	float *dL_dw1_ptr = (float *)dL_dw1.data_ptr();
	float *dL_db1_ptr = (float *)dL_db1.data_ptr();
	float *dL_d_conditioning_ptr = (float *)dL_d_conditioning.data_ptr();
	float *dL_d_deatures_ptr = (float *)dL_d_deatures.data_ptr();
	float *dL_dm_ptr = (float *)dL_dm.data_ptr();
	float *dL_ds_ptr = (float *)dL_ds.data_ptr();
	float *dL_dq_ptr = (float *)dL_dq.data_ptr();

	float4 *depth_reg_accums_ptr = (float4 *)depth_reg_accums.data_ptr();
	float4 *depth_normal_reg_prefix_sums_ptr = (float4 *)depth_normal_reg_prefix_sums.data_ptr();

	float2 *depth_and_index_ptr = (float2 *)depth_and_index.data_ptr();
	float3 *surface_normal_ptr = (float3 *)surface_normal.data_ptr();
	float4 *normal_reg_accums_ptr = (float4 *)normal_reg_accums.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.Backward(
		conditioning_variable_ptr,
		features_P_hit_object_ptr,

		A1_ptr,
		b1_ptr,
		A2_ptr,
		b2_ptr,
		A3_ptr,
		b3_ptr,

		O_ptr,
		v_ptr,

		size1[0], // number of rays

		make_float3(bg_color_R, bg_color_G, bg_color_B),

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		I_ptr,
		dL_dI_ptr,
		dL_dRGB_ptr,
		dL_dA_ptr,
		dL_d_kappa_ptr,
		dL_dw3_ptr,
		dL_db3_ptr,
		dL_dw2_ptr,
		dL_db2_ptr,
		dL_dw1_ptr,
		dL_db1_ptr,
		dL_d_conditioning_ptr,
		dL_d_deatures_ptr,
		dL_dm_ptr,
		dL_ds_ptr,
		dL_dq_ptr,

		T_threshold,

		depth_reg_accums_ptr,
		depth_normal_reg_prefix_sums_ptr,
		lambda_depth,
		reg_a,
		reg_b,

		depth_and_index_ptr,
		surface_normal_ptr,
		normal_reg_accums_ptr,
		lambda_normal
	);
}
