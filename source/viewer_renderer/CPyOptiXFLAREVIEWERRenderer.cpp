#ifdef _WIN32
	#define NOMINMAX
#endif

#include "constants.h"
#include "CPyOptiXFLAREVIEWERMeshInstance.h"
#include "CPyOptiXFLAREVIEWERRenderer.h"

// *** *** *** *** ***

CPyOptiXFLAREVIEWERRenderer::CPyOptiXFLAREVIEWERRenderer(
	int number_of_sides, float chi_square_squared_radius, int max_batch_size, int max_recursion_depth
) :
	renderer(number_of_sides, chi_square_squared_radius, max_batch_size, max_recursion_depth)
{
}

// *** *** *** *** ***

void CPyOptiXFLAREVIEWERRenderer::SetGeometry(
	torch::Tensor &m, torch::Tensor &s, torch::Tensor &q, torch::Tensor &opacity, torch::Tensor &kappa,
	torch::Tensor &mesh_instances,
	torch::Tensor &mesh_hitgroup_records
) {
	float *m_ptr = (float *)m.data_ptr();
	float *s_ptr = (float *)s.data_ptr();
	float *q_ptr = (float *)q.data_ptr();
	float *opacity_ptr = (float *)opacity.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();
	float *mesh_instances_ptr = (float *)mesh_instances.data_ptr();
	float *mesh_hitgroup_records_ptr = (float *)mesh_hitgroup_records.data_ptr();
	c10::IntArrayRef sizes1 = m.sizes();
	c10::IntArrayRef sizes2 = mesh_instances.sizes();
	c10::IntArrayRef sizes3 = mesh_hitgroup_records.sizes();
	renderer.SetGeometry(
		m_ptr, s_ptr, q_ptr, opacity_ptr, kappa_ptr,
		sizes1[0],
		mesh_instances_ptr,
		sizes2[0],
		mesh_hitgroup_records_ptr,
		sizes3[0]
	);
}

// *** *** *** *** ***

CPyOptiXFLAREVIEWERMeshInstance CPyOptiXFLAREVIEWERRenderer::CreateMeshInstance(
	torch::Tensor &vertices,
	torch::Tensor &indices,
	torch::Tensor &normals,
	unsigned sbtOffset
) {
	float3 *vertices_ptr = (float3 *)vertices.data_ptr();
	c10::IntArrayRef vertices_size = vertices.sizes();
	int3 *indices_ptr = (int3 *)indices.data_ptr();
	c10::IntArrayRef indices_size = indices.sizes();
	float3 *normals_ptr = (float3 *)normals.data_ptr();

	CPyOptiXFLAREVIEWERMeshInstance instance;

	instance.vertex_buffer = vertices;
	instance.indices_buffer = indices;
	instance.normals_buffer = normals;

	instance.instance = renderer.CreateMeshInstance(
		vertices_ptr, vertices_size[0],
		indices_ptr, indices_size[0],
		normals_ptr,
		sbtOffset
	);

	return instance;
}

// *** *** *** *** ***

void CPyOptiXFLAREVIEWERRenderer::Forward(
	torch::Tensor &conditioning_variable,
	torch::Tensor &features_P_hit_object,
	torch::Tensor &RGB,

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
	torch::Tensor &A,
	torch::Tensor &kappa,

	float T_threshold,

	torch::Tensor &materials,
	torch::Tensor &M4N,
	torch::Tensor &lights,
	float epsilon,
	float shadow_multiplier,
	float Ia_R, float Ia_G, float Ia_B
) {
	half2 *conditioning_variable_ptr = (half2 *)conditioning_variable.data_ptr();
	float *features_P_hit_object_ptr = (float *)features_P_hit_object.data_ptr();
	float3 *RGB_ptr = (float3 *)RGB.data_ptr();

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
	float *A_ptr = (float *)A.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	float *materials_ptr = (float *)materials.data_ptr();
	float *M4N_ptr = (float *)M4N.data_ptr();
	float *lights_ptr = (float *)lights.data_ptr();

	c10::IntArrayRef size1 = O.sizes();
	c10::IntArrayRef size2 = m.sizes();
	c10::IntArrayRef size3 = lights.sizes();

	renderer.Forward(
		conditioning_variable_ptr,
		features_P_hit_object_ptr,
		RGB_ptr,

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
		A_ptr,
		kappa_ptr,

		T_threshold,

		size2[0], // number_of_Gaussians,

		materials_ptr,
		M4N_ptr,
		lights_ptr,
		size3[0], // number_of_lights
		epsilon,
		shadow_multiplier,
		make_float3(Ia_R, Ia_G, Ia_B)
	);
}