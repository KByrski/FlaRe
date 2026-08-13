#pragma once

// *** *** *** *** ***

#include "CPyOptiXFLAREVIEWERMeshInstance.h"
#include "CPyOptiXFLAREVIEWERRenderer_CUDA.h"
#include "Header.cuh"

#include <torch/extension.h>

// *** *** *** *** ***

class CPyOptiXFLAREVIEWERRenderer {
	public:
		CPyOptiXFLAREVIEWERRenderer(
			int number_of_sides, float chi_square_squared_radius, int max_batch_size, int max_recursion_depth
		);

		void SetGeometry(
			torch::Tensor &m, torch::Tensor &s, torch::Tensor &q, torch::Tensor &opacity, torch::Tensor &kappa,
			torch::Tensor &mesh_instances,
			torch::Tensor &mesh_hitgroup_records
		);

		CPyOptiXFLAREVIEWERMeshInstance CreateMeshInstance(
			torch::Tensor &vertices,
			torch::Tensor &indices,
			torch::Tensor &normals,
			unsigned sbtOffset
		);

		void Forward(
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
		);

	private:
		CPyOptiXFLAREVIEWERRenderer_CUDA renderer;
};