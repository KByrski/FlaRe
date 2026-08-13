#pragma once

// *** *** *** *** ***

#include "CPyOptiXFLARERenderer_CUDA.h"
#include "Header.cuh"

#include <torch/extension.h>

// *** *** *** *** ***

class CPyOptiXFLARERenderer {
	public:
		CPyOptiXFLARERenderer(
			int number_of_sides, float chi_square_squared_radius, int max_batch_size
		);

		void SetGeometry(torch::Tensor &m, torch::Tensor &s, torch::Tensor &q, torch::Tensor &opacity, torch::Tensor &kappa);

		// For testing model changes during development
		std::tuple<torch::Tensor, torch::Tensor> Render(
			torch::Tensor &O, torch::Tensor &v, torch::Tensor &t_min,
			int size
		);

		void GetMedianDepth_base(
			torch::Tensor &O,
			torch::Tensor &v,

			torch::Tensor &m,
			torch::Tensor &s,
			torch::Tensor &q,
			torch::Tensor &RGBA,
			torch::Tensor &kappa,

			float T_threshold,

			torch::Tensor &depth_and_index
		);

		void Forward_training_base(
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
		);

		void Forward_inference_base(
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
		);

		void Forward_inference_base_with_geometry(
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
		);

		void Backward_base(
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
		);

		void GetMedianDepth(
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
		);

		void Forward_training(
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
		);

		void Forward_inference(
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
		);

		void Forward_pca(
			torch::Tensor &conditioning_variable,
			torch::Tensor &features_P_hit_object,

			torch::Tensor &A1,
			torch::Tensor &b1,
			torch::Tensor &A2,
			torch::Tensor &b2,
			torch::Tensor &A3,
			torch::Tensor &b3,
			torch::Tensor &override_colors,

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
		);

		void Forward_inference_with_geometry(
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
		);

		void Backward(
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
		);

	private:
		CPyOptiXFLARERenderer_CUDA renderer;
};
