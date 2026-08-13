#pragma once

// *** *** *** *** ***

#include "Header.cuh"

// *** *** *** *** ***

class CPyOptiXFLARERenderer_CUDA {
	public:
		CPyOptiXFLARERenderer_CUDA(
			int number_of_sides, float chi_square_squared_radius, int max_batch_size
		);

		void SetGeometry(
			float *m, float *s, float *q, float *opacity, float *kappa,
			int number_of_Gaussians
		);

		void Render(
			float *O, float *v, float *t_min,
			int size,
			float *t, int *indices
		);

		void GetMedianDepth_base(
			float3 *O,
			float3 *v,

			int number_of_rays,

			float3 *m,
			float2 *s,
			float4 *q,
			float4 *RGBA,
			float *kappa,

			float T_threshold,

			float2 *depth_and_index
		);

		void Forward_training_base(
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
		);

		void Forward_inference_base(
			float3 *O,
			float3 *v,
			float3 *bitmap,
			float3 *normal,
			float *alpha,
			float *expected_depth_numerator,

			int number_of_rays,

			float3 bg_color,

			float3 *m,
			float2 *s,
			float4 *q,
			float4 *RGBA,
			float *kappa,

			float T_threshold
		);

		void Backward_base(
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
		);

		void GetMedianDepth(
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

			float3 *m,
			float2 *s,
			float4 *q,
			float4 *RGBA,
			float *kappa,

			float T_threshold,

			float2 *depth_and_index
		);

		void Forward_training(
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
		);

		void Forward_inference(
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
			float3 *normal,
			float *alpha,
			float *expected_depth_numerator,

			int number_of_rays,

			float3 bg_color,

			float3 *m,
			float2 *s,
			float4 *q,
			float4 *RGBA,
			float *kappa,

			float T_threshold,
			float3 *override_colors
		);

		void Backward(
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
		);

	private:
		float chi_square_squared_radius;

		OptixDeviceContext optixContext;
		OptixModule module;
		OptixProgramGroup raygenPG_test;
		OptixProgramGroup raygenPG;
		OptixProgramGroup missPG;
		OptixProgramGroup hitgroupPG;
		OptixPipeline pipeline;
		OptixShaderBindingTable *sbt_test;
		OptixShaderBindingTable *sbt;

		void *raygenRecordsBuffer_test;
		void *raygenRecordsBuffer;
		void *missRecordsBuffer;
		void *hitgroupRecordsBuffer;

		float3 *Gaussian_as_polygon_vertices;
		int3 *Gaussian_as_polygon_indices;
		OptixTraversableHandle GAS;

		unsigned long long *compactedSizeBuffer;
		unsigned long long tempBufferSize;
		void *tempBuffer;
		unsigned outputBufferSize;
		void *outputBuffer;
		void *GASBuffer;
		unsigned long long instancesBufferSize;
		void *instancesBuffer;

		OptixTraversableHandle IAS;

		unsigned long long IASBufferSize;
		void *IASBuffer;
		void *launchParamsBuffer;

		// *** *** *** *** ***

		int SM_count;

		// *** *** *** *** ***

		cudaStream_t stream;

		float *t_min;
		float *T;
		int *ray_indices[2];
		float *t_hit;
		int *indices_hit;
		int *is_active;
};
