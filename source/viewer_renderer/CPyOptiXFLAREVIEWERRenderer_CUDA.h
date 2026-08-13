#pragma once

// *** *** *** *** ***

#include "CPyOptiXFLAREVIEWERMeshInstance_CUDA.h"
#include "Header.cuh"

// *** *** *** *** ***

class CPyOptiXFLAREVIEWERRenderer_CUDA {
	public:
		CPyOptiXFLAREVIEWERRenderer_CUDA(
			int number_of_sides, float chi_square_squared_radius, int max_batch_size, int max_recursion_depth
		);

		void SetGeometry(
			float *m, float *s, float *q, float *opacity, float *kappa,
			int number_of_Gaussians,
			float *mesh_instances,
			int number_of_mesh_instances,
			float *mesh_hitgroup_records,
			int number_of_mesh_hitgroup_records
		);

		CPyOptiXFLAREVIEWERMeshInstance_CUDA CreateMeshInstance(
			float3 *vertices, int number_of_vertices,
			int3 *indices, int number_of_indices,
			float3 *normals,
			unsigned sbtOffset
		);

		void Forward(
			half2 *conditioning_variable,
			float *features_P_hit_object,
			float3 *RGB,

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
			float *A,
			float *kappa,

			float T_threshold,

			int number_of_Gaussians,

			float *materials,
			float *M4N,
			float *lights,
			int number_of_lights,
			float epsilon,
			float shadow_multiplier,
			float3 Ia
		);

	private:
		float chi_square_squared_radius;

		OptixDeviceContext optixContext;
		OptixModule module;
		OptixProgramGroup raygenPG;
		OptixProgramGroup raygenPG_shadows;
		OptixProgramGroup missPG;
		OptixProgramGroup missPG_shadows;
		OptixProgramGroup hitgroupPG;
		OptixProgramGroup hitgroupPG_mesh;
		OptixPipeline pipeline;
		OptixShaderBindingTable *sbt;
		OptixShaderBindingTable *sbt_shadows;

		void *raygenRecordsBuffer;
		void *raygenRecordsBuffer_shadows;
		void *missRecordsBuffer;

		SbtHitgroupRecord rec_hitgroup;
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
		int *ray_indices_initial;
		int *ray_indices[2];
		float *t_hit;
		int *indices_hit;
		int *is_active;

		// *** *** *** *** ***

		float4 *buffer1;
		float4 *buffer2;
		int *is_mesh;

		int max_recursion_depth;
		SStackEntry *stack;
};