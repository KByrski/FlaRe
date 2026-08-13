#pragma once

#include "constants.h"

#include "cuda_fp16.h"
#include "cuda_bf16.h"

#include "optix_host.h"
#include "optix_stack_size.h"
#include "optix_stubs.h"

// *************************************************************************************************

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
	char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// *************************************************************************************************

struct SbtHitgroupRecordData {
	float3 *vertex_buffer;
	int3 *indices_buffer;
	float3 *normals_buffer;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtHitgroupRecord {
	char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	SbtHitgroupRecordData data;
};

// *************************************************************************************************

struct SStackEntry {
	int *ray_indices_initial;
	float3 *O;
	float3 *v;
	float *T;
	float *t_min;
	int *is_active;

	int number_of_rays_initial;
	int number_of_rays;
	bool processed;
};

// *************************************************************************************************

struct SLaunchParams {
	float3 *O;
	float3 *v;
	float *t_min;
	float *t;
	int *indices;
	OptixTraversableHandle AS;
	int *ray_indices;

	int number_of_Gaussians;

	float3 *bitmap;
	int *ray_indices_initial;
	int *is_mesh;
	float4 *buffer1;
	float4 *buffer2;
	float *materials;
	float *M4N;
	float *T_ptr;
	float T_threshold;
	float3 bg_color;
	int *is_active;
	float *lights;
	int number_of_lights;
	SStackEntry entry;
	float epsilon;

	float3 Ia;
};