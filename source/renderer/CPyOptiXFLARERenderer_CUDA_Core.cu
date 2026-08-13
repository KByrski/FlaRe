#define _USE_MATH_DEFINES

#include <math.h>
#include <stdio.h> // Needed on Linux
#include <string>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "CPyOptiXFLARERenderer_CUDA.h"
#include "Header.cuh"

// !!! !!! !!!
#include "optix_function_table_definition.h" // Included only in one file
// !!! !!! !!!

// *** *** *** *** ***

namespace {

std::string GetShadersPath() {
#ifdef _WIN32
	return "shaders.cu.ptx";
#else
	Dl_info module_info = {};
	if (dladdr(reinterpret_cast<void *>(&GetShadersPath), &module_info) == 0 ||
		module_info.dli_fname == nullptr) {
		return "shaders.cu.ptx";
	}

	std::string module_path(module_info.dli_fname);
	const std::string::size_type separator = module_path.find_last_of('/');
	if (separator == std::string::npos) return "shaders.cu.ptx";
	return module_path.substr(0, separator + 1) + "shaders.cu.ptx";
#endif
}

} // namespace

// *** *** *** *** ***

CPyOptiXFLARERenderer_CUDA::CPyOptiXFLARERenderer_CUDA(
	int number_of_sides, float chi_square_squared_radius, int max_batch_size
) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;
	CUresult error_CUDA_Driver_API;

	// *********************************************************************************************

	error_CUDA = cudaSetDevice(0);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_OptiX = optixInit();
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	CUcontext cudaContext;
	error_CUDA_Driver_API = cuCtxGetCurrent(&cudaContext);
	if (error_CUDA_Driver_API != CUDA_SUCCESS) throw 0;

	error_OptiX = optixDeviceContextCreate(cudaContext, 0, &optixContext);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	const std::string shaders_path = GetShadersPath();
	FILE *f = fopen(shaders_path.c_str(), "rb");
	if (f == nullptr) throw 0;
	fseek(f, 0, SEEK_END);
	int shadersSize = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *shaders = (char *)malloc(sizeof(char) * (shadersSize + 1));
	fread(shaders, 1, shadersSize, f);
	fclose(f);
	shaders[shadersSize] = 0;

	// *********************************************************************************************

	OptixModuleCompileOptions moduleCompileOptions = {};
	OptixPipelineCompileOptions pipelineCompileOptions = {};

	moduleCompileOptions.maxRegisterCount = 40;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 0;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
	pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

	error_OptiX = optixModuleCreate(
		optixContext,
		&moduleCompileOptions,
		&pipelineCompileOptions,
		shaders,
		strlen(shaders),
		NULL, NULL,
		&module
	);

	free(shaders);

	// *********************************************************************************************

	OptixStackSizes oss;
	oss.cssRG = 0;
	oss.cssMS = 0;
	oss.cssCH = 0;
	oss.cssAH = 0;
	oss.cssIS = 0;
	oss.cssCC = 0;
	oss.dssDC = 0;

	// *********************************************************************************************

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc;

	// *********************************************************************************************

	pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = module;           
	pgDesc.raygen.entryFunctionName = "__raygen__test";

	error_OptiX = optixProgramGroupCreate(
		optixContext,
		&pgDesc,
		1,
		&pgOptions,
		NULL, NULL,
		&raygenPG_test
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_OptiX = optixUtilAccumulateStackSizes(raygenPG_test, &oss, NULL);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = module;           
	pgDesc.raygen.entryFunctionName = "__raygen__";

	error_OptiX = optixProgramGroupCreate(
		optixContext,
		&pgDesc,
		1,
		&pgOptions,
		NULL, NULL,
		&raygenPG
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_OptiX = optixUtilAccumulateStackSizes(raygenPG, &oss, NULL);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

	error_OptiX = optixProgramGroupCreate(
		optixContext,
		&pgDesc,
		1, 
		&pgOptions,
		NULL, NULL,
		&missPG
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_OptiX = optixUtilAccumulateStackSizes(missPG, &oss, NULL);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc.hitgroup.moduleAH            = module;
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__";

	error_OptiX = optixProgramGroupCreate(
		optixContext,
		&pgDesc,
		1, 
		&pgOptions,
		NULL, NULL,
		&hitgroupPG
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_OptiX = optixUtilAccumulateStackSizes(hitgroupPG, &oss, NULL);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	OptixPipelineLinkOptions pipelineLinkOptions = {};
	pipelineLinkOptions.maxTraceDepth = 1;

	OptixProgramGroup program_groups[] = { raygenPG_test, raygenPG, missPG, hitgroupPG };

	error_OptiX = optixPipelineCreate(
		optixContext,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		program_groups,
		4,
		NULL, NULL,
		&pipeline
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	unsigned int directCallableStackSizeFromTraversal;
	unsigned int directCallableStackSizeFromState;
	unsigned int continuationStackSize;

	error_OptiX = optixUtilComputeStackSizes(
		&oss,
		1,
		0,
		0,
		&directCallableStackSizeFromTraversal,
		&directCallableStackSizeFromState,
		&continuationStackSize 
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_OptiX = optixPipelineSetStackSize(
		pipeline, 
		directCallableStackSizeFromTraversal,
		directCallableStackSizeFromState,
		continuationStackSize,
		2
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	SbtRecord rec;

	// *********************************************************************************************

	sbt_test = new OptixShaderBindingTable();
	sbt = new OptixShaderBindingTable();

	// *********************************************************************************************

	error_OptiX = optixSbtRecordPackHeader(raygenPG_test, &rec);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaMalloc(&raygenRecordsBuffer_test, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMemcpy(raygenRecordsBuffer_test, &rec, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) throw 0;

	sbt_test->raygenRecord = (CUdeviceptr)raygenRecordsBuffer_test;

	// *********************************************************************************************

	error_OptiX = optixSbtRecordPackHeader(raygenPG, &rec);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaMalloc(&raygenRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMemcpy(raygenRecordsBuffer, &rec, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) throw 0;

	sbt->raygenRecord = (CUdeviceptr)raygenRecordsBuffer;

	// *********************************************************************************************

	error_OptiX = optixSbtRecordPackHeader(missPG, &rec);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaMalloc(&missRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMemcpy(missRecordsBuffer, &rec, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) throw 0;

	sbt_test->missRecordBase = (CUdeviceptr)missRecordsBuffer;
	sbt_test->missRecordStrideInBytes = sizeof(SbtRecord);
	sbt_test->missRecordCount = 1;

	sbt->missRecordBase = (CUdeviceptr)missRecordsBuffer;
	sbt->missRecordStrideInBytes = sizeof(SbtRecord);
	sbt->missRecordCount = 1;

	// *********************************************************************************************

	error_OptiX = optixSbtRecordPackHeader(hitgroupPG, &rec);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaMalloc(&hitgroupRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMemcpy(hitgroupRecordsBuffer, &rec, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) throw 0;

	sbt_test->hitgroupRecordBase          = (CUdeviceptr)hitgroupRecordsBuffer;
	sbt_test->hitgroupRecordStrideInBytes = sizeof(SbtRecord);
	sbt_test->hitgroupRecordCount         = 1;

	sbt->hitgroupRecordBase          = (CUdeviceptr)hitgroupRecordsBuffer;
	sbt->hitgroupRecordStrideInBytes = sizeof(SbtRecord);
	sbt->hitgroupRecordCount         = 1;

	// *********************************************************************************************

	// Paper: "Mesh representation of a 2D Gaussian primitive". Build one
	// canonical regular polygon; every primitive reuses it through an OptiX instance.
	float3 *Gaussian_as_polygon_vertices_host = (float3 *)malloc(sizeof(float3) * 1 * number_of_sides);
	int3 *Gaussian_as_polygon_indices_host = (int3 *)malloc(sizeof(int3) * 1 * (number_of_sides - 2));

	for (int i = 0; i < number_of_sides; ++i)
		/*Gaussian_as_polygon_vertices_host[i] = make_float3(
			cosf(i * ((2.0f * M_PI) / number_of_sides)) * sqrtf(chi_square_squared_radius),
			sinf(i * ((2.0f * M_PI) / number_of_sides)) * sqrtf(chi_square_squared_radius),
			0.0f
		);*/
		Gaussian_as_polygon_vertices_host[i] = make_float3(
			cosf(i * ((2.0f * M_PI) / number_of_sides)),
			sinf(i * ((2.0f * M_PI) / number_of_sides)),
			0.0f
		);
	for (int i = 0; i < number_of_sides - 2; ++i)
		Gaussian_as_polygon_indices_host[i] = make_int3(0, i + 1, i + 2);

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&Gaussian_as_polygon_vertices, sizeof(float3) * 1 * number_of_sides);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMemcpy(Gaussian_as_polygon_vertices, Gaussian_as_polygon_vertices_host, sizeof(float3) * 1 * number_of_sides, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMalloc(&Gaussian_as_polygon_indices, sizeof(int3) * 1 * (number_of_sides - 2));
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMemcpy(Gaussian_as_polygon_indices, Gaussian_as_polygon_indices_host, sizeof(int3) * 1 * (number_of_sides - 2), cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	free(Gaussian_as_polygon_vertices_host);
	free(Gaussian_as_polygon_indices_host);

	// *********************************************************************************************

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	// *********************************************************************************************

	OptixBuildInput mesh_input = {};
	mesh_input.type                           = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	mesh_input.triangleArray.vertexBuffers    = (CUdeviceptr *)&Gaussian_as_polygon_vertices;
	mesh_input.triangleArray.numVertices      = 1 * number_of_sides;
	mesh_input.triangleArray.vertexFormat     = OPTIX_VERTEX_FORMAT_FLOAT3;
	mesh_input.triangleArray.indexBuffer      = (CUdeviceptr)Gaussian_as_polygon_indices;
	mesh_input.triangleArray.numIndexTriplets = 1 * (number_of_sides - 2);
	mesh_input.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

	int mesh_input_flags[1]                = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
	mesh_input.triangleArray.flags         = ((const unsigned int *)mesh_input_flags);
	mesh_input.triangleArray.numSbtRecords = 1;

	// *********************************************************************************************

	OptixAccelBufferSizes blasBufferSizes;
	error_OptiX = optixAccelComputeMemoryUsage(
		optixContext,
		&accel_options,
		&mesh_input,
		1,
		&blasBufferSizes
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	unsigned long long *compactedSizeBuffer;
	error_CUDA = cudaMalloc(&compactedSizeBuffer, sizeof(unsigned long long) * 1);
	if (error_CUDA != cudaSuccess) throw 0;

	OptixAccelEmitDesc emitDesc;
	emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = (CUdeviceptr)compactedSizeBuffer;

	void *tempBuffer;

	error_CUDA = cudaMalloc(&tempBuffer, blasBufferSizes.tempSizeInBytes);
	if (error_CUDA != cudaSuccess) throw 0;

	void *outputBuffer;

	error_CUDA = cudaMalloc(&outputBuffer, blasBufferSizes.outputSizeInBytes);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_OptiX = optixAccelBuild(
		optixContext,
		0,
		&accel_options,
		&mesh_input,
		1,  
		(CUdeviceptr)tempBuffer,
		blasBufferSizes.tempSizeInBytes,
		(CUdeviceptr)outputBuffer,
		blasBufferSizes.outputSizeInBytes,
		&GAS,
		&emitDesc,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) throw 0;

	unsigned long long compactedSize;

	error_CUDA = cudaMemcpy(&compactedSize, compactedSizeBuffer, sizeof(unsigned long long) * 1, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMalloc(&GASBuffer, compactedSize);
	if (error_CUDA != cudaSuccess) throw 0;

	error_OptiX = optixAccelCompact(
		optixContext,
		0,
		GAS,
		(CUdeviceptr)GASBuffer,
		compactedSize,
		&GAS
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaFree(compactedSizeBuffer);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaFree(tempBuffer);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaFree(outputBuffer);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	this->chi_square_squared_radius = chi_square_squared_radius;
	instancesBuffer = NULL; // !!! !!! !!!
	IASBuffer = NULL; // !!! !!! !!!

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&launchParamsBuffer, sizeof(SLaunchParams) * 1);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	int device;

	error_CUDA = cudaGetDevice(&device);
	if (error_CUDA != cudaSuccess) throw 0;

	cudaDeviceProp prop;

	error_CUDA = cudaGetDeviceProperties(&prop, device);
	if (error_CUDA != cudaSuccess) throw 0;

	SM_count = prop.multiProcessorCount;

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&t_min, sizeof(float) * max_batch_size);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&T, sizeof(float) * max_batch_size);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&ray_indices[0], sizeof(int) * max_batch_size);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMalloc(&ray_indices[1], sizeof(int) * max_batch_size);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&t_hit, sizeof(float) * HIT_BUFFER_SIZE * max_batch_size);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&indices_hit, sizeof(int) * HIT_BUFFER_SIZE * max_batch_size);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&is_active, sizeof(int) * max_batch_size);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_CUDA = cudaStreamCreate(&stream);
	if (error_CUDA != cudaSuccess) throw 0;
}

// *** *** *** *** ***

// Paper: instantiate the shared proxy with each primitive's center, tangent-plane
// scales, quaternion, opacity threshold, and generalized-Gaussian shape.
static __global__ void GenerateInstances(
	float3 *m_ptr, float2 *s_ptr, float4 *q_ptr,
	float *opacities, float *kappas,
	float chi_square_squared_radius,
	int numberOfGaussians,
	OptixTraversableHandle GAS,
	float *instances
) {
	extern __shared__ float tmp[];

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int wid = tid >> 5;
	int number_of_warps = (numberOfGaussians + 31) >> 5;

	// *********************************************************************************************

	if (wid < number_of_warps) {
		int index = ((tid < numberOfGaussians) ? tid : (numberOfGaussians - 1));

		// *****************************************************************************************

		float3 m_param = m_ptr[index];
		float2 s_param = s_ptr[index];
		float4 q_param = q_ptr[index];
		float opacity = opacities[index];
		float kappa = kappas[index];

		// *****************************************************************************************

		float aa = q_param.x * q_param.x;
		float bb = q_param.y * q_param.y;
		float cc = q_param.z * q_param.z;
		float dd = q_param.w * q_param.w;
		float s = 2.0f / (aa + bb + cc + dd);

		float bs = q_param.y * s;  float cs = q_param.z * s;  float ds = q_param.w * s;
		float ab = q_param.x * bs; float ac = q_param.x * cs; float ad = q_param.x * ds;
		bb = bb * s;			   float bc = q_param.y * cs; float bd = q_param.y * ds;
		cc = cc * s;			   float cd = q_param.z * ds;       dd = dd * s;
	
		// !!! !!! !!!
		/*float scale_squared = 1.0f + ((2.0f * logf(opacity)) / chi_square_squared_radius);
		scale_squared = (scale_squared < 0.0f) ? 0.0f : scale_squared;

		float scale = sqrtf(scale_squared);

		s_param.x *= scale;
		s_param.y *= scale;*/
		// !!! !!! !!!

		// !!! !!! !!!
		float scale_power = kappa * (11.3449f + (2.0f * logf(opacity)));
		scale_power = (scale_power < 0.0f) ? 0.0f : scale_power;

		float scale = powf(scale_power, 1.0f / (2.0f * kappa));

		s_param.x *= scale;
		s_param.y *= scale;
		// !!! !!! !!!

		float Q11 = s_param.x * (1.0f - cc - dd);
		float Q12 = s_param.y * (bc - ad);
		float Q13 = bd + ac;

		float Q21 = s_param.x * (bc + ad);
		float Q22 = s_param.y * (1.0f - bb - dd);
		float Q23 = cd - ab;

		float Q31 = s_param.x * (bd - ac);
		float Q32 = s_param.y * (cd + ab);
		float Q33 = 1.0f - bb - cc;

		// *****************************************************************************************

		float *base_address = &tmp[(threadIdx.x * 20) + (threadIdx.x >> 3)];

		// transform
		base_address[0] = Q11;
		base_address[1] = Q12;
		base_address[2] = Q13;
		base_address[3] = m_param.x;

		base_address[4] = Q21;
		base_address[5] = Q22;
		base_address[6] = Q23;
		base_address[7] = m_param.y;

		base_address[8] = Q31;
		base_address[9] = Q32;
		base_address[10] = Q33;
		base_address[11] = m_param.z;

		// instanceId
		base_address[12] = 0.0f;

		// sbtOffset
		base_address[13] = 0.0f;

		// visibilityMask
		base_address[14] = __uint_as_float(255);

		// flags
		base_address[15] = __uint_as_float(OPTIX_INSTANCE_FLAG_NONE);

		// traversableHandle
		base_address[16] = __uint_as_float(GAS);
		base_address[17] = __uint_as_float(GAS >> 32);

		// pad
		base_address[18] = 0.0f;
		base_address[19] = 0.0f;
	}

	// *********************************************************************************************

	__syncthreads();

	// *********************************************************************************************

	if (wid < number_of_warps) {
		int lane_id = threadIdx.x & 31;

		float *base_address_1 = &instances[(tid & -32) * 20];
		float *base_address_2 = &tmp[((threadIdx.x & -32) * 20) + ((threadIdx.x & -32) >> 3)];

		base_address_1[lane_id      ] = base_address_2[lane_id      ];
		base_address_1[lane_id + 32 ] = base_address_2[lane_id + 32 ];
		base_address_1[lane_id + 64 ] = base_address_2[lane_id + 64 ];
		base_address_1[lane_id + 96 ] = base_address_2[lane_id + 96 ];
		base_address_1[lane_id + 128] = base_address_2[lane_id + 128];

		base_address_1[lane_id + 160] = base_address_2[lane_id + 160 + 1];
		base_address_1[lane_id + 192] = base_address_2[lane_id + 192 + 1];
		base_address_1[lane_id + 224] = base_address_2[lane_id + 224 + 1];
		base_address_1[lane_id + 256] = base_address_2[lane_id + 256 + 1];
		base_address_1[lane_id + 288] = base_address_2[lane_id + 288 + 1];

		base_address_1[lane_id + 320] = base_address_2[lane_id + 320 + 2];
		base_address_1[lane_id + 352] = base_address_2[lane_id + 352 + 2];
		base_address_1[lane_id + 384] = base_address_2[lane_id + 384 + 2];
		base_address_1[lane_id + 416] = base_address_2[lane_id + 416 + 2];
		base_address_1[lane_id + 448] = base_address_2[lane_id + 448 + 2];

		base_address_1[lane_id + 480] = base_address_2[lane_id + 480 + 3];
		base_address_1[lane_id + 512] = base_address_2[lane_id + 512 + 3];
		base_address_1[lane_id + 544] = base_address_2[lane_id + 544 + 3];
		base_address_1[lane_id + 576] = base_address_2[lane_id + 576 + 3];
		base_address_1[lane_id + 608] = base_address_2[lane_id + 608 + 3];
	}
}

// *** *** *** *** ***

// Paper: explicit proxy instances form the hardware-accelerated, directly
// ray-queryable scene representation.
void CPyOptiXFLARERenderer_CUDA::SetGeometry(
	float *m, float *s, float *q, float *opacity, float *kappa,
	int number_of_Gaussians
) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;

	// *********************************************************************************************

	if (instancesBuffer != NULL) {
		error_CUDA = cudaFree(instancesBuffer);
		if (error_CUDA != cudaSuccess) throw 0;
	}
	error_CUDA = cudaMalloc(&instancesBuffer, sizeof(OptixInstance) * ((number_of_Gaussians + 31) & -32)); // !!! !!! !!!
	if (error_CUDA != cudaSuccess) throw 0;

	GenerateInstances<<<(number_of_Gaussians + 63) >> 6, 64, ((20 * 64) + 7) << 2>>>(
		(float3 *)m, (float2 *)s, (float4 *)q,
		opacity, kappa,
		chi_square_squared_radius,
		number_of_Gaussians,
		GAS,
		(float *)instancesBuffer
	);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	// *********************************************************************************************

	OptixBuildInput instances_input = {};
	instances_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instances_input.instanceArray.instances    = (CUdeviceptr)instancesBuffer;
	instances_input.instanceArray.numInstances = number_of_Gaussians;

	// *********************************************************************************************

	OptixAccelBufferSizes blasBufferSizes;
	error_OptiX = optixAccelComputeMemoryUsage(
		optixContext,
		&accel_options,
		&instances_input,
		1,
		&blasBufferSizes
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	unsigned long long *compactedSizeBuffer;
	error_CUDA = cudaMalloc(&compactedSizeBuffer, sizeof(unsigned long long) * 1);
	if (error_CUDA != cudaSuccess) throw 0;

	OptixAccelEmitDesc emitDesc;
	emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = (CUdeviceptr)compactedSizeBuffer;

	void *tempBuffer;

	error_CUDA = cudaMalloc(&tempBuffer, blasBufferSizes.tempSizeInBytes);
	if (error_CUDA != cudaSuccess) throw 0;

	void *outputBuffer;

	error_CUDA = cudaMalloc(&outputBuffer, blasBufferSizes.outputSizeInBytes);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_OptiX = optixAccelBuild(
		optixContext,
		0,
		&accel_options,
		&instances_input,
		1,  
		(CUdeviceptr)tempBuffer,
		blasBufferSizes.tempSizeInBytes,
		(CUdeviceptr)outputBuffer,
		blasBufferSizes.outputSizeInBytes,
		&IAS,
		&emitDesc,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) throw 0;

	unsigned long long compactedSize;

	error_CUDA = cudaMemcpy(&compactedSize, compactedSizeBuffer, sizeof(unsigned long long) * 1, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) throw 0;

	if (IASBuffer != NULL) {
		error_CUDA = cudaFree(IASBuffer);
		if (error_CUDA != cudaSuccess) throw 0;
	}
	error_CUDA = cudaMalloc(&IASBuffer, compactedSize);
	if (error_CUDA != cudaSuccess) throw 0;

	error_OptiX = optixAccelCompact(
		optixContext,
		0,
		IAS,
		(CUdeviceptr)IASBuffer,
		compactedSize,
		&IAS
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaFree(compactedSizeBuffer);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaFree(tempBuffer);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaFree(outputBuffer);
	if (error_CUDA != cudaSuccess) throw 0;
}

// *** *** *** *** ***

void CPyOptiXFLARERenderer_CUDA::Render(
	float *O, float *v, float *t_min,
	int size,
	float *t, int *indices
) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;

	// *********************************************************************************************

	SLaunchParams launchParams;

	launchParams.O = (float3 *)O;
	launchParams.v = (float3 *)v;
	launchParams.t_min = t_min;
	launchParams.AS = IAS;
	launchParams.t = t;
	launchParams.indices = indices;

	error_CUDA = cudaMemcpy(launchParamsBuffer, &launchParams, sizeof(SLaunchParams) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_OptiX = optixLaunch(
		pipeline,
		0,
		(CUdeviceptr)launchParamsBuffer,
		sizeof(SLaunchParams) * 1,
		sbt_test,
		size,
		1,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) throw 0;
}