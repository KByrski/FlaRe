#include "Header.cuh"
#include "optix_device.h"

// *** *** *** *** ***

extern "C" __constant__ SLaunchParams optixLaunchParams;

// *** *** *** *** ***

// Paper: G(r) is the bounded, ordered set of ray-proxy intersections used
// by the neural renderer for front-to-back compositing.
struct SRayPayload {
	int number_of_hits;
	float2 data[HIT_BUFFER_SIZE];
};

// *** *** *** *** ***

extern "C" __global__ void __raygen__test() {
	uint3 launch_index = optixGetLaunchIndex();
	uint3 launch_dimensions = optixGetLaunchDimensions();

	int pixel_ind = launch_index.x;
	int number_of_rays = launch_dimensions.x;

	float3 O = optixLaunchParams.O[pixel_ind];
	float3 v = optixLaunchParams.v[pixel_ind];
	float t_min = optixLaunchParams.t_min[pixel_ind];

	float *t = optixLaunchParams.t;
	int *indices = optixLaunchParams.indices;

	// *********************************************************************************************

	SRayPayload rp;

	unsigned long long rp_addr = ((unsigned long long)&rp);
	unsigned rp_addr_lo = rp_addr;
	unsigned rp_addr_hi = rp_addr >> 32;

	// *********************************************************************************************

	rp.number_of_hits = 0;

	optixTrace(
		optixLaunchParams.AS,
		O,
		v,
		t_min,
		INFINITY,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
		0,
		1,
		0,

		rp_addr_lo,
		rp_addr_hi
	);

	// *********************************************************************************************

	for (int i = 0; i < HIT_BUFFER_SIZE; ++i) {
		int element_index = (i * number_of_rays) + pixel_ind;

		if (i < rp.number_of_hits) {
			float2 tmp = rp.data[i];

			t[element_index] = tmp.x;
			indices[element_index] = __float_as_int(tmp.y);
		} else {
			t[element_index] = INFINITY;
			indices[element_index] = -1;
		}
	}
}

// *** *** *** *** ***

extern "C" __global__ void __raygen__() {
	uint3 launch_index = optixGetLaunchIndex();
	uint3 launch_dimensions = optixGetLaunchDimensions();

	int pixel_ind = launch_index.x;
	int ray_ind = optixLaunchParams.ray_indices[pixel_ind];
	int number_of_rays = launch_dimensions.x;

	float3 O = optixLaunchParams.O[ray_ind];
	float3 v = optixLaunchParams.v[ray_ind];
	float t_min = optixLaunchParams.t_min[ray_ind];

	float *t = optixLaunchParams.t;
	int *indices = optixLaunchParams.indices;

	// *********************************************************************************************

	SRayPayload rp;

	unsigned long long rp_addr = ((unsigned long long)&rp);
	unsigned rp_addr_lo = rp_addr;
	unsigned rp_addr_hi = rp_addr >> 32;

	// *********************************************************************************************

	rp.number_of_hits = 0;

	optixTrace(
		optixLaunchParams.AS,
		O,
		v,
		t_min,
		INFINITY,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
		0,
		1,
		0,

		rp_addr_lo,
		rp_addr_hi
	);

	// *********************************************************************************************

	for (int i = 0; i < HIT_BUFFER_SIZE; ++i) {
		int element_index = (i * number_of_rays) + pixel_ind;

		if (i < rp.number_of_hits) {
			float2 tmp = rp.data[i];

			t[element_index] = tmp.x;
			indices[element_index] = __float_as_int(tmp.y);
		} else {
			t[element_index] = INFINITY;
			indices[element_index] = -1;
		}
	}
}

// *** *** *** *** ***

// Paper: collect every relevant primitive hit and keep the payload sorted by t.
extern "C" __global__ void __anyhit__() {
	SRayPayload *rp;

	unsigned long long rp_addr_lo = optixGetPayload_0();
	unsigned long long rp_addr_hi = optixGetPayload_1();
	*((unsigned long long *)&rp) = rp_addr_lo + (rp_addr_hi << 32);

	// *********************************************************************************************

	float t_hit = optixGetRayTmax();
	
	float3 O = optixGetObjectRayOrigin();
	float3 v = optixGetObjectRayDirection();
	
	unsigned Gauss_ind = optixGetInstanceIndex();

	int number_of_hits = rp->number_of_hits;
	float2 *data = rp->data;

	float *t = optixLaunchParams.t;
	int *indices = optixLaunchParams.indices;

	// *** *** *** *** ***

	float2 tmp1 = make_float2(t_hit, __int_as_float(Gauss_ind));
	float2 tmp2;

	for (int i = 0; i < number_of_hits; ++i) {
		tmp2 = data[i];

		if (tmp1.x < tmp2.x) {
			data[i] = tmp1;
			tmp1 = tmp2;
		}
	}

	if (number_of_hits < HIT_BUFFER_SIZE) {
		data[number_of_hits] = tmp1;
		rp->number_of_hits = number_of_hits + 1;

		optixIgnoreIntersection();
	} else {
		if (t_hit <= tmp2.x) optixIgnoreIntersection();
	}
}
