#include "Header.cuh"

// *** *** *** *** ***

extern "C" __constant__ SLaunchParams optixLaunchParams;

// *** *** *** *** ***

struct SRayPayload {
	int number_of_hits;
	float2 data[HIT_BUFFER_SIZE];
	int number_of_Gaussians;
	int closest_mesh_face_ind;
	SbtHitgroupRecordData *hitgroup_record_data;
	float2 uv;
};

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
	rp.number_of_Gaussians = optixLaunchParams.number_of_Gaussians;

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

	int last_instance_ind = -1; // !!! !!! !!!
	float last_t;

	for (int i = 0; i < HIT_BUFFER_SIZE; ++i) {
		int element_index = (i * number_of_rays) + pixel_ind;

		float t_out = INFINITY;
		int ind_out = -1;

		if (i < rp.number_of_hits) {
			float2 tmp = rp.data[i];

			last_t = tmp.x;
			last_instance_ind = __float_as_int(tmp.y);

			if (last_instance_ind < optixLaunchParams.number_of_Gaussians) {
				t_out = tmp.x;
				ind_out = last_instance_ind;

				bool is_lit = true;

				for (int j = 0; j < optixLaunchParams.number_of_lights; ++j) {
					float *lights_addr = ((float *)optixLaunchParams.lights) + (j * 6); // !!! !!! !!!
					float3 L = *((float3 *)(lights_addr + 3));

					float3 P_hit = make_float3(
						__fmaf_rn(v.x, t_out, O.x),
						__fmaf_rn(v.y, t_out, O.y),
						__fmaf_rn(v.z, t_out, O.z)
					);

					unsigned int is_lit_current = 0;

					optixTrace(
						optixLaunchParams.AS,
						P_hit,
						make_float3(-L.x, -L.y, -L.z),
						optixLaunchParams.epsilon,
						INFINITY,
						0.0f,
						OptixVisibilityMask(15),
						OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
						0,
						1,
						1, // !!! !!! !!!

						is_lit_current
					);

					is_lit = is_lit & is_lit_current;
				}

				if (!is_lit)
					// We code the shadow boolean flag as the sign bit... .
					t_out = -t_out; // !!! !!! !!!
			}
		}

		t[element_index] = t_out;
		indices[element_index] = ind_out;
	}

	// *********************************************************************************************

	// Ostatni element hit bufora jest punktem przeciêcia promienia z meshem. Wyznaczamy promieñ odbity i za³amany w oparciu o hitgroup_record_data... .
	if (last_instance_ind >= optixLaunchParams.number_of_Gaussians) {
		unsigned long long hitgroup_record_data_addr = ((unsigned long long)rp.hitgroup_record_data);
		unsigned hitgroup_record_data_addr_lo = hitgroup_record_data_addr;
		unsigned hitgroup_record_data_addr_hi = hitgroup_record_data_addr >> 32;

		optixLaunchParams.is_mesh[ray_ind] = 1;
		optixLaunchParams.buffer1[ray_ind] = make_float4(
			__int_as_float(hitgroup_record_data_addr_lo),
			__int_as_float(hitgroup_record_data_addr_hi),
			__int_as_float(last_instance_ind - optixLaunchParams.number_of_Gaussians),
			__int_as_float(rp.closest_mesh_face_ind)
		);
		optixLaunchParams.buffer2[ray_ind] = make_float4(
			rp.uv.x,
			rp.uv.y,
			last_t,
			0.0f
		);
	} else
		optixLaunchParams.is_mesh[ray_ind] = 0;
}

// *** *** *** *** ***

static __device__ __forceinline__ float __frcp_approx(float x) {
	float result;
	asm volatile ("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
	return result;
}

// *** *** *** *** ***

static __device__ __forceinline__ float __fsqrt_approx(float x) {
	float result;
	asm volatile ("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
	return result;
}

// *** *** *** *** ***

static __device__ __forceinline__ float __frsqrt_approx(float x) {
	float result;
	asm volatile ("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
	return result;
}

// *** *** *** *** ***

static __device__ __forceinline__ float3 Reflect(float3 N, float3 v, float cos_theta_1) {
	float tmp = -2.0f * cos_theta_1;
	return make_float3(
		__fmaf_rn(N.x, tmp, v.x),
		__fmaf_rn(N.y, tmp, v.y),
		__fmaf_rn(N.z, tmp, v.z)
	);
}

// *** *** *** *** ***

static __device__ __forceinline__  bool GetCosTheta2ForRefraction(float cos_theta_1, float n_inv, float &cos_theta_2) {
	float cos_theta_2_squared = __fmaf_rn(
		-n_inv * n_inv,
		((__fmaf_rn(-cos_theta_1, cos_theta_1, 1.0f) < 0.0f) ? 0.0f : __fmaf_rn(-cos_theta_1, cos_theta_1, 1.0f)),
		1.0f
	);
	if (cos_theta_2_squared < 0.0f) return true;
	else {
		cos_theta_2 = __fsqrt_approx(cos_theta_2_squared);
		return false;
	}
}

// *** *** *** *** ***

static __device__ __forceinline__  float3 Refract(
	float3 N,
	float3 v,
	float n_inv,
	float cos_theta_1, float cos_theta_2
) {
	float tmp = __fmaf_rn(-cos_theta_1, n_inv, copysignf(cos_theta_2, cos_theta_1));
	return make_float3(
		(v.x * n_inv) + (N.x * tmp),
		(v.y * n_inv) + (N.y * tmp),
		(v.z * n_inv) + (N.z * tmp)
	);
}

// *** *** *** *** ***

static __device__ __forceinline__ float GetFresnelFactor(float cos_theta_1, float eta, float cos_theta_2) {
	float tmp1 = (1.0f - eta) * __frcp_approx(1.0f + eta);
	float R0 = tmp1 * tmp1;
	float tmp2 = 1.0f - ((eta <= 1.0f) ? cos_theta_2 : cos_theta_1);
	float tmp3 = tmp2 * tmp2;
	tmp3 = tmp3 * tmp3;
	tmp3 = tmp2 * tmp3;
	return __fmaf_rn(1.0f - R0, tmp3, R0);
}

// *** *** *** *** ***

extern "C" __global__ void __raygen__shadows() {
	uint3 launch_index = optixGetLaunchIndex();

	int tid = launch_index.x;
	int ray_ind = optixLaunchParams.ray_indices_initial[tid];

	float T_old = optixLaunchParams.T_ptr[ray_ind];

	float T_R = T_old;
	float T_T = T_old;

	float3 color = optixLaunchParams.bitmap[ray_ind];

	if (optixLaunchParams.is_mesh[ray_ind]) {
		float4 data1 = optixLaunchParams.buffer1[ray_ind];
		float4 data2 = optixLaunchParams.buffer2[ray_ind];

		unsigned long long hitgroup_record_data_addr = (((unsigned long long) __float_as_uint(data1.y)) << 32) + ((unsigned long long) __float_as_uint(data1.x));
		SbtHitgroupRecordData *hitgroup_record_data = (SbtHitgroupRecordData *)hitgroup_record_data_addr;
		int instance_ind = __float_as_int(data1.z);
		int face_ind = __float_as_int(data1.w);

		float w = 1.0f - data2.x - data2.y;

		// Computing normal vector
		int3 indices = hitgroup_record_data->indices_buffer[face_ind];
		float3 N1 = hitgroup_record_data->normals_buffer[indices.x];
		float3 N2 = hitgroup_record_data->normals_buffer[indices.y];
		float3 N3 = hitgroup_record_data->normals_buffer[indices.z];
		float3 N_raw = make_float3(
			__fmaf_rn(N1.x, w, __fmaf_rn(N2.x, data2.x, N3.x * data2.y)),
			__fmaf_rn(N1.y, w, __fmaf_rn(N2.y, data2.x, N3.y * data2.y)),
			__fmaf_rn(N1.z, w, __fmaf_rn(N2.z, data2.x, N3.z * data2.y))
		);

		float *M4N_addr = ((float *)optixLaunchParams.M4N) + (instance_ind * 9);
		float3 M4N1 = ((float3 *)M4N_addr)[0];
		float3 M4N2 = ((float3 *)M4N_addr)[1];
		float3 M4N3 = ((float3 *)M4N_addr)[2];

		float3 N = make_float3(
			__fmaf_rn(M4N1.x, N_raw.x, __fmaf_rn(M4N1.y, N_raw.y, M4N1.z * N_raw.z)),
			__fmaf_rn(M4N2.x, N_raw.x, __fmaf_rn(M4N2.y, N_raw.y, M4N2.z * N_raw.z)),
			__fmaf_rn(M4N3.x, N_raw.x, __fmaf_rn(M4N3.y, N_raw.y, M4N3.z * N_raw.z))
		);

		float N_norm_inv = __frsqrt_approx(__fmaf_rn(N.x, N.x, __fmaf_rn(N.y, N.y, N.z * N.z)));
		N = make_float3(
			N.x * N_norm_inv,
			N.y * N_norm_inv,
			N.z * N_norm_inv
		);

		// Acquiring material properties
		float *material_addr = ((float *)optixLaunchParams.materials) + (instance_ind * 6); // !!! !!! !!!
		float3 Id = *((float3 *)(material_addr + 0));
		float kd_old = *((float *)(material_addr + 3));

		float eta_old = *((float *)(material_addr + 4));
		float eta_old_inv = __frcp_approx(eta_old);

		float n = *((float *)(material_addr + 5));

		// Hit point
		float3 O = optixLaunchParams.O[ray_ind];
		float3 v = optixLaunchParams.v[ray_ind];
		float3 P_hit = make_float3(
			__fmaf_rn(v.x, data2.z, O.x),
			__fmaf_rn(v.y, data2.z, O.y),
			__fmaf_rn(v.z, data2.z, O.z)
		);

		// Reflected vector
		float cos_theta_1 = __fmaf_rn(N.x, v.x, __fmaf_rn(N.y, v.y, N.z * v.z));
		float3 R = Reflect(N, v, cos_theta_1);

		// Refracted vector
		float eta;
		float eta_inv;
		float kd;
		if (cos_theta_1 <= 0.0f) {
			eta = eta_old;
			eta_inv = eta_old_inv;

			kd = kd_old;
		} else {
			eta_inv = eta_old;
			eta = eta_old_inv;

			kd = 0.0f; // !!! !!! !!!
		}
		float cos_theta_2;
		bool TIR = GetCosTheta2ForRefraction(cos_theta_1, eta_inv, cos_theta_2);

		float3 T;
		float F1;
		if (!TIR) {
			T = Refract(N, v, eta_inv, cos_theta_1, cos_theta_2);
			F1 = GetFresnelFactor(cos_theta_1, eta_inv, cos_theta_2);
		} else
			F1 = 1.0f;

		// !!! !!! !!!
		T_R *= F1;
		T_T *= (1.0f - F1) * (1.0f - kd);
		// !!! !!! !!!

		// *************************************************************************************

		float3 color_ds = make_float3(0.0f, 0.0f, 0.0f);

		if (cos_theta_1 < 0.0f) {
			color_ds = make_float3(
				__fmaf_rn(Id.x * (1.0f - F1) * kd_old, optixLaunchParams.Ia.x, color_ds.x),
				__fmaf_rn(Id.y * (1.0f - F1) * kd_old, optixLaunchParams.Ia.y, color_ds.y),
				__fmaf_rn(Id.z * (1.0f - F1) * kd_old, optixLaunchParams.Ia.z, color_ds.z)
			);
		}

		for (int i = 0; i < optixLaunchParams.number_of_lights; ++i) {
			float *lights_addr = ((float *)optixLaunchParams.lights) + (i * 6); // !!! !!! !!!
			float3 Is = *((float3 *)(lights_addr + 0));
			float3 L = *((float3 *)(lights_addr + 3));

			// *************************************************************************************

			unsigned int is_lit = 0;

			optixTrace(
				optixLaunchParams.AS,
				P_hit,
				make_float3(-L.x, -L.y, -L.z),
				optixLaunchParams.epsilon,
				INFINITY,
				0.0f,
				OptixVisibilityMask(15),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
				0,
				1,
				0, // !!! !!! !!!

				is_lit
			);

			// *************************************************************************************

			// Diffuse
			float L_norm_inv = __frsqrt_approx(__fmaf_rn(L.x, L.x, __fmaf_rn(L.y, L.y, L.z * L.z)));
			L = make_float3(
				L.x * L_norm_inv,
				L.y * L_norm_inv,
				L.z * L_norm_inv
			);
			cos_theta_1 = __fmaf_rn(N.x, L.x, __fmaf_rn(N.y, L.y, N.z * L.z));

			if (cos_theta_1 <= 0.0f) {
				eta = eta_old;
				eta_inv = eta_old_inv;
			} else {
				eta_inv = eta_old;
				eta = eta_old_inv;
			}
			TIR = GetCosTheta2ForRefraction(cos_theta_1, eta_inv, cos_theta_2);

			float F2;
			if (!TIR) {
				F2 = GetFresnelFactor(cos_theta_1, eta_inv, cos_theta_2);
			} else
				F2 = 1.0f;

			float diffuse = (1.0f - F1) * (1.0f - F2) * kd_old * __saturatef(-cos_theta_1);
			if (!is_lit)
				diffuse = 0.0f;

			// *********************************************************************************

			// Specular
			float3 H = make_float3(
				v.x + L.x,
				v.y + L.y,
				v.z + L.z
			);
			float H_norm_inv = __frsqrt_approx(__fmaf_rn(H.x, H.x, __fmaf_rn(H.y, H.y, H.z * H.z)));
			H = make_float3(
				H.x * H_norm_inv,
				H.y * H_norm_inv,
				H.z * H_norm_inv
			);
			cos_theta_1 = __saturatef(-__fmaf_rn(N.x, H.x, __fmaf_rn(N.y, H.y, N.z * H.z)));

			float specular = F2 * cos_theta_1 * __frcp_approx(__fmaf_rn(n, 1.0f - cos_theta_1, cos_theta_1));
			if (!is_lit)
				specular = 0.0f;

			// *********************************************************************************

			color_ds = make_float3(
				__fmaf_rn(Is.x, __fmaf_rn(Id.x, diffuse, specular), color_ds.x),
				__fmaf_rn(Is.y, __fmaf_rn(Id.y, diffuse, specular), color_ds.y),
				__fmaf_rn(Is.z, __fmaf_rn(Id.z, diffuse, specular), color_ds.z)
			);
		}

		// *************************************************************************************

		// Bitmap update
		optixLaunchParams.bitmap[ray_ind] = make_float3(
			__saturatef(__fmaf_rn(T_old, color_ds.x, color.x)),
			__saturatef(__fmaf_rn(T_old, color_ds.y, color.y)),
			__saturatef(__fmaf_rn(T_old, color_ds.z, color.z))
		);

		// *************************************************************************************

		float T1;
		float3 v1;
		float T2;
		float3 v2;

		if (T_T <= T_R) {
			T1 = T_R;
			v1 = R;
			T2 = T_T;
			v2 = T;
		} else {
			T1 = T_T;
			v1 = T;
			T2 = T_R;
			v2 = R;
		}

		// Ray 1
		optixLaunchParams.O[ray_ind] = P_hit;
		optixLaunchParams.t_min[ray_ind] = optixLaunchParams.epsilon;

		optixLaunchParams.v[ray_ind] = v1;
		optixLaunchParams.T_ptr[ray_ind] = T1;
		optixLaunchParams.is_active[tid] = (T1 >= optixLaunchParams.T_threshold);

		// Ray 2
		optixLaunchParams.entry.O[ray_ind] = P_hit;
		optixLaunchParams.entry.t_min[ray_ind] = optixLaunchParams.epsilon;

		optixLaunchParams.entry.v[ray_ind] = v2;
		optixLaunchParams.entry.T[ray_ind] = T2;
		optixLaunchParams.entry.is_active[tid] = (T2 >= optixLaunchParams.T_threshold);
	} else {
		optixLaunchParams.bitmap[ray_ind] = make_float3(
			__fmaf_rn(optixLaunchParams.bg_color.x, T_R, color.x),
			__fmaf_rn(optixLaunchParams.bg_color.y, T_R, color.y),
			__fmaf_rn(optixLaunchParams.bg_color.z, T_R, color.z)
		);

		optixLaunchParams.is_active[tid] = 0;
		optixLaunchParams.entry.is_active[tid] = 0;
	}
}

// *** *** *** *** ***

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

// *** *** *** *** ***

extern "C" __global__ void __anyhit__mesh() {
	SRayPayload *rp;

	unsigned long long rp_addr_lo = optixGetPayload_0();
	unsigned long long rp_addr_hi = optixGetPayload_1();
	*((unsigned long long *)&rp) = rp_addr_lo + (rp_addr_hi << 32);

	// *********************************************************************************************

	float t_hit = optixGetRayTmax();
	unsigned instance_ind = optixGetInstanceIndex();

	int number_of_hits = rp->number_of_hits;
	float2 *data = rp->data;

	float *t = optixLaunchParams.t;
	int *indices = optixLaunchParams.indices;

	// *** *** *** *** ***

	float2 tmp1 = make_float2(t_hit, __int_as_float(instance_ind));
	float2 tmp2 = make_float2(INFINITY, -1); // !!! !!! !!!

	int i;
	for (i = 0; i < number_of_hits; ++i) {
		tmp2 = data[i];

		if (tmp1.x < tmp2.x)
			break;
	}

	if (
		(i < HIT_BUFFER_SIZE) && (
			(__float_as_int(tmp2.y) < rp->number_of_Gaussians) ||
			(i < number_of_hits)
		)
	) {
		data[i] = tmp1;
		rp->closest_mesh_face_ind = optixGetPrimitiveIndex();
		rp->hitgroup_record_data = (SbtHitgroupRecordData *)optixGetSbtDataPointer();
		rp->uv = optixGetTriangleBarycentrics();
		rp->number_of_hits = i + 1;
	}
}

// *** *** *** *** ***

extern "C" __global__ void __miss__shadows() {
	optixSetPayload_0(1);
}
