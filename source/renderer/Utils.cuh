#pragma once

#include "Header.cuh"

// *** *** *** *** ***

__device__ __forceinline__ uint32_t __half22uint32_t(half2 x) {
	return ((uint32_t &)x);
}

// *** *** *** *** ***

__device__ __forceinline__ uint32_t __bfloat1622uint32_t(__nv_bfloat162 x) {
	return ((uint32_t &)x);
}