#pragma once

// *** *** *** *** ***

#include "CPyOptiXFLAREVIEWERMeshInstance_CUDA.h"
#include "Header.cuh"

#include <torch/extension.h>

// *** *** *** *** ***

class CPyOptiXFLAREVIEWERRenderer;

// *** *** *** *** ***

class CPyOptiXFLAREVIEWERMeshInstance {
	public:
		torch::Tensor GetInstanceAsTensor();
		torch::Tensor GetHitgroupRecordAsTensor();

	private:
		torch::Tensor vertex_buffer;
		torch::Tensor indices_buffer;
		torch::Tensor normals_buffer;

		CPyOptiXFLAREVIEWERMeshInstance_CUDA instance;

	friend class CPyOptiXFLAREVIEWERRenderer;
};
