#pragma once

// *** *** *** *** ***

#include "Header.cuh"

// *** *** *** *** ***

class CPyOptiXFLAREVIEWERMeshInstance;
class CPyOptiXFLAREVIEWERRenderer_CUDA;

// *** *** *** *** ***

class CPyOptiXFLAREVIEWERMeshInstance_CUDA {
	private:
		void *GASBuffer;
		OptixInstance instance;
		OptixInstance *dev_instance;
		SbtHitgroupRecord *dev_hitgroup_record;

	friend class CPyOptiXFLAREVIEWERMeshInstance;
	friend class CPyOptiXFLAREVIEWERRenderer_CUDA;
};