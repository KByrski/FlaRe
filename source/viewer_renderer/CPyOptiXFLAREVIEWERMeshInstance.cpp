#ifdef _WIN32
	#define NOMINMAX
#endif

#include "CPyOptiXFLAREVIEWERMeshInstance.h"

// *** *** *** *** ***

torch::Tensor CPyOptiXFLAREVIEWERMeshInstance::GetInstanceAsTensor() {
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	return torch::from_blob(instance.dev_instance, {sizeof(OptixInstance) / sizeof(float)}, options);
}

// *** *** *** *** ***

torch::Tensor CPyOptiXFLAREVIEWERMeshInstance::GetHitgroupRecordAsTensor() {
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	return torch::from_blob(instance.dev_hitgroup_record, {sizeof(SbtHitgroupRecord) / sizeof(float)}, options);
}