#ifdef _WIN32
	#define NOMINMAX

	#include "framework.h"
#endif

#include "CPyOptiXFLAREVIEWERMeshInstance.h"
#include "CPyOptiXFLAREVIEWERRenderer.h"
#include "GenerateRays.h"

// *** *** *** *** ***

PYBIND11_MODULE(PYOPTIXFLAREVIEWER, m) {
	m.doc() = "Isolated OptiX ray tracer for the FlaRe viewer";

	// *********************************************************************************************

	pybind11::class_<CPyOptiXFLAREVIEWERMeshInstance>(m, "CPyOptiXFLAREVIEWERMeshInstance")
		.def("GetInstanceAsTensor", &CPyOptiXFLAREVIEWERMeshInstance::GetInstanceAsTensor)
		.def("GetHitgroupRecordAsTensor", &CPyOptiXFLAREVIEWERMeshInstance::GetHitgroupRecordAsTensor);

	// *********************************************************************************************

	pybind11::class_<CPyOptiXFLAREVIEWERRenderer>(m, "CPyOptiXFLAREVIEWERRenderer")
		.def(pybind11::init<int, float, int, int>(),
			py::arg("number_of_sides"),
			py::arg("chi_square_squared_radius"),
			py::arg("max_batch_size"),
			py::arg("max_recursion_depth")
		)
		.def("SetGeometry", &CPyOptiXFLAREVIEWERRenderer::SetGeometry, "Sets the means, scales and the quaternions of the Gaussians")
		.def("CreateMeshInstance", &CPyOptiXFLAREVIEWERRenderer::CreateMeshInstance)

		.def("Forward",
			&CPyOptiXFLAREVIEWERRenderer::Forward,
			"Renders the image for the inference phase"
		);

	// *********************************************************************************************

	m.def("GenerateRays", &GenerateRays, "Generates the direction vectors of the rays");
}