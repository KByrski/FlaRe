#ifdef _WIN32
	#define NOMINMAX

	#include "framework.h"
#endif

#include "CPyOptiXFLARERenderer.h"
#include "GenerateRays.h"

// *** *** *** *** ***

PYBIND11_MODULE(PYOPTIXFLARERENDERER, m) {
	m.doc() = "Python OptiX FLARE Renderer";

	// *********************************************************************************************

	pybind11::class_<CPyOptiXFLARERenderer>(m, "CPyOptiXFLARERenderer")
		.def(pybind11::init<int, float, int>(),
			py::arg("number_of_sides"),
			py::arg("chi_square_squared_radius"),
			py::arg("max_batch_size")
		)
		.def("SetGeometry", &CPyOptiXFLARERenderer::SetGeometry, "Sets the means, scales and the quaternions of the Gaussians")
		.def("Render", &CPyOptiXFLARERenderer::Render, "Computes the t parameters and the indices of the Gaussians intersected by the rays")
		.def("GetMedianDepth_base",
			&CPyOptiXFLARERenderer::GetMedianDepth_base,
			"Gets the transmittance-median depth and primitive index for the base model"
		)
		
		.def("Forward_training_base",
			&CPyOptiXFLARERenderer::Forward_training_base,
			"Renders the image for the base model training phase"
		)
		.def("Forward_inference_base",
			&CPyOptiXFLARERenderer::Forward_inference_base,
			"Renders the image for the base model inference phase"
		)
		.def("Forward_inference_base_with_geometry",
			&CPyOptiXFLARERenderer::Forward_inference_base_with_geometry,
			"Renders base RGB, alpha, expected depth, and composited normals"
		)
		.def("Backward_base",
			&CPyOptiXFLARERenderer::Backward_base,
			"Computes the base model gradient"
		)

		.def("GetMedianDepth",
			&CPyOptiXFLARERenderer::GetMedianDepth,
			"Gets the transmittance-median depth and primitive index for FlaRe"
		)
		.def("Forward_training",
			&CPyOptiXFLARERenderer::Forward_training,
			"Renders the image for the training phase"
		)
		.def("Forward_inference",
			&CPyOptiXFLARERenderer::Forward_inference,
			"Renders the image for the inference phase"
		)
		.def("Forward_pca",
			&CPyOptiXFLARERenderer::Forward_pca,
			"Renders PCA colors for per-Gaussian appearance embeddings"
		)
		.def("Forward_inference_with_geometry",
			&CPyOptiXFLARERenderer::Forward_inference_with_geometry,
			"Renders FlaRe RGB, alpha, expected depth, and composited normals"
		)
		.def("Backward",
			&CPyOptiXFLARERenderer::Backward,
			"Computes the gradient"
		);

	// *********************************************************************************************

	m.def("GenerateRays", &GenerateRays, "Generates the direction vectors of the rays");
}