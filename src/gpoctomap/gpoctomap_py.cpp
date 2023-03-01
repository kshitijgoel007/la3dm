#include "gpoctomap.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;

PYBIND11_MODULE(gpoctomap_py, m)
{
  py::class_<la3dm::GPOctoMap>(m, "GPOctoMap")
	.def(py::init<>());
}
