#include "gpoctomap.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;

pcl::PointCloud<pcl::PointXYZI> eigenToPCL(Eigen::MatrixXf &m)
{
  pcl::PointCloud<pcl::PointXYZI> xyz;
  xyz.points.reserve(m.rows());
  for (int i = 0; i < m.rows(); ++i)
  {
    pcl::PointXYZI p;
    p.x = m(i, 0);
    p.y = m(i, 1);
    p.z = m(i, 2);
    p.intensity = m(i, 3);
    xyz.points.push_back(p);
  }
  return xyz;
}

PYBIND11_MODULE(gpoctomap_py, m)
{
  py::class_<la3dm::GPOctoMap>(m, "GPOctoMap")
      .def(py::init<>())
      .def("set_resolution", &la3dm::GPOctoMap::set_resolution)
      .def("get_resolution", &la3dm::GPOctoMap::get_resolution)
      .def("insert_color_pointcloud", [](la3dm::GPOctoMap& gptree, Eigen::MatrixXf& pcld) {
        la3dm::PCLPointCloud pcld_converted = eigenToPCL(pcld);
        gptree.insert_pointcloud(pcld_converted, la3dm::point3f(0.0, 0.0, 0.0), -1);
      });
}
