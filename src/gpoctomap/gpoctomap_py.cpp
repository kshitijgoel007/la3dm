#include "gpoctomap.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;

pcl::PointCloud<pcl::PointXYZI> eigenToPCLXYZI(Eigen::MatrixXf &m)
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

pcl::PointCloud<pcl::PointXYZ> eigenToPCLXYZ(Eigen::MatrixXf &m)
{
  pcl::PointCloud<pcl::PointXYZ> xyz;
  xyz.points.reserve(m.rows());
  for (int i = 0; i < m.rows(); ++i)
  {
    pcl::PointXYZ p;
    p.x = m(i, 0);
    p.y = m(i, 1);
    p.z = m(i, 2);
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
      .def("insert_color_pointcloud", [](la3dm::GPOctoMap& gptree, Eigen::MatrixXf& pcld, float ds_resolution) {
        la3dm::PCLPointCloud pcld_converted = eigenToPCLXYZ(pcld);
        gptree.insert_pointcloud(pcld_converted, la3dm::point3f(0.0, 0.0, 0.0), ds_resolution);
      })
      .def("get_pointcloud", [](la3dm::GPOctoMap& gptree) {
        std::vector<Eigen::Vector3f> points;
        for (auto it = gptree.begin_leaf(); it != gptree.end_leaf(); ++it)
        {
          la3dm::point3f p = it.get_loc();

          if (it.get_node().get_state() == la3dm::State::OCCUPIED)
          {
            // auto pruned = it.get_pruned_locs();
            // for (auto n = pruned.cbegin(); n < pruned.cend(); ++n)
            //   points.push_back(Eigen::Vector3f(n->x(), n->y(), n->z()));
            points.push_back(Eigen::Vector3f(p.x(), p.y(), p.z()));
          }
        }

        Eigen::MatrixXf final_pcld = Eigen::MatrixXf(points.size(), 3);
        for (int i = 0; i < points.size(); i++)
        {
          final_pcld(i, Eigen::all) = points[i];
        }

        return final_pcld;
      });
}
