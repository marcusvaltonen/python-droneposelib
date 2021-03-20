#include "relpose.hpp"
#include "get_valtonenornhag_arxiv_2021.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

py::list get_valtonenornhag_arxiv_2021_fEf_wrapper(const Eigen::Ref<const Eigen::MatrixXd> &x1,
                                                   const Eigen::Ref<const Eigen::MatrixXd> &x2,
                                                   const Eigen::Ref<const Eigen::MatrixXd> &R1,
                                                   const Eigen::Ref<const Eigen::MatrixXd> &R2)
{

    if (x1.rows() != 2 || x1.cols() != 3) {
        throw std::invalid_argument("First argument should be of size 2x3");
    }
    if (x2.rows() != 2 || x2.cols() != 3) {
        throw std::invalid_argument("Second argument should be of size 2x3");
    }
    if (R1.rows() != 3 || R1.cols() != 3) {
        throw std::invalid_argument("Third argument should be of size 3x3");
    }
    if (R2.rows() != 3 || R2.cols() != 3) {
        throw std::invalid_argument("Fourth argument should be of size 3x3");
    }

    std::vector<DronePoseLib::RelPose> posedata = DronePoseLib::ValtonenOrnhagArxiv2021::get_fEf(x1, x2, R1, R2);

    py::list lst;
    for (int i = 0; i < posedata.size(); i++) {
        py::dict d;
        d["F"] = posedata[i].F;
        d["f"] = posedata[i].f;
        lst.append(d);
    }

    return lst;
}

py::list get_valtonenornhag_arxiv_2021_frEfr_wrapper(const Eigen::Ref<const Eigen::MatrixXd> &x1,
                                                     const Eigen::Ref<const Eigen::MatrixXd> &x2,
                                                     const Eigen::Ref<const Eigen::MatrixXd> &R1,
                                                     const Eigen::Ref<const Eigen::MatrixXd> &R2,
                                                     const bool use_fast_solver)
{

    if (x1.rows() != 2 || x1.cols() != 4) {
        throw std::invalid_argument("First argument should be of size 2x4");
    }
    if (x2.rows() != 2 || x2.cols() != 4) {
        throw std::invalid_argument("Second argument should be of size 2x4");
    }
    if (R1.rows() != 3 || R1.cols() != 3) {
        throw std::invalid_argument("Third argument should be of size 3x3");
    }
    if (R2.rows() != 3 || R2.cols() != 3) {
        throw std::invalid_argument("Fourth argument should be of size 3x3");
    }

    std::vector<DronePoseLib::RelPose> posedata =
        DronePoseLib::ValtonenOrnhagArxiv2021::get_frEfr(x1, x2, R1, R2, use_fast_solver);

    py::list lst;
    for (int i = 0; i < posedata.size(); i++) {
        py::dict d;
        d["F"] = posedata[i].F;
        d["f"] = posedata[i].f;
        d["r"] = posedata[i].r;
        lst.append(d);
    }

    return lst;
}


PYBIND11_MODULE(droneposelib, m) {
  m.doc() = R"doc(
        Python module
        -----------------------
        .. currentmodule:: droneposelib
        .. autosummary::
           :toctree: _generate

           get_valtonenornhag_arxiv_2021_fEf
           get_valtonenornhag_arxiv_2021_frEfr
    )doc";

  m.def("get_valtonenornhag_arxiv_2021_fEf", &get_valtonenornhag_arxiv_2021_fEf_wrapper, R"doc(
        Valtonen Ornhag et al. (ArXiV, 2021) 3-point relative pose using IMU data.

        More info to be added...
    )doc");

  m.def("get_valtonenornhag_arxiv_2021_frEfr", &get_valtonenornhag_arxiv_2021_frEfr_wrapper, R"doc(
        Valtonen Ornhag et al. (ArXiV, 2021) 4-point radial relative pose using IMU data.

        More info to be added...
    )doc");
}
