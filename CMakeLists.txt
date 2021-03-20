cmake_minimum_required(VERSION 3.4)
project(python-droneposelib)

find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

SET(DRONEPOSELIB_DIR "lib/DronePoseLib")

SET(SOURCE_DIR "src")
# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR})
include_directories(${DRONEPOSELIB_DIR})
include_directories(${DRONEPOSELIB_DIR}/includes/DronePoseLib)
include_directories(${DRONEPOSELIB_DIR}/src/helpers)
include_directories(${DRONEPOSELIB_DIR}/src/solvers/valtonenornhag_arxiv_2021/fEf)
include_directories(${DRONEPOSELIB_DIR}/src/solvers/valtonenornhag_arxiv_2021/frEfr)
SET(SRCS_DRONEPOSELIB
    ${DRONEPOSELIB_DIR}/src/helpers/normalize2dpts.cpp
    ${DRONEPOSELIB_DIR}/src/helpers/quartic.cpp
    ${DRONEPOSELIB_DIR}/src/helpers/radial.cpp
    ${DRONEPOSELIB_DIR}/src/solvers/valtonenornhag_arxiv_2021/fEf/get_fEf.cpp
    ${DRONEPOSELIB_DIR}/src/solvers/valtonenornhag_arxiv_2021/frEfr/coeffs_frEfr.cpp
    ${DRONEPOSELIB_DIR}/src/solvers/valtonenornhag_arxiv_2021/frEfr/get_frEfr.cpp
    ${DRONEPOSELIB_DIR}/src/solvers/valtonenornhag_arxiv_2021/frEfr/solver_frEfr.cpp
)
SET(HDRS_DRONEPOSELIB
    ${DRONEPOSELIB_DIR}/includes/DronePoseLib/relpose.hpp
    ${DRONEPOSELIB_DIR}/includes/DronePoseLib/get_valtonenornhag_arxiv_2021.hpp
    ${DRONEPOSELIB_DIR}/src/helpers/charpoly.h
    ${DRONEPOSELIB_DIR}/src/helpers/normalize2dpts.hpp
    ${DRONEPOSELIB_DIR}/src/helpers/quartic.hpp
    ${DRONEPOSELIB_DIR}/src/helpers/radial.hpp
    ${DRONEPOSELIB_DIR}/src/helpers/sturm.h
    ${DRONEPOSELIB_DIR}/src/solvers/valtonenornhag_arxiv_2021/frEfr/coeffs_frEfr.hpp
    ${DRONEPOSELIB_DIR}/src/solvers/valtonenornhag_arxiv_2021/frEfr/solver_frEfr.hpp
)

# Generate python module
add_subdirectory(lib/pybind11)
pybind11_add_module(droneposelib "${SOURCE_DIR}/bindings.cpp" ${HDRS_DRONEPOSELIB} ${SRCS_DRONEPOSELIB})
target_link_libraries(droneposelib PRIVATE Eigen3::Eigen)

add_library(${PROJECT_NAME} STATIC ${HDRS_DRONEPOSELIB} ${SRCS_DRONEPOSELIB})

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)
