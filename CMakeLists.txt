# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("Developable Mesh")

#SURFACE MESH
include_directories("$ENV{Surface_Mesh_DIR}/include")

#Eigen
find_package(Eigen3 REQUIRED)

#Boost
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

#VTK
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#alglib
include_directories("$ENV{Alglib_DIR}/inc")

#Mosek
include_directories("$ENV{MOSEK_DIR}/h")

#Cinolib
include_directories("$ENV{Cinolib_DIR}/include")

#CGAL
find_package(CGAL REQUIRED)
include_directories("${CGAL_DIR}/include")
include_directories("${GMP_INCLUDE_DIR}")

add_executable(Interpolation "src/Interpolation/interpolation.cpp" "src/Interpolation/interpolation.h")
target_link_libraries(Interpolation debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(Interpolation optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(Interpolation Eigen3::Eigen)
target_link_libraries(Interpolation ${VTK_LIBRARIES})

add_executable(L0Interpolation "src/L0Inter/L0Inter.cpp" "src/L0Inter/L0Inter.h")
target_link_libraries(L0Interpolation debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(L0Interpolation optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(L0Interpolation Eigen3::Eigen)
target_link_libraries(L0Interpolation ${VTK_LIBRARIES})

add_executable(L1Interpolation "src/L1Inter/L1Inter.cpp" "src/L1Inter/L1Inter.h")
target_link_libraries(L1Interpolation debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(L1Interpolation optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(L1Interpolation Eigen3::Eigen)
target_link_libraries(L1Interpolation ${VTK_LIBRARIES})
target_link_libraries(L1Interpolation debug "$ENV{Alglib_DIR}/lib/alglib_debug.lib" )
target_link_libraries(L1Interpolation optimized  "$ENV{Alglib_DIR}/lib/alglib_release.lib")

add_executable(Least_Norm "src/LeastNorm/least_norm.cpp" "src/LeastNorm/least_norm.h")
target_link_libraries(Least_Norm debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(Least_Norm optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(Least_Norm Eigen3::Eigen)
target_link_libraries(Least_Norm ${VTK_LIBRARIES})

add_executable(NewtonMethod "src/NewtonMethod/newtonMethod.cpp" "src/NewtonMethod/newtonMethod.h")
target_link_libraries(NewtonMethod debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(NewtonMethod optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(NewtonMethod Eigen3::Eigen)
target_link_libraries(NewtonMethod ${VTK_LIBRARIES})

add_executable(LaplasianSmooth "src/LaplasianSmooth.cpp")
target_link_libraries(LaplasianSmooth debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(LaplasianSmooth optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(LaplasianSmooth Eigen3::Eigen)
target_link_libraries(LaplasianSmooth ${VTK_LIBRARIES})

add_executable(TestObject "src/test.cpp")
target_link_libraries(TestObject debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(TestObject optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(TestObject Eigen3::Eigen)
target_link_libraries(TestObject ${VTK_LIBRARIES})
target_link_libraries(TestObject debug "$ENV{Alglib_DIR}/lib/alglib_debug.lib" )
target_link_libraries(TestObject optimized  "$ENV{Alglib_DIR}/lib/alglib_release.lib")
target_link_libraries(TestObject "$ENV{MOSEK_DIR}/bin/mosek9_1.lib")

add_executable(DifferMat "src/DifferentiationMatrix.cpp")
target_link_libraries(DifferMat debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(DifferMat optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(DifferMat Eigen3::Eigen)

add_executable(Temp "src/temp.cpp")
target_link_libraries(Temp debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(Temp optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(Temp Eigen3::Eigen)
target_link_libraries(Temp ${VTK_LIBRARIES})

add_executable(opt_solver "src/solver/func_opt.cpp" "src/solver/func_opt.h" "src/solver/newton_solver.cpp" "src/solver/newton_solver.h" "src/solver/residual_func.h" "src/solver/main.cpp")
target_link_libraries(opt_solver ${Boost_LIBRARIES})
target_link_libraries(opt_solver Eigen3::Eigen)
target_link_libraries(opt_solver debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(opt_solver optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(opt_solver ${VTK_LIBRARIES})

add_executable(LBFGS "src/LBFGS/LBFGS.cpp" "src/LBFGS/LBFGS.h")
target_link_libraries(LBFGS debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(LBFGS optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(LBFGS Eigen3::Eigen)
target_link_libraries(LBFGS ${VTK_LIBRARIES})
target_link_libraries(LBFGS debug "$ENV{Alglib_DIR}/lib/alglib_debug.lib" )
target_link_libraries(LBFGS optimized  "$ENV{Alglib_DIR}/lib/alglib_release.lib")

add_executable(ADMM "src/ADMM/ADMM.cpp" "src/ADMM/ADMM.h" "src/ADMM/mosek_solver.h")
target_link_libraries(ADMM debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(ADMM optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(ADMM Eigen3::Eigen)
target_link_libraries(ADMM ${VTK_LIBRARIES})
target_link_libraries(ADMM "$ENV{MOSEK_DIR}/bin/mosek9_1.lib")

add_executable(QP "src/QP/QP.cpp" "src/QP/QP.h" "src/QP/mosek_solver.h")
target_link_libraries(QP debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(QP optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(QP Eigen3::Eigen)
target_link_libraries(QP ${VTK_LIBRARIES})
target_link_libraries(QP "$ENV{MOSEK_DIR}/bin/mosek9_1.lib")

add_executable(PlanB "src/PlanB/PlanB.cpp" "src/PlanB/PlanB.h" "src/PlanB/mosek_solver.h")
target_link_libraries(PlanB debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(PlanB optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(PlanB Eigen3::Eigen)
target_link_libraries(PlanB ${VTK_LIBRARIES})
target_link_libraries(PlanB "$ENV{MOSEK_DIR}/bin/mosek9_1.lib")

add_executable(Plan3 "src/Plan3/Plan3.cpp" "src/Plan3/Plan3.h")
target_link_libraries(Plan3 debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(Plan3 optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(Plan3 Eigen3::Eigen)
target_link_libraries(Plan3 ${VTK_LIBRARIES})
target_link_libraries(Plan3 "$ENV{MOSEK_DIR}/bin/mosek9_1.lib")

add_executable(Plan4 "src/Plan4/Plan4.cpp" "src/Plan4/Plan4.h")
target_link_libraries(Plan4 debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(Plan4 optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(Plan4 Eigen3::Eigen)
target_link_libraries(Plan4 ${VTK_LIBRARIES})

add_executable(DoF "src/DoF/DoF.cpp" "src/DoF/DoF.h")
target_link_libraries(DoF debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(DoF optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(DoF Eigen3::Eigen)
target_link_libraries(DoF ${VTK_LIBRARIES})
target_link_libraries(DoF debug "$ENV{Alglib_DIR}/lib/alglib_debug.lib" )
target_link_libraries(DoF optimized  "$ENV{Alglib_DIR}/lib/alglib_release.lib")