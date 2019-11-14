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

add_executable(DifferOperators "src/DifferentialOperators.cpp" "src/DifferentialOperators.h")
target_link_libraries(DifferOperators debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(DifferOperators optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(DifferOperators Eigen3::Eigen)
target_link_libraries(DifferOperators ${VTK_LIBRARIES})

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

add_executable(DifferMat "src/DifferentiationMatrix.cpp")
target_link_libraries(DifferMat debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(DifferMat optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(DifferMat Eigen3::Eigen)

add_executable(Temp "src/temp.cpp")
target_link_libraries(Temp debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(Temp optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(Temp Eigen3::Eigen)
target_link_libraries(Temp ${VTK_LIBRARIES})

add_executable(MeshIO "src/mesh_io.cpp" "src/mesh_io.h")
target_link_libraries(MeshIO Eigen3::Eigen)

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

add_executable(ADMM "src/ADMM/ADMM.cpp" "src/ADMM/ADMM.h")
target_link_libraries(ADMM debug "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_debug.lib")
target_link_libraries(ADMM optimized "$ENV{Surface_Mesh_DIR}/lib/surface_mesh_release.lib")
target_link_libraries(ADMM Eigen3::Eigen)
target_link_libraries(ADMM ${VTK_LIBRARIES})