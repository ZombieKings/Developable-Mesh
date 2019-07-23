# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("Developable Mesh")

#SURFACE MESH
include_directories("$ENV{Surface_Mesh_DIR}/include")

#Eigen
find_package(Eigen3 REQUIRED)

#OpenCV
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

#VTK
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#alglib
include_directories("$ENV{Alglib_DIR}/inc")

add_executable(Interpolation "src/Interpolation/main.cpp" "src/Interpolation/interpolation.cpp" "src/Interpolation/interpolation.h" "src/mesh_tools.h")
target_link_libraries(Interpolation "$ENV{Surface_Mesh_DIR}/lib/surface_mesh.lib")
target_link_libraries(Interpolation Eigen3::Eigen)
target_link_libraries(Interpolation ${OpenCV_LIBS})
target_link_libraries(Interpolation ${VTK_LIBRARIES})

add_executable(Least_Norm "src/LeastNorm/main.cpp" "src/LeastNorm/least_norm.cpp" "src/LeastNorm/least_norm.h" "src/mesh_tools.h" "src/LeastNorm/myVisualizer.cpp" "src/LeastNorm/myVisualizer.h")
target_link_libraries(Least_Norm "$ENV{Surface_Mesh_DIR}/lib/surface_mesh.lib")
target_link_libraries(Least_Norm Eigen3::Eigen)
target_link_libraries(Least_Norm ${OpenCV_LIBS})
target_link_libraries(Least_Norm ${VTK_LIBRARIES})
target_link_libraries(Least_Norm debug "$ENV{Alglib_DIR}/lib/alglib_debug.lib" )
target_link_libraries(Least_Norm optimized  "$ENV{Alglib_DIR}/lib/alglib_release.lib")

add_executable(MeshUpdate "src/MeshUpdate/meshupdate.cpp" "src/MeshUpdate/myVisualizer.cpp" "src/MeshUpdate/myVisualizer.h")
target_link_libraries(MeshUpdate "$ENV{Surface_Mesh_DIR}/lib/surface_mesh.lib")
target_link_libraries(MeshUpdate Eigen3::Eigen)
target_link_libraries(MeshUpdate ${VTK_LIBRARIES})

add_executable(TestObject "src/test.cpp")
target_link_libraries(TestObject "$ENV{Surface_Mesh_DIR}/lib/surface_mesh.lib")
target_link_libraries(TestObject Eigen3::Eigen)
target_link_libraries(TestObject ${OpenCV_LIBS})

add_executable(Temp "src/temp.cpp")
target_link_libraries(Temp Eigen3::Eigen)
target_link_libraries(Temp ${OpenCV_LIBS})
