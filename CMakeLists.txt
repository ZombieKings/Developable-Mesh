# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("Developable Mesh Deformation")

#SURFACE MESH
include_directories("$ENV{Surface_Mesh_DIR}/include")

#Eigen
find_package(Eigen3 REQUIRED)

add_executable(Deformation "src/main.cpp" "src/Dev_Inter.cpp" "src/Dev_Inter.h")
target_link_libraries(Deformation "$ENV{Surface_Mesh_DIR}/lib/surface_mesh.lib")
target_link_libraries(Deformation Eigen3::Eigen)