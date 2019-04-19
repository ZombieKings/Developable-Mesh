# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("Developable Mesh Deformation")

#SURFACE MESH
include_directories("$ENV{Surface_Mesh_DIR}/include")

#Eigen
find_package(Eigen3 REQUIRED)

#PCL
find_package(PCL 1.3 REQUIRED COMPONENTS common visualization)

#OpenCV
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(Deformation "src/main.cpp" "src/Dev_Inter.cpp" "src/Dev_Inter.h")
target_link_libraries(Deformation "$ENV{Surface_Mesh_DIR}/lib/surface_mesh.lib")
target_link_libraries(Deformation Eigen3::Eigen)
target_link_libraries(Deformation ${PCL_LIBRARIES})
target_link_libraries(Deformation ${OpenCV_LIBS})

add_executable(TestObject "src/test.cpp")
target_link_libraries(TestObject "$ENV{Surface_Mesh_DIR}/lib/surface_mesh.lib")
target_link_libraries(TestObject Eigen3::Eigen)
target_link_libraries(TestObject ${PCL_LIBRARIES})
target_link_libraries(TestObject ${OpenCV_LIBS})

add_executable(LDLT "src/LDLT.cpp")
target_link_libraries(TestObject Eigen3::Eigen)
target_link_libraries(TestObject ${OpenCV_LIBS})
