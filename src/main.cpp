#include "Dev_Inter.h"

#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#define SIZE 

int main(int argc, char** argv)
{
	Surface_mesh mesh;
	//for (int i = 0; i < SIZE; ++i)
	//	for (int j = 0; j < SIZE; ++j)
	//		mesh.add_vertex(Point(i + 1, j + 1, 1));

	//for (int i = 0; i < SIZE - 1; ++i)
	//	for (int j = 0; j < SIZE - 1; ++j)
	//	{
	//		mesh.add_triangle(Surface_mesh::Vertex(j + i * SIZE), Surface_mesh::Vertex((j + 1) + i * SIZE), Surface_mesh::Vertex((j + 1) + (i + 1) * SIZE));
	//		mesh.add_triangle(Surface_mesh::Vertex(j + i * SIZE), Surface_mesh::Vertex((j + 1) + (i + 1) * SIZE), Surface_mesh::Vertex(j + (i + 1) * SIZE));
	//	}
	mesh.read("Skirt.obj");

	//std::cout << "vertices: " << mesh.n_vertices() << std::endl;
	//std::cout << "edges: " << mesh.n_edges() << std::endl;
	//std::cout << "faces: " << mesh.n_faces() << std::endl;

	//std::vector<unsigned int> anchor_idx{ 0 };
	//std::vector<Point> anchor_pos;
	//anchor_pos.push_back(Point(108, 108, 102));


	//for (auto vit: mesh.vertices())
	//{
	//	if (mesh.position(vit).x == 108 || mesh.position(vit).y == 108)
	//	{ 
	//		anchor_idx.push_back(vit.idx());
	//		anchor_pos.push_back(mesh.position(vit));
	//	}
	//}
	//for (auto vit : mesh.vertices())
	//{
	//	if (mesh.position(vit).x == 107 || mesh.position(vit).y == 107)
	//	{
	//		anchor_idx.push_back(vit.idx());
	//		anchor_pos.push_back(mesh.position(vit));
	//	}
	//}

	//for (int i = 1; i <= SIZE; ++i)
	//{
	//	anchor_idx.push_back(i * SIZE - 1);
	//	anchor_pos.push_back(mesh.position(Surface_mesh::Vertex(i * SIZE - 1)));
	//}

	//for (int i = 0; i < SIZE; ++i)
	//{
	//	anchor_idx.push_back((SIZE)*(SIZE - 1) + i);
	//	anchor_pos.push_back(mesh.position(Surface_mesh::Vertex((SIZE)*(SIZE - 1) + i)));
	//}

	std::vector<unsigned int> anchor_idx;
	std::vector<Point> anchor_pos;

	for (const auto &vit : mesh.vertices())
		if (mesh.position(vit)[1] > 24.5 || mesh.position(vit)[1] < -35 || abs(mesh.position(vit)[1]) < 1)
		{
			anchor_idx.push_back(vit.idx());
			anchor_pos.push_back(mesh.position(vit));
		}

	Dev_Inter cDef(mesh, anchor_pos, anchor_idx);
	cDef.SetConditions(0.001, 0.05, 0.01, 0.01, 0.01, 0.15, 0.01);
	cDef.Deformation();
	Surface_mesh result_mesh = cDef.Get_Result();

	//-----------Visualizer------------
	//将网格导入点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);
	//for (auto vit : result_mesh.vertices())
	//{
	//	cloud->push_back(pcl::PointXYZ(result_mesh.position(vit).x, result_mesh.position(vit).y, result_mesh.position(vit).z));
	//}	
	for (auto vit : mesh.vertices())
	{
		cloud->push_back(pcl::PointXYZ(mesh.position(vit).x, mesh.position(vit).y, mesh.position(vit).z));
	}

	pcl::PolygonMesh::Ptr polygon_ptr(new pcl::PolygonMesh);
	toPCLPointCloud2(*cloud, polygon_ptr->cloud);

	std::vector<pcl::Vertices> polygon;
	for (auto fit : result_mesh.faces())
	{
		pcl::Vertices vt;
		for (auto fvit : result_mesh.vertices(fit))
			vt.vertices.push_back(fvit.idx());
		polygon_ptr->polygons.push_back(vt);
	}

	for (size_t i = 0; i < anchor_idx.size(); ++i)
	{
		cloud2->push_back(pcl::PointXYZ(anchor_pos[i].x, anchor_pos[i].y, anchor_pos[i].z));
	}

	pcl::visualization::PCLVisualizer viewer("viewer");
	viewer.setBackgroundColor(0, 0, 0);
	//viewer.addCoordinateSystem(1);
	viewer.addPolygonMesh(*polygon_ptr);

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud2_handler(cloud2, 255, 0, 0);
	viewer.addPointCloud(cloud2, cloud2_handler, "cloud2");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "cloud2");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}


	return 0;
}