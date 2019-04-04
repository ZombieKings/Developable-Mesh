#include "Dev_Inter.h"

//#include <pcl/point_types.h>
//#include <pcl/common/transforms.h>
//#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char** argv)
{
	Surface_mesh mesh;
	mesh.read("test.off");
	//std::cout << "vertices: " << mesh.n_vertices() << std::endl;
	//std::cout << "edges: " << mesh.n_edges() << std::endl;
	//std::cout << "faces: " << mesh.n_faces() << std::endl;

	std::vector<unsigned int> anchor_idx{ 0 };
	std::vector<Point> anchor_pos;
	anchor_pos.push_back(Point(0, 0, 6));
	for (int i = 1; i <= 10; ++i)
	{
		anchor_idx.push_back(i * 10 - 1);
		anchor_pos.push_back(mesh.position(Surface_mesh::Vertex(i * 10 - 1)));
	}

	for (int i = 0; i < 10; ++i)
	{
		anchor_idx.push_back(90 + i);
		anchor_pos.push_back(mesh.position(Surface_mesh::Vertex(90 + i)));
	}

	Dev_Inter cDef(mesh, anchor_pos, anchor_idx);

	cDef.SetConditions(0.001, 0.05, 0.01, 0.01, 0.01, 0.15, 0.01);
	cDef.Deformation();
	Surface_mesh result_mesh = cDef.Get_Result();

	//	//-----------Visualizer------------
	//	//将网格导入点云
	//	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	//	for (auto vit : result_mesh.vertices())
	//	{
	//		cloud->push_back(pcl::PointXYZ(result_mesh.position(vit).x, result_mesh.position(vit).y, result_mesh.position(vit).z));
	//	}
	//	pcl::PolygonMesh::Ptr polygon_ptr(new pcl::PolygonMesh);
	//	toPCLPointCloud2(*cloud, polygon_ptr->cloud);
	//
	//	std::vector<pcl::Vertices> polygon;
	//	for (auto fit : result_mesh.faces())
	//	{
	//		pcl::Vertices vt;
	//		for (auto fvit : result_mesh.vertices(fit))
	//			vt.vertices.push_back(fvit.idx());
	//		polygon_ptr->polygons.push_back(vt);
	//	}
	//
	//	pcl::visualization::PCLVisualizer viewer("viewer");
	//	viewer.setBackgroundColor(0, 0, 0);
	//	//viewer.addCoordinateSystem(1);
	//	viewer.addPolygonMesh(*polygon_ptr);
	//
	//	while (!viewer.wasStopped())
	//	{
	//		viewer.spinOnce();
	//	}
	//
	//
	//	return 0;
}