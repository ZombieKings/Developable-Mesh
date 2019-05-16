#include "Dev_Creator.h"


int main(int argc, char** argv)
{
	Dev_Creator DC;
	DC.Read_File("data.txt");
	DC.CreatMesh(20);
	DC.Deformation();
	Surface_mesh result_mesh(DC.Get_Result());

	//-----------Visualizer------------
	//将网格导入点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	for (auto vit : result_mesh.vertices())
	{
		cloud->push_back(pcl::PointXYZ(result_mesh.position(vit).x, result_mesh.position(vit).y, result_mesh.position(vit).z));
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

	pcl::visualization::PCLVisualizer viewer("viewer");
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addPolygonMesh(*polygon_ptr);

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}

	return 0;
}