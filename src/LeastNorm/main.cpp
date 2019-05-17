#include "least_norm.h"


int main(int argc, char** argv)
{
	Dev_LN cD;
	if (!cD.Load_Mesh("1.off"))
	{
		return 0;
	}
	cD.Deformation();
	Surface_mesh result_mesh(cD.Get_Result());

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

	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);
	//for (auto vit : mesh.vertices())
	//{
	//	if (mesh.is_boundary(vit))
	//	{
	//		cloud2->push_back(pcl::PointXYZ(mesh.position(vit).x, mesh.position(vit).y, mesh.position(vit).z));
	//	}
	//}
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler2(cloud2, 255, 0, 0);
	//viewer.addPointCloud(cloud2, handler2, "cloud2");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud2");



	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}

	return 0;
}