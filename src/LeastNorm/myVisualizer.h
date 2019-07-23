#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>

#include <vtkSmartPointer.h>
#include <vtkProperty.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkTriangle.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>

#include <vtkOBJReader.h>
#include <vtkPLYReader.h>
#include <vtkPolyDataReader.h>
#include <vtksys/SystemTools.hxx>

#include <vtkLookupTable.h>
#include <vtkColorTransferFunction.h>

#include <Eigen/Core>

#include <surface_mesh/Surface_mesh.h>

#include <string>

class myVisualizer
{
public:
	myVisualizer();
	~myVisualizer() {};
private:
	//user datas
	vtkSmartPointer<vtkPolyData> polydata_;

public:
	int LoadFile(const std::string& fileName);
	int LoadMesh(const surface_mesh::Surface_mesh& mesh);
	int LoadMatrix(const Eigen::Matrix3Xf &V, const Eigen::Matrix3Xi &F);

	int InputColor(const Eigen::VectorXd &Color);

	int Initialize();

	int Run();
private:
	//Render related
	vtkSmartPointer<vtkRenderWindow> window_;
	vtkSmartPointer<vtkRenderer> renderer_;
	vtkSmartPointer<vtkRenderWindowInteractor> interactor_;

	//Color related
	bool color_flag_ = false;
	vtkSmartPointer<vtkLookupTable> lut_;
	vtkSmartPointer<vtkColorTransferFunction> ctf_;
	vtkSmartPointer<vtkDoubleArray> scalar_;
};