#include "least_norm.h"

#include "myVisualizer.h"

int main(int argc, char** argv)
{
	Dev_LN cD;
	if (!cD.Load_Mesh("2.off"))
	{
		return 0;
	}
	cD.Deformation();
	Surface_mesh result_mesh(cD.Get_Result());

	myVisualizer mv;
	mv.LoadMesh(result_mesh);
	mv.Run();

	return 0;
}