#include "Dev_Inter.h"

int main(int argc, char** argv)
{
	// instantiate a Surface_mesh object
	Surface_mesh mesh;
	mesh.read("test.off");
	std::cout << "vertices: " << mesh.n_vertices() << std::endl;
	std::cout << "edges: " << mesh.n_edges() << std::endl;
	std::cout << "faces: " << mesh.n_faces() << std::endl;
	return 0;
}