#include "cal_normals.h"

void Zombie::cal_normal_per_vertex(Eigen::Matrix3Xd& V,
	const Eigen::MatrixXi& F,
	const Eigen::VectorXi& interVidx,
	Eigen::Matrix3Xd& Normals)
{
	const int DIM = F.rows();
	Normals.setConstant(DIM, V.cols(), 0);
	for (int f = 0; f < F.cols(); ++f)
	{
		const Eigen::Vector3i& fv = F.col(f);
		const Eigen::Vector3d& p0 = V.col(fv[0]);
		const Eigen::Vector3d& p1 = V.col(fv[1]);
		const Eigen::Vector3d& p2 = V.col(fv[2]);
		const Eigen::Vector3d crosstemp = (p1 - p0).cross(p2 - p0);
		for (size_t vi = 0; vi < DIM; ++vi)
			if (interVidx(fv[vi]) != -1)
				Normals.col(fv[vi]) += crosstemp;
	}

	for (int v = 0; v < Normals.cols(); ++v)
		if (interVidx(v) != -1)
			Normals.col(v).normalize();
}

void Zombie::cal_normal_per_vertex(Eigen::Matrix3Xd& V,
	const Eigen::MatrixXi& F,
	const Eigen::VectorXi& interVidx,
	Eigen::VectorXd& Normals)
{
	const int DIM = F.rows();
	Eigen::Matrix3Xd matTemp;
	cal_normal_per_vertex(V, F, interVidx, matTemp);
	Normals = Eigen::Map<Eigen::VectorXd>(matTemp.data(), V.cols() * DIM, 1);
}

void Zombie::cal_normal_per_face(Eigen::Matrix3Xd& V,
	const Eigen::MatrixXi& F,
	const Eigen::VectorXi& interVidx,
	Eigen::Matrix3Xd& Normals)
{
	const int DIM = F.rows();
	Normals.setConstant(DIM, F.cols(), 0);
	for (int f = 0; f < F.cols(); ++f)
	{
		const Eigen::Vector3i& fv = F.col(f);
		const Eigen::Vector3d& p0 = V.col(fv[0]);
		const Eigen::Vector3d& p1 = V.col(fv[1]);
		const Eigen::Vector3d& p2 = V.col(fv[2]);
		Normals.col(f) = (p1 - p0).cross(p2 - p0).normalized();
	}
}

void Zombie::cal_normal_per_face(Eigen::Matrix3Xd& V,
	const Eigen::MatrixXi& F,
	const Eigen::VectorXi& interVidx,
	Eigen::VectorXd& Normals)
{
	const int DIM = F.rows();
	Eigen::Matrix3Xd matTemp;
	cal_normal_per_face(V, F, interVidx, matTemp);
	Normals = Eigen::Map<Eigen::VectorXd>(matTemp.data(), F.cols() * DIM, 1);
}
