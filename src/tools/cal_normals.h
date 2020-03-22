#ifndef ZOMBIE_CAL_NORMALS_H
#define ZOMBIE_CAL_NORMALS_H

#include <Eigen/Dense>

namespace Zombie
{
	void cal_normal_per_vertex(Eigen::Matrix3Xd& V,
		const Eigen::MatrixXi& F,
		const Eigen::VectorXi& interVidx,
		Eigen::Matrix3Xd& Normals);
	void cal_normal_per_vertex(Eigen::Matrix3Xd& V,
		const Eigen::MatrixXi& F,
		const Eigen::VectorXi& interVidx,
		Eigen::VectorXd& Normals);

	void cal_normal_per_face(Eigen::Matrix3Xd& V,
		const Eigen::MatrixXi& F,
		const Eigen::VectorXi& interVidx,
		Eigen::Matrix3Xd& Normals);
	void cal_normal_per_face(Eigen::Matrix3Xd& V,
		const Eigen::MatrixXi& F,
		const Eigen::VectorXi& interVidx,
		Eigen::VectorXd& Normals);
}

#include "cal_normals.cpp"

#endif 