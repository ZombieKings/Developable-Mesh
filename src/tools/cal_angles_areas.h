#ifndef ZOMBIE_CAL_ANGLES_AREAS_H
#define ZOMBIE_CAL_ANGLES_AREAS_H

#include <Eigen/Dense>

namespace Zombie
{
	void cal_angles(const Eigen::Matrix3Xd& V,
		const Eigen::MatrixXi& F,
		Eigen::VectorXd& vecAngles);

	void cal_angles(const Eigen::Matrix3Xd& V,
		const Eigen::MatrixXi& F,
		const Eigen::VectorXi& Vtype,
		Eigen::VectorXd& vecAngles,
		Eigen::Matrix3Xd& matAngles);

	void cal_angles_and_areas(const Eigen::Matrix3Xd& V,
		const Eigen::MatrixXi& F,
		const Eigen::VectorXi& Vtype,
		Eigen::VectorXd& vecAngles,
		Eigen::VectorXd& areas,
		Eigen::Matrix3Xd& matAngles);

	void cal_angles_and_areas_with_edges(int Vnum,
		const Eigen::Matrix3Xi& F,
		const Eigen::Matrix3Xi& FE, 
		const Eigen::VectorXi& Vtype,
		Eigen::VectorXd& vecLength,
		Eigen::VectorXd& vecAreas,
		Eigen::VectorXd& vecAngles,
		Eigen::Matrix3Xd& matAngles);
}

#include "cal_angles_areas.cpp"

#endif 