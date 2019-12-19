#include "mosek.h" /* Include the MOSEK definition file. */

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"

class mosek_solver
{
private:
	int numCon_;   /* Number of constraints.             */
	int numVar_;   /* Number of variables.               */
	int numANZ_;   /* Number of non-zeros in A.           */
	int numLTNZ_;    /* Number of non-zeros in Q.           */

	double cf_;
	double* c_;
	MSKboundkeye* bkc_;
	double* blc_;
	double* buc_;

	MSKboundkeye* bkx_;
	double* blx_;
	double* bux_;

	MSKint32t* aptrb_;
	MSKint32t* aptre_;
	MSKint32t* asub_;
	double* aval_;

	MSKint32t* qsubi_;
	MSKint32t* qsubj_;
	double* qval_;

	double* xx_;
private:
	static void MSKAPI printstr(void* handle, const char str[])
	{
		printf("%s", str);
	} /* printstr */

public:
	~mosek_solver()
	{
		free(c_);
		free(bkc_);
		free(blc_);
		free(buc_);

		free(bkx_);
		free(blx_);
		free(bux_);

		free(aptrb_);
		free(aptre_);
		free(asub_);
		free(aval_);

		free(qsubi_);
		free(qsubj_);
		free(qval_);

		free(xx_);
	}
	mosek_solver(const Eigen::SparseMatrix<double>& Q,
		const Eigen::SparseMatrix<double>& A,
		const Eigen::VectorXd& cl,
		const Eigen::VectorXd& b,
		const int Vnum,
		const double cf) :numVar_(Vnum), cf_(cf)
	{
		numCon_ = b.size();   /* Number of constraints.             */
		numANZ_ = A.nonZeros();   /* Number of non-zeros in A.           */

		//变量取值的上下界
		bkx_ = (MSKboundkeye*)malloc(numVar_ * sizeof(MSKboundkeye));
		blx_ = (double*)malloc(numVar_ * sizeof(double));
		bux_ = (double*)malloc(numVar_ * sizeof(double));
		for (size_t i = 0; i < numVar_; ++i)
		{
			bkx_[i] = MSK_BK_FR;
			blx_[i] = -MSK_INFINITY;
			bux_[i] = +MSK_INFINITY;
		}

		//二次项系数矩阵的下三角部分
		Eigen::SparseMatrix<double> LT = Q.triangularView<Eigen::Lower>();
		numLTNZ_ = LT.nonZeros();
		qsubi_ = (MSKint32t*)malloc(numLTNZ_ * sizeof(MSKint32t));
		qsubj_ = (MSKint32t*)malloc(numLTNZ_ * sizeof(MSKint32t));
		qval_ = (double*)malloc(numLTNZ_ * sizeof(double));
		int count2 = 0;
		for (int i = 0; i < LT.outerSize(); ++i)
		{
			for (Eigen::SparseMatrix<double>::InnerIterator it(LT, i); it; ++it)
			{
				qsubi_[count2] = it.row();
				qsubj_[count2] = it.col();
				qval_[count2] = it.value();
				++count2;
			}
		}

		//线性项系数向量
		c_ = (double*)malloc(numVar_ * sizeof(double));
		for (size_t i = 0; i < numVar_; ++i)
		{
			c_[i] = cl[i];
		}

		//约束的上下界
		bkc_ = (MSKboundkeye*)malloc(numCon_ * sizeof(MSKboundkeye));
		blc_ = (double*)malloc(numCon_ * sizeof(double));
		buc_ = (double*)malloc(numCon_ * sizeof(double));
		for (size_t i = 0; i < numCon_; ++i)
		{
			bkc_[i] = MSK_BK_LO;
			blc_[i] = b[i];
			buc_[i] = +MSK_INFINITY;
		}

		//约束项系数矩阵
		aptrb_ = (MSKint32t*)malloc(numVar_ * sizeof(MSKint32t));
		aptre_ = (MSKint32t*)malloc(numVar_ * sizeof(MSKint32t));
		asub_ = (MSKint32t*)malloc(numANZ_ * sizeof(MSKint32t));
		aval_ = (double*)malloc(numANZ_ * sizeof(double));
		int count1 = 0;
		for (int i = 0; i < A.outerSize(); ++i)
		{
			aptrb_[i] = count1;
			for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it)
			{
				asub_[count1] = it.row();
				aval_[count1] = it.value();
				++count1;
			}
			aptre_[i] = count1;
		}
	}

	int solve()
	{
		MSKenv_t      env = NULL;
		MSKtask_t     task = NULL;
		MSKrescodee   r;

		xx_ = (double*)malloc(numVar_ * sizeof(double));
		MSKint32t i, j;

		// Create the mosek environment. 
		r = MSK_makeenv(&env, NULL);

		if (r == MSK_RES_OK)
		{
			// Create the optimization task. 
			r = MSK_maketask(env, numCon_, numVar_, &task);

			if (r == MSK_RES_OK)
			{
				r = MSK_linkfunctotaskstream(task, MSK_STREAM_LOG, NULL, printstr);

				// Append 'NUMCON' empty constraints.
				// The constraints will initially have no bounds. 
				if (r == MSK_RES_OK)
					r = MSK_appendcons(task, numCon_);

				// Append 'NUMVAR' variables.
				// The variables will initially be fixed at zero (x=0). 
				if (r == MSK_RES_OK)
					r = MSK_appendvars(task, numVar_);

				// Optionally add a constant term to the objective.
				if (r == MSK_RES_OK)
					r = MSK_putcfix(task, cf_);
				for (j = 0; j < numVar_ && r == MSK_RES_OK; ++j)
				{
					// Set the linear term c_j in the objective.
					if (r == MSK_RES_OK)
						r = MSK_putcj(task, j, c_[j]);

					// Set the bounds on variable j.
					// blx[j] <= x_j <= bux[j]
					if (r == MSK_RES_OK)
						r = MSK_putvarbound(task,
							j,           /* Index of variable.*/
							bkx_[j],      /* Bound key.*/
							blx_[j],      /* Numerical value of lower bound.*/
							bux_[j]);     /* Numerical value of upper bound.*/

					 /* Input column j of A */
					if (r == MSK_RES_OK)
						r = MSK_putacol(task,
							j,                 /* Variable (column) index.*/
							aptre_[j] - aptrb_[j], /* Number of non-zeros in column j.*/
							asub_ + aptrb_[j],   /* Pointer to row indexes of column j.*/
							aval_ + aptrb_[j]);  /* Pointer to Values of column j.*/

				}

				// Set the bounds on constraints.
				//  for i=1, ...,NUMCON : blc[i] <= constraint i <= buc[i] 
				for (i = 0; i < numCon_ && r == MSK_RES_OK; ++i)
					r = MSK_putconbound(task,
						i,           /* Index of constraint.*/
						bkc_[i],      /* Bound key.*/
						blc_[i],      /* Numerical value of lower bound.*/
						buc_[i]);     /* Numerical value of upper bound.*/

				if (r == MSK_RES_OK)
				{
					// Input the Q for the objective. 
					r = MSK_putqobj(task, numLTNZ_, qsubi_, qsubj_, qval_);
				}

				if (r == MSK_RES_OK)
				{
					MSKrescodee trmcode;

					// Run optimizer 
					r = MSK_optimizetrm(task, &trmcode);

					// Print a summary containing information
					//   about the solution for debugging purposes
					MSK_solutionsummary(task, MSK_STREAM_MSG);

					if (r == MSK_RES_OK)
					{
						MSKsolstae solsta;
						int j;

						MSK_getsolsta(task, MSK_SOL_ITR, &solsta);

						switch (solsta)
						{
						case MSK_SOL_STA_OPTIMAL:
							MSK_getxx(task,
								MSK_SOL_ITR,    /* Request the interior solution. */
								xx_);

							//printf("Optimal primal solution\n");
							//for (j = 0; j < numVar_; ++j)
							//	printf("x[%d]: %e\n", j, xx_[j]);
							break;

						case MSK_SOL_STA_DUAL_INFEAS_CER:
						case MSK_SOL_STA_PRIM_INFEAS_CER:
							printf("Primal or dual infeasibility certificate found.\n");
							break;

						case MSK_SOL_STA_UNKNOWN:
							printf("The status of the solution could not be determined. Termination code: %d.\n", trmcode);
							break;

						default:
							printf("Other solution status.");
							break;
						}
					}
					else
					{
						printf("Error while optimizing.\n");
					}
				}

				if (r != MSK_RES_OK)
				{
					/* In case of an error print error code and description. */
					char symname[MSK_MAX_STR_LEN];
					char desc[MSK_MAX_STR_LEN];

					printf("An error occurred while optimizing.\n");
					MSK_getcodedesc(r,
						symname,
						desc);
					printf("Error %s - '%s'\n", symname, desc);
				}
			}
			MSK_deletetask(&task);
		}
		MSK_deleteenv(&env);
		return (r);
	}

	double* get_result() const
	{
		return xx_;
	}
};
