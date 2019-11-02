#ifndef RESIDUAL_FUNC_H
#define RESIDUAL_FUNC_H

#include <stdint.h>
#include <cstdlib>
#include <cassert>
#include <typeinfo>

#include <Eigen/Sparse>

namespace common {
	template <typename T> inline char type2char(void) { return 0; }

	template <> inline char type2char<float>(void) { return 's'; }
	template <> inline char type2char<double>(void) { return 'd'; }
	template <> inline char type2char<int32_t>(void) { return 'i'; }
	template <> inline char type2char<int64_t>(void) { return 'l'; }

	class  func_ctx
	{
	public:
		virtual ~func_ctx() {}
	};

	class  residual_func
	{
	public:
		virtual size_t dim_of_x(void) const = 0;
		virtual size_t dim_of_f(void) const = 0;

		template <typename VAL_TYPE>
		func_ctx* new_ctx(const VAL_TYPE* x) {
			assert(get_value_type() == type2char<VAL_TYPE>());
			return new_ctx(reinterpret_cast<const void*>(x));
		}

		template <typename VAL_TYPE>
		int val(const VAL_TYPE* x, VAL_TYPE* f, func_ctx* ctx = 0) {
			assert(get_value_type() == type2char<VAL_TYPE>());
			return val(reinterpret_cast<const void*>(x), reinterpret_cast<void*>(f), ctx);
		}

		template <typename VAL_TYPE>
		int jac(const VAL_TYPE* x, Eigen::SparseMatrix<double>& Jt, func_ctx* ctx = 0) {
			assert(get_value_type() == type2char<VAL_TYPE>());
			return jac(reinterpret_cast<const void*>(x), Jt, ctx);
		}

		virtual size_t jac_nnz(void) const {
			return dim_of_x() * dim_of_f();
		}

		virtual char get_value_type(void) const = 0;
		virtual char get_int_type(void) const = 0;

		virtual ~residual_func() {}
		//protected:
		virtual func_ctx* new_ctx(const void* x) { return 0; }
		virtual int val(const void* x, void* f, func_ctx* ctx = 0) = 0;
		//! @brief a |x|*|f| csc matrix with dim_of_nz_x()*|f| nnz
		virtual int jac(const void* x, Eigen::SparseMatrix<double>& Jt, func_ctx* ctx = 0) = 0;

	protected:
		std::vector<double*>  cache_val_ptr_;
	};

	//! generic function with known value and int type
	template <typename VAL_TYPE, typename INT_TYPE>
	class  residual_func_t : public residual_func
	{
	public:
		typedef VAL_TYPE value_type;
		typedef INT_TYPE int_type;

		virtual func_ctx* new_ctx(const value_type* x) { return 0; }
		virtual int val(const value_type* x, value_type* f, func_ctx* ctx = 0) = 0;
		virtual int jac(const value_type* x, Eigen::SparseMatrix<double>& Jt, func_ctx* ctx = 0) = 0;

		virtual char get_value_type(void) const {
			return type2char<value_type>();
		}
		virtual char get_int_type(void) const {
			return type2char<int_type>();
		}
	protected:
		virtual func_ctx* new_ctx(const void* x) {
			return new_ctx(reinterpret_cast<const value_type*>(x));
		}
		virtual int val(const void* x, void* f, func_ctx* ctx = 0) {
			return val(reinterpret_cast<const value_type*>(x), reinterpret_cast<value_type*>(f), ctx);
		}
		virtual int jac(const void* x, Eigen::SparseMatrix<double>& Jt, func_ctx* ctx = 0) {
			return jac(reinterpret_cast<const value_type*>(x), Jt, ctx);
		}
	};
}

#endif
