#ifndef VOODOO_HH
#define VOODOO_HH

#include <assert.h>             // for assert
#include <cstdlib>              // for abs
#include <blitz/array.h>        // for Array, Range, shape, any
#include "types.hh"             // for Array2D, ACF

namespace autoreg {

	template<class T>
	Array2D<T>
	AC_matrix_block(const ACF<T>& acf, int i0, int j0) {
		const int n = acf.extent(2);
		Array2D<T> block(blitz::shape(n, n));
		for (int i=0; i<n; ++i) {
			for (int j=0; j<n; ++j) {
				block(i, j) = acf(i0, j0, std::abs(i-j));
			}
		}
		return block;
	}

	template<class T>
	void
	append_column_block(Array2D<T>& lhs, const Array2D<T>& rhs) {
		if (lhs.numElements() == 0) {
			lhs.resize(rhs.shape());
			lhs = rhs;
		} else {
			using blitz::Range;
			assert(lhs.rows() == rhs.rows());
			const int old_cols = lhs.columns();
			lhs.resizeAndPreserve(lhs.rows(), old_cols + rhs.columns());
			lhs(Range::all(), Range(old_cols, blitz::toEnd)) = rhs;
		}
	}

	template<class T>
	void
	append_row_block(Array2D<T>& lhs, const Array2D<T>& rhs) {
		if (lhs.numElements() == 0) {
			lhs.resize(rhs.shape());
			lhs = rhs;
		} else {
			using blitz::Range;
			assert(lhs.columns() == rhs.columns());
			const int old_rows = lhs.rows();
			lhs.resizeAndPreserve(old_rows + rhs.rows(), lhs.columns());
			lhs(Range(old_rows, blitz::toEnd), Range::all()) = rhs;
		}
	}

	template<class T>
	Array2D<T>
	AC_matrix_block(const ACF<T>& acf, int i0) {
		const int n = acf.extent(1);
		Array2D<T> result;
		for (int i=0; i<n; ++i) {
			Array2D<T> row;
			for (int j=0; j<n; ++j) {
				append_column_block(row, AC_matrix_block(acf, i0, std::abs(i-j)));
			}
			append_row_block(result, row);
		}
		return result;
	}

	template<class T>
	Array2D<T>
	generate_AC_matrix(const ACF<T>& acf) {
		const int n = acf.extent(0);
		Array2D<T> result;
		for (int i=0; i<n; ++i) {
			Array2D<T> row;
			for (int j=0; j<n; ++j) {
				append_column_block(row, AC_matrix_block(acf, std::abs(i-j)));
			}
			append_row_block(result, row);
		}
		return result;
	}

}

#endif // VOODOO_HH
