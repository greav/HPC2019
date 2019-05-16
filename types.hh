#ifndef VECTOR_N_HH
#define VECTOR_N_HH

#include <blitz/array.h>

/**
@file domain-specific types and templates
*/

namespace autoreg {

	template<class T, size_t N>
	using Vector = blitz::TinyVector<T, N>;

	typedef Vector<int, 3> size3;
	typedef Vector<int, 2> size2;
	typedef Vector<int, 1> size1;

	template<class T> using Vec3 = Vector<T, 3>;
	template<class T> using Vec2 = Vector<T, 2>;
	template<class T> using Vec1 = Vector<T, 1>;

	template<class T> using ACF = blitz::Array<T,3>;
	template<class T> using AR_coefs = blitz::Array<T,3>;
	template<class T> using Zeta = blitz::Array<T,3>;
	template<class T> using Array2D = blitz::Array<T,2>;
	template<class T> using Array1D = blitz::Array<T,1>;

}

#endif // VECTOR_N_HH
