#ifndef SYSV_HH
#define SYSV_HH

#include <sstream>    // for operator<<, basic_ostream::operator<<, basic_os...
#include <stdexcept>  // for invalid_argument
#include <valarray>   // for valarray

/// @file
/// C/C++ interface to ``sysv'' LAPACK routine.

extern "C" void ssysv_(char*, int*, int*, float*, int*, int*, float*, int*, float*, int*, int*);
extern "C" void dsysv_(char*, int*, int*, double*, int*, int*, double*, int*, double*, int*, int*);

template<class T>
void sysv(char type, int m, int nrhs, T* a, int lda, T* b, int ldb);


void
check_info(int info) {
	if (info != 0) {
		std::stringstream s;
		s << "sysv error, A(" << info << ", " << info << ")=0";
		throw std::invalid_argument(s.str());
	}
}

template<>
void sysv<float>(char type, int m, int nrhs, float* a, int lda, float* b, int ldb) {
	int info = 0;
	int lwork = m;
	std::valarray<float> work(lwork);
	std::valarray<int> ipiv(m);
	ssysv_(&type, &m, &nrhs, a, &lda, &ipiv[0], b, &ldb, &work[0], &lwork, &info);
	check_info(info);
}

template<>
void sysv<double>(char type, int m, int nrhs, double* a, int lda, double* b, int ldb) {
	int info = 0;
	int lwork = m;
	std::valarray<double> work(lwork);
	std::valarray<int> ipiv(m);
	dsysv_(&type, &m, &nrhs, a, &lda, &ipiv[0], b, &ldb, &work[0], &lwork, &info);
	check_info(info);
}

#endif // SYSV_HH
