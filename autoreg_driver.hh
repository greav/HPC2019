#ifndef AUTOREG_DRIVER_HH
#define AUTOREG_DRIVER_HH

#include <iostream>     // for operator<<, basic_ostream, clog
#include <iomanip>      // for operator<<, setw
#include <fstream>      // for ofstream
#include <stdexcept>    // for runtime_error
#include <string>       // for operator==, basic_string, string, getline

#include "types.hh"     // for size3, Vector, Zeta, ACF, AR_coefs
#include "autoreg.hh"   // for mean, variance, ACF_variance, approx_acf, comp...

/// @file
/// Some abbreviations used throughout the programme.
/// AR      autoregressive
/// ACF     auto-covariate function
/// zeta    ocean wavy surface
/// phi     AR model coefficients
/// YW      Yule-Walker
/// WN      white noise
/// var     variance
/// MT      Mersenne Twister (pseudo-random number generator)

namespace autoreg {

/// Class that reads paramters from the input files,
/// calls all subroutines, and prints the result.
template<class T>
struct Autoreg_model {

	Autoreg_model():
	zsize(768, 24, 24),
	zdelta(1, 1, 1),
	acf_size(10, 10, 10),
	acf_delta(zdelta),
	fsize(acf_size),
	zsize2(zsize)
	{}

	void act() {
		echo_parameters();
		ACF<T> acf_model = approx_acf<T>(alpha, beta, gamm, acf_delta, acf_size);
		//{ std::ofstream out("acf"); out << acf_model; }
		AR_coefs<T> ar_coefs = compute_AR_coefs(acf_model);
		T var_wn = white_noise_variance(ar_coefs, acf_model);
		std::clog << "ACF variance = " << ACF_variance(acf_model) << std::endl;
		std::clog << "WN variance = " << var_wn << std::endl;
		Zeta<T> zeta2 = generate_white_noise(zsize2, var_wn);
		std::clog << "mean(eps) = " << mean(zeta2) << std::endl;
		std::clog << "variance(eps) = " << variance(zeta2) << std::endl;
		generate_zeta(ar_coefs, zeta2);
		std::clog << "mean(zeta) = " << mean(zeta2) << std::endl;
		std::clog << "variance(zeta) = " << variance(zeta2) << std::endl;
		Zeta<T> zeta = trim_zeta(zeta2, zsize);
		write_zeta(zeta);
	}

	/// Read AR model parameters from an input stream, generate default ACF and
	/// validate all the parameters.
	template<class V>
	friend std::istream&
	operator>>(std::istream& in, Autoreg_model<V>& m) {
		m.read_parameters(in);
		m.validate_parameters();
		return in;
	}

private:

	T size_factor() const { return T(zsize2[0]) / T(zsize[0]); }

	/// Read AR model parameters from an input stream.
	void
	read_parameters(std::istream& in) {
		std::string name;
		T size_factor = 1.2;
		while (std::getline(in, name, '=')) {
			if (name.size() > 0 && name[0] == '#') in.ignore(1024*1024, '\n');
			else if (name == "zsize"       ) in >> zsize;
			else if (name == "zdelta"      ) in >> zdelta;
			else if (name == "acf_size"    ) in >> acf_size;
			else if (name == "size_factor" ) in >> size_factor;
			else if (name == "alpha"       ) in >> alpha;
			else if (name == "beta"        ) in >> beta;
			else if (name == "gamma"       ) in >> gamm;
			else {
				in.ignore(1024*1024, '\n');
				std::stringstream str;
				str << "Unknown parameter: " << name << '.';
				throw std::runtime_error(str.str().c_str());
			}
			in >> std::ws;
		}

		if (size_factor < T(1)) {
			std::stringstream str;
			str << "Invalid size factor: " << size_factor;
			throw std::runtime_error(str.str().c_str());
		}

		zsize2 = size3(zsize*size_factor);
		acf_delta = zdelta;
		fsize = acf_size;
	}

	/// Check for common input/logical errors and numerical implementation constraints.
	void validate_parameters() {
		check_non_zero(zsize, "zsize");
		check_non_zero(zdelta, "zdelta");
		check_non_zero(acf_size, "acf_size");
		for (int i=0; i<3; ++i) {
			if (zsize2[i] < zsize[i]) {
				throw std::runtime_error("size_factor < 1, zsize2 < zsize");
			}
		}
		int part_sz = zsize[0];
		int fsize_t = fsize[0];
		if (fsize_t > part_sz) {
			std::stringstream tmp;
			tmp << "fsize[0] > zsize[0], should be 0 < fsize[0] < zsize[0]\n";
			tmp << "fsize[0]  = " << fsize_t << '\n';
			tmp << "zsize[0] = " << part_sz << '\n';
			throw std::runtime_error(tmp.str());
		}
	}

	/// Check that all components of vector @sz are non-zero,
	/// i.e. it is valid size specification.
	template<class V>
	void check_non_zero(const Vector<V, 3>& sz, const char* var_name) {
		if (blitz::product(sz) == 0) {
			std::stringstream str;
			str << "Invalid " << var_name << ": " << sz;
			throw std::runtime_error(str.str().c_str());
		}
	}

	void
	echo_parameters() {
		std::clog << std::left;
		write_key_value(std::clog, "acf_size:"   , acf_size);
		write_key_value(std::clog, "zsize:"      , zsize);
		write_key_value(std::clog, "zsize2:"     , zsize2);
		write_key_value(std::clog, "zdelta:"     , zdelta);
		write_key_value(std::clog, "size_factor:", size_factor());
	}

	template<class V>
	std::ostream&
	write_key_value(std::ostream& out, const char* key, V value) {
		return out << std::setw(20) << key << value << std::endl;
	}

	void write_zeta(const Zeta<T>& zeta) {
		std::ofstream out("zeta");
		out << zeta;
	}

	/// Wavy surface size.
	size3 zsize;

	/// Wavy surface grid granularity.
	Vector<T, 3> zdelta;

	/// Auto-covariate function size.
	size3 acf_size;

	/// Auto-covariate function grid granularity.
	Vector<T, 3> acf_delta;

	/// Size of the array of AR coefficients.
	size3 fsize;

	/// Size of enlarged wavy surface. Equals to @zsize multiplied
	/// by size_factor read from input file.
	size3 zsize2;

	/// ACF parameters
	/// @see approx_acf
	T alpha = 0.06;
	T beta = 0.8;
	T gamm = 1.0;

};

}

#endif // AUTOREG_DRIVER_HH
