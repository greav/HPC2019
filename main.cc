#include <fstream>

#include "autoreg_driver.hh"

int main() {

	using namespace autoreg;

	/// floating point type (float, double, long double or multiprecision number C++ class)
	typedef float Real;

	/// input file with various model parameters
	const char* input_filename = "autoreg.model";
	Autoreg_model<Real> model;
	std::ifstream cfg(input_filename);
	cfg >> model;
	model.act();
	return 0;
}
