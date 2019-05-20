#ifndef AUTOREG_HH
#define AUTOREG_HH

#include <algorithm>             // for min, any_of, copy_n, for_each, generate
#include <cassert>               // for assert
#include <chrono>                // for duration, steady_clock, steady_clock...
#include <cmath>                 // for isnan
#include <cstdlib>               // for abs
#include <functional>            // for bind
#include <iostream>              // for operator<<, cerr, endl
#include <fstream>               // for ofstream
#include <random>                // for mt19937, normal_distribution
#include <stdexcept>             // for runtime_error
#include <vector>
#include <thread>
#include <mutex>
#include <list>

#include <blitz/array.h>         // for Array, Range, shape, any

#include "sysv.hh"               // for sysv
#include "types.hh"              // for size3, ACF, AR_coefs, Zeta, Array2D
#include "voodoo.hh"             // for generate_AC_matrix
#include "parallel_mt.hh"


/// @file
/// File with subroutines for AR model, Yule-Walker equations
/// and some others.

namespace autoreg {

	template<class T>
	ACF<T>
	approx_acf(T alpha, T beta, T gamm, const Vec3<T>& delta, const size3& acf_size) {
		ACF<T> acf(acf_size);
		blitz::firstIndex t;
		blitz::secondIndex x;
		blitz::thirdIndex y;
		acf = gamm
			* blitz::exp(-alpha * (t*delta[0] + x*delta[1] + y*delta[2]))
//	 		* blitz::cos(beta * (t*delta[0] + x*delta[1] + y*delta[2]));
	 		* blitz::cos(beta * t * delta[0])
	 		* blitz::cos(beta * x * delta[1])
	 		* blitz::cos(beta * y * delta[2]);
		return acf;
	}

	template<class T>
	T white_noise_variance(const AR_coefs<T>& ar_coefs, const ACF<T>& acf) {
		return acf(0,0,0) - blitz::sum(ar_coefs * acf);
	}

	template<class T>
	T ACF_variance(const ACF<T>& acf) {
		return acf(0,0,0);
	}

	/// Удаление участков разгона из реализации.
	template<class T>
	Zeta<T>
	trim_zeta(const Zeta<T>& zeta2, const size3& zsize) {
		using blitz::Range;
		using blitz::toEnd;
		size3 zsize2 = zeta2.shape();
		return zeta2(
			Range(zsize2(0) - zsize(0), toEnd),
			Range(zsize2(1) - zsize(1), toEnd),
			Range(zsize2(2) - zsize(2), toEnd)
		);
	}

	template<class T>
	bool is_stationary(AR_coefs<T>& phi) {
		return !blitz::any(blitz::abs(phi) > T(1));
	}

	template<class T>
	AR_coefs<T>
	compute_AR_coefs(const ACF<T>& acf) {
		using blitz::Range;
		using blitz::toEnd;
		const int m = acf.numElements()-1;
		Array2D<T> acm = generate_AC_matrix(acf);
		//{ std::ofstream out("acm"); out << acm; }

		/**
		eliminate the first equation and move the first column of the remaining
		matrix to the right-hand side of the system
		*/
		Array1D<T> rhs(m);
		rhs = acm(Range(1, toEnd), 0);
		//{ std::ofstream out("rhs"); out << rhs; }

		// lhs is the autocovariance matrix without first
		// column and row
		Array2D<T> lhs(blitz::shape(m,m));
		lhs = acm(Range(1, toEnd), Range(1, toEnd));
		//{ std::ofstream out("lhs"); out << lhs; }

		assert(lhs.extent(0) == m);
		assert(lhs.extent(1) == m);
		assert(rhs.extent(0) == m);
		sysv<T>('U', m, 1, lhs.data(), m, rhs.data(), m);
		AR_coefs<T> phi(acf.shape());
		assert(phi.numElements() == rhs.numElements() + 1);
		phi(0,0,0) = 0;
		std::copy_n(rhs.data(), rhs.numElements(), phi.data()+1);
		//{ std::ofstream out("ar_coefs"); out << phi; }
		if (!is_stationary(phi)) {
			std::cerr << "phi.shape() = " << phi.shape() << std::endl;
			std::for_each(
				phi.begin(),
				phi.end(),
				[] (T val) {
					if (std::abs(val) > T(1)) {
						std::cerr << val << std::endl;
					}
				}
			);
			throw std::runtime_error("AR process is not stationary, i.e. |phi| > 1");
		}
		return phi;
	}

	template<class T>
	bool
	isnan(T rhs) noexcept {
		return std::isnan(rhs);
	}

	/// Генерация белого шума по алгоритму Вихря Мерсенна и
	/// преобразование его к нормальному распределению по алгоритму Бокса-Мюллера.
	template<class T>
	Zeta<T>
	generate_white_noise(const size3& size, const T variance) {
		if (variance < T(0)) {
			throw std::runtime_error("variance is less than zero");
		}

		int n = 8;
        std::ifstream file("init_data_config");
        std::vector<parallel_mt> generators;
                        
        for (int i = 0; i < n; i++) {
            mt_config config;
            file >> config;
            generators.push_back(parallel_mt(config));
        }
   		
        std::normal_distribution<T> normal(T(0), std::sqrt(variance));
        
		Zeta<T> eps(size);
        
        std::vector<std::thread> threads;
        auto cur_begin = std::begin(eps);
        int step = size(0) * size(1) * size(2) / n;
       
        for (int i = 0; i < n; i++) {
            auto cur_end = std::next(cur_begin, step);
            auto cur_gen = std::bind(normal, generators[i]);
            std::thread cur_thread(std::generate<decltype(cur_begin), decltype(cur_gen)>, cur_begin, 
            	 		           cur_end, cur_gen);
					
            threads.push_back(std::move(cur_thread));
			cur_begin = cur_end;
        }
        auto cur_end = std::end(eps);
        auto cur_gen = std::bind(normal, generators[n - 1]);
        std::thread cur_thread(std::generate<decltype(cur_begin), decltype(cur_gen)>, cur_begin, 
        					   cur_end, cur_gen);
        threads.push_back(std::move(cur_thread));
		
		for(auto& cur_thread : threads){
			cur_thread.join();
		}

		if (std::any_of(std::begin(eps), std::end(eps), &::autoreg::isnan<T>)) {
			throw std::runtime_error("white noise generator produced some NaNs");
		}
		return eps;
	}

	struct ZetaBlock {
        int t_begin;
        int x_begin;
        int y_begin;

        int t_end;
        int x_end;
        int y_end;

        int x_id;
        int y_id;
        int t_id;
    };

    bool is_available(ZetaBlock &block, std::mutex &completed_blocks_mutex, 
    				  blitz::Array<bool, 3> &completed_blocks) {
        int t_id = block.t_id;
        int x_id = block.x_id;
        int y_id = block.y_id;

        int t_id_prev = block.t_id - 1;
        int x_id_prev = block.x_id - 1;
        int y_id_prev = block.y_id - 1;

        std::lock_guard <std::mutex> lock(completed_blocks_mutex);


        if (t_id_prev >= 0 && !completed_blocks(t_id_prev, x_id, y_id)) {
            return false;
        }

        if (x_id_prev >= 0 && !completed_blocks(t_id, x_id_prev, y_id)) {
            return false;
        }

        if (y_id_prev >= 0 && !completed_blocks(t_id, x_id, y_id_prev)) {
            return false;
        }

        if (t_id_prev >= 0 && x_id_prev >= 0 && !completed_blocks(t_id_prev, x_id_prev, y_id)) {
            return false;
        }

        if (t_id_prev >= 0 && y_id_prev >= 0 && !completed_blocks(t_id_prev, x_id, y_id_prev)) {
            return false;
        }

        if (x_id_prev >= 0 && y_id_prev >= 0 && !completed_blocks(t_id, x_id_prev, y_id_prev)) {
            return false;
        }

        if (t_id_prev >= 0 && x_id_prev >= 0 && y_id_prev >= 0 && !completed_blocks(t_id_prev, x_id_prev, y_id_prev)) {
            return false;
        }

        return true;
    }


	template <class T>
	void generate_zeta_block(const AR_coefs<T>& phi, Zeta<T>& zeta,
							 std::mutex &list_of_blocks_mutex, std::list<ZetaBlock> &list_of_blocks, 
							 std::mutex &completed_blocks_mutex, blitz::Array<bool, 3> &completed_blocks){
		const size3 fsize = phi.shape();
		const size3 zsize = zeta.shape();

		while (true) {
			ZetaBlock block;

			bool found_block;

			// поиск блока для генерации поверхности
			list_of_blocks_mutex.lock();
	        found_block = false;
	        for (auto current = list_of_blocks.begin(); current != list_of_blocks.end(); ++current) {
	            if (is_available(*current, completed_blocks_mutex, completed_blocks)) {
	                block = *current;
	                list_of_blocks.erase(current);
	                found_block = true;
	                break;
	            }
	        }
	        list_of_blocks_mutex.unlock();
        	
	        // генерация блока поверхности
			if (!found_block)
			{
				std::lock_guard<std::mutex> lock(list_of_blocks_mutex);
				if (list_of_blocks.size() == 0)
					break;
			} 
			else 
			{
				for (int t = block.t_begin; t < block.t_end; ++t) {
					for (int x = block.x_begin; x < block.x_end; ++x) {
						for (int y = block.y_begin; y < block.y_end; ++y) {
							const int m1 = std::min(t+1, fsize[0]);
							const int m2 = std::min(x+1, fsize[1]);
							const int m3 = std::min(y+1, fsize[2]);
							T sum = 0;
							for (int k=0; k<m1; k++)
								for (int i=0; i<m2; i++)
									for (int j=0; j<m3; j++)
										sum += phi(k, i, j)*zeta(t-k, x-i, y-j);
							zeta(t, x, y) += sum;
						}
					}
				}

				std::lock_guard <std::mutex> lock(completed_blocks_mutex);
        		completed_blocks(block.t_id, block.x_id, block.y_id) = true;
			}
		}
	}

	/// Генерация отдельных частей реализации волновой поверхности.
	template<class T>
	void generate_zeta(const AR_coefs<T>& phi, Zeta<T>& zeta) {
		const size3 fsize = phi.shape();
		const size3 zsize = zeta.shape();
		const int t1 = zsize[0];
		const int x1 = zsize[1];
		const int y1 = zsize[2];

		int t_size = 20;
		int x_size = 4;
		int y_size = 4;

		const int t_step = std::max(fsize[0], zsize[0]/t_size);
		const int x_step = std::max(fsize[1], zsize[1]/x_size);
		const int y_step = std::max(fsize[2], zsize[2]/y_size);

		t_size = ceil(double(t1) / t_step);
        x_size = ceil(double(x1) / x_step);
        y_size = ceil(double(y1) / y_step);

    	std::clog<<"Block size: "<< "("<< t_step<<", "<< x_step <<", "<<y_step << ")" << std::endl;
		std::clog<<"Number of blocks: "<< ceil(zsize[0] * zsize[1] * zsize[2] / t_step / x_step / y_step) << std::endl;

	  	blitz::Array<bool, 3> completed_blocks;
        std::list <ZetaBlock> list_of_blocks;

        std::mutex list_of_blocks_mutex;
        std::mutex completed_blocks_mutex;

        completed_blocks.resize(t_size, x_size, y_size);
        completed_blocks(blitz::Range::all(), blitz::Range::all(), blitz::Range::all()) = false;

        for (int current_t_begin = 0; current_t_begin < t1; current_t_begin += t_step) {
    		int current_t_end;
    		if ((current_t_begin + t_step) <= t1)
        		current_t_end = current_t_begin + t_step;
    		else
        		current_t_end = t1;

    		for (int current_x_begin = 0; current_x_begin < x1; current_x_begin += x_step) {
       			int current_x_end;
        		if ((current_x_begin + x_step) <= x1)
            		current_x_end = current_x_begin + x_step;
        		else
            		current_x_end = x1;

        		for (int current_y_begin = 0; current_y_begin < y1; current_y_begin += y_step) {
            		int current_y_end;
            		if ((current_y_begin + y_step) <= y1)
                		current_y_end = current_y_begin + y_step;
            		else
                		current_y_end = y1;

		            ZetaBlock block;
		            block.t_begin = current_t_begin;
		            block.t_end = current_t_end;
		            block.x_begin = current_x_begin;
		            
		            block.x_end = current_x_end;
		            block.y_begin = current_y_begin;
		            block.y_end = current_y_end;
		            
		            block.t_id = current_t_begin / t_step;
		            block.x_id = current_x_begin / x_step;
		            block.y_id = current_y_begin / y_step;

            		list_of_blocks.push_back(block);
        		}
    		}
		}

		int n_threads = std::max(atoi(std::getenv("OMP_NUM_THREADS")), 1);	
		std::clog << "n_threads: " << n_threads << std::endl;
		
		std::vector<std::thread> threads;

		for(int i = 0; i < n_threads; ++i){
			std::thread current_thread(generate_zeta_block<T>, std::ref(phi), std::ref(zeta),
									   std::ref(list_of_blocks_mutex), std::ref(list_of_blocks), 
									   std::ref(completed_blocks_mutex), std::ref(completed_blocks));
			threads.push_back(std::move(current_thread));
		}

		for(std::thread& cur_thread : threads){
			cur_thread.join();
		}

	}

	template<class T, int N>
	T mean(const blitz::Array<T,N>& rhs) {
		return blitz::sum(rhs) / rhs.numElements();
	}

	template<class T, int N>
	T variance(const blitz::Array<T,N>& rhs) {
		assert(rhs.numElements() > 0);
		const T m = mean(rhs);
		return blitz::sum(blitz::pow(rhs-m, 2)) / (rhs.numElements() - 1);
	}

}

#endif // AUTOREG_HH
