#include <iostream>
#include <fstream>
#include "parallel_mt.hh"

using namespace autoreg;

int main() {
    int n_config = 8, seed = 0;

    std::ofstream file("init_data_config");
    parallel_mt_seq<> initializer(seed);

    for (int i = 0; i < n_config; ++i){
        mt_config config = initializer();
        file << config;
    }
}