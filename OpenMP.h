// OpenMP.h
#pragma once

#include <string>

// OpenMP implementation of DCT compression
void runOpenMP(const std::string& input_folder,
    const std::string& output_folder,
    const std::string& csv_path,
    int num_threads = 8);