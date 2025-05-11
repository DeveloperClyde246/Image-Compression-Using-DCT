// main.cpp
#include "Serial.h"
#include "Cuda.h"
#include "OpenCL.h"
#include "m_pi.h"
#include "OpenMP.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string input_folder = (argc > 1 ? argv[1] : "./images");
    int num_threads = (argc > 4 ? std::stoi(argv[4]) : 8); // Changed to use argv[4] for thread count

    // 1) Run serial version
    std::cout << "==================Serial compression==============\n";
    std::string serial_out = "./compressed_images_serial";
    std::string serial_csv = "serial_results.csv";
    runSerial(input_folder, serial_out, serial_csv);
    std::cout << "Serial compression complete.\n"
        << "  Images ¡ú " << serial_out << "\n"
        << "  Details ¡ú " << serial_csv << "\n\n";

    // 2) Run CUDA version
    std::cout << "==================CUDA compression==============\n";
    std::string cuda_out = "./compressed_images_cuda";
    std::string cuda_csv = "cuda_results.csv";
    runCuda(input_folder, cuda_out, cuda_csv);
    std::cout << "CUDA compression complete.\n"
        << "  Images ¡ú " << cuda_out << "\n"
        << "  Details ¡ú " << cuda_csv << "\n";


    // 3) Run MPI version
    std::cout << "==================MPI compression==============\n";
    std::string mpi_out = "./compressed_images_mpi";
    std::string mpi_csv = "mpi_results.csv";
    runMPI(input_folder, mpi_out, mpi_csv);
    std::cout << "Serial compression complete.\n"
        << "  Images ¡ú " << mpi_out << "\n"
        << "  Details ¡ú " << mpi_csv << "\n\n";


    // 2) Run OpenCL version
    std::cout << "==================OpenCL compression==============\n";
    std::string opencl_out = "./compressed_images_opencl";
    std::string opencl_csv = "opencl_results.csv";
    runOpenCL(input_folder, opencl_out, opencl_csv);
    std::cout << "OpenCL compression complete.\n"
        << "  Images ¡ú " << opencl_out << "\n"
        << "  Details ¡ú " << opencl_csv << "\n\n";

    // 3) Run OpenMP version
    std::cout << "==================OpenMP compression==============\n";
    std::string openmp_out = "./compressed_images_openmp";
    std::string openmp_csv = "openmp_results.csv";
    runOpenMP(input_folder, openmp_out, openmp_csv, num_threads);
    std::cout << "OpenMP compression complete.\n"
        << " Images ¡ú " << openmp_out << "\n"
        << " Details ¡ú " << openmp_csv << "\n\n";

    return 0;
}
