#define _USE_MATH_DEFINES // Enable M_PI in <cmath> for MSVC
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <map>
#include <sstream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <type_traits>
#include <filesystem>
#include <CL/cl.h>
#include "OpenCL.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Quantization table for quality 90% (JPEG standard, scaled)
const int cl_quant_table[8][8] = {
    {3, 2, 2, 3, 5, 8, 10, 12},
    {2, 2, 3, 4, 5, 12, 12, 11},
    {3, 3, 3, 5, 8, 11, 14, 11},
    {3, 3, 4, 6, 10, 17, 16, 12},
    {4, 4, 7, 11, 14, 22, 21, 15},
    {5, 7, 11, 13, 16, 21, 23, 18},
    {10, 13, 16, 17, 21, 24, 24, 20},
    {14, 18, 20, 20, 22, 23, 24, 22}
};

// Zigzag ordering for 8x8 block
const int cl_zigzag_order[64] = {
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
};

// OpenCL kernel source for DCT, IDCT, Quantize, and Dequantize (batch processing)
const char* cl_kernel_source = R"(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define M_PI 3.14159265358979323846

__kernel void cl_dct_2d(__global double* cl_all_blocks, __global double* cl_dct_blocks, int num_blocks) {
    int block_id = get_global_id(0);
    int u = get_global_id(1);
    int v = get_global_id(2);

    if (block_id < num_blocks && u < 8 && v < 8) {
        double sum = 0.0;
        double cu = (u == 0) ? 1.0 / sqrt(2.0) : 1.0;
        double cv = (v == 0) ? 1.0 / sqrt(2.0) : 1.0;

        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                int idx = block_id * 64 + x * 8 + y;
                sum += cl_all_blocks[idx] *
                       cos((2 * x + 1) * u * M_PI / 16.0) *
                       cos((2 * y + 1) * v * M_PI / 16.0);
            }
        }

        int out_idx = block_id * 64 + u * 8 + v;
        cl_dct_blocks[out_idx] = 0.25 * cu * cv * sum;
    }
}


__kernel void cl_idct_2d(__global double* cl_all_blocks, __global double* cl_idct_blocks, int num_blocks) {
    int block_id = get_global_id(0);
    int x = get_global_id(1);
    int y = get_global_id(2);

    if (block_id < num_blocks && x < 8 && y < 8) {
        double sum = 0.0;

        for (int u = 0; u < 8; u++) {
            for (int v = 0; v < 8; v++) {
                double cu = (u == 0) ? 1.0 / sqrt(2.0) : 1.0;
                double cv = (v == 0) ? 1.0 / sqrt(2.0) : 1.0;
                int idx = block_id * 64 + u * 8 + v;

                sum += cu * cv * cl_all_blocks[idx] *
                       cos((2 * x + 1) * u * M_PI / 16.0) *
                       cos((2 * y + 1) * v * M_PI / 16.0);
            }
        }

        int out_idx = block_id * 64 + x * 8 + y;
        cl_idct_blocks[out_idx] = 0.25 * sum;
    }
}

__kernel void cl_quantize(__global double* cl_all_blocks, __constant int* cl_quant_table, int num_blocks) {
    int block_id = get_global_id(0);
    int cl_x = get_global_id(1);
    int cl_y = get_global_id(2);
    if (block_id < num_blocks && cl_x < 8 && cl_y < 8) {
        int cl_idx = block_id * 64 + cl_y * 8 + cl_x;
        cl_all_blocks[cl_idx] = round(cl_all_blocks[cl_idx] / cl_quant_table[cl_y * 8 + cl_x]);
    }
}

__kernel void cl_dequantize(__global double* cl_all_blocks, __constant int* cl_quant_table, int num_blocks) {
    int block_id = get_global_id(0);
    int cl_x = get_global_id(1);
    int cl_y = get_global_id(2);
    if (block_id < num_blocks && cl_x < 8 && cl_y < 8) {
        int cl_idx = block_id * 64 + cl_y * 8 + cl_x;
        cl_all_blocks[cl_idx] = cl_all_blocks[cl_idx] * cl_quant_table[cl_y * 8 + cl_x];
    }
}
)";

// OpenCL resources struct to hold shared resources
struct OpenCLResources {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel dct_kernel;
    cl_kernel idct_kernel;
    cl_kernel quant_kernel;
    cl_kernel dequant_kernel;
    cl_mem quant_buf;
};

// Initialize OpenCL resources once
bool cl_init_opencl(OpenCLResources& resources) {
    cl_int cl_err;
    cl_uint cl_num_platforms;
    cl_err = clGetPlatformIDs(0, NULL, &cl_num_platforms);
    if (cl_err != CL_SUCCESS || cl_num_platforms == 0) {
        cout << "No OpenCL platforms found." << endl;
        return false;
    }
    vector<cl_platform_id> cl_platforms(cl_num_platforms);
    cl_err = clGetPlatformIDs(cl_num_platforms, cl_platforms.data(), NULL);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to get platform IDs." << endl;
        return false;
    }

    cl_uint cl_num_devices;
    cl_err = clGetDeviceIDs(cl_platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &cl_num_devices);
    if (cl_err != CL_SUCCESS || cl_num_devices == 0) {
        cout << "No OpenCL GPU devices found." << endl;
        return false;
    }
    vector<cl_device_id> cl_devices(cl_num_devices);
    cl_err = clGetDeviceIDs(cl_platforms[0], CL_DEVICE_TYPE_GPU, cl_num_devices, cl_devices.data(), NULL);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to get device IDs." << endl;
        return false;
    }

    resources.context = clCreateContext(NULL, 1, &cl_devices[0], NULL, NULL, &cl_err);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to create context." << endl;
        return false;
    }

    cl_command_queue_properties cl_properties[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    resources.queue = clCreateCommandQueueWithProperties(resources.context, cl_devices[0], cl_properties, &cl_err);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to create command queue." << endl;
        clReleaseContext(resources.context);
        return false;
    }

    resources.program = clCreateProgramWithSource(resources.context, 1, &cl_kernel_source, NULL, &cl_err);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to create program." << endl;
        clReleaseCommandQueue(resources.queue);
        clReleaseContext(resources.context);
        return false;
    }

    cl_err = clBuildProgram(resources.program, 1, &cl_devices[0], NULL, NULL, NULL);
    if (cl_err != CL_SUCCESS) {
        size_t cl_log_size;
        clGetProgramBuildInfo(resources.program, cl_devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &cl_log_size);
        vector<char> cl_build_log(cl_log_size);
        clGetProgramBuildInfo(resources.program, cl_devices[0], CL_PROGRAM_BUILD_LOG, cl_log_size, cl_build_log.data(), NULL);
        cout << "OpenCL Build Error: " << cl_build_log.data() << endl;
        clReleaseProgram(resources.program);
        clReleaseCommandQueue(resources.queue);
        clReleaseContext(resources.context);
        return false;
    }

    resources.dct_kernel = clCreateKernel(resources.program, "cl_dct_2d", &cl_err);
    resources.idct_kernel = clCreateKernel(resources.program, "cl_idct_2d", &cl_err);
    resources.quant_kernel = clCreateKernel(resources.program, "cl_quantize", &cl_err);
    resources.dequant_kernel = clCreateKernel(resources.program, "cl_dequantize", &cl_err);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to create kernels." << endl;
        clReleaseProgram(resources.program);
        clReleaseCommandQueue(resources.queue);
        clReleaseContext(resources.context);
        return false;
    }

    resources.quant_buf = clCreateBuffer(resources.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_quant_table), (void*)cl_quant_table, &cl_err);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to create quant_table buffer." << endl;
        clReleaseKernel(resources.dct_kernel);
        clReleaseKernel(resources.idct_kernel);
        clReleaseKernel(resources.quant_kernel);
        clReleaseKernel(resources.dequant_kernel);
        clReleaseProgram(resources.program);
        clReleaseCommandQueue(resources.queue);
        clReleaseContext(resources.context);
        return false;
    }

    return true;
}

// Cleanup OpenCL resources
void cl_cleanup_opencl(OpenCLResources& resources) {
    clReleaseMemObject(resources.quant_buf);
    clReleaseKernel(resources.dct_kernel);
    clReleaseKernel(resources.idct_kernel);
    clReleaseKernel(resources.quant_kernel);
    clReleaseKernel(resources.dequant_kernel);
    clReleaseProgram(resources.program);
    clReleaseCommandQueue(resources.queue);
    clReleaseContext(resources.context);
}

// Run-Length Encoding (RLE) with short
vector<pair<short, short>> cl_rle_encode(const vector<short>& cl_data) {
    vector<pair<short, short>> cl_rle;
    if (cl_data.empty()) return cl_rle;
    short cl_count = 1;
    for (size_t cl_i = 1; cl_i < cl_data.size(); cl_i++) {
        if (cl_data[cl_i] == cl_data[cl_i - 1] && cl_count < 255) {
            cl_count++;
        }
        else {
            cl_rle.push_back({ cl_data[cl_i - 1], cl_count });
            cl_count = 1;
        }
    }
    cl_rle.push_back({ cl_data.back(), cl_count });
    return cl_rle;
}

// Run-Length Decoding
vector<short> cl_rle_decode(const vector<pair<short, short>>& cl_rle) {
    vector<short> cl_data;
    for (const auto& cl_p : cl_rle) {
        for (short cl_i = 0; cl_i < cl_p.second; cl_i++) {
            cl_data.push_back(cl_p.first);
        }
    }
    return cl_data;
}

// Huffman Encoding
struct HuffmanNode {
    int cl_value;
    int cl_freq;
    HuffmanNode* cl_left, * cl_right;
    HuffmanNode(int cl_val, int cl_fr) : cl_value(cl_val), cl_freq(cl_fr), cl_left(nullptr), cl_right(nullptr) {}
};

struct Compare {
    bool operator()(HuffmanNode* cl_l, HuffmanNode* cl_r) {
        return cl_l->cl_freq > cl_r->cl_freq;
    }
};

void cl_build_huffman_codes(HuffmanNode* cl_root, string cl_code, map<short, string>& cl_huffman_codes) {
    if (!cl_root) return;
    if (!cl_root->cl_left && !cl_root->cl_right) {
        cl_huffman_codes[cl_root->cl_value] = cl_code.empty() ? "0" : cl_code;
    }
    cl_build_huffman_codes(cl_root->cl_left, cl_code + "0", cl_huffman_codes);
    cl_build_huffman_codes(cl_root->cl_right, cl_code + "1", cl_huffman_codes);
}

pair<vector<unsigned char>, map<short, string>> cl_huffman_encode(const vector<short>& cl_data) {
    map<short, int> cl_freq;
    for (short cl_val : cl_data) cl_freq[cl_val]++;

    priority_queue<HuffmanNode*, vector<HuffmanNode*>, Compare> cl_pq;
    for (auto& cl_p : cl_freq) {
        cl_pq.push(new HuffmanNode(cl_p.first, cl_p.second));
    }

    while (cl_pq.size() > 1) {
        HuffmanNode* cl_left = cl_pq.top(); cl_pq.pop();
        HuffmanNode* cl_right = cl_pq.top(); cl_pq.pop();
        HuffmanNode* cl_node = new HuffmanNode(-1, cl_left->cl_freq + cl_right->cl_freq);
        cl_node->cl_left = cl_left;
        cl_node->cl_right = cl_right;
        cl_pq.push(cl_node);
    }

    map<short, string> cl_huffman_codes;
    if (cl_pq.empty()) return { {}, cl_huffman_codes };
    cl_build_huffman_codes(cl_pq.top(), "", cl_huffman_codes);

    string cl_bitstream;
    for (short cl_val : cl_data) {
        cl_bitstream += cl_huffman_codes[cl_val];
    }

    vector<unsigned char> cl_encoded;
    for (size_t cl_i = 0; cl_i < cl_bitstream.size(); cl_i += 8) {
        unsigned char cl_byte = 0;
        for (size_t cl_j = 0; cl_j < 8 && cl_i + cl_j < cl_bitstream.size(); cl_j++) {
            cl_byte |= (cl_bitstream[cl_i + cl_j] == '1' ? 1 : 0) << (7 - cl_j);
        }
        cl_encoded.push_back(cl_byte);
    }

    return { cl_encoded, cl_huffman_codes };
}

// Huffman Decoding
vector<short> cl_huffman_decode(const vector<unsigned char>& cl_encoded_bytes, const map<short, string>& cl_huffman_codes, size_t cl_bit_length) {
    string cl_bitstream;
    for (unsigned char cl_byte : cl_encoded_bytes) {
        for (int cl_j = 7; cl_j >= 0; cl_j--) {
            cl_bitstream += (cl_byte & (1 << cl_j)) ? '1' : '0';
        }
    }
    if (cl_bitstream.size() > cl_bit_length) {
        cl_bitstream = cl_bitstream.substr(0, cl_bit_length);
    }

    vector<short> cl_decoded;
    map<string, short> cl_reverse_codes;
    for (const auto& cl_p : cl_huffman_codes) {
        cl_reverse_codes[cl_p.second] = cl_p.first;
    }

    string cl_current;
    for (char cl_bit : cl_bitstream) {
        cl_current += cl_bit;
        if (cl_reverse_codes.find(cl_current) != cl_reverse_codes.end()) {
            cl_decoded.push_back(cl_reverse_codes[cl_current]);
            cl_current = "";
        }
    }
    return cl_decoded;
}

// Process one channel (compression) with OpenCL
void cl_process_channel(const Mat& cl_channel, vector<short>& cl_compressed_data, vector<unsigned char>& cl_encoded_data, map<short, string>& cl_huffman_codes, size_t& cl_bit_length, int cl_channel_idx, const string& cl_output_folder, const string& cl_base_name, OpenCLResources& resources) {
    int cl_height = cl_channel.rows;
    int cl_width = cl_channel.cols;

    size_t cl_original_bits = static_cast<size_t>(cl_width) * cl_height * 8;
    size_t cl_original_bytes = static_cast<size_t>(cl_width) * cl_height;

    cout << "Channel " << cl_channel_idx << ": original bits = " << cl_original_bits << " bits (" << (cl_original_bits / 8) << " bytes)" << endl;

    cl_compressed_data.clear();
    int cl_blocks_x = (cl_width + 7) / 8;
    int cl_blocks_y = (cl_height + 7) / 8;
    int cl_num_blocks = cl_blocks_x * cl_blocks_y;
    vector<double> cl_all_blocks(cl_num_blocks * 64, 0.0);

    // Extract all 8x8 blocks (matching serial logic)
    for (int cl_i = 0, cl_block_idx = 0; cl_i < cl_height; cl_i += 8) {
        for (int cl_j = 0; cl_j < cl_width; cl_j += 8, cl_block_idx++) {
            for (int cl_x = 0; cl_x < 8 && cl_i + cl_x < cl_height; cl_x++) {
                for (int cl_y = 0; cl_y < 8 && cl_j + cl_y < cl_width; cl_y++) {
                    cl_all_blocks[cl_block_idx * 64 + cl_x * 8 + cl_y] = static_cast<double>(cl_channel.at<uchar>(cl_i + cl_x, cl_j + cl_y)) - 128;
                }
            }
        }
    }

    cl_int cl_err;
    cl_mem cl_blocks_buf = clCreateBuffer(resources.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * cl_num_blocks * 64, cl_all_blocks.data(), &cl_err);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to create blocks buffer." << endl;
        return;
    }

    // DCT
    cl_mem cl_dct_buf = clCreateBuffer(resources.context, CL_MEM_READ_WRITE, sizeof(double) * cl_num_blocks * 64, NULL, &cl_err);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to create DCT output buffer." << endl;
        clReleaseMemObject(cl_blocks_buf);
        return;
    }

    cl_err = clSetKernelArg(resources.dct_kernel, 0, sizeof(cl_mem), &cl_blocks_buf); // input
    cl_err |= clSetKernelArg(resources.dct_kernel, 1, sizeof(cl_mem), &cl_dct_buf);   // output
    cl_err |= clSetKernelArg(resources.dct_kernel, 2, sizeof(int), &cl_num_blocks);   // num_blocks

    if (cl_err != CL_SUCCESS) {
        cout << "Failed to set DCT kernel args." << endl;
        clReleaseMemObject(cl_blocks_buf);
        return;
    }
    size_t cl_global_work_size[3] = { (size_t)cl_num_blocks, 8, 8 };
    cl_err = clEnqueueNDRangeKernel(resources.queue, resources.dct_kernel, 3, NULL, cl_global_work_size, NULL, 0, NULL, NULL);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to execute DCT kernel." << endl;
        clReleaseMemObject(cl_blocks_buf);
        return;
    }

    // Quantize
    cl_err = clSetKernelArg(resources.quant_kernel, 0, sizeof(cl_mem), &cl_blocks_buf);
    cl_err |= clSetKernelArg(resources.quant_kernel, 1, sizeof(cl_mem), &resources.quant_buf);
    cl_err |= clSetKernelArg(resources.quant_kernel, 2, sizeof(int), &cl_num_blocks);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to set Quantize kernel args." << endl;
        clReleaseMemObject(cl_blocks_buf);
        return;
    }
    cl_err = clEnqueueNDRangeKernel(resources.queue, resources.quant_kernel, 3, NULL, cl_global_work_size, NULL, 0, NULL, NULL);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to execute Quantize kernel." << endl;
        clReleaseMemObject(cl_blocks_buf);
        return;
    }

    // Read back results
    cl_event cl_read_event;
    size_t buffer_size = sizeof(double) * 16 * 64; // = 1024 doubles
    cl_err = clEnqueueReadBuffer(resources.queue, cl_blocks_buf, CL_FALSE, 0,buffer_size, cl_all_blocks.data(), 0, NULL, &cl_read_event);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to read buffer." << endl;
        clReleaseMemObject(cl_blocks_buf);
        return;
    }
    clWaitForEvents(1, &cl_read_event);
    clReleaseEvent(cl_read_event);

    // Zigzag and RLE (matching serial)
    for (int cl_block_idx = 0; cl_block_idx < cl_num_blocks; cl_block_idx++) {
        vector<short> cl_zigzag(64);
        for (int cl_k = 0; cl_k < 64; cl_k++) {
            int cl_idx = cl_zigzag_order[cl_k];
            int cl_x = cl_idx / 8;
            int cl_y = cl_idx % 8;
            cl_zigzag[cl_k] = static_cast<short>(cl_all_blocks[cl_block_idx * 64 + cl_x * 8 + cl_y]);
        }
        auto cl_rle = cl_rle_encode(cl_zigzag);
        for (auto cl_p : cl_rle) {
            cl_compressed_data.push_back(cl_p.first);
            cl_compressed_data.push_back(cl_p.second);
        }
    }

    // Huffman encode
    auto cl_result = cl_huffman_encode(cl_compressed_data);
    cl_encoded_data = cl_result.first;
    cl_huffman_codes = cl_result.second;
    cl_bit_length = 0;
    for (short cl_val : cl_compressed_data) {
        cl_bit_length += cl_huffman_codes[cl_val].size();
    }

    size_t cl_compressed_bits = cl_bit_length;
    size_t cl_compressed_bytes = cl_encoded_data.size();

    cout << "Channel " << cl_channel_idx << ": compressed = " << cl_compressed_bits << " bits (" << cl_compressed_bytes << " bytes)\n";

    double cl_ratio_bits = static_cast<double>(cl_original_bits) / cl_compressed_bits;
    double cl_ratio_bytes = static_cast<double>(cl_original_bytes) / cl_compressed_bytes;

    cout << fixed << setprecision(2) << "Channel " << cl_channel_idx << ": compression ratio = " << cl_ratio_bytes << ":1 (¡Ö " << cl_ratio_bits << ":1 bits-based)\n\n";

    clReleaseMemObject(cl_blocks_buf);
}

// Reconstruct one channel (decompression) with OpenCL
Mat cl_reconstruct_channel(const vector<unsigned char>& cl_encoded_data, const map<short, string>& cl_huffman_codes, size_t cl_bit_length, int cl_height, int cl_width, OpenCLResources& resources) {
    Mat cl_channel(cl_height, cl_width, CV_8U, Scalar(0));

    // Decode Huffman
    vector<short> cl_decoded_data = cl_huffman_decode(cl_encoded_data, cl_huffman_codes, cl_bit_length);

    // Decode RLE
    vector<pair<short, short>> cl_rle_data;
    for (size_t cl_i = 0; cl_i + 1 < cl_decoded_data.size(); cl_i += 2) {
        cl_rle_data.push_back({ cl_decoded_data[cl_i], cl_decoded_data[cl_i + 1] });
    }
    vector<short> cl_zigzag = cl_rle_decode(cl_rle_data);

    int cl_blocks_x = (cl_width + 7) / 8;
    int cl_blocks_y = (cl_height + 7) / 8;
    int cl_num_blocks = cl_blocks_x * cl_blocks_y;
    vector<double> cl_all_blocks(cl_num_blocks * 64, 0.0);

    // Reconstruct blocks from zigzag (matching serial)
    size_t cl_zigzag_idx = 0;
    for (int cl_block_idx = 0; cl_block_idx < cl_num_blocks && cl_zigzag_idx < cl_zigzag.size(); cl_block_idx++) {
        for (int cl_k = 0; cl_k < 64 && cl_zigzag_idx < cl_zigzag.size(); cl_k++) {
            int cl_pos = cl_zigzag_order[cl_k];
            int cl_x = cl_pos / 8;
            int cl_y = cl_pos % 8;
            cl_all_blocks[cl_block_idx * 64 + cl_x * 8 + cl_y] = static_cast<double>(cl_zigzag[cl_zigzag_idx++]);
        }
    }

    cl_int cl_err;
    size_t buffer_size = sizeof(double) * 16 * 64; // = 1024 doubles


    cl_mem cl_blocks_buf = clCreateBuffer(resources.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buffer_size, cl_all_blocks.data(), &cl_err);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to create blocks buffer." << endl;
        return cl_channel;
    }

    // Dequantize
    cl_err = clSetKernelArg(resources.dequant_kernel, 0, sizeof(cl_mem), &cl_blocks_buf);
    cl_err |= clSetKernelArg(resources.dequant_kernel, 1, sizeof(cl_mem), &resources.quant_buf);
    cl_err |= clSetKernelArg(resources.dequant_kernel, 2, sizeof(int), &cl_num_blocks);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to set Dequantize kernel args." << endl;
        clReleaseMemObject(cl_blocks_buf);
        return cl_channel;
    }
    size_t cl_global_work_size[3] = { 16, 8, 8 };

    cl_err = clEnqueueNDRangeKernel(resources.queue, resources.dequant_kernel, 3, NULL, cl_global_work_size, NULL, 0, NULL, NULL);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to execute Dequantize kernel." << endl;
        clReleaseMemObject(cl_blocks_buf);
        return cl_channel;
    }

    // IDCT
    cl_mem cl_idct_buf = clCreateBuffer(resources.context, CL_MEM_WRITE_ONLY, sizeof(double) * cl_num_blocks * 64, NULL, &cl_err);
    if (cl_err != CL_SUCCESS) {
        std::cerr << "Failed to create cl_idct_buf. Error: " << cl_err << std::endl;
        clReleaseMemObject(cl_blocks_buf);
        return cl_channel;
    }

    cl_err = clSetKernelArg(resources.idct_kernel, 0, sizeof(cl_mem), &cl_blocks_buf); // input
    cl_err |= clSetKernelArg(resources.idct_kernel, 1, sizeof(cl_mem), &cl_idct_buf);   // output
    cl_err |= clSetKernelArg(resources.idct_kernel, 2, sizeof(int), &cl_num_blocks);    // num_blocks

    if (cl_err != CL_SUCCESS) {
        cout << "Failed to set IDCT kernel args." << endl;
        clReleaseMemObject(cl_blocks_buf);
        return cl_channel;
    }
    cl_err = clEnqueueNDRangeKernel(resources.queue, resources.idct_kernel, 3, NULL, cl_global_work_size, NULL, 0, NULL, NULL);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to execute IDCT kernel." << endl;
        clReleaseMemObject(cl_blocks_buf);
        return cl_channel;
    }

    // Read back results
    cl_event cl_read_event;
    cl_err = clEnqueueReadBuffer(resources.queue, cl_blocks_buf, CL_FALSE, 0, buffer_size, cl_all_blocks.data(), 0, NULL, &cl_read_event);
    if (cl_err != CL_SUCCESS) {
        cout << "Failed to read buffer." << endl;
        clReleaseMemObject(cl_blocks_buf);
        return cl_channel;
    }
    clWaitForEvents(1, &cl_read_event);
    clReleaseEvent(cl_read_event);

    // Store back (matching serial)
    for (int cl_i = 0, cl_block_idx = 0; cl_i < cl_height; cl_i += 8) {
        for (int cl_j = 0; cl_j < cl_width; cl_j += 8, cl_block_idx++) {
            for (int cl_x = 0; cl_x < 8 && cl_i + cl_x < cl_height; cl_x++) {
                for (int cl_y = 0; cl_y < 8 && cl_j + cl_y < cl_width; cl_y++) {
                    int idx = cl_block_idx * 64 + cl_x * 8 + cl_y;
                    int val = static_cast<int>(round(cl_all_blocks[idx] + 128));
                    cl_channel.at<uchar>(cl_i + cl_x, cl_j + cl_y) = static_cast<uchar>(std::min(255, std::max(0, val)));
                }
            }
        }
    }

    clReleaseMemObject(cl_blocks_buf);
    return cl_channel;
}


double cl_calculate_psnr_raw(const vector<vector<vector<uint8_t>>>& cl_orig, const vector<vector<vector<uint8_t>>>& cl_recon) {
    double cl_mse = 0.0;
    int cl_H = cl_orig.size(), cl_W = cl_orig[0].size(), cl_C = cl_orig[0][0].size();
    for (int cl_y = 0; cl_y < cl_H; cl_y++) {
        for (int cl_x = 0; cl_x < cl_W; cl_x++) {
            for (int cl_c = 0; cl_c < cl_C; cl_c++) {
                int cl_diff = static_cast<int>(cl_orig[cl_y][cl_x][cl_c]) - static_cast<int>(cl_recon[cl_y][cl_x][cl_c]);
                cl_mse += cl_diff * cl_diff;
            }
        }
    }
    cl_mse /= (cl_H * cl_W * cl_C);
    return (cl_mse == 0.0) ? INFINITY : 10.0 * log10((255.0 * 255.0) / cl_mse);
}

vector<vector<vector<uint8_t>>> cl_mat_to_array(const Mat& cl_img) {
    int cl_H = cl_img.rows, cl_W = cl_img.cols;
    vector<vector<vector<uint8_t>>> cl_arr(cl_H, vector<vector<uint8_t>>(cl_W, vector<uint8_t>(3)));
    for (int cl_y = 0; cl_y < cl_H; cl_y++) {
        for (int cl_x = 0; cl_x < cl_W; cl_x++) {
            Vec3b cl_p = cl_img.at<Vec3b>(cl_y, cl_x);
            for (int cl_c = 0; cl_c < 3; cl_c++) cl_arr[cl_y][cl_x][cl_c] = cl_p[cl_c];
        }
    }
    return cl_arr;
}

vector<string> cl_get_image_files(const string& cl_folder_path) {
    vector<pair<pair<int, int>, string>> cl_indexed_files;
    for (const auto& cl_entry : fs::directory_iterator(cl_folder_path)) {
        if (cl_entry.is_regular_file()) {
            string cl_path = cl_entry.path().string();
            string cl_ext = cl_entry.path().extension().string();
            transform(cl_ext.begin(), cl_ext.end(), cl_ext.begin(), ::tolower);
            if (cl_ext == ".jpg" || cl_ext == ".jpeg" || cl_ext == ".png") {
                Mat cl_img = imread(cl_path, IMREAD_COLOR);
                if (!cl_img.empty()) {
                    int cl_width = cl_img.cols;
                    int cl_height = cl_img.rows;
                    int cl_min_dim = min(cl_width, cl_height);
                    int cl_max_dim = max(cl_width, cl_height);
                    cl_indexed_files.emplace_back(make_pair(cl_min_dim, cl_max_dim), cl_path);
                }
            }
        }
    }
    sort(cl_indexed_files.begin(), cl_indexed_files.end());
    vector<string> cl_image_files;
    for (const auto& cl_p : cl_indexed_files) {
        cl_image_files.push_back(cl_p.second);
    }
    return cl_image_files;
}

void cl_save_compact_byte_stream(const string& cl_filename,
    int cl_width, int cl_height,
    const vector<unsigned char>& cl_compact_stream,
    const map<short, string>& cl_huffman_codes,
    size_t cl_bit_length) {
    ofstream cl_out(cl_filename, ios::binary);
    cl_out.write(reinterpret_cast<const char*>(&cl_width), sizeof(int));
    cl_out.write(reinterpret_cast<const char*>(&cl_height), sizeof(int));
    cl_out.write(reinterpret_cast<const char*>(&cl_bit_length), sizeof(size_t));

    uint32_t cl_table_size = cl_huffman_codes.size();
    cl_out.write(reinterpret_cast<const char*>(&cl_table_size), sizeof(uint32_t));
    for (const auto& [cl_val, cl_code] : cl_huffman_codes) {
        cl_out.write(reinterpret_cast<const char*>(&cl_val), sizeof(short));
        uint8_t cl_len = static_cast<uint8_t>(cl_code.size());
        cl_out.write(reinterpret_cast<const char*>(&cl_len), sizeof(uint8_t));
        cl_out.write(cl_code.c_str(), cl_len);
    }

    uint32_t cl_byte_count = cl_compact_stream.size();
    cl_out.write(reinterpret_cast<const char*>(&cl_byte_count), sizeof(uint32_t));
    cl_out.write(reinterpret_cast<const char*>(cl_compact_stream.data()), cl_byte_count);
    cl_out.close();
}

void cl_load_compact_byte_stream(const string& cl_filename,
    int& cl_width, int& cl_height,
    vector<unsigned char>& cl_compact_stream,
    map<short, string>& cl_huffman_codes,
    size_t& cl_bit_length) {
    ifstream cl_in(cl_filename, ios::binary);
    cl_in.read(reinterpret_cast<char*>(&cl_width), sizeof(int));
    cl_in.read(reinterpret_cast<char*>(&cl_height), sizeof(int));
    cl_in.read(reinterpret_cast<char*>(&cl_bit_length), sizeof(size_t));

    uint32_t cl_table_size;
    cl_in.read(reinterpret_cast<char*>(&cl_table_size), sizeof(uint32_t));
    cl_huffman_codes.clear();
    for (uint32_t cl_i = 0; cl_i < cl_table_size; cl_i++) {
        short cl_val;
        uint8_t cl_len;
        cl_in.read(reinterpret_cast<char*>(&cl_val), sizeof(short));
        cl_in.read(reinterpret_cast<char*>(&cl_len), sizeof(uint8_t));
        string cl_code(cl_len, '\0');
        cl_in.read(&cl_code[0], cl_len);
        cl_huffman_codes[cl_val] = cl_code;
    }

    uint32_t cl_byte_count;
    cl_in.read(reinterpret_cast<char*>(&cl_byte_count), sizeof(uint32_t));
    cl_compact_stream.resize(cl_byte_count);
    cl_in.read(reinterpret_cast<char*>(cl_compact_stream.data()), cl_byte_count);
    cl_in.close();
}

double cl_calculate_compression_ratio(size_t cl_original_bytes, size_t cl_compressed_bytes) {
    return static_cast<double>(cl_original_bytes) / cl_compressed_bytes;
}

void runOpenCL(const std::string& cl_input_folder,
    const std::string& cl_output_folder,
    const std::string& cl_csv_path) {
    // Initialize OpenCL resources once
    OpenCLResources resources = { 0 };
    if (!cl_init_opencl(resources)) {
        cout << "Failed to initialize OpenCL resources." << endl;
        return;
    }

    try {
        fs::create_directories(cl_output_folder);
    }
    catch (const fs::filesystem_error& cl_e) {
        cout << "Error creating output folder: " << cl_e.what() << endl;
        cl_cleanup_opencl(resources);
        return;
    }

    vector<string> cl_image_files = cl_get_image_files(cl_input_folder);
    if (cl_image_files.empty()) {
        cout << "No image files found in " << cl_input_folder << endl;
        cl_cleanup_opencl(resources);
        return;
    }

    ofstream cl_csv_file(cl_csv_path);
    cl_csv_file << "Image,Width,Height,Pixels,Time_ms,Orig_Size,Compact_Size,Reduction,Pixel_Diff_Min,Pixel_Diff_Max,Changed_Pixels,PSNR\n";

    int cl_image_count = 1;
    for (const auto& cl_image_file : cl_image_files) {
        Mat cl_img = imread(cl_image_file, IMREAD_COLOR);
        if (cl_img.empty()) {
            cout << "Failed to load " << cl_image_file << endl;
            continue;
        }

        int cl_W = cl_img.cols, cl_H = cl_img.rows;
        vector<Mat> cl_channels(3);
        split(cl_img, cl_channels);

        double cl_t0 = getTickCount();
        vector<vector<unsigned char>> cl_compact_streams(3);
        vector<map<short, string>> cl_huffman_codes(3);
        vector<size_t> cl_bit_lengths(3);

        string cl_base_name = "image" + to_string(cl_image_count);
        for (int cl_c = 0; cl_c < 3; cl_c++) {
            vector<short> cl_compressed_data;
            cl_process_channel(cl_channels[cl_c], cl_compressed_data, cl_compact_streams[cl_c], cl_huffman_codes[cl_c], cl_bit_lengths[cl_c], cl_c, cl_output_folder, cl_base_name, resources);
        }
        double cl_time_ms = (getTickCount() - cl_t0) / getTickFrequency() * 1000.0;

        int cl_compact_total_size = 0;
        for (int cl_c = 0; cl_c < 3; cl_c++) {
            string cl_bin_path = (fs::path(cl_output_folder) / (cl_base_name + "_ch" + to_string(cl_c) + ".bin")).string();
            cl_save_compact_byte_stream(cl_bin_path, cl_W, cl_H, cl_compact_streams[cl_c], cl_huffman_codes[cl_c], cl_bit_lengths[cl_c]);
            cl_compact_total_size += static_cast<int>(fs::file_size(cl_bin_path));
        }

        vector<Mat> cl_recon_ch(3);
        for (int cl_c = 0; cl_c < 3; cl_c++) {
            cl_recon_ch[cl_c] = cl_reconstruct_channel(cl_compact_streams[cl_c], cl_huffman_codes[cl_c], cl_bit_lengths[cl_c], cl_H, cl_W, resources);
        }

        Mat cl_recon;
        merge(cl_recon_ch, cl_recon);

        double cl_psnr = cl_calculate_psnr_raw(cl_mat_to_array(cl_img), cl_mat_to_array(cl_recon));

        int cl_raw_size = cl_W * cl_H * cl_img.channels();
        double cl_reduction = 100.0 * (1.0 - static_cast<double>(cl_compact_total_size) / cl_raw_size);
        double cl_compression_ratio = cl_calculate_compression_ratio(cl_raw_size, cl_compact_total_size);

        string cl_output_path = (fs::path(cl_output_folder) / (cl_base_name + ".jpg")).string();
        imwrite(cl_output_path, cl_recon, { IMWRITE_JPEG_QUALITY, 90 });

        Mat cl_diff;
        absdiff(cl_img, cl_recon, cl_diff);
        int cl_min_diff = 255, cl_max_diff = 0, cl_changed = 0;
        for (int cl_y = 0; cl_y < cl_H; cl_y++) {
            for (int cl_x = 0; cl_x < cl_W; cl_x++) {
                Vec3b cl_p = cl_diff.at<Vec3b>(cl_y, cl_x);
                bool cl_any = false;
                for (int cl_k = 0; cl_k < 3; cl_k++) {
                    if (cl_p[cl_k] > 0) {
                        cl_min_diff = min(cl_min_diff, int(cl_p[cl_k]));
                        cl_max_diff = max(cl_max_diff, int(cl_p[cl_k]));
                        cl_any = true;
                    }
                }
                if (cl_any) cl_changed++;
            }
        }
        if (cl_changed == 0) cl_min_diff = 0;

        cout << fixed << setprecision(1) << "Execution time: " << cl_time_ms << " ms\n";
        cout << cl_image_count << ". " << cl_image_file << " ¡ú " << cl_base_name << " (" << cl_W << "x" << cl_H << ", " << fs::file_size(cl_image_file) << " bytes, comp=" << cl_compact_total_size << ")\n";
        cout << "PSNR: " << fixed << setprecision(2) << cl_psnr << " dB\n";
        cout << "-------------------------------------------------------------------------------------------------\n";

        cl_csv_file << "\"" << cl_image_file << "\"," << cl_W << "," << cl_H << "," << (cl_W * cl_H) << "," << fixed << setprecision(1) << cl_time_ms << "," << cl_raw_size << "," << cl_compact_total_size << "," << setprecision(1) << cl_reduction << "," << cl_min_diff << "," << cl_max_diff << "," << cl_changed << "," << setprecision(2) << cl_psnr << "\n";

        cl_image_count++;
    }

    cl_csv_file.close();
    cl_cleanup_opencl(resources);
}