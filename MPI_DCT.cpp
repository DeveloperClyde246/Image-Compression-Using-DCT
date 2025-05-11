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
#include <filesystem>
#include <mpi.h>
#include <stdexcept>
#include "m_pi.h"

// Fallback for M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Quantization table for quality 90% (JPEG standard, scaled)
const int quant_table[8][8] = {
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
const int zigzag_order[64] = {
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
};

// DCT and IDCT functions with double precision
void mpi_dct_2d(double block[8][8]) {
    double temp[8][8];
    double cu, cv;
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            double sum = 0.0;
            cu = (u == 0) ? 1.0 / sqrt(2) : 1.0;
            cv = (v == 0) ? 1.0 / sqrt(2) : 1.0;
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    sum += block[x][y] * cos((2 * x + 1) * u * M_PI / 16.0) * cos((2 * y + 1) * v * M_PI / 16.0);
                }
            }
            temp[u][v] = 0.25 * cu * cv * sum;
        }
    }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            block[i][j] = temp[i][j];
}

void mpi_idct_2d(double block[8][8]) {
    double temp[8][8];
    double cu, cv;
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            double sum = 0.0;
            for (int u = 0; u < 8; u++) {
                for (int v = 0; v < 8; v++) {
                    cu = (u == 0) ? 1.0 / sqrt(2) : 1.0;
                    cv = (v == 0) ? 1.0 / sqrt(2) : 1.0;
                    sum += cu * cv * block[u][v] * cos((2 * x + 1) * u * M_PI / 16.0) * cos((2 * y + 1) * v * M_PI / 16.0);
                }
            }
            temp[x][y] = 0.25 * sum;
        }
    }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            block[i][j] = temp[i][j];
}

// Quantization and Dequantization
void mpi_quantize(double block[8][8]) {
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            block[i][j] = round(block[i][j] / quant_table[i][j]);
}

void mpi_dequantize(double block[8][8]) {
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            block[i][j] = block[i][j] * quant_table[i][j];
}

// Run-Length Encoding (RLE) with short
vector<pair<short, short>> mpi_rle_encode(const vector<short>& data) {
    vector<pair<short, short>> rle;
    if (data.empty()) return rle;
    short count = 1;
    for (size_t i = 1; i < data.size(); i++) {
        if (data[i] == data[i - 1] && count < 255) {
            count++;
        }
        else {
            rle.push_back({ data[i - 1], count });
            count = 1;
        }
    }
    rle.push_back({ data.back(), count });
    return rle;
}

// Run-Length Decoding
vector<short> mpi_rle_decode(const vector<pair<short, short>>& rle) {
    vector<short> data;
    for (const auto& p : rle) {
        for (short i = 0; i < p.second; i++) {
            data.push_back(p.first);
        }
    }
    return data;
}

// Huffman Encoding
struct HuffmanNode {
    int value;
    int freq;
    HuffmanNode* left, * right;
    HuffmanNode(int val, int fr) : value(val), freq(fr), left(nullptr), right(nullptr) {}
};

struct Compare {
    bool operator()(HuffmanNode* l, HuffmanNode* r) {
        return l->freq > r->freq;
    }
};

void mpi_build_huffman_codes(HuffmanNode* root, string code, map<short, string>& huffman_codes) {
    if (!root) return;
    if (!root->left && !root->right) {
        huffman_codes[root->value] = code.empty() ? "0" : code;
    }
    mpi_build_huffman_codes(root->left, code + "0", huffman_codes);
    mpi_build_huffman_codes(root->right, code + "1", huffman_codes);
}

pair<vector<unsigned char>, map<short, string>> mpi_huffman_encode(const vector<short>& data) {
    map<short, int> freq;
    for (short val : data) freq[val]++;
    priority_queue<HuffmanNode*, vector<HuffmanNode*>, Compare> pq;
    for (auto& p : freq) {
        pq.push(new HuffmanNode(p.first, p.second));
    }
    while (pq.size() > 1) {
        HuffmanNode* left = pq.top(); pq.pop();
        HuffmanNode* right = pq.top(); pq.pop();
        HuffmanNode* node = new HuffmanNode(-1, left->freq + right->freq);
        node->left = left;
        node->right = right;
        pq.push(node);
    }
    map<short, string> huffman_codes;
    if (pq.empty()) return { {}, huffman_codes };
    mpi_build_huffman_codes(pq.top(), "", huffman_codes);
    string bitstream;
    for (short val : data) {
        bitstream += huffman_codes[val];
    }
    vector<unsigned char> encoded;
    for (size_t i = 0; i < bitstream.size(); i += 8) {
        unsigned char byte = 0;
        for (size_t j = 0; j < 8 && i + j < bitstream.size(); j++) {
            byte |= (bitstream[i + j] == '1' ? 1 : 0) << (7 - j);
        }
        encoded.push_back(byte);
    }
    return { encoded, huffman_codes };
}

// Huffman Decoding
vector<short> mpi_huffman_decode(const vector<unsigned char>& encoded_bytes, const map<short, string>& huffman_codes, size_t bit_length) {
    string bitstream;
    for (unsigned char byte : encoded_bytes) {
        for (int j = 7; j >= 0; j--) {
            bitstream += (byte & (1 << j)) ? '1' : '0';
        }
    }
    if (bitstream.size() > bit_length) {
        bitstream = bitstream.substr(0, bit_length);
    }
    vector<short> decoded;
    map<string, short> reverse_codes;
    for (const auto& p : huffman_codes) {
        reverse_codes[p.second] = p.first;
    }
    string current;
    for (char bit : bitstream) {
        current += bit;
        if (reverse_codes.find(current) != reverse_codes.end()) {
            decoded.push_back(reverse_codes[current]);
            current = "";
        }
    }
    return decoded;
}

// Process one channel (compression)
void mpi_process_channel(const Mat& channel, vector<short>& compressed_data, vector<unsigned char>& encoded_data, map<short, string>& huffman_codes, size_t& bit_length) {
    int height = channel.rows;
    int width = channel.cols;
    compressed_data.clear();
    for (int i = 0; i < height; i += 8) {
        for (int j = 0; j < width; j += 8) {
            double block[8][8] = { 0 };
            for (int x = 0; x < 8 && i + x < height; x++) {
                for (int y = 0; y < 8 && j + y < width; y++) {
                    block[x][y] = static_cast<double>(channel.at<uchar>(i + x, j + y)) - 128;
                }
            }
            mpi_dct_2d(block);
            mpi_quantize(block);
            vector<short> zigzag(64);
            for (int k = 0; k < 64; k++) {
                int idx = zigzag_order[k];
                int x = idx / 8;
                int y = idx % 8;
                zigzag[k] = static_cast<short>(block[x][y]);
            }
            auto rle = mpi_rle_encode(zigzag);
            for (auto p : rle) {
                compressed_data.push_back(p.first);
                compressed_data.push_back(p.second);
            }
        }
    }
    auto result = mpi_huffman_encode(compressed_data);
    encoded_data = result.first;
    huffman_codes = result.second;
    bit_length = 0;
    for (short val : compressed_data) {
        bit_length += huffman_codes[val].size();
    }
}

// Reconstruct one channel (decompression)
Mat mpi_reconstruct_channel(const vector<unsigned char>& encoded_data, const map<short, string>& huffman_codes, size_t bit_length, int height, int width) {
    Mat channel(height, width, CV_8U, Scalar(0));
    vector<short> decoded_data = mpi_huffman_decode(encoded_data, huffman_codes, bit_length);
    vector<pair<short, short>> rle_data;
    for (size_t i = 0; i + 1 < decoded_data.size(); i += 2) {
        rle_data.push_back({ decoded_data[i], decoded_data[i + 1] });
    }
    vector<short> zigzag = mpi_rle_decode(rle_data);
    size_t idx = 0;
    for (int i = 0; i < height && idx < zigzag.size(); i += 8) {
        for (int j = 0; j < width && idx < zigzag.size(); j += 8) {
            double block[8][8] = { 0 };
            for (int k = 0; k < 64 && idx < zigzag.size(); k++) {
                int pos = zigzag_order[k];
                int x = pos / 8;
                int y = pos % 8;
                block[x][y] = static_cast<double>(zigzag[idx++]);
            }
            mpi_dequantize(block);
            mpi_idct_2d(block);
            for (int x = 0; x < 8 && i + x < height; x++) {
                for (int y = 0; y < 8 && j + y < width; y++) {
                    int val = static_cast<int>(block[x][y] + 128);
                    channel.at<uchar>(i + x, j + y) = max(0, min(255, val));
                }
            }
        }
    }
    return channel;
}

// Calculate PSNR
double mpi_calculate_psnr(const Mat& orig, const Mat& recon) {
    Mat diff;
    absdiff(orig, recon, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    double mse = sum(diff)[0] / (orig.rows * orig.cols * orig.channels());
    if (mse == 0) return INFINITY;
    return 10 * log10((255 * 255) / mse);
}

// Get image files from folder and sort by smallest dimension
vector<string> mpi_get_image_files(const string& folder_path, int rank) {
    cout << "Rank " << rank << ": Entering mpi_get_image_files for folder: " << folder_path << endl;
    cout.flush();
    vector<pair<pair<int, int>, string>> indexed_files; // (min_dim, max_dim, path)
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            string path = entry.path().string();
            string ext = entry.path().extension().string();
            transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                Mat img = imread(path, IMREAD_COLOR);
                if (!img.empty()) {
                    int width = img.cols;
                    int height = img.rows;
                    int min_dim = min(width, height);
                    int max_dim = max(width, height);
                    indexed_files.emplace_back(make_pair(min_dim, max_dim), path);
                    cout << "Rank " << rank << ": Found image: " << path << " (" << width << "x" << height << ")" << endl;
                    cout.flush();
                }
                else {
                    cout << "Rank " << rank << ": Failed to load image: " << path << endl;
                    cout.flush();
                }
            }
        }
    }
    sort(indexed_files.begin(), indexed_files.end());
    vector<string> image_files;
    for (const auto& p : indexed_files) {
        image_files.push_back(p.second);
    }
    cout << "Rank " << rank << ": Found " << image_files.size() << " images in " << folder_path << endl;
    cout.flush();
    return image_files;
}

void mpi_parallel_process_channel(const cv::Mat& channel, std::vector<short>& local_data, int rank, int size) {
    int height = channel.rows;
    int width = channel.cols;

    std::vector<std::pair<int, int>> blocks;
    for (int i = 0; i < height; i += 8) {
        for (int j = 0; j < width; j += 8) {
            blocks.emplace_back(i, j);
        }
    }

    int num_blocks = blocks.size();
    int blocks_per_proc = num_blocks / size;
    int remainder = num_blocks % size;
    int start_idx = rank * blocks_per_proc + std::min(rank, remainder);
    int local_blocks = blocks_per_proc + (rank < remainder ? 1 : 0);

    for (int idx = 0; idx < local_blocks; ++idx) {
        int global_idx = start_idx + idx;
        int i = blocks[global_idx].first;
        int j = blocks[global_idx].second;

        double block[8][8] = { 0 };
        for (int x = 0; x < 8 && i + x < height; ++x) {
            for (int y = 0; y < 8 && j + y < width; ++y) {
                block[x][y] = static_cast<double>(channel.at<uchar>(i + x, j + y)) - 128;
            }
        }

        mpi_dct_2d(block);
        mpi_quantize(block);

        std::vector<short> zigzag(64);
        for (int k = 0; k < 64; ++k) {
            int idx = zigzag_order[k];
            int x = idx / 8;
            int y = idx % 8;
            zigzag[k] = static_cast<short>(block[x][y]);
        }

        auto rle = mpi_rle_encode(zigzag);
        for (const auto& p : rle) {
            local_data.push_back(p.first);
            local_data.push_back(p.second);
        }
    }
}

// Struct to hold compression results
struct CompressionResult {
    char image_name[128];
    int width;
    int height;
    int pixels;
    float time_ms;
    int orig_size;
    int comp_size;
    float reduction;
    int pixel_diff_min;
    int pixel_diff_max;
    int changed_pixels;
    float psnr;
};

// Define MPI datatype for CompressionResult
MPI_Datatype MPI_CompressionResult;
void mpi_create_mpi_compression_result_type() {
    CompressionResult dummy; // For address calculations

    // Define the block lengths, displacements, and types
    int block_lengths[] = {
        128, // image_name
        1,   // width
        1,   // height
        1,   // pixels
        1,   // time_ms
        1,   // orig_size
        1,   // comp_size
        1,   // reduction
        1,   // pixel_diff_min
        1,   // pixel_diff_max
        1,   // changed_pixels
        1    // psnr
    };

    MPI_Aint displacements[12];
    MPI_Aint base_address;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&dummy.image_name[0], &displacements[0]);
    MPI_Get_address(&dummy.width, &displacements[1]);
    MPI_Get_address(&dummy.height, &displacements[2]);
    MPI_Get_address(&dummy.pixels, &displacements[3]);
    MPI_Get_address(&dummy.time_ms, &displacements[4]);
    MPI_Get_address(&dummy.orig_size, &displacements[5]);
    MPI_Get_address(&dummy.comp_size, &displacements[6]);
    MPI_Get_address(&dummy.reduction, &displacements[7]);
    MPI_Get_address(&dummy.pixel_diff_min, &displacements[8]);
    MPI_Get_address(&dummy.pixel_diff_max, &displacements[9]);
    MPI_Get_address(&dummy.changed_pixels, &displacements[10]);
    MPI_Get_address(&dummy.psnr, &displacements[11]);

    // Adjust displacements relative to base address
    for (int i = 0; i < 12; i++) {
        displacements[i] -= base_address;
    }

    MPI_Datatype types[] = {
        MPI_CHAR,  // image_name
        MPI_INT,   // width
        MPI_INT,   // height
        MPI_INT,   // pixels
        MPI_FLOAT, // time_ms
        MPI_INT,   // orig_size
        MPI_INT,   // comp_size
        MPI_FLOAT, // reduction
        MPI_INT,   // pixel_diff_min
        MPI_INT,   // pixel_diff_max
        MPI_INT,   // changed_pixels
        MPI_FLOAT  // psnr
    };

    MPI_Type_create_struct(12, block_lengths, displacements, types, &MPI_CompressionResult);
    MPI_Type_commit(&MPI_CompressionResult);
}

void runMPI(const std::string& input_folder, const std::string& output_folder, const std::string& csv_path) {
    std::cout << "Running binary built on: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "Program started" << std::endl;
    std::cout.flush();

    MPI_Init(NULL, NULL);
    mpi_create_mpi_compression_result_type();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        fs::create_directories(output_folder);
    }
    catch (const fs::filesystem_error& e) {
        cout << "Error creating output folder: " << e.what() << endl;
        return;
    }


    if (rank == 0) fs::create_directories(output_folder);

    std::vector<std::string> image_files;
    if (rank == 0) {
        image_files = mpi_get_image_files(input_folder, rank);
        if (image_files.empty()) {
            std::cerr << "Rank 0: No images found." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast image file names to all ranks
    std::string all_filenames;
    if (rank == 0) {
        for (const auto& f : image_files) all_filenames += f + '\0';
    }
    int buffer_size = static_cast<int>(all_filenames.size());
    MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<char> file_buffer(buffer_size);
    if (rank == 0) memcpy(file_buffer.data(), all_filenames.data(), buffer_size);
    MPI_Bcast(file_buffer.data(), buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        std::string current;
        for (char c : file_buffer) {
            if (c == '\0') {
                image_files.push_back(current);
                current.clear();
            }
            else current += c;
        }
    }

    std::ofstream csv_file;
    if (rank == 0) {
        csv_file.open("mpi_results.csv");
        csv_file << "Image,Width,Height,Pixels,Time_ms,Orig_Size,Comp_Size,Reduction,Pixel_Diff_Min,Pixel_Diff_Max,Changed_Pixels,PSNR,Num_Processes\n";
    }

    for (const auto& image_file : image_files) {
        cv::Mat img = imread(image_file, IMREAD_COLOR);
        if (img.empty()) continue;
        int height = img.rows, width = img.cols;

        std::vector<cv::Mat> channels(3);
        split(img, channels);

        double start_time = MPI_Wtime();

        std::vector<std::vector<short>> all_local_data(3);
        for (int c = 0; c < 3; ++c) {
            mpi_parallel_process_channel(channels[c], all_local_data[c], rank, size);
        }

        std::vector<std::vector<short>> global_data(3);
        for (int c = 0; c < 3; ++c) {
            int local_size = static_cast<int>(all_local_data[c].size());
            std::vector<int> sizes(size);
            MPI_Gather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

            std::vector<int> displs(size);
            if (rank == 0) {
                displs[0] = 0;
                for (int i = 1; i < size; ++i)
                    displs[i] = displs[i - 1] + sizes[i - 1];
                global_data[c].resize(displs[size - 1] + sizes[size - 1]);
            }

            MPI_Gatherv(all_local_data[c].data(), local_size, MPI_SHORT,
                global_data[c].data(), sizes.data(), displs.data(), MPI_SHORT,
                0, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            std::vector<cv::Mat> recon_channels;
            double total_bit_len = 0;
            int total_comp_size = 0;
            for (int c = 0; c < 3; ++c) {
                auto [encoded, codes] = mpi_huffman_encode(global_data[c]);
                size_t bit_len = 0;
                for (auto& s : global_data[c]) bit_len += codes[s].size();
                total_bit_len += bit_len;
                total_comp_size += (bit_len + 7) / 8;
                recon_channels.push_back(mpi_reconstruct_channel(encoded, codes, bit_len, height, width));
            }

            cv::Mat recon_img;
            merge(recon_channels, recon_img);

            std::string output_filename = (fs::path(output_folder) / fs::path(image_file).filename()).string();
            imwrite(output_filename, recon_img);

            double time_ms = (MPI_Wtime() - start_time) * 1000.0;
            int orig_size = height * width * 3;
            double reduction = 100.0 * (1.0 - (double)total_comp_size / orig_size);

            cv::Mat diff;
            absdiff(img, recon_img, diff);
            int min_d = 255, max_d = 0, changed = 0;
            for (int i = 0; i < diff.rows; ++i) {
                for (int j = 0; j < diff.cols; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        int val = diff.at<cv::Vec3b>(i, j)[k];
                        if (val > 0) {
                            min_d = std::min(min_d, val);
                            max_d = std::max(max_d, val);
                            changed++;
                        }
                    }
                }
            }

            double psnr = mpi_calculate_psnr(img, recon_img);

            std::cout << "Processed: " << image_file << " | Time: " << time_ms << " ms | PSNR: " << psnr << " dB\n";

            csv_file << image_file << "," << width << "," << height << "," << width * height << ","
                << time_ms << "," << orig_size << "," << total_comp_size << "," << reduction << ","
                << min_d << "," << max_d << "," << changed << "," << psnr << "," << size << "\n";
        }
    }

    if (rank == 0) csv_file.close();

    MPI_Type_free(&MPI_CompressionResult);
    MPI_Finalize();
}