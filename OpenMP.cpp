#define _USE_MATH_DEFINES // Enable M_PI in <cmath> for MSVC
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <map>
#include <iomanip>
#include <filesystem>
#include "OpenMP.h"

// Fallback for M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Quantization table for quality 90% (JPEG standard, scaled)
const int omp_quant_table[8][8] = {
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
const int omp_zigzag_order[64] = {
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
};

// DCT and IDCT functions with double precision
void omp_dct_2d(double block[8][8]) {
    double temp[8][8];
    double cu, cv;

#pragma omp parallel for private(cu, cv) shared(block, temp) collapse(2)
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

void omp_idct_2d(double block[8][8]) {
    double temp[8][8];
    double cu, cv;

#pragma omp parallel for private(cu, cv) shared(block, temp) collapse(2)
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
void omp_quantize(double block[8][8]) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            block[i][j] = round(block[i][j] / omp_quant_table[i][j]);
}

void omp_dequantize(double block[8][8]) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            block[i][j] = block[i][j] * omp_quant_table[i][j];
}

// Run-Length Encoding (RLE) with short
vector<pair<short, short>> omp_rle_encode(const vector<short>& data) {
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
vector<short> omp_rle_decode(const vector<pair<short, short>>& rle) {
    vector<short> data;
    data.reserve(rle.size() * 2); // Reserve space to avoid frequent reallocations

    for (const auto& p : rle) {
        for (short i = 0; i < p.second; i++) {
            data.push_back(p.first);
        }
    }
    return data;
}

// Huffman Encoding
struct OmpHuffmanNode {
    int value;
    int freq;
    OmpHuffmanNode* left, * right;
    OmpHuffmanNode(int val, int fr) : value(val), freq(fr), left(nullptr), right(nullptr) {}
};

struct OmpCompare {
    bool operator()(OmpHuffmanNode* l, OmpHuffmanNode* r) {
        return l->freq > r->freq;
    }
};

void omp_build_huffman_codes(OmpHuffmanNode* root, string code, map<short, string>& huffman_codes) {
    if (!root) return;
    if (!root->left && !root->right) {
        huffman_codes[root->value] = code.empty() ? "0" : code;
    }
    omp_build_huffman_codes(root->left, code + "0", huffman_codes);
    omp_build_huffman_codes(root->right, code + "1", huffman_codes);
}

pair<vector<unsigned char>, map<short, string>> omp_huffman_encode(const vector<short>& data) {
    map<short, int> freq;

    // Count frequencies in parallel
#pragma omp parallel
    {
        map<short, int> local_freq;

#pragma omp for nowait
        for (int i = 0; i < static_cast<int>(data.size()); i++) {
            local_freq[data[i]]++;
        }

#pragma omp critical
        {
            for (const auto& pair : local_freq) {
                freq[pair.first] += pair.second;
            }
        }
    }

    priority_queue<OmpHuffmanNode*, vector<OmpHuffmanNode*>, OmpCompare> pq;
    for (auto& p : freq) {
        pq.push(new OmpHuffmanNode(p.first, p.second));
    }

    while (pq.size() > 1) {
        OmpHuffmanNode* left = pq.top(); pq.pop();
        OmpHuffmanNode* right = pq.top(); pq.pop();
        OmpHuffmanNode* node = new OmpHuffmanNode(-1, left->freq + right->freq);
        node->left = left;
        node->right = right;
        pq.push(node);
    }

    map<short, string> huffman_codes;
    if (pq.empty()) return { {}, huffman_codes };
    omp_build_huffman_codes(pq.top(), "", huffman_codes);

    // Free memory
    while (!pq.empty()) {
        OmpHuffmanNode* node = pq.top();
        pq.pop();
        delete node;
    }

    // Build bitstream in parallel segments
    const int chunk_size = 1000; // Adjust based on your needs
    const int num_chunks = (data.size() + chunk_size - 1) / chunk_size;
    vector<string> bitstream_chunks(num_chunks);

#pragma omp parallel for
    for (int c = 0; c < num_chunks; c++) {
        int start = c * chunk_size;
        int end = min(start + chunk_size, static_cast<int>(data.size()));
        for (int i = start; i < end; i++) {
            bitstream_chunks[c] += huffman_codes[data[i]];
        }
    }

    // Combine chunks
    string bitstream;
    for (const auto& chunk : bitstream_chunks) {
        bitstream += chunk;
    }

    // Pack bits into bytes
    vector<unsigned char> encoded;
    encoded.reserve((bitstream.size() + 7) / 8);

    for (int i = 0; i < static_cast<int>(bitstream.size()); i += 8) {
        unsigned char byte = 0;
        for (int j = 0; j < 8 && i + j < static_cast<int>(bitstream.size()); j++) {
            byte |= (bitstream[i + j] == '1' ? 1 : 0) << (7 - j);
        }
        encoded.push_back(byte);
    }

    return { encoded, huffman_codes };
}

// Huffman Decoding
vector<short> omp_huffman_decode(const vector<unsigned char>& encoded_bytes, const map<short, string>& huffman_codes, size_t bit_length) {
    // Unpack bytes into a bitstream string
    string bitstream;
    bitstream.reserve(encoded_bytes.size() * 8);

    for (unsigned char byte : encoded_bytes) {
        for (int j = 7; j >= 0; j--) {
            bitstream += (byte & (1 << j)) ? '1' : '0';
        }
    }

    // Truncate to the actual bit length
    if (bitstream.size() > bit_length) {
        bitstream = bitstream.substr(0, bit_length);
    }

    vector<short> decoded;
    decoded.reserve(bit_length / 2); // Rough estimate for reserving space

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

// Process block function for parallel processing
void omp_process_block(const Mat& channel, int start_i, int start_j, int height, int width,
    vector<short>& block_data, double reconstructed_block[8][8]) {

    double block[8][8] = { 0 };

    // Extract 8x8 block
    for (int x = 0; x < 8 && start_i + x < height; x++) {
        for (int y = 0; y < 8 && start_j + y < width; y++) {
            block[x][y] = static_cast<double>(channel.at<uchar>(start_i + x, start_j + y)) - 128;
        }
    }

    // DCT
    omp_dct_2d(block);

    // Quantize
    omp_quantize(block);

    // Copy to reconstructed block for later IDCT
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reconstructed_block[i][j] = block[i][j];
        }
    }

    // Zigzag
    vector<short> zigzag(64);
    for (int k = 0; k < 64; k++) {
        int idx = omp_zigzag_order[k];
        int x = idx / 8;
        int y = idx % 8;
        zigzag[k] = static_cast<short>(block[x][y]);
    }

    // RLE
    auto rle = omp_rle_encode(zigzag);

    // Save RLE data
    for (auto p : rle) {
        block_data.push_back(p.first);
        block_data.push_back(p.second);
    }
}

// Process one channel (compression)
void omp_process_channel(const Mat& channel, vector<short>& compressed_data,
    vector<unsigned char>& encoded_data, map<short, string>& huffman_codes,
    size_t& bit_length, int channel_idx, const string& output_folder,
    const string& base_name, Mat& reconstructed_channel) {

    int height = channel.rows;
    int width = channel.cols;

    // Initialize reconstructed channel
    reconstructed_channel = Mat(height, width, CV_8U, Scalar(0));

    // 1) Raw sizes
    size_t original_bits = static_cast<size_t>(width) * height * 8;        // bits
    size_t original_bytes = static_cast<size_t>(width) * height;            // bytes

    // Calculate and display the raw bit count
    cout << "Channel " << channel_idx
        << ": original bits = " << original_bits
        << " bits (" << (original_bits / 8)
        << " bytes)" << endl;

    // Create a vector to store all block data
    const int blocks_h = (height + 7) / 8;
    const int blocks_w = (width + 7) / 8;
    vector<vector<short>> all_blocks_data(blocks_h * blocks_w);

    // We can't use a vector of fixed-size arrays directly, so we'll use a custom struct
    struct BlockDct {
        double values[8][8];
    };
    vector<BlockDct> all_blocks_dct(blocks_h * blocks_w);

    // Process blocks in parallel
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < height; i += 8) {
        for (int j = 0; j < width; j += 8) {
            int block_idx = (i / 8) * blocks_w + (j / 8);
            double reconstructed_block[8][8];
            omp_process_block(channel, i, j, height, width, all_blocks_data[block_idx], reconstructed_block);

            // Save reconstructed block data
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    all_blocks_dct[block_idx].values[x][y] = reconstructed_block[x][y];
                }
            }
        }
    }

    // Combine all blocks data
    compressed_data.clear();
    for (const auto& block_data : all_blocks_data) {
        compressed_data.insert(compressed_data.end(), block_data.begin(), block_data.end());
    }

    // Reconstruct blocks in parallel
#pragma omp parallel for collapse(2)
    for (int i = 0; i < height; i += 8) {
        for (int j = 0; j < width; j += 8) {
            int block_idx = (i / 8) * blocks_w + (j / 8);
            double block[8][8];

            // Copy from saved DCT data
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    block[x][y] = all_blocks_dct[block_idx].values[x][y];
                }
            }

            // Dequantize
            omp_dequantize(block);

            // IDCT
            omp_idct_2d(block);

            // Store back
            for (int x = 0; x < 8 && i + x < height; x++) {
                for (int y = 0; y < 8 && j + y < width; y++) {
                    int val = static_cast<int>(block[x][y] + 128);
                    reconstructed_channel.at<uchar>(i + x, j + y) = static_cast<uchar>(max(0, min(255, val)));
                }
            }
        }
    }

    // Huffman encode
    auto result = omp_huffman_encode(compressed_data);
    encoded_data = result.first;
    huffman_codes = result.second;

    // Calculate bit length
    bit_length = 0;
    for (short val : compressed_data) {
        bit_length += huffman_codes[val].size();
    }

    // 3) Compressed sizes
    size_t compressed_bits = bit_length;             // bits in stream
    size_t compressed_bytes = encoded_data.size();    // packed bytes

    cout << "Channel " << channel_idx
        << ": compressed = " << compressed_bits << " bits ("
        << compressed_bytes << " bytes)\n";

    // 4) Compression ratio
    double ratio_bits = static_cast<double>(original_bits) / compressed_bits;
    double ratio_bytes = static_cast<double>(original_bytes) / compressed_bytes;

    cout << fixed << setprecision(2)
        << "Channel " << channel_idx
        << ": compression ratio = "
        << ratio_bytes << ":1 (≈ "      // bytes-based
        << ratio_bits << ":1 bits-based)\n\n";
}

// Save compact byte stream
void omp_save_compact_byte_stream(const string& filename,
    int width, int height,
    const vector<unsigned char>& compact_stream,
    const map<short, string>& huffman_codes,
    size_t bit_length) {
    ofstream out(filename, ios::binary);
    out.write(reinterpret_cast<const char*>(&width), sizeof(int));
    out.write(reinterpret_cast<const char*>(&height), sizeof(int));
    out.write(reinterpret_cast<const char*>(&bit_length), sizeof(size_t));

    uint32_t table_size = huffman_codes.size();
    out.write(reinterpret_cast<const char*>(&table_size), sizeof(uint32_t));
    for (const auto& [val, code] : huffman_codes) {
        out.write(reinterpret_cast<const char*>(&val), sizeof(short));
        uint8_t len = static_cast<uint8_t>(code.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(uint8_t));
        out.write(code.c_str(), len);
    }

    uint32_t byte_count = compact_stream.size();
    out.write(reinterpret_cast<const char*>(&byte_count), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(compact_stream.data()), byte_count);
    out.close();
}

// Calculate PSNR
double omp_calculate_psnr_raw(const vector<vector<vector<uint8_t>>>& orig, const vector<vector<vector<uint8_t>>>& recon) {
    double mse = 0.0;
    int H = orig.size(), W = orig[0].size(), C = orig[0][0].size();

#pragma omp parallel reduction(+:mse)
    {
        double local_mse = 0.0;

#pragma omp for collapse(2)
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                for (int c = 0; c < C; c++) {
                    int diff = static_cast<int>(orig[y][x][c]) - static_cast<int>(recon[y][x][c]);
                    local_mse += diff * diff;
                }
            }
        }

        mse += local_mse;
    }

    mse /= (H * W * C);
    return (mse == 0.0) ? INFINITY : 10.0 * log10((255.0 * 255.0) / mse);
}

vector<vector<vector<uint8_t>>> omp_mat_to_array(const Mat& img) {
    int H = img.rows, W = img.cols;
    vector<vector<vector<uint8_t>>> arr(H, vector<vector<uint8_t>>(W, vector<uint8_t>(3)));

#pragma omp parallel for collapse(2)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            Vec3b p = img.at<Vec3b>(y, x);
            for (int c = 0; c < 3; c++) arr[y][x][c] = p[c];
        }
    }
    return arr;
}

vector<string> omp_get_image_files(const string& folder_path) {
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
                }
            }
        }
    }
    sort(indexed_files.begin(), indexed_files.end());
    vector<string> image_files;
    for (const auto& p : indexed_files) {
        image_files.push_back(p.second);
    }
    return image_files;
}

double omp_calculate_compression_ratio(size_t original_bytes, size_t compressed_bytes) {
    return static_cast<double>(original_bytes) / compressed_bytes;
}

void runOpenMP(const std::string& input_folder,
    const std::string& output_folder,
    const std::string& csv_path,
    int num_threads) {

    // Set number of threads
    omp_set_num_threads(num_threads);
    cout << "Using " << num_threads << " threads\n";

    try {
        fs::create_directories(output_folder);
    }
    catch (const fs::filesystem_error& e) {
        cout << "Error creating output folder: " << e.what() << endl;
        return;
    }

    vector<string> image_files = omp_get_image_files(input_folder);
    if (image_files.empty()) {
        cout << "No image files found in " << input_folder << endl;
        return;
    }

    ofstream csv_file(csv_path);
    csv_file << "Image,Width,Height,Pixels,Time_ms,Orig_Size,Compact_Size,Reduction,Pixel_Diff_Min,Pixel_Diff_Max,Changed_Pixels,PSNR\n";

    int image_count = 1;
    for (const auto& image_file : image_files) {
        Mat img = imread(image_file, IMREAD_COLOR);
        if (img.empty()) {
            cout << "Failed to load " << image_file << endl;
            continue;
        }

        int W = img.cols, H = img.rows;
        vector<Mat> channels(3);
        split(img, channels);

        double t0 = omp_get_wtime();
        vector<vector<unsigned char>> compact_streams(3);
        vector<map<short, string>> huffman_codes(3);
        vector<size_t> bit_lengths(3);
        vector<Mat> recon_channels(3);

        string base_name = "image" + to_string(image_count);

        // Process channels in parallel
#pragma omp parallel for
        for (int c = 0; c < 3; c++) {
            vector<short> compressed_data;
            omp_process_channel(channels[c], compressed_data, compact_streams[c],
                huffman_codes[c], bit_lengths[c], c, output_folder,
                base_name, recon_channels[c]);
        }

        double time_ms = (omp_get_wtime() - t0) * 1000.0;

        // Save compact byte streams to disk
        int compact_total_size = 0;

        for (int c = 0; c < 3; c++) {
            string bin_path = (fs::path(output_folder) / (base_name + "_ch" + to_string(c) + ".bin")).string();
            omp_save_compact_byte_stream(bin_path, W, H, compact_streams[c], huffman_codes[c], bit_lengths[c]);
            compact_total_size += static_cast<int>(fs::file_size(bin_path));
        }

        // Merge reconstructed channels
        Mat recon;
        merge(recon_channels, recon);

        // Compute PSNR
        double psnr = omp_calculate_psnr_raw(omp_mat_to_array(img), omp_mat_to_array(recon));

        // Calculate compression ratio based on actual .bin files
        int raw_size = W * H * img.channels();
        double reduction = 100.0 * (1.0 - static_cast<double>(compact_total_size) / raw_size);
        double compression_ratio = omp_calculate_compression_ratio(raw_size, compact_total_size);

        // Save reconstructed image
        string output_path = (fs::path(output_folder) / (base_name + ".jpg")).string();
        imwrite(output_path, recon, { IMWRITE_JPEG_QUALITY, 90 });

        // Pixel diff stats
        Mat diff;
        absdiff(img, recon, diff);
        int min_diff = 255, max_diff = 0, changed = 0;

#pragma omp parallel
        {
            int local_min = 255;
            int local_max = 0;
            int local_changed = 0;

#pragma omp for collapse(2)
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    Vec3b p = diff.at<Vec3b>(y, x);
                    bool any = false;
                    for (int k = 0; k < 3; k++) {
                        if (p[k] > 0) {
                            local_min = min(local_min, int(p[k]));
                            local_max = max(local_max, int(p[k]));
                            any = true;
                        }
                    }
                    if (any) local_changed++;
                }
            }

#pragma omp critical
            {
                min_diff = min(min_diff, local_min);
                max_diff = max(max_diff, local_max);
                changed += local_changed;
            }
        }

        if (changed == 0) min_diff = 0;

        cout << fixed << setprecision(1)
            << "Execution time: " << time_ms << " ms\n";

        // Log output
        cout << image_count << ". " << image_file << " → " << base_name
            << " (" << W << "x" << H << ", " << fs::file_size(image_file)
            << " bytes, comp=" << compact_total_size << ")\n";
        cout << "PSNR: " << fixed << setprecision(2) << psnr << " dB\n";
        cout << "-------------------------------------------------------------------------------------------------" << "\n";

        // Write CSV row
        csv_file << "\"" << image_file << "\"," << W << "," << H << "," << (W * H) << ","
            << fixed << setprecision(1) << time_ms << ","
            << raw_size << "," << compact_total_size << ","
            << setprecision(1) << reduction << ","
            << min_diff << "," << max_diff << "," << changed << ","
            << setprecision(2) << psnr << "\n";

        image_count++;
    }

    csv_file.close();
    cout << "Completed processing all images.\n";
}