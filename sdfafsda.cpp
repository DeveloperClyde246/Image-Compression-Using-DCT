#define _USE_MATH_DEFINES // Enable M_PI in <cmath> for MSVC
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <map>
#include "Serial.h"
#include <sstream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <type_traits>
#include <filesystem>

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
void dct_2d(double block[8][8]) {
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

void idct_2d(double block[8][8]) {
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
void quantize(double block[8][8]) {
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            block[i][j] = round(block[i][j] / quant_table[i][j]);
}

void dequantize(double block[8][8]) {
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            block[i][j] = block[i][j] * quant_table[i][j];
}

// Run-Length Encoding (RLE) with short
vector<pair<short, short>> rle_encode(const vector<short>& data) {
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
vector<short> rle_decode(const vector<pair<short, short>>& rle) {
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

void build_huffman_codes(HuffmanNode* root, string code, map<short, string>& huffman_codes) {
    if (!root) return;
    if (!root->left && !root->right) {
        huffman_codes[root->value] = code.empty() ? "0" : code;
    }
    build_huffman_codes(root->left, code + "0", huffman_codes);
    build_huffman_codes(root->right, code + "1", huffman_codes);
}

pair<vector<unsigned char>, map<short, string>> huffman_encode(const vector<short>& data) {
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
    build_huffman_codes(pq.top(), "", huffman_codes);

    // Encode data into a bitstream and pack into bytes
    string bitstream;
    for (short val : data) {
        bitstream += huffman_codes[val];
    }

    // Pack bits into bytes
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
vector<short> huffman_decode(const vector<unsigned char>& encoded_bytes, const map<short, string>& huffman_codes, size_t bit_length) {
    // Unpack bytes into a bitstream string
    string bitstream;
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
void process_channel(const Mat& channel, vector<short>& compressed_data, vector<unsigned char>& encoded_data, map<short, string>& huffman_codes, size_t& bit_length, int channel_idx, const string& output_folder, const string& base_name) {
    int height = channel.rows;
    int width = channel.cols;

    compressed_data.clear();
    for (int i = 0; i < height; i += 8) {
        for (int j = 0; j < width; j += 8) {
            double block[8][8] = { 0 };
            // Extract 8x8 block
            for (int x = 0; x < 8 && i + x < height; x++) {
                for (int y = 0; y < 8 && j + y < width; y++) {
                    block[x][y] = static_cast<double>(channel.at<uchar>(i + x, j + y)) - 128;
                }
            }
            // DCT
            dct_2d(block);
            // Quantize
            quantize(block);
            // Zigzag
            vector<short> zigzag(64);
            for (int k = 0; k < 64; k++) {
                int idx = zigzag_order[k];
                int x = idx / 8;
                int y = idx % 8;
                zigzag[k] = static_cast<short>(block[x][y]);
            }
            // RLE
            auto rle = rle_encode(zigzag);
            for (auto p : rle) {
                compressed_data.push_back(p.first);
                compressed_data.push_back(p.second);
            }
        }
    }

    // Huffman encode
    auto result = huffman_encode(compressed_data);
    encoded_data = result.first;
    huffman_codes = result.second;
    bit_length = 0;
    for (short val : compressed_data) {
        bit_length += huffman_codes[val].size();
    }

    // Generate Huffman-encoded bitstream
    string bitstream;
    for (short val : compressed_data) {
        bitstream += huffman_codes[val];
    }

    // Display Huffman-encoded bitstream with spaces every 5 bits
    cout << "Huffman-encoded bitstream for channel " << channel_idx << " (first 100 bits, grouped by 5): ";
    size_t display_length = min(bitstream.size(), static_cast<size_t>(100));
    for (size_t i = 0; i < display_length; i++) {
        cout << bitstream[i];
        if (i % 5 == 4 && i != display_length - 1) cout << " ";
    }
    if (bitstream.size() > display_length) cout << "...";
    cout << endl;
    cout << "Total Huffman bitstream length: " << bitstream.size() << " bits (" << (bitstream.size() + 7) / 8 << " bytes)" << endl;

    // Save the bitstream to a .txt file
    string txt_path = (fs::path(output_folder) / (base_name + "_ch" + to_string(channel_idx) + "_huffman.txt")).string();
    ofstream txt_file(txt_path);
    if (!txt_file.is_open()) {
        cout << "Failed to open " << txt_path << " for writing" << endl;
        return;
    }
    for (size_t i = 0; i < bitstream.size(); i++) {
        txt_file << bitstream[i];
        if (i % 5 == 4 && i != bitstream.size() - 1) txt_file << " ";
    }
    txt_file.close();
    cout << "Huffman bitstream saved to: " << txt_path << endl;
}

// Reconstruct one channel (decompression)
Mat reconstruct_channel(const vector<unsigned char>& encoded_data, const map<short, string>& huffman_codes, size_t bit_length, int height, int width) {
    Mat channel(height, width, CV_8U, Scalar(0));

    // Decode Huffman
    vector<short> decoded_data = huffman_decode(encoded_data, huffman_codes, bit_length);

    // Decode RLE
    vector<pair<short, short>> rle_data;
    for (size_t i = 0; i + 1 < decoded_data.size(); i += 2) {
        rle_data.push_back({ decoded_data[i], decoded_data[i + 1] });
    }
    vector<short> zigzag = rle_decode(rle_data);

    // Reconstruct blocks
    size_t idx = 0;
    for (int i = 0; i < height && idx < zigzag.size(); i += 8) {
        for (int j = 0; j < width && idx < zigzag.size(); j += 8) {
            double block[8][8] = { 0 };
            // Reconstruct block from zigzag
            for (int k = 0; k < 64 && idx < zigzag.size(); k++) {
                int pos = zigzag_order[k];
                int x = pos / 8;
                int y = pos % 8;
                block[x][y] = static_cast<double>(zigzag[idx++]);
            }
            // Dequantize
            dequantize(block);
            // IDCT
            idct_2d(block);
            // Store back
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

double calculate_psnr_raw(const vector<vector<vector<uint8_t>>>& orig, const vector<vector<vector<uint8_t>>>& recon) {
    double mse = 0.0;
    int H = orig.size(), W = orig[0].size(), C = orig[0][0].size();
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < C; c++) {
                int diff = static_cast<int>(orig[y][x][c]) - static_cast<int>(recon[y][x][c]);
                mse += diff * diff;
            }
        }
    }
    mse /= (H * W * C);
    return (mse == 0.0) ? INFINITY : 10.0 * log10((255.0 * 255.0) / mse);
}

vector<vector<vector<uint8_t>>> mat_to_array(const Mat& img) {
    int H = img.rows, W = img.cols;
    vector<vector<vector<uint8_t>>> arr(H, vector<vector<uint8_t>>(W, vector<uint8_t>(3)));
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            Vec3b p = img.at<Vec3b>(y, x);
            for (int c = 0; c < 3; c++) arr[y][x][c] = p[c];
        }
    }
    return arr;
}

vector<string> get_image_files(const string& folder_path) {
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

void save_compact_byte_stream(const string& filename,
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

void load_compact_byte_stream(const string& filename,
    int& width, int& height,
    vector<unsigned char>& compact_stream,
    map<short, string>& huffman_codes,
    size_t& bit_length) {
    ifstream in(filename, ios::binary);
    in.read(reinterpret_cast<char*>(&width), sizeof(int));
    in.read(reinterpret_cast<char*>(&height), sizeof(int));
    in.read(reinterpret_cast<char*>(&bit_length), sizeof(size_t));

    uint32_t table_size;
    in.read(reinterpret_cast<char*>(&table_size), sizeof(uint32_t));
    huffman_codes.clear();
    for (uint32_t i = 0; i < table_size; i++) {
        short val;
        uint8_t len;
        in.read(reinterpret_cast<char*>(&val), sizeof(short));
        in.read(reinterpret_cast<char*>(&len), sizeof(uint8_t));
        string code(len, '\0');
        in.read(&code[0], len);
        huffman_codes[val] = code;
    }

    uint32_t byte_count;
    in.read(reinterpret_cast<char*>(&byte_count), sizeof(uint32_t));
    compact_stream.resize(byte_count);
    in.read(reinterpret_cast<char*>(compact_stream.data()), byte_count);
    in.close();
}

double calculate_compression_ratio(size_t original_bytes, size_t compressed_bytes) {
    return static_cast<double>(original_bytes) / compressed_bytes;
}

void runSerial(const std::string& input_folder,
    const std::string& output_folder,
    const std::string& csv_path) {
    try {
        fs::create_directories(output_folder);
    }
    catch (const fs::filesystem_error& e) {
        cout << "Error creating output folder: " << e.what() << endl;
        return;
    }

    vector<string> image_files = get_image_files(input_folder);
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

        double t0 = getTickCount();
        vector<vector<unsigned char>> compact_streams(3);
        vector<map<short, string>> huffman_codes(3);
        vector<size_t> bit_lengths(3);

        string base_name = "image" + to_string(image_count);
        for (int c = 0; c < 3; c++) {
            vector<short> compressed_data;
            process_channel(channels[c], compressed_data, compact_streams[c], huffman_codes[c], bit_lengths[c], c, output_folder, base_name);
        }
        double time_ms = (getTickCount() - t0) / getTickFrequency() * 1000.0;

        // Save compact byte streams to disk
        int compact_total_size = 0;
        for (int c = 0; c < 3; c++) {
            string bin_path = (fs::path(output_folder) / (base_name + "_ch" + to_string(c) + ".bin")).string();
            save_compact_byte_stream(bin_path, W, H, compact_streams[c], huffman_codes[c], bit_lengths[c]);
            compact_total_size += static_cast<int>(fs::file_size(bin_path));
        }

        // Reconstruct
        vector<Mat> recon_ch(3);
        for (int c = 0; c < 3; c++) {
            recon_ch[c] = reconstruct_channel(compact_streams[c], huffman_codes[c], bit_lengths[c], H, W);
        }

        Mat recon;
        merge(recon_ch, recon);

        // Compute PSNR
        double psnr = calculate_psnr_raw(mat_to_array(img), mat_to_array(recon));

        // Calculate compression ratio based on actual .bin files
        int raw_size = W * H * img.channels();
        double reduction = 100.0 * (1.0 - static_cast<double>(compact_total_size) / raw_size);
        double compression_ratio = calculate_compression_ratio(raw_size, compact_total_size);

        // Save reconstructed image (optional visualization)
        string output_path = (fs::path(output_folder) / (base_name + ".jpg")).string();
        imwrite(output_path, recon, { IMWRITE_JPEG_QUALITY, 90 });

        // Pixel diff stats
        Mat diff;
        absdiff(img, recon, diff);
        int min_diff = 255, max_diff = 0, changed = 0;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                Vec3b p = diff.at<Vec3b>(y, x);
                bool any = false;
                for (int k = 0; k < 3; k++) {
                    if (p[k] > 0) {
                        min_diff = min(min_diff, int(p[k]));
                        max_diff = max(max_diff, int(p[k]));
                        any = true;
                    }
                }
                if (any) changed++;
            }
        }
        if (changed == 0) min_diff = 0;

        // Log output
        cout << image_count << ". " << image_file << " → " << base_name
            << " (" << W << "x" << H << ", " << fs::file_size(image_file)
            << " bytes, comp=" << compact_total_size << ")\n";
        cout << "Compression Ratio (orig/comp): " << fixed << setprecision(2) << compression_ratio << "\n";
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
}