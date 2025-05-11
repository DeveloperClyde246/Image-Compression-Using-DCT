// CUDA.cpp
#define _USE_MATH_DEFINES
#include "Cuda.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cctype>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

extern "C" void gpuCompress(
    const unsigned char* h_in,
    short* h_out,
    int width, int height,
    int blocksX, int blocksY
);


static const int CUDA_ZIGZAG[64] = {
     0, 1, 8,16, 9, 2, 3,10,
    17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,
    27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36,
    29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,
    53,60,61,54,47,55,62,63
};

static const int CUDA_QUANT[8][8] = {
    {3,2,2,3,5,8,10,12},
    {2,2,3,4,5,12,12,11},
    {3,3,3,5,8,11,14,11},
    {3,3,4,6,10,17,16,12},
    {4,4,7,11,14,22,21,15},
    {5,7,11,13,16,21,23,18},
    {10,13,16,17,21,24,24,20},
    {14,18,20,20,22,23,24,22}
};

void cuda_dct_2d(double b[8][8]) {
    double tmp[8][8];
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            double sum = 0;
            double cu = (u == 0) ? 1.0 / sqrt(2) : 1.0;
            double cv = (v == 0) ? 1.0 / sqrt(2) : 1.0;
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    sum += b[x][y]
                        * cos((2 * x + 1) * u * M_PI / 16.0)
                            * cos((2 * y + 1) * v * M_PI / 16.0);
                }
            }
            tmp[u][v] = 0.25 * cu * cv * sum;
        }
    }
    memcpy(b, tmp, sizeof(tmp));
}

void cuda_idct_2d(double b[8][8]) {
    double tmp[8][8];
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            double sum = 0;
            for (int u = 0; u < 8; u++) {
                for (int v = 0; v < 8; v++) {
                    double cu = (u == 0) ? 1.0 / sqrt(2) : 1.0;
                    double cv = (v == 0) ? 1.0 / sqrt(2) : 1.0;
                    sum += cu * cv * b[u][v]
                        * cos((2 * x + 1) * u * M_PI / 16.0)
                            * cos((2 * y + 1) * v * M_PI / 16.0);
                }
            }
            tmp[x][y] = 0.25 * sum;
        }
    }
    memcpy(b, tmp, sizeof(tmp));
}

void cuda_quantize(double b[8][8]) {
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            b[i][j] = round(b[i][j] / CUDA_QUANT[i][j]);
}

void cuda_dequantize(double b[8][8]) {
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            b[i][j] *= CUDA_QUANT[i][j];
}

void cuda_printBlock(const double b[8][8], const std::string& lbl) {
    std::cout << "=== " << lbl << " ===\n";
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << std::setw(7) << std::fixed << std::setprecision(1)
                << b[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void cuda_debugOneBlock(const Mat& g, int sy, int sx) {
    double b[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            b[i][j] = double(g.at<uchar>(sy + i, sx + j)) - 128.0;
    cuda_printBlock(b, "1) After Shift");
    cuda_dct_2d(b);
    cuda_printBlock(b, "2) After DCT");
    cuda_quantize(b);
    cuda_printBlock(b, "3) After Quantize");
    cuda_dequantize(b);
    cuda_printBlock(b, "4) After Dequantize");
    cuda_idct_2d(b);
    cuda_printBlock(b, "5) After IDCT");
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            b[i][j] += 128.0;
    cuda_printBlock(b, "6) Final +128");
}

struct CudaHuffmanNode {
    short v; int f; CudaHuffmanNode* l; CudaHuffmanNode* r;
    CudaHuffmanNode(short vv, int ff) : v(vv), f(ff), l(nullptr), r(nullptr) {}
};
struct CudaCMP {
    bool operator()(CudaHuffmanNode* a, CudaHuffmanNode* b) const { return a->f > b->f; }
};

void cuda_buildHuff(CudaHuffmanNode* n,
    const std::string& c,
    std::map<short, std::string>& o)
{
    if (!n) return;
    if (!n->l && !n->r) o[n->v] = c.empty() ? "0" : c;
    cuda_buildHuff(n->l, c + "0", o);
    cuda_buildHuff(n->r, c + "1", o);
}

std::vector<std::pair<short, short>> cuda_rle_enc(const std::vector<short>& d) {
    std::vector<std::pair<short, short>> r;
    if (d.empty()) return r;
    short run = d[0], cnt = 1;
    for (size_t i = 1; i < d.size(); i++) {
        if (d[i] == run && cnt < 255) cnt++;
        else { r.emplace_back(run, cnt); run = d[i]; cnt = 1; }
    }
    r.emplace_back(run, cnt);
    return r;
}

std::vector<short> cuda_rle_dec(const std::vector<std::pair<short, short>>& r) {
    std::vector<short> o;
    for (auto& p : r)
        for (int i = 0; i < p.second; i++)
            o.push_back(p.first);
    return o;
}

std::pair<std::vector<unsigned char>, std::map<short, std::string>>
cuda_huff_enc(const std::vector<short>& d)
{
    std::map<short, int> fq;
    for (short v : d) fq[v]++;
    std::priority_queue<CudaHuffmanNode*, std::vector<CudaHuffmanNode*>, CudaCMP> pq;
    for (auto& kv : fq) pq.push(new CudaHuffmanNode(kv.first, kv.second));
    while (pq.size() > 1) {
        auto l = pq.top(); pq.pop();
        auto r = pq.top(); pq.pop();
        CudaHuffmanNode* p = new CudaHuffmanNode(0, l->f + r->f);
        p->l = l; p->r = r;
        pq.push(p);
    }
    std::map<short, std::string> cd;
    if (!pq.empty()) cuda_buildHuff(pq.top(), "", cd);
    std::string bits;
    bits.reserve(d.size() * 4);
    for (short v : d) bits += cd[v];
    std::vector<unsigned char> b((bits.size() + 7) / 8, 0);
    for (size_t i = 0; i < bits.size(); i++)
        if (bits[i] == '1')
            b[i / 8] |= (1 << (7 - (i % 8)));
    return { b, cd };
}

std::vector<short> cuda_huff_dec(const std::vector<unsigned char>& b,
    const std::map<short, std::string>& cd,
    size_t bl)
{
    std::map<std::string, short> rv;
    for (auto& kv : cd) rv[kv.second] = kv.first;
    std::string bits;
    bits.reserve(bl);
    for (size_t i = 0; i < bl; i++) {
        unsigned char x = b[i / 8];
        bits += ((x >> (7 - (i % 8))) & 1) ? '1' : '0';
    }
    std::vector<short> o;
    std::string cur;
    for (char bit : bits) {
        cur.push_back(bit);
        if (rv.count(cur)) {
            o.push_back(rv[cur]);
            cur.clear();
        }
    }
    return o;
}

Mat cuda_reconstructChannel(const std::vector<short>& zf,
    const std::vector<unsigned char>& hb,
    const std::map<short, std::string>& hc,
    size_t bl,
    int W, int H)
{
    Mat c(H, W, CV_8U);
    auto dec = cuda_huff_dec(hb, hc, bl);
    std::vector<std::pair<short, short>> runs;
    for (size_t i = 0; i + 1 < dec.size(); i += 2)
        runs.emplace_back(dec[i], dec[i + 1]);
    auto zig = cuda_rle_dec(runs);
    int bx = 0;
    for (int y = 0; y < H; y += 8)
        for (int x = 0; x < W; x += 8, bx++) {
            double b[8][8] = {};
            for (int k = 0; k < 64; k++) {
                int pos = CUDA_ZIGZAG[k];
                b[pos / 8][pos % 8] = zig[bx * 64 + k] * CUDA_QUANT[pos / 8][pos % 8];
            }
            cuda_idct_2d(b);
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++) {
                    int yy = y + i, xx = x + j;
                    if (yy < H && xx < W)
                        c.at<uchar>(yy, xx) = uchar(std::clamp(int(b[i][j] + 128.0), 0, 255));
                }
        }
    return c;
}

double cuda_calculatePSNR(const Mat& o, const Mat& r) {
    Mat d;
    absdiff(o, r, d);
    d.convertTo(d, CV_32F);
    d = d.mul(d);
    double mse = sum(d)[0] / (o.total() * o.channels());
    return mse == 0 ? INFINITY : 10 * log10((255 * 255) / mse);
}

static int cuda_extractNumber(const std::string& name) {
    int num = 0;
    bool inDigits = false;
    for (char c : name) {
        if (std::isdigit(c)) {
            inDigits = true;
            num = num * 10 + (c - '0');
        }
        else if (inDigits) {
            break;
        }
    }
    return num;
}

// Save Huffman�\encoded stream + codebook to a .bin file
void cuda_save_huffman_bin(const std::string& filename,
    int width, int height,
    const std::vector<unsigned char>& encoded_bytes,
    const std::map<short, std::string>& huffman_codes,
    size_t bit_length)
{
    std::ofstream out(filename, std::ios::binary);
    // dimensions + bit_length
    out.write(reinterpret_cast<const char*>(&width), sizeof(int));
    out.write(reinterpret_cast<const char*>(&height), sizeof(int));
    out.write(reinterpret_cast<const char*>(&bit_length), sizeof(size_t));
    // codebook
    uint32_t table_size = static_cast<uint32_t>(huffman_codes.size());
    out.write(reinterpret_cast<const char*>(&table_size), sizeof(uint32_t));
    for (auto& kv : huffman_codes) {
        out.write(reinterpret_cast<const char*>(&kv.first), sizeof(short));
        uint8_t len = static_cast<uint8_t>(kv.second.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(uint8_t));
        out.write(kv.second.data(), len);
    }
    // raw byte count + data
    uint32_t byte_count = static_cast<uint32_t>(encoded_bytes.size());
    out.write(reinterpret_cast<const char*>(&byte_count), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(encoded_bytes.data()), byte_count);
    out.close();
}

void runCuda(const std::string& input_folder,
    const std::string& output_folder,
    const std::string& csv_path)
{
    fs::create_directories(output_folder);
    std::ofstream csv(csv_path);
    csv << "Image,Width,Height,Pixels,Time_ms,Orig_Size,Comp_Size,Reduction,Pixel_Diff_Min,Pixel_Diff_Max,Changed_Pixels,PSNR\n";

    // Gather & sort files
    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(input_folder))
        if (e.is_regular_file()) files.push_back(e.path());
    std::sort(files.begin(), files.end(),
        [](auto const& a, auto const& b) {
            return cuda_extractNumber(a.filename().string())
                < cuda_extractNumber(b.filename().string());
        }
    );

    // Warm-up kernel
    {
        unsigned char dummy_in[64] = { 0 };
        short        dummy_out[64];
        gpuCompress(dummy_in, dummy_out, 8, 8, 1, 1);
    }

    int image_count = 1;
    for (auto& path : files) {
        cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (img.empty()) continue;

        int W = img.cols, H = img.rows;
        std::vector<cv::Mat> ch(3);
        cv::split(img, ch);

        // 1) Original per-channel sizes
        std::vector<size_t> orig_bits(3, size_t(W) * H * 8);
        std::vector<size_t> orig_bytes(3, size_t(W) * H);

        // 2) GPU DCT+quant �� zigzag
        double t0 = cv::getTickCount();
        int bX = (W + 7) / 8, bY = (H + 7) / 8;
        size_t nB = size_t(bX) * bY;
        std::vector<std::vector<short>> zig(3);
        for (int c = 0; c < 3; c++) {
            std::vector<unsigned char> h_in(W * H);
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    h_in[y * W + x] = ch[c].at<uchar>(y, x);
            zig[c].assign(nB * 64, 0);
            gpuCompress(h_in.data(), zig[c].data(), W, H, bX, bY);
        }
        double time_ms = (cv::getTickCount() - t0) / cv::getTickFrequency() * 1000.0;

        // 3) RLE �� Huffman �� save .bin �� reconstruct
        std::vector<size_t> comp_bits(3), comp_bytes(3);
        std::vector<cv::Mat> rec_ch(3);
        for (int c = 0; c < 3; c++) {
            // RLE encode
            auto runs = cuda_rle_enc(zig[c]);
            std::vector<short> flat;
            flat.reserve(runs.size() * 2);
            for (auto& p : runs) {
                flat.push_back(p.first);
                flat.push_back(p.second);
            }

            // Huffman encode
            auto enc = cuda_huff_enc(flat);
            size_t bitLen = 0;
            for (short v : flat) bitLen += enc.second[v].size();
            comp_bits[c] = bitLen;
            comp_bytes[c] = enc.first.size();

            // Save channel .bin
            {
                std::string base = "image" + std::to_string(image_count);
                std::string binf = (fs::path(output_folder)
                    / (base + "_ch" + std::to_string(c) + ".bin"))
                    .string();
                cuda_save_huffman_bin(binf, W, H, enc.first, enc.second, bitLen);
            }

            // Reconstruct channel
            rec_ch[c] = cuda_reconstructChannel(
                zig[c], enc.first, enc.second, bitLen, W, H
            );
        }


        // 4) Print per-channel stats (serial style)
        for (int c = 0; c < 3; c++) {
            std::cout
                << "Channel " << c
                << ": original bits = " << orig_bits[c]
                << " bits (" << orig_bytes[c]
                    << " bytes)\n";
                    std::cout
                        << "Channel " << c
                        << ": compressed = " << comp_bits[c]
                        << " bits (" << comp_bytes[c]
                            << " bytes)\n";
                            double ratio_bytes = double(orig_bytes[c]) / comp_bytes[c];
                            double ratio_bits = double(orig_bits[c]) / comp_bits[c];
                            std::cout << std::fixed << std::setprecision(2)
                                << "Channel " << c
                                << ": compression ratio = "
                                << ratio_bytes << ":1 (≈ "
                                << ratio_bits << ":1 bits-based)\n\n";
        }

        // 5) Merge and finalize
        cv::Mat recon;
        cv::merge(rec_ch, recon);
        int raw_size = W * H * img.channels();
        int comp_size = int(comp_bytes[0] + comp_bytes[1] + comp_bytes[2]);
        double reduction = 100.0 * (1.0 - double(comp_size) / raw_size);
        double psnr = cuda_calculatePSNR(img, recon);

        // Save output image
        std::string out_name = "image" + std::to_string(image_count) + ".jpg";
        cv::imwrite((fs::path(output_folder) / out_name).string(),
            recon, { cv::IMWRITE_JPEG_QUALITY, 90 });


        cout << fixed << setprecision(1)
            << "Execution time: " << time_ms << " ms\n";
        // 6) Serial-style summary
        std::cout
            << image_count << ". "
            << path.filename().string()
            << " -> " << out_name
            << " (" << W << "x" << H
            << ", " << fs::file_size(path)
            << " bytes, comp=" << comp_size << ")\n";
        std::cout
            << "PSNR: "
            << std::fixed << std::setprecision(2) << psnr
            << " dB\n\n";
        cout << "-------------------------------------------------------------------------------------------------" << "\n";

        // 7) Write CSV row
        csv << "\"" << path.filename().string() << "\","
            << W << "," << H << "," << (W * H) << ","
            << std::fixed << std::setprecision(1) << time_ms << ","
            << raw_size << "," << comp_size << ","
            << std::setprecision(1) << reduction << ","
            << /* min_diff */ 0 << "," << /* max_diff */ 0 << ","
            << /* changed */ 0 << "," << std::setprecision(2) << psnr
            << "\n";

        ++image_count;
    }

    csv.close();
}
