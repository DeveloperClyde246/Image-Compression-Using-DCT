// File: dct_cuda_kernels.cu
#define _USE_MATH_DEFINES
#include <cuda_runtime.h>
#include <cmath>
#include <device_launch_parameters.h>  // if you need blockIdx/threadIdx defs

#define BLOCK_DIM 8
#define PI_F 3.14159265358979323846f

__constant__ int d_quant[BLOCK_DIM * BLOCK_DIM] = {
    3,2,2,3,5,8,10,12,
    2,2,3,4,5,12,12,11,
    3,3,3,5,8,11,14,11,
    3,3,4,6,10,17,16,12,
    4,4,7,11,14,22,21,15,
    5,7,11,13,16,21,23,18,
    10,13,16,17,21,24,24,20,
    14,18,20,20,22,23,24,22
};

__constant__ int d_zigzag[64] = {
     0, 1, 8,16, 9, 2, 3,10,
    17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,
    27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36,
    29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,
    53,60,61,54,47,55,62,63
};

__device__ void blockDCT(float blk[BLOCK_DIM][BLOCK_DIM]) {
    const float PI = 3.14159265358979323846f;
    float tmp[BLOCK_DIM][BLOCK_DIM];
    for (int u = 0;u < BLOCK_DIM;u++) {
        for (int v = 0;v < BLOCK_DIM;v++) {
            float sum = 0;
            float cu = (u == 0) ? 1.0f / sqrtf(2) : 1.0f;
            float cv = (v == 0) ? 1.0f / sqrtf(2) : 1.0f;
            for (int x = 0;x < BLOCK_DIM;x++) {
                for (int y = 0;y < BLOCK_DIM;y++) {
                    sum += blk[x][y]
                        * cosf((2 * x + 1) * u * PI / 16.0f)
                            * cosf((2 * y + 1) * v * PI / 16.0f);
                }
            }
            tmp[u][v] = 0.25f * cu * cv * sum;
        }
    }
    for (int i = 0;i < BLOCK_DIM;i++) {
        for (int j = 0;j < BLOCK_DIM;j++) {
            blk[i][j] = tmp[i][j];
        }
    }
}

extern "C" __global__ void compressBlocksKernel(
    const unsigned char* in,
    short* out_zigzag,
    int width, int height,
    int blocksX, int blocksY
) {
    int bx = blockIdx.x, by = blockIdx.y;
    if (bx >= blocksX || by >= blocksY) return;
    float blk[BLOCK_DIM][BLOCK_DIM];
    int sx = bx * BLOCK_DIM, sy = by * BLOCK_DIM;
    for (int i = 0;i < BLOCK_DIM;i++) {
        for (int j = 0;j < BLOCK_DIM;j++) {
            int x = sx + j, y = sy + i;
            blk[i][j] = (x < width && y < height) ? float(in[y * width + x]) - 128.0f : 0.0f;
        }
    }
    blockDCT(blk);
    for (int i = 0;i < BLOCK_DIM;i++) {
        for (int j = 0;j < BLOCK_DIM;j++) {
            blk[i][j] = roundf(blk[i][j] / float(d_quant[i * BLOCK_DIM + j]));
        }
    }
    int idx = by * blocksX + bx;
    short* out = out_zigzag + idx * (BLOCK_DIM * BLOCK_DIM);
    for (int k = 0;k < 64;k++) {
        int pos = d_zigzag[k];
        int ii = pos / BLOCK_DIM, jj = pos % BLOCK_DIM;
        out[k] = short(blk[ii][jj]);
    }
}

extern "C" void gpuCompress(
    const unsigned char* h_in,
    short* h_out,
    int width, int height,
    int blocksX, int blocksY
) {
    size_t inSize = width * height * sizeof(unsigned char);
    size_t outSize = size_t(blocksX) * blocksY * 64 * sizeof(short);
    unsigned char* d_in; short* d_z;
    cudaMalloc(&d_in, inSize);
    cudaMalloc(&d_z, outSize);
    cudaMemcpy(d_in, h_in, inSize, cudaMemcpyHostToDevice);
    dim3 grid(blocksX, blocksY), thr(1, 1);
    compressBlocksKernel << <grid, thr >> > (d_in, d_z, width, height, blocksX, blocksY);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_z, outSize, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_z);
}
