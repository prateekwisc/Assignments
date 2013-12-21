/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

// You can use any other block size you wish.
#define BLK_SIZE 256
#define DEFAULT_NUM_ELEMENTS 16777216
#define MAX_RAND 2

// reduce phase, reduction
template <typename T>
__global__ void reduce(T *data, int num_elements, int last_tid, int offset)
{
    volatile __shared__ T temp[BLK_SIZE];
    // load data into shared memory
    int uid = blockIdx.y*blockDim.x*gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int i1 = offset*(2*uid+1)-1;
    int i2 = i1 + offset;
    if (uid < num_elements) { // prevent out of range in last iteration
        temp[tid] = data[i1] + data[i2];  // one reduction on loading
    }
    __syncthreads();

    int n, d;
    for (d = 1, n = BLK_SIZE>>1; d < BLK_SIZE; d <<= 1, n >>= 1) {
        if (tid < n) {
            temp[d*(2*tid+2)-1] += temp[d*(2*tid+1)-1];
        }
        __syncthreads();
    }

    // write updated entry back to global memory
    if (uid < num_elements) {
        data[i2] = temp[tid];
    }
    if (tid == last_tid) {  // clear last element
        data[i2] = 0;
    }
}

// down-speed phase
template <typename T>
__global__ void down_sweep(T *data, int global_d, int num_elements, int offset)
{
    volatile __shared__ T temp[2*BLK_SIZE];
    // load data into shared memory
    int uid = blockIdx.y*blockDim.x*gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int i1 = offset*(2*uid+1)-1;
    int i2 = i1 + offset;
    if (i2 < num_elements) {
        temp[2*tid] = data[i1];
        temp[2*tid+1] = data[i2];
    }
    __syncthreads();

    int d, n;
    if (BLK_SIZE < global_d)
        d = BLK_SIZE;
    else
        d = global_d;

    for (n = BLK_SIZE/d; d > 0; d >>= 1, n <<= 1) {
        if (tid < n) {
            int a1 = d*(2*tid+1)-1;
            int a2 = a1 + d;
            T t = temp[a1];
            temp[a1] = temp[a2];
            temp[a2] += t;
        }
        __syncthreads();
    }

    // write updated entry back to global memory
    if (i2 < num_elements) {
        data[i1] = temp[2*tid];
        data[i2] = temp[2*tid+1];
    }
}

// kernel launch
template <typename T>
void prescanArray(T *Array, int numElements)
{
    int num_blocks;
    int offset;
    int n;

    // reduce phase
    for (offset = 1, n = numElements/2; offset < numElements; offset*=(2*BLK_SIZE), n/=(2*BLK_SIZE)) {
        num_blocks = (n + BLK_SIZE - 1) / (BLK_SIZE);
        int num_x = num_blocks, num_y = 1;
        if (num_blocks > 32768) {
            num_x = 32768;
            num_y = num_blocks / 32768;
        }
        dim3 grid_size(num_x, num_y);
        reduce<<<grid_size, BLK_SIZE>>>(Array, n, n-1, offset);
    }

    // down-sweep phase
    num_blocks = 1;
    n = numElements/2;
    for (offset = numElements/2; ; offset /= BLK_SIZE*2) {
        int finish_offset = offset / BLK_SIZE;
        if (finish_offset == 0) finish_offset = 1;  // last iteration

        if (num_blocks * BLK_SIZE > numElements)
            num_blocks = (numElements+BLK_SIZE-1)/BLK_SIZE;
        int num_x = num_blocks, num_y = 1;
        if (num_blocks > 32768) {
            num_x = 32768;
            num_y = num_blocks / 32768;
        }
        dim3 grid_size(num_x, num_y);
        down_sweep<<<grid_size, BLK_SIZE>>>(Array, n, numElements, finish_offset);
        num_blocks *= BLK_SIZE*2;
        n /= BLK_SIZE*2;
        if (finish_offset == 1) break;
    }
}

// declaration, forward
void runTest(int argc, char** argv);

template<typename T>
void computeGold(T* reference, T* idata, const unsigned int len);

template<typename T>
int compare(const T* reference, const T* data, const unsigned int len);

// Program main
int main(int argc, char** argv)
{
	// cudaSetDevice(1);
    runTest(argc, argv);
    return EXIT_SUCCESS;
}

//! Run a scan test for CUDA
void runTest(int argc, char** argv)
{
    int num_elements = 0;

    if (argc == 1) {
        num_elements = DEFAULT_NUM_ELEMENTS;
    } else if (argc == 2) {
        num_elements = atoi(argv[1]);
    } else {
        fprintf(stderr, "Usage: %s <# Elements>\n", argv[0]);
        exit(1);
    }

    // allocate host memory
    size_t mem_size = sizeof(double) * num_elements;
    double *h_data;

    cudaMallocHost(&h_data, mem_size);  // use pinned memory
    // initialize the input data
    for(unsigned int i = 0; i < num_elements; ++i)
    {
        h_data[i] = (int)(rand() % MAX_RAND);
        h_data[i] = 1;
    }

    // compute reference solution
    double* reference = (double*) malloc(mem_size);

    printf("Processing %d elements...\n", num_elements);
    computeGold(reference, h_data, num_elements);

    // padding
    int N = 0;
    while (num_elements > ( (unsigned int)1 << N ) ) {
        N++;
    }
    int d_num_elements = (unsigned int)1 << N;
    size_t d_mem_size = sizeof(double) * d_num_elements;
    printf("Padded to %d elements for GPU\n", d_num_elements);

    // allocate device memory
    double* d_data = NULL;

    if (cudaMalloc((void**) &d_data, d_mem_size) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc fails. Exit ...\n");
        exit(1);
    }

    // Run once to remove startup overhead for more accurate performance measurement
    prescanArray(d_data, 16);

    // copy data
    cudaMemcpy(d_data, h_data, mem_size, cudaMemcpyHostToDevice);

    // launch kernel
    prescanArray(d_data, d_num_elements);

    // copy result back, only copy un-padded part back
    cudaMemcpy(h_data, d_data, mem_size, cudaMemcpyDeviceToHost);

    // Check correctness
    int result_regtest = compare(reference, h_data, num_elements);
    if (result_regtest == 0)
        fprintf(stdout, "Test PASSED\n");
    else
        fprintf(stdout, "Test Failed: %d errors\n", result_regtest);

    cudaFreeHost(h_data);
    free(reference);
    cudaFree(d_data);
}

template <typename T>
void computeGold( T* reference, T* idata, const unsigned int len)
{
    reference[0] = 0;
    T total_sum = 0;
    for(unsigned int i = 1; i < len; ++i)
    {
        total_sum += idata[i-1];
        reference[i] = idata[i-1] + reference[i-1];
    }
    if (total_sum != reference[len-1])
        fprintf(stderr, "Warning: exceeding single-precision accuracy.  Scan will be inaccurate.\n");
}

template <typename T>
int compare(const T* reference, const T* data, const unsigned int len)
{
    int num_errors = 0;
    T eps = (T)0.0001;
    for (int i = 0; i < len; i++) {
        // relax eps as len grows
        eps = 0.0001 * log(len) / log(2);
        T error = reference[i] - data[i];
        if (abs(error) > eps) {
            num_errors++;
        }
    }
    return num_errors;
}
