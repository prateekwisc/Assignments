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

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vector_reduction_gold.cpp"
#define BLOCK_SIZE 256

template <typename T>
__global__ void reduction(T *data_in, T *data_out)
{
    volatile __shared__ T sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // load half of data into shared memory
    if (tid < BLOCK_SIZE/2)
        sdata[tid] = data_in[globalIdx] + data_in[globalIdx + BLOCK_SIZE/2];
    __syncthreads();
    // unroll the loop
    if (BLOCK_SIZE >= 1024) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    }
    if (BLOCK_SIZE >= 512) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 64)  { sdata[tid] += sdata[tid + 64];  __syncthreads(); }
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 32)  { sdata[tid] += sdata[tid + 32];  __syncthreads(); }
    }
    // assume it's larger than 64
    if (tid < 16)  {
		{ sdata[tid] += sdata[tid + 16];   }
		{ sdata[tid] += sdata[tid + 8];    }
		{ sdata[tid] += sdata[tid + 4];    }
		{ sdata[tid] += sdata[tid + 2];    }
	}
    // write back to data_out[blockIdx.x]
    if (tid == 0) data_out[blockIdx.x] = sdata[0] + sdata[1];
}

// declaration, forward
void runTest(int argc, char** argv);
extern "C" void computeGold(double* reference, double* idata, const unsigned int len);
extern "C" void computeUnroll(double* reference, double* idata, const unsigned int len);

template<typename T>
T computeOnDevice(T* h_data, int len);

// Program main
int main(int argc, char** argv) 
{
	//// cudaSetDevice(1);
    runTest( argc, argv);
    return 0;
}

// Run test
void runTest(int argc, char** argv) 
{

  float tcpu;
  cudaEvent_t start1, stop1;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N: size of array> <M: range>\n", argv[0]);
        exit(1);
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);

    const int array_mem_size = sizeof(double) * N;

    cudaEventRecord(start1, NULL);
    // allocate host memory
    double *h_data = (double*) malloc(array_mem_size);

    // initialize the input data on the host; [-M, M]
    for (int i = 0; i < N; i++) {
        h_data[i] = 2 * M * (rand()/(double)RAND_MAX) - M;
    }

    // CPU version
    double reference = 0;
    double result = 0;

    computeGold(&reference, h_data, N);

    computeUnroll(&reference, h_data, N);

    // GPU version
    result = computeOnDevice(h_data, N);

cudaEventRecord(stop1, NULL);
cudaEventSynchronize(stop1);
cudaEventElapsedTime(&tcpu, start1, stop1);
printf("\n CPU time is: %f", tcpu);

    // Run accuracy test
    double epsilon = 0.0001;
    unsigned int result_regtest = (abs(result - reference) <= epsilon);
    fprintf(stdout, "host   result: %f\n", reference);
    fprintf(stdout, "device result: %f\n", result);
    fprintf(stdout, "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    // cleanup memory
    free(h_data);
}

// setup device computation
template<typename T>
T computeOnDevice(T* h_data, int len)
{
    
  float tgpu;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  T **d_data;
    d_data = (T**) malloc(2*sizeof(T*));
    // grid & block dimension
    int num_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // padding
    int d_len = num_blocks * BLOCK_SIZE;
  
    cudaEventRecord(start, NULL);
    
    // allocate space on device
  
    if (cudaMalloc((void**) &d_data[0], sizeof(T) * d_len) != cudaSuccess ||
            cudaMalloc((void**) &d_data[1], sizeof(T) * d_len) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc fails. Exit ...\n");
        exit(1);
    }
    // clear array for padded part
    cudaMemset(d_data[0], 0, sizeof(T)*d_len);
    cudaMemset(d_data[1], 0, sizeof(T)*d_len);

    // copy data to device
    cudaMemcpy(d_data[0], h_data, sizeof(T)*len, cudaMemcpyHostToDevice);
    // multiple iteration of the kernel
    int in = 0;
    int last_grid_size = 0;
    for (int grid_size = num_blocks; last_grid_size != 1; ) {
        // grid size for next iter
        int next_grid_size = (grid_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // clear output array, only for region of interest
        cudaMemset(d_data[1-in], 0, next_grid_size*BLOCK_SIZE*sizeof(T));
        // kernel launch
        reduction<<<grid_size, BLOCK_SIZE>>>(d_data[in], d_data[1-in]);
        // flip data_in and data_out
        in = 1 - in;
        // new grid size
        last_grid_size = grid_size;
        grid_size = next_grid_size;
    }
    // copy result back
    cudaMemcpy(h_data, d_data[in], sizeof(T), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tgpu, start, stop);
    printf("\n GPU time for reduction is: %f", tgpu);
	cudaFree(d_data[0]);
	cudaFree(d_data[1]);

	free(d_data);

    return h_data[0];
}
