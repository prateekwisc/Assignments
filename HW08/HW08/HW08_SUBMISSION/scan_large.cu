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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <cutil>
#include<cuda.h>
#include<fstream>
#include<float.h>
// You can use any other block size you wish.
#define BLOCK_SIZE 1024
//#define DEFAULT_NUM_ELEMENTS 16777216 
#define MAX_RAND 1
#include"scan_gold.cpp"

// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls.
__global__ void prescanArray1(float *outArray, float *inArray, int numElements)
{

volatile extern __shared__ float temp[];


int tid = threadIdx.x + blockIdx.x*blockDim.x;
int tx = threadIdx.x;
int i = 1;

temp[2*tx] = inArray[2*tid];
temp[2*tx + 1] = inArray[2*tid + 1];


for(int j = blockDim.x; j > 0;j >>= 1)
{
 
__syncthreads();

 if(tx <j)
 {
   int x = i * (2*tx + 1)-1;
   int y = i * (2*tx + 2)-1;
   temp[y] += temp[x];
 }
 i <<=1;
}

if(tx == 0)
  temp[2*blockDim.x-1] = inArray[2*blockDim.x *(blockDim.x +1)-1];

for(int j = 1; j < blockDim.x<<1;j <<= 1)
{
 i >>=1;
 __syncthreads();

if(tx < j)
{
   int x = i * (2*tx + 1)-1;
   int y = i * (2*tx + 2)-1;
float d = temp[x];
temp[x] = temp[y];
temp[y] += d;
}
}

__syncthreads();
if(tx == 0) temp[0] = inArray[tid];
outArray[2 * tid] = temp[2 * tx];
outArray[2 * tid + 1] = temp[2 *tx + 1];
}

__global__ void prescanArray(float *outArray, float *inArray, int numElements)
{

volatile extern __shared__ float temp[];

int tid = threadIdx.x + blockIdx.x*blockDim.x;
int tx = threadIdx.x;
int pout = 0, pin=1;
temp[tx] = inArray[tid];

if(tid<numElements)
{
for(int offset = 1; offset < blockDim.x; offset <<= 1)
{
 pout = 1- pout;
 pin = 1-pin;

 if(tx >= offset)
   temp[pout* blockDim.x + tx] = temp[pin* blockDim.x + tx] + temp[pin* blockDim.x + tx - offset];
 else
   temp[pout * blockDim.x + tx] = temp[pin * blockDim.x + tx];
 __syncthreads();
}

outArray[tid] = temp[pout*blockDim.x + tx];
if(tx == blockDim.x-1)
  inArray[tid] = temp[pout*blockDim.x + tx];
}
}


__global__ void prescanArray2(float *out, float *in)
{
  volatile extern __shared__ float temp[];
 
int tid = threadIdx.x + blockIdx.x*blockDim.x;
int bid = blockIdx.x + 1;
int tx = threadIdx.x;

temp[tx] = out[tid + blockDim.x];

for(int k = 0; k<bid; k++)
  temp[tx] += in[blockDim.x + k*blockDim.x -1];
__syncthreads();

out[tid + blockDim.x] = temp[tx];
}



// **===-----------------------------------------------------------===**



////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int ReadFile(float*, char* file_name, int size);
void WriteFile(float*, char* file_name, int size);

//extern "C" 
//unsigned int compare( const float* reference, const float* data, 
//                     const unsigned int len);
extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
   // int errorM = 0;
    float device_time;
    float host_time;

         int  num_elements = atoi(argv[1]);
    // allocate host memory to store the input data
    unsigned int mem_size = sizeof( float) * num_elements;
   // float* h_data = (float*) malloc( mem_size);
    
   float* h_data; cudaMallocHost(&h_data, mem_size);
            // allocate host memory to store the input data
            mem_size = sizeof( float) * num_elements;
          // float* h_data = (float*) malloc( mem_size);

            for( unsigned int i = 0; i < num_elements; ++i) 
            {
                h_data[i] = (int)(rand() % MAX_RAND);
            }

    unsigned int j, result_regtest;
    float timer,timer1;
    cudaEvent_t start, stop,start1,stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    dim3 grid1,grid2;
    grid1.x = (num_elements + BLOCK_SIZE -1)/(BLOCK_SIZE);
    grid2.x = (num_elements + 2*BLOCK_SIZE-1)/(2*BLOCK_SIZE);
    dim3 block(BLOCK_SIZE,1,1);
    int left = grid1.x -1;

    // compute reference solution
   // float* reference = (float*) malloc( mem_size);  

    float* reference; cudaMallocHost(&reference, mem_size);
    
    
    cudaEventRecord(start, NULL);
    computeGold( reference, h_data, num_elements);
	cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timer, start, stop);
    printf("\n\n**===-------------------------------------------------===**\n");
    printf("Processing %d elements...\n", num_elements);
    printf("Host CPU Processing time: %f (ms)\n", timer);
    host_time = timer;
   


    // allocate device memory input and output arrays
    float* d_idata = NULL;
    float* d_odata = NULL;
    float size = sizeof(float) * (BLOCK_SIZE*2);
    float size1 = sizeof(float) * (BLOCK_SIZE);
    
    cudaMalloc( (void**) &d_idata, mem_size);
    cudaMalloc( (void**) &d_odata, mem_size);
    
    // copy host memory to device input array
    cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice);
    // initialize all the other device arrays to be safe
  //  cudaMemcpy( d_odata, h_data, mem_size, cudaMemcpyHostToDevice);

    // **===-------- Allocate data structure here -----------===**
    // preallocBlockSums(num_elements);
    // **===-----------------------------------------------------------===**

cudaEventRecord(start1, NULL);
    
 prescanArray<<<grid1,block,size>>>(d_odata, d_idata,num_elements);

if(left > 0)
  prescanArray2<<<left, BLOCK_SIZE, size1>>>(d_odata, d_idata);

cudaMemcpy( h_data, d_odata, sizeof(float) * num_elements, cudaMemcpyDeviceToHost); 


cudaEventRecord(stop1, NULL);
cudaEventSynchronize(stop1);
cudaEventElapsedTime(&timer1, start1, stop1);
printf("CUDA Processing time: %f (ms)\n", timer1);
device_time = timer1;
printf("Speedup: %fX\n", host_time/device_time);


    // Check if the result is equivalent to the expected soluion
   double epsilon = 0.0001f;
  
   for(j=0; j<num_elements; ++j)
   {
       result_regtest = (abs( h_data[j] - reference[j]) <= epsilon);
   }
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    printf("device: %f host: %f\n", h_data, reference);

    // cleanup memory
   // cutDeleteTimer(timer);
   cudaFree( h_data);
    cudaFree( reference);
    cudaFree( d_odata);
    cudaFree( d_idata);
}



