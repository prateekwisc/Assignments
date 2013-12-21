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

/* Matrix multiplication: C = A * B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
// includes, project
#include "2Dconvolution.h"
#include "2Dconvolution_gold.cpp"
#define radius 2

using namespace std;
////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
bool CompareResults(float* A, float* B, int elements, float eps);
bool ReadParams(int* params, int size, char* file_name);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);

////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
////////////////////////////////////////////////////////////////////////////////
__constant__ float dataM[KERNEL_SIZE][KERNEL_SIZE];

__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P)
{

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = tx + blockIdx.x * blockDim.x;
  int y = ty + blockIdx.y * blockDim.y;
  int tid = x + y * N.width;
  int i,j;

__shared__ float dataN[BLOCK_SIZE + radius * 2][BLOCK_SIZE + radius * 2];

i = x - radius;
j = y - radius;
if(i<0 || j<0)
  dataN[ty][tx] = 0;
else
  dataN[ty][tx] = N.elements[tid - radius - (N.width * radius)];


i = x + radius;
j = y - radius;
if ( i> N.width-1 || j < 0 )
dataN[ty][tx + 2*radius] = 0;
else 
dataN[ty][tx + 2*radius] = N.elements[tid + radius - (N.width * radius)];


i = x - radius;
j = y + radius;
if (i < 0 || j > N.height-1)
dataN[ty + 2*radius][tx] = 0;
else 
dataN[ty + 2* radius][tx] = N.elements[tid - radius + (N.width * radius)];

i = x + radius;
j = y + radius;
if ( i > N.width-1 || j > N.height-1)
dataN[ty + 2*radius][tx + 2*radius] = 0;
else 
dataN[ty + 2*radius][tx + 2*radius] = N.elements[tid + radius + (N.width * radius)]; 

__syncthreads();

// convolution
float s = 0.0;

for (i = 0; i < KERNEL_SIZE; i++)
for (j = 0; j < KERNEL_SIZE; j++)
s += dataN[ty + i][tx + j] * dataM[i][j];

if(tx < N.width && ty< N.height)
P.elements[tid] = s; 
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	Matrix  M;
	Matrix  N;
	Matrix  P;
    
	srand(2013);
	
        if(argc == 2) {
          int in = atoi(argv[1]);

		M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 1);

		N  = AllocateMatrix(in,in,1);

               P  = AllocateMatrix(N.height, N.width, 0);
        }
        else  if(argc != 5 && argc != 4 && argc!=2) 
	{
		// Allocate and initialize the matrices
		M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 1);
		N  = AllocateMatrix((rand() % 1024) + 1, (rand() % 1024) + 1, 1);
		P  = AllocateMatrix(N.height, N.width, 0);
	}
	else
	{
		// Allocate and read in matrices from disk
		int* params = (int*)malloc(2 * sizeof(int));
		unsigned int data_read = 2;
      	if(ReadParams(params, data_read, argv[1])){
         	printf("Error reading parameter file\n");
         	return 1;
      	}

		M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 0);
		N  = AllocateMatrix(params[0], params[1], 0);		
		P  = AllocateMatrix(params[0], params[1], 0);
		(void)ReadFile(&M, argv[2]);
		(void)ReadFile(&N, argv[3]);
	}

	// M * N on the device
    ConvolutionOnDevice(M, N, P);
    
    // compute the matrix multiplication on the CPU for comparison
    Matrix reference = AllocateMatrix(P.height, P.width, 0);
    
    
float t1;
cudaEvent_t start1, stop1;
cudaEventCreate(&start1);
cudaEventCreate(&stop1);
cudaEventRecord(start1, NULL);

computeGold(reference.elements, M.elements, N.elements, N.height, N.width);

cudaEventRecord(stop1, NULL);
cudaEventSynchronize(stop1);
cudaEventElapsedTime(&t1, start1, stop1);
printf("\n CPU execution time = %f", t1*1000);

    // in this case check if the result is equivalent to the expected soluion

    bool res = CompareResults(reference.elements, P.elements, P.width * P.height, 0.01f);;
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    
    if(argc == 5)
    {
		WriteFile(P, argv[4]);
	}
	else if(argc == 2)
	{
	    WriteFile(P, argv[1]);
	}   

	// Free matrices
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{

  
    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    
    Matrix Nd = AllocateDeviceMatrix(N);
    

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    

    // Setup the execution configuration
dim3 grid((P.width + BLOCK_SIZE -1)/BLOCK_SIZE, (P.height + BLOCK_SIZE-1)/BLOCK_SIZE, 1);
dim3 block(BLOCK_SIZE,BLOCK_SIZE, 1);

float t;
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, NULL);
//CopyToDeviceMatrix(Md, M);

cudaMemcpyToSymbol(dataM, M.elements, M.width * M.height * sizeof(float));
CopyToDeviceMatrix(Nd,N);
CopyToDeviceMatrix(Pd,P);//Clear Memory

 
    // Launch the device computation threads!
ConvolutionKernel<<<grid, block>>>(Md,Nd,Pd);
    // Read P from the device

CopyFromDeviceMatrix(P, Pd); 
cudaEventRecord(stop, NULL);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&t, start, stop);



   // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);

    printf("\n GPU execution time = %f", t*1000);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
		if(rand() % 2)
			M.elements[i] = - M.elements[i];
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.

void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

//compare the data stored in two arrays on the host
bool CompareResults(float* A, float* B, int elements, float eps)
{
   for(unsigned int i = 0; i < elements; i++){
      float error = A[i]-B[i];
      if(error>eps){
         return false;
      } 
   }
   return true;
}

bool ReadParams(int* params, int size, char* file_name){
   ifstream ifile(file_name);
   int i=0;
   for(int i=0; i<size; i++){
      if(ifile.fail()==false){
         ifile>>params[i];
      }
   }
   return (i==size)? 1:0;

}

// Read a 16x16 floating point matrix in from file
int ReadFile(Matrix* M, char* file_name)
{
   unsigned int data_read = M->height * M->width;
   std::ifstream ifile(file_name);

   for(unsigned int i = 0; i < data_read; i++){
      ifile>>M->elements[i];
   }
   ifile.close();
   return data_read;

}



// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
   std::ofstream ofile(file_name);
   for(unsigned int i = 0; i < M.width*M.height; i++){
      ofile<<M.elements[i];
   }
   ofile.close();
}

