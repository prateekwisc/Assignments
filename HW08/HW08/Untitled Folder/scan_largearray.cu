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
//#define BLOCK_SIZE 256
//#define DEFAULT_NUM_ELEMENTS 16000000 
#define MAX_RAND 1
#include"scan_gold.cpp"

// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls.
__global__ void prescanArray(float *outArray, float *inArray, int numElements)
{

volatile __shared__ float temp[2048];

int tx = threadIdx.x;
int pout = 0, pin =1;

//load input into shared memory
//Exclusive scan
if(tx == 0)
  temp[tx] = 0;
else
temp[tx] =  inArray[tx-1];
__syncthreads();

for(int offset = 1; offset < numElements; offset <<= 1)
{
 pout = 1- pout;
 pin = 1-pin;

 if(tx >= offset)
   temp[pout*numElements + tx] = temp[pin*numElements + tx] + temp[pin*numElements + tx - offset];
 else
   temp[pout * numElements + tx] = temp[pin * numElements + tx];
 __syncthreads();
}

outArray[tx] = temp[pout*numElements + tx];
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
  //  int* size = NULL; //(int*)malloc(1 * sizeof(int));
  //  unsigned int data2read = 1;
  //  int num_elements; // Must support large, non-power-of-2 arrays

         int  num_elements = atoi(argv[1]);
    // allocate host memory to store the input data
    unsigned int mem_size = sizeof( float) * num_elements;
    float* h_data = (float*) malloc( mem_size);
    int tpb;
   
    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Randomly generate input data and write the result to
    //   file name specified by first argument
    // * Two arguments: Read the first argument which indicates the size of the array,
    //   randomly generate input data and write the input data
    //   to the second argument. (for generating random input data)
    // * Three arguments: Read the first file which indicate the size of the array,
    //   then input data from the file name specified by 2nd argument and write the
    //   SCAN output to file name specified by the 3rd argument.
   /* switch(argc-1)
    {      
        case 2: 
            // Determine size of array
         errorM =   ReadFile(h_data, argv[1],NUM_ELEMENTS);
            if(data2read != 1){
                printf("Error reading parameter file\n");
                exit(1);
            } */

            // allocate host memory to store the input data
            mem_size = sizeof( float) * num_elements;
          // float* h_data = (float*) malloc( mem_size);

            for( unsigned int i = 0; i < num_elements; ++i) 
            {
                h_data[i] = (int)(rand() % MAX_RAND);
            }
          //  WriteFile(h_data, argv[2], num_elements);
       // break;
    
        /*case 3:  // Three Arguments
            ReadFile(argv[1], &size, &data2read, true);
            if(data2read != 1){
                printf("Error reading parameter file\n");
                exit(1);
            }

            num_elements = size[0];
            
            // allocate host memory to store the input data
            mem_size = sizeof( float) * num_elements;
            h_data = (float*) malloc( mem_size);

            errorM = ReadFile(h_data, argv[2], size[0]);
            if(errorM != 1)
            {
                printf("Error reading input file!\n");
                exit(1);
            }
        break;
        
        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            // Use DEFAULT_NUM_ELEMENTS num_elements
            num_elements = DEFAULT_NUM_ELEMENTS;
            
            // allocate host memory to store the input data
            mem_size = sizeof( float) * num_elements;
            h_data = (float*) malloc( mem_size);

            // initialize the input data on the host
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
//                h_data[i] = 1.0f;
                h_data[i] = (int)(rand() % MAX_RAND);
            }
        break;  
    }  */ 

    unsigned int j, result_regtest;
    float timer,timer1;
    cudaEvent_t start, stop,start1,stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    // compute reference solution
    float* reference = (float*) malloc( mem_size);  
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

    cudaMalloc( (void**) &d_idata, mem_size);
    cudaMalloc( (void**) &d_odata, mem_size);
    
    // copy host memory to device input array
    cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice);
    // initialize all the other device arrays to be safe
    cudaMemcpy( d_odata, h_data, mem_size, cudaMemcpyHostToDevice);

    // **===-------- Allocate data structure here -----------===**
    // preallocBlockSums(num_elements);
    // **===-----------------------------------------------------------===**

    // Run just once to remove startup overhead for more accurate performance 
    // measurement
    tpb = 1024;
   // prescanArray<<<1,tpb>>>(d_odata, d_idata, 16);

    // Run the prescan
    //cutCreateTimer(&timer);
    //cutStartTimer(timer);
    

	cudaEventRecord(start1, NULL);

    // **===-------- Modify the body of this function -----------===**
    prescanArray<<<1,tpb>>>(d_odata, d_idata, num_elements);
    // **===-----------------------------------------------------------===**
    
	cudaEventRecord(stop1, NULL);
        cudaEventSynchronize(stop1);
        cudaEventElapsedTime(&timer1, start1, stop1);
    
   // cudaThreadSynchronize();

  //  cutStopTimer(timer);
    printf("CUDA Processing time: %f (ms)\n", timer1);
    device_time = timer1;
    printf("Speedup: %fX\n", host_time/device_time);

    // **===-------- Deallocate data structure here -----------===**
    // deallocBlockSums();
    // **===-----------------------------------------------------------===**


    // copy result from device to host
    cudaMemcpy( h_data, d_odata, sizeof(float) * num_elements, 
                               cudaMemcpyDeviceToHost);

  /*  if ((argc - 1) == 3)  // Three Arguments, write result to file
    {
        WriteFile(h_data, argv[3], num_elements);
    }
    else if ((argc - 1) == 1)  // One Argument, write result to file
    {
        WriteFile(h_data, argv[1], num_elements);
    } */


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
    free( h_data);
    free( reference);
    cudaFree( d_odata);
    cudaFree( d_idata);
}


/*int ReadFile(float* M, char* file_name, int size)
{
	unsigned int elements_read = size;
	if (ReadFile(file_name, &M, &elements_read, true))
        return 1;
    else
        return 0;
}

void WriteFile(float* M, char* file_name, int size)
{
    WriteFile(file_name, M, size, 0.0001f);
}*/

