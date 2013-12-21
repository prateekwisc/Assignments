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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <fstream>
using namespace std;
// includes, project

// includes, kernels
#include "vector_reduction_kernel2.cuh"

#include "vector_reduction_gold.cpp"
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int ReadFile(double*, char* file_name, int numm);
double computeOnDevice(double* h_data, int array_mem_size);

extern "C" void computeGold( double* reference, double* idata, const unsigned int len);

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
//! Run test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
float tc;
cudaEvent_t start,stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

    int num_elements = atoi(argv[1]);
    //int errorM = 0;
	int M = atoi(argv[2]);

    const unsigned int array_mem_size = sizeof( double) * num_elements;

cudaEventRecord(start, NULL);
    double* h_data = (double*) malloc(array_mem_size);
	for( unsigned int i = 0; i < num_elements; ++i)
	{
		h_data[i] = 1.0*((rand()%(2*M+1))-M);
 	}


    // allocate host memory to store the input data

    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Read the input data array from the given file.
   /* switch(argc-1)
    {      
        case 1:  // One Argument-
            errorM = ReadFile(h_data, argv[1], NUM_ELEMENTS);
            if(errorM != 1)
            {
                printf("Error reading input file!\n");
                exit(1);
            }
        break;
        
		case 2: //Two Arguments
            		//errorM = ReadFile(h_data, argv[1], NUM_ELEMENTS);
			num_elements = atoi(argv[1]);
			M=atoi(argv[2]);
			
            		//errorM = ReadFile(h_data, argv[1], num_elements);
			for( unsigned int i = 0; i < num_elements; ++i) 
            {
                h_data[i] = 1.0*((rand()%(2*M+1))-M);
            }
		
		break;
		
        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
                h_data[i] = floorf(1000*(rand()/(double)RAND_MAX));
            }
        break;  
    }*/
    // compute reference solution
    double reference = 0.0f;  
    
computeGold(&reference , h_data, num_elements);


cudaEventRecord(stop, NULL);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&tc, start, stop);
printf("\n CPU time is: %f", tc);
    
    // **===-------- Modify the body of this function -----------===**
    double result = computeOnDevice(h_data, num_elements);
    // **===-----------------------------------------------------------===**


    // Run accuracy test
    double epsilon = 0.0001f;
    unsigned int result_regtest = (abs(result - reference) <= epsilon);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    printf( "device: %f  host: %f\n", result, reference);
    // cleanup memory
    free( h_data);
}




// **===----------------- Modify this function ---------------------===**
// Take h_data from host, copies it to device, setup grid and thread 
// dimensions, excutes kernel function, and copy result of scan back
// to h_data.
// Note: double* h_data is both the input and the output of this function.
double computeOnDevice(double* h_data, int num_elements)
{
float tg;
cudaEvent_t start1,stop1;
cudaEventCreate(&start1);
cudaEventCreate(&stop1);

  // placeholder
  int tpb,bpg;
  int size = num_elements * sizeof(double);
  double * d_data;
  
 cudaEventRecord(start1, NULL);

  cudaMalloc(&d_data, size);
  cudaMemcpy(d_data,h_data,size, cudaMemcpyHostToDevice);
 
 tpb = 1024;
 //bpg = (num_elements + tpb-1)/tpb;
 reduction<<<1,tpb>>>(d_data,num_elements);
 cudaMemcpy(h_data,d_data,size,cudaMemcpyDeviceToHost);

cudaEventRecord(stop1, NULL);
cudaEventSynchronize(stop1);
cudaEventElapsedTime(&tg, start1, stop1);
printf("\n GPU time is: %f", tg);

 return h_data[0]; 

} 
