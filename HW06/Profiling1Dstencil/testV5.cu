#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <cuda.h>
#include <math.h>

//#define N 1000000
#define RADIUS 3

int checkResults(int startElem, int endElem, float* cudaRes, float* res)
{
    int nDiffs=0;
    const float smallVal = 0.000001f;
    for(int i=startElem; i<endElem; i++)
        if(fabs(cudaRes[i]-res[i])>smallVal)
            nDiffs++;
    return nDiffs;
}

void initializeWeights(float* weights, int rad)
{
    // for now hardcoded for RADIUS=3
    weights[0] = 0.50f;
    weights[1] = 0.75f;
    weights[2] = 1.25f;
    weights[3] = 2.00f;
    weights[4] = 1.25f;
    weights[5] = 0.75f;
    weights[6] = 0.50f;
}

void initializeArray(float* arr, int nElements)
{
    const int myMinNumber = -5;
    const int myMaxNumber = 5;
    srand(time(NULL));
    for( int i=0; i<nElements; i++)
        arr[i] = (float)(rand() % (myMaxNumber - myMinNumber + 1) + myMinNumber);
}

void applyStencil1D_SEQ(int sIdx, int eIdx, const float *weights, float *in, float *out) {
  
  for (int i = sIdx; i < eIdx; i++) {   
    out[i] = 0;
    //loop over all elements in the stencil
    for (int j = -RADIUS; j <= RADIUS; j++) {
      out[i] += weights[j + RADIUS] * in[i + j]; 
    }
    out[i] = out[i] / (2 * RADIUS + 1);
  }
}

__global__ void applyStencil1D(int sIdx, int eIdx, const float *weights, float *input, float *out) {

__shared__ float s_x[1024 + 2 * RADIUS];
__shared__ float sw[2 * RADIUS + 1];


//int tx = threadIdx.x;
//int ty = threadIdx.y;
int ix = blockIdx.x*blockDim.x + threadIdx.x;
int iy = blockIdx.y*blockDim.y + threadIdx.y;
int x = threadIdx.x + RADIUS;
int tid = iy * blockDim.x * gridDim.x + ix;

s_x[x] = input[tid];

if(threadIdx.x < RADIUS)
{
s_x[x - RADIUS] = input[tid - RADIUS];
s_x[x + blockDim.x] = input[tid + blockDim.x];
}

if(threadIdx.x < 2* RADIUS + 1)
sw[threadIdx.x] = weights[threadIdx.x];
__syncthreads();

float result = 0.f;
if( tid < eIdx ) {
//        float result = 0.f;
        result += sw[0]*s_x[x-3];
        result += sw[1]*s_x[x-2];
        result += sw[2]*s_x[x-1];
        result += sw[3]*s_x[x];
        result += sw[4]*s_x[x+1];
        result += sw[5]*s_x[x+2];
        result += sw[6]*s_x[x+3];
        result /=7.f;

    }
out[tid] = result;
}

int main(int argc, char *argv[]) {

  
  if(argc != 2) {
    printf("Missing input N \n");
    exit(1);
  }

  int N = atoi (argv[1]);
  int size = N * sizeof(float); 
  int wsize = (2 * RADIUS + 1) * sizeof(float); 
  //allocate resources
  float *weights  = (float *) malloc(wsize);
  float *in       = (float *)malloc(size);
  float *out      = (float *)malloc(size); 
  float *cuda_out = (float *)malloc(size); 
  initializeWeights(weights, RADIUS);
  initializeArray(in, N);
  float *d_weights;  cudaMalloc((void**)&d_weights, wsize);
  float *d_in;       cudaMalloc((void**)&d_in, size);
  float *d_out;      cudaMalloc((void**)&d_out, size);

float tc,tg;
cudaEvent_t start1,stop1;
cudaEvent_t start2,stop2;
cudaEventCreate(&start1);
cudaEventCreate(&stop1);
cudaEventCreate(&start2);
cudaEventCreate(&stop2);

cudaEventRecord(start1, NULL);

  cudaMemcpy(d_weights,weights,wsize,cudaMemcpyHostToDevice);
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
 
  
dim3 grid;
    if (N<67108864)
      grid.x = (N + 1023)/1024;
    else {
      grid.x = 1 + (N + 1023)/1024/2;
      grid.y = 2;
    }
  dim3 block(1024,1,1);

  applyStencil1D<<<grid,block>>>(RADIUS, N-RADIUS, d_weights, d_in, d_out);
  cudaMemcpy(cuda_out, d_out, size, cudaMemcpyDeviceToHost);
 
cudaEventRecord(stop1, NULL);
cudaEventSynchronize(stop1);
cudaEventElapsedTime(&tg, start1, stop1);
  
  
cudaEventRecord(start2, NULL);
  applyStencil1D_SEQ(RADIUS, N-RADIUS, weights, in, out);

cudaEventRecord(stop2, NULL);
cudaEventSynchronize(stop2);
cudaEventElapsedTime(&tc, start2, stop2);
 
int nDiffs = checkResults(RADIUS, N-RADIUS, cuda_out, out);
  nDiffs==0? std::cout<<"Looks good.\n": std::cout<<"Doesn't look good: " << nDiffs << "differences\n";

  printf("GPU time = %f , CPU time = %f", tg*1000,tc*1000);

  //free resources
  cudaFree(weights); cudaFree(in); cudaFree(out); cudaFree(cuda_out);
  cudaFree(d_weights);  cudaFree(d_in);  cudaFree(d_out);
  return 0;
}
