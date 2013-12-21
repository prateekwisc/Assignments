#include<stdio.h>
#include<cuda.h>
#include<math.h>
#include<sys/time.h>

__global__
void Matadd(double* A, double* B, double* C, int N)
{
      
    int i = blockIdx.x* blockDim.x +threadIdx.x;
    if(i<N)
      C[i] = A[i] + B[i];
	__syncthreads();
 }

int main()
{     

for(int j=10;j<=20;j++)
{
      struct timeval start1,end1,start2,end2,start3,end3,start4,end4; 
      float time1, time2,time3,time4;
      int i;
      int N = pow(2,j);
     
size_t size = N * sizeof(double);

      
//allocate input matrices hA, hB, hC,refC in host memory
double* hA = (double*)malloc(size);
double* hB = (double*)malloc(size);
double* hC = (double*)malloc(size);
double* refC = (double*)malloc(size);


for(i=0;i<N;i++)
{
hA[i] = rand()%20-10;
hB[i] = rand()%20-10;

refC[i] = hA[i] + hB[i];
 }
//allocate memory on the device (GPU)
double* dA;
cudaMalloc(&dA,size);
double* dB;
cudaMalloc(&dB,size);
double* dC;
cudaMalloc(&dC,size);

gettimeofday(&start1, NULL);
//copy vectors from host memory to devie memory

      cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);

      cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
        
gettimeofday(&start2, NULL);      
//invoke GPU kernel, with two blocks each having eight threads
      
int threadsperblock = 32;
int blockspergrid = (N + threadsperblock - 1)/ threadsperblock;

Matadd<<<blockspergrid,threadsperblock>>>(dA,dB,dC,N);
gettimeofday(&end1, NULL);
//bring the result back from the device memory into the host array
      cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

gettimeofday(&end2, NULL);

time1 = ((end1.tv_sec*1000000 + end1.tv_usec) - (start1.tv_sec*1000000 + start1.tv_usec));
     
time2 = ((end2.tv_sec*1000000 + end2.tv_usec) - (start2.tv_sec*1000000 + start2.tv_usec));


printf("\n The inclusive time for 32 threads in microseconds for 2 to power %d is %f and exclusive time is %f respectively \n",j,time1,time2);

gettimeofday(&start3, NULL);

//copy vectors from host memory to devie memory

      cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);

      cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

gettimeofday(&start4, NULL);      

//invoke GPU kernel, with two blocks each having eight threads
      
threadsperblock = 1024;
blockspergrid = (N + threadsperblock - 1)/ threadsperblock;

Matadd<<<blockspergrid,threadsperblock>>>(dA,dB,dC,N);
gettimeofday(&end3, NULL);
//bring the result back from the device memory into the host array
      cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

gettimeofday(&end4, NULL);

time3 = ((end3.tv_sec*1000000 + end3.tv_usec) - (start3.tv_sec*1000000 + start3.tv_usec));
     
time4 = ((end4.tv_sec*1000000 + end4.tv_usec) - (start4.tv_sec*1000000 + start4.tv_usec));
/*printing hte result
printf("Values stored in hostarray: ");

for(i=0;i<N;i++)
{
printf("\t %f + %f = %f\n",hA[i], hB[i], hC[i]);
}
      
printf("Values stored in refernece array: ");

for(i=0;i<N;i++)
{
printf("\t %f \n",refC[i]);
} */

//printing the inclusive and exclusive time in microseconds

printf("The inclusive time for 1024 threads for 2 to power %d in microseconds is %f and exclusive time is %f respectively \n",j,time3,time4);
//release the memory allocated on the GPU

free(hA);
free(hB);
free(hC);
free(refC);

cudaFree(dA);
cudaFree(dB);
cudaFree(dC);
}
return 0;
}
