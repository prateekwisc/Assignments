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
      
      cudaEvent_t start1,start2,start3,stop1,stop2,stop3,start4,stop4; 
      float time1,time2,time3, time4;
      int i;
      int N = pow(2,j);
      size_t size = N * sizeof(double);
printf ("\n The value of N is %d",N);

cudaEventCreate(&start1);
cudaEventCreate(&stop1);     

cudaEventCreate(&start2);
cudaEventCreate(&stop2);     

cudaEventCreate(&start3);
cudaEventCreate(&stop3);     

cudaEventCreate(&start4);
cudaEventCreate(&stop4);     


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

//timing start for inclusive timing
cudaEventRecord(start1, 0);
     
//copy vectors from host memory to devie memory
      cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);

      cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
        
//invoke GPU kernel, with two blocks each having eight threads

      
int threadsperblock = 32;
int blockspergrid = (N + threadsperblock - 1)/ threadsperblock;

//timing start for exclusive timing
cudaEventRecord(start2, 0);
Matadd<<<blockspergrid,threadsperblock>>>(dA,dB,dC,N);
//timing stop for exclusive timing
cudaEventRecord(stop2, 0);
cudaEventSynchronize(stop2);
      
cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

//timing stop for inclusive timing
cudaEventRecord(stop1, 0);
cudaEventSynchronize(stop1);

     
//timing start for inclusive timing

cudaEventRecord(start3, 0);


//copy vectors from host memory to devie memory
      cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);

      cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);


      
//invoke GPU kernel, with two blocks each having eight threads
threadsperblock = 1024;
blockspergrid = (N + threadsperblock - 1)/ threadsperblock;

//timing start for exclusive timing
cudaEventRecord(start4, 0);
Matadd<<<blockspergrid,threadsperblock>>>(dA,dB,dC,N);
//timing stop for exclusive timing
cudaEventRecord(stop4, 0);
cudaEventSynchronize(stop4);

//bring the result back from the device memory into the host array
      cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

cudaEventRecord(stop3, 0);
cudaEventSynchronize(stop3);

for (i=0;i<N;i++)
{
if(fabs(refC[i] - hC[i]) > 1e-12)
{
printf("Erratic Value \n");
exit(1);
}
}

     
cudaEventElapsedTime(&time1,start1,stop1);
      
cudaEventElapsedTime(&time2,start2,stop2);

printf("\n The inclusive time and exclusive time for 32 threads in microseconds for 2 to power %d is %f and %f respectively \n",j,time1,time2);


cudaEventElapsedTime(&time3,start3,stop3);

cudaEventElapsedTime(&time4,start4,stop4);

printf("\n The inclusive time and exclusive time for 1024 threads in microseconds for 2 to power %d is %f and %f respectively \n",j,time3,time4);

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
