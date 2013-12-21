#include<stdio.h>
#include<cuda.h>
#include<math.h>
#include<sys/time.h>

__global__
void Matadd(char* A, char*B, int N)
 {
      
 }

int main()
{     

for(int j=0;j<=25;j++)
{     
      
      cudaEvent_t start1,stop1,start2,stop2; 
      float time1,time2, time3;
      int i;
      int N = pow(2,j);
      size_t size = N;
printf ("\n The value of N is %d",N);

cudaEventCreate(&start1);
cudaEventCreate(&stop1);     

cudaEventCreate(&start2);
cudaEventCreate(&stop2);     


//allocate input matrices hA, hB, hC,refC in host memory
char* hA; cudaMallocHost(&hA, size);
char* hB; cudaMallocHost(&hB, size);

for(i=0;i<N;i++)
{
hA[i] = rand()%20-10;

 }
//allocate memory on the device at location A (GPU)
char* dA;
cudaMalloc((void**) &dA,size);


//allocate memory on the device at location B (GPU)
char* dB;
cudaMalloc((void**) &dB,size);

//timing start for inclusive timing
cudaEventRecord(start1, 0);
     
//copy vectors from host memory to devie memory
      cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);


cudaEventRecord(stop1, 0);

cudaEventSynchronize(stop1);

//invoke GPU kernel, with two blocks each having eight threads

      
int threadsperblock = 16;
int blockspergrid = (N + threadsperblock - 1)/ threadsperblock;


cudaEventRecord(start2, 0);

//timing start for exclusive timing
//cudaEventRecord(start2, 0);
Matadd<<<blockspergrid,threadsperblock>>>(dA,dB,N);


cudaMemcpy(hB, dB, size, cudaMemcpyDeviceToHost);


cudaEventRecord(stop2, 0);
     
cudaEventSynchronize(stop2);

cudaEventElapsedTime(&time1,start1,stop1);
      
cudaEventElapsedTime(&time2,start2,stop2);

printf("\n The Host to Device time  for location A in microseconds for 2 to power %d is %f respectively \n",j,time1);

printf("\n The Device to Host time  for location B in microseconds for 2 to power %d is %f respectively \n",j,time2);

time3 = time1 + time2;

printf("\n The total data transfer time  in microseconds for 2 to power %d is %f respectively \n",j,time3);


cudaFree(hA);
cudaFree(hB);

cudaFree(dA);
cudaFree(dB);
}
return 0;
}
