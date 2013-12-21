#include<stdio.h>
#include<cuda.h>
#include<math.h>
#include<sys/time.h>
#include<stdlib.h>
__global__
void Matadd(char* A,int N)
 {
      
 }

int main()
{     

  char newline = '\n';
  
FILE *fp;
fp = fopen("HW7_1-1.txt","w");

for(int j=0;j<=25;j++)
{     
      
      cudaEvent_t start1,stop1; 
      float time1;
      int i;
      int N = pow(2,j);
      size_t size = N;
printf ("\n The value of N is %d",N);

cudaEventCreate(&start1);
cudaEventCreate(&stop1);     



//allocate input matrices hA, hB, hC,refC in host memory
char* hA = (char*)malloc(size);


for(i=0;i<N;i++)
{
hA[i] = rand()%20-10;

 }
//allocate memory on the device (GPU)
char* dA;
cudaMalloc((void**) &dA,size);

//timing start for inclusive timing
cudaEventRecord(start1, 0);
     
//copy vectors from host memory to devie memory
      cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);

        
cudaEventRecord(stop1, 0);

cudaEventSynchronize(stop1);


//invoke GPU kernel, with two blocks each having eight threads

      
int threadsperblock = 16;
int blockspergrid = (N + threadsperblock - 1)/ threadsperblock;

//timing start for exclusive timing
//cudaEventRecord(start2, 0);
Matadd<<<blockspergrid,threadsperblock>>>(dA,N);

cudaMemcpy(hA, dA, size, cudaMemcpyDeviceToHost);



//cudaEventRecord(stop1, 0);
     
//cudaEventSynchronize(stop1);
cudaEventElapsedTime(&time1,start1,stop1);
      

printf("\n The transfer time in microseconds for 2 to power %d is %f respectively \n",j,time1 );

fwrite(&j,sizeof(j),1,fp);
fwrite(&time1,sizeof(time1),1,fp);
fwrite(&newline,sizeof(newline),1,fp);


cudaFree(hA);

cudaFree(dA);

}
fclose(fp);

return 0;
}
