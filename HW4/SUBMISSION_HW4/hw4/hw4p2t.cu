#include<stdio.h>
#include<cuda.h>
#include<math.h>



__global__
void Matmultkernel(int* A , int* b, int* C)
{
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int P = 0;

for(int k= 0; k<32; k++)
 { 
  int MA = A[col * 32 + k];
  int Mb = b[k];
  P = P + MA * Mb;
 }
    
  C[col] = P;   
}

int main()
{
  FILE * pFile;
int i,j;

pFile = fopen("problem1.out","w");    

int d_A[16][32];
int d_b[32][1];
int d_C[16][1]; 
float time;

cudaEvent_t start1, stop1;

cudaEventCreate(&start1);

cudaEventCreate(&stop1);

for(i = 0;i<16;i++)
 {
   for (j=0;j<32;j++)
    {
     d_A[i][j] = i + j;
    printf(" %d ",d_A[i][j]);
    }
 printf("\n");
 }
printf("\n");

 for(i=0; i<32; i++)
 {
   d_b[i][0] = i;
   printf(" %d \t", d_b[i][0]);
 }


size_t sizeA = 16 * 32 * sizeof(int);

size_t sizeb = 32 * sizeof(int);

size_t sizeC = 16 * sizeof(int);

int* A;
cudaMalloc(&A,sizeA);

int* b;
cudaMalloc(&b,sizeb);

int* C;
cudaMalloc(&C,sizeC);

//Allocate and Load A and B into device memory


cudaDeviceProp deviceProp;
const int currentDevice = 0;
if(cudaGetDeviceProperties(&deviceProp, currentDevice) == cudaSuccess)
  printf("Device %d: %s \n", currentDevice, deviceProp.name);


cudaEventRecord(start1, 0);

cudaMemcpy(A, d_A, sizeA, cudaMemcpyHostToDevice);

cudaMemcpy(b, d_b, sizeb, cudaMemcpyHostToDevice);


// Invoke kernel

Matmultkernel<<<1,16>>>(A, b, C);

//bring the result back from the device memory into the host 
cudaMemcpy(d_C, C, sizeC, cudaMemcpyDeviceToHost);

cudaEventRecord(stop1, 0);
cudaEventSynchronize(stop1);

for(i = 0; i<16;i++)
{ 
  printf("\n %d", d_C[i][0]);
  fprintf(pFile, "%d \n",d_C[i][0]);
}

fclose (pFile);
cudaFree(A);
cudaFree(b);
cudaFree(C);

cudaEventElapsedTime(&time, start1, stop1);
printf("\n Inclusive time is %f", time);
cudaEventDestroy(start1);
cudaEventDestroy(stop1);

return 0;
}
