#include<stdio.h>
#include<iostream>
#include<cuda.h>

__global__
void simpleKernel(int* data)
{

    data[(blockIdx.x* blockDim.x)+threadIdx.x] = blockIdx.x + threadIdx.x;
   //to print the result in the device array
      
     printf("\n %d   + \t %d \t %d",threadIdx.x, blockIdx.x, data[(blockIdx.x * blockDim.x)+ threadIdx.x]);
    
}

int main()
{
      int hostarray[16], *devarray,i;
//allocate memory on the device (GPU)
      cudaMalloc((void**) &devarray,sizeof(int)*16);
      
//invoke GPU kernel, with two blocks each having eight threads
      simpleKernel<<<2,8>>>(devarray);
//bring the result back from the GPU into the host array
      cudaMemcpy(&hostarray, devarray, sizeof(int)*16, cudaMemcpyDeviceToHost);

      
//printing hte result
printf("\n Values stored in hostarray: ");

for(i=0;i<16;i++)
{
printf("\t %d",hostarray[i]);
}
      
//release the memory allocated on the GPU
cudaFree(devarray);

return 0;
}
