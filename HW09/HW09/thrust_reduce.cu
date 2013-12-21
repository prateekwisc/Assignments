#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/reduce.h>
#include<thrust/scan.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<stdlib.h>
#define DATA_SIZE 66292994
int main(void)
{
 
  float t_reduce;
  cudaEvent_t start1,stop1, start2, stop2;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  //initialze random values on host
  thrust::host_vector<int> data(DATA_SIZE);
  
  thrust::generate(data.begin(), data.end(), rand);



//compute sum on host(CPU)
int h_sreduce = thrust::reduce(data.begin(), data.end());


//for inclusive time
cudaEventRecord(start1, NULL);
//copy values on device
  thrust::device_vector<int> gpudata = data;




cudaEventRecord(start2, NULL);
//compute sum on device(GPU)  
  int d_sreduce = thrust::reduce(gpudata.begin(), gpudata.end());
//copy back to host
  thrust::copy(gpudata.begin(), gpudata.end(), data.begin());

cudaEventRecord(stop1, NULL);
cudaEventSynchronize(stop1);
cudaEventElapsedTime(&t_reduce, start1, stop1);
printf("\n Reduce time is %f ms", t_reduce);
// thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

printf("\n host sum = %d, gpu sum = %d",h_sreduce,d_sreduce);

}
