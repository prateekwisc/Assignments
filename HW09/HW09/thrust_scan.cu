#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/reduce.h>
#include<thrust/scan.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<stdlib.h>
#define DATA_SIZE 10000000
int main(void)
{
  int count = 0; 
  int d_scan; 
  float t_scan;
  cudaEvent_t start1,stop1;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  
  //initialze random values on host
  thrust::host_vector<int> data(DATA_SIZE);
  thrust::host_vector<int> h(DATA_SIZE);

  thrust::generate(data.begin(), data.end(), rand);
  thrust::exclusive_scan(data.begin(), data.end(), h);

cudaEventRecord(start1,NULL);


//copy host vector to device

  thrust::device_vector<int> gpudata = data;

thrust::exclusive_scan(gpudata.begin(), gpudata.end(), gpudata.begin());



//copy back to host
  thrust::copy(gpudata.begin(), gpudata.end(), data.begin());

cudaEventRecord(stop1, NULL);
cudaEventSynchronize(stop1);
cudaEventElapsedTime(&t_scan, start1, stop1);
printf("\n Scan time is %f ms", t_scan);
// thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

 
for (int i= 0; i<DATA_SIZE; i++)
{
if(fabs(h[i] - gpudata[i])==0.001)
count++;
else
  break;
}

if (count<DATA_SIZE)
printf("\n Error!!");

else
printf("Looks good");

}
