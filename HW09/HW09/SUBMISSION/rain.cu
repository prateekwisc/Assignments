#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/sort.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/inner_product.h>
#include<iostream>
#include<thrust/copy.h>
//#include<binary_search.h>

using namespace thrust::placeholders;

template <typename Vector>
int compute_numdays_with_rainfall(const Vector &day)
{

  return thrust::inner_product(day.begin(), day.end() - 1, day.begin() + 1,0,thrust::plus<int>(),thrust::not_equal_to<int>());
}


template <typename Vector>
int count_days_where_rainfall_exceeded_5(Vector& day,Vector& measurement)
{
  size_t N = compute_numdays_with_rainfall(day);

  thrust::device_vector<int> day_a(N);
  thrust::device_vector<int> measurement_a(N);

  thrust::reduce_by_key(day.begin(), day.end(), measurement.begin(), day.begin(), measurement.begin());
return thrust::count_if(measurement.begin(), measurement.end(), _1 > 5);

}

//int result=  count_days_where_rainfall_exceeded_5(data.day[]);
//printf("\n Result = %d", result);

template<typename Vector>
void compute_rainfall_per_site(Vector& site, Vector& measurement)
{
  Vector tmp_site(site);
  Vector tmp_measurement(measurement);

  thrust::sort_by_key(tmp_site.begin(), tmp_site.end(), tmp_measurement.begin());
  thrust::reduce_by_key(tmp_site.begin(), tmp_site.end(), tmp_measurement.begin(), site.begin(), measurement.begin());

for(int i=0; i<5; i++)
{
  std :: cout<<"\n  measurement at site "<<site[i]<< "is "<< measurement[i];
}


}

//printf("\n measurement = %d", measurement.begin());

int main(void) {

  
  thrust::device_vector<int>temp(5);
  thrust::device_vector<int>day(15);
  thrust::device_vector<int>site(15);
  thrust::device_vector<int>measurement(15);
  thrust::device_vector<int>day_a(15);
  thrust::device_vector<int>measurement_a(15);

day[0]=0; day[1]=0; day[2]=1; day[3]=2; day[4]=5; day[5]=5;day[6]=6;day[7]=6;day[8]=7;day[9]=8;day[10]=9;day[11]=9;day[12]=9;day[13]=10; day[14]=11;
site[0]=2; site[1]=3; site[2]=0; site[3]=1; site[4]=1; site[5]=2; site[6]=0; site[7]=1; site[8]=2; site[9]=1; site[10]=3; site[11]=4; site[12]=0; site[13]=1; site[14]=2;measurement[0]=9;measurement[1]=5;measurement[2]=6;measurement[3]=3;measurement[4]=3;measurement[5]=8;measurement[6]=2;measurement[7]=6;measurement[8]=5;measurement[9]=10;measurement[10]=9;measurement[11]=11;measurement[12]=8;measurement[13]=4;measurement[14]=1;  

thrust::copy(day.begin(),day.end(),day_a.begin());
thrust::copy(measurement.begin(),measurement.end(),measurement_a.begin());



compute_rainfall_per_site(site, measurement);

int result = count_days_where_rainfall_exceeded_5(day_a, measurement_a);
printf("\n Days with rainfall exceeded 5 are : %d", result-1);


return 0;
}
