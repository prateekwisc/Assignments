#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/sort.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/inner_product.h>

using namespace thrust::placeholders;

struct Data
{
  thrust::device_vector<int> day;
   thrust::device_vector<int> site;
 thrust::device_vector<int> measurement;
  
}data;





/*struct one_site_measurement
{
 int siteofint;
 one_site_measurement(int site) : siteofint(site) {}
  
__host__ __device__
  int operator()(thrust::tuple<int,int> t)
  {
  if(thrust::get<0>(t) == siteofint)
    return thrust::get<1>(t);
  else 
    return 0;
  }
};



template <typename Vector>
int total_r(int siteID, const Vector& site, const Vector& measurement)
{
return thrust::transform_reduce
  (thrust::make_zip_iterator(thrust::make_tuple(site.begin(), measurement.begin())),
   thrust::make_zip_iterator(thrust::make_tuple(site.end(), measurement.end())),
   one_site_measurement(siteID),0,thrust::plus<int>());
}*/

template <typename Vector>
int compute_numdays_with_rainfall(const Data &data)
{

  return thrust::inner_product(data.day.begin(), data.day.end() - 1, data.day.begin() + 1,1,thrust::plus<int>(),thrust::not_equal_to<int>());
}

//using namespace thrust::placeholders;

template <typename Vector>
int count_days_where_rainfall_exceeded_5(const Data &data)
{
  size_t N = compute_numdays_with_rainfall(&data);

  thrust::device_vector<int> day(N);
  thrust::device_vector<int> measurement(N);

  thrust::reduce_by_key(data.day.begin(), data.day.end(), data.measurement.begin(), data.day.begin(), measurement.begin());
return thrust::count_if(measurement.begin(), measurement.end(), _1 > 5);
}

//int result=  count_days_where_rainfall_exceeded_5(data.day[]);
//printf("\n Result = %d", result);

template<typename Vector>
void compute_rainfall_per_site(const Data &data,Vector &site, Vector &measurement)
{
  Vector tmp_site(data.site);
  Vector tmp_measurement(data.measurement);

  thrust::sort_by_key(tmp_site.begin(), tmp_site.end(), tmp_measurement.begin());
  thrust::reduce_by_key(tmp_site.begin(), tmp_site.end(), tmp_measurement.begin(), site.begin(), measurement.begin());
}

//printf("\n measurement = %d", measurement.begin());

int main(void) {

  
  thrust::device_vector<int>temp(5);
  thrust::device_vector<int>day(15);
  thrust::device_vector<int>site(15);
  thrust::device_vector<int>measurement(15);


day[0]=0; day[1]=0; day[2]=1; day[3]=2; day[4]=5; day[5]=5;day[6]=6;day[7]=6;day[8]=7;day[9]=8;day[10]=9;day[11]=9;day[12]=9;day[13]=10; day[14]=11;
site[0]=2; site[1]=3; site[2]=0; site[3]=1; site[4]=1; site[5]=2; site[6]=0; site[7]=1; site[8]=2; site[9]=1; site[10]=3; site[11]=4; site[12]=0; site[13]=1; site[14]=2;measurement[0]=9;measurement[1]=5;measurement[2]=6;measurement[3]=3;measurement[4]=3;measurement[5]=8;measurement[6]=2;measurement[7]=6;measurement[8]=5;measurement[9]=10;measurement[10]=9;measurement[11]=11;measurement[12]=8;measurement[13]=4;measurement[14]=1;  


//day[15] = {0,0,1,2,5,5,6,6,7,8,9,9,9,10,11};
 //site[15] = {2,3,0,1,1,2,0,1,2,1,3,4,0,1,2};
 // measurement[15] = {9,5,6,3,3,8,2,6,5,10,9,11,8,4,1};


int result=  count_days_where_rainfall_exceeded_5(data.day);
printf("\n Result = %d", result);

//compute_rainfall_per_site(site,measurement);
//for(int i=0;i<5;i++)

//  printf("\n Result = %d", res);


return 0;
}
