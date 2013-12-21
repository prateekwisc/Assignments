#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"omp.h"

int main(int argc, char* argv[])
{
  double start, end;

  int num = atoi(argv[1]);
  double step = 100./(double(num));
  double sum;
  int omp_get_max_threads();
  int omp_get_num_procs();

start = omp_get_wtime();  

#pragma omp parallel for reduction(+:sum)

   for(int i=0; i<num; i++)
   {
    
     double x = (i + 0.5)* step;
 
   sum += exp(sin(x))*cos((x)/40);

   }

// end = omp_get_wtime();
double integral = sum * step;

 end = omp_get_wtime();

  printf("***************************************************");
  printf("\n The value of integral is: %f", integral);
  printf("\n ****************************************************");
  printf("\n Timing result in milli seconds: %f", (end-start)*1000);

  printf("\n No. of processors available to the program: %d", omp_get_num_procs());
  
  printf("\n No. of max. threads available to the program: %d", omp_get_max_threads());
  
  return 0;
}
