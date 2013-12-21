#include "mpi.h"
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

using namespace std;
int main(int argc, char *argv[])
{
int n, rank, size, i;
double res,ans,h,sum,x;
char processor_name[MPI_MAX_PROCESSOR_NAME];
int namelen;

MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD,&size);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Get_processor_name(processor_name, &namelen);

printf("\n Hello from process %d of %d on %s ",rank, size, processor_name);

if (rank == 0){
if(argc<2 || argc>2)
  n=0;
else
  n=atoi(argv[1]);
}

MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
if(n>0){
h = 100.0/(double(n));
sum = 0.0;
for (i=rank +1; i<=n ; i+= size){
x = h * ((double)i - 0.5);
sum += exp(sin(x)) * cos((x)/40);
}
res = h * sum;
MPI_Reduce(&res, &ans,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
if(rank == 0)
  printf(" \n Value of integral is: %f", ans);
}
MPI_Finalize();
return 0;
}
