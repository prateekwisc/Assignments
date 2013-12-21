#include"mpi.h"
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
int main(int argc, char *argv[])
{
double start, end;
int my_rank;
int p,N;
int source;
int dest;
double MPI_Wtime();
//int tag = 0;
int i;
MPI_Status status;
char *data;
N = atoi(argv[1]);
size_t size = N;

MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
MPI_Comm_size(MPI_COMM_WORLD, &p);

data = (char*)malloc(N*sizeof(char));

for(i = 0; i<N; i++)
{
  data[i] =(char)rand()%20;
}


start = MPI_Wtime();
  //code for process 0
if(my_rank == 0)
{ 

MPI_Ssend(data, N, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

}
//code for process 1
else {
MPI_Recv(data ,N, MPI_CHAR,0, 0, MPI_COMM_WORLD, &status);

printf("received %ld \n", data);

}
end = MPI_Wtime();

printf("Time required for transfer for rank %d in milliseconds: %f \n ", my_rank,(end-start)*1000);
MPI_Finalize();
return 0;
}
