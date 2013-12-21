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
//char message[100];
MPI_Status status;
char *data, *data1;
N = atoi(argv[1]);
/*for(int j=0; j<30; j++)
{
  N = pow(2,j);
}*/
size_t size = N;

MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
MPI_Comm_size(MPI_COMM_WORLD, &p);

data = (char*)malloc(N*sizeof(char));
data1 = (char*)malloc(N*sizeof(char));

for(i = 0; i<N; i++)
{
  data[i] =(char)rand()%20;
}


start = MPI_Wtime();
  //code for process 0
if(my_rank == 0)
{ 

MPI_Ssend(data, N, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

MPI_Recv(data1,N, MPI_CHAR,1, 0, MPI_COMM_WORLD, &status);
}
//code for process 1
if(my_rank ==1) {
MPI_Recv(data1,N, MPI_CHAR,0, 0, MPI_COMM_WORLD, &status);

MPI_Ssend(data, N, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
printf("received %ld \n", data);

}
end = MPI_Wtime();

printf("Time required for transfer for rank %d in milliseconds: %f \n ", my_rank,(end-start)*1000);
MPI_Finalize();
return 0;
}
