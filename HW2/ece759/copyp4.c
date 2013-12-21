#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

void sort(long int *,int);
int main(int argc, char *argv[])
{
  
  int i,j, n=0;
  long int arr[30];
  FILE *pFile;
  
  pFile = fopen(argv[1],"r");
  /* char x[100];
  for(i=0; i<=sizeof(x);i++)
  fwrite(x, sizeof(x[0]),100, pFile);*/
  if(pFile== NULL)
   {
    printf("cannot open file");
    exit(1);
   }
 i=0;

//arr = (long int*)(malloc(1000000*sizeof(argv[1])));

while(n!=EOF)
{
fread(&n,sizeof(n),1,pFile);
arr[i]=n;

printf("%d \t %ld\n",n,arr[i]);
i++;
}
i=i-1;

sort(arr,i);

//printf("Size of argv %d",sizeof(argv[1]));

fclose(pFile);
//free(arr);
return 0;
}


void sort(long int *a, int n)
{
int i,j;
long int t;
long delta;
struct timeval start, end;
printf("\n Array is-");

   for(i=0;i<n;i++)
     {
          printf("\t %ld",a[i]);
     }
gettimeofday(&start, NULL);   
for(i=0;i<n;i++)
{
  for(j=0;j<(n-i-1);j++)
      {
         if(a[j]>a[j+1])
           { 
             t = a[j];
             a[j] = a[j+1];
             a[j+1] = t;
           }
      }
}
gettimeofday(&end, NULL);
delta = (end.tv_usec - start.tv_usec);

printf("\n Array in ascending order is");
for(i=0;i<n;i++)
{ printf("\t %ld",a[i]);
}
printf("min = %ld, max = %ld, total numbers = %d, time in ms = %ld",a[0],a[n-1],n,delta);
}



