#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<sys/stat.h>

void sort(long int *,long int);
int main(int argc, char *argv[])
{
  struct stat f_s;
  long int i,j, n=0,size;

  long int *arr;
  FILE *pFile;
  
  pFile = fopen(argv[1],"r");

  if(pFile== NULL)
   {
    printf("cannot open file");
    exit(1);
   }
 i=0;

if(stat(argv[1],&f_s)<0)
return 1;
size = f_s.st_size;
arr = (malloc(size));

while(n!=EOF)
{
fread(&n,sizeof(n),1,pFile);
arr[i]=n;
fread(&n,sizeof(n),1,pFile);
printf("%ld \t %ld\n",n,arr[i]);
i++;
}
i=i-1;

sort(arr,i);

printf("Size of file %ld",size);

fclose(pFile);
free(arr);
return 0;
}


void sort(long int *a, long int n)
{
long int i,j;
long int t;
long long int delta;

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
printf("min = %ld, max = %ld, total numbers = %ld, time in ms = %lld",a[0],a[n-1],n,delta);
}



