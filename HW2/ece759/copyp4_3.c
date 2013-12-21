#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<sys/stat.h>

void sort(int *,int);
int main(int argc, char *argv[])
{
  struct stat f_s;
  int i,j, n=0,size;

  int *arr;
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

if(stat(argv[1],&f_s)<0)
return 1;
size = f_s.st_size;
arr = (malloc(size));

while(n!=EOF)
{
fread(&n,sizeof(n),1,pFile);
arr[i]=n;
fread(&n,sizeof(n),1,pFile);
printf("%d \t %d\n",n,arr[i]);
i++;
}
i=i-1;

sort(arr,i);

printf("Size of file %d",size);

fclose(pFile);
free(arr);
return 0;
}


void sort(int *a, int n)
{
long int i,j;
long int t;
long delta;
struct timeval start, end;
printf("\n Array is-");

   for(i=0;i<n;i++)
     {
          printf("\t %d",a[i]);
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
{ printf("\t %d",a[i]);
}
printf("min = %d, max = %d, total numbers = %d, time in ms = %ld",a[0],a[n-1],n,delta);
}



