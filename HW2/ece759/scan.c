#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<sys/stat.h>
#include<time.h>

void scan(int *,int *,int *,int);
int cmp (const void *, const void *);
int main(int argc, char *argv[])
{
  struct stat f_s;
  
  float t2;
  int i,j, n=0,size;
  int delta; 
  int *arr, *arr2, *arr3;
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
arr2 = (malloc(size));
arr3= (malloc(size));
while(n!=EOF)
{
fread(&n,sizeof(n),1,pFile);
arr[i]=n;
fread(&n,sizeof(n),1,pFile);

i++;
}
i=i-1;

scan(arr,arr2,arr3,i);

printf("Size of file %d",size);

fclose(pFile);
free(arr);
free(arr2);
free(arr3);
return 0;
}

//**************SCAN ALGORITHM***************//

void scan(int *a,int *b,int *c,int n)
{
struct timeval start,end;
long int i;
long int t=0;

double t1;
 
gettimeofday(&start, NULL);
for(i=0;i<n;i++)
   {   
    t += a[i];
    b[i] = t;
      }
c[0]=0;
for(i=0;i<n;i++)
{
c[i+1]=b[i];
}

gettimeofday(&end, NULL);

t1 = ((end.tv_sec*1000000 + end.tv_usec) - (start.tv_sec*1000000 + start.tv_usec));
printf("\n Array is");
for(i=0;i<n;i++)
{ printf("\t %d",c[i]);
}

printf("total numbers = %d,last entry in scan array =%d,time in microsec = %f",n,c[n-1],t1);
}

