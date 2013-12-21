#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<sys/stat.h>
#include<time.h>

void scan(long int *,long int *,long int *,long int);
int cmp (const void *, const void *);
int main(int argc, char *argv[])
{
  struct stat f_s;
  clock_t start1,end1;
  double t2;
  long int i,j, n=0,size;
  long long int delta; 
  long int *arr, *arr2, *arr3;
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

printf("Size of file %ld",size);

fclose(pFile);
free(arr);
free(arr2);
free(arr3);
return 0;
}


void scan(long int *a,long int *b,long int *c, long int n)
{

long int i;
long int t=0;
clock_t start,end;
double t1;

/*printf("\n Array is-");

   for(i=0;i<n;i++)
     {
          printf("\t %ld",a[i]);
     }*/
start = clock();   
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
end = clock();
t1 = (end-start)/CLOCKS_PER_SEC;

printf("\n Array is");
for(i=0;i<n;i++)
{ printf("\t %ld",c[i]);
}

printf("total numbers = %ld,last entry in scan array =%ld,time in sec = %f",n,c[n-1],t1);
}

