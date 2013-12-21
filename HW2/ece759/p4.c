#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<sys/stat.h>
#include<time.h>

void sort(int *,int);
int cmp (const void *, const void *);
int main(int argc, char *argv[])
{
  struct stat f_s;
  struct timeval start,end;
 
  float t2;
  int i,j, n=0,size;
  int delta; 
  int *arr;
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
printf("%d\n",n,arr[i]);
i++;
}
i=i-1;

//For Calculating the time of quick sort//
gettimeofday(&start, NULL);
qsort(arr,i,sizeof(int),cmp);

gettimeofday(&end, NULL);

t2= ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec*1000000 + start.tv_usec))/1000;
printf("time in millisec = %f",t2);
sort(arr,i);

printf("Size of file %d",size);

fclose(pFile);
free(arr);
return 0;
}

int cmp(const void *a, const void *b)
{
return (*(int*)a - *(int*)b);
}

void sort(int *a, int n)
{
struct timeval start1,end1;
int i,j;
int t;
float t1;
//Loop for calculating the time of our sorting//
gettimeofday(&start1, NULL);   
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

gettimeofday(&end1, NULL);

t1= ((end1.tv_sec * 1000000 + end1.tv_usec)-(start1.tv_sec * 1000000 + start1.tv_usec))/1000;

printf("\n Values of our sorting");
printf("\n min = %d, max = %d, total numbers = %d, time in millisec = %f",a[0],a[n-1],n,t1);
}






