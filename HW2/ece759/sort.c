#include<stdio.h>

void main()
{
int a[10],i,j,n,t;


printf("\t Enter the size of an array");
scanf("\t %d",&n);


printf("\n Now enter the elements in the array");
for(i=0;i<n;i++)
{  scanf("%d",&a[i]);
}

printf("\n Array is-");

   for(i=0;i<n;i++)
     {
          printf("\t %d",a[i]);
     }
   
for(i=0;i<n;i++)
{
  for(j=0;j<n-i;j++)
      {
         if(a[j]>a[j+1])
           { 
             t = a[j];
             a[j] = a[j+1];
             a[j+1] = t;
           }
      }
}
printf("\n Array in ascending order is");
for(i=0;i<n;i++)
{ printf("\t %d",a[i]);
}
}
