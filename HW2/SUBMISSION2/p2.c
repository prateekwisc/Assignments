#include<stdio.h>
#include<string.h>

void main(int argc, char* argv[])
{
int len = 0;
char ctr;

if  (argc<2)
printf("\n Invalid string!");

else
for(ctr = 1; ((ctr<argc) && (ctr!=' '));ctr++)
{
len = len + strlen(argv[ctr]);
}
printf("\n Length of string is %d", len);
}
