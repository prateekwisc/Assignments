#include<stdio.h>
void main(int argc, char* argv[])
{
int ctr;
for(ctr=1;ctr<argc;ctr++)
{
printf("Command line argument %d=%s",ctr,argv[ctr]);
}
}