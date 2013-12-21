#include<stdio.h>
#include<stdlib.h>

int main()
{
  int ch = EOF;
  int i, n;
  long int x;
  FILE *pFile;
  pFile = fopen("prateekfile.bin","wb");
  /* char x[100];
  for(i=0; i<=sizeof(x);i++)
  fwrite(x, sizeof(x[0]),100, pFile);*/
  if(pFile== NULL)
   {
    printf("cannot open file");
    exit(1);
   }
// seed random sumber genrator//

x=rand()%10+1;

//ch='\n';
for(i=0;i<x;i++)
{

n= rand()%10000;
fwrite(&n, sizeof(n),1,pFile);
//fwrite(&ch, sizeof(ch),1,pFile);
//printf("\n %d",n);
}

fwrite(&ch,sizeof(ch),1,pFile);
fclose(pFile);
pFile = fopen("prateekfile.bin","rb");
/*if(sFile==NULL)
{
printf("cannot open file");
exit(1);
}*/

for(i=0;i<x;i++)
{
fread(&n,sizeof(n),1,pFile);
//fread(&ch,sizeof(ch),1,pFile);
printf("\t %d",n);
}

return 0;
}
