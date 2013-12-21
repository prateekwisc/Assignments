#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main()
{
  int ch;
  int i, n;
  int x;
  FILE *pFile;
  pFile = fopen("prateekfile.bin","wb");
  
  if(pFile== NULL)
   {
    printf("cannot open file");
    exit(1);
   }
// seed random sumber genrator//

x=pow(2,12);

ch='\n';
for(i=0;i<x;i++)
{

n= rand()%20-10;
fwrite(&n, sizeof(n),1,pFile);
fwrite(&ch, sizeof(ch),1,pFile);

}
ch = EOF;
fwrite(&ch,sizeof(ch),1,pFile);
fclose(pFile);
pFile = fopen("prateekfile.bin","rb");


for(i=0;i<x;i++)
{
fread(&n,sizeof(n),1,pFile);
fread(&ch,sizeof(ch),1,pFile);
printf("\t %d",n);
}

return 0;
}
