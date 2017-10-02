#include<stdio.h> 
#include<stdlib.h>

#include "mgard_capi.h" 

int main(int argc, char* argv[])
{
  int i, nrow, ncol;

  nrow = 16;
  ncol = 60;
  
  double* in_buff = (double*) malloc(nrow*ncol*sizeof(double));
  unsigned char* mgard_comp_buff;
  
  for(i = 0; i<nrow*ncol; ++i)
    {
      in_buff[i] = i*i + 1 ;
    }

  double norm0 = 0;
  for(i = 0; i<nrow*ncol; ++i)
    {
      double temp = abs(in_buff[i]);
      if(temp > norm0) norm0 = temp;
    }

  
  int iflag = 1; //0 -> float, 1 -> double
  int out_size;
  double tol = 1e-8;

  mgard_comp_buff = mgard_compress(iflag, in_buff, &out_size,  nrow,  ncol, &tol );
  

  double* mgard_out_buff; 
  
  mgard_out_buff = mgard_decompress(iflag, mgard_comp_buff, out_size,  nrow,  ncol); 
  

  double norm = 0;
  for(i = 0; i<nrow*ncol; ++i)
    {
      double temp = abs((i*i+1.0) - mgard_out_buff[i]);
            
      if(temp > norm) norm = temp;
    }
  
  printf ("Rel. L-infty error: %8.5f \n", norm/norm0); 
      
}
