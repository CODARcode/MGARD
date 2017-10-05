// Copyright 2017, Brown University, Providence, RI.
//
//                         All Rights Reserved
//
// Permission to use, copy, modify, and distribute this software and
// its documentation for any purpose other than its incorporation into a
// commercial product or service is hereby granted without fee, provided
// that the above copyright notice appear in all copies and that both
// that copyright notice and this permission notice appear in supporting
// documentation, and that the name of Brown University not be used in
// advertising or publicity pertaining to distribution of the software
// without specific, written prior permission.
//
// BROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
// INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
// PARTICULAR PURPOSE.  IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR
// ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//
//
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.1
//
// This file is part of MGARD.
//
// MGARD is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// MGARD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with MGARD.  If not, see <http://www.gnu.org/licenses/>.




#include<stdio.h> 
#include<stdlib.h>
#include<math.h>
#include<string.h>

#include "mgard_capi.h" 

int main(int argc, char *argv[])
{
  int i, j, nrow, ncol;
  double tol;

  if(argc < 6)
    {
      fprintf (stderr, "%s: Not enough arguments! Usage: %s infile outfile tolerance\n", argv[0], argv[0]);
      return 1;
    }
  else
    {
      tol = atof(argv[5]);
    }
  

  FILE * pFile;
  long lSize;
  char * buffer;
  size_t result;

  pFile = fopen ( argv[1] , "rb" );
  if (pFile==NULL) {fputs ("File error",stderr); exit (1);}

  fseek (pFile , 0 , SEEK_END);
  lSize = ftell (pFile);
  rewind (pFile);

  buffer = (char*) malloc (sizeof(char)*lSize);
  if (buffer == NULL) {fputs ("Memory error",stderr); exit (2);}

  result = fread (buffer,1,lSize,pFile);
  if (result != lSize) {fputs ("Reading error",stderr); exit (3);}


  fclose (pFile);

  
  double* in_buff = (double*) malloc (sizeof(char)*lSize);

  memcpy (in_buff, buffer, lSize);

  unsigned char* mgard_comp_buff;
  
  double norm0 = 0;
  for(i = 0; i<nrow; ++i)
    {
      for(j = 0; j<ncol; ++j)
        {
          double temp = fabs(in_buff[ncol*i+j]);
          if(temp > norm0) norm0 = temp;
        }
    }

  int iflag = 1; //0 -> float, 1 -> double
  int out_size;

  mgard_comp_buff = mgard_compress(iflag, in_buff, &out_size,  nrow,  ncol, &tol );


  FILE *qfile;
  qfile = fopen ( argv[2] , "wb" );

  char* outbuffer = ((char*)mgard_comp_buff);
    
  result = fwrite (outbuffer, 1, out_size, qfile);
  fclose(qfile);
  
  
  printf ("In size:  %10d  Out size: %10d  Compression ratio: %10d \n", lSize, out_size, lSize/out_size);
  
  double* mgard_out_buff; 
  
  mgard_out_buff = mgard_decompress(iflag, mgard_comp_buff, out_size,  nrow,  ncol); 

  in_buff = ((double*)buffer);

  double norm = 0;
  for(i = 0; i<nrow; ++i)
    {
      for(j = 0; j<ncol; ++j)
        {
          double temp = fabs( in_buff[ncol*i+j] - mgard_out_buff[ncol*i+j] );
          if(temp > norm) norm = temp;
        }
    }

  free(mgard_out_buff);
  
  printf ("Rel. L-infty error: %10.5E \n", norm/norm0);

  if( norm/norm0 < tol)
    {
      printf("SUCCESS: Error tolerance met!\n");
      return 0;
    }
  else{
      printf("FAILURE: Error tolerance not met!\n");
      return 1;
  }
  

}
