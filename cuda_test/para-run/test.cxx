
#include <iostream>
#include <stdlib.h>
#include <iomanip>
#include <mpi.h>
#include <string>

#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <bits/stdc++.h> 

#include <mgard_api.h>
#include <mgard_api_cuda.h>

#include <chrono>


void refactorize(double * data, int nrow, int ncol, int nfib, 
                 std::string csv_prefix, int device) {
  mgard_cuda_handle * handle = new mgard_cuda_handle(32, csv_prefix);
  int out_size;
  double * mgard_refac_buff;
  if (device == 0) { // CPU
    mgard_refac_buff = (double *)mgard_compress(1, data, out_size, nrow, ncol, nfib, 0.1, csv_prefix);
  } else {
    mgard_refac_buff = (double *)mgard_compress_cuda(1, data, out_size, 
                                                    nrow, ncol, nfib, 
                                                    0.1, 3, 16, true, *handle);
  }

  double dummy = 0;
  out_size = nrow * ncol * nfib * sizeof(double);
  double * tmp_data;
  if (device == 0) {
    data = mgard_decompress(1, dummy, (unsigned char*)mgard_refac_buff, out_size,  nrow,  ncol, nfib, csv_prefix);
  } else {
    data = mgard_decompress_cuda(1, dummy, (unsigned char*)mgard_refac_buff, out_size,  
                                 nrow,  ncol, nfib, 3, 16, true, *handle);
  }

}


int main(int argc, char *argv[]) {

  MPI_Init(NULL, NULL);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, nproc;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);
  

  int L = atoi(argv[1]);
  int L2 = atoi(argv[2]);
  int L3 = atoi(argv[3]);
  int l = atoi(argv[4]);
  int l2 = atoi(argv[5]);
  int l3 = atoi(argv[6]);
  int n = atoi(argv[7]);
  int device = atoi(argv[8]);

  std::string root_csv_prefix = "";
  if (device == 0)  root_csv_prefix = "./results-cpu";
  if (device == 1)  root_csv_prefix = "./results-gpu";
  std::string cmd_rmdir = "rm -rf " + root_csv_prefix;
  if (rank == 0)
    std::system(cmd_rmdir.c_str());
  MPI_Barrier(comm);

  for (int i = (n/nproc)*rank; i < (n/nproc)*(rank+1); i++) {
  
    std::string infile = "";
    if (L3 == 1) {
      infile = "../bp2bin/gs_bin_data/gs_" + std::to_string(L) +
      "_" + std::to_string(L2) + "_2D_" + std::to_string(i) + ".dat";
    } else {
      infile = "../bp2bin/gs_bin_data/gs_" + std::to_string(L) +
      "_" + std::to_string(L2) + "_" + std::to_string(L3) + "_3D_" + 
      std::to_string(i) + ".dat";
    }

    std::cout << infile << "\n";

    FILE *pFile;
    double * buffer = new double[l * l2 * l3 * sizeof(double)];
    pFile = fopen ( infile.c_str() , "rb" );
    if (pFile==NULL) {fputs ("File error",stderr); exit (1);}
    fseek (pFile , 0 , SEEK_END);
    long lSize = ftell (pFile);
    
    rewind (pFile);

    long num_bytes = l * l2 * l3 * sizeof(double);
    lSize = num_bytes;

    if (lSize != num_bytes) {
      fprintf( stderr,
        "%s contains %lu bytes when %lu were expected. Exiting.\n",
        infile.c_str(), lSize, num_bytes );
      return 1;
    }

    size_t result = fread (buffer,1,lSize,pFile);
    if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
    fclose (pFile);

    
    std::string cmd_mkdir = "mkdir -p " + root_csv_prefix + "/" + std::to_string(rank) + "-" + std::to_string(i-(n/nproc)*rank);
    std::cout << cmd_mkdir << "\n";
    std::system(cmd_mkdir.c_str());

    std::string csv_prefix = root_csv_prefix + "/" +std::to_string(rank) + "-" + std::to_string(i-(n/nproc)*rank) + "/";

    MPI_Barrier(comm);
    refactorize(buffer, l, l2, l3, csv_prefix, device);
    MPI_Barrier(comm);
  }

  MPI_Finalize();
  return 0;
}
