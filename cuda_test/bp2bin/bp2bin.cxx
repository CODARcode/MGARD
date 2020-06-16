
#include <adios2.h>

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

#include <chrono>

int main(int argc, char *argv[]) {

  MPI_Init(NULL, NULL);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, nproc;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);


  int L = std::atoi(argv[1]);
  int L2 = std::atoi(argv[2]);
  int L3 = std::atoi(argv[3]);
  std::string bpfile = argv[4];
  int step = std::atoi(argv[5]);
  int d = std::atoi(argv[6]);

  int dims[3] = {};
  int coords[3] = {};
  const int periods[3] = {1, 1, 1};
  MPI_Dims_create(nproc, 3, dims);
  int npx = dims[0];
  int npy = dims[1];
  int npz = dims[2];

  MPI_Comm cart_comm;
  MPI_Cart_create(comm, 3, dims, periods, 0, &cart_comm);
  MPI_Cart_coords(cart_comm, rank, 3, coords);
  int px = coords[0];
  int py = coords[1];
  int pz = coords[2];

  int size_x = L / npx;
  int size_y = L2 / npy;
  int size_z = L3 / npz;

  if (px < L % npx) {
      size_x++;
  }
  if (py < L2 % npy) {
      size_y++;
  }
  if (pz < L3 % npz) {
      size_z++;
  }

  int offset_x = (L / npx * px) + std::min(L % npx, px);
  int offset_y = (L2 / npy * py) + std::min(L2 % npy, py);
  int offset_z = (L3 / npz * pz) + std::min(L3 % npz, pz);
    
  printf("rank %d, (%d, %d, %d)\n", rank, size_x, size_y, size_z);

  adios2::ADIOS adios(comm, adios2::DebugON);
  adios2::IO inIO = adios.DeclareIO("SimulationOutput");
  inIO.SetEngine("BP4");
  adios2::Engine reader = inIO.Open(bpfile, adios2::Mode::Read);
  adios2::Variable<double> inVarU = inIO.InquireVariable<double>("U");
  adios2::Dims shapeU = inVarU.Shape();
  adios2::Box<adios2::Dims> sel({offset_z, offset_y, offset_x},
                                {size_z, size_y, size_x});

  inVarU.SetSelection(sel);

  adios2::Variable<int> inStep = inIO.InquireVariable<int>("step");
  std::vector<int> step_data;

  std::vector<double> gs_data;

  for (int i = 0; i < step-1; i++) {
    adios2::StepStatus status = reader.BeginStep();
    if (status != adios2::StepStatus::OK) {
        std::cout << "Step error\n";
        break;
    }
    reader.Get(inVarU, gs_data);
    reader.Get(inStep, step_data);
    reader.EndStep();
    std::cout << "Skipping step: " << step_data[0] <<"/" << step << " data: " << gs_data.size() << std::endl; 
  }

  reader.BeginStep();
  reader.Get(inVarU, gs_data);
  reader.Get(inStep, step_data);
  reader.EndStep();
  std::cout << "Dumping step: " << step_data[0] <<"/" << step << " data: " << gs_data.size() << std::endl; 

  if (d == 3) {
    std::string bin_file3 = "gs_"+std::to_string(L) + "_" + std::to_string(L2) + "_" + std::to_string(L3) +"_3D_"+ std::to_string(rank) + ".dat";
    std::ofstream wf3(bin_file3, std::ios::out | std::ios::binary);
    wf3.write((char*)gs_data.data(), L*L2*L3 * sizeof(double));
    wf3.close();
  }
  if (d == 2) {
    std::string bin_file2 = "gs_"+std::to_string(L) + "_" + std::to_string(L2) + +"_2D_"+ std::to_string(rank) + ".dat";
    std::ofstream wf2(bin_file2, std::ios::out | std::ios::binary);
    wf2.write((char*)gs_data.data(), L*L2* sizeof(double));
    wf2.close();
  }

  MPI_Finalize();
  return 0;
}
