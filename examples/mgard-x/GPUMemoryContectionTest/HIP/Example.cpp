#include "mgard/compress_x_lowlevel.hpp"

#include <iostream>
// #include <vector>
#include <cstdlib>
#include <mpi.h>
int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  mgard_x::DeviceRuntime<mgard_x::HIP>::SelectDevice(rank);

  printf("Rank %d selecting GPU %d\n", rank,
         mgard_x::DeviceRuntime<mgard_x::HIP>::GetDevice());

  int n = atoi(argv[1]);
  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  MPI_Barrier(MPI_COMM_WORLD);

  mgard_x::Byte *data;
  mgard_x::MemoryManager<mgard_x::HIP>::Malloc1D(data, (size_t)n, 0);
  mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);

  MPI_Barrier(MPI_COMM_WORLD);
  timer.end();
  MPI_Barrier(MPI_COMM_WORLD);
  if (!rank) {
    timer.print("allocation");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  MPI_Barrier(MPI_COMM_WORLD);

  mgard_x::MemoryManager<mgard_x::HIP>::Free(data, 0);
  mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);

  MPI_Barrier(MPI_COMM_WORLD);
  timer.end();
  MPI_Barrier(MPI_COMM_WORLD);
  if (!rank) {
    timer.print("free");
  }
  MPI_Finalize();
  return 0;
}