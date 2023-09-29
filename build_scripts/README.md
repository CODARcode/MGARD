# Build scripts for MGARD
The build scripts provided here are example scripts for building MGARD on systems with different processors. Each of them can also be modified for building MGARD other systems.

#### For CPU processors:

* `build_mgard_serial.sh` builds MGARD for single-core CPU
* `build_mgard_openmp_cpu.sh` builds MGARD for multi-thread CPUs using OpenMP
* `build_mgard_sycl_x86.sh` builds MGARD for multi-thread x86 CPUs using SYCL
* `build_mgard_apple_silicon.sh` builds MGARD for multi-thread Apple Silicon CPUs


#### For NVIDIA GPUs:
* `build_mgard_cuda_ampere.sh` builds MGARD for NVIDIA Ampere GPUs
* `build_mgard_cuda_turing.sh` builds MGARD for NVIDIA Turing GPUs
* `build_mgard_cuda_summit.sh` builds MGARD for NVIDIA Volta GPUs on the Summit supercomputer at OLCF
* `build_mgard_cuda_andes.sh` builds MGARD for NVIDIA Kepler GPUs on the Andes supercomputer at OLCF


#### For AMD GPUs:
* `build_mgard_hip_frontier.sh` builds MGARD for AMD MI-250X GPUs on the Frontier supercomputer at OLCF

#### For Intel GPUs:
* `build_mgard_sycl_gen9.sh` builds MGARD for Intel Gen9 integrated GPUs
* `build_mgard_sycl_xehp.sh` builds MGARD for Intel XeHP high performacne GPUs


#### For integration with ADIOS2 I/O library:
* `build_mgard_adios2_cuda_summit.sh` builds MGARD with ADIOS2 for NVIDIA Volta GPUs on the Summit supercomputer at OLCF
* `build_mgard_adios2_hip_frontier.sh` builds MGARD with ADIOS2 for AMD MI-250X GPUs on the Frontier supercomputer at OLCF