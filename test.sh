#!/bin/bash
set -e
# set -x



make -j 8
make -s install > /dev/null

###### Necessary CUDA profiler binaries #######
# For details, refer to: 
# (1) NCU: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
# (2) NVPROF: https://docs.nvidia.com/cuda/profiler-users-guide/index.html
NCU=/usr/local/cuda-11.4/bin/ncu
NVPROF=nvprof

####### User's executable binary #######
MgardSerialExec="./bin/MgardSerialExec"
MgardCudaExec="./bin/mgard-x"
TestRefactor="./bin/test_refactor"
TestReconstructor="./bin/test_reconstructor"

TestRefactorOrg="/home/jieyang/dev/MDR/build/test/test_refactor"
TestReconstructorOrg="/home/jieyang/dev/MDR/build/test/test_reconstructor"

CPU=0
GPU=1


# test_group_l_inf d rel $1

DATA=../../512x512x512/velocity_x.dat
# $MgardCudaExec -z -i $DATA -c $DATA.mgard -t s -n 3 6 6 6    -m rel -e 1e-5 -s 0 -l 1 -v -d $1
# $MgardCudaExec -z -i $DATA -c $DATA.mgard -t s -n 3 5 5 5 -m rel -e 1e-3 -s 0 -l 1 -v -d $1
# $MgardCudaExec -z -i $DATA -c $DATA.mgard -t s -n 3 512 512 512 -m abs -e 1e6 -s 0 -l 1 -v -d $1
# $MgardCudaExec -z -i $DATA -c $DATA.mgard -t s -n 3 512 512 512 -m rel -e 1e-4 -s 0 -l 2 -v -d $1
# $MgardCudaExec -z -i $DATA -c $DATA.mgard -t s -n 3 129 129 129 -m abs -e 1e6 -s inf -l 2 -v -d $1

# ./bin/test_flying_edges -i $DATA -n 3 512 512 512
# ./bin/test_flying_edges -i random -n 3 5 5 5


# cuda-memcheck 
# valgrind 
# $MgardCudaExec -z -i random -c random.out -t d -n 3 5 5 5   -m abs -e 1e-3 -s inf -l 1 -v -d $1

# $MgardCudaExec -z -i random -c random.out -t d -n 3 129 129 129  -m rel -e 1e-4 -s inf -l 1 -v -d $1
# $MgardCudaExec -z -i random -c random.out -t d -n 1 1000 -m rel -e 1e-4 -s 0 -l 1 -v -d $1
# $MgardCudaExec -z -i random -c random.out -t s -n 1 5  -m abs -e 1 -s inf -l 1 -v -d cuda


# $MgardSerialExec -z -i $DATA -c $DATA.mgard -t s -n 3 512 512 512 -m abs -e 1e5 -s inf -v
# $MgardSerialExec -z -i $DATA -c $DATA.mgard -t s -n 3 129 129 129 -m abs -e 1e5 -s inf -v

# DATA=../../data/temperature_sides.dat
# $MgardCudaExec -z -i $DATA -c $DATA.mgard -t s -n 3 72 1444 359 -m rel -e 5e0 -s inf -l 2 -v
# $MgardCudaExec -x -c $DATA.mgard -d $DATA.1e-1.decompressed -v

# $MgardCudaExec -z -i $DATA -c $DATA.mgard -t s -n 3 72 1444 359 -m rel -e 4e-1 -s inf -l 2 -v
# $MgardCudaExec -x -c $DATA.mgard -d $DATA.1e-2.decompressed -v

# $MgardCudaExec -z -i $DATA -c $DATA.mgard -t s -n 3 72 1444 359 -m rel -e 4e-2 -s inf -l 2 -v
# $MgardCudaExec -x -c $DATA.mgard -d $DATA.1e-3.decompressed -v


# mkdir -p refactored_data
# $TestRefactor $DATA 1 32 3 3 3 3 
# $TestRefactor $DATA 8 32 3 512 512 512 
# $TestReconstructor $DATA 1 3 1e12 1e10 1e8 0


# mkdir -p refactored_data
# $TestRefactorOrg $DATA 3 32 3 512 512 512 
# $TestReconstructorOrg $DATA 1 1 1e5 0


# DATA=$HOME/dev/data/d3d_coarse_v2_700.bin
# $MgardCudaExec -z -i $DATA -c $DATA.mgard -t d -n 4 8 39 16395 39 -m abs -e 1e14 -s 0 -l 2 -v
# $MgardCudaExec -z -i $DATA -c $DATA.mgard -t d -n 3 312 16395 39 -m abs -e 1e15 -s inf -l 2 -v

# BIN="$EXEC $DATA d 1 100 abs 1e15 inf gpu"
# DATA=/home/jieyang/dev/data/pk.data
# DATA=/home/jieyang/dev/data/enst.dat
# 
cd ../examples/mgard-x/CompareMgardXAndCpu && rm -rf ../examples/gpu-cuda/CompareMgardXAndCpu/build && ./build_script.sh && ./build/BatchTests random $1
# cmake --build ../../vtk-m/build -j
# cmake --build ../examples/gpu-cuda/FlyingEdges/build 
# ../examples/gpu-cuda/FlyingEdges/build/FlyingEdges -i random -n 3 800 800 800 -s 1.5
# ../examples/gpu-cuda/FlyingEdges/build/FlyingEdges -i random -n 3 900 900 900 -s 1.5

# ../examples/gpu-cuda/FlyingEdges/build/FlyingEdges -i random -n 3 1000 1000 1000 -s 1.5

# ../examples/gpu-cuda/FlyingEdges/build/FlyingEdges -i $DATA -n 3 512 512 512 -s 2e6

# cd ../examples/gpu-cuda/FlyingEdges && rm -rf build && ./build_script.sh && ./build/FlyingEdges random -n 3 5 5 5


# $XGC_4D
# $XGC_3D
# $XGC_4D_CPU
# $XGC_3D_CPU
# $BIN
# $BIN2
# $BIN3
# $BIN4
# $BIN5


# for N in 513 257 129 65 33 17 9
# do
	# BIN="./build/bin/mgard_x_double_test $N $N $N 1 1"
	# $NVPROF --print-gpu-trace --csv --normalized-time-unit ms --log-file kernel_trace.csv $BIN
	# ./parse_trace.py kernel_trace.csv 0 pi_Ql_cpt2 cpt_to_pow2p1_add
# done

####### Get trace data of user's GPU kernels #######
# Single process version
# $NVPROF --print-gpu-trace --csv --log-file kernel_trace.csv $BIN
# MPI version
NPROC=2
# mpirun -np $NPROC $NVPROF --print-gpu-trace --csv --log-file kernel_trace.%q{OMPI_COMM_WORLD_RANK}.csv $BIN

####### Get trace data of CUDA runtime API calls that called by user's code #######
# Single process version
# $NVPROF --print-api-trace --csv --log-file api_trace.csv $BIN
# MPI version
NPROC=2
# mpirun -np $NPROC $NVPROF --print-api-trace --csv --log-file api_trace.%q{OMPI_COMM_WORLD_RANK}.csv $BIN

####### Kernel level performace data #######
# Refer here for a list of availble performance matrices: https://docs.nvidia.com/cupti/Cupti/r_main.html#r_host_raw_metrics_api 
# Following example measure the memory throughput of the 1st invocation of the kernel "_pi_Ql_cpt" 
# "sudo" is requred to access low level performance counters
# KERNEL=gpk_reo
# KERNEL=lpk_reo_1
KERNEL=Kernel
# KERNEL=lpk_reo_2
# KERNEL=ipk_2
# KERNEL=lpk_reo_3
# KERNEL=ipk_3
# KERNEL=lwpk
INVOCAION=1
# METRIC=l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second

BIN=./bin/test_encoding_warp
# sudo $NCU $BIN
# sudo $NCU --kernel-id ::$KERNEL:$INVOCAION --section LaunchStats --section Occupancy --target-processes all  $BIN
# sudo $NCU --kernel-id ::$KERNEL:$INVOCAION --metric $METRIC $BIN

# METRIC=l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second
# sudo $NCU --kernel-id ::$KERNEL:$INVOCAION --metric $METRIC $BIN

# METRIC=l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
# sudo $NCU --kernel-id ::$KERNEL:$INVOCAION --metric $METRIC $BIN

# METRIC=l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
# sudo $NCU --kernel-id ::$KERNEL:$INVOCAION --metric $METRIC $BIN

# for i in {1..500}
# do
#  test_group_l_inf $BIN_DOUBLE_REL 1
#  test_group_l_inf $BIN_DOUBLE_ABS 1
#  test_group_l_inf $BIN_FLOAT_REL 1
#  test_group_l_inf $BIN_FLOAT_ABS 1
# done


# test_cpu_gpu $BIN_DOUBLE_ABS 0
# 
# test_real_data

# $EXEC ../../512x512x512/velocity_x.dat s 3 512 512 512 abs 1e4 0 $1

# $EXEC random d 2 5 5 rel 0.1 0 $1

# test_amr $BIN_DOUBLE_REL

# for i in {1..1}
# do
# test_perf
# done



# for i in {5..64}
# do
	# $BIN_DOUBLE 0 3 64 5 5 0.00001 0 1
# done

# cuobjdump /home/jieyang/dev/MGARD/build/lib/libmgard.so -res-usage | grep -A 1 $KERNEL

# $NVPROF --print-gpu-trace $BIN 2> >(grep $KERNEL)

# $NVPROF --print-gpu-trace $BIN
# $NVPROF --print-gpu-trace --csv --normalized-time-unit ms --log-file kernel_trace.csv $BIN

# $NVPROF $BIN


# $NVPROF -f --export-profile timeline.prof $BIN_FLOAT_REL 1 ../../512x512x512/velocity_x.dat 3 512 512 512 0.1 inf 1
# $NVPROF $BIN
# KERNEL=_mass_multiply_1_cpt
# $NVPROF --print-gpu-trace $BIN 2> >(grep $KERNEL)
