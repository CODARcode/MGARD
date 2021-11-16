#!/bin/bash
set -e
set -x
make -j 8
make install

###### Necessary CUDA profiler binaries #######
# For details, refer to: 
# (1) NCU: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
# (2) NVPROF: https://docs.nvidia.com/cuda/profiler-users-guide/index.html
NCU=/usr/local/cuda-11.4/bin/ncu
NVPROF=nvprof

####### User's executable binary #######
BIN_DOUBLE_REL="./bin/TestDoubleRelativeError"
BIN_FLOAT_REL="./bin/TestFloatRelativeError"
BIN_DOUBLE_ABS="./bin/TestDoubleAbsoluteError"
BIN_FLOAT_ABS="./bin/TestFloatAbsoluteError"
MgardSerialExec="./bin/MgardSerialExec"
MgardCudaExec="./bin/MgardCudaExec"
TestRefactor="./bin/test_refactor"
TestReconstructor="./bin/test_reconstructor"

TestRefactorOrg="/home/jieyang/dev/MDR/build/test/test_refactor"
TestReconstructorOrg="/home/jieyang/dev/MDR/build/test/test_reconstructor"

CPU=0
GPU=1




# ./convert.py ../512x512x512/velocity_x.dat 512 512 512 s data_513_d.dat 513 513 513 d 
# ./convert.py ../512x512x512/velocity_x.dat 512 512 512 s data_257_d.dat 257 257 257 d 
# ./convert.py ../512x512x512/velocity_x.dat 512 512 512 s data_129_d.dat 129 129 129 d 
# ./convert.py ../512x512x512/velocity_x.dat 512 512 512 s data_65_d.dat 65 65 65 d 
# ./convert.py ../512x512x512/velocity_x.dat 512 512 512 s data_33_d.dat 33 33 33 d 

# ./convert.py ../512x512x512/velocity_x.dat 512 512 512 s data_513_s.dat 513 513 513 s 
# ./convert.py ../512x512x512/velocity_x.dat 512 512 512 s data_257_s.dat 257 257 257 s 
# ./convert.py ../512x512x512/velocity_x.dat 512 512 512 s data_129_s.dat 129 129 129 s 
# ./convert.py ../512x512x512/velocity_x.dat 512 512 512 s data_65_s.dat 65 65 65 s 
# ./convert.py ../512x512x512/velocity_x.dat 512 512 512 s data_33_s.dat 33 33 33 s 

# BIN="./build/bin/mgard_cuda_float_test ./test.dat 65 65 65 1 1"


# BIN="./bin/mgard_cuda_double_test 0 1 100  0.001 1"
# BIN="./bin/mgard_cuda_double_test 0 2 10 10 0.01 1"
# BIN="./bin/mgard_cuda_double_test 0 3 64 5 12 0.00001 0 1"

BIN=$BIN_DOUBLE" 0 3 513 513 513 1 0 1"


# BIN="./bin/mgard_cuda_double_test 1 ../data_513_d.dat 3 513 513 513 0.1 0 1"
#BIN="./bin/mgard_cuda_float_test 1 ../../512x512x512/velocity_x.dat 3 512 512 512 1 0 1"
# BIN="./bin/mgard_cuda_float_test 0 3 513 513 513 1 1"



# BIN="./bin/mgard_cuda_double_test 0 3 70 170 170 0.001 0 1"
# BIN="./bin/mgard_cuda_double_test 0 4 7 7 7 7 1 1"
# BIN="./bin/mgard_cuda_double_test 0 4 30 50 50 50 0.00001 1"
# BIN="./bin/mgard_cuda_double_test 0 4 5 5 5 5 0.000001 1"
# BIN="./bin/mgard_cuda_double_test 0 5 5 5 5 5 5 0.000001 1"


# BIN="./build/bin/mgard_cuda_double_test 5 5 5 1 1"

test_amr () { 
   $1 0 2 9 9 0.001 0 1
}


test_cpu_gpu() {
   $1 0 1 17 0.1 0 $2


}
test_group_l2 () { 
   $1 0 1 100 0.001 0 $2
   $1 0 2 3 70 0.001 0 $2
   # $BIN_DOUBLE 1 ../data_513_d.dat 3 513 513 513 0.001 0 1
   $1 0 3 70 170 180 0.001 0 $2
   $1 0 3 808 80 80 0.001 0 $2
   $1 0 3 80 80 10000 0.001 0 $2
   $1 0 3 80 10000 80 0.001 0 $2
   $1 0 3 10000 80 80 0.001 0 $2
   $1 0 3 64 5 12 0.00001 0 $2
   $1 0 4 5 5 5 5 0.00001 0 $2
   $1 0 4 70 50 10 30 0.0001 0 $2
   # $BIN_FLOAT 1 ../../512x512x512/velocity_x.dat 3 512 512 512 0.001 0 1

   #NO TEST YET
   # $BIN_DOUBLE 0 5 5 5 5 5 5 0.000001 0 1
}


test_group_l_inf () { 
   # $EXEC random $1 1 5 $2 1e-4 1 $3
   # $EXEC random $1 2 3 70 $2 0.001 inf $3
   # $EXEC random $1 3 70 170 180 $2 0.001 inf $3
   # $EXEC random $1 3 808 80 80 $2 0.001 inf $3
   # $EXEC random $1 3 80 80 10000 $2 0.001 inf $3
   # $EXEC random $1 3 80 10000 80 $2 0.001 inf $3
   # $EXEC random $1 3 10000 80 80 $2 0.001 inf $3
   # $EXEC random $1 3 64 5 12 $2 0.00001 inf $3
   # $EXEC random $1 5 5 5 5 5 5 $2 0.0001 0 $3
   # $EXEC random $1 4 10 100 10 100 $2 0.1 inf $3

   $EXEC random $1 2 5 5 $2 0.1 inf $3

   #NO TEST YET
   # $BIN_DOUBLE 0 5 5 5 5 5 5 0.000001 0 1
}

# test_group_l_inf d rel $1

DATA=../../512x512x512/velocity_x.dat
# $MgardCudaExec -z -i $DATA -c $DATA.mgard -t s -n 3 512 512 512 -m rel -e 1e-3 -s 0 -l 1 -v
# $MgardCudaExec -z -i $DATA -c $DATA.mgard -t s -n 3 129 129 129 -m abs -e 1e6 -s inf -l 2 -v

# ./bin/test_flying_edges -i $DATA -n 3 512 512 512
# ./bin/test_flying_edges -i random -n 3 5 5 5


# $MgardCudaExec -z -i random -c random.out -t s -n 2 500 50 -m abs -e 1e-2 -s inf -l 1 -v

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

# BIN="$EXEC random d 3 10 5 5 rel 0.00001 inf gpu"
# BIN2="$EXEC random d 3 10 5 5 rel 0.0001 inf gpu"
# BIN3="$EXEC random d 3 10 5 5 rel 0.001 inf gpu"
# BIN4="$EXEC random d 3 10 5 5 rel 0.01 inf gpu"
# BIN5="$EXEC random d 3 10 5 5 rel 0.1 inf gpu"
# BIN="$EXEC $DATA d 2 12168 16395 abs 1e15 inf gpu"

# BIN="$EXEC $DATA d 1 100 abs 1e15 inf gpu"
# DATA=/home/jieyang/dev/data/pk.data
# DATA=/home/jieyang/dev/data/enst.dat

cd ../examples/gpu-cuda/CompareCpuAndGpu && rm -rf ../examples/gpu-cuda/CompareCpuAndGpu/build && ./build_script.sh && ./build/BatchTests random
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

test_real_data() {
   $BIN_FLOAT_REL 1 ../../512x512x512/velocity_x.dat 3 512 512 512 0.000001 0 1
   # $BIN_FLOAT_REL 1 ../../512x512x512/velocity_x.dat 3 5 5 5 0.01 0 1
   # $BIN_FLOAT_REL 1 $DATA 3 768 768 768 1 inf 1
   # $BIN_FLOAT_REL 1 $DATA 3 768 768 768 1 inf 1
}


test_perf () { 

	$BIN_DOUBLE_REL 0 5 5 5 5 5 5 0.000001 0 1
   # $BIN_DOUBLE_REL 0 3 513 513 513 1 0 1
   # $BIN_DOUBLE_REL 0 3 5 5 5 0.00001 0 0
   # $BIN_FLOAT_REL 1 ../../512x512x512/velocity_x.dat 3 512 512 512 0.1 0 1
}


# for N in 513 257 129 65 33 17 9
# do
	# BIN="./build/bin/mgard_cuda_double_test $N $N $N 1 1"
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
