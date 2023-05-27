#!/bin/bash

# set -e
set -x

exec=./build/adios2-test-external-compressor

IN_DATA=/home/jieyang/dev/data/xgc.f0.00200.bp
OUT_DATA=$IN_DATA.cmp.bp
DEC_DATA=$IN_DATA.dec.bp
# IN_DATA=/home/jieyang/dev/data/wrfout_d01_2019-11-26_23:50:00.bp
# OUT_DATA=/home/jieyang/dev/data/wrfout_d01_2019-11-26_23:50:00.mgard.bp
# DEC_DATA=/home/jieyang/dev/data/wrfout_d01_2019-11-26_23:50:00.mgard_dec_linf_1e-6.bp

# eb6=1.5e-5
# eb4=2.5e-3
eb2=2.5e-1

./build_script.sh
rm -rf $OUT_DATA
# ./build/adios2-test -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t s -n 2 1200 1500 -m rel -e $eb6 -s inf -v T2 -b 0 -d 0 
# ./build/adios2-test -x -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t s -n 2 1200 1500 -m rel -e $eb6 -s inf -v T2 -b 0 -d 0 

# IN_DATA=../../reconstructed_subarray1e3.dat
# IN_DATA=../../reconstructed_subarray1e2.dat
# IN_DATA=../../reconstructed_subarray1e1.dat
# IN_DATA=../../reconstructed_subarray1e0.dat
# IN_DATA=../../partial_reconstructed_subarray1e3.dat
# IN_DATA=../../partial_reconstructed_subarray1e2.dat
# IN_DATA=../../partial_reconstructed_subarray1e1.dat
# IN_DATA=../../partial_reconstructed_subarray1e0.dat
# OUT_DATA=$IN_DATA.bp
# ./build/adios2-test -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t s -n 2 1200 1500 -m rel -e $eb6 -s inf -v T2 -b 0 -d 0 


# ./build/adios2-test-external-compressor -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t d -n 3 264 10000 37 -m abs -e 25 -s inf -v i_f -b 0 -d 0 -p 0 -u 1
# ./build/adios2-test-external-compressor -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t d -n 3 264 10000 37 -m abs -e 25 -s inf -v i_f -b 0 -d 0 -p 1 -u 1

# ./build/adios2-test-external-compressor -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t s -n 2 1200 1500 -m abs -e 25 -s inf -v T2 -b 0 -d 0 -p 0
# ./build/adios2-test-external-compressor -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t s -n 2 1200 1500 -m abs -e 25 -s inf -v T2 -b 0 -d 0 -p 1


IN_DATA=$HOME/dev/data/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.f32
OUT_DATA=$IN_DATA.cmp
$exec -z -i $IN_DATA -c $OUT_DATA -o nocomp_${N}_${i}.csv -t s -n 3 512 512 512 -m abs -e 1e4 -s inf -v i_f -b 0 -d 0 -p 0 -u 0 -r 5 -k 1
# $exec -z -i $IN_DATA -c $OUT_DATA -o mgard_${N}_${i}.csv  -t s -n 3 512 512 512 -m abs -e 1e4 -s inf -v i_f -b 0 -d 0 -p 1 -u 0 -r 5 -k 1
# $exec -z -i $IN_DATA -c $OUT_DATA -o sz_${N}_${i}.csv     -t s -n 3 512 512 512 -m abs -e 1e3 -s inf -v i_f -b 0 -d 0 -p 2 -u 0 -r 5 -k 1
# $exec -z -i $IN_DATA -c $OUT_DATA -o zfp_${N}_${i}.csv    -t s -n 3 512 512 512 -m abs -e 11   -s inf -v i_f -b 0 -d 0 -p 3 -u 0 -r 10 -k 1
# $exec -z -i $IN_DATA -c $OUT_DATA -o lz4_${N}_${i}.csv    -t s -n 3 512 512 512 -m abs -e 11   -s inf -v i_f -b 0 -d 0 -p 4 -u 0 -r 10 -k 1

IN_DATA=$HOME/dev/data/summit.20220527.hires_atm.hifreq_write.F2010.ne120pg2_r0125_oRRS18to6v3.eam.h6.0001_side.bp
# IN_DATA=$HOME/dev/data/e3sm_PSL.dat
OUT_DATA=$IN_DATA.cmp.bp
DEC_DATA=$IN_DATA.dec.bp

N=1
# for i in {1..2}
# do
# $exec -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t s -n 3 2880 240 960 -m abs -e 1e3 -s inf -v PSL -b 0 -d 0 -p 0 -u 1 -r 5 -k 1 > nocmp_${N}_${i}.csv
# $exec -z -i $IN_DATA -c $OUT_DATA -o mgard_${N}_${i}.csv -t s -n 3 2880 240 960 -m abs -e 4e3 -s inf -v PSL -b 0 -d 0 -p 1 -u 1 -r 2 -k 1
# $exec -z -i $IN_DATA -c $OUT_DATA -o sz_${N}_${i}.csv -t s -n 3 2880 240 960 -m abs -e 1e2 -s inf -v PSL -b 0 -d 0 -p 2 -u 1 -r 2 -k 4
# $exec -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t s -n 3 2880 240 960 -m abs -e 6   -s inf -v PSL -b 0 -d 0 -p 3 -u 1 -r 2 -k 4 > zfp_${N}_${i}.csv
# done


IN_DATA=$HOME/dev/data/summit.20220527.hires_atm.hifreq_write.F2010.ne120pg2_r0125_oRRS18to6v3.eam.h5.0001-01-01-00000_side.bp
OUT_DATA=$IN_DATA.cmp.bp
DEC_DATA=$IN_DATA.dec.bp
# $exec -z -i $IN_DATA -c $OUT_DATA -o nocomp_${N}_${i}.csv -t s -n 3 720 240 960 -m abs -e 1e3 -s inf -v PSL -b 0 -d 0 -p 0 -u 1 -r 10 -k 1
# $exec -z -i $IN_DATA -c $OUT_DATA -o mgard_${N}_${i}.csv  -t s -n 3 720 240 960 -m abs -e 1e4 -s inf -v PSL -b 0 -d 0 -p 1 -u 1 -r 10 -k 1
# $exec -z -i $IN_DATA -c $OUT_DATA -o sz_${N}_${i}.csv     -t s -n 3 720 240 960 -m abs -e 1e2 -s inf -v PSL -b 0 -d 0 -p 2 -u 1 -r 10 -k 1
# $exec -z -i $IN_DATA -c $OUT_DATA -o zfp_${N}_${i}.csv    -t s -n 3 720 240 960 -m abs -e 6   -s inf -v PSL -b 0 -d 0 -p 3 -u 1 -r 10 -k 1

IN_DATA=$HOME/dev/data/d3d_coarse_v2_700.bin
OUT_DATA=$IN_DATA.cmp.bp
DEC_DATA=$IN_DATA.dec.bp
# $exec -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t d -n 3 50 50 50 -m abs -e 1e19 -s inf -v i_f -b 0 -d 0 -p 0 -u 0 -r 1
# $exec -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t d -n 3 50 50 50 -m abs -e 1e19 -s inf -v i_f -b 0 -d 0 -p 1 -u 0 -r 1
# $exec -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t d -n 3 50 50 50 -m abs -e 20 -s inf -v i_f -b 0 -d 0 -p 2 -u 0 -r 1

# eb6=2e-6
# eb4=2.3e-4
# eb2=2.6e-2

# ./build_script.sh
# rm -rf $OUT_DATA
# ./build/adios2-test -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t s -n 2 1200 1500 -m rel -e $eb2 -s 0 -v T2 -b 0 -d 2 
# ./build/adios2-test -x -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t s -n 2 1200 1500 -m rel -e $eb2 -s 0 -v T2 -b 0 -d 2 