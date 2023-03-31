#!/bin/bash

DPATH=../dataset/
FILE=hires_atm.hifreq_write.F2010.ne120pg2_r0125_oRRS18to6v3.eam.h5.0001-12-27.bp
THRESH="1 7 3 3 3 0.025"
EB=1e-4
echo ./build/mgard_roi $DPATH $FILE PSL $EB $THRESH
mpirun -np 12 ./build/mgard_roi $DPATH $FILE PSL $EB 3 $THRESH 
