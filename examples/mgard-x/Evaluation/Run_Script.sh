MgardSerialExec="../../../build/bin/MgardSerialExec"
MgardXExec="../../../build/bin/mgard-x"

DATA_DIR=$HOME/dev/data
NYX_DATA=$DATA_DIR/512x512x512/velocity_x.dat
XGC_DATA=$DATA_DIR/d3d_coarse_v2_700.bin
E3SM_DATA=$DATA_DIR/temperature.dat


# $MgardSerialExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5e-3 -s inf -v # actual error 1e-4
# $MgardSerialExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5e-2 -s inf -v # actual error 1e-3
# $MgardSerialExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5.5e-1 -s inf -v # actual error 1e-2
# $MgardSerialExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5e0 -s inf -v # actual error 1e-1

# $MgardXExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5e-3 -s inf -l 0 -v -d $1
# $MgardXExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5e-2 -s inf -l 0 -v -d $1
# $MgardXExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5.5e-1 -s inf -l 0 -v -d $1
# $MgardXExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5e0 -s inf -l 0 -v -d $1

# $MgardXExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5e-3 -s inf -l 1 -v -d $1
# $MgardXExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5e-2 -s inf -l 1 -v -d $1
# $MgardXExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5.5e-1 -s inf -l 1 -v -d $1
# $MgardXExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5e0 -s inf -l 1 -v -d $1

# $MgardXExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5e-3 -s inf -l 2 -v -d $1
# $MgardXExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5e-2 -s inf -l 2 -v -d $1
# $MgardXExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5.5e-1 -s inf -l 2 -v -d $1
# $MgardXExec -z -i $NYX_DATA -c out.mgard -t s -n 3 512 512 512 -m rel -e 5e0 -s inf -l 2 -v -d $1



e1=4e-3
e2=4e-2
e3=4e-1
e4=5e0
# $MgardSerialExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e1 -s inf -v # actual error 1e-4
# $MgardSerialExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e2 -s inf -v # actual error 1e-3
# $MgardSerialExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e3 -s inf -v # actual error 1e-2
# $MgardSerialExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e4 -s inf -v # actual error 1e-1


e1=1.7e-3
e2=1.8e-2
e3=1.7e-1
e4=4e0
# $MgardXExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e1 -s inf -l 0 -v -d $1
# $MgardXExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e2 -s inf -l 0 -v -d $1
# $MgardXExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e3 -s inf -l 0 -v -d $1
# $MgardXExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e4 -s inf -l 0 -v -d $1

# $MgardXExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e1 -s inf -l 1 -v -d $1
# $MgardXExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e2 -s inf -l 1 -v -d $1
# $MgardXExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e3 -s inf -l 1 -v -d $1
# $MgardXExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e4 -s inf -l 1 -v -d $1

# $MgardXExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e1 -s inf -l 2 -v -d $1
# $MgardXExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e2 -s inf -l 2 -v -d $1
# $MgardXExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e3 -s inf -l 2 -v -d $1
# $MgardXExec -z -i $XGC_DATA -c out.mgard -t d -n 3 312 16395 39 -m rel -e $e4 -s inf -l 2 -v -d $1



e1=4.5e-3
e2=4e-2
e3=4e-1
e4=5e0
# $MgardSerialExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e1 -s inf -v # actual error 1e-4
# $MgardSerialExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e2 -s inf -v # actual error 1e-3
# $MgardSerialExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e3 -s inf -v # actual error 1e-2
# $MgardSerialExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e4 -s inf -v # actual error 1e-1


e1=1e-3
e2=1e-2
e3=1.041e-1
e3=2.041e-1
e4=1.05e0
# $MgardXExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e1 -s inf -l 0 -v -d $1
# $MgardXExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e2 -s inf -l 0 -v -d $1
# $MgardXExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e3 -s inf -l 0 -v -d $1
# $MgardXExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e4 -s inf -l 0 -v -d $1

# $MgardXExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e1 -s inf -l 1 -v -d $1
# $MgardXExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e2 -s inf -l 1 -v -d $1
# $MgardXExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e3 -s inf -l 1 -v -d $1
# $MgardXExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e4 -s inf -l 1 -v -d $1

# $MgardXExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e1 -s inf -l 2 -v -d $1
# $MgardXExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e2 -s inf -l 2 -v -d $1
# $MgardXExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e3 -s inf -l 2 -v -d $1
# $MgardXExec -z -i $E3SM_DATA -c out.mgard -t s -n 3 72 1444 359 -m rel -e $e4 -s inf -l 2 -v -d $1