#!/bin/bash
[ ! -d "gs_data" ] && mkdir -p gs_data

NPROC=16

launch_simulation() {
	cp ./simulation/settings-files.json ./settings-files.json
 	sed -i 's/LLL/'"$1/"'g' ./settings-files.json
 	sed -i 's/MMM/'"$2/"'g' ./settings-files.json
 	sed -i 's/KKK/'"$3/"'g' ./settings-files.json
	mpirun -np $NPROC ./build/gray-scott ./settings-files.json
}


# data for single GPU or single CPU runs
# launch_simulation 33 33 33
# launch_simulation 65 65 65
# launch_simulation 129 129 129
# launch_simulation 257 257 257

launch_simulation 128 128 128
#launch_simulation 129 129 129
#launch_simulation 17 17 17

#########Large runs############
# launch_simulation 513 513 513
#launch_simulation 1025 1025 17
#launch_simulation 2049 2049 17
#launch_simulation 4097 4097 17
#launch_simulation 8193 8193 17

# data for single node (multi-GPU/multicore CPU) or multi-node runs
#launch_simulation 1026 1026 1026
#launch_simulation 1539 1026 57351
#launch_simulation 16386 32772 17
#launch_simulation 49158 57351 17
