#!/bin/bash
[ ! -d "gs_bin_data" ] && mkdir -p gs_bin_data


convert_data_2d(){
	mpirun -np $4 ./build/bp2bin $1 $2 $3 ../gray-scott/gs_data/gs-$1-$2-$3.bp 1 2
	mv *.dat ./gs_bin_data/
}

convert_data_3d(){
	mpirun -np $4 ./build/bp2bin $1 $2 $3 ../gray-scott/gs_data/gs-$1-$2-$3.bp 300 3
	mv *.dat ./gs_bin_data/
}


# convert for single GPU or single CPU runs
# convert_data_2d 33 33 33 1
# convert_data_2d 65 65 65 1
# convert_data_2d 129 129 129 1
# convert_data_2d 257 257 257 1

# convert_data_3d 33 33 33 1
# convert_data_3d 65 65 65 1
# convert_data_3d 129 129 129 1
# convert_data_3d 257 257 257 1

# convert_data_3d 128 128 128 1
# convert_data_3d 129 129 129 1
convert_data_3d 17 17 17 1

######for large data######
# convert_data_3d 513 513 513 1
#convert_data_2d 513 513 513 1
#convert_data_2d 1025 1025 17 1
#convert_data_2d 2049 2049 17 1
# convert_data_2d 4097 4097 17 1
# convert_data_2d 8193 8193 17 1



# data for single node (multi-GPU/multicore CPU) or multi-node runs
# convert_data_3d 1026 1026 1026 8
# convert_data_3d 1539 1026 57351 8
# convert_data_3d 16386 32772 17 8
# convert_data_3d 49158 57351 17 8
