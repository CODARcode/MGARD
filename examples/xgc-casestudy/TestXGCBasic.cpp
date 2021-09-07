#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

#include "adios2.h"
#include "mgard/compress.hpp"

#define SECONDS(d) std::chrono::duration_cast<std::chrono::seconds>(d).count()

template<typename Type>
void FileWriter_ad(const char *filename, Type *data, std::vector<size_t> global_dim, 
    std::vector<size_t>local_dim, size_t para_dim)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
    adios2::Variable<Type> bp_fdata = bpIO.DefineVariable<Type>(
          "i_f_4d", global_dim, {0, para_dim*rank, 0, 0}, local_dim,  adios2::ConstantDims);
    // Engine derived class, spawned to start IO operations //
    adios2::Engine bpFileWriter = bpIO.Open(filename, adios2::Mode::Write);
    bpFileWriter.Put<Type>(bp_fdata, data);
    bpFileWriter.Close();
}

template<typename Type>
void CheckReconstruction(Type *ori_buff, Type *rct_buff, double tol, 
        size_t local_sz)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double gb_L_inf, error_L_inf_norm = 0;
    double gb_L2, error_L2 = 0;
    for (size_t it=0; it<local_sz; it++) {
        double temp = fabs(ori_buff[it] - rct_buff[it]);
        if (temp > error_L_inf_norm)
            error_L_inf_norm = temp;
        error_L2 += temp * temp;
    }
    error_L2 = sqrt(error_L2/local_sz);
    MPI_Allreduce(&error_L2, &gb_L2, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&error_L_inf_norm, &gb_L_inf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank==0) {
        if (gb_L2 < tol) {
            printf("SUCCESS: Error tolerance met!...requested %f, measured L2=%f, L-inf=%f\n", tol, gb_L2, gb_L_inf);
        } else {
            printf("FAILURE: Error tolerance NOT met!...requested %f, measured L2= %f, L-inf=%f\n", tol, gb_L2, gb_L_inf);
        }
    }
}

// MPI parallelize the second dimension -- # of mesh nodes 
// argv[1]: data path
// argv[2]: filename
// argv[3]: compression dimension 
// argv[4]: eb
// agrv[5]: snorm
// XGC data: n_phi x n_nodes x vx x vy
int main(int argc, char **argv) {
    if (argc != 6) {
        printf("Inputs: \n");
        printf("-- data files directory\n");
        printf("-- data file prefix (suffix is the timestep, 0, 1, 2, ...)\n");
		printf("-- Dimensions used for compression\n");
        printf("-- eb \n");
        printf("-- snorm (default is 0)\n");
        return -1;
    }
    MPI_Init(&argc, &argv);
    int rank, np_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

	int nargv = 0;
    char datapath[2048], filename[2048], readin_f[2048], write_f[2048];
    strcpy(datapath, argv[++nargv]);
    strcpy(filename, argv[++nargv]);
	sprintf(readin_f, "%s%s", datapath, filename);
	sprintf(write_f, "%s%s", filename, ".mgard");
	int D      = atoi(argv[++nargv]);
	double tol = atof(argv[++nargv]);
    double s_norm = atof(argv[++nargv]);
    
    unsigned char *compressed_data = 0;
    std::vector<std::size_t> MGARD_shape(D);

    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io = ad.DeclareIO("XGC");

    adios2::Engine reader = reader_io.Open(readin_f, adios2::Mode::Read);
    // Inquire variable
    adios2::Variable<double> var_i_f_in;
    var_i_f_in = reader_io.InquireVariable<double>("i_f");
    std::vector<std::size_t> shape = var_i_f_in.Shape();
    if (rank == 0) {
        printf("Read in: %s\n", readin_f);
        printf(" shape: %dD ( ", D);
        for (int d = 0; d < D; d ++) {
            printf("%lu ", shape[d]);
        }
        printf(")\n");
        printf("Absolute error tolerance = %f\n", tol);
        printf("Snorm: %f\n", s_norm);
		printf("XGC data shape: {%ld, %ld, %ld, %ld}\n", shape[0], shape[1], shape[2], shape[3]);
		switch (D) {
			case 2: 
				printf("compression shape: {%ld, %ld}\n", shape[0]*shape[1], shape[2]*shape[3]);
				break;
			case 3:
				printf("compression shape: {%ld, %ld, %ld}\n", shape[0]*shape[1], shape[2], shape[3]);
				break;
			case 4: 
				printf("compression shape: {%ld, %ld, %ld, %ld}\n",  shape[0], shape[1], shape[2], shape[3]);
				break;
			default:
				printf("Only support the test of 2D, 3D, 4D...\n");
				break;
		}
    }
    MPI_Barrier(MPI_COMM_WORLD);
	if ((D<2) || (D>4)) {
		reader.Close();
	    MPI_Finalize();
		return -1;
	}

    size_t temp_dim  = (size_t)ceil((float)shape[1]/np_size);
    size_t local_dim = ((rank==np_size-1) ? (shape[1]-temp_dim*rank) : temp_dim);
    size_t local_sz  = local_dim * shape[0]*shape[2]*shape[3]; 

    size_t start_pos = temp_dim*rank;
    size_t read_node = local_dim;
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, start_pos, 0, 0}, {shape[0], read_node, shape[2], shape[3]}));
    std::vector<double> i_f;
    reader.Get<double>(var_i_f_in, i_f);
    reader.Close();
    if (rank == 0) {
        printf("begin compression...\n");
	}
	
	size_t gb_compressed, compressed_sz;
	double *mgard_out_buff;
	if (D == 2) {
		const mgard::TensorMeshHierarchy<2, double> hierarchy({shape[0]*local_dim, shape[2]*shape[3]});
        const mgard::CompressedDataset<2, double> compressed = mgard::compress(hierarchy, i_f.data(), s_norm, tol);
        compressed_sz = compressed.size();
        if (rank==0) printf("begin decompression...\n");
        const mgard::DecompressedDataset<2, double> decompressed = mgard::decompress(compressed);
        mgard_out_buff = (double *)decompressed.data();
        CheckReconstruction(i_f.data(), mgard_out_buff, tol, local_sz);
        FileWriter_ad(write_f, mgard_out_buff, {shape[0], shape[1], shape[2], shape[3]}, {shape[0], local_dim, shape[2], shape[3]}, temp_dim);
        
	} else if (D == 3) {
		const mgard::TensorMeshHierarchy<3, double> hierarchy({shape[0]*local_dim, shape[2], shape[3]});
		const mgard::CompressedDataset<3, double> compressed = mgard::compress(hierarchy, i_f.data(), s_norm, tol);
        compressed_sz = compressed.size();
        if (rank==0) printf("begin decompression...\n");
        const mgard::DecompressedDataset<3, double> decompressed = mgard::decompress(compressed);
        mgard_out_buff = (double *)decompressed.data();
  	    CheckReconstruction(i_f.data(), mgard_out_buff, tol, local_sz);
        FileWriter_ad(write_f, mgard_out_buff, {shape[0], shape[1], shape[2], shape[3]}, {shape[0], local_dim, shape[2], shape[3]}, temp_dim);	
	} else if (D == 4) {
		const mgard::TensorMeshHierarchy<4, double> hierarchy({shape[0], local_dim, shape[2], shape[3]});
	    const mgard::CompressedDataset<4, double> compressed = mgard::compress(hierarchy, i_f.data(), s_norm, tol);
        compressed_sz = compressed.size();
        if (rank==0) printf("begin decompression...\n");
        const mgard::DecompressedDataset<4, double> decompressed = mgard::decompress(compressed);
        mgard_out_buff = (double *)decompressed.data();
	    CheckReconstruction(i_f.data(), mgard_out_buff, tol, local_sz);
        if (rank ==0) printf("%s\n", write_f);
        FileWriter_ad(write_f, mgard_out_buff, {shape[0], shape[1], shape[2], shape[3]}, {shape[0], local_dim, shape[2], shape[3]}, temp_dim);	
	}
	MPI_Allreduce(&compressed_sz, &gb_compressed, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
        printf("finish...\n");
		printf("Compression ratio = %.3f\n", ((double)8.0*shape[0]*shape[1]*shape[2]*shape[3]) / gb_compressed);
	}
    MPI_Finalize();
	return 0;
}
