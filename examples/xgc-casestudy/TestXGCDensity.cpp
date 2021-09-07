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
    std::cout << "processor " << rank << ": {" << global_dim[0] << ", " << global_dim[1] << ", " << global_dim[2] << ", " << global_dim[3] << "}, {";
    std::cout << local_dim[0] << ", " <<local_dim[1] << ", " << local_dim[2] << ", " << local_dim[3] << "}, " << para_dim*rank << "\n";
    adios2::Variable<Type> bp_fdata = bpIO.DefineVariable<Type>(
          "i_f_2d", global_dim, {0, para_dim*rank, 0, 0}, local_dim,  adios2::ConstantDims);
    // Engine derived class, spawned to start IO operations //
    adios2::Engine bpFileWriter = bpIO.Open(filename, adios2::Mode::Write);
    bpFileWriter.Put<Type>(bp_fdata, data);
    bpFileWriter.Close();
}

int main(int argc, char **argv) {
    if (argc != 6) {
        printf("Inputs: \n");
        printf("-- data files directory\n");
        printf("-- data file prefix (suffix is the timestep, 0, 1, 2, ...)\n");
		printf("-- snorm variable file (in the same directory as data file)\n");
        printf("-- eb \n");
        return -1;
    }
    MPI_Init(&argc, &argv);
    int rank, np_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

    int nargv = 0;
    char datapath[2048], filename[2048], readin_f[2048], snorm_f[2048], write_f[2048];
    strcpy(datapath, argv[++nargv]);
    strcpy(filename, argv[++nargv]);
    sprintf(readin_f, "%s%s", datapath, filename);
	sprintf(snorm_f, "%s%s", datapath, argv[++nargv]);
    sprintf(write_f, "%s%s", filename, ".mgard");
    double tol = atof(argv[++nargv]);

    unsigned char *compressed_data = 0;
    double *i_f_5d;

    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io  = ad.DeclareIO("XGC");
    adios2::IO reader_so = ad.DeclareIO("XGC_snorm");
    adios2::Engine reader = reader_io.Open(readin_f, adios2::Mode::Read);
    // Inquire variable
    adios2::Variable<double> var_i_f_in;
    var_i_f_in = reader_io.InquireVariable<double>("i_f");

    std::vector<std::size_t> shape = var_i_f_in.Shape();
    size_t temp_dim  = (size_t)ceil((float)shape[1]/np_size);
    size_t local_dim = ((rank==np_size-1) ? (shape[1]-temp_dim*rank) : temp_dim);
    size_t temp_sz   = temp_dim  * shape[0]*shape[2]*shape[3];
    size_t local_sz  = local_dim * shape[0]*shape[2]*shape[3]; 
	if (rank == 0) {
        printf("Read in data file: %s\n", readin_f);
        printf("Snorm parameter file: %s\n", snorm_f);
		printf("Requested error tolerance: %f\n", tol);
        printf("global shape: {%ld, %ld, %ld, %ld}\n", shape[0], shape[1], shape[2], shape[3]);
		printf("compression shape: {%ld, %ld}\n", shape[2], shape[3]);
    }
    size_t start_pos = temp_dim*rank;
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, start_pos, 0, 0}, {shape[0], local_dim, shape[2], shape[3]}));
    std::vector<double> i_f;
    reader.Get<double>(var_i_f_in, i_f);
    reader.Close();

    adios2::Engine reader_s = reader_so.Open(snorm_f, adios2::Mode::Read);
    adios2::Variable<double> var_s_in;
    var_s_in = reader_so.InquireVariable<double>("Q_norm");
    var_s_in.SetSelection(adios2::Box<adios2::Dims>({start_pos}, {local_dim}));
    std::vector<double> s_norm;
    reader_s.Get<double>(var_s_in, s_norm);
    reader_s.Close();

    if (rank == 0) {
        printf("begin compression...\n");
    }
    double *dcp_i_f = (double *)malloc(local_sz * sizeof(double));   
    size_t compressed_sz = 0;
    const mgard::TensorMeshHierarchy<2, double> hierarchy({shape[2], shape[3]});
    size_t vnode = shape[2]*shape[3];
    const size_t ndof = hierarchy.ndof();
    for (size_t iphi=0; iphi<shape[0]; iphi++) {
        double *i_f_2d = i_f.data() + iphi*local_dim*vnode;
        double *dcp_i_f_2d = &dcp_i_f[iphi*local_dim*vnode];
        for (size_t inode=0; inode<local_dim; inode++) {
            double s_tol = tol / s_norm.at(inode);
            const mgard::CompressedDataset<2, double> compressed = mgard::compress(hierarchy, i_f_2d + inode*vnode, -1.0, s_tol);
            compressed_sz += compressed.size();
            const mgard::DecompressedDataset<2, double> decompressed = mgard::decompress(compressed);
            memcpy(&dcp_i_f_2d[inode*vnode], decompressed.data(), vnode*sizeof(double)); 
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    FileWriter_ad(write_f, dcp_i_f, {shape[0], shape[1], shape[2], shape[3]}, {shape[0], local_dim, shape[2], shape[3]}, temp_dim);
    size_t gb_compressed;
    MPI_Allreduce(&compressed_sz, &gb_compressed, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

    free(dcp_i_f);
    MPI_Finalize();
	return 0;
}
