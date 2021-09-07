#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

#include "adios2.h"
#include "mgard/compress.hpp"

// only compress the plane 0 

#define SECONDS(d) std::chrono::duration_cast<std::chrono::seconds>(d).count()

template <typename Type>
void FileWriter_bin(const char *filename, Type *data, size_t size)
{
  std::ofstream fout(filename, std::ios::binary);
  fout.write((const char*)(data), size*sizeof(Type));
  fout.close();
}

template<typename Type>
void FileWriter_ad(const char *filename, Type *data, std::vector<size_t> global_dim)
{
    adios2::ADIOS ad;
    adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
    adios2::Variable<Type> bp_fdata = bpIO.DefineVariable<Type>(
          "i_f_4d", global_dim, {0, 0, 0, 0}, global_dim,  adios2::ConstantDims);
    // Engine derived class, spawned to start IO operations //
    printf("write...%s\n", filename);
    adios2::Engine bpFileWriter = bpIO.Open(filename, adios2::Mode::Write);
    bpFileWriter.Put<Type>(bp_fdata, data);
    bpFileWriter.Close();
}

// argv[1]: data path
// argv[2]: filename
// argv[3]: eb
// agrv[4]: snorm
// XGC data: n_phi x n_nodes x vx x vy
// Require xgc.f0.mesh.bp file located in the same folder as the data file 
int main(int argc, char **argv) {

	int nargv = 0;
    char datapath[2048], filename[2048], readin_f[2048], write_f[2048];
    strcpy(datapath, argv[++nargv]);
    strcpy(filename, argv[++nargv]);
    sprintf(readin_f, "%s%s", datapath, filename);
    sprintf(write_f, "%s%s", filename, ".mgard.non");
    double tol = atof(argv[++nargv]);
    double s_norm = atof(argv[++nargv]);

	unsigned char *compressed_data = 0;

    adios2::ADIOS ad;
    adios2::IO reader_io = ad.DeclareIO("XGC");
    adios2::Engine reader = reader_io.Open(readin_f, adios2::Mode::Read);
    // Inquire variable
    adios2::Variable<double> var_i_f_in;
    var_i_f_in = reader_io.InquireVariable<double>("i_f");
    std::vector<std::size_t> shape = var_i_f_in.Shape();
	var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, 0, 0, 0}, {shape[0], shape[1], shape[2], shape[3]}));
    std::vector<double> i_f;
    reader.Get<double>(var_i_f_in, i_f);
    reader.Close();
    size_t num_elements = shape[0]*shape[1]*shape[2]*shape[3];

    printf("Read in: %s\n", readin_f);
    printf(" XGC data shape: (%ld, %ld, %ld, %ld)\n ", shape[0], shape[1], shape[2], shape[3]);
    printf("Absolute error tolerance = %f\n", tol);
	printf("Snorm: %f\n", s_norm);
	printf("This program requires a xgc.f0.mesh.bp file for the non-uniform coordinates\n");

    adios2::IO read_vol_io = ad.DeclareIO("xgc_vol");
    char vol_file[2048];
    sprintf(vol_file, "%sxgc.f0.mesh.bp", datapath);
    adios2::Engine reader_vol = read_vol_io.Open(vol_file, adios2::Mode::Read);
    var_i_f_in = read_vol_io.InquireVariable<double>("f0_grid_vol_vonly");
    size_t nnodes = var_i_f_in.Shape()[1];
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, 0}, {1, nnodes}));
    std::vector<double> grid_vol;
    reader_vol.Get<double>(var_i_f_in, grid_vol);
    reader_vol.Close();

    double FSL = std::accumulate(grid_vol.begin(), grid_vol.end(), 0.0);
    std::cout << FSL << "\n";
    double div = 1.0/FSL;
    std::vector<double>coords_z(nnodes, div); 
    coords_z.at(0) = grid_vol.at(0) * 0.5 * div;
    for (size_t i=1; i<nnodes; i++) {
        coords_z.at(i) = coords_z.at(i-1) + (grid_vol.at(i) + grid_vol.at(i-1)) * 0.5 * div;
//        if (i < 100) std::cout << coords_z.at(i) << "\n";
    }
    size_t vx=shape[2], vy=shape[3], nphi=shape[0];
    double base_x = (1.0-1.0/vx)/(vx-1), base_y=(1.0-1.0/vy)/(vy-1), base_p=1.0/nphi;
    std::vector<double>coords_x(vx, 0.0), coords_y(vy, 0.0), coords_p(nphi, 0.0);
    coords_x.at(0) = 0.5/vx;
    coords_y.at(0) = 0.5/vy;
    for (size_t idx=1; idx<vx; idx++) coords_x.at(idx) = base_x * idx + coords_x.at(0);
    for (size_t idx=1; idx<vy; idx++) coords_y.at(idx) = base_y * idx + coords_y.at(0);
    for (size_t idx=0; idx<nphi; idx++) coords_p.at(idx) = idx * base_p;
//    std::cout << "max x: " << coords_x.at(vx-1) << ", max y: " << coords_y.at(vy-1) << ", max z: " << coords_z.at(nnodes-1) << ", max p: " << coords_p.at(nphi-1) << "\n";

    std::cout << "begin compression...\n";
    const std::array<std::vector<double>, 4> coords = {coords_p, coords_z, coords_y, coords_x};
    const std::array<std::size_t, 4> dims = {shape[0], shape[1], shape[2], shape[3]};
    const mgard::TensorMeshHierarchy<4, double> hierarchy(dims, coords);
    const size_t ndof = hierarchy.ndof();
    const mgard::CompressedDataset<4, double> compressed = mgard::compress(hierarchy, i_f.data(), s_norm, tol); 
    const mgard::DecompressedDataset<4, double> decompressed = mgard::decompress(compressed);
    size_t compressed_sz = compressed.size();
	printf("Compression ratio: %.2f\n", ((double)8.0*num_elements) / compressed.size());

    FileWriter_ad(write_f, (double *)decompressed.data(), {shape[0], shape[1], shape[2], shape[3]});

    return 0;
}
