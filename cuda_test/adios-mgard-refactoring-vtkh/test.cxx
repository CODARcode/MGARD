#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkm/filter/CellMeasures.h>


#include <adios2.h>

#include <vtkm/Types.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/DataSetFieldAdd.h>

#include <iostream>
#include <stdlib.h>
#include <iomanip>
#include <mpi.h>
#include <string>

#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <bits/stdc++.h> 

#include <mgard_api.h>
#include <mgard_api_cuda.h>

#include <chrono>

void gen_data(double * data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = rand() % 10 + 1;
  }
}

void refactorize(double * data, int nrow, int ncol, int nfib, 
                 double * refactorized_data, std::string csv_prefix,
                 int device) {
  mgard_cuda_handle * handle = new mgard_cuda_handle(8, csv_prefix);
  int out_size;
  double * mgard_refac_buff;
  if (device == 0) { // CPU
    mgard_refac_buff = (double *)mgard_compress(1, data, out_size, nrow, ncol, nfib, 0.1, csv_prefix);
  } else {
    mgard_refac_buff = (double *)mgard_compress_cuda(1, data, out_size, 
                                                    nrow, ncol, nfib, 
                                                    0.1, 3, 16, true, *handle);
  }
  for (int i = 0; i < nrow * ncol * nfib; i++) {
    refactorized_data[i] = mgard_refac_buff[i];
  }
  delete [] mgard_refac_buff;
}

bool comp(std::pair<double, int>  i1, std::pair<double, int>  i2) 
{ 
    return abs(i1.first) > abs(i2.first); 
} 


void sortArr(double * refactorized_data, int data_size, int * index) {
  std::vector<std::pair<double, int> > vp;
  for (int i = 0; i < data_size; ++i) { 
    vp.push_back(std::make_pair(refactorized_data[i], i)); 
  } 
  sort(vp.begin(), vp.end(), comp); 
  for (int i = 0; i < data_size; ++i) { 
    refactorized_data[i] = vp[i].first;
    index[i] = vp[i].second;
  }
}

void reconstruct(double * data, int nrow, int ncol, int nfib, 
                 double * recomposed_data, std::string csv_prefix,
                 int device) {
  mgard_cuda_handle * handle = new mgard_cuda_handle(8, csv_prefix);
  double dummy = 0;
  int out_size = nrow * ncol * nfib * sizeof(double);
  double * tmp_data;
  if (device == 0) {
    tmp_data = mgard_decompress(1, dummy, (unsigned char*)recomposed_data, out_size,  nrow,  ncol, nfib, csv_prefix);
  } else {
    tmp_data = mgard_decompress_cuda(1, dummy, (unsigned char*)recomposed_data, out_size,  
                                 nrow,  ncol, nfib, 3, 16, true, *handle);
  }
  for (int i = 0; i < nrow * ncol * nfib; i++) {
    data[i] = tmp_data[i];
  }
  delete [] tmp_data;
}

void decompose(double * refactorized_data, 
               int data_size,
               int num_of_class, 
               double ** decomposed_data,
               int ** index,
               size_t * counters) {
  int * tmp_index = new int[data_size];
  sortArr(refactorized_data, data_size, tmp_index);
  for (int i = 0; i < num_of_class; i++) {
    int s = floor((float)(i * data_size)/num_of_class);
    int e = floor((float)((i+1) * data_size)/num_of_class) - 1;
    counters[i] = 0;
    for (int j = s; j <= e; j++) {
      decomposed_data[i][counters[i]] = refactorized_data[j];
      index[i][counters[i]] = tmp_index[j];
      counters[i]++;
    }
  }
  delete [] tmp_index;
}

void recompose(double * recomposed_data, 
               int data_size,
               int num_of_class, 
               double * decomposed_data,
               int *index,
               size_t counter) {
  for (int j = 0; j < counter; j++) {
    recomposed_data[index[j]] = decomposed_data[j];
  }
}

void adios_write(int class_idx, double * decomposed_data, 
                 int * index, size_t counter, 
                 std::string var_name, std::string idx_name,
                 std::string filename, std::ofstream &timing_results) {

  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, nproc;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);

  adios2::ADIOS adios(comm, adios2::DebugON);
  adios2::IO outIO = adios.DeclareIO("SimulationOutput");
  outIO.SetEngine("BP4");
  adios2::Engine writer = outIO.Open(filename, adios2::Mode::Write);
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::duration<double> elapsed;


  adios2::Variable<double> outVarV = outIO.DefineVariable<double>(var_name, {1, 1, counter * nproc},
                                                          {0, 0, counter * rank},
                                                          {1, 1, counter});
  adios2::Variable<int> outIdx = outIO.DefineVariable<int>(idx_name, {1, 1, counter * nproc},
                                                      {0, 0, counter * rank},
                                                      {1, 1, counter});
  start = std::chrono::high_resolution_clock::now();
  writer.BeginStep();
  writer.Put<double>(outVarV, decomposed_data);
  writer.Put<int>(outIdx, index);
  writer.EndStep();
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  timing_results << std::to_string(elapsed.count()) << std::endl;
}


void adios_read(int class_idx, double * decomposed_data, 
                 int * index, size_t counter, 
                 std::string var_name, std::string idx_name,
                 std::string filename, std::ofstream &timing_results) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, nproc;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);

  adios2::ADIOS adios(comm, adios2::DebugON);
  adios2::IO inIO = adios.DeclareIO("SimulationOutput");
  inIO.SetEngine("BP4");
  adios2::Engine reader = inIO.Open(filename, adios2::Mode::Read);

  adios2::Box<adios2::Dims> sel({0, 0, counter * rank}, 
                                {1, 1, counter});

  adios2::Variable<double> inVarV = inIO.InquireVariable<double>(var_name);
  adios2::Variable<int> inIdx = inIO.InquireVariable<int>(idx_name);
  inVarV.SetSelection(sel);
  inIdx.SetSelection(sel);

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::duration<double> elapsed;

  std::vector<double> tmp_decomposed_data;
  std::vector<int> tmp_index;

  start = std::chrono::high_resolution_clock::now();
  reader.BeginStep();
  reader.Get(inVarV, tmp_decomposed_data);
  reader.Get(inIdx, tmp_index);
  reader.EndStep();

  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  timing_results << std::to_string(elapsed.count()) << std::endl;

  std::copy(tmp_decomposed_data.begin(), tmp_decomposed_data.end(), decomposed_data);
  std::copy(tmp_index.begin(), tmp_index.end(), index);
}


void vis(int rank,
         int org_z, int org_y, int org_x,
         int dim_z, int dim_y, int dim_x,
         int shape_z, int shape_y, int shape_x,
         double * data, std::string fidldname,
         double iso_value, double * surface_result) {

  vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));
  //vtkh::ForceOpenMP();
  vtkm::Id3 dims(dim_z, dim_y, dim_x);
  vtkm::Id3 org(org_z, org_y, org_x);
  vtkm::Id3 spc(1, 1, 1);
  vtkm::cont::DataSetBuilderUniform dataSetBuilder;
  vtkm::cont::DataSet dataSet = dataSetBuilder.Create(dims, org, spc);
  vtkm::cont::DataSetFieldAdd dsf;
  std::vector<double> vec_data(data, data+(dim_z*dim_y*dim_x));
  dsf.AddPointField(dataSet, fidldname, vec_data);
  vtkh::DataSet data_set;
  data_set.AddDomain(dataSet, rank);

  vtkh::MarchingCubes marcher;
  marcher.SetInput(&data_set);
  marcher.SetField(fidldname);
  const int num_vals = 1;
  double iso_vals [num_vals];
  iso_vals[0] = iso_value;
  marcher.SetIsoValues(iso_vals, num_vals);
  marcher.AddMapField(fidldname);
  marcher.Update();
  vtkh::DataSet *iso_output = marcher.GetOutput();

  vtkm::filter::CellMeasures<vtkm::Area> vols;

  vtkm::cont::DataSet iso_dataset = iso_output->GetDomainById(rank);
  vtkm::cont::DataSet outputData = vols.Execute(iso_dataset);
  vols.SetCellMeasureName("measure");
  auto temp = outputData.GetField(vols.GetCellMeasureName()).GetData();
  vtkm::cont::ArrayHandle<vtkm::FloatDefault> resultArrayHandle;
  temp.CopyTo(resultArrayHandle);

  //std::cout << "GetNumberOfValues: " <<resultArrayHandle.GetNumberOfValues() << std::endl;
  for (int i = 0; i < resultArrayHandle.GetNumberOfValues(); i++) {
    //std::cout << "Area: " <<resultArrayHandle.GetPortalConstControl().Get(vtkm::Id(i)) << std::endl;
    *surface_result += resultArrayHandle.GetPortalConstControl().Get(vtkm::Id(i));
  }
  std::cout << "surface: " <<  *surface_result << "\n";
  /*
  vtkm::Bounds bounds(vtkm::Range(0, shape_z-1), vtkm::Range(0, shape_y-1), vtkm::Range(0, shape_x-1));
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkh::Render render = vtkh::MakeRender(1024,  1024, camera, *iso_output, "img_test", bg_color);

  vtkh::Scene scene;
  scene.AddRender(render);

  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField(fidldname); 

  vtkm::cont::ColorTable color_map("Rainbow Uniform");
  tracer.SetColorTable(color_map);
  tracer.SetRange(vtkm::Range(0, 0.5));
  scene.AddRenderer(&tracer);  
  scene.Render();
  */
}

template <typename T>
void print(T * data, int n) {
  for (int i = 0; i < n; i++) {
    std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
}



int main(int argc, char *argv[]) {

  MPI_Init(NULL, NULL);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, nproc;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);
  MPI_Comm cart_comm;
  int dims[3] = {};
  const int periods[3] = {1, 1, 1};
  int coords[3] = {};

  MPI_Dims_create(nproc, 3, dims);
  size_t npx = dims[0];
  size_t npy = dims[1];
  size_t npz = dims[2];

  MPI_Cart_create(comm, 3, dims, periods, 0, &cart_comm);
  MPI_Cart_coords(cart_comm, rank, 3, coords);
  size_t px = coords[0];
  size_t py = coords[1];
  size_t pz = coords[2];


  int n = atoi(argv[1]);
  int num_of_class = atoi(argv[2]);
  std::string root_csv_prefix = argv[3];
  int device = atoi(argv[4]);
  

  std::string cmd_rmdir = "rm -rf " + root_csv_prefix + std::to_string(rank);
  std::string cmd_mkdir = "mkdir -p " + root_csv_prefix + std::to_string(rank);
  std::system(cmd_rmdir.c_str());
  std::system(cmd_mkdir.c_str());

  std::string csv_prefix = root_csv_prefix + std::to_string(rank) + "/";

  
  int data_size = n*n*n;
  double * data = new double[data_size];
  double * data2 = new double[data_size];
  double * refactorized_data = new double[data_size];
  double * recomposed_data = new double[data_size];
  double ** decomposed_data = new double *[num_of_class];
  int ** index = new int *[num_of_class];
  double ** decomposed_data2 = new double *[num_of_class];
  int ** index2 = new int *[num_of_class];
  size_t * counters = new size_t[num_of_class];

  std::string * filenames = new std::string[num_of_class];
  std::string * var_names = new std::string[num_of_class];
  std::string * idx_names = new std::string[num_of_class];
  for (int i = 0; i < num_of_class; i++) {
    decomposed_data[i] = new double[data_size];
    index[i] = new int[data_size];
    decomposed_data2[i] = new double[data_size];
    index2[i] = new int[data_size];
    var_names[i] = "V";// + std::to_string(i);
    idx_names[i] = "IDX";// + std::to_string(i);
    filenames[i] = root_csv_prefix + "/coeff_" + std::to_string(i);
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::duration<double> elapsed;

  std::ofstream timing_results;
  timing_results.open (csv_prefix + "workflow.csv");

  if (rank == 0) std::cout << "Generating data\n";
  gen_data(data, data_size);


  if (rank == 0) std::cout << "Refactoring\n";
  refactorize(data, n, n, n, refactorized_data, csv_prefix, device);
  //print(refactorized_data, data_size);

  if (rank == 0) std::cout << "Decompose\n";
  decompose(refactorized_data, data_size, num_of_class, 
            decomposed_data, index, counters);
  // for (int i = 0; i < num_of_class; i++) {
  //   std::cout << "i = " << i << std::endl;
  //   print(decomposed_data[i], counters[i]);
  //   print(index[i], counters[i]);
  // }

  for (int i = 0; i < num_of_class; i++) {
    if (rank == 0) std::cout << "ADIOS write\n";
    adios_write(i, decomposed_data[i], index[i], counters[i], 
              var_names[i], idx_names[i], filenames[i], timing_results);
  }

  for (int i = 0; i < num_of_class; i++) {
    if (rank == 0) std::cout << "ADIOS read\n";
    adios_read(i, decomposed_data2[i], index2[i], counters[i], 
                var_names[i], idx_names[i], filenames[i], timing_results);

    if (rank == 0) std::cout << "Recompose\n";
    recompose(recomposed_data, data_size, i, decomposed_data2[i],
              index2[i], counters[i]);

    if (rank == 0) std::cout << "Reconstruct\n";
    reconstruct(data2, n, n, n, 
                recomposed_data, csv_prefix, device);
    if (device == 0) {
      std::string old_file = csv_prefix + "recompose_3D.csv";
      std::string new_file = csv_prefix + "recompose_3D" + std::to_string(i) +".csv";
      std::string cmd_mv = "mv " + old_file + " " + new_file;
      std::system(cmd_mv.c_str());
    } else {
      std::string old_file = csv_prefix + "recompose_3D_cuda_cpt_l2_sm.csv";
      std::string new_file = csv_prefix + "recompose_3D_cuda_cpt_l2_sm_" + std::to_string(i) +".csv";
      std::string cmd_mv = "mv " + old_file + " " + new_file;
      std::system(cmd_mv.c_str());
    }

    // if (rank == 0) std::cout << "Vis\n";
    // start = std::chrono::high_resolution_clock::now();
    // double surface_result = 0.0;
    // vis(rank,
    //     pz * n, py * n, px * n,
    //     n, n, n,
    //     n * npz, n * npy, n * npx,
    //     data2, "field_v",
    //     5, &surface_result);
    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // timing_results << std::to_string(elapsed.count()) << std::endl;
    // timing_results << std::to_string(surface_result) << std::endl;

  }

  timing_results.close();

  MPI_Finalize();
  return 0;
}
