

#include <vtkm/cont/Initialize.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/filter/MapFieldPermutation.h>
#include <vtkm/filter/contour/worklet/Contour.h>
#include <vtkm/filter/contour/Contour.h>

#include <vtkm/rendering/Actor.h>
#include <vtkm/rendering/MapperWireframer.h>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperRayTracer.h>
#include <vtkm/rendering/Scene.h>
#include <vtkm/rendering/View3D.h>

#include <vtkm/io/VTKDataSetReader.h>
#include <vtkm/io/VTKDataSetWriter.h>

#include "mgard/mgard-x/DataRefactoring/MultiDimension/DataRefactoring.hpp"
#include "mgard/mgard-x/DataRefactoring/MultiDimension/DataRefactoringAdaptiveResolution.hpp"
#include "mgard/mgard-x/Utilities/ErrorCalculator.h"

#include <iostream>
#include <vector>

// using namespace mgard_x;

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
struct SurfaceDetect {
  T iso_value;

  SurfaceDetect(T iso_value): iso_value(iso_value) {}
  bool operator()(AdaptiveResolutionTreeNode<D, T, DeviceType> * node, 
                  typename AdaptiveResolutionTreeNode<D, T, DeviceType>::T_error error, 
                  SubArray<D, T, DeviceType> v){
    SIZE data_index[D];
    T max_data = std::numeric_limits<T>::min();
    T min_data = std::numeric_limits<T>::max();
    for (int i = 0; i < std::pow(2, D); i++) {
      for (int d = 0; d < D; d++) {
        int linearized_index = i;
        if (linearized_index % 2 == 0) {
          data_index[d] = node->index_start_reordered[d];
        } else {
          data_index[d] = node->index_end_reordered[d];
        }
        linearized_index /= 2;
      }
      T data = 0;
      MemoryManager<DeviceType>::Copy1D(&data, v(data_index), 1, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      max_data = std::max(max_data, data);
      min_data = std::min(min_data, data);
    }

    std::cout << "max_data: " << max_data << " "
              << "min_data: " << min_data << " "
              << "error: " << error << "\n";
    if (max_data + error >= iso_value &&
        min_data - error <= iso_value) {
      return true;
    } else {
      return false;
    }

  }
};

template <DIM D, typename T>
void test(T * data, std::vector<SIZE> shape, T tol, T iso_value) {
  std::cout << "Preparing data...";
  //... load data into in_array_cpu
  Hierarchy<D, T, CUDA> hierarchy(shape);
  Array<D, T, CUDA> in_array(shape);
  in_array.load(data);
  SubArray in_subarray(in_array);

  Array<D, T, CUDA> org_array = in_array;
  std::cout << "Done\n";

  // PrintSubarray("Input data", SubArray(org_array));

  std::cout << "Decomposing with MGARD-X CUDA backend...\n";
  decompose(hierarchy, in_subarray, hierarchy.l_target, 0);

  // PrintSubarray("Decomposed data", in_subarray);

  std::cout << "Done\n";

  std::cout << "Recomposing with MGARD-X CUDA backend...\n";
  bool interpolate_full_resolution = true;
  SurfaceDetect<D, T, CUDA> surface_detector(iso_value);
  Array<D, T, CUDA> out_array = recompose_adaptive_resolution(hierarchy, in_subarray, tol, interpolate_full_resolution, surface_detector, 0); 
  std::cout << "Done\n";

  DeviceRuntime<CUDA>::SyncQueue(0);

  size_t n = 1;
  for (int i = 0; i < shape.size(); i++) n *= shape[i];
  enum error_bound_type mode = error_bound_type::ABS;
  std::cout << "L_inf_error: " << L_inf_error(n, org_array.hostCopy(), out_array.hostCopy(), mode) << "\n";

  


}

}


VTKM_CONT bool DoMapField(vtkm::cont::DataSet& result,
                          const vtkm::cont::Field& field,
                          vtkm::worklet::Contour& worklet)
{
  if (field.IsFieldPoint())
  {
    auto functor = [&](const auto& concrete) {
      auto fieldArray = worklet.ProcessPointField(concrete);
      result.AddPointField(field.GetName(), fieldArray);
    };
    field.GetData()
      .CastAndCallForTypesWithFloatFallback<vtkm::TypeListField, VTKM_DEFAULT_STORAGE_LIST>(
        functor);
    return true;
  }
  else if (field.IsFieldCell())
  {
    // Use the precompiled field permutation function.
    vtkm::cont::ArrayHandle<vtkm::Id> permutation = worklet.GetCellIdMap();
    return vtkm::filter::MapFieldPermutation(field, permutation, result);
  }
  else if (field.IsFieldGlobal())
  {
    result.AddField(field);
    return true;
  }
  else
  {
    return false;
  }
}

template <mgard_x::DIM D, typename T>
void test_vtkm(int argc, char *argv[], T * data, std::vector<mgard_x::SIZE> shape, T tol, T iso_value) {
  vtkm::cont::Initialize(argc, argv);
  vtkm::cont::ScopedRuntimeDeviceTracker(vtkm::cont::DeviceAdapterTagCuda{});
  vtkm::cont::DataSet dataSet;
  vtkm::cont::DataSetBuilderUniform dataSetBuilder;
  vtkm::cont::DataSetFieldAdd dsf;
  if (D == 2) shape.push_back(1);
  vtkm::Id3 dims(shape[2], shape[1], shape[0]);
  vtkm::Id3 org(0,0,0);
  vtkm::Id3 spc(1,1,1);
  dataSet = dataSetBuilder.Create(dims, org, spc);
  std::vector<T> data_vec(shape[0]*shape[1]*shape[2]);
  for (int i = 0; i < shape[0]*shape[1]*shape[2]; i++) {
    data_vec[i] = data[i];
  }
  std::string field_name = "test_field";
  dsf.AddPointField(dataSet, field_name, data_vec);
  vtkm::filter::Contour contour_filter;
  contour_filter.SetGenerateNormals(true);
  contour_filter.SetMergeDuplicatePoints(false);
  contour_filter.SetIsoValue(0, iso_value);
  // contour_filter.SetIsoValue(1, iso_value+10);
  contour_filter.SetActiveField(field_name);
  contour_filter.SetFieldsToPass({ field_name });
  vtkm::cont::DataSet outputData = contour_filter.Execute(dataSet);

  

  vtkm::cont::DataSet outputData2;
  vtkm::cont::Field field = dataSet.GetField(0);
  // outputData2.AddField(field);
  std::vector<T> iso_data_vec(shape[0]*shape[1]*shape[2], iso_value);
  outputData2.AddPointField(field_name, iso_data_vec);

  vtkm::cont::CellSetSingleType<> cellset = outputData.GetCellSet().AsCellSet<vtkm::cont::CellSetSingleType<>>();

  vtkm::cont::CellSetSingleType<> cellset2;
  vtkm::cont::ArrayHandle<vtkm::Id, VTKM_DEFAULT_CONNECTIVITY_STORAGE_TAG> connectivity;
  connectivity.Allocate(cellset.GetNumberOfCells()*3);
  vtkm::cont::ArrayHandle<vtkm::Id, VTKM_DEFAULT_CONNECTIVITY_STORAGE_TAG>::WritePortalType writePortal = connectivity.WritePortal();
  for (vtkm::Id i = 0; i < cellset.GetNumberOfCells(); i++) {
    vtkm::Vec<vtkm::Id, 3> ids;
    cellset.GetIndices(i, ids);
    writePortal.Set(i*3, ids[0]);
    writePortal.Set(i*3+1, ids[1]);
    writePortal.Set(i*3+2, ids[2]);
  }

  cellset2.Fill(cellset.GetNumberOfPoints(),
                vtkm::CELL_SHAPE_TRIANGLE, 3,
                connectivity);

  outputData2.SetCellSet(cellset2);
  outputData2.AddCoordinateSystem(outputData.GetCoordinateSystem());

  vtkm::io::VTKDataSetWriter writer_output("contour_output.vtk");
  writer_output.WriteDataSet(outputData);

  vtkm::io::VTKDataSetWriter writer_input("input.vtk");
  writer_input.WriteDataSet(dataSet);

  std::cout << "outputData.GetNumberOfCells() = " << outputData.GetNumberOfCells() << "\n";
  std::cout << "outputData.GetNumberOfPoints() = " << outputData.GetNumberOfPoints() << "\n";
 
  // using Mapper = vtkm::rendering::MapperWireframer;
  using Mapper = vtkm::rendering::MapperRayTracer;
  using Canvas = vtkm::rendering::CanvasRayTracer;

  vtkm::rendering::Scene scene;
  vtkm::cont::ColorTable colorTable("inferno");
  vtkm::cont::ColorTable colorTable2("inferno");
  scene.AddActor(vtkm::rendering::Actor(outputData2.GetCellSet(),
                                      outputData2.GetCoordinateSystem(),
                                      outputData2.GetField(field_name),
                                      colorTable));

  // scene.AddActor(vtkm::rendering::Actor(outputData.GetCellSet(),
  //                                     outputData.GetCoordinateSystem(),
  //                                     outputData.GetField(field_name),
  //                                     colorTable));

  Mapper mapper;
  Canvas canvas(1024, 1024);

  vtkm::rendering::Color bg(0.2f, 0.2f, 0.2f, 1.0f);

  const vtkm::cont::CoordinateSystem coords = outputData2.GetCoordinateSystem();
  vtkm::Bounds coordsBounds = coords.GetBounds();
  vtkm::rendering::Camera camera = vtkm::rendering::Camera();
  // camera.SetViewUp(vtkm::make_Vec(0.f, 0.f, 1.f));
  camera.ResetToBounds(coordsBounds);

  vtkm::Vec<vtkm::Float32, 3> totalExtent;
  totalExtent[0] = vtkm::Float32(shape[2]);
  totalExtent[1] = vtkm::Float32(shape[1]);
  totalExtent[2] = vtkm::Float32(shape[0]);
  vtkm::Float32 mag = vtkm::Magnitude(totalExtent);
  vtkm::Normalize(totalExtent);
  camera.SetLookAt(totalExtent * (mag * .5f));
  camera.SetViewUp(vtkm::make_Vec(0.f, 0.f, 1.f));
  // camera.SetClippingRange(1.f, 1000.f);
  camera.SetFieldOfView(60.f);
  camera.SetPosition(totalExtent * (mag * 2.f));
  vtkm::rendering::View3D view(scene, mapper, canvas, camera, bg);
  view.Initialize();
  view.Paint();
  string outputfile = "output";
  view.SaveAs(outputfile + " .pnm"); 
}





bool require_arg(int argc, char *argv[], std::string option) {
  for (int i = 0; i < argc; i++) {
    if (option.compare(std::string(argv[i])) == 0) {
      return true;
    }
  }
  std::cout << "missing option: " + option + "\n";
  return false;
}

std::string get_arg(int argc, char *argv[], std::string option) {
  if (require_arg(argc, argv, option)) {
    for (int i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        return std::string(argv[i + 1]);
      }
    }
  }
  return std::string("");
}

std::vector<mgard_x::SIZE> get_arg_dims(int argc, char *argv[],
                                        std::string option) {
  std::vector<mgard_x::SIZE> shape;
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int arg_idx = 0, i;
    for (i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
        arg_idx = i + 1;
      }
    }
    try {
      int d = std::stoi(arg);
      for (int i = 0; i < d; i++) {
        shape.push_back(std::stoi(argv[arg_idx + 1 + i]));
      }
      return shape;
    } catch (std::invalid_argument const &e) {
      std::cout << "illegal argument for option " + option + "\n";
      return shape;
    }
  }
  return shape;
}

double get_arg_double(int argc, char *argv[], std::string option) {
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int i;
    for (i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
      }
    }
    try {
      double d = std::stod(arg);
      return d;
    } catch (std::invalid_argument const &e) {
      std::cout << "illegal argument for option " + option + "\n";
    }
  }
  return 0;
}

template <typename T> size_t readfile(const char *input_file, T *&in_buff) {
  std::cout << "Loading file: " << input_file << "\n";

  FILE *pFile;
  pFile = fopen(input_file, "rb");
  if (pFile == NULL) {
    std::cout << "file open error!\n";
    exit(1);
  }
  fseek(pFile, 0, SEEK_END);
  size_t lSize = ftell(pFile);
  rewind(pFile);
  in_buff = (T *)malloc(lSize);
  lSize = fread(in_buff, 1, lSize, pFile);
  fclose(pFile);
  // min_max(lSize/sizeof(T), in_buff);
  return lSize;
}

int main(int argc, char *argv[]) {
  std::string input_file = get_arg(argc, argv, "-i");
  std::string dt = get_arg(argc, argv, "-t");
  std::vector<mgard_x::SIZE> shape = get_arg_dims(argc, argv, "-n");
  double tol = get_arg_double(argc, argv, "-e");
  double iso_value = get_arg_double(argc, argv, "-v");

  if (dt.compare("s") == 0) {
    float * data = NULL;
    readfile(input_file.c_str(), data);
    if (shape.size() == 2) {
      test_vtkm<2, float>(argc, argv, data, shape, (float)tol, (float)iso_value);
      // mgard_x::test<2, float>(data, shape, (float)tol, (float)iso_value);
    } else if (shape.size() == 3) {
      test_vtkm<3, float>(argc, argv, data, shape, (float)tol, (float)iso_value);
      // mgard_x::test<3, float>(data, shape, (float)tol, (float)iso_value);
    } else {
      std::cout << "wrong num of dim.\n";
    }
  } else if (dt.compare("d") == 0) {
    double * data = NULL;
    readfile(input_file.c_str(), data);
    if (shape.size() == 2) {
      test_vtkm<2, double>(argc, argv, data, shape, (float)tol, (float)iso_value);
      // mgard_x::test<2, double>(data, shape, (double)tol, (float)iso_value);
    } else if (shape.size() == 3) {
      test_vtkm<3, double>(argc, argv, data, shape, (float)tol, (float)iso_value);
      // mgard_x::test<3, double>(data, shape, (double)tol, (float)iso_value);
    } else {
      std::cout << "wrong num of dim.\n";
    }
  } else {
    std::cout << "wrong data type.\n";
  }
  
  
  // prepare
  // std::cout << "Preparing data...";
  // double *in_array_cpu = new double[n1 * n2 * n3];
  // //... load data into in_array_cpu
  // std::vector<SIZE> shape{n1, n2, n3};
  // Hierarchy<3, double, CUDA> hierarchy(shape);
  // Array<3, double, CUDA> in_array(shape);
  // in_array.load(in_array_cpu);
  // SubArray in_subarray(in_array);

  // std::cout << "Done\n";

  // std::cout << "Decomposing with MGARD-X CUDA backend...";
  // decompose(hierarchy, in_subarray, hierarchy.l_target, 0);
  // Get compressed size in number of bytes.
  // size_t compressed_size = compressed_array.shape()[0];
  // unsigned char *compressed_array_cpu = compressed_array.hostCopy();
  // std::cout << "Done\n";

  // std::cout << "Decompressing with MGARD-X CUDA backend...";
  // // decompression
  // Array<3, double, CUDA> decompressed_array =
  //     decompress(hierarchy, compressed_array,
  //                         error_bound_type::REL, tol, s, norm, config);
  // delete[] in_array_cpu;
  // double *decompressed_array_cpu = decompressed_array.hostCopy();
  // std::cout << "Done\n";
}