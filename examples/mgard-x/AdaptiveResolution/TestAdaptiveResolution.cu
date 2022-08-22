

#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/cont/Initialize.h>
#include <vtkm/filter/MapFieldPermutation.h>
#include <vtkm/filter/contour/Contour.h>
#include <vtkm/filter/contour/worklet/Contour.h>

#include <vtkm/rendering/Actor.h>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperRayTracer.h>
#include <vtkm/rendering/MapperWireframer.h>
#include <vtkm/rendering/Scene.h>
#include <vtkm/rendering/View3D.h>

#include <vtkm/io/VTKDataSetReader.h>
#include <vtkm/io/VTKDataSetWriter.h>

#include "mgard/mgard-x/DataRefactoring/MultiDimension/Coefficient/DenseToCompressedSparseCell.hpp"
#include "mgard/mgard-x/DataRefactoring/MultiDimension/DataRefactoring.hpp"
#include "mgard/mgard-x/DataRefactoring/MultiDimension/DataRefactoringAdaptiveResolution.hpp"
#include "mgard/mgard-x/Utilities/ErrorCalculator.h"

#include "SparseFlyingCells.hpp"
#include "SparseFlyingEdges.hpp"

#include "FlyingEdgesBatched.hpp"

#include <iostream>
#include <unordered_map>
#include <vector>

#include "mgard/mdr_x.hpp"

// using namespace mgard_x;

void vtkm_render(vtkm::cont::DataSet dataSet, std::vector<mgard_x::SIZE> shape,
                 std::string field_name, std::string output) {
  using Mapper = vtkm::rendering::MapperRayTracer;
  using Canvas = vtkm::rendering::CanvasRayTracer;

  vtkm::rendering::Scene scene;
  vtkm::cont::ColorTable colorTable("inferno");
  scene.AddActor(vtkm::rendering::Actor(
      dataSet.GetCellSet(), dataSet.GetCoordinateSystem(),
      dataSet.GetField(field_name), colorTable));

  Mapper mapper;
  Canvas canvas(1024, 1024);

  vtkm::rendering::Color bg(0.2f, 0.2f, 0.2f, 1.0f);

  const vtkm::cont::CoordinateSystem coords = dataSet.GetCoordinateSystem();
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
  view.SaveAs(output + ".pnm");
}

template <typename T>
vtkm::cont::DataSet
ArrayToDataset(std::vector<mgard_x::SIZE> shape, T iso_value,
               mgard_x::Array<1, mgard_x::SIZE, mgard_x::CUDA> TrianglesArray,
               mgard_x::Array<1, T, mgard_x::CUDA> PointsArray,
               std::string field_name) {
  mgard_x::SIZE *Triangles = new mgard_x::SIZE[TrianglesArray.shape(0)];
  T *Points = new T[PointsArray.shape(0)];

  mgard_x::SIZE numTriangles = TrianglesArray.shape(0) / 3;
  mgard_x::SIZE numPoints = PointsArray.shape(0) / 3;

  memcpy(Triangles, TrianglesArray.hostCopy(),
         numTriangles * 3 * sizeof(mgard_x::SIZE));
  memcpy(Points, PointsArray.hostCopy(), numPoints * 3 * sizeof(T));

  // mgard_x::PrintSubarray("Triangles", mgard_x::SubArray(TrianglesArray));
  // mgard_x::PrintSubarray("Points", mgard_x::SubArray(PointsArray));

  vtkm::cont::DataSet ds_from_mc;
  std::vector<T> iso_data_vec(shape[0] * shape[1] * shape[2], iso_value);
  ds_from_mc.AddPointField(field_name, iso_data_vec);
  vtkm::cont::CellSetSingleType<> cellset;
  vtkm::cont::ArrayHandle<vtkm::Id, VTKM_DEFAULT_CONNECTIVITY_STORAGE_TAG>
      connectivity;
  connectivity.Allocate(TrianglesArray.shape(0));

  // std::cout << "connectivity.GetNumberOfValues() = " <<
  // connectivity.GetNumberOfValues() << "\n";

  vtkm::cont::ArrayHandle<vtkm::Id, VTKM_DEFAULT_CONNECTIVITY_STORAGE_TAG>::
      WritePortalType writePortal = connectivity.WritePortal();
  for (vtkm::Id i = 0; i < numTriangles; i++) {
    writePortal.Set(i * 3, Triangles[i * 3]);
    writePortal.Set(i * 3 + 1, Triangles[i * 3 + 1]);
    writePortal.Set(i * 3 + 2, Triangles[i * 3 + 2]);
  }

  cellset.Fill(numPoints, vtkm::CELL_SHAPE_TRIANGLE, 3, connectivity);
  ds_from_mc.SetCellSet(cellset);

  vtkm::cont::ArrayHandle<vtkm::Vec3f> coordinate_points;
  coordinate_points.Allocate(numPoints);
  for (vtkm::Id pointId = 0; pointId < numPoints; pointId++) {
    vtkm::Vec3f point;
    point[0] = Points[pointId * 3];
    point[1] = Points[pointId * 3 + 1];
    point[2] = Points[pointId * 3 + 2];
    coordinate_points.WritePortal().Set(pointId, point);
  }
  vtkm::cont::CoordinateSystem coordinate_system("cs", coordinate_points);
  ds_from_mc.AddCoordinateSystem(coordinate_system);

  return ds_from_mc;
}

template <mgard_x::DIM D, typename T>
void test_vtkm(int argc, char *argv[], T *data,
               std::vector<mgard_x::SIZE> shape, T tol, T iso_value) {
  vtkm::cont::Initialize(argc, argv);
  vtkm::cont::ScopedRuntimeDeviceTracker(vtkm::cont::DeviceAdapterTagCuda{});
  vtkm::cont::DataSet dataSet;
  vtkm::cont::DataSetBuilderUniform dataSetBuilder;
  if (D == 2)
    shape.push_back(1);
  vtkm::Id3 dims(shape[2], shape[1], shape[0]);
  vtkm::Id3 org(0, 0, 0);
  vtkm::Id3 spc(1, 1, 1);
  dataSet = dataSetBuilder.Create(dims, org, spc);
  std::vector<T> data_vec(shape[0] * shape[1] * shape[2]);
  for (int i = 0; i < shape[0] * shape[1] * shape[2]; i++) {
    data_vec[i] = data[i];
  }
  std::string field_name = "test_field";
  dataSet.AddPointField(field_name, data_vec);
  vtkm::filter::contour::Contour contour_filter;
  contour_filter.SetGenerateNormals(true);
  contour_filter.SetMergeDuplicatePoints(false);
  contour_filter.SetIsoValue(0, iso_value);
  // contour_filter.SetIsoValue(1, iso_value+10);
  contour_filter.SetActiveField(field_name);
  contour_filter.SetFieldsToPass({field_name});
  mgard_x::Timer t;
  t.start();
  mgard_x::DeviceRuntime<mgard_x::CUDA>::SyncDevice();
  vtkm::cont::DataSet outputData = contour_filter.Execute(dataSet);
  mgard_x::DeviceRuntime<mgard_x::CUDA>::SyncDevice();
  t.end();
  t.print("VTKM");
  t.clear();

  std::cout << "vtkm::FlyingEdges::numPoints: "
            << outputData.GetNumberOfPoints() << "\n";
  std::cout << "vtkm::FlyingEdges::numTris: " << outputData.GetNumberOfCells()
            << "\n";

  vtkm_render(outputData, shape, field_name, "vtkm_render_output");
}

template <mgard_x::DIM D, typename T>
void test_mine(T *original_data, std::vector<mgard_x::SIZE> shape,
               T iso_value) {

  mgard_x::SIZE numTriangles;
  mgard_x::SIZE *Triangles;
  mgard_x::SIZE numPoints;
  T *Points;

  mgard_x::Array<3, T, mgard_x::CUDA> v(shape);
  v.load(original_data);

  // mgard_x::PrintSubarray("input", mgard_x::SubArray<3, T, mgard_x::CUDA>(v));

  mgard_x::Array<1, mgard_x::SIZE, mgard_x::CUDA> TrianglesArray;
  mgard_x::Array<1, T, mgard_x::CUDA> PointsArray;
  double pass1_time = 0, pass2_time = 0, pass3_time = 0, pass4_time = 0;
  mgard_x::flying_edges::FlyingEdges<T, mgard_x::CUDA>().Execute(
      shape[0], shape[1], shape[2], mgard_x::SubArray<3, T, mgard_x::CUDA>(v),
      iso_value, TrianglesArray, PointsArray, pass1_time, pass2_time,
      pass3_time, pass4_time, 0);
  mgard_x::DeviceRuntime<mgard_x::CUDA>::SyncQueue(0);
  // printf("mgard_x::FlyingEdges: %f\n", pass1_time + pass2_time + pass3_time +
  // pass4_time);
  double total_time = pass1_time + pass2_time + pass3_time + pass4_time;
  printf("Full res: \n");
  printf("Pass1 time: %f s\n", pass1_time);
  printf("Pass2 time: %f s\n", pass2_time);
  printf("Pass3 time: %f s\n", pass3_time);
  printf("Pass4 time: %f s\n", pass4_time);
  printf("Total time: %f s\n", total_time);
  std::cout << "mgard_x::FlyingEdges::numPoints: " << PointsArray.shape(0) / 3
            << "\n";
  std::cout << "mgard_x::FlyingEdges::numTris: " << TrianglesArray.shape(0) / 3
            << "\n";

  std::string field_name = "test_field";
  vtkm::cont::DataSet dataset =
      ArrayToDataset(shape, iso_value, TrianglesArray, PointsArray, field_name);
  vtkm_render(dataset, shape, field_name, "my_flying_edges");
}

namespace mgard_x {

template <DIM D, typename T, typename DeviceType> struct SurfaceDetect {
  T iso_value;

  SurfaceDetect(T iso_value) : iso_value(iso_value) {}
  bool operator()(
      AdaptiveResolutionTreeNode<D, T, DeviceType> *node,
      typename AdaptiveResolutionTreeNode<D, T, DeviceType>::T_error error,
      SubArray<D, T, DeviceType> v) {
    SIZE data_index[D];
    T max_data = std::numeric_limits<T>::min();
    T min_data = std::numeric_limits<T>::max();
    // std::cout << "cell: ";
    for (int i = 0; i < std::pow(2, D); i++) {
      int linearized_index = i;
      for (int d = 0; d < D; d++) {
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
      // std::cout << data << ", ";
    }

    // std::cout << "("<<node->index_start[2] << ", " << node->index_start[1] <<
    // ", " << node->index_start[0] << "), "; std::cout <<
    // "("<<node->index_end[2] << ", " << node->index_end[1] << ", " <<
    // node->index_end[0] << ")";

    // return true;
    if (max_data + error >= iso_value && min_data - error <= iso_value) {
      // std::cout << "max_data: " << max_data << " "
      //        << "min_data: " << min_data << " "
      //        << "error: " << error << "\n";
      // std::cout << "Keep\n";
      return true;
    } else {
      // std::cout << "Discard\n";
      return false;
    }
  }
};

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>
refactor_and_recompose_data_levelwise(Array<D, T, DeviceType> in_array,
                                      std::vector<SIZE> shape, T tol,
                                      T iso_value, bool debug) {
  T max_data = std::numeric_limits<T>::min();
  T min_data = std::numeric_limits<T>::max();
  T *data = in_array.hostCopy();
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      for (int k = 0; k < shape[2]; k++) {
        max_data = std::max(max_data,
                            data[i * shape[1] * shape[0] + j * shape[0] + k]);
        min_data = std::min(min_data,
                            data[i * shape[1] * shape[0] + j * shape[0] + k]);
      }
    }
  }

  // printf("max_data: %.10f, min_data: %.10f\n", max_data, min_data);

  // std::cout << "Preparing data...";
  //... load data into in_array_cpu
  Hierarchy<D, T, DeviceType> hierarchy(shape);
  // Array<D, T, DeviceType> in_array(shape);
  // in_array.load(data);
  SubArray in_subarray(in_array);

  Array<D, T, DeviceType> org_array = in_array;

  Array<D + 1, T, DeviceType> *max_abs_coefficient =
      new Array<D + 1, T, DeviceType>[hierarchy.l_target()];
  SubArray<D + 1, T, DeviceType> *max_abs_coefficient_subarray =
      new SubArray<D + 1, T, DeviceType>[hierarchy.l_target()];
  for (int l = 0; l < hierarchy.l_target(); l++) {
    std::vector<SIZE> max_abs_coefficient_shape(D + 1);
    for (int d = 1; d < D + 1; d++) {
      max_abs_coefficient_shape[d] = hierarchy.shapes_vec[l + 1][D - d] - 1;
    }
    max_abs_coefficient_shape[0] = l + 1;
    max_abs_coefficient[l] =
        Array<D + 1, T, DeviceType>(max_abs_coefficient_shape);
    max_abs_coefficient[l].memset(0);
    max_abs_coefficient_subarray[l] = SubArray(max_abs_coefficient[l]);
  }

  Array<D, SIZE, DeviceType> *refinement_flag =
      new Array<D, SIZE, DeviceType>[hierarchy.l_target() + 1];
  SubArray<D, SIZE, DeviceType> *refinement_flag_subarray =
      new SubArray<D, SIZE, DeviceType>[hierarchy.l_target() + 1];
  for (int l = 0; l < hierarchy.l_target() + 1; l++) {
    std::vector<SIZE> refinement_flag_shape(D);
    for (int d = 0; d < D; d++) {
      refinement_flag_shape[d] = hierarchy.shapes_vec[l][D - 1 - d] - 1;
    }
    refinement_flag[l] =
        Array<D, SIZE, DeviceType>(refinement_flag_shape, false);
    refinement_flag[l].memset(0);
    refinement_flag_subarray[l] = SubArray(refinement_flag[l]);
    // if (l == hierarchy.l_target) {
    //   SIZE one = 1;
    //   for (int i = 0; i < refinement_flag_subarray[l].shape(0); i++) {
    //     for (int j = 0; j < refinement_flag_subarray[l].shape(1); j++) {
    //       for (int k = 0; k < refinement_flag_subarray[l].shape(2); k++) {
    //         MemoryManager<DeviceType>::Copy1D(refinement_flag_subarray[l](i,
    //         j, k), &one, 1, 0);
    //       }
    //     }
    //   }
    //   DeviceRuntime<DeviceType>::SyncQueue(0);
    // }
  }

  Array<1, T, DeviceType> level_max({hierarchy.l_target() + 1});
  SubArray<1, T, DeviceType> level_max_subarray(level_max);

  // std::cout << "Done\n";

  // PrintSubarray("Input data", SubArray(org_array));

  // std::cout << "Decomposing with MGARD-X CUDA backend...\n";
  multidim_refactoring_debug_print = debug;
  decompose_adaptive_resolution(hierarchy, in_subarray, hierarchy.l_target(),
                                level_max_subarray,
                                max_abs_coefficient_subarray, 0);

  DeviceRuntime<DeviceType>::SyncQueue(0);

  // PrintSubarray("Decomposed data", in_subarray);
  // PrintSubarray("level_max", level_max_subarray);
  // for (int l = 0; l < hierarchy.l_target; l++) {
  //   PrintSubarray4D("max_abs_coefficient_subarray level = " +
  //   std::to_string(l), max_abs_coefficient_subarray[l]);
  // }

  // std::cout << "Done\n";

  // std::cout << "Recomposing with MGARD-X CUDA backend...\n";

  multidim_refactoring_debug_print = debug;
  Array<D, T, DeviceType> result_data = recompose_adaptive_resolution(
      hierarchy, in_subarray, hierarchy.l_target(), iso_value, tol,
      level_max_subarray, max_abs_coefficient_subarray,
      refinement_flag_subarray, 0);

  // printf("shape: %u %u %u\n", result_data.shape(0), result_data.shape()[1],
  // result_data.shape()[2]);
  if (debug) {
    PrintSubarray("result_data", SubArray(result_data));
  }

  return result_data;

  /*
  SubArray<1, SIZE, CUDA> * refinement_flag_linearized_subarray = new
  SubArray<1, SIZE, CUDA>[hierarchy.l_target+1]; SIZE num_cell = 0; for (int l =
  hierarchy.l_target; l >=0 ; l--) { Array<1, SIZE, CUDA> result({1}); SubArray
  result_subarray(result); refinement_flag_linearized_subarray[l] =
  refinement_flag_subarray[l].Linearize();
    DeviceCollective<CUDA>::Sum(refinement_flag_linearized_subarray[l].getShape(0),
  refinement_flag_linearized_subarray[l], result_subarray, 0);
    DeviceRuntime<CUDA>::SyncQueue(0);
    std::cout << "feature cells[" << l << "]: " << *result.hostCopy() << "/" <<
  refinement_flag_linearized_subarray[l].getShape(0) << "\n"; if (l == 0)
  num_cell = *result.hostCopy();
  }

  std::vector<SIZE> write_index_shape(D);
  for (int d = 0; d < D; d++) {
    write_index_shape[d] = hierarchy.shapes_vec[0][D-1-d]-1;
  }
  Array<D, SIZE, CUDA> write_index_array(write_index_shape, false);
  write_index_array.memset(0);
  SubArray<D, SIZE, CUDA> write_index(write_index_array);
  SubArray<1, SIZE, CUDA> write_index_linearized = write_index.Linearize();
  SubArray<1, SIZE, CUDA> refinement_flag_linearized =
  refinement_flag_subarray[0].Linearize();
  DeviceCollective<CUDA>::ScanSumExclusive(write_index_linearized.getShape(0),
  refinement_flag_linearized, write_index_linearized, 0);
  DeviceRuntime<CUDA>::SyncQueue(0);

  // PrintSubarray("refinement_flag_subarray[0]", refinement_flag_subarray[0]);
  // PrintSubarray("write_index_linearized", write_index_linearized);

  CompressedSparseCell<T, CUDA> csc(num_cell);

  DenseToCompressedSparseCell(in_subarray, refinement_flag_subarray[0],
                              write_index, csc, 0);
  DeviceRuntime<CUDA>::SyncQueue(0);
  // csc.Print();

  mgard_x::Array<1, mgard_x::SIZE, mgard_x::CUDA> TrianglesArray;
  mgard_x::Array<1, T, mgard_x::CUDA> PointsArray;
  SparseFlyingCells<D, T, CUDA>().Execute(csc, iso_value, TrianglesArray,
  PointsArray, 0);


  // bool interpolate_full_resolution = true;
  // SurfaceDetect<D, T, CUDA> surface_detector(iso_value);
  // std::vector<CompressedSparseEdge<D, T, CUDA>> cse_list;
  // CompressedSparseCell<T, CUDA> csc;
  // Array<D, T, CUDA> out_array = recompose_adaptive_resolution(hierarchy,
  in_subarray, tol, interpolate_full_resolution, surface_detector, cse_list,
  csc, 0);
  // std::cout << "Done\n";

  // DeviceRuntime<CUDA>::SyncQueue(0);
  // size_t n = 1;
  // for (int i = 0; i < shape.size(); i++) n *= shape[i];
  // enum error_bound_type mode = error_bound_type::ABS;
  // std::cout << "L_inf_error: " << L_inf_error(n, org_array.hostCopy(),
  out_array.hostCopy(), mode) << "\n";

  // mgard_x::Array<1, mgard_x::SIZE, mgard_x::CUDA> TrianglesArray;
  // mgard_x::Array<1, T, mgard_x::CUDA> PointsArray;
  // SparseFlyingCells<D, T, CUDA>().Execute(csc, iso_value, TrianglesArray,
  PointsArray, 0); std::string field_name = "test_field"; vtkm::cont::DataSet
  dataset = ArrayToDataset(shape, iso_value, TrianglesArray, PointsArray,
  field_name); vtkm_render(dataset, shape, field_name, "my_csc_render_output");


  // for (int i = 0; i < cse_list.size(); i++) {
  //   if(!cse_list[i].empty) {
  //     mgard_x::Array<1, mgard_x::SIZE, mgard_x::CUDA> TrianglesArray;
  //     mgard_x::Array<1, T, mgard_x::CUDA> PointsArray;
  //     SparseFlyingEdges<D, T, CUDA>().Execute(cse_list[i], iso_value,
  TrianglesArray, PointsArray, 0);
  //     std::string field_name = "test_field";
  //     vtkm::cont::DataSet dataset = ArrayToDataset(shape, iso_value,
  TrianglesArray, PointsArray, field_name);
  //     vtkm_render(dataset, shape, field_name, "my_render_output");
  //   }
  // }
  */
}

template <DIM D, class T_data, class T_bitplane, class T_error,
          typename DeviceType, class Decomposer, class Interleaver,
          class Encoder, class Compressor, class ErrorCollector, class Writer>
void refactor(Array<D, T_data, DeviceType> data, std::vector<SIZE> shape,
              int target_level, int num_bitplanes,
              Hierarchy<D, T_data, DeviceType> &hierarchy,
              Decomposer decomposer, Interleaver interleaver, Encoder encoder,
              Compressor compressor, ErrorCollector collector, Writer writer) {

  auto refactor =
      MDR::ComposedRefactor<D, T_data, T_bitplane, T_error, Decomposer,
                            Interleaver, Encoder, Compressor, ErrorCollector,
                            Writer, DeviceType>(hierarchy, decomposer,
                                                interleaver, encoder,
                                                compressor, collector, writer);
  refactor.refactor(data, shape, target_level, num_bitplanes);
}

template <DIM D, class T_data, class T_stream, class T_error,
          typename DeviceType, class Decomposer, class Interleaver,
          class Encoder, class Compressor, class ErrorEstimator,
          class SizeInterpreter, class Retriever>
Array<D, T_data, DeviceType>
reconstructor(T_error tol, Hierarchy<D, T_data, DeviceType> &hierarchy,
              Decomposer decomposer, Interleaver interleaver, Encoder encoder,
              Compressor compressor, ErrorEstimator estimator,
              SizeInterpreter interpreter, Retriever retriever) {
  auto reconstructor =
      MDR::ComposedReconstructor<D, T_data, T_stream, Decomposer, Interleaver,
                                 Encoder, Compressor, SizeInterpreter,
                                 ErrorEstimator, Retriever, DeviceType>(
          hierarchy, decomposer, interleaver, encoder, compressor, interpreter,
          retriever);
  reconstructor.load_metadata();
  std::vector<T_error> tolerance = {tol};
  return reconstructor.reconstruct(tolerance[0]);
}

template <DIM D, typename T_data, typename DeviceType>
Array<D, T_data, DeviceType> refactor_and_recompose_data_bitplanwise(
    Array<D, T_data, DeviceType> in_array, std::vector<SIZE> shape,
    SIZE block_id, double tol, double iso_value, bool debug) {
  using T_stream = uint32_t;
  using T_error = double;
  int num_bitplanes = 64;
  int uniform_coord_mode = 0; // is this necessary?
  Hierarchy<D, T_data, DeviceType> hierarchy(shape, uniform_coord_mode);
  SIZE target_level = hierarchy.l_target();
  std::string metadata_file =
      "refactored_data/block_" + std::to_string(block_id) + "_metadata.bin";
  std::vector<std::string> files;
  for (int i = 0; i <= target_level; i++) {
    std::string filename = "refactored_data/block_" + std::to_string(block_id) +
                           "_level_" + std::to_string(i) + ".bin";
    files.push_back(filename);
  }
  auto decomposer =
      MDR::MGARDOrthoganalDecomposer<D, T_data, DeviceType>(hierarchy);
  auto interleaver = MDR::DirectInterleaver<D, T_data, DeviceType>(hierarchy);
  auto encoder = MDR::GroupedBPEncoder<T_data, T_stream, T_error, DeviceType>();
  auto compressor = MDR::DefaultLevelCompressor<T_stream, DeviceType>();
  auto collector = MDR::MaxErrorCollector<T_data>();
  auto writer = MDR::ConcatLevelFileWriter(metadata_file, files);
  refactor<D, T_data, T_stream, T_error, DeviceType>(
      in_array, shape, target_level, num_bitplanes, hierarchy, decomposer,
      interleaver, encoder, compressor, collector, writer);

  auto retriever = MDR::ConcatLevelFileRetriever(metadata_file, files);
  auto estimator = MDR::MaxErrorEstimatorOB<T_data>(D);
  auto interpreter = MDR::SignExcludeGreedyBasedSizeInterpreter<
      MDR::MaxErrorEstimatorOB<T_data>>(estimator);
  // auto reconstructor = MDR::ComposedReconstructor(hierarchy, decomposer,
  // interleaver, encoder, compressor, interpreter, retriever);
  // reconstructor.load_metadata();
  // auto reconstructed_data =
  // reconstructor.progressive_reconstruct(tolerance[0]);
  return reconstructor<D, T_data, T_stream, T_error, DeviceType>(
      tol, hierarchy, decomposer, interleaver, encoder, compressor, estimator,
      interpreter, retriever);
}

template <typename T, typename DeviceType>
std::string hash_key(SubArray<3, T, DeviceType> subArray) {
  return std::to_string(subArray.shape(0)) + "-" +
         std::to_string(subArray.shape(1)) + "-" +
         std::to_string(subArray.shape(2));
}

template <DIM D, typename T>
void test_adaptive_resolution(T *data, std::vector<SIZE> shape, SIZE block_size,
                              T tol, T iso_value) {

  Array<D, T, CUDA> data_array(shape);
  data_array.load(data);
  SubArray data_subarray(data_array);
  std::unordered_map<std::string, std::vector<SubArray<D, T, CUDA>>> umap;

  // PrintSubarray("data", data_subarray);
  SIZE num_block_r = (shape[0] - 1) / block_size + 1;
  SIZE num_block_c = (shape[1] - 1) / block_size + 1;
  SIZE num_block_f = (shape[2] - 1) / block_size + 1;

  Array<D, T, CUDA> *data_block =
      new Array<D, T, CUDA>[num_block_r * num_block_c * num_block_f];
  Array<D, T, CUDA> *recomposed_data_blocks =
      new Array<D, T, CUDA>[num_block_r * num_block_c * num_block_f];
  Array<D, T, CUDA> *recomposed_data_blocks2 =
      new Array<D, T, CUDA>[num_block_r * num_block_c * num_block_f];
  double pass1_time = 0, pass2_time = 0, pass3_time = 0, pass4_time = 0,
         total_time = 0;
  for (SIZE r = 0; r < num_block_r; r++) {
    for (SIZE c = 0; c < num_block_c; c++) {
      for (SIZE f = 0; f < num_block_f; f++) {
        SIZE start_r = r * block_size;
        SIZE block_size_r = std::min(block_size, shape[0] - start_r);
        SIZE start_c = c * block_size;
        SIZE block_size_c = std::min(block_size, shape[1] - start_c);
        SIZE start_f = f * block_size;
        SIZE block_size_f = std::min(block_size, shape[2] - start_f);
        SIZE linearized_index =
            r * num_block_c * num_block_f + c * num_block_f + f;
        std::vector<SIZE> block_shape{block_size_r, block_size_c, block_size_f};
        data_block[linearized_index] = Array<D, T, CUDA>(block_shape);
        SubArray data_block_src = data_subarray;
        data_block_src.offset({start_f, start_c, start_r});
        data_block_src.resize({block_size_f, block_size_c, block_size_r});
        SubArray data_block_des(data_block[linearized_index]);
        CopyND(data_block_src, data_block_des, 0);
        DeviceRuntime<CUDA>::SyncQueue(0);

        bool debug = false;
        // printf("block %u ", linearized_index);
        // PrintSubarray("data_block_src", data_block_src);
        if (linearized_index == 0) {
          debug = false;
        }
        // recomposed_data_blocks[linearized_index] =
        // refactor_and_recompose_data_levelwise(data_block[linearized_index],
        // {block_size_r, block_size_c, block_size_f}, tol, iso_value, debug);
        recomposed_data_blocks[linearized_index] =
            refactor_and_recompose_data_bitplanwise(
                data_block[linearized_index],
                {block_size_r, block_size_c, block_size_f}, linearized_index,
                tol, iso_value, debug);
        SubArray recomposed_data_block(
            recomposed_data_blocks[linearized_index]);

        // if (recomposed_data_blocks[linearized_index].shape(0) == 64) {
        //   printf("linearized_index: %u\n", linearized_index);
        // }
        // { // debug
        //   if (linearized_index == 300) {
        //     SubArray org(data_block[linearized_index]);
        //     org.resize({5,5,5});
        //     SubArray rep(recomposed_data_blocks[linearized_index]);
        //     rep.resize({5,5,5});
        //     PrintSubarray("org", org);
        //     PrintSubarray("rep", rep);
        //   }
        // }

        // Run dense FlyingEdges
        Array<1, SIZE, CUDA> TrianglesArray;
        Array<1, T, CUDA> PointsArray;
        double pass1_time_block = 0, pass2_time_block = 0, pass3_time_block = 0,
               pass4_time_block = 0, total_time_block = 0;
        flying_edges::FlyingEdges<T, CUDA>().Execute(
            recomposed_data_block.shape(0), recomposed_data_block.shape(1),
            recomposed_data_block.shape(2), recomposed_data_block, iso_value,
            TrianglesArray, PointsArray, pass1_time_block, pass2_time_block,
            pass3_time_block, pass4_time_block, 0);
        DeviceRuntime<CUDA>::SyncQueue(0);
        // if (recomposed_data_blocks[linearized_index].shape(0) == 64) {
        //   std::cout << "block " << linearized_index << "
        //   mgard_x::FlyingEdges::numPoints: " << PointsArray.shape(0)/3 <<
        //   "\n"; std::cout << "block " << linearized_index << "
        //   mgard_x::FlyingEdges::numTris: " << TrianglesArray.shape(0)/3 <<
        //   "\n";
        // }
        pass1_time += pass1_time_block;
        pass2_time += pass2_time_block;
        pass3_time += pass3_time_block;
        pass4_time += pass4_time_block;
      }
    }
  }
  total_time = pass1_time + pass2_time + pass3_time + pass4_time;

  printf("Non-batched: \n");
  printf("Pass1 time: %f s\n", pass1_time);
  printf("Pass2 time: %f s\n", pass2_time);
  printf("Pass3 time: %f s\n", pass3_time);
  printf("Pass4 time: %f s\n", pass4_time);
  printf("Total time: %f s\n", total_time);

  for (SIZE i = 0; i < num_block_r * num_block_c * num_block_f; i++) {
    SubArray recomposed_data_block(recomposed_data_blocks[i]);
    std::string key = hash_key(recomposed_data_block);
    umap[key].push_back(recomposed_data_block);
  }

  for (auto x : umap) {
    pass1_time = 0, pass2_time = 0, pass3_time = 0, pass4_time = 0,
    total_time = 0;
    std::cout << x.first << " " << x.second.size() << std::endl;
    std::vector<SubArray<D, T, CUDA>> recomposed_data_blocks_vec = x.second;
    SIZE num_batches = (SIZE)recomposed_data_blocks_vec.size();
    const bool pitched = false;
    const bool managed = true;
    Array<1, SubArray<D, T, CUDA>, CUDA> recomposed_data_blocks_subarray(
        {num_batches}, pitched, managed);
    SubArray subarray_of_subarray(recomposed_data_blocks_subarray);
    for (int i = 0; i < num_batches; i++) {
      *subarray_of_subarray(i) = recomposed_data_blocks_vec[i];
    }
    SIZE nr = recomposed_data_blocks_vec[0].shape(0);
    SIZE nc = recomposed_data_blocks_vec[0].shape(1);
    SIZE nf = recomposed_data_blocks_vec[0].shape(2);
    Array<1, Array<1, SIZE, CUDA>, CUDA> Triangles;
    Array<1, Array<1, T, CUDA>, CUDA> Points;
    flying_edges_batched::FlyingEdgesBatched<T, CUDA>().Execute(
        nr, nc, nf, subarray_of_subarray, iso_value, Triangles, Points,
        pass1_time, pass2_time, pass3_time, pass4_time, 0);
    total_time = pass1_time + pass2_time + pass3_time + pass4_time;
    printf("Batched: \n");
    printf("Pass1 time: %f s\n", pass1_time);
    printf("Pass2 time: %f s\n", pass2_time);
    printf("Pass3 time: %f s\n", pass3_time);
    printf("Pass4 time: %f s\n", pass4_time);
    printf("Total time: %f s\n", total_time);
  }

  // printf("mgard_x::FlyingEdges adapt res: %f\n", total_time);
}

} // end of namespace mgard_x

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

mgard_x::SIZE get_arg_int(int argc, char *argv[], std::string option) {
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int i;
    for (i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
      }
    }
    try {
      mgard_x::SIZE d = std::stoi(arg);
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

template <typename T>
T *get_data(std::string input_file, size_t original_size) {
  T *data = NULL;
  if (std::string(input_file).compare("random") == 0) {
    data = new T[original_size];
    srand(7117);
    for (size_t i = 0; i < original_size; i++) {
      // data[i] = (T)rand() / RAND_MAX;
      if (i < 2)
        data[i] = 1;
      else
        data[i] = 0;
    }
  } else {
    readfile(input_file.c_str(), data);
  }
  return data;
}

int main(int argc, char *argv[]) {
  std::string input_file = get_arg(argc, argv, "-i");
  std::string dt = get_arg(argc, argv, "-t");
  std::vector<mgard_x::SIZE> shape = get_arg_dims(argc, argv, "-n");
  mgard_x::SIZE block_size = get_arg_int(argc, argv, "-b");
  double tol = get_arg_double(argc, argv, "-e");
  double iso_value = get_arg_double(argc, argv, "-v");
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < shape.size(); i++)
    original_size *= shape[i];

  if (dt.compare("s") == 0) {
    float *data = get_data<float>(input_file, original_size);
    test_vtkm<3, float>(argc, argv, data, shape, (float)tol, (float)iso_value);
    test_mine<3, float>(data, shape, (float)iso_value);
    mgard_x::test_adaptive_resolution<3, float>(data, shape, block_size,
                                                (float)tol, (float)iso_value);
  } else if (dt.compare("d") == 0) {
    double *data = get_data<double>(input_file, original_size);
    test_vtkm<3, double>(argc, argv, data, shape, (float)tol, (float)iso_value);
    test_mine<3, double>(data, shape, (float)iso_value);
    mgard_x::test_adaptive_resolution<3, double>(
        data, shape, block_size, (double)tol, (double)iso_value);

  } else {
    std::cout << "wrong data type.\n";
  }
}