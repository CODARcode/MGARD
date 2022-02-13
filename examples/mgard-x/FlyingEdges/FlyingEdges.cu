

// #include "/home/jieyang/dev/MGARD/include/cuda/FlyingEdges.hpp"

#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandlePermutation.h>
#include <vtkm/cont/ArrayHandleUniformPointCoordinates.h>
#include <vtkm/cont/ColorTable.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/cont/Initialize.h>
#include <vtkm/cont/Invoker.h>
#include <vtkm/io/reader/VTKDataSetReader.h>
#include <vtkm/io/writer/VTKDataSetWriter.h>

#include <vtkm/worklet/SurfaceNormals.h>
#include <vtkm/worklet/contour/CommonState.h>
#include <vtkm/worklet/contour/FieldPropagation.h>
#include <vtkm/worklet/contour/FlyingEdges.h>
#include <vtkm/worklet/contour/MarchingCells.h>

#include <vtkm/filter/Contour.h>
#include <vtkm/filter/FieldMetadata.h>
#include <vtkm/filter/FilterDataSetWithField.h>
#include <vtkm/filter/MapFieldPermutation.h>
#include <vtkm/filter/PolicyBase.h>

#include <vtkm/rendering/Actor.h>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperRayTracer.h>
#include <vtkm/rendering/MapperWireframer.h>
#include <vtkm/rendering/Scene.h>
#include <vtkm/rendering/View3D.h>

#include "FlyingEdges.hpp"

bool require_arg(int argc, char *argv[], std::string option) {
  for (int i = 0; i < argc; i++) {
    if (option.compare(std::string(argv[i])) == 0) {
      return true;
    }
  }
  exit(-1);
}

template <typename T> size_t readfile(const char *input_file, T *&in_buff) {
  std::cout << mgard_x::log::log_info << "Loading file: " << input_file << "\n";

  FILE *pFile;
  pFile = fopen(input_file, "rb");
  if (pFile == NULL) {
    std::cout << mgard_x::log::log_err << "file open error!\n";
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

int get_arg_int(int argc, char *argv[], std::string option) {
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int i;
    for (i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
      }
    }
    try {
      int d = std::stoi(arg);
      return d;
    } catch (std::invalid_argument const &e) {
      exit(-1);
    }
  }
  return 0;
}

std::vector<mgard_x::SIZE> get_arg_dims(int argc, char *argv[],
                                        std::string option) {
  std::vector<mgard_x::SIZE> shape;
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int arg_idx = 0;
    for (int i = 0; i < argc; i++) {
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
      exit(-1);
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
      exit(-1);
    }
  }
  return 0;
}

template <typename T>
void test_vtkm(int argc, char *argv[], std::vector<mgard_x::SIZE> shape,
               T *original_data, T iso_value, mgard_x::SIZE &numTriangles,
               mgard_x::SIZE *&Triangles, mgard_x::SIZE &numPoints,
               T *&Points) {
  // vtkm::cont::InitializeOptions options = vtkm::cont::InitializeOptions::
  vtkm::cont::Initialize(argc, argv);
  vtkm::cont::RuntimeDeviceTracker &deviceTracker =
      vtkm::cont::GetRuntimeDeviceTracker();
  deviceTracker.ForceDevice(vtkm::cont::DeviceAdapterTagCuda());
  vtkm::Id3 dims(shape[0], shape[1], shape[2]);
  vtkm::Id3 org(0, 0, 0);
  vtkm::Id3 spc(1, 1, 1);

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < shape.size(); i++)
    original_size *= shape[i];

  vtkm::cont::DataSet inputDataSet;
  vtkm::cont::DataSetBuilderUniform dataSetBuilderUniform;
  vtkm::cont::DataSetFieldAdd dsf;
  inputDataSet = dataSetBuilderUniform.Create(dims, org, spc);
  std::vector<T> vec_data(original_data, original_data + original_size);
  dsf.AddPointField(inputDataSet, "v", vec_data);

  vtkm::filter::Contour contour;
  contour.SetGenerateNormals(true);
  contour.SetMergeDuplicatePoints(true);
  contour.SetNumberOfIsoValues(1);
  contour.SetIsoValue(0, iso_value);
  contour.SetActiveField("v");

  vtkm::cont::DataSet ds_from_mc = contour.Execute(inputDataSet);

  vtkm::cont::CellSetSingleType<> TriangleCells =
      ds_from_mc.GetCellSet().template Cast<vtkm::cont::CellSetSingleType<>>();
  Triangles = new mgard_x::SIZE[TriangleCells.GetNumberOfCells() * 3];
  numTriangles = TriangleCells.GetNumberOfCells();

  if (numTriangles == 0) {
    return;
  }

  int index = 0;
  for (vtkm::Id cellId = 0; cellId < TriangleCells.GetNumberOfCells();
       cellId++) {
    vtkm::Id ids[3];
    TriangleCells.GetCellPointIds(cellId, ids);
    Triangles[index] = ids[0];
    Triangles[index + 1] = ids[1];
    Triangles[index + 2] = ids[2];
    index += 3;
  }

  vtkm::cont::CoordinateSystem PointField = ds_from_mc.GetCoordinateSystem();
  vtkm::cont::ArrayHandle<vtkm::Vec3f> PointArray =
      PointField.GetData()
          .template AsArrayHandle<vtkm::cont::ArrayHandle<vtkm::Vec3f>>();

  Points = new T[PointField.GetNumberOfPoints() * 3];
  numPoints = PointField.GetNumberOfPoints();

  if (numPoints == 0) {
    return;
  }

  index = 0;
  for (vtkm::Id pointId = 0; pointId < PointField.GetNumberOfPoints();
       pointId++) {
    vtkm::Vec3f coord = PointArray.ReadPortal().Get(pointId);

    Points[index] = coord[0];
    Points[index + 1] = coord[1];
    Points[index + 2] = coord[2];
    index += 3;
  }
}

template <typename T>
void test_mine(std::vector<mgard_x::SIZE> shape, T *original_data, T iso_value,
               mgard_x::SIZE &numTriangles, mgard_x::SIZE *&Triangles,
               mgard_x::SIZE &numPoints, T *&Points) {
  mgard_x::Array<3, T, mgard_x::CUDA> v({shape[2], shape[1], shape[0]});
  v.loadData(original_data);

  mgard_x::Array<1, mgard_x::SIZE, mgard_x::CUDA> TrianglesArray;
  mgard_x::Array<1, T, mgard_x::CUDA> PointsArray;

  mgard_x::FlyingEdges<T, mgard_x::CUDA>().Execute(
      shape[2], shape[1], shape[0], mgard_x::SubArray<3, T, mgard_x::CUDA>(v),
      iso_value, TrianglesArray, PointsArray, 0);

  numTriangles = TrianglesArray.getShape()[0] / 3;
  numPoints = PointsArray.getShape()[0] / 3;

  if (numTriangles == 0 || numPoints == 0) {
    printf("returing %u %u from test_mine\n", numTriangles, numPoints);
    return;
  }

  Triangles = new mgard_x::SIZE[TrianglesArray.getShape()[0]];
  Points = new T[PointsArray.getShape()[0]];

  memcpy(Triangles, TrianglesArray.getDataHost(),
         numTriangles * 3 * sizeof(mgard_x::SIZE));
  memcpy(Points, PointsArray.getDataHost(), numPoints * 3 * sizeof(T));

  // mgard_x::PrintSubarray("Triangles", mgard_x::SubArray(TrianglesArray));
  // mgard_x::PrintSubarray("Points", mgard_x::SubArray(PointsArray));
}

int main(int argc, char *argv[]) {

  std::cout << "start\n";
  std::string input_file = get_arg(argc, argv, "-i");
  mgard_x::DIM D = get_arg_int(argc, argv, "-n");
  std::vector<mgard_x::SIZE> shape = get_arg_dims(argc, argv, "-n");
  float iso_value = (float)get_arg_double(argc, argv, "-s");

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  float *original_data;
  size_t in_size = 0;

  if (std::string(input_file).compare("random") == 0) {
    std::cout << "generating data...";
    in_size = original_size * sizeof(float);
    original_data = new float[original_size];

    for (size_t i = 0; i < shape[2]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        float value = j % 20;
        for (size_t k = 0; k < shape[0]; k++) {
          original_data[i * shape[1] * shape[0] + j * shape[0] + k] = value;
        }
      }
    }
    std::cout << "Done\n";
  } else {
    in_size = readfile(input_file.c_str(), original_data);
  }
  if (in_size != original_size * sizeof(float)) {
    std::cout << mgard_x::log::log_err << "input file size mismatch" << in_size
              << "vs." << original_size * sizeof(float) << "!\n";
  }

  mgard_x::SIZE numTriangles_vtkm;
  mgard_x::SIZE *Triangles_vtkm;
  mgard_x::SIZE numPoints_vtkm;
  float *Points_vtkm;

  mgard_x::SIZE numTriangles_mine;
  mgard_x::SIZE *Triangles_mine;
  mgard_x::SIZE numPoints_mine;
  float *Points_mine;

  // float iso_value = 2e6;
  // std::cout << "test_vtkm\n";
  // test_vtkm(argc, argv, shape, original_data, iso_value, numTriangles_vtkm,
  // Triangles_vtkm, numPoints_vtkm, Points_vtkm);
  std::cout << "test_mine\n";
  test_mine(shape, original_data, iso_value, numTriangles_mine, Triangles_mine,
            numPoints_mine, Points_mine);

  // if (numTriangles_vtkm != numTriangles_mine) { printf("numTriangles
  // mismatch! %u vs. %u\n", numTriangles_vtkm, Triangles_mine);} bool match =
  // true; for (size_t i = 0; i < numTriangles_vtkm * 3; i++) {
  //   if (Triangles_vtkm[i] != Triangles_mine[i]) {
  //     match = false;
  //     // printf("diff at %u: %u %u\n", i, Triangles_vtkm[i],
  //     Triangles_mine[i]); break;
  //   }
  // }

  // printf("Triangles: %s\n", match ? "pass": "no pass");

  // if (numPoints_vtkm != numPoints_mine) { printf("numPoints mismatch! %u vs.
  // %u\n", numPoints_vtkm, numPoints_mine);} match = true; for (size_t i = 0; i
  // < numPoints_vtkm * 3; i++) {
  //   if (std::fabs(Points_vtkm[i] - Points_mine[i]) > 1e-2) {
  //     match = false;
  //     // printf("diff at %u: %f %f\n", i, Points_vtkm[i], Points_mine[i]);
  //     break;
  //   }
  // }

  // printf("Points: %s\n", match ? "pass": "no pass");
}