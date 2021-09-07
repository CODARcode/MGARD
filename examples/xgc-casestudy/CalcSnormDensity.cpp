#include <math.h>
#include <cstddef>

#include <array>
#include <iostream>
#include <stdexcept>

#include "mgard/TensorQuantityOfInterest.hpp"
#include "mgard/compress.hpp"
#include "adios2.h"

class AverageFunctional {
public:
  AverageFunctional(const std::array<std::size_t, 2> lower_left,
                    const std::array<std::size_t, 2> upper_right, double vol)
      : lower_left(lower_left), upper_right(upper_right), vol(vol) {
    for (std::size_t i = 0; i < 2; ++i) {
      if (upper_right.at(i) <= lower_left.at(i)) {
        throw std::invalid_argument("invalid region");
      }
    }
  }

  double operator()(const mgard::TensorMeshHierarchy<2, double> &hierarchy,
                   double  const *const u) const {
    const std::array<std::size_t, 2> shape = hierarchy.shapes.back();
    const std::size_t n = shape.at(0);
    const std::size_t m = shape.at(1);
    if (upper_right.at(0) > n || upper_right.at(1) > m) {
      throw std::invalid_argument("region isn't contained in domain");
    }
    double total = 0;
    std::size_t count = 0;
    for (std::size_t i = lower_left.at(0); i < upper_right.at(0); ++i) {
      for (std::size_t j = lower_left.at(1); j < upper_right.at(1); ++j) {
        double coeff = (i==lower_left.at(0) || i==upper_right.at(0) || j==lower_left.at(1) || j==upper_right.at(1)) ?
            (((i==lower_left.at(0) || i==upper_right.at(0)) && (j==lower_left.at(1) || j==upper_right.at(1))) ? 0.25 : 0.5) : 1.0;
        total += u[n * i + j] * coeff;
        ++count;
      }
    }
    return total * vol;
  }

private:
  std::array<std::size_t, 2> lower_left;
  std::array<std::size_t, 2> upper_right;
  double vol;
};

// input: path to xgc.f0.mesh.bp
int main(int argc, char **argv) {
  char readin_f[2048];
  strcpy(readin_f, argv[1]);
  size_t vx, vy, nnodes;
  adios2::ADIOS ad;
  adios2::IO reader_io = ad.DeclareIO("XGC");
  adios2::Engine reader_vol = reader_io.Open(readin_f, adios2::Mode::Read);
  adios2::Variable<double> var_i_f_in;
  var_i_f_in = reader_io.InquireVariable<double>("f0_grid_vol_vonly");
  nnodes = var_i_f_in.Shape()[1];
  std::cout << "number of nnodes: " << nnodes << "\n";
  var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, 0}, {1, nnodes}));
  std::vector<double> grid_vol;
  reader_vol.Get<double>(var_i_f_in, grid_vol);
  reader_vol.PerformGets();
  int32_t temp;
  adios2::Variable<int32_t> var_sz_in;
  var_sz_in = reader_io.InquireVariable<int32_t>("f0_nmu");
  reader_vol.Get<int32_t>(var_sz_in, &temp);
  reader_vol.PerformGets();
  vx = (size_t)temp*2 + 1;
  var_sz_in = reader_io.InquireVariable<int32_t>("f0_nvp");
  reader_vol.Get<int32_t>(var_sz_in, &temp);
  reader_vol.PerformGets();
  vy = (size_t)temp + 1;
  reader_vol.Close();

  const mgard::TensorMeshHierarchy<2, double> hierarchy({vx, vy});
  std::vector<double> Q_norm(nnodes);
  adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
  adios2::Variable<double> bp_fdata = bpIO.DefineVariable<double>(
          "Q_norm", {nnodes}, {0}, {nnodes},  adios2::ConstantDims);
  adios2::Engine bpFileWriter = bpIO.Open("RsQ.bp", adios2::Mode::Write); 
  for (size_t innode=0; innode<nnodes; innode++) {
    const AverageFunctional average({0, 0}, {vx, vy}, grid_vol.at(innode));
    const mgard::TensorQuantityOfInterest<2, double> Q(hierarchy, average);
    Q_norm.at(innode) = Q.norm(0); 
    if (innode % 100 ==0)
        std::cout << "norm of the average as a functional on L^2: " << Q_norm.at(innode) << " for vol = " << grid_vol.at(innode) <<  std::endl;

  }
  bpFileWriter.Put<double>(bp_fdata, Q_norm.data());
  bpFileWriter.Close();
  
  return 0;
}
