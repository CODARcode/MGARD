// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ben Whitney, Qing Liu
//
// See LICENSE for details.
#ifndef COMPRESS_TPP
#define COMPRESS_TPP

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

#include "MGARDConfig.hpp"
#include "TensorMultilevelCoefficientQuantizer.hpp"
#include "TensorNorms.hpp"
#include "adaptive_roi.hpp"
#include "compressors.hpp"
#include "decompose.hpp"
#include "format.hpp"
#include "quantize.hpp"
#include "shuffle.hpp"

namespace mgard {

using DEFAULT_INT_T = std::int64_t;

template <std::size_t N, typename Real>
CompressedDataset<N, Real>
compress(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v,
         const Real s, const Real tolerance) {
  const std::size_t ndof = hierarchy.ndof();
  Real *const u = new Real[ndof];
  shuffle(hierarchy, v, u);
  pb::Header header;
  populate_defaults(header);
  hierarchy.populate(header);
  decompose(hierarchy, header, u);
  {
    pb::ErrorControl &e = *header.mutable_error_control();
    e.set_mode(pb::ErrorControl::ABSOLUTE);
    if (s == std::numeric_limits<Real>::infinity()) {
      e.set_norm(pb::ErrorControl::L_INFINITY);
    } else {
      e.set_norm(pb::ErrorControl::S_NORM);
      e.set_s(s);
    }
    e.set_tolerance(tolerance);
  }
  MemoryBuffer<unsigned char> quantized = quantization_buffer(header, ndof);
  quantize(hierarchy, header, s, tolerance, u, quantized.data.get());
  MemoryBuffer<unsigned char> buffer =
      compress(header, quantized.data.get(), quantized.size);
  delete[] u;
  return CompressedDataset<N, Real>(hierarchy, header, s, tolerance,
                                    buffer.data.release(), buffer.size);
}

template <std::size_t N, typename Real>
DecompressedDataset<N, Real>
decompress(const CompressedDataset<N, Real> &compressed) {
  const std::size_t ndof = compressed.hierarchy.ndof();
  Real *const dequantized = new Real[ndof];
  Real *const v = new Real[ndof];
  MemoryBuffer<unsigned char> quantized =
      quantization_buffer(compressed.header, ndof);

  decompress(compressed.header, const_cast<void *>(compressed.data()),
             compressed.size(), quantized.data.get(), quantized.size);
  dequantize(compressed, quantized.data.get(), dequantized);

  recompose(compressed.hierarchy, compressed.header, dequantized);
  unshuffle(compressed.hierarchy, dequantized, v);
  delete[] dequantized;
  return DecompressedDataset<N, Real>(compressed, v);
}

template <std::size_t N, typename Real>
CompressedDataset<N, Real>
compress_roi(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v,
             const Real s, const Real tolerance, const std::vector<Real> thresh,
             const std::vector<size_t> init_bw,
             const std::vector<size_t> bw_ratio, const size_t l_th,
             const char *filename, bool wr /*1 for write 0 for read*/) {
  const std::size_t ndof = hierarchy.ndof();
  Real *const u = new Real[ndof];
  shuffle(hierarchy, v, u);
  pb::Header header;
  populate_defaults(header);
  hierarchy.populate(header);
  decompose(hierarchy, header, u);

  {
    pb::ErrorControl &e = *header.mutable_error_control();
    e.set_mode(pb::ErrorControl::ABSOLUTE);
    if (s == std::numeric_limits<Real>::infinity()) {
      e.set_norm(pb::ErrorControl::L_INFINITY);
    } else {
      e.set_norm(pb::ErrorControl::S_NORM);
      e.set_s(s);
    }
    e.set_tolerance(tolerance);
  }
  // QG: create a map for adaptive compression
  //  Real *const unshuffled_u = static_cast<Real *>(std::malloc(ndof *
  //  sizeof(Real)));
  std::vector<Real> unshuffled_u(ndof);
  unshuffle(hierarchy, u, unshuffled_u.data());
  const std::array<std::size_t, N> &SHAPE = hierarchy.shapes.back();
  Real *u_map = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));

  if ((wr == 0) && (filename != NULL)) { // load from existing map files
    FILE *file = fopen(filename, "rb");
    fread(u_map, sizeof(Real), ndof, file);
    fclose(file);
    std::cout << "load umap from file..." << filename << "\n";
  } else {
    int Dim2, r, c, h;
    struct customized_hierarchy<size_t> c_hierarchy;
    c_hierarchy.level = new size_t[ndof];
    c_hierarchy.L = hierarchy.L;
    c_hierarchy.Row = SHAPE[0];
    if (N > 3) {
      std::cout << "Adaptive compression does not support dim > 3!!!\n";
      exit;
    }
    if (N == 1) {
      c_hierarchy.Col = 1;
      c_hierarchy.Height = 1;
    } else {
      c_hierarchy.Col = SHAPE[1];
      if (N == 3) {
        c_hierarchy.Height = SHAPE[2];
      } else {
        c_hierarchy.Height = 1;
      }
    }
    Dim2 = c_hierarchy.Height * c_hierarchy.Col;
    // QG: get the level of each node (can be improved)
    size_t k;
    std::array<std::size_t, N> multiindex;
    if (N == 1) {
      for (std::size_t r = 0; r < SHAPE[0]; r++) {
        multiindex[0] = r;
        c_hierarchy.level[r] = hierarchy.date_of_birth(multiindex);
        u_map[r] = (c_hierarchy.level[r] < l_th) ? BUFFER_ZONE : BACKGROUND;
      }
    } else if (N == 2) {
      for (std::size_t r = 0; r < SHAPE[0]; r++) {
        for (std::size_t c = 0; c < SHAPE[1]; c++) {
          k = r * SHAPE[1] + c;
          multiindex[0] = r;
          multiindex[1] = c;
          c_hierarchy.level[k] = hierarchy.date_of_birth(multiindex);
          u_map[k] = (c_hierarchy.level[k] < l_th) ? BUFFER_ZONE : BACKGROUND;
        }
      }
    } else if (N == 3) {
      for (std::size_t r = 0; r < SHAPE[0]; r++) {
        for (std::size_t c = 0; c < SHAPE[1]; c++) {
          for (std::size_t h = 0; h < SHAPE[2]; h++) {
            k = r * Dim2 + c * SHAPE[0] + h;
            multiindex[0] = r;
            multiindex[1] = c;
            multiindex[2] = h;
            c_hierarchy.level[k] = hierarchy.date_of_birth(multiindex);
            u_map[k] = (c_hierarchy.level[k] < l_th) ? BUFFER_ZONE : BACKGROUND;
          }
        }
      }
    }

    c_hierarchy.l_th = l_th;
    // number of bins in the 1st layer of histogram
    struct std::vector<cube_<size_t>> bin_w(thresh.size() + 1);
    bin_w[0] = {c_hierarchy.Row, c_hierarchy.Col, c_hierarchy.Height};
    bin_w[1].r = init_bw.at(0);
    bin_w[1].c = (init_bw.size() > 1) ? init_bw.at(1) : 1;
    bin_w[1].h = (init_bw.size() > 2) ? init_bw.at(2) : 1;
    for (int i = 2; i < thresh.size() + 1; i++) {
      bin_w[i].r = (size_t)std::ceil((double)bin_w[i - 1].r / bw_ratio[i - 2]);
      bin_w[i].c = (size_t)std::ceil((double)bin_w[i - 1].c / bw_ratio[i - 2]);
      bin_w[i].h = (size_t)std::ceil((double)bin_w[i - 1].h / bw_ratio[i - 2]);
    }

    // depth first search for hierachical block refinement
    if ((thresh.size() == 1) && (bin_w[1].r * bin_w[1].c * bin_w[1].h == 1)) {
      amr_gb_bw1<N, Real, size_t>(unshuffled_u, c_hierarchy, thresh.at(0),
                                  bin_w, u_map);
    } else {
      amr_gb<N, Real, size_t>(unshuffled_u.data(), c_hierarchy, thresh, bin_w,
                              u_map);
      //        amr_gb_recursion<N, Real, size_t>(unshuffled_u.data(),
      //        c_hierarchy, thresh, bin_w, u_map);
    }
    if (filename != NULL) {
      FILE *fp = fopen(filename, "wb");
      fwrite(u_map, sizeof(Real), ndof, fp);
      fclose(fp);
    }
  }
  size_t cnt_bz = 0, cnt_roi = 0;
  for (size_t i = 0; i < ndof; i++) {
    if (u_map[i] == BUFFER_ZONE)
      cnt_bz++;
    else if (u_map[i] == ROI)
      cnt_roi++;
  }
  // std::cout << "percentage of bz " << (float)cnt_bz / (float)ndof * 100.0
  //           << "% out of " << (float)cnt_roi / (float)ndof * 100.0 << "%
  //           roi\n";
  shuffle(hierarchy, u_map, unshuffled_u.data());

  // QG: scalar of eb used for coefficients in non-RoI : RoI
  // 2D case is bounded by the horizontal direction of coefficient_nodal error
  // propagation 3rd peak, scalar=125 for 2D if nodal nodal and scalar=50 if
  // nodal coefficient vertical
  //  1D case: scalar = 22 for d=2h  (0.634 * w^2), w=0.268
  //  2D case: scalar = 23 for d=2h  (0.5915 * w^2)
  size_t scalar = (N == 3) ? 25 : 23; // 20

  MemoryBuffer<unsigned char> quantized = quantization_buffer(header, ndof);
  quantize_roi(hierarchy, header, s, tolerance, scalar, unshuffled_u.data(), u,
               quantized.data.get());
  MemoryBuffer<unsigned char> buffer =
      compress(header, quantized.data.get(), quantized.size);
  delete[] u;
  return CompressedDataset<N, Real>(hierarchy, header, s, tolerance,
                                    buffer.data.release(), buffer.size);
}

} // namespace mgard

#endif
