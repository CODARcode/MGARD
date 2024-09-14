#include "adaptive_roi.hpp"
#include <algorithm>
#include <cmath>
#include <time.h> /* clock_t, clock, CLOCKS_PER_SEC */

namespace mgard {

template <typename T>
void hist_blc_coord(std::vector<cube_<T>> &blc_set /*top left coord*/,
                    cube_<T> nbin, cube_<T> bin_w, cube_<T> init) {
  size_t k, r, c, h, dim2;
  dim2 = nbin.c * nbin.h;
  for (r = 0; r < nbin.r; r++) {
    for (c = 0; c < nbin.c; c++) {
      for (h = 0; h < nbin.h; h++) {
        k = r * dim2 + c * nbin.h + h;
        blc_set.at(k).r = init.r + r * bin_w.r;
        blc_set.at(k).c = init.c + c * bin_w.c;
        blc_set.at(k).h = init.h + h * bin_w.h;
      }
    }
  }
}

template <typename T>
size_t blc_coord_gb(std::vector<cube_<int>> &c_blc,
                    std::vector<cube_<int>> p_blc, size_t nblc,
                    std::vector<cube_<T>> bin_w, size_t depth) {
  size_t k = 0, r, c, h;
  struct cube_<size_t> nbin;
  T row = bin_w[0].r;
  T col = bin_w[0].c;
  T hig = bin_w[0].h;
  struct cube_<T> prev_bw = {
    bin_w[depth - 1].r, bin_w[depth - 1].c, bin_w[depth - 1].h
  };
  struct cube_<T> curr_bw = {
    bin_w[depth].r, bin_w[depth].c, bin_w[depth].h
  };
  for (size_t i = 0; i < nblc; i++) {
    nbin.r = (p_blc[i].r + prev_bw.r > row)
                 ? (size_t)std::ceil(((float)(row - p_blc[i].r)) / curr_bw.r)
                 : (size_t)std::ceil(((float)prev_bw.r) / curr_bw.r);
    nbin.c = (p_blc[i].c + prev_bw.c > col)
                 ? (size_t)std::ceil(((float)(col - p_blc[i].c)) / curr_bw.c)
                 : (size_t)std::ceil(((float)prev_bw.c) / curr_bw.c);
    nbin.h = (p_blc[i].h + prev_bw.h > hig)
                 ? (size_t)std::ceil(((float)(hig - p_blc[i].h)) / curr_bw.h)
                 : (size_t)std::ceil(((float)prev_bw.h) / curr_bw.h);
    for (r = 0; r < nbin.r; r++) {
      for (c = 0; c < nbin.c; c++) {
        for (h = 0; h < nbin.h; h++) {
          c_blc.at(k).r = p_blc[i].r + r * curr_bw.r;
          c_blc.at(k).c = p_blc[i].c + c * curr_bw.c;
          c_blc.at(k).h = p_blc[i].h + h * curr_bw.h;
          k++;
        }
      }
    }
  }
  return k;
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  std::stable_sort(idx.begin(), idx.end(),
                   [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

  return idx;
}

#define Alpha 1.414

template <typename T1> struct KVal {
  T1 key;
  size_t val;
};

template <typename T1> struct greater1 {
  bool operator()(const struct KVal<T1> &a, const struct KVal<T1> &b) const {
    return a.key < b.key;
  }
};

// return the id of blocks whose coefficients sum are large and expand the block
// boundary
template <typename T1, typename T2>
void filter_hist_blc(const T1 *u_mc, customized_hierarchy<T2> &c_hierarchy,
                     std::vector<cube_<int>> &blc_set,
                     std::vector<cube_<int>> &filtered_set, const T1 thresh,
                     cube_<T2> bin_w, size_t &nbins) {
  //  std::cout << "number of bins: " << nbins << "\n";
  std::vector<struct KVal<T1>> hist_w(nbins);
  size_t dim2 = c_hierarchy.Col * c_hierarchy.Height;
  size_t k;
  T1 area, r1, c1, h1;
  for (size_t i = 0; i < nbins; i++) {
    r1 = blc_set.at(i).r + bin_w.r;
    area = bin_w.r;
    if (r1 > c_hierarchy.Row) {
      area = r1 - c_hierarchy.Row;
      r1 = c_hierarchy.Row;
    }
    c1 = blc_set.at(i).c + bin_w.c;
    if (c1 > c_hierarchy.Col) {
      area = area * (c1 - c_hierarchy.Col);
      c1 = c_hierarchy.Col;
    } else {
      area = area * bin_w.c;
    }
    h1 = blc_set.at(i).h + bin_w.h;
    if (h1 > c_hierarchy.Height) {
      area = area * (h1 - c_hierarchy.Height);
      h1 = c_hierarchy.Height;
    } else {
      area = area * bin_w.h;
    }
    hist_w[i].key = 0;
    hist_w[i].val = i;
    for (T2 r = blc_set.at(i).r; r < r1; r++) {
      T2 r0 = r * dim2;
      for (T2 c = blc_set.at(i).c; c < c1; c++) {
        T2 c0 = c * c_hierarchy.Height;
        for (T2 h = blc_set.at(i).h; h < h1; h++) {
          k = r0 + c0 + h;
          T2 l = c_hierarchy.level[k];
          if (l > 0) {
            hist_w[i].key += std::abs(u_mc[k]); // / factor;
          }
          //                    if (l<c_hierarchy.l_th)
          //                        hist_w[i].key += std::abs(u_mc[k]) /
          //                        std::pow(Alpha, c_hierarchy.L-l);
        }
      }
    }
    // normalize to remove the edge effect
    hist_w[i].key = hist_w[i].key / area;
  }

  nbins = (size_t)std::ceil(thresh * nbins);
  std::make_heap(hist_w.begin(), hist_w.end(), greater1<T1>());
  for (size_t i = 0; i < nbins; i++) {
    filtered_set.at(i) = blc_set.at(hist_w.front().val);
    pop_heap(hist_w.begin(), hist_w.end(), greater1<T1>());
    hist_w.pop_back();
  }
  //    std::vector<size_t> sid_umc = sort_indexes<T1>(hist_w);
  //    for (int i=0; i<nbins; i++) {
  //        filtered_set.at(i) = blc_set.at(sid_umc.at(i));
  //    }
}

template <size_t N, typename T1, typename T2>
void set_buffer_zone(customized_hierarchy<T2> &c_hierarchy, T1 *u_map,
                     struct cube_<int> roi_min, struct cube_<int> roi_max,
                     struct cube_<int> bz_min, struct cube_<int> bz_max,
                     const int lr) {
  size_t l, k, delta, lth;
  lth = c_hierarchy.L - lr;
  // separate the buffer zone memset into 4 (2D) or 6 (3D) regions
  if (N == 2) {
    // top region
    delta = bz_max.c - bz_min.c;
    for (int r = bz_min.r; r < roi_min.r; r++) {
      k = r * c_hierarchy.Col + bz_min.c;
      if (lr == 0) { // no need to check the level of the BG points
        memset(&u_map[k], 0, sizeof(T1) * delta);
      } else {
        for (size_t d = k; d < k + delta; d++) {
          l = c_hierarchy.level[d];
          if (l <= lth) {
            u_map[d] = 0;
          }
        }
      }
    }
    // bottom region
    for (int r = roi_max.r; r < bz_max.r; r++) {
      k = r * c_hierarchy.Col + bz_min.c;
      if (lr == 0) { // no need to check the level of the BG points
        memset(&u_map[k], 0, sizeof(T1) * delta);
      } else {
        for (size_t d = k; d < k + delta; d++) {
          l = c_hierarchy.level[d];
          if (l <= lth) {
            u_map[d] = 0;
          }
        }
      }
    }
    // left region
    delta = roi_min.c - bz_min.c;
    for (int r = roi_min.r; r < roi_max.r; r++) {
      k = r * c_hierarchy.Col + bz_min.c;
      if (lr == 0) {
        memset(&u_map[k], 0, sizeof(T1) * delta);
      } else {
        for (size_t d = k; d < k + delta; d++) {
          l = c_hierarchy.level[d];
          if (l <= lth) {
            u_map[d] = 0;
          }
        }
      }
    }
    // right region
    delta = bz_max.c - roi_max.c;
    for (int r = roi_min.r; r < roi_max.r; r++) {
      k = r * c_hierarchy.Col + roi_max.c;
      if (lr == 0) {
        memset(&u_map[k], 0, sizeof(T1) * delta);
      } else {
        for (size_t d = k; d < k + delta; d++) {
          l = c_hierarchy.level[d];
          if (l <= lth) {
            u_map[d] = 0;
          }
        }
      }
    }
  } else if (N == 3) {
    // top region
    delta = bz_max.h - bz_min.h;
    size_t dim2 = c_hierarchy.Col * c_hierarchy.Height;
    size_t rr;
    size_t thresh = c_hierarchy.Height * c_hierarchy.Col * c_hierarchy.Row;
    for (int r = bz_min.r; r < roi_min.r; r++) {
      rr = r * dim2;
      for (int c = bz_min.c; c < bz_max.c; c++) {
        k = rr + c * c_hierarchy.Height + bz_min.h;
        //                if ((k>thresh) || (delta>thresh))
        //                    std::cout << "invalid top...roi.r: " << roi_max.r
        //                    << ", " << roi_min.r << ", roi.c: " << roi_max.c
        //                    << ", " << roi_min.c << ", roi.h: " << roi_max.h
        //                    << ", " << roi_min.h << ", bz.r: " << bz_max.r <<
        //                    ", " << bz_min.r << ", bz.c: " << bz_max.c << ", "
        //                    << bz_min.c << ", bz.h: " << bz_max.h << ", " <<
        //                    bz_min.h << ", lr: " << "\n";
        if (lr == 0) {
          memset(&u_map[k], 0, sizeof(T1) * delta);
        } else {
          T2 *p_l = &c_hierarchy.level[k];
          T1 *p_umap = &u_map[k];
          for (size_t d = 0; d < delta; d++) {
            if ((*p_l) <= lth) {
              *p_umap = 0;
            }
            p_l++;
            p_umap++;
          }

          /*                    size_t d=0;
                              while (d<delta) {
                                  if ((*p_l) <= lth) {
                                      *p_umap = 0;
                                      p_l   +=2;
                                      p_umap+=2;
                                      d += 2;
                                  } else {
                                      p_l++;
                                      p_umap++;
                                      d ++;
                                  }
                              }*/
        }
      }
    }
    // bottom region
    for (int r = roi_max.r; r < bz_max.r; r++) {
      rr = r * dim2;
      for (int c = bz_min.c; c < bz_max.c; c++) {
        k = rr + c * c_hierarchy.Height + bz_min.h;
        if (lr == 0) {
          memset(&u_map[k], 0, sizeof(T1) * delta);
        } else {
          T2 *p_l = &c_hierarchy.level[k];
          T1 *p_umap = &u_map[k];
          for (size_t d = 0; d < delta; d++) {
            if ((*p_l) <= lth) {
              *p_umap = 0;
            }
            p_l++;
            p_umap++;
          }
          /*
                              size_t d=0;
                              while (d<delta) {
                                  if ((*p_l) <= lth) {
                                      *p_umap = 0;
                                      p_l   +=2;
                                      p_umap+=2;
                                      d += 2;
                                  } else {
                                      p_l++;
                                      p_umap++;
                                      d ++;
                                  }
                              }*/
        }
      }
    }
    // left region
    delta = roi_min.h - bz_min.h;
    for (int r = roi_min.r; r < roi_max.r; r++) {
      rr = r * dim2;
      for (int c = bz_min.c; c < bz_max.c; c++) {
        k = rr + c * c_hierarchy.Height + bz_min.h;
        if (lr == 0) {
          memset(&u_map[k], 0, sizeof(T1) * delta);
        } else {
          T2 *p_l = &c_hierarchy.level[k];
          T1 *p_umap = &u_map[k];
          for (size_t d = 0; d < delta; d++) {
            if ((*p_l) <= lth) {
              *p_umap = 0;
            }
            p_l++;
            p_umap++;
          } /*
           size_t d=0;
           while (d<delta) {
               if ((*p_l) <= lth) {
                   *p_umap = 0;
                   p_l   +=2;
                   p_umap+=2;
                   d += 2;
               } else {
                   p_l++;
                   p_umap++;
                   d ++;
               }
           }*/
        }
      }
    }
    // right region
    delta = bz_max.h - roi_max.h;
    for (int r = roi_min.r; r < roi_max.r; r++) {
      rr = r * dim2;
      for (int c = bz_min.c; c < bz_max.c; c++) {
        k = rr + c * c_hierarchy.Height + roi_max.h;
        if (lr == 0) {
          memset(&u_map[k], 0, sizeof(T1) * delta);
        } else {
          T2 *p_l = &c_hierarchy.level[k];
          T1 *p_umap = &u_map[k];
          for (size_t d = 0; d < delta; d++) {
            if ((*p_l) <= lth) {
              *p_umap = 0;
            }
            p_l++;
            p_umap++;
          } /*
           size_t d=0;
           while (d<delta) {
               if ((*p_l) <= lth) {
                   *p_umap = 0;
                   p_l   +=2;
                   p_umap+=2;
                   d += 2;
               } else {
                   p_l++;
                   p_umap++;
                   d ++;
               }
           }*/
        }
      }
    }
    // back region
    delta = roi_max.h - roi_min.h;
    for (int r = roi_min.r; r < roi_max.r; r++) {
      rr = r * dim2;
      for (int c = bz_min.c; c < roi_min.c; c++) {
        k = rr + c * c_hierarchy.Height + roi_min.h;
        if (lr == 0) {
          memset(&u_map[k], 0, sizeof(T1) * delta);
        } else {
          T2 *p_l = &c_hierarchy.level[k];
          T1 *p_umap = &u_map[k];
          for (size_t d = 0; d < delta; d++) {
            if ((*p_l) <= lth) {
              *p_umap = 0;
            }
            p_l++;
            p_umap++;
          } /*
           size_t d=0;
           while (d<delta) {
               if ((*p_l) <= lth) {
                   *p_umap = 0;
                   p_l   +=2;
                   p_umap+=2;
                   d += 2;
               } else {
                   p_l++;
                   p_umap++;
                   d ++;
               }
           }*/
        }
      }
    }
    // front region
    for (int r = roi_min.r; r < roi_max.r; r++) {
      rr = r * dim2;
      for (int c = roi_max.c; c < bz_max.c; c++) {
        k = rr + c * c_hierarchy.Height + roi_min.h;
        if (lr == 0) {
          memset(&u_map[k], 0, sizeof(T1) * delta);
        } else {
          T2 *p_l = &c_hierarchy.level[k];
          T1 *p_umap = &u_map[k];
          for (size_t d = 0; d < delta; d++) {
            if ((*p_l) <= lth) {
              *p_umap = 0;
            }
            p_l++;
            p_umap++;
          } /*
           size_t d=0;
           while (d<delta) {
               if ((*p_l) <= lth) {
                   *p_umap = 0;
                   p_l   +=2;
                   p_umap+=2;
                   d += 2;
               } else {
                   p_l++;
                   p_umap++;
                   d ++;
               }
           }*/
        }
      }
    }
  }
}

template <typename T1, typename T2>
void set_buffer_zone_bw1_2d(customized_hierarchy<T2> &c_hierarchy, T1 *u_map,
                            struct cube_<int> roi_min,
                            struct cube_<int> roi_max, struct cube_<int> bz_min,
                            struct cube_<int> bz_max, const int lr,
                            const int rad) {
  size_t l, k, d, delta, lth;
  lth = c_hierarchy.L - lr;
  // separate the buffer zone memset into 4 (2D) or 6 (3D) regions
  if (lr == 0) {
    delta = 1;
    d = roi_min.c;
    for (int r = bz_min.r; r <= roi_min.r; r++) {
      k = r * c_hierarchy.Col + d;
      memset(&u_map[k], 0, sizeof(T1) * delta);
      d--;
      delta += 2;
    } // d = bz_min.c; delta = 2*rad+1
    for (int r = roi_min.r + 1; r < bz_max.r; r++) {
      k = r * c_hierarchy.Col + d;
      memset(&u_map[k], 0, sizeof(T1) * delta);
      d++;
      delta -= 2;
    }
  } else {
    // top & bottom region
    int c1 = roi_min.c + rad;
    ;
    int c2 = c1 + 1;
    int rtop = bz_min.r;
    int rbot = bz_max.r - 1;
    size_t k1, k2;
    while (rtop <= roi_min.r) {
      k1 = rtop * c_hierarchy.Col + c1;
      k2 = rbot * c_hierarchy.Col + c1;
      for (int c = c1; c < c2; c++) {
        if (c_hierarchy.level[k1] <= lth)
          u_map[k1] = 0;
        k1++;
        if (c_hierarchy.level[k2] <= lth)
          u_map[k2] = 0;
        k2++;
      }
      c1--;
      c2++;
      rtop++;
      rbot--;
    }
    // middle top region
    c1 = roi_min.c - 1;
    c2 = roi_min.c + rad + 1;
    for (int r = roi_min.r + 1; r < roi_min.r + rad; r++) {
      k = r * c_hierarchy.Col;
      k1 = k + c1;
      k2 = k + c2;
      for (d = k1; d < k1 + rad; d++) {
        if (c_hierarchy.level[d] <= lth)
          u_map[d] = 0;
      }
      for (d = k2; d < k2 + rad; d++) {
        if (c_hierarchy.level[d] <= lth)
          u_map[d] = 0;
      }
      c1--;
      c2++;
    }
    // middle bottom region
    c1 = bz_min.c;
    c2 = roi_max.c;
    for (int r = roi_min.r + rad; r < roi_max.r; r++) {
      k = r * c_hierarchy.Col;
      k1 = k + c1;
      k2 = k + c2;
      for (d = k1; d < k1 + rad; d++) {
        if (c_hierarchy.level[d] <= lth)
          u_map[d] = 0;
      }
      for (d = k2; d < k2 + rad; d++) {
        if (c_hierarchy.level[d] <= lth)
          u_map[d] = 0;
      }
      c1++;
      c2--;
    }
  }
}

template <typename T1, typename T2>
void set_buffer_zone_bw1_3d(customized_hierarchy<T2> &c_hierarchy, T1 *u_map,
                            struct cube_<int> roi_min,
                            struct cube_<int> roi_max, struct cube_<int> bz_min,
                            struct cube_<int> bz_max, const int lr,
                            const int rad) {
  size_t l, k, delta, lth, rr;
  lth = c_hierarchy.L - lr;
  size_t dim2 = c_hierarchy.Col * c_hierarchy.Height;
  if (lr == 0) {
    delta = 1;
    int h = roi_min.h;
    for (int r = bz_min.r; r < roi_min.r; r++) {
      rr = r * dim2;
      delta += 2;
      h--;
      for (int c = bz_min.c; c < bz_max.c; c++) {
        k = rr + c * c_hierarchy.Height + h;
        memset(&u_map[k], 0, sizeof(T1) * delta);
      }
      //            std::cout << "delta = " << delta << "\n";
    } // h = bz_max.h
    rr = roi_min.r * dim2;
    for (int c = bz_min.c; c < bz_max.c; c++) {
      k = rr + c * c_hierarchy.Height + h;
      memset(&u_map[k], 0, sizeof(T1) * delta);
    }
    for (int r = roi_min.r + 1; r < bz_max.r; r++) {
      rr = r * dim2;
      for (int c = bz_min.c; c < bz_max.c; c++) {
        k = rr + c * c_hierarchy.Height + h;
        memset(&u_map[k], 0, sizeof(T1) * delta);
      }
      delta -= 2;
      h++;
    }
    /*        delta = bz_max.h - bz_min.h;
            for (int r=bz_min.r; r<bz_max.r; r++) {
                rr = r*dim2;
                k = rr+bz_min.c*c_hierarchy.Height + bz_min.h;
                for (int c=bz_min.c; c<bz_max.c; c++) {
                    memset(&u_map[k], 0, sizeof(T1)*delta);
                    k += c_hierarchy.Height;
                }
            }*/
  } else {
    // top region
    delta = bz_max.h - bz_min.h;
    size_t dim2 = c_hierarchy.Col * c_hierarchy.Height;
    size_t rr;
    for (int r = bz_min.r; r < roi_min.r; r++) {
      rr = r * dim2;
      for (int c = bz_min.c; c < bz_max.c; c++) {
        k = rr + c * c_hierarchy.Height + bz_min.h;
        T2 *p_l = &c_hierarchy.level[k];
        T1 *p_umap = &u_map[k];
        for (size_t d = 0; d < delta; d++) {
          if ((*p_l) <= lth) {
            *p_umap = 0;
          }
          p_l++;
          p_umap++;
        }
      }
    }
    // bottom region
    for (int r = roi_max.r; r < bz_max.r; r++) {
      rr = r * dim2;
      for (int c = bz_min.c; c < bz_max.c; c++) {
        k = rr + c * c_hierarchy.Height + bz_min.h;
        T2 *p_l = &c_hierarchy.level[k];
        T1 *p_umap = &u_map[k];
        for (size_t d = 0; d < delta; d++) {
          if ((*p_l) <= lth) {
            *p_umap = 0;
          }
          p_l++;
          p_umap++;
        }
      }
    }
    // left region
    delta = roi_min.h - bz_min.h;
    for (int r = roi_min.r; r < roi_max.r; r++) {
      rr = r * dim2;
      for (int c = bz_min.c; c < bz_max.c; c++) {
        k = rr + c * c_hierarchy.Height + bz_min.h;
        T2 *p_l = &c_hierarchy.level[k];
        T1 *p_umap = &u_map[k];
        for (size_t d = 0; d < delta; d++) {
          if ((*p_l) <= lth) {
            *p_umap = 0;
          }
          p_l++;
          p_umap++;
        }
      }
    }
    // right region
    delta = bz_max.h - roi_max.h;
    for (int r = roi_min.r; r < roi_max.r; r++) {
      rr = r * dim2;
      for (int c = bz_min.c; c < bz_max.c; c++) {
        k = rr + c * c_hierarchy.Height + roi_max.h;
        T2 *p_l = &c_hierarchy.level[k];
        T1 *p_umap = &u_map[k];
        for (size_t d = 0; d < delta; d++) {
          if ((*p_l) <= lth) {
            *p_umap = 0;
          }
          p_l++;
          p_umap++;
        }
      }
    }
    // back region
    delta = roi_max.h - roi_min.h;
    for (int r = roi_min.r; r < roi_max.r; r++) {
      rr = r * dim2;
      for (int c = bz_min.c; c < roi_min.c; c++) {
        k = rr + c * c_hierarchy.Height + roi_min.h;
        T2 *p_l = &c_hierarchy.level[k];
        T1 *p_umap = &u_map[k];
        for (size_t d = 0; d < delta; d++) {
          if ((*p_l) <= lth) {
            *p_umap = 0;
          }
          p_l++;
          p_umap++;
        }
      }
    }
    // front region
    for (int r = roi_min.r; r < roi_max.r; r++) {
      rr = r * dim2;
      for (int c = roi_max.c; c < bz_max.c; c++) {
        k = rr + c * c_hierarchy.Height + roi_min.h;
        T2 *p_l = &c_hierarchy.level[k];
        T1 *p_umap = &u_map[k];
        for (size_t d = 0; d < delta; d++) {
          if ((*p_l) <= lth) {
            *p_umap = 0;
          }
          p_l++;
          p_umap++;
        }
      }
    }
  }
  /*
          // top region
          size_t dim2 = c_hierarchy.Col * c_hierarchy.Height;
          size_t rr;
          int h = roi_min.h;
          delta = roi_max.h - roi_min.h;
          for (int r=bz_min.r; r<roi_min.r; r++) {
              rr = r*dim2;
              delta += 2;
              h --;
              for (int c=bz_min.c; c<bz_max.c; c++) {
                  k = rr+c*c_hierarchy.Height + h;
                  T2 *p_l = &c_hierarchy.level[k];
                  T1 *p_umap = &u_map[k];
                  for (size_t d=0; d<delta; d++) {
                      if ((*p_l) <= lth) {
                          *p_umap = 0;
                      }
                      p_l++;
                      p_umap++;
                  }
              }
          }
  //        std::cout << "after top, h = " << h << ", bz_min.h = " << bz_min.h
  << ", delta = " << delta << "\n";
          // bottom region
          for (int r=roi_max.r+1; r<bz_max.r; r++) {
              rr = r*dim2;
              for (int c=bz_min.c; c<bz_max.c; c++) {
                  k = rr+c*c_hierarchy.Height + h;
                  T2 *p_l = &c_hierarchy.level[k];
                  T1 *p_umap = &u_map[k];
                  for (size_t d=0; d<delta; d++) {
                      if ((*p_l) <= lth) {
                          *p_umap = 0;
                      }
                      p_l++;
                      p_umap++;
                  }
              }
              delta -= 2;
              h ++;
          }
          // left region
          delta = roi_min.h - bz_min.h;
          h = bz_min.h;
          for (int r=roi_min.r; r<=roi_max.r; r++) {
              rr = r*dim2;
              for (int c=bz_min.c; c<bz_max.c; c++) {
                  k = rr+c*c_hierarchy.Height + h;
                  T2 *p_l = &c_hierarchy.level[k];
                  T1 *p_umap = &u_map[k];
                  for (size_t d=0; d<delta; d++) {
                      if ((*p_l) <= lth) {
                          *p_umap = 0;
                      }
                      p_l++;
                      p_umap++;
                  }
              }
          }
          // right region
          delta = bz_max.h - roi_max.h;
          h = roi_max.h;
          for (int r=roi_min.r; r<=roi_max.r; r++) {
              rr = r*dim2;
              for (int c=bz_min.c; c<bz_max.c; c++) {
                  k = rr+c*c_hierarchy.Height+h;
                  T2 *p_l = &c_hierarchy.level[k];
                  T1 *p_umap = &u_map[k];
                  for (size_t d=0; d<delta; d++) {
                      if ((*p_l) <= lth) {
                          *p_umap = 0;
                      }
                      p_l++;
                      p_umap++;
                  }
              }
          }
          // back region
          delta = roi_max.h - roi_min.h;;
          h = roi_min.h;
          for (int r=roi_min.r; r<roi_max.r; r++) {
              rr = r*dim2;
              delta += 2;
              h --;
              for (int c=bz_min.c; c<roi_min.c; c++) {
                  k = rr+c*c_hierarchy.Height+h;
                  T2 *p_l = &c_hierarchy.level[k];
                  T1 *p_umap = &u_map[k];
                  for (size_t d=0; d<delta; d++) {
                      if ((*p_l) <= lth) {
                          *p_umap = 0;
                      }
                      p_l++;
                      p_umap++;
                  }
              }
          }
          // front region
          for (int r=roi_min.r; r<roi_max.r; r++) {
              rr = r*dim2;
              delta -= 2;
              h++;
              for (int c=roi_max.c; c<bz_max.c; c++) {
                  k = rr+c*c_hierarchy.Height+h;
                  T2 *p_l = &c_hierarchy.level[k];
                  T1 *p_umap = &u_map[k];
                  for (size_t d=0; d<delta; d++) {
                      if ((*p_l) <= lth) {
                          *p_umap = 0;
                      }
                      p_l++;
                      p_umap++;
                  }
              }
          }
      }*/
}

// bin_w: [{Row, Col, Height}, ...], size == depth+1
//        minimally bin_w will have two levels
// depth: thresh.size()
template <size_t N, typename T1, typename T2>
void amr_gb(const T1 *u_mc, customized_hierarchy<T2> &c_hierarchy,
            const std::vector<T1> thresh, const std::vector<cube_<T2>> bin_w,
            T1 *u_map) {
  clock_t start, end;
  float duration_bz = 0.0;
  float factor_ = 2;
  size_t depth = thresh.size();
  // initialize parent, children blocks (maximal #)
  size_t nr = (size_t)std::ceil((double)bin_w[0].r / bin_w[depth].r);
  size_t nc = (size_t)std::ceil((double)bin_w[0].c / bin_w[depth].c);
  size_t nh = (size_t)std::ceil((double)bin_w[0].h / bin_w[depth].h);
  std::vector<struct cube_<int>> child_blc(nr * nc * nh);
  std::vector<struct cube_<int>> parent_blc(nr * nc * nh);

  // mesh refinement
  size_t n_pblc = 1;
  struct cube_<int> blc_min, blc_max, epa_min, epa_max, bz_min, bz_max;
  start = clock();
  for (size_t d = 0; d < depth; d++) {
    n_pblc = blc_coord_gb<T2>(child_blc, parent_blc, n_pblc, bin_w, d + 1);
    filter_hist_blc<T1, T2>(u_mc, c_hierarchy, child_blc, parent_blc, thresh[d],
                            bin_w[d + 1], n_pblc);
  } // parent_blc contains the filtered blocks
  end = clock();
  duration_bz = ((float)(end - start)) / CLOCKS_PER_SEC;
  // std::cout << "AMR takes " << duration_bz << " seconds\n";
  //    std::cout << "final bins..." << n_pblc << "\n";

  // RoI expansion
  size_t dim2 = (size_t)c_hierarchy.Col * c_hierarchy.Height;
  size_t r0, c0, r00, c00, k, l, epa_h;
  int rad, rr, cc, hh;

  //    std::cout << "number of roi blocks: " << n_pblc << "\n";
  size_t nest_r = c_hierarchy.L - c_hierarchy.l_th + 1;
  for (size_t i = 0; i < n_pblc; i++) {
    blc_min = parent_blc[i];
    blc_max.r = blc_min.r + bin_w[depth].r;
    blc_max.c = blc_min.c + bin_w[depth].c;
    blc_max.h = blc_min.h + bin_w[depth].h;
    // extend the RoI blocks by max_R
    /*        rad = (double)(1<<(c_hierarchy.L-c_hierarchy.l_th));
            epa_min.r = (blc_min.r-rad > 0) ? (blc_min.r-rad) : 0;
            epa_max.r = (blc_max.r+rad+1<c_hierarchy.Row) ? (blc_max.r+rad+1) :
       c_hierarchy.Row; epa_min.c = (blc_min.c-rad>0) ? (blc_min.c-rad) : 0;
            epa_max.c = (blc_max.c+rad+1<c_hierarchy.Col) ? (blc_max.c+rad+1) :
       c_hierarchy.Col; epa_min.h = (blc_min.h-rad>0) ? (blc_min.h-rad) : 0;
            epa_max.h = (blc_max.h+rad+1<c_hierarchy.Height) ? (blc_max.h+rad+1)
       : c_hierarchy.Height; epa_h     = (epa_max.h - epa_min.h) * sizeof(T1);
            for (rr=epa_min.r; rr<epa_max.r; rr++) {
                r00 = rr*dim2;
                for (cc=epa_min.c; cc<epa_max.c; cc++) {
                    c00 = cc*c_hierarchy.Height;
                    memset(&u_map[r00+c00+epa_min.h], 0, epa_h);
                }
            }*/
    // non-extension
    epa_min = blc_min;
    epa_max = blc_max;
    // buffer zone search
    //        start =clock();
    rad = 0;
    for (int lr = 0; lr < nest_r; lr++) {
      rad = (int)(factor_ * (double)(1 << (lr + 1)) - rad);
      bz_min.r = (epa_min.r - rad > 0) ? (epa_min.r - rad) : 0;
      bz_max.r = (epa_max.r + rad + 1 < c_hierarchy.Row) ? (epa_max.r + rad + 1)
                                                         : c_hierarchy.Row;
      bz_min.c = (epa_min.c - rad > 0) ? (epa_min.c - rad) : 0;
      bz_max.c = (epa_max.c + rad + 1 < c_hierarchy.Col) ? (epa_max.c + rad + 1)
                                                         : c_hierarchy.Col;
      bz_min.h = (epa_min.h - rad > 0) ? (epa_min.h - rad) : 0;
      bz_max.h = (epa_max.h + rad + 1 < c_hierarchy.Height)
                     ? (epa_max.h + rad + 1)
                     : c_hierarchy.Height;
      set_buffer_zone<N, T1, T2>(c_hierarchy, u_map, epa_min, epa_max, bz_min,
                                 bz_max, lr);
      epa_min = bz_min;
      epa_max = bz_max;
    }
    //        end = clock();
    //        duration_bz += ((float)(end-start))/CLOCKS_PER_SEC;
  }
  //    std::cout << "buffer zone search takes " << duration_bz << " seconds\n";
}

template <size_t N, typename T1, typename T2>
void amr_gb_bw1(std::vector<T1> u_mc, customized_hierarchy<T2> &c_hierarchy,
                const T1 thresh, const std::vector<cube_<T2>> bin_w,
                T1 *u_map) {
  float factor_ = 2;
  size_t n_pblc = bin_w[0].r * bin_w[0].c * bin_w[0].h;
  // initialize children blocks (maximal #)
  std::vector<struct cube_<int>> amr_blc(n_pblc);

  // mesh refinement
  clock_t start, end;
  float duration_bz = 0.0;
  struct cube_<int> blc_min, blc_max, bz_min, bz_max;
  //    start = clock();
  //    std::vector<size_t> sid_umc = sort_indexes<T1>(u_mc);
  std::vector<struct KVal<T1>> hist_w(n_pblc);
  for (size_t i = 0; i < n_pblc; i++) {
    hist_w[i].val = i;
    hist_w[i].key = u_mc[i];
  }
  std::make_heap(hist_w.begin(), hist_w.end(), greater1<T1>());
  n_pblc = (size_t)std::ceil(thresh * n_pblc);
  //    std::cout << "n_pblc: " << n_pblc << "\n";
  std::vector<size_t> sid_umc(n_pblc);
  for (size_t i = 0; i < n_pblc; i++) {
    sid_umc.at(i) = hist_w.front().val;
    //        std::cout << sid_umc.at(i) << "\n";
    pop_heap(hist_w.begin(), hist_w.end(), greater1<T1>());
    hist_w.pop_back();
  }
  //    end = clock();
  hist_w.clear();
  //    duration_bz += ((float)(end-start))/CLOCKS_PER_SEC;
  //    std::cout << "sorting takes " << duration_bz << " seconds\n";
  size_t r, c, h, k;
  size_t dim2 = (size_t)c_hierarchy.Col * c_hierarchy.Height;
  for (int i = 0; i < n_pblc; i++) {
    k = sid_umc.at(i);
    amr_blc.at(i).r = (int)std::floor((float)k / (float)dim2);
    amr_blc.at(i).c = (int)std::floor((float)(k - dim2 * amr_blc.at(i).r) /
                                      (float)c_hierarchy.Height);
    amr_blc.at(i).h =
        k - amr_blc.at(i).r * dim2 - amr_blc.at(i).c * c_hierarchy.Height;
  }
  //    std::cout << "final bins..." << n_pblc << "\n";
  sid_umc.clear();

  // non RoI expansion
  size_t r0, c0, r00, c00, l, epa_h;
  int rad, rr, cc, hh;
  size_t nest_r = c_hierarchy.L - c_hierarchy.l_th + 1;
  //    size_t l7_cnt = 0;
  for (size_t i = 0; i < n_pblc; i++) {
    blc_min = amr_blc[i];
    blc_max.r = (amr_blc[i].r + 1 < c_hierarchy.Row) ? amr_blc[i].r + 1
                                                     : c_hierarchy.Row;
    blc_max.c = (amr_blc[i].c + 1 < c_hierarchy.Col) ? amr_blc[i].c + 1
                                                     : c_hierarchy.Col;
    blc_max.h = (amr_blc[i].h + 1 < c_hierarchy.Height) ? amr_blc[i].h + 1
                                                        : c_hierarchy.Height;
    // buffer zone search
    //        start =clock();
    rad = 0;
    for (int lr = 0; lr < nest_r; lr++) {
      rad = (int)(factor_ * (double)(1 << (lr + 1)) - rad);
      //            std::cout << "rad = " << rad << "\n";
      bool edge = false;
      if (blc_min.r - rad > 0) {
        bz_min.r = blc_min.r - rad;
      } else {
        bz_min.r = 0;
        edge = true;
      }
      if (blc_max.r + rad < c_hierarchy.Row) {
        bz_max.r = blc_max.r + rad;
      } else {
        bz_max.r = c_hierarchy.Row;
        edge = true;
      }
      if (blc_min.c - rad > 0) {
        bz_min.c = blc_min.c - rad;
      } else {
        bz_min.c = 0;
        edge = true;
      }
      if (blc_max.c + rad < c_hierarchy.Col) {
        bz_max.c = blc_max.c + rad;
      } else {
        bz_max.c = c_hierarchy.Col;
        edge = true;
      }
      if (blc_min.h - rad > 0) {
        bz_min.h = blc_min.h - rad;
      } else {
        bz_min.h = 0;
        edge = true;
      }
      if (blc_max.h + rad < c_hierarchy.Height) {
        bz_max.h = blc_max.h + rad;
      } else {
        bz_max.h = c_hierarchy.Height;
        edge = true;
      }
      //            std::cout << "min: " << bz_min.r << "," << bz_min.c << ", "
      //            << bz_min.h << ", max: " << bz_max.r << ", " << bz_max.c <<
      //            ", " << bz_max.h << "\n";
      if (edge) {
        set_buffer_zone<N, T1, T2>(c_hierarchy, u_map, blc_min, blc_max, bz_min,
                                   bz_max, lr);
      } else {
        if (N == 2) {
          set_buffer_zone_bw1_2d<T1, T2>(c_hierarchy, u_map, blc_min, blc_max,
                                         bz_min, bz_max, lr, rad);
        } else if (N == 3) {
          set_buffer_zone_bw1_3d<T1, T2>(c_hierarchy, u_map, blc_min, blc_max,
                                         bz_min, bz_max, lr, rad);
        }
      }
      blc_min = bz_min;
      blc_max = bz_max;
    }
    //        end = clock();
    //        duration_bz += ((float)(end-start))/CLOCKS_PER_SEC;
  }
  //    std::cout << "buffer zone search takes " << duration_bz << " seconds\n";
}

/*
// for each point in RoI, expand buffer zone by its level, then search for
buffer zone template <size_t N, typename T1, typename T2> void amr_gb(const T1
*u_mc, customized_hierarchy <T2> &c_hierarchy, const std::vector<T1> thresh,
            const std::vector<cube_<T2>> bin_w, T1* u_map, const
std::vector<size_t> R2)
{
    float factor_ = (N==3) ? 2.5 : 2;
    size_t depth = thresh.size();
    // initialize parent blocks (maximal #)
    size_t nr = (size_t)std::ceil((double)bin_w[0].r / bin_w[depth].r);
    size_t nc = (size_t)std::ceil((double)bin_w[0].c / bin_w[depth].c);
    size_t nh = (size_t)std::ceil((double)bin_w[0].h / bin_w[depth].h);
    // initialize children blocks (maximal #)
    nr = (size_t)std::ceil((double)bin_w[0].r / bin_w[depth].r);
    nc = (size_t)std::ceil((double)bin_w[0].c / bin_w[depth].c);
    nh = (size_t)std::ceil((double)bin_w[0].h / bin_w[depth].h);
    std::vector<struct cube_<int>> child_blc(nr*nc*nh);
    std::vector<struct cube_<int>> parent_blc(nr*nc*nh);

    // mesh refinement
    size_t n_pblc = 1;
    struct cube_<int> blc_min, blc_max, epa_min, epa_max, bz_min, bz_max;
    for (size_t d=0; d<depth; d++) {
        n_pblc = blc_coord_gb<T2>(child_blc, parent_blc, n_pblc, bin_w, d+1);
        filter_hist_blc<T1, T2>(u_mc, c_hierarchy, child_blc, parent_blc,
thresh[d], bin_w[d+1], n_pblc); } // parent_blc contains the filtered blocks
//    std::cout << "final bins..." << n_pblc << "\n";

    // RoI expansion
    size_t dim2 = (size_t)c_hierarchy.Col * c_hierarchy.Height;
    size_t r0, c0, r00, c00,  k, l, epa_h;
    int rad, rr, cc, hh;
    clock_t start, end;
    float duration_bz=0.0;
    for (size_t i=0; i<n_pblc; i++) {
        blc_min = parent_blc[i];
        blc_max.r = blc_min.r + bin_w[depth].r;
        blc_max.c = blc_min.c + bin_w[depth].c;
        blc_max.h = blc_min.h + bin_w[depth].h;
//        std::cout << "blc_min.r = " << blc_min.r << ", .c = " << blc_min.c <<
", .h = " << blc_min.h << "\n";
//        std::cout << "blc_max.r = " << blc_max.r << ", .c = " << blc_max.c <<
", .h = " << blc_max.h << "\n"; for (int r=blc_min.r; r<blc_max.r; r++) { r0 =
r*dim2; for (int c=blc_min.c; c<blc_max.c; c++) { c0 = c*c_hierarchy.Height; for
(int h=blc_min.h; h<blc_max.h; h++) { k = r0 + c0 + h; l = c_hierarchy.level[k];
                    if (l>=c_hierarchy.l_th) {
                        rad  = static_cast<int>(1<<(c_hierarchy.L - l));
                        epa_min.r = (r-rad > 0) ? (r-rad) : 0;
                        epa_max.r = (r+rad+1<c_hierarchy.Row) ? (r+rad+1) :
c_hierarchy.Row; epa_min.c = (c-rad>0) ? c-rad : 0; epa_max.c =
(c+rad+1<c_hierarchy.Col) ? (c+rad+1) : c_hierarchy.Col; epa_min.h = (h-rad>0) ?
h-rad : 0; epa_max.h = (h+rad+1<c_hierarchy.Height) ? h+rad+1 :
c_hierarchy.Height; epa_h     = epa_max.h - epa_min.h; for (rr=epa_min.r;
rr<epa_max.r; rr++) { r00 = rr*dim2; for (cc=epa_min.c; cc<epa_max.c; cc++) {
                                c00 = cc*c_hierarchy.Height;
                                memset(&u_map[r00+c00+epa_min.h], 0, epa_h *
sizeof(T1));
                            }
                        }
                    } else {
                        epa_min = {r, c, h};
                        epa_max = {r, c, h};
                    }
                    // check for surrounding buffer zone
                    size_t nest_r = c_hierarchy.L - c_hierarchy.l_th + 1;
//                    start =clock();
                    rad = 0;
                    for (int lr=0; lr < nest_r; lr ++) {
                        rad      = (int)(factor_ * (double)(1<<(lr+1)) - rad);
                        bz_min.r = (epa_min.r - rad>0) ? (epa_min.r-rad) : 0;
                        bz_max.r = (epa_max.r+rad+1 < c_hierarchy.Row) ?
(epa_max.r+rad+1) : c_hierarchy.Row; bz_min.c = (epa_min.c - rad>0) ?
(epa_min.c-rad) : 0; bz_max.c = (epa_max.c+rad+1 < c_hierarchy.Col) ?
(epa_max.c+rad+1) : c_hierarchy.Col; bz_min.h = (epa_min.h - rad>0) ?
(epa_min.h-rad) : 0; bz_max.h = (epa_max.h+rad+1 < c_hierarchy.Height) ?
(epa_max.h+rad+1) : c_hierarchy.Height; set_buffer_zone<T1, T2>(c_hierarchy,
u_map, epa_min, epa_max, bz_min, bz_max, lr, N); epa_min = bz_min; epa_max =
bz_max;
                    }
//                    end = clock();
//                    duration_bz += ((float)(end-start))/CLOCKS_PER_SEC;
                }
            }
        }
    }
//    std::cout << "buffer zone search takes " << duration_bz << " seconds\n";
}
*/
} // namespace mgard
