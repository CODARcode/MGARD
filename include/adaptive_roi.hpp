#ifndef ADAPTIVE_ROI
#define ADAPTIVE_ROI

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mgard {

#define BUFFER_ZONE 125
#define BACKGROUND 255
#define ROI 0

template <typename T> struct customized_hierarchy {
  T *level;
  T L;
  T Row;
  T Col;
  T Height;
  T l_th;
};

template <typename T> struct cube_ {
  T r;
  T c;
  T h;
};

template <typename T>
void hist_blc_coord(std::vector<cube_<T>> &blc_set /*top left coord*/,
                    cube_<T> nbin, cube_<T> bin_w, cube_<T> init);

template <typename T> std::vector<size_t> sort_indexes(const std::vector<T> &v);

template <typename T1, typename T2>
void filter_hist_blc(const T1 *u_mc, customized_hierarchy<T2> &c_hierarchy,
                     std::vector<cube_<int>> &blc_set,
                     std::vector<cube_<int>> &filtered_set, const T1 thresh,
                     cube_<T2> bin_w, size_t &nbins);

template <typename T>
size_t blc_coord_gb(std::vector<cube_<int>> &c_blc,
                    std::vector<cube_<int>> p_blc, size_t nblc,
                    std::vector<cube_<T>> bin_w, size_t depth);

template <typename T1, typename T2>
bool buffer_zone(customized_hierarchy<T2> &c_hierarchy, T1 *u_map,
                 struct cube_<int> pos, const int rad);

template <size_t N, typename T1, typename T2>
void amr_gb(const T1 *u_mc, customized_hierarchy<T2> &c_hierarchy,
            const std::vector<T1> thresh, std::vector<cube_<T2>> bin_w,
            T1 *u_map);

template <size_t N, typename T1, typename T2>
void amr_gb_bw1(std::vector<T1> u_mc, customized_hierarchy<T2> &c_hierarchy,
                const T1 thresh, const std::vector<cube_<T2>> bin_w, T1 *u_map);

template <typename T1, typename T2>
void set_buffer_zone_bw1_3d(customized_hierarchy<T2> &c_hierarchy, T1 *u_map,
                            struct cube_<int> roi_min,
                            struct cube_<int> roi_max, struct cube_<int> bz_min,
                            struct cube_<int> bz_max, const int lr,
                            const int rad);

template <typename T1, typename T2>
void set_buffer_zone_bw1_2d(customized_hierarchy<T2> &c_hierarchy, T1 *u_map,
                            struct cube_<int> roi_min,
                            struct cube_<int> roi_max, struct cube_<int> bz_min,
                            struct cube_<int> bz_max, const int lr,
                            const int rad);

template <size_t N, typename T1, typename T2>
void set_buffer_zone(customized_hierarchy<T2> &c_hierarchy, T1 *u_map,
                     struct cube_<int> roi_min, struct cube_<int> roi_max,
                     struct cube_<int> bz_min, struct cube_<int> bz_max,
                     const int lr);

} // namespace mgard
#include "adaptive_roi.tpp"
#endif
