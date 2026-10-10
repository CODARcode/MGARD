#ifndef MGARD_X_MDR_RECONSTRUCT_PIPELINE_QOI_HPP
#define MGARD_X_MDR_RECONSTRUCT_PIPELINE_QOI_HPP

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "mgard-x/Config/Config.h"
#include "mgard-x/MDRHighLevel/MDRDataHighLevel.hpp"
#include "mgard-x/MDRHighLevel/MDRHighLevel.hpp"
#include "mgard-x/MDRHighLevel/MaxAbsIndexKernel.hpp"
#include "mgard-x/MDRHighLevel/QoIKernel.hpp"

namespace mgard_x {
namespace MDR {

// f(x) = x^2
template <typename T> inline double compute_bound_x_square(T x, T eb) {
  return 2 * fabs(x) * eb + eb * eb;
}

// f(x) = sqrt(x)
template <class T> inline double compute_bound_square_root_x(T x, T eb) {
  if (x == 0) {
    return sqrt(eb);
  }
  if (x > eb) {
    return eb / (sqrt(x - eb) + sqrt(x));
  } else {
    return eb / sqrt(x);
  }
}

template <typename T>
inline void error_bound_uniform_decrease(T vx, T vy, T vz, double tau,
                                         double max_error,
                                         std::vector<double> &ebs) {
  double V_TOT_2 = vx * vx + vy * vy + vz * vz;
  double estimate_error = max_error;
  double eb_vx = ebs[0];
  double eb_vy = ebs[1];
  double eb_vz = ebs[2];
  {
    double e_V_TOT_2 = compute_bound_x_square((double)vx, eb_vx) +
                       compute_bound_x_square((double)vy, eb_vy) +
                       compute_bound_x_square((double)vz, eb_vz);
    estimate_error = compute_bound_square_root_x(V_TOT_2, e_V_TOT_2);
    // std::cout << "validation of max error = " << estimate_error << std::endl;
  }
  while (estimate_error > tau) {
    eb_vx = eb_vx / 1.5;
    eb_vy = eb_vy / 1.5;
    eb_vz = eb_vz / 1.5;
    double e_V_TOT_2 = compute_bound_x_square((double)vx, eb_vx) +
                       compute_bound_x_square((double)vy, eb_vy) +
                       compute_bound_x_square((double)vz, eb_vz);
    estimate_error = compute_bound_square_root_x(V_TOT_2, e_V_TOT_2);
  }
  ebs[0] = eb_vx;
  ebs[1] = eb_vy;
  ebs[2] = eb_vz;
  return;
}

template <DIM D, typename T, typename DeviceType, typename ReconstructorType>
void reconstruct_pipeline_qoi(
    DomainDecomposer<D, T, ReconstructorType, DeviceType> &domain_decomposer,
    Config &config, RefactoredMetadata &refactored_metadata,
    RefactoredData &refactored_data, ReconstructedData &reconstructed_data) {
  Timer timer_series, qoi_timer;

  using Cache = ReconstructorCache<D, T, DeviceType, ReconstructorType>;
  using HierarchyType = typename ReconstructorType::HierarchyType;

  ReconstructorType &reconstructor = *Cache::cache.reconstructor;
  Array<D, T, DeviceType> *device_subdomain_buffer =
      Cache::cache.device_subdomain_buffer;
  MDRData<DeviceType> *mdr_data = Cache::cache.mdr_data;

  Array<D, double, DeviceType> error_out(domain_decomposer.subdomain_shape(0));
  Array<1, double, DeviceType> error_final_out({1});
  Array<1, uint32_t, DeviceType> max_index_d({1});
  Array<1, Byte, DeviceType> workspace;

  for (int i = 0; i < 2; i++) {
    error_final_out.resize({1}, i);
    DeviceCollective<DeviceType>::AbsMax(
        domain_decomposer.subdomain_shape(0)[0] *
            domain_decomposer.subdomain_shape(0)[1] *
            domain_decomposer.subdomain_shape(0)[2],
        SubArray<1, double, DeviceType>(), SubArray<1, double, DeviceType>(),
        workspace, false, 0);
  }

  if (config.mdr_qoi_num_variables != domain_decomposer.num_subdomains()) {
    throw std::runtime_error(
        "QOI mode requires the number of variables to be equal to the "
        "number of subdomains");
  }

  log::info("Adjust device buffers");
  if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(0),
                                     domain_decomposer.uniform)) {
    Cache::cache.ClearHierarchyCache();
  }
  for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
    if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(id),
                                       domain_decomposer.uniform)) {
      Cache::cache.InsertHierarchyCache(
          domain_decomposer.subdomain_hierarchy(id));
    }
    mdr_data[id].Resize(reconstructor, refactored_metadata.metadata[id], 0);
    device_subdomain_buffer[id].resize(domain_decomposer.subdomain_shape(id),
                                       0);
    // Reset all signs to 0 for the initial QOI reconstruction
    if (!reconstructed_data.qoi_in_progress) {
      mdr_data[id].ResetSigns(0);
    }
  }

  // Size the reused buffers for the largest subdomain: they are not grown
  // when a later subdomain is bigger than the first.
  std::vector<SIZE> largest_shape =
      domain_decomposer.subdomain_shape(domain_decomposer.largest_subdomain());
  HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(largest_shape);
  reconstructor.Adapt(hierarchy, config, 0);

  int current_buffer = 0;
  int current_queue = 0;

  DeviceRuntime<DeviceType>::SyncDevice();
  timer_series.start();
  // Prefetch the first subdomain
  mdr_data[current_buffer].CopyFromRefactoredData(
      refactored_metadata.metadata[0], refactored_data.data[0], current_queue);

  SIZE total_size = 0;
  uint32_t max_iter = 500;
  uint32_t iter = 0;
  double error_final_out_host;
  int buffer_for_variable[3];
  std::vector<double> ebs(3);
  std::vector<double> last_ebs(3);
  double last_maximal_error;
  double tol = refactored_metadata.metadata[0].tau;

  reconstructed_data.qoi_in_progress = true;

  while ((reconstructed_data.qoi_in_progress) && (iter < max_iter)) {
    iter++;
    // std::cout << "======= Iteration " << iter << " =======" << std::endl;
    for (SIZE curr_subdomain_id = 0;
         curr_subdomain_id < domain_decomposer.num_subdomains();
         curr_subdomain_id++) {

      SIZE next_subdomain_id;
      int next_buffer =
          (current_buffer + 1) % domain_decomposer.num_subdomains();
      int next_queue = (current_queue + 1) % domain_decomposer.num_subdomains();
      HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
          domain_decomposer.subdomain_shape(curr_subdomain_id));
      log::info("Adapt Refactor to hierarchy");
      reconstructor.Adapt(hierarchy, config, current_queue);
      if (iter == 1)
        total_size += hierarchy.total_num_elems() * sizeof(T);
      reconstructor.LoadMetadata(
          refactored_metadata.metadata[curr_subdomain_id],
          mdr_data[current_buffer], current_queue);
      reconstructor.Decompress(refactored_metadata.metadata[curr_subdomain_id],
                               mdr_data[current_buffer], current_queue);
      if (curr_subdomain_id + 1 < domain_decomposer.num_subdomains()) {
        // Prefetch the next subdomain
        next_subdomain_id = curr_subdomain_id + 1;
        mdr_data[next_buffer].CopyFromRefactoredData(
            refactored_metadata.metadata[next_subdomain_id],
            refactored_data.data[next_subdomain_id], next_queue);
      }

      if (curr_subdomain_id == config.mdr_qoi_num_variables - 1) {
        // We are about to finish reconstructing all variables
        // so, we need to fetch more data
        //
        // We need to update the metadata for all variables
        if (refactored_metadata.decrease_method) {
          ebs[0] = refactored_metadata.metadata[0].corresponding_error;
          ebs[1] = refactored_metadata.metadata[1].corresponding_error;
          ebs[2] = refactored_metadata.metadata[2].corresponding_error;
        } else {
          ebs[0] = refactored_metadata.metadata[0].requested_tol;
          ebs[1] = refactored_metadata.metadata[1].requested_tol;
          ebs[2] = refactored_metadata.metadata[2].requested_tol;
        }
        // std::cout << "current ebs : ";
        // for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
        //   if (refactored_metadata.decrease_method) std::cout <<
        //   refactored_metadata.metadata[id].corresponding_error << ", "; else
        //   std::cout << refactored_metadata.metadata[id].requested_tol << ",
        //   ";
        //   // refactored_metadata.metadata[id].requested_size =
        //   usr_def_requested_size; //new tolerance

        // }
        // std::cout << std::endl;

        // for (auto &metadata : refactored_metadata.metadata) {
        //   metadata.PrintStatus();
        // }
        // size_t size_read = read_mdr(refactored_metadata, refactored_data,
        // "/home/linusli037/Polaris/MGARD/build-cuda-turing/mgard/miniNYX/XYZ",
        // false, config); refactored_metadata.total_size += size_read;*****
        // initiate the bitplane transfer for the 1st variable which
        // should coorespond to the next_buffer
        // mdr_data[0].CopyFromRefactoredData(
        //     refactored_metadata.metadata[0],
        //     refactored_data.data[0], next_queue);
      }

      std::stringstream ss;
      for (DIM d = 0; d < D; d++) {
        ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
      }
      log::info("Reconstruct subdomain " + std::to_string(curr_subdomain_id) +
                " with shape: " + ss.str());

      // Reconstruct

      reconstructor.ProgressiveReconstruct(
          refactored_metadata.metadata[curr_subdomain_id],
          mdr_data[current_buffer], config.mdr_adaptive_resolution,
          device_subdomain_buffer[current_buffer], current_queue);

      if (curr_subdomain_id == config.mdr_qoi_num_variables - 1) {

        // DeviceRuntime<DeviceType>::SyncQueue(current_queue);

        // for (int q = 0; q < 2; q++) {
        //   DeviceRuntime<DeviceType>::SyncQueue(q);
        // }

        // We are done with reconstructing all variables now
        // Do error estimation here
        // Var0 can be accessed from device_subdomain_buffer[0].data()
        // Var1 can be accessed from device_subdomain_buffer[1].data()
        // Var2 can be accessed from device_subdomain_buffer[2].data()

        //  if (tol NOT met) {
        //    need to contine reconstructing. Device buffers will NOT be
        //    released reconstructed_data.qoi_in_progress = true;
        //  } else {
        //     will stop reconstructing. Device buffers will be released
        //     reconstructed_data.qoi_in_progress = false;
        //  }
        //  we set it true for testing only

        if (log::level & log::TIME) {
          DeviceRuntime<DeviceType>::SyncQueue(current_queue);
          qoi_timer.start();
        }
        DeviceLauncher<DeviceType>::Execute(
            mgard_x::data_refactoring::multi_dimension::QoIKernel<D, T,
                                                                  DeviceType>(
                SubArray(device_subdomain_buffer[0]),
                SubArray(device_subdomain_buffer[1]),
                SubArray(device_subdomain_buffer[2]), SubArray(error_out),
                ebs[0], ebs[1], ebs[2], tol),
            current_queue);
        SubArray<1, double, DeviceType> out_1d(
            {device_subdomain_buffer[0].shape(0) *
             device_subdomain_buffer[0].shape(1) *
             device_subdomain_buffer[0].shape(2)},
            error_out.data());
        // std::vector<double>
        // out_vec(refactored_metadata.metadata[0].num_elements); std::cout <<
        // "num_elements = " << refactored_metadata.metadata[0].num_elements <<
        // std::endl; std::cout << "out_vec.data() = " << out_vec.data() <<
        // std::endl; MemoryManager<DeviceType>::Copy1D(out_vec.data(),
        // out_1d.data(), refactored_metadata.metadata[0].num_elements,
        //                                   current_queue);
        // std::cout << "max est error = " << *std::max_element(out_vec.begin(),
        // out_vec.end()) << std::endl;
        DeviceCollective<DeviceType>::AbsMax(
            device_subdomain_buffer[0].shape(0) *
                device_subdomain_buffer[0].shape(1) *
                device_subdomain_buffer[0].shape(2),
            out_1d, SubArray(error_final_out), workspace, true, current_queue);
        if (log::level || log::TIME) {
          DeviceRuntime<DeviceType>::SyncQueue(current_queue);
          qoi_timer.end();
          qoi_timer.print("QoI error estimation: ", total_size / 3);
          qoi_timer.clear();
        }
        MemoryManager<DeviceType>::Copy1D(
            &error_final_out_host, error_final_out.data(), 1, current_queue);
        DeviceRuntime<DeviceType>::SyncQueue(current_queue);
        // reconstructed_data.qoi_in_progress = error_final_out_host ? true :
        // false; std::cout << "==== maximal est error = " <<
        // error_final_out_host << " ====" << std::endl;
        reconstructed_data.qoi_in_progress =
            (error_final_out_host > tol) ? true : false;
        if (reconstructed_data.qoi_in_progress) {
          // CPU version
          if (refactored_metadata.decrease_method == 0) {
            DeviceLauncher<DeviceType>::Execute(
                mgard_x::data_refactoring::multi_dimension::MaxAbsIndexKernel<
                    1, double, DeviceType>(out_1d, SubArray(error_final_out),
                                           SubArray(max_index_d)),
                current_queue);
            uint32_t max_index_h;
            MemoryManager<DeviceType>::Copy1D(&max_index_h, max_index_d.data(),
                                              1, current_queue);
            DeviceRuntime<DeviceType>::SyncQueue(current_queue);
            std::vector<double> new_ebs = ebs;

            T vx, vy, vz;

            T *vx_ptr = device_subdomain_buffer[0].data();
            T *vy_ptr = device_subdomain_buffer[1].data();
            T *vz_ptr = device_subdomain_buffer[2].data();

            MemoryManager<DeviceType>::Copy1D(&vx, &vx_ptr[max_index_h], 1,
                                              current_queue);
            MemoryManager<DeviceType>::Copy1D(&vy, &vy_ptr[max_index_h], 1,
                                              current_queue);
            MemoryManager<DeviceType>::Copy1D(&vz, &vz_ptr[max_index_h], 1,
                                              current_queue);

            error_bound_uniform_decrease<T>(vx, vy, vz, tol,
                                            error_final_out_host, new_ebs);

            // std::cout << "new ebs : ";
            for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
              refactored_metadata.metadata[id].requested_tol = new_ebs[id];
              // std::cout << refactored_metadata.metadata[id].requested_tol <<
              // ", ";
              reconstructor.GenerateRequest(refactored_metadata.metadata[id]);
            }
            // std::cout << std::endl;
          } else if (refactored_metadata.decrease_method == 1) {
            // Segmented
            for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
              reconstructor.GenerateRequest(refactored_metadata.metadata[id]);
            }
          } else if (refactored_metadata.decrease_method == 2) {
            // Hybrid Threshold = 2
            if (error_final_out_host / tol > 2 &&
                (refactored_metadata.metadata[0].corresponding_error_return)) {
              // std::cout << "new ebs : ";
              for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
                refactored_metadata.metadata[id].requested_tol = std::max(
                    refactored_metadata.metadata[id].corresponding_error / 4,
                    tol / error_final_out_host *
                        refactored_metadata.metadata[id].corresponding_error);
                // std::cout << refactored_metadata.metadata[id].requested_tol
                // << ", ";
                reconstructor.GenerateRequest(refactored_metadata.metadata[id]);
              }
              // std::cout << std::endl;
            } else {
              for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
                if (refactored_metadata.metadata[id]
                        .corresponding_error_return) {
                  // std::cout << "Switch to Segmented ..." << std::endl;
                  refactored_metadata.metadata[id].corresponding_error_return =
                      false;
                  refactored_metadata.metadata[id].segmented = true;
                  refactored_metadata.metadata[id].requested_size = 1;
                }
                reconstructor.GenerateRequest(refactored_metadata.metadata[id]);
              }
            }
          } else if (refactored_metadata.decrease_method >= 3) {
            // Hybrid Threshold = 10 with relative or uniform value range eb
            if (error_final_out_host / tol > 10 &&
                (refactored_metadata.metadata[0].corresponding_error_return)) {
              // std::cout << "new ebs : ";
              for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
                refactored_metadata.metadata[id].requested_tol = std::max(
                    refactored_metadata.metadata[id].corresponding_error / 4,
                    tol / error_final_out_host *
                        refactored_metadata.metadata[id].corresponding_error);
                // std::cout << refactored_metadata.metadata[id].requested_tol
                // << ", ";
                reconstructor.GenerateRequest(refactored_metadata.metadata[id]);
              }
              // std::cout << std::endl;
            } else {
              for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
                if (refactored_metadata.metadata[id]
                        .corresponding_error_return) {
                  // std::cout << "Switch to Segmented ..." << std::endl;
                  refactored_metadata.metadata[id].corresponding_error_return =
                      false;
                  refactored_metadata.metadata[id].segmented = true;
                  refactored_metadata.metadata[id].requested_size = 1;
                }
                reconstructor.GenerateRequest(refactored_metadata.metadata[id]);
              }
            }
            // IO_timer.start();
            // size_t size_read = read_mdrx(refactored_metadata,
            // refactored_data, false, config); IO_timer.end();
            // refactored_metadata.IO_time += IO_timer.get();
          }

          mdr_data[0].CopyFromRefactoredData(refactored_metadata.metadata[0],
                                             refactored_data.data[0],
                                             next_queue);
        }
        last_maximal_error = error_final_out_host;
        // std::cout << "reconstructed_data.qoi_in_progress = " <<
        // reconstructed_data.qoi_in_progress << std::endl;
      }

      DeviceRuntime<DeviceType>::SyncQueue(current_queue);

      current_buffer = next_buffer;
      current_queue = next_queue;
      last_ebs[0] = ebs[0];
      last_ebs[1] = ebs[1];
      last_ebs[2] = ebs[2];
    }
  }

  refactored_metadata.metadata[0].requested_tol = ebs[0];
  refactored_metadata.metadata[1].requested_tol = ebs[1];
  refactored_metadata.metadata[2].requested_tol = ebs[2];
  // Copy final data out if we are done with reconstructing
  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {
    // Update reconstructed data
    domain_decomposer.copy_subdomain(
        device_subdomain_buffer[curr_subdomain_id], curr_subdomain_id,
        subdomain_copy_direction::SubdomainToOriginal, current_queue);
  }

  DeviceRuntime<DeviceType>::SyncDevice();
  timer_series.end();
  timer_series.print("Reconstruct pipeline", total_size);
  if (!refactored_metadata.MPI_enabled) {
    std::cout << "Reconstruct pipeline: " << timer_series.get() << "s ("
              << (double)total_size / timer_series.get() / 1e9 << " GB/s)"
              << std::endl;
  } else {
    refactored_metadata.kernel_time = timer_series.get();
  }
  timer_series.clear();

  // std::cout << "Iterations = " << iter << std::endl;
  // std::cout << "Est_max_error = " << error_final_out_host << std::endl;
  refactored_metadata.max_est_error = error_final_out_host;
}

} // namespace MDR
} // namespace mgard_x

#endif // MGARD_X_MDR_RECONSTRUCT_PIPELINE_QOI_HPP
