#!/usr/bin/python
import subprocess
import csv
import numpy as np
import matplotlib.pyplot as plt

def read_levels(filename):
  file = open(filename)
  csv_reader = csv.reader(file)
  results = []
  for row in csv_reader:
    results.append(row[0])
  return results

def read_kernel_names(filename):
  file = open(filename)
  csv_reader = csv.reader(file)
  results = []
  for row in csv_reader:
    results.append(row[1])
  return results

def read_timing(filename):
  file = open(filename)
  csv_reader = csv.reader(file)
  results = []
  for row in csv_reader:
    results.append(float(row[2]))
  return results

def rename_file(name_before, name_after):
  cmd = ['mv', 
          str(name_before),
          str(name_after)]
  subprocess.call(' '.join(cmd), shell = True)

def sum_time_by_kernel(result, kernel):
  sum = 0.0;
  for i in range(len(result[0])):
    if (result[1][i] == kernel):
      sum += result[2][i]
  return sum


def run_fake_data(nrow, ncol, nfib, opt, B):
  tol = 0.001
  s = 0
  profile = 1
  cmd = ['../build/bin/mgard_check_cuda_fake_data', 
          str(nrow), str(ncol), str(nfib), 
          str(tol), str(s), str(opt), str(B), str(profile)]
  subprocess.call(' '.join(cmd), shell = True)
  if (nfib == 1): # 2D
    if (opt == -1):
      refactor_result_before = 'refactor_2D.csv'
      refactor_result_after = 'refactor_2D_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)
      recompose_result_before = 'recompose_2D.csv'
      recompose_result_after = 'recompose_2D_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)
    if (opt == 0):
      refactor_result_before = 'refactor_2D_cuda.csv'
      refactor_result_after = 'refactor_2D_cuda_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)
      recompose_result_before = 'recompose_2D_cuda.csv'
      recompose_result_after = 'recompose_2D_cuda_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)
    if (opt == 1):
      refactor_result_before = 'refactor_2D_cuda_cpt_l1.csv'
      refactor_result_after = 'refactor_2D_cuda_cpt_l1_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)
      recompose_result_before = 'recompose_2D_cuda.csv'
      recompose_result_after = 'recompose_2D_cuda_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)
    if (opt == 2):
      refactor_result_before = 'refactor_2D_cuda_cpt_l2.csv'
      refactor_result_after = 'refactor_2D_cuda_cpt_l2_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)
      recompose_result_before = 'recompose_2D_cuda.csv'
      recompose_result_after = 'recompose_2D_cuda_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)
    if (opt == 3):
      refactor_result_before = 'refactor_2D_cuda_cpt_l2_sm.csv'
      refactor_result_after = 'refactor_2D_cuda_cpt_l2_sm_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)
      recompose_result_before = 'recompose_2D_cuda_cpt_l2_sm.csv'
      recompose_result_after = 'recompose_2D_cuda_cpt_l2_sm_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)

  else: # 3D
    if (opt == -1):
      refactor_result_before = 'refactor_3D.csv'
      refactor_result_after = 'refactor_3D_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)
      recompose_result_before = 'recompose_3D.csv'
      recompose_result_after = 'recompose_3D_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)
    if (opt == 3):
      refactor_result_before = 'refactor_3D_cuda_cpt_l2_sm.csv'
      refactor_result_after = 'refactor_3D_cuda_cpt_l2_sm_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)
      recompose_result_before = 'recompose_3D_cuda_cpt_l2_sm.csv'
      recompose_result_after = 'recompose_3D_cuda_cpt_l2_sm_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B)


  rename_file(refactor_result_before, refactor_result_after)
  rename_file(recompose_result_before, recompose_result_after)
  return [refactor_result_after, recompose_result_after]


def avg_fake_run(nrow, ncol, nfib, opt, B, num_runs):

  results = run_fake_data(nrow, ncol, nfib, opt, B)

  refactor_levels = read_levels(results[0]) # refactor
  recompose_levels = read_levels(results[1]) # recompose

  refactor_kernel_names = read_kernel_names(results[0]) # refactor
  recompose_kernel_names = read_kernel_names(results[1]) # recompose

  refactor_timing_results_all = []
  recompose_timing_results_all = []
  for i in range(num_runs):
    results = run_fake_data(nrow, ncol, nfib, opt, B)
    refactor_timing_results = read_timing(results[0]) # refactor
    recompose_timing_results = read_timing(results[1]) # recompose
    refactor_timing_results_all.append(refactor_timing_results)
    recompose_timing_results_all.append(recompose_timing_results)

  refactor_timing_results_avg = np.average(np.array(refactor_timing_results_all), axis=0)
  recompose_timing_results_avg = np.average(np.array(recompose_timing_results_all), axis=0)

  ret1 = [refactor_levels, refactor_kernel_names, refactor_timing_results_avg.tolist()]
  ret2 = [recompose_levels, recompose_kernel_names, recompose_timing_results_avg.tolist()]
  return [ret1, ret2]

ret = avg_fake_run(65, 65, 1,  0, 16, 3)
print sum_time_by_kernel(ret[0], 'pi_Ql_cuda_time')

def plot_cuda_speedup(kernel_names, cpu_speedup, cuda_speedup, cuda_o1_speedup, cuda_o2_speedup):
  n_groups = cpu_speedup.shape[0];
  print(n_groups)
  fig, ax = plt.subplots()
  index = np.arange(n_groups)
  bar_width = 0.2
  opacity = 0.8

  rects1 = plt.bar(index, cpu_speedup, bar_width,
  alpha=opacity,
  color='b',
  label='CPU')

  rects1 = plt.bar(index + bar_width, cuda_speedup, bar_width,
  alpha=opacity,
  color='g',
  label='CUDA')

  rects1 = plt.bar(index + bar_width + bar_width, cuda_o1_speedup, bar_width,
  alpha=opacity,
  color='r',
  label='CUDA-O1')

  rects1 = plt.bar(index + bar_width + bar_width + bar_width, cuda_o2_speedup, bar_width,
  alpha=opacity,
  color='m',
  label='CUDA-O2')

  plt.xlabel('Kernels')
  plt.ylabel('Speed up')
  plt.title('MGARD CPU vs. MGARD CUDA')
  plt.xticks(index + bar_width, kernel_names, rotation='vertical')
  plt.legend()

  plt.tight_layout()
  plt.show()

# run_fake_data(65, 65, 1, -1, 16)
# avg_fake_run(65, 65, 1,  0, 16, 3)
# run_fake_data(65, 65, 1,  1, 16)
# run_fake_data(65, 65, 1,  2, 16)
# run_fake_data(65, 65, 1,  3, 16)

# run_fake_data(65, 65, 65,  -1, 16)
# run_fake_data(65, 65, 65,  3, 16)

# warming_up_gpu=3
# for i in range(warming_up_gpu):
#   ret = run_cuda_version(600, 400, 1, 0.01, 0, 0)

# kernel_names = get_kernel_names(600, 400, 1, 0.01, 0)


# num_runs = 10
# cpu_results = []
# cuda_results = []
# cuda_o1_results = []
# cuda_o2_results = []
# for i in range(num_runs):
#   ret = run_cpu_version(600, 400, 1, 0.01, 0)
#   cpu_results.append(ret)

#   ret = run_cuda_version(600, 400, 1, 0.01, 0, 0)
#   cuda_results.append(ret)

#   ret = run_cuda_version(600, 400, 1, 0.01, 0, 1)
#   cuda_o1_results.append(ret)

#   ret = run_cuda_version(600, 400, 1, 0.01, 0, 2)
#   cuda_o2_results.append(ret)

# cpu_avg = np.average(np.array(cpu_results), axis=0)
# print(cpu_avg)

# cuda_avg = np.average(np.array(cuda_results), axis=0)
# print(cuda_avg)

# cuda_o1_avg = np.average(np.array(cuda_o1_results), axis=0)
# print(cuda_o1_avg)

# cuda_o2_avg = np.average(np.array(cuda_o2_results), axis=0)
# print(cuda_o2_avg)

# cpu_speedup = np.full(cpu_avg.shape, 1)
# cuda_speedup = cpu_avg/cuda_avg
# cuda_o1_speedup = cpu_avg/cuda_o1_avg
# cuda_o2_speedup = cpu_avg/cuda_o2_avg
# print(cpu_speedup)
# print(cuda_speedup)

# plot_cuda_speedup(kernel_names, cpu_speedup, cuda_speedup, cuda_o1_speedup, cuda_o2_speedup)


