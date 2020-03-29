#!/usr/bin/python
import subprocess
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

PLATFORM = "gtx2080ti"
CSV_PREFIX="./" + PLATFORM + "/"

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


refactor_2D_kernels_list = ['pi_Ql',
                            'copy_level_l',
                            'assign_num_level_l',
                            'mass_mult_l_row',
                            'restriction_l_row',
                            'solve_tridiag_M_l_row',
                            'mass_mult_l_col',
                            'restriction_l_col',
                            'solve_tridiag_M_l_col',
                            'add_level_l']
recompose_2D_kernels_list = ['copy_level_l',
                            'assign_num_level_l',
                            'mass_mult_l_row',
                            'restriction_l_row',
                            'solve_tridiag_M_l_row',
                            'mass_mult_l_col',
                            'restriction_l_col',
                            'solve_tridiag_M_l_col',
                            'subtract_level_l',
                            'prolongate_l_row',
                            'prolongate_l_col']

refactor_3D_kernels_list = ['pi_Ql',
                            'copy_level_l',
                            'assign_num_level_l',
                            'mass_mult_l_row',
                            'restriction_l_row',
                            'solve_tridiag_M_l_row',
                            'mass_mult_l_col',
                            'restriction_l_col',
                            'solve_tridiag_M_l_col',
                            'mass_mult_l_fib',
                            'restriction_l_fib',
                            'solve_tridiag_M_l_fib',
                            'add_level_l']
recompose_3D_kernels_list = ['copy_level_l',
                            'assign_num_level_l',
                            'mass_mult_l_row',
                            'restriction_l_row',
                            'solve_tridiag_M_l_row',
                            'mass_mult_l_col',
                            'restriction_l_col',
                            'solve_tridiag_M_l_col',
                            'mass_mult_l_fib',
                            'restriction_l_fib',
                            'solve_tridiag_M_l_fib',
                            'subtract_level_l',
                            'prolongate_l_row',
                            'prolongate_l_col',
                            'prolongate_l_fib']


refactor_3D_kernels_fused_list = ['pi_Ql',
                                  'copy_level_l',
                                  'assign_num_level_l',
                                  'correction_caclculation_fused'
                                  'add_level_l']
recompose_3D_kernels_fused_list = ['copy_level_l',
                                  'assign_num_level_l',
                                  'correction_caclculation_fused',
                                  'subtract_level_l',
                                  'prolongate_calculation_fused']

kernels_list_gpu_ex = ['pow2p1_to_cpt',
                   'cpt_to_pow2p1'
                   #'org_to_pow2p1',
                   #'pow2p1_to_org'
                   ]

kernel_list_cpu_ex = ['copy_slice',
                      'copy_from_slice']


def Union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list 


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

def write_csv(filename, data):
  file = open(filename, 'w')
  csv_writer = csv.writer(file)
  for i in range(len(data[0])):
    csv_writer.writerow([data[0][i], data[1][i], data[2][i]])

def read_csv(filename):
  levels = read_levels(filename)
  kernels = read_kernel_names(filename)
  timing = read_timing(filename)
  return [levels, kernels, timing]

def rename_file(name_before, name_after):
  cmd = ['mv', 
          str(name_before),
          str(name_after)]
  subprocess.call(' '.join(cmd), shell = True)

def sum_time_all(result):
  sum = 0.0;
  for i in range(len(result[0])):
    sum += result[2][i]
  return sum

def sum_time_by_kernel(result, kernel):
  sum = 0.0;
  for i in range(len(result[0])):
    if (result[1][i] == kernel):
      sum += result[2][i]
  return sum


def get_refactor_csv_name(nrow, ncol, nfib, opt, B, num_of_queues):
  if (nfib == 1): # 2D
    if (opt == -1):
      return CSV_PREFIX + 'refactor_2D_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 0):
      return CSV_PREFIX + 'refactor_2D_cuda_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 1):
      return CSV_PREFIX + 'refactor_2D_cuda_cpt_l1_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 2):
      return CSV_PREFIX + 'refactor_2D_cuda_cpt_l2_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 3):
      return CSV_PREFIX + 'refactor_2D_cuda_cpt_l2_sm_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)

  else: # 3D
    if (opt == -1):
      return CSV_PREFIX + 'refactor_3D_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 3):
      return CSV_PREFIX + 'refactor_3D_cuda_cpt_l2_sm_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)

def get_recompose_csv_name(nrow, ncol, nfib, opt, B, num_of_queues):
  if (nfib == 1): # 2D
    if (opt == -1):
      return CSV_PREFIX + 'recompose_2D_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 0):
      return CSV_PREFIX + 'recompose_2D_cuda_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 1):
      return CSV_PREFIX + 'recompose_2D_cuda_cpt_l1_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 2):
      return CSV_PREFIX + 'recompose_2D_cuda_cpt_l2_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 3):
      return CSV_PREFIX + 'recompose_2D_cuda_cpt_l2_sm_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)

  else: # 3D
    if (opt == -1):
      return CSV_PREFIX + 'recompose_3D_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 3):
      return CSV_PREFIX + 'recompose_3D_cuda_cpt_l2_sm_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)


def run_fake_data(nrow, ncol, nfib, opt, B, num_of_queues):
  tol = 0.001
  s = 0
  profile = 1
  cmd = ['../build/bin/mgard_check_cuda_fake_data', 
          str(nrow), str(ncol), str(nfib), 
          str(tol), str(s), str(opt), str(B), str(profile),
          str(num_of_queues), str(CSV_PREFIX)]
  print(' '.join(cmd))
  subprocess.call(' '.join(cmd), shell = True)
  if (nfib == 1): # 2D
    if (opt == -1):
      refactor_result_before = CSV_PREFIX + 'refactor_2D.csv'
      recompose_result_before = CSV_PREFIX + 'recompose_2D.csv'
    if (opt == 0):
      refactor_result_before = CSV_PREFIX + 'refactor_2D_cuda.csv'
      recompose_result_before = CSV_PREFIX + 'recompose_2D_cuda.csv'
    if (opt == 1):
      refactor_result_before = CSV_PREFIX + 'refactor_2D_cuda_cpt_l1.csv'
      recompose_result_before = CSV_PREFIX + 'recompose_2D_cuda.csv'
    if (opt == 2):
      refactor_result_before = CSV_PREFIX + 'refactor_2D_cuda_cpt_l2.csv'
      recompose_result_before = CSV_PREFIX + 'recompose_2D_cuda.csv'
    if (opt == 3):
      refactor_result_before = CSV_PREFIX + 'refactor_2D_cuda_cpt_l2_sm.csv'
      recompose_result_before = CSV_PREFIX + 'recompose_2D_cuda_cpt_l2_sm.csv'
  else: # 3D
    if (opt == -1):
      refactor_result_before = CSV_PREFIX + 'refactor_3D.csv'
      recompose_result_before = CSV_PREFIX + 'recompose_3D.csv'
    if (opt == 3):
      refactor_result_before = CSV_PREFIX + 'refactor_3D_cuda_cpt_l2_sm.csv'
      recompose_result_before = CSV_PREFIX + 'recompose_3D_cuda_cpt_l2_sm.csv'

  refactor_result_after = get_refactor_csv_name(nrow, ncol, nfib, opt, B, num_of_queues)
  recompose_result_after = get_recompose_csv_name(nrow, ncol, nfib, opt, B, num_of_queues)

  rename_file(refactor_result_before, refactor_result_after)
  rename_file(recompose_result_before, recompose_result_after)
  return [refactor_result_after, recompose_result_after]


def avg_fake_run(nrow, ncol, nfib, opt, B, num_of_queues, num_runs):
  refactor_timing_results_all = []
  recompose_timing_results_all = []
  for i in range(num_runs):
    results = run_fake_data(nrow, ncol, nfib, opt, B, num_of_queues)
    refactor_levels = read_levels(results[0]) # refactor
    recompose_levels = read_levels(results[1]) # recompose

    refactor_kernel_names = read_kernel_names(results[0]) # refactor
    recompose_kernel_names = read_kernel_names(results[1]) # recompose
    refactor_timing_results = read_timing(results[0]) # refactor
    recompose_timing_results = read_timing(results[1]) # recompose
    refactor_timing_results_all.append(refactor_timing_results)
    recompose_timing_results_all.append(recompose_timing_results)

  refactor_timing_results_avg = np.average(np.array(refactor_timing_results_all), axis=0)
  recompose_timing_results_avg = np.average(np.array(recompose_timing_results_all), axis=0)

  ret1 = [refactor_levels, refactor_kernel_names, refactor_timing_results_avg.tolist()]
  ret2 = [recompose_levels, recompose_kernel_names, recompose_timing_results_avg.tolist()]
  write_csv(results[0], ret1)
  write_csv(results[1], ret2)
  return [results[0], results[1]]


def plot_speedup_kernel(nrow, ncol, nfib, opt1, opt2, B, num_of_queues):

  result_refactor_cpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_refactor_gpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
  result_recompose_cpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_recompose_gpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))

  refactor_cpu_kernel = []
  refactor_gpu_kernel = []

  recompose_cpu_kernel = []
  recompose_gpu_kernel = []

  if (nfib == 1):
    refactor_kernels_list = refactor_2D_kernels_list
    recompose_kernels_list = recompose_2D_kernels_list
  else:
    refactor_kernels_list = refactor_3D_kernels_list
    recompose_kernels_list = recompose_3D_kernels_list

  for kernel in refactor_kernels_list:
    t = sum_time_by_kernel(result_refactor_cpu, kernel)
    refactor_cpu_kernel.append(t)
  

  for kernel in refactor_kernels_list:
    t = sum_time_by_kernel(result_refactor_gpu, kernel)
    refactor_gpu_kernel.append(t)
  

  for kernel in recompose_kernels_list:
    t = sum_time_by_kernel(result_recompose_cpu, kernel)
    recompose_cpu_kernel.append(t)
    

  for kernel in recompose_kernels_list:
    t = sum_time_by_kernel(result_recompose_gpu, kernel)
    recompose_gpu_kernel.append(t)
    
  refactor_speedup_kernel = np.array(refactor_cpu_kernel)/np.array(refactor_gpu_kernel)
  recompose_speedup_kernel = np.array(recompose_cpu_kernel)/np.array(recompose_gpu_kernel)

  print(refactor_cpu_kernel)
  print(refactor_gpu_kernel)
  print(recompose_cpu_kernel)
  print(recompose_gpu_kernel)
  print(refactor_speedup_kernel)
  print(recompose_speedup_kernel)

  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
  width = 0.25
  x_idx = np.array(range(len(refactor_kernels_list)))
  y_idx = np.array(range(0, int(np.ceil(np.amax(refactor_speedup_kernel))), 100))
  p1 = ax1.bar(x_idx, refactor_speedup_kernel, width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(refactor_kernels_list)
  ax1.set_xlabel("Kernels")
  ax1.tick_params(axis='x', rotation=90)
  ax1.set_yticks(y_idx)
  ax1.set_ylabel("Speedup")
  ax1.grid(which='major', axis='y')
  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'speedup_refactor_kernel_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))

  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
  width = 0.25
  x_idx = np.array(range(len(recompose_kernels_list)))
  y_idx = np.array(range(0, int(np.ceil(np.amax(recompose_speedup_kernel))), 100))
  p1 = ax1.bar(x_idx, recompose_speedup_kernel, width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(recompose_kernels_list)
  ax1.set_xlabel("Kernels")
  ax1.tick_params(axis='x', rotation=90)
  ax1.set_yticks(y_idx)
  ax1.set_ylabel("Speedup")
  ax1.grid(which='major', axis='y')
  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'speedup_recompose_kernel_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))

  

def plot_speedup_all(nrow, ncol, nfib, opt1, opt2, B, num_of_queues, max_level):
  refactor_speedup_all = []
  recompose_speedup_all = []
  size_all = []
  for i in range(max_level):
    n = pow(2, i) + 1
    if (n >= 33):
      r = n
      c = n
      if (nfib == 1):
        f = 1
      else:
        f = n
      result_refactor_cpu = read_csv(get_refactor_csv_name(r, c, f, opt1, B, num_of_queues))
      result_refactor_gpu = read_csv(get_refactor_csv_name(r, c, f, opt2, B, num_of_queues))
      result_recompose_cpu = read_csv(get_recompose_csv_name(r, c, f, opt1, B, num_of_queues))
      result_recompose_gpu = read_csv(get_recompose_csv_name(r, c, f, opt2, B, num_of_queues))
      refractor_cpu_all = sum_time_all(result_refactor_cpu)
      refractor_gpu_all = sum_time_all(result_refactor_gpu)
      recompose_cpu_all = sum_time_all(result_recompose_cpu)
      recompose_gpu_all = sum_time_all(result_recompose_gpu)
      refactor_speedup_all.append(refractor_cpu_all / refractor_gpu_all)
      recompose_speedup_all.append(recompose_cpu_all / recompose_gpu_all)
      if (nfib == 1):
        size_all.append('${}^2$'.format(n))
      else:
        size_all.append('${}^3$'.format(n))

  print(refactor_speedup_all)
  print(recompose_speedup_all)

  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
  bar_width = 0.25
  x_idx = np.array(range(len(size_all)))
  y_idx = np.array(range(0, int(np.ceil(np.amax(refactor_speedup_all))), 10))
  p1 = ax1.bar(x_idx, refactor_speedup_all, align='center', width=bar_width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(size_all)
  ax1.set_xlabel("Input Size")
  ax1.tick_params(axis='x', rotation=0)
  ax1.set_yticks(y_idx)
  ax1.set_yticklabels(y_idx)
  ax1.set_ylabel("Speedup")
  ax1.grid(which='major', axis='y')
  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'speedup_refactor_all_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))

  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
  bar_width = 0.25
  x_idx = np.array(range(len(size_all)))
  y_idx = np.array(range(0, int(np.ceil(np.amax(recompose_speedup_all))), 10))
  p1 = ax1.bar(x_idx, recompose_speedup_all, align='center', width=bar_width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(size_all)
  ax1.set_xlabel("Input Size")
  ax1.tick_params(axis='x', rotation=0)
  ax1.set_yticks(y_idx)
  ax1.set_yticklabels(y_idx)
  ax1.set_ylabel("Speedup")
  ax1.grid(which='major', axis='y')
  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'speedup_recompose_all_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))


def plot_time_breakdown(nrow, ncol, nfib, opt1, opt2, B, num_of_queues):
  result_refactor_cpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_refactor_gpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
  result_recompose_cpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_recompose_gpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))

  cpu_kernel_all = []
  gpu_kernel_all = []

  if (nfib == 1):
    refactor_cpu_kernels_list = refactor_2D_kernels_list + kernel_list_cpu_ex
    recompose_cpu_kernels_list = recompose_2D_kernels_list + kernel_list_cpu_ex
    refactor_gpu_kernels_list = refactor_2D_kernels_list + kernels_list_gpu_ex
    recompose_gpu_kernels_list = recompose_2D_kernels_list + kernels_list_gpu_ex
  else:

    refactor_cpu_kernels_list = refactor_3D_kernels_list + kernel_list_cpu_ex
    recompose_cpu_kernels_list = recompose_3D_kernels_list + kernel_list_cpu_ex
    if (num_of_queues == 1):
      refactor_gpu_kernels_list = refactor_3D_kernels_list + kernels_list_gpu_ex
      recompose_gpu_kernels_list = recompose_3D_kernels_list + kernels_list_gpu_ex
    else:
      refactor_gpu_kernels_list = refactor_3D_kernels_fused_list + kernels_list_gpu_ex
      recompose_gpu_kernels_list = recompose_3D_kernels_fused_list + kernels_list_gpu_ex

  cpu_kernels_list = Union(refactor_cpu_kernels_list, recompose_cpu_kernels_list)
  gpu_kernels_list = Union(refactor_gpu_kernels_list, recompose_gpu_kernels_list)

  for kernel in cpu_kernels_list:
    t1 = sum_time_by_kernel(result_refactor_cpu, kernel)
    t2 = sum_time_by_kernel(result_recompose_cpu, kernel)
    cpu_kernel_all.append([t1, t2])
    
  for kernel in gpu_kernels_list:
    t1 = sum_time_by_kernel(result_refactor_gpu, kernel)
    t2 = sum_time_by_kernel(result_recompose_gpu, kernel)
    gpu_kernel_all.append([t1, t2])

  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
  bar_width = 0.25
  y_idx = np.array([1, 0]) # reverse the order of refactor and recompose
  #y_idx = np.array(range(0, int(np.ceil(np.amax(refactor_speedup_all))), 1))
  last_bar=[0,0]
  bars = []
  for i in range(len(cpu_kernels_list)):
    print("CPU: ", cpu_kernels_list[i], ": ", cpu_kernel_all[i])
    bar = ax1.barh(y_idx, cpu_kernel_all[i], align='center', left=last_bar, height=bar_width)
    last_bar = [last_bar[0] + cpu_kernel_all[i][0], last_bar[1] + cpu_kernel_all[i][1]]
    bars.append(bar)

  ax1.set_yticks(y_idx)
  ax1.set_yticklabels(['refactor', 'recompose'])
  ax1.tick_params(axis='y', rotation=0)
  #ax1.set_yticks(y_idx)
  #ax1.set_yticklabels(y_idx)
  ax1.grid(which='major', axis='x')
  ax1.set_xlabel("Time (sec)")
  ax1.legend(tuple(bars), cpu_kernels_list, loc='upper left', bbox_to_anchor=(0,-0.2), ncol=3)
  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'cpu_time_breakdown_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))


  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
  bar_width = 0.25
  y_idx = np.array([1, 0]) # reverse the order of refactor and recompose
  #y_idx = np.array(range(4))
  last_bar=[0, 0]
  bars = []
  #for i in range(len(Union(refactor_cpu_kernels_list, recompose_cpu_kernels_list))):
  for i in range(len(gpu_kernels_list)):
    print("GPU", gpu_kernels_list[i], ": ", gpu_kernel_all[i])
    b = ax1.barh(y_idx, gpu_kernel_all[i], align='center', left=last_bar, height=bar_width)
    last_bar = [last_bar[0] + gpu_kernel_all[i][0], last_bar[1] + gpu_kernel_all[i][1]]
    bars.append(b)

  ax1.set_yticks(y_idx)
  ax1.set_yticklabels(['refactor', 'recompose'])
  ax1.tick_params(axis='y', rotation=0)
  #ax1.set_yticks(y_idx)
  #ax1.set_yticklabels(y_idx)
  ax1.grid(which='major', axis='x')
  ax1.set_xlabel("Time (sec)")
  ax1.legend(tuple(bars), gpu_kernels_list, loc='upper left', bbox_to_anchor=(0,-0.2), ncol=3)
  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'gpu_time_breakdown_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))


def plot_num_of_queues(nrow, ncol, nfib, opt1, opt2, B, max_level):
  refactor_speedup_all = []
  recompose_speedup_all = []
  queues_all = []
  for i in range(max_level):
    num_of_queues = pow(2, i)
    result_refactor_cpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
    result_refactor_gpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
    result_recompose_cpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
    result_recompose_gpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
    refractor_cpu_all = sum_time_all(result_refactor_cpu)
    refractor_gpu_all = sum_time_all(result_refactor_gpu)
    recompose_cpu_all = sum_time_all(result_recompose_cpu)
    recompose_gpu_all = sum_time_all(result_recompose_gpu)
    refactor_speedup_all.append(refractor_cpu_all / refractor_gpu_all)
    recompose_speedup_all.append(recompose_cpu_all / recompose_gpu_all)
    queues_all.append('{}'.format(num_of_queues))

  print(refactor_speedup_all)
  print(recompose_speedup_all)
  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
  bar_width = 0.25
  x_idx = np.array(range(len(queues_all)))
  y_idx = np.array(range(0, int(np.ceil(np.amax(refactor_speedup_all))), 10))
  p1 = ax1.bar(x_idx, refactor_speedup_all, align='center', width=bar_width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(queues_all)
  ax1.set_xlabel("Number of CUDA Streams")
  ax1.tick_params(axis='x', rotation=0)
  ax1.set_yticks(y_idx)
  ax1.set_yticklabels(y_idx)
  ax1.set_ylabel("Speedup")
  ax1.grid(which='major', axis='y')
  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'speedup_refactor_all_queue_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))

  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
  bar_width = 0.25
  x_idx = np.array(range(len(queues_all)))
  y_idx = np.array(range(0, int(np.ceil(np.amax(recompose_speedup_all))), 10))
  p1 = ax1.bar(x_idx, recompose_speedup_all, align='center', width=bar_width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(queues_all)
  ax1.set_xlabel("Number of CUDA Streams")
  ax1.tick_params(axis='x', rotation=0)
  ax1.set_yticks(y_idx)
  ax1.set_yticklabels(y_idx)
  ax1.set_ylabel("Speedup")
  ax1.grid(which='major', axis='y')
  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'speedup_recompose_all_queue_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))


def get_bw(nrow, ncol, nfib, opt1, opt2, B, num_of_queues, nproc, rank):

  sizeof_double = 8
  result_refactor_cpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_refactor_gpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
  result_recompose_cpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_recompose_gpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
  refractor_cpu_all = sum_time_all(result_refactor_cpu)
  refractor_gpu_all = sum_time_all(result_refactor_gpu)
  recompose_cpu_all = sum_time_all(result_recompose_cpu)
  recompose_gpu_all = sum_time_all(result_recompose_gpu)

  refractor_cpu_all_bw = (nrow * ncol * nfib * sizeof_double) / refractor_cpu_all /1e9
  refractor_gpu_all_bw = (nrow * ncol * nfib * sizeof_double) / refractor_gpu_all /1e9

  recompose_cpu_all_bw = (nrow * ncol * nfib * sizeof_double) / recompose_cpu_all /1e9
  recompose_gpu_all_bw = (nrow * ncol * nfib * sizeof_double) / recompose_gpu_all /1e9

  return np.array([refractor_cpu_all_bw, refractor_gpu_all_bw, recompose_cpu_all_bw, recompose_gpu_all_bw])



def bw_at_scale(nrow, ncol, nfib, opt1, opt2, B, num_of_queues):
  refractor_cpu_all_bw = np.array([])
  refractor_gpu_all_bw = np.array([])
  recompose_cpu_all_bw = np.array([])
  recompose_gpu_all_bw = np.array([])


  for nproc in [1, 8, 64, 512, 4096]:
    bw_sum = np.array([0.0, 0.0, 0.0, 0.0])
    for rank in range(nproc):
      bw = get_bw(nrow, ncol, nfib, opt1, opt2, B, num_of_queues, nproc, rank)
      bw_sum = bw + bw_sum

    refractor_cpu_all_bw = np.append(refractor_cpu_all_bw, bw_sum[0])
    refractor_gpu_all_bw = np.append(refractor_gpu_all_bw, bw_sum[1])
    recompose_cpu_all_bw = np.append(recompose_cpu_all_bw, bw_sum[2])
    recompose_gpu_all_bw = np.append(recompose_gpu_all_bw, bw_sum[3])

  print(refractor_cpu_all_bw)
  print(refractor_gpu_all_bw)
  print(recompose_cpu_all_bw)
  print(recompose_gpu_all_bw)

  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
  bar_width = 0.25
  x_idx = np.array(range(len(refractor_cpu_all_bw)))
  y_idx = np.array(range(0, int(np.ceil(np.amax(refractor_gpu_all_bw))), 1000))
  nproc_list = ['1', '8', '64', '512', '4096']
  p1, = ax1.plot(x_idx, refractor_cpu_all_bw, 'b-s')
  p2, = ax1.plot(x_idx, refractor_gpu_all_bw, 'g-o')
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(nproc_list)
  ax1.set_xlabel("Number of GPUs")
  ax1.tick_params(axis='x', rotation=0)
  ax1.set_yticks(y_idx)
  ax1.set_yticklabels(y_idx)
  ax1.set_yscale("log")
  ax1.set_ylabel("Throughput (GB/s)")
  ax1.grid(which='major', axis='y')
  ax1.legend(tuple([p1, p2]), ['CPU', 'GPU'])
  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'bw_refactor_all_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))


  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
  bar_width = 0.25
  x_idx = np.array(range(len(recompose_cpu_all_bw)))
  y_idx = np.array(range(0, int(np.ceil(np.amax(recompose_gpu_all_bw))), 1000))
  nproc_list = ['1', '8', '64', '512', '4096']
  p1, = ax1.plot(x_idx, recompose_cpu_all_bw, 'b-s')
  p2, = ax1.plot(x_idx, recompose_gpu_all_bw, 'g-o')
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(nproc_list)
  ax1.set_xlabel("Number of GPUs")
  ax1.tick_params(axis='x', rotation=0)
  ax1.set_yticks(y_idx)
  ax1.set_yticklabels(y_idx)
  ax1.set_yscale("log")
  ax1.set_ylabel("Throughput (GB/s)")
  ax1.grid(which='major', axis='y')
  ax1.legend(tuple([p1, p2]), ['CPU', 'GPU'])
  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'bw_rcompose_all_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))


########Global Configuration########
B = 16
num_runs = 3

########Run 2D All Size########
num_of_queues=1
max_level = 14 #8193^2
for i in range(max_level):
  n = pow(2, i) + 1
  if (n > 3):
    avg_fake_run(n, n, 1, -1, B, num_of_queues, num_runs)
    avg_fake_run(n, n, 1, 3, B, num_of_queues, num_runs)

########Plot 2D All Size########
# plot_speedup_all(n, n, 1, -1, 3, B, num_of_queues, max_level)


########Run 3D All Size########
num_of_queues=32
max_level = 10 #513^3
for i in range(max_level):
  n = pow(2, i) + 1
  if (n > 3):
    avg_fake_run(n, n, n, -1, B, num_of_queues, num_runs)
    avg_fake_run(n, n, n, 3, B, num_of_queues, num_runs)

########Plot 3D All Size########
# plot_speedup_all(n, n, n, -1, 3, B, num_of_queues, max_level)


########Run 3D All Queues########
n = 513
max_queues = 7 #128 queues
for i in range(max_queues):
  num_of_queues = pow(2, i)
  avg_fake_run(n, n, n, -1, B, num_of_queues, num_runs)
  avg_fake_run(n, n, n, 3, B, num_of_queues, num_runs)

########Plot 3D All Queues########
# plot_num_of_queues(n, n, n, -1, 3, B, max_queues)


########Run 2D One Size########
n = 8193
num_of_queues=1
# avg_fake_run(n, n, 1, -1, B, num_of_queues, num_runs)
# avg_fake_run(n, n, 1, 3, B, num_of_queues, num_runs)

########Plot 2D One Size Kernel Speedup########
# plot_speedup_kernel(n, n, 1, -1, 3, B, num_of_queues)
########Plot 2D One Size Time Breakdown########
# plot_time_breakdown(n, n, 1, -1, 3, B, num_of_queues)

########Run 3D One Size########
n = 513
num_of_queues=1
# avg_fake_run(n, n, n, -1, B, num_of_queues, num_runs)
# avg_fake_run(n, n, n, 3, B, num_of_queues, num_runs)

########Plot 3D One Size Kernel Speedup########
# plot_speedup_kernel(n, n, n, -1, 3, B, num_of_queues)
########Plot 3D One Size Time Breakdown########
# plot_time_breakdown(n, n, n, -1, 3, B, num_of_queues)

n = 513
num_of_queues=32
#bw_at_scale(n, n, n, -1, 3, B, num_of_queues)


n = 8193
num_of_queues=1
#bw_at_scale(n, n, 1, -1, 3, B, num_of_queues)