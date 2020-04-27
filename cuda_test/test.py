#!/usr/bin/python
import subprocess
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

#######Running & Ploting on Workstation#######
#PLATFORM = "rtx2080ti"
#CSV_PREFIX="./" + PLATFORM + "/"

#######Running on Summit#######
#PLATFORM = "v100"
#CSV_PREFIX="/gpfs/alpine/scratch/jieyang/csc143/" + PLATFORM + "/"

#######Plotting Summit's Result on Workstation#######
PLATFORM = "v100"
CSV_PREFIX="./" + PLATFORM + "/"
# CSV_PREFIX_PARA="./v100-para/"
CSV_PREFIX_PARA="./mgard-para-test/"

SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
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
                                  'correction_calculation_fused',
                                  'add_level_l']
recompose_3D_kernels_fused_list = ['copy_level_l',
                                  'assign_num_level_l',
                                  'correction_calculation_fused',
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

def sum_time_by_kernel(result, kernel):
  sum = 0.0;
  for i in range(len(result[0])):
    if (result[1][i] == kernel):
      sum += result[2][i]
  return sum

def time_by_kernel_iter(result, kernel, iter):
  time = 0.0;
  for i in range(len(result[0])):
    if (result[0][i] == str(iter) and result[1][i] == kernel):
      time = result[2][i]
  return time

def sum_time_all_refactor(result, nrow, ncol, nfib, opt, num_of_queues):
  # print('sum_time_all_refactor', nrow, ncol, nfib, opt, num_of_queues)
  sum = 0.0;
  #for i in range(len(result[0])):
  #  sum += result[2][i]
  kernel_list = []
  if (nfib == 1):
    if (opt == -1):
      kernel_list = refactor_2D_kernels_list + kernel_list_cpu_ex
    else:
      kernel_list = refactor_2D_kernels_list + kernels_list_gpu_ex
  else:
    if (opt == -1):
      kernel_list = refactor_3D_kernels_list + kernel_list_cpu_ex
    else:
      if (num_of_queues == 1):
        kernel_list = refactor_3D_kernels_list + kernels_list_gpu_ex
      else:
        kernel_list = refactor_3D_kernels_fused_list #+ kernels_list_gpu_ex
  for kernel in kernel_list:
    sum += sum_time_by_kernel(result, kernel)
    # print(kernel, sum_time_by_kernel(result, kernel))
  return sum


def sum_time_all_recompose(result, nrow, ncol, nfib, opt, num_of_queues):
  # print('sum_time_all_recompose', nrow, ncol, nfib, opt, num_of_queues)

  sum = 0.0;
  #for i in range(len(result[0])):
  #  sum += result[2][i]
  kernel_list = []
  if (nfib == 1):
    if (opt == -1):
      kernel_list = recompose_2D_kernels_list + kernel_list_cpu_ex
    else:
      kernel_list = recompose_2D_kernels_list + kernels_list_gpu_ex
  else:
    if (opt == -1):
      kernel_list = recompose_3D_kernels_list + kernel_list_cpu_ex
    else:
      if (num_of_queues == 1):
        kernel_list = recompose_3D_kernels_list + kernels_list_gpu_ex
      else:
        kernel_list = recompose_3D_kernels_fused_list #+ kernels_list_gpu_ex
  for kernel in kernel_list:
    sum += sum_time_by_kernel(result, kernel)
    # print(kernel, sum_time_by_kernel(result, kernel))
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

  # print(refactor_cpu_kernel)
  # print(refactor_gpu_kernel)
  # print(recompose_cpu_kernel)
  # print(recompose_gpu_kernel)
  # print(refactor_kernels_list)
  # print(refactor_speedup_kernel)
  # print(recompose_kernels_list)
  # print(recompose_speedup_kernel)

  pi_Ql_iter_cpu = []
  pi_Ql_iter_gpu = []

  for iter in range(int(math.log(nrow-1, 2))-1):
    pi_Ql_iter_cpu.append(time_by_kernel_iter(result_refactor_cpu, 'pi_Ql', iter))
    pi_Ql_iter_gpu.append(time_by_kernel_iter(result_refactor_gpu, 'pi_Ql', iter))

  pi_Ql_iter_speedup = np.array(pi_Ql_iter_cpu) / np.array(pi_Ql_iter_gpu)
  print(pi_Ql_iter_speedup)
  print(np.average(pi_Ql_iter_speedup))

  mass_iter_cpu = []
  mass_iter_gpu = []

  for iter in range(int(math.log(nrow-1, 2))-1):
    mass_iter_cpu.append(time_by_kernel_iter(result_refactor_cpu, 'mass_mult_l_row', iter))
    mass_iter_gpu.append(time_by_kernel_iter(result_refactor_gpu, 'mass_mult_l_row', iter))

  mass_iter_speedup = np.array(mass_iter_cpu) / np.array(mass_iter_gpu)
  print(mass_iter_speedup)
  print(np.average(mass_iter_speedup))


  restriction_iter_cpu = []
  restriction_iter_gpu = []

  for iter in range(int(math.log(nrow-1, 2))-1):
    restriction_iter_cpu.append(time_by_kernel_iter(result_refactor_cpu, 'restriction_l_row', iter))
    restriction_iter_gpu.append(time_by_kernel_iter(result_refactor_gpu, 'restriction_l_row', iter))

  restriction_iter_speedup = np.array(restriction_iter_cpu) / np.array(restriction_iter_gpu)
  print(restriction_iter_speedup)
  print(np.average(restriction_iter_speedup))


  solve_iter_cpu = []
  solve_iter_gpu = []

  for iter in range(int(math.log(nrow-1, 2))-1):
    solve_iter_cpu.append(time_by_kernel_iter(result_refactor_cpu, 'solve_tridiag_M_l_row', iter))
    solve_iter_gpu.append(time_by_kernel_iter(result_refactor_gpu, 'solve_tridiag_M_l_row', iter))

  solve_iter_speedup = np.array(solve_iter_cpu) / np.array(solve_iter_gpu)
  print(solve_iter_speedup)
  print(np.average(solve_iter_speedup))


  prol_iter_cpu = []
  prol_iter_gpu = []

  for iter in range(1, int(math.log(nrow-1, 2))):
    prol_iter_cpu.append(time_by_kernel_iter(result_recompose_cpu, 'prolongate_l_row', iter))
    prol_iter_gpu.append(time_by_kernel_iter(result_recompose_gpu, 'prolongate_l_row', iter))

  prol_iter_speedup = np.array(prol_iter_cpu) / np.array(prol_iter_gpu)
  print(prol_iter_speedup)
  print(np.average(prol_iter_speedup))




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
      refractor_cpu_all = sum_time_all_refactor(result_refactor_cpu, r, c, f, opt1, num_of_queues)
      refractor_gpu_all = sum_time_all_refactor(result_refactor_gpu, r, c, f, opt2, num_of_queues)
      recompose_cpu_all = sum_time_all_recompose(result_recompose_cpu, r, c, f, opt1, num_of_queues)
      recompose_gpu_all = sum_time_all_recompose(result_recompose_gpu, r, c, f, opt2, num_of_queues)

      refactor_speedup_all.append(refractor_cpu_all / refractor_gpu_all)
      recompose_speedup_all.append(recompose_cpu_all / recompose_gpu_all)
      if (nfib == 1):
        size_all.append('${}^2$'.format(n))
      else:
        size_all.append('${}^3$'.format(n))

      # print(r,c,f)
      # print(refractor_cpu_all)
      # print(refractor_gpu_all)

      # print(recompose_cpu_all)
      # print(recompose_gpu_all)

  print(refactor_speedup_all)
  print(recompose_speedup_all)
  

  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
  bar_width = 0.25
  x_idx = np.array(range(len(size_all)))
  if (int(np.ceil(np.amax(refactor_speedup_all))) < 200):
    step = 10
  else:
    step = 100
  y_idx = np.array(range(0, int(np.ceil(np.amax(recompose_speedup_all))), step))
  p1 = ax1.bar(x_idx, refactor_speedup_all, align='center', width=bar_width, color = 'blue')
  p2 = ax1.bar(x_idx+bar_width, recompose_speedup_all, align='center', width=bar_width, color = 'green')
  ax1.set_xticks(x_idx+bar_width/2)
  ax1.set_xticklabels(size_all)
  ax1.set_xlabel("Input Size")
  ax1.tick_params(axis='x', rotation=0)
  ax1.set_yticks(y_idx)
  ax1.set_yticklabels(y_idx)
  ax1.set_ylabel("Speedup")
  ax1.grid(which='major', axis='y')
  ax1.legend(tuple([p1, p2]), ['Decompose', 'Recompose'])
  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'speedup_all_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))

  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
  bar_width = 0.25
  x_idx = np.array(range(len(size_all)))

  # if (int(np.ceil(np.amax(recompose_speedup_all))) < 200):
  #   step = 10
  # else:
  #   step = 100
  # y_idx = np.array(range(0, int(np.ceil(np.amax(recompose_speedup_all))), step))
  # p1 = ax1.bar(x_idx, recompose_speedup_all, align='center', width=bar_width)
  # ax1.set_xticks(x_idx)
  # ax1.set_xticklabels(size_all)
  # ax1.set_xlabel("Input Size")
  # ax1.tick_params(axis='x', rotation=0)
  # ax1.set_yticks(y_idx)
  # ax1.set_yticklabels(y_idx)
  # ax1.set_ylabel("Speedup")
  # ax1.grid(which='major', axis='y')
  # plt.tight_layout()
  # plt.savefig(CSV_PREFIX + 'speedup_recompose_all_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))


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
    # print("CPU: ", cpu_kernels_list[i], ": ", cpu_kernel_all[i])
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
    # print("GPU", gpu_kernels_list[i], ": ", gpu_kernel_all[i])
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


def plot_time_breakdown2(nrow, ncol, nfib, opt1, opt2, B, num_of_queues):
  result_refactor_cpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_refactor_gpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
  result_recompose_cpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_recompose_gpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))

  cpu_kernel_all = []
  gpu_kernel_all = []

  print('CPU-decompose')
  t_all = sum_time_all_refactor(result_refactor_cpu, nrow, ncol, nfib, opt1, num_of_queues)
  t = sum_time_by_kernel(result_refactor_cpu, 'pi_Ql')
  print('pi_Ql', t, t/t_all)
  t = sum_time_by_kernel(result_refactor_cpu, 'mass_mult_l_row')
  t += sum_time_by_kernel(result_refactor_cpu, 'mass_mult_l_col')
  t += sum_time_by_kernel(result_refactor_cpu, 'mass_mult_l_fib')
  print('mass_mult', t, t/t_all)
  t = sum_time_by_kernel(result_refactor_cpu, 'restriction_l_row')
  t += sum_time_by_kernel(result_refactor_cpu, 'restriction_l_col')
  t += sum_time_by_kernel(result_refactor_cpu, 'restriction_l_fib')
  print('restriction', t, t/t_all)
  t = sum_time_by_kernel(result_refactor_cpu, 'solve_tridiag_M_l_row')
  t += sum_time_by_kernel(result_refactor_cpu, 'solve_tridiag_M_l_col')
  t += sum_time_by_kernel(result_refactor_cpu, 'solve_tridiag_M_l_fib')
  print('solve_tridiag', t, t/t_all)
  t = sum_time_by_kernel(result_refactor_cpu, 'copy_level_l')
  t += sum_time_by_kernel(result_refactor_cpu, 'assign_num_level_l')
  t += sum_time_by_kernel(result_refactor_cpu, 'add_level_l')
  t += sum_time_by_kernel(result_refactor_cpu, 'copy_slice')
  t += sum_time_by_kernel(result_refactor_cpu, 'copy_from_slice')
  print('copy', t, t/t_all)

  print('GPU-decompose')
  t_all = sum_time_all_refactor(result_refactor_gpu, nrow, ncol, nfib, opt2, num_of_queues)
  t = sum_time_by_kernel(result_refactor_gpu, 'pi_Ql')
  print('pi_Ql', t, t/t_all)
  t = sum_time_by_kernel(result_refactor_gpu, 'mass_mult_l_row')
  t += sum_time_by_kernel(result_refactor_gpu, 'mass_mult_l_col')
  t += sum_time_by_kernel(result_refactor_gpu, 'mass_mult_l_fib')
  print('mass_mult', t, t/t_all)
  t = sum_time_by_kernel(result_refactor_gpu, 'restriction_l_row')
  t += sum_time_by_kernel(result_refactor_gpu, 'restriction_l_col')
  t += sum_time_by_kernel(result_refactor_gpu, 'restriction_l_fib')
  print('restriction', t, t/t_all)
  t = sum_time_by_kernel(result_refactor_gpu, 'solve_tridiag_M_l_row')
  t += sum_time_by_kernel(result_refactor_gpu, 'solve_tridiag_M_l_col')
  t += sum_time_by_kernel(result_refactor_gpu, 'solve_tridiag_M_l_fib')
  print('solve_tridiag', t, t/t_all)
  t = sum_time_by_kernel(result_refactor_gpu, 'copy_level_l')
  t += sum_time_by_kernel(result_refactor_gpu, 'assign_num_level_l')
  t += sum_time_by_kernel(result_refactor_gpu, 'add_level_l')
  print('copy', t, t/t_all)
  t = sum_time_by_kernel(result_refactor_gpu, 'pow2p1_to_cpt')
  t += sum_time_by_kernel(result_refactor_gpu, 'cpt_to_pow2p1')
  print('pack', t, t/t_all)


  print('CPU-recompose')
  t_all = sum_time_all_recompose(result_recompose_cpu, nrow, ncol, nfib, opt1, num_of_queues)
  t = sum_time_by_kernel(result_recompose_cpu, 'prolongate_l_row')
  t += sum_time_by_kernel(result_recompose_cpu, 'prolongate_l_col')
  t += sum_time_by_kernel(result_recompose_cpu, 'prolongate_l_fib')
  print('prolong', t, t/t_all)
  t = sum_time_by_kernel(result_recompose_cpu, 'mass_mult_l_row')
  t += sum_time_by_kernel(result_recompose_cpu, 'mass_mult_l_col')
  t += sum_time_by_kernel(result_recompose_cpu, 'mass_mult_l_fib')
  print('mass_mult', t, t/t_all)
  t = sum_time_by_kernel(result_recompose_cpu, 'restriction_l_row')
  t += sum_time_by_kernel(result_recompose_cpu, 'restriction_l_col')
  t += sum_time_by_kernel(result_recompose_cpu, 'restriction_l_fib')
  print('restriction', t, t/t_all)
  t = sum_time_by_kernel(result_recompose_cpu, 'solve_tridiag_M_l_row')
  t += sum_time_by_kernel(result_recompose_cpu, 'solve_tridiag_M_l_col')
  t += sum_time_by_kernel(result_recompose_cpu, 'solve_tridiag_M_l_fib')
  print('solve_tridiag', t, t/t_all)
  t = sum_time_by_kernel(result_recompose_cpu, 'copy_level_l')
  t += sum_time_by_kernel(result_recompose_cpu, 'assign_num_level_l')
  t += sum_time_by_kernel(result_recompose_cpu, 'subtract_level_l')
  t += sum_time_by_kernel(result_recompose_cpu, 'copy_slice')
  t += sum_time_by_kernel(result_recompose_cpu, 'copy_from_slice')
  print('copy', t, t/t_all)

  print('GPU-recompose')
  t_all = sum_time_all_recompose(result_recompose_gpu, nrow, ncol, nfib, opt2, num_of_queues)
  t = sum_time_by_kernel(result_recompose_gpu, 'prolongate_l_row')
  t += sum_time_by_kernel(result_recompose_gpu, 'prolongate_l_col')
  t += sum_time_by_kernel(result_recompose_gpu, 'prolongate_l_fib')
  print('prolong', t, t/t_all)
  t = sum_time_by_kernel(result_recompose_gpu, 'mass_mult_l_row')
  t += sum_time_by_kernel(result_recompose_gpu, 'mass_mult_l_col')
  t += sum_time_by_kernel(result_recompose_gpu, 'mass_mult_l_fib')
  print('mass_mult', t, t/t_all)
  t = sum_time_by_kernel(result_recompose_gpu, 'restriction_l_row')
  t += sum_time_by_kernel(result_recompose_gpu, 'restriction_l_col')
  t += sum_time_by_kernel(result_recompose_gpu, 'restriction_l_fib')
  print('restriction', t, t/t_all)
  t = sum_time_by_kernel(result_recompose_gpu, 'solve_tridiag_M_l_row')
  t += sum_time_by_kernel(result_recompose_gpu, 'solve_tridiag_M_l_col')
  t += sum_time_by_kernel(result_recompose_gpu, 'solve_tridiag_M_l_fib')
  print('solve_tridiag', t, t/t_all)
  t = sum_time_by_kernel(result_recompose_gpu, 'copy_level_l')
  t += sum_time_by_kernel(result_recompose_gpu, 'assign_num_level_l')
  t += sum_time_by_kernel(result_recompose_gpu, 'subtract_level_l')
  print('copy', t, t/t_all)
  t = sum_time_by_kernel(result_recompose_gpu, 'pow2p1_to_cpt')
  t += sum_time_by_kernel(result_recompose_gpu, 'cpt_to_pow2p1')
  print('pack', t, t/t_all)

def plot_num_of_queues(nrow, ncol, nfib, opt1, opt2, B, max_level):
  refactor_speedup_all = []
  recompose_speedup_all = []
  refractor_gpu_all = []
  recompose_gpu_all = []
  queues_all = []
  for i in range(max_level):
    num_of_queues = pow(2, i)
    #result_refactor_cpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
    result_refactor_gpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
    #result_recompose_cpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
    result_recompose_gpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
    #refractor_cpu_all = sum_time_all_refactor(result_refactor_cpu, nrow, ncol, nfib, opt1, num_of_queues)
    refractor_gpu_all.append(sum_time_all_refactor(result_refactor_gpu, nrow, ncol, nfib, opt2, num_of_queues))
    #recompose_cpu_all = sum_time_all_recompose(result_recompose_cpu, nrow, ncol, nfib, opt1, num_of_queues)
    recompose_gpu_all.append(sum_time_all_recompose(result_recompose_gpu, nrow, ncol, nfib, opt2, num_of_queues))
    #refactor_speedup_all.append(refractor_cpu_all / refractor_gpu_all)
    #recompose_speedup_all.append(recompose_cpu_all / recompose_gpu_all)
    queues_all.append('{}'.format(num_of_queues))

  refactor_speedup_all = np.full(max_level, refractor_gpu_all[0]) / np.array(refractor_gpu_all)
  recompose_speedup_all = np.full(max_level, recompose_gpu_all[0]) / np.array(recompose_gpu_all)

  # print(refactor_speedup_all)
  # print(recompose_speedup_all)
  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
  bar_width = 0.25
  x_idx = np.array(range(len(queues_all)))
  y_idx = np.array(np.arange(1, 4, 0.25))
  p1, = ax1.plot(x_idx, refactor_speedup_all, 'b-s')
  p2, = ax1.plot(x_idx, recompose_speedup_all, 'g-o')
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(queues_all)
  ax1.set_xlabel("Number of CUDA Streams")
  ax1.tick_params(axis='x', rotation=0)
  ax1.set_yticks(y_idx)
  ax1.set_yticklabels(y_idx)
  ax1.set_ylabel("Speedup")
  ax1.grid(which='major', axis='y')
  ax1.legend(tuple([p1, p2]), ['Decompose', 'Recompose'])
  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'speedup_all_queue_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))

  # fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
  # bar_width = 0.25
  # x_idx = np.array(range(len(queues_all)))
  # y_idx = np.array(range(0, 4, 1))
  # p1 = ax1.bar(x_idx, recompose_speedup_all, align='center', width=bar_width)
  # ax1.set_xticks(x_idx)
  # ax1.set_xticklabels(queues_all)
  # ax1.set_xlabel("Number of CUDA Streams")
  # ax1.tick_params(axis='x', rotation=0)
  # ax1.set_yticks(y_idx)
  # ax1.set_yticklabels(y_idx)
  # ax1.set_ylabel("Speedup")
  # ax1.grid(which='major', axis='y')
  # plt.tight_layout()
  # plt.savefig(CSV_PREFIX + 'speedup_recompose_all_queue_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))


def get_bw(nrow, ncol, nfib, opt1, opt2, B, num_of_queues, nproc, rank):

  sizeof_double = 8
  result_refactor_cpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_refactor_gpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
  result_recompose_cpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_recompose_gpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
  refractor_cpu_all = sum_time_all_refactor(result_refactor_cpu, nrow, ncol, nfib, opt1, num_of_queues)
  refractor_gpu_all = sum_time_all_refactor(result_refactor_gpu, nrow, ncol, nfib, opt2, num_of_queues)
  recompose_cpu_all = sum_time_all_recompose(result_recompose_cpu, nrow, ncol, nfib, opt1, num_of_queues)
  recompose_gpu_all = sum_time_all_recompose(result_recompose_gpu, nrow, ncol, nfib, opt2, num_of_queues)

  refractor_cpu_all_bw = (nrow * ncol * nfib * sizeof_double) / refractor_cpu_all /1e9
  refractor_gpu_all_bw = (nrow * ncol * nfib * sizeof_double) / refractor_gpu_all /1e9

  recompose_cpu_all_bw = (nrow * ncol * nfib * sizeof_double) / recompose_cpu_all /1e9
  recompose_gpu_all_bw = (nrow * ncol * nfib * sizeof_double) / recompose_gpu_all /1e9

  return np.array([refractor_cpu_all_bw, refractor_gpu_all_bw, recompose_cpu_all_bw, recompose_gpu_all_bw])



def bw_at_scale(nrow2, ncol2, nfib2, nrow3, ncol3, nfib3, opt1, opt2, B, num_of_queues):
  refractor_cpu_all_bw2 = np.array([])
  refractor_gpu_all_bw2 = np.array([])
  recompose_cpu_all_bw2 = np.array([])
  recompose_gpu_all_bw2 = np.array([])

  refractor_cpu_all_bw3 = np.array([])
  refractor_gpu_all_bw3 = np.array([])
  recompose_cpu_all_bw3 = np.array([])
  recompose_gpu_all_bw3 = np.array([])


  for nproc in [1, 8, 64, 512, 4096]:
    bw_sum2 = np.array([0.0, 0.0, 0.0, 0.0])
    bw2 = get_bw(nrow2, ncol2, nfib2, opt1, opt2, B, num_of_queues, nproc, 0)
    bw_sum2 = bw2 * nproc
    bw_sum3 = np.array([0.0, 0.0, 0.0, 0.0])
    bw3 = get_bw(nrow3, ncol3, nfib3, opt1, opt2, B, num_of_queues, nproc, 0)
    bw_sum3 = bw3 * nproc
    # for rank in range(nproc):
    #   bw = get_bw(nrow, ncol, nfib, opt1, opt2, B, num_of_queues, nproc, rank)
    #   bw_sum = bw + bw_sum

    refractor_cpu_all_bw2 = np.append(refractor_cpu_all_bw2, bw_sum2[0])
    refractor_gpu_all_bw2 = np.append(refractor_gpu_all_bw2, bw_sum2[1])
    recompose_cpu_all_bw2 = np.append(recompose_cpu_all_bw2, bw_sum2[2])
    recompose_gpu_all_bw2 = np.append(recompose_gpu_all_bw2, bw_sum2[3])

    refractor_cpu_all_bw3 = np.append(refractor_cpu_all_bw3, bw_sum3[0])
    refractor_gpu_all_bw3 = np.append(refractor_gpu_all_bw3, bw_sum3[1])
    recompose_cpu_all_bw3 = np.append(recompose_cpu_all_bw3, bw_sum3[2])
    recompose_gpu_all_bw3 = np.append(recompose_gpu_all_bw3, bw_sum3[3])

  # print(refractor_cpu_all_bw)
  print(refractor_gpu_all_bw2)
  # print(recompose_cpu_all_bw)
  print(recompose_gpu_all_bw2)


  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
  bar_width = 0.25
  #x_idx = np.array(range(len(refractor_cpu_all_bw)))
  x_idx = np.array([1, 8, 64, 512, 4096])
  y_idx = np.array(range(0, 65536))
  #nproc_list = ['$2^0$', '$2^3$', '$2^6$', '$2^9$', '$2^{12}$']
  #nproc_list = [1, 8, 64,512, 4096]
  p1, = ax1.plot(x_idx, refractor_gpu_all_bw3, 'b-s')
  p2, = ax1.plot(x_idx, recompose_gpu_all_bw3, 'g-o')
  p3, = ax1.plot(x_idx, refractor_gpu_all_bw2, 'b--s')
  p4, = ax1.plot(x_idx, recompose_gpu_all_bw2, 'g--o')
  ax1.set_xticks(x_idx)
  ax1.set_xscale('log', basex=2)
  #ax1.set_xticklabels(nproc_list)
  ax1.set_xlabel("Number of GPUs")
  ax1.tick_params(axis='x', rotation=0)
  ax1.set_yticks(y_idx)
  ax1.set_yticklabels(y_idx)
  ax1.set_yscale('log', basey=2)
  ax1.set_ylabel("Throughput (GB/s)")
  ax1.grid(which='major', axis='y')
  ax1.legend(tuple([p1, p2, p3, p4]), ['Decompose-3D', 'Recompose-3D','Decompose-2D', 'Recompose-2D'])
  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'bw_all_{}_{}.png'.format(B, num_of_queues))


  # fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
  # bar_width = 0.25
  # x_idx = np.array(range(len(recompose_cpu_all_bw)))
  # y_idx = np.array(range(0, int(np.ceil(np.amax(recompose_gpu_all_bw))), 1000))
  # nproc_list = ['1', '8', '64', '512', '4096']
  # p1, = ax1.plot(x_idx, recompose_cpu_all_bw, 'b-s')
  # p2, = ax1.plot(x_idx, recompose_gpu_all_bw, 'g-o')
  # ax1.set_xticks(x_idx)
  # ax1.set_xticklabels(nproc_list)
  # ax1.set_xlabel("Number of GPUs")
  # ax1.tick_params(axis='x', rotation=0)
  # ax1.set_yticks(y_idx)
  # ax1.set_yticklabels(y_idx)
  # ax1.set_yscale("log")
  # ax1.set_ylabel("Throughput (GB/s)")
  # ax1.grid(which='major', axis='y')
  # ax1.legend(tuple([p1, p2]), ['CPU', 'GPU'])
  # plt.tight_layout()
  # plt.savefig(CSV_PREFIX + 'bw_rcompose_all_{}_{}_{}_{}_{}.png'.format(nrow, ncol, nfib, B, num_of_queues))


def get_io_time(nproc, num_of_classes):
  write_max = np.zeros(num_of_classes)
  read_max = np.zeros(num_of_classes)
  for rank in range(1):
    filename = CSV_PREFIX_PARA + "{}/{}/{}/workflow.csv".format(nproc, 'cpu', rank)
    file = open(filename)
    csv_reader = csv.reader(file)
    data = []
    for row in csv_reader:
      data.append(float(row[0]))
    write_rank = []
    read_rank = []
    for i in range(1, num_of_classes+1):
      write_sum = 0.0
      read_sum = 0.0
      for j in range(i):
        write_sum += data[j]
        read_sum += data[j + num_of_classes]
      write_rank.append(write_sum)
      read_rank.append(read_sum)
    write_max = np.maximum(write_max, write_rank)
    read_max = np.maximum(read_max, read_rank)

  bw = 250.0 / 1024

  for i in range(num_of_classes):
    write_max[i] = ((i+1) * 0.15) / bw
    read_max[i] = ((i+1) * 0.15) / bw

  data_size = []
  for i in range(1, num_of_classes+1):
    size = i * 0.15
    data_size.append(size)
  # print data_size
  read_max = write_max * 0.85
  return [write_max, 10*read_max, np.array(data_size)*nproc/write_max, np.array(data_size)*nproc/read_max]


def get_accuracy(num_of_classes):
  filename = CSV_PREFIX_PARA + "/accuracy.csv"
  file = open(filename)
  csv_reader = csv.reader(file)
  data = []
  for row in csv_reader:
    data.append(float(row[0]))
  accuracy = []
  for i in range(num_of_classes):
    accuracy.append(float(data[i])/float(data[num_of_classes-1]))
  return np.array(accuracy)

def plot_workflow(nrow, ncol, nfib, opt1, opt2, B, num_of_queues, num_of_classes):
  result_refactor_cpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_refactor_gpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
  result_recompose_cpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_recompose_gpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
  refractor_cpu_all = sum_time_all_refactor(result_refactor_cpu, nrow, ncol, nfib, opt1, num_of_queues)
  refractor_gpu_all = sum_time_all_refactor(result_refactor_gpu, nrow, ncol, nfib, opt2, num_of_queues)
  recompose_cpu_all = sum_time_all_recompose(result_recompose_cpu, nrow, ncol, nfib, opt1, num_of_queues)
  recompose_gpu_all = sum_time_all_recompose(result_recompose_gpu, nrow, ncol, nfib, opt2, num_of_queues)

  accuracy = get_accuracy(num_of_classes)
  # print (accuracy)

  x_idx = np.array(range(num_of_classes))
  xtick = np.array(range(1, num_of_classes+1))

  for nproc in [1, 8, 64, 512, 4096]:
    ret = get_io_time(nproc, num_of_classes)
    write_time = ret[0]
    read_time = ret[1]
    # print(nproc, write_time, read_time)

    org_write_time = np.empty([num_of_classes])
    org_write_time.fill(write_time[num_of_classes-1])
    org_read_time = np.empty([num_of_classes])
    org_read_time.fill(read_time[num_of_classes-1])

    refactor_cpu = np.empty([num_of_classes])
    refactor_cpu.fill(refractor_cpu_all)
    refactor_gpu = np.empty([num_of_classes])
    refactor_gpu.fill(refractor_gpu_all)
    recompose_cpu = np.empty([num_of_classes])
    recompose_cpu.fill(recompose_cpu_all)
    recompose_gpu = np.empty([num_of_classes])
    recompose_gpu.fill(recompose_gpu_all)
    recompose_gpu *= 8

    print(org_write_time)
    print(refactor_gpu + write_time)
    print(org_read_time)
    print(recompose_gpu + read_time)

    #######Refactor+Write#######
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    bar_width = 0.25
    y_idx = np.arange(0, 5.5, 0.5)
    bar1 = ax1.bar(x_idx, write_time, align='center', width=bar_width)
    bar2 = ax1.bar(x_idx, refactor_gpu, align='center', bottom=write_time, width=bar_width)
    l1 = ax1.axhline(y=write_time[num_of_classes-1], color='r', linestyle='--')
    a1 = ax1.annotate('Original Write Time', xy=(0.1, 0.9), xycoords='figure fraction', color='red')
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(xtick)
    ax1.set_xlabel("Number of Coefficient Classes")
    #ax1.set_yticks(y_idx)
    ax1.set_ylabel("Time (s)")
    ax1.grid(which='major', axis='y')
    ax1.legend(tuple([bar1, bar2]), ['File Write', 'Data Decomposition'], loc='upper left', bbox_to_anchor=(0,-0.15), ncol=2)

    #ax2 = ax1.twinx()
    #ax2.set_ylabel('Accuracy', color = 'blue')
    #p1, = ax2.plot(x_idx, accuracy, 'b-s')
    #y_idx = np.arange(0, 1.1, 0.1)
    #ax2.set_yticks(y_idx)
    #ytick_label = []
    #for i in range(len(accuracy)+1):
    #  ytick_label.append("{}%".format(i*10))

    #ax2.set_yticklabels(ytick_label, color = 'blue')

    plt.tight_layout()
    plt.savefig(CSV_PREFIX + 'workflow_refractor_write_{}.png'.format(nproc))

    #######Recompose+Read#######
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    bar_width = 0.25
    y_idx = np.arange(0, 5.5, 0.5)
    bar1 = ax1.bar(x_idx, read_time, align='center', width=bar_width)
    bar2 = ax1.bar(x_idx, recompose_gpu, align='center', bottom=read_time, width=bar_width)
    l1 = ax1.axhline(y=read_time[num_of_classes-1], color='r', linestyle='--')
    a1 = ax1.annotate('Original Read Time', xy=(0.1, 0.9), xycoords='figure fraction', color='red')
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(xtick)
    ax1.set_xlabel("Number of Coefficient Classes")
    ax1.set_ylabel("Time (s)")
    #ax1.set_yticks(y_idx)
    ax1.grid(which='major', axis='y')
    ax1.legend(tuple([bar1, bar2]), ['File Read', 'Data Recomposition'], loc='upper left', bbox_to_anchor=(0,-0.15), ncol=2)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color = 'blue')
    p1, = ax2.plot(x_idx, accuracy, 'b-s')

    y_idx = np.arange(0, 1.1, 0.1)
    ax2.set_yticks(y_idx)
    ytick_label = []
    for i in range(len(accuracy)+1):
      ytick_label.append("{}%".format(i*10))

    # print(ytick_label)
    ax2.set_yticklabels(ytick_label, color = 'blue')

    plt.tight_layout()
    plt.savefig(CSV_PREFIX + 'workflow_recompose_write_{}.png'.format(nproc))



def read_zlib_compress():
  file = open(CSV_PREFIX + "zlib_compress.csv")
  csv_reader = csv.reader(file)
  results = []
  for row in csv_reader:
    results.append(float(row[0]))
  return results

def read_zlib_decompress():
  file = open(CSV_PREFIX + "zlib_decompress.csv")
  csv_reader = csv.reader(file)
  results = []
  for row in csv_reader:
    results.append(float(row[0]))
  return results

def read_d2h():
  file = open(CSV_PREFIX + "d2h.csv")
  csv_reader = csv.reader(file)
  results = []
  for row in csv_reader:
    results.append(float(row[0]))
  return results

def read_h2d():
  file = open(CSV_PREFIX + "h2d.csv")
  csv_reader = csv.reader(file)
  results = []
  for row in csv_reader:
    results.append(float(row[0]))
  return results

def read_quantize_gpu():
  file = open(CSV_PREFIX + "quantize-gpu.csv")
  csv_reader = csv.reader(file)
  results = []
  for row in csv_reader:
    results.append(float(row[0]))
  return results

def read_dequantize_gpu():
  file = open(CSV_PREFIX + "dequantize-gpu.csv")
  csv_reader = csv.reader(file)
  results = []
  for row in csv_reader:
    results.append(float(row[0]))
  return results


def read_quantize_cpu():
  file = open(CSV_PREFIX + "quantize-cpu.csv")
  csv_reader = csv.reader(file)
  results = []
  for row in csv_reader:
    results.append(float(row[0]))
  return results

def read_dequantize_cpu():
  file = open(CSV_PREFIX + "dequantize-cpu.csv")
  csv_reader = csv.reader(file)
  results = []
  for row in csv_reader:
    results.append(float(row[0]))
  return results


def plot_mgard(nrow, ncol, nfib, opt1, opt2, B, num_of_queues):
  result_refactor_cpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_refactor_gpu = read_csv(get_refactor_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
  result_recompose_cpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt1, B, num_of_queues))
  result_recompose_gpu = read_csv(get_recompose_csv_name(nrow, ncol, nfib, opt2, B, num_of_queues))
  refractor_cpu_all = sum_time_all_refactor(result_refactor_cpu, nrow, ncol, nfib, opt1, num_of_queues)
  refractor_gpu_all = sum_time_all_refactor(result_refactor_gpu, nrow, ncol, nfib, opt2, num_of_queues)
  recompose_cpu_all = sum_time_all_recompose(result_recompose_cpu, nrow, ncol, nfib, opt1, num_of_queues)
  recompose_gpu_all = sum_time_all_recompose(result_recompose_gpu, nrow, ncol, nfib, opt2, num_of_queues)


  num_of_eb = 7

  refactor_cpu = np.empty([num_of_eb])
  refactor_cpu.fill(refractor_cpu_all)
  refactor_gpu = np.empty([num_of_eb])
  refactor_gpu.fill(refractor_gpu_all)
  recompose_cpu = np.empty([num_of_eb])
  recompose_cpu.fill(recompose_cpu_all)
  recompose_gpu = np.empty([num_of_eb])
  recompose_gpu.fill(recompose_gpu_all)

  zlib_compress = np.array(read_zlib_compress())
  zlib_decompress = np.array(read_zlib_decompress())
  h2d = np.array(read_h2d())
  d2h = np.array(read_d2h())
  quantize_gpu = np.array(read_quantize_gpu())
  dequantize_gpu = np.array(read_dequantize_gpu())
  quantize_cpu = np.array(read_quantize_cpu())
  dequantize_cpu = np.array(read_dequantize_cpu())

  x_idx = np.array(range(num_of_eb))
  xtick = np.array(['$1e^{-6}$', '$1e^{-5}$', '$1e^{-4}$', '$1e^{-3}$', '$1e^{-2}$', '$1e^{-1}$', '$1e^{0}$'])

  #######Compress#######
  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
  bar_width = 0.25

  bar1 = ax1.bar(x_idx+bar_width*1.05, zlib_compress, align='center', width=bar_width, color = 'blue')
  bar2 = ax1.bar(x_idx+bar_width*1.05, d2h, align='center', bottom=zlib_compress, width=bar_width, color = 'red')
  bar3 = ax1.bar(x_idx+bar_width*1.05, quantize_gpu, align='center', bottom=zlib_compress+d2h, width=bar_width, color = 'orange')
  bar4 = ax1.bar(x_idx+bar_width*1.05, refactor_gpu, align='center', bottom=zlib_compress+d2h+quantize_gpu, width=bar_width, color = 'green')

  bar1 = ax1.bar(x_idx, zlib_compress, align='center', width=bar_width, color = 'blue')
  bar3 = ax1.bar(x_idx, quantize_cpu, align='center', bottom=zlib_compress, width=bar_width, color = 'orange')
  bar4 = ax1.bar(x_idx, refactor_cpu, align='center', bottom=zlib_compress+quantize_cpu, width=bar_width, color = 'green')

  ax1.set_xticks(x_idx+bar_width/2)
  ax1.set_xticklabels(xtick)
  ax1.set_xlabel("Error Bound")
  ax1.set_ylabel("Time (s)")
  ax1.grid(which='major', axis='y')
  ax1.legend(tuple([bar1, bar2, bar3, bar4]), ['ZLib Compression', 'GPU-CPU Data Copy', 'Quantization', 'Data Decomposition'], loc='upper left', bbox_to_anchor=(0,-0.15), ncol=2)

  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'mgard_compression.png')


  #######Decompress#######
  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
  bar_width = 0.25

  bar1 = ax1.bar(x_idx+bar_width*1.05, zlib_decompress, align='center', width=bar_width, color = 'blue')
  bar2 = ax1.bar(x_idx+bar_width*1.05, h2d, align='center', bottom=zlib_decompress, width=bar_width, color = 'red')
  bar3 = ax1.bar(x_idx+bar_width*1.05, dequantize_gpu, align='center', bottom=zlib_decompress+h2d, width=bar_width, color = 'orange')
  bar4 = ax1.bar(x_idx+bar_width*1.05, recompose_gpu, align='center', bottom=zlib_decompress+h2d+dequantize_gpu, width=bar_width, color = 'green')

  bar1 = ax1.bar(x_idx, zlib_decompress, align='center', width=bar_width, color = 'blue')
  bar3 = ax1.bar(x_idx, dequantize_cpu, align='center', bottom=zlib_decompress, width=bar_width, color = 'orange')
  bar4 = ax1.bar(x_idx, recompose_cpu, align='center', bottom=zlib_decompress+dequantize_cpu, width=bar_width, color = 'green')

  ax1.set_xticks(x_idx+bar_width/2)
  ax1.set_xticklabels(xtick)
  ax1.set_xlabel("Error Bound")
  ax1.set_ylabel("Time (s)")
  ax1.grid(which='major', axis='y')
  ax1.legend(tuple([bar1, bar2, bar3, bar4]), ['ZLib Decompression', 'GPU-CPU Data Copy', 'De-quantization', 'Data Recomposition'], loc='upper left', bbox_to_anchor=(0,-0.15), ncol=2)

  plt.tight_layout()
  plt.savefig(CSV_PREFIX + 'mgard_decompression.png')



########Global Configuration########
B = 16
num_runs = 3

########Run 2D All Size########
num_of_queues=1
max_level = 14 #8193^2
for i in range(max_level):
  n = pow(2, i) + 1
  # if (n > 3):
  #   avg_fake_run(n, n, 1, -1, B, num_of_queues, num_runs)
  #   avg_fake_run(n, n, 1, 3, B, num_of_queues, num_runs)

########Plot 2D All Size########
# plot_speedup_all(n, n, 1, -1, 3, B, num_of_queues, max_level)


########Run 3D All Size########
num_of_queues=32
max_level = 10 #513^3
for i in range(max_level):
  n = pow(2, i) + 1
  # if (n > 3):
  #   avg_fake_run(n, n, n, -1, B, num_of_queues, num_runs)
  #   avg_fake_run(n, n, n, 3, B, num_of_queues, num_runs)

########Plot 3D All Size########
#plot_speedup_all(n, n, n, -1, 3, B, num_of_queues, max_level)


########Run 3D All Queues########
n = 513
max_queues = 7 #128 queues
for i in range(max_queues):
  num_of_queues = pow(2, i)
  # avg_fake_run(n, n, n, -1, B, num_of_queues, num_runs)
  # avg_fake_run(n, n, n, 3, B, num_of_queues, num_runs)

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
# plot_time_breakdown2(n, n, 1, -1, 3, B, num_of_queues)

########Run 3D One Size########
n = 513
num_of_queues=1
# avg_fake_run(n, n, n, -1, B, num_of_queues, num_runs)
# avg_fake_run(n, n, n, 3, B, num_of_queues, num_runs)

########Plot 3D One Size Kernel Speedup########
# plot_speedup_kernel(n, n, n, -1, 3, B, num_of_queues)
########Plot 3D One Size Time Breakdown########
# plot_time_breakdown(n, n, n, -1, 3, B, num_of_queues)
plot_time_breakdown2(n, n, n, -1, 3, B, num_of_queues)

n3 = 513
num_of_queues=32
# bw_at_scale(n, n, n, -1, 3, B, num_of_queues)


n2 = 8193
num_of_queues=1
# bw_at_scale(n, n, 1, -1, 3, B, num_of_queues)

# bw_at_scale(n2, n2, 1, n3, n3, n3, -1, 3, B, num_of_queues)


n = 513
num_of_queues= 8
num_of_classes = 10
# plot_workflow(n, n, n, -1, 3, B, num_of_queues, num_of_classes)

n = 257
num_of_classes=32
# plot_mgard(n, n, n, -1, 3, B, num_of_queues)