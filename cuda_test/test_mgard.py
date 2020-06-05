#!/usr/bin/python
import subprocess
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math


#######Configure platfrom#######
PLATFORM = "rtx2080ti"
#PLATFORM = "v100"

CSV_PREFIX="./" + PLATFORM + "/"


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




def run_test(nrow, ncol, nfib, opt, B, num_of_queues):
  tol = 0.001
  s = 0
  profile = 1
  if (nfib == 1):
    DATAIN = "./bp2bin/gs_bin_data/gs_{}_{}_2D_0.dat".format(nrow, ncol)
    DATAOUT = "./bp2bin/gs_bin_data/gs_{}_{}_2D_0.dat.out".format(nrow, ncol)
  else:
    DATAIN = "./bp2bin/gs_bin_data/gs_{}_{}_{}_3D_0.dat".format(nrow, ncol, nfib)
    DATAOUT = "./bp2bin/gs_bin_data/gs_{}_{}_{}_3D_0.dat.out".format(nrow, ncol, nfib)
  cmd = ['../build/bin/mgard_check_cuda_and_cpu', str(1), str(DATAIN), str(DATAOUT),
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


def avg_run(nrow, ncol, nfib, opt, B, num_of_queues, num_runs):
  refactor_timing_results_all = []
  recompose_timing_results_all = []
  for i in range(num_runs):
    results = run_test(nrow, ncol, nfib, opt, B, num_of_queues)
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



########Global Configuration########
B = 16
num_runs = 3

########Run 2D All Size########
num_of_queues=1

####CPU-2D####
avg_run(33, 33, 1, -1, B, num_of_queues, num_runs)
avg_run(65, 65, 1, -1, B, num_of_queues, num_runs)
avg_run(129, 129, 1, -1, B, num_of_queues, num_runs)
avg_run(257, 257, 1, -1, B, num_of_queues, num_runs)

####GPU-2D####
avg_run(33, 33, 1, 3, B, num_of_queues, num_runs)
avg_run(65, 65, 1, 3, B, num_of_queues, num_runs)
avg_run(129, 129, 1, 3, B, num_of_queues, num_runs)
avg_run(257, 257, 1, 3, B, num_of_queues, num_runs)




########Run 3D All Size########
num_of_queues=32
####CPU-2D####
avg_run(33, 33, 33, -1, B, num_of_queues, num_runs)
avg_run(65, 65, 65, -1, B, num_of_queues, num_runs)
avg_run(129, 129, 129, -1, B, num_of_queues, num_runs)
avg_run(257, 257, 257, -1, B, num_of_queues, num_runs)

####GPU-2D####
avg_run(33, 33, 33, 3, B, num_of_queues, num_runs)
avg_run(65, 65, 65, 3, B, num_of_queues, num_runs)
avg_run(129, 129, 129, 3, B, num_of_queues, num_runs)
avg_run(257, 257, 257, 3, B, num_of_queues, num_runs)
