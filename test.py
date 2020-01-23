#!/usr/bin/python
import subprocess
import csv
import numpy as np
import matplotlib.pyplot as plt

def read_timing(filename):
  file = open(filename)
  csv_reader = csv.reader(file)
  results = []
  for row in csv_reader:
    results.append(float(row[1]))
  return results

def read_kernel_names(filename):
  file = open(filename)
  csv_reader = csv.reader(file)
  results = []
  for row in csv_reader:
    results.append(row[0])
  return results

def run_cpu_version(m, n, k, tol, s):
  cmd = ['./build/bin/mgard_check','./data/data_600x400_orig','./data/data_600x400_orig.mgard', str(m), str(n), str(k), str(tol), str(s)]
  subprocess.call(' '.join(cmd), shell = True)
  prep_2D = read_timing('prep_2D.csv')
  refactor_2D = read_timing('refactor_2D.csv')
  recompose_2D = read_timing('recompose_2D.csv')
  postp_2D = read_timing('postp_2D.csv')
  return prep_2D + refactor_2D + recompose_2D + postp_2D


def run_cuda_version(m, n, k, tol, s, opt):
  cmd = ['./build/bin/mgard_check_cuda','./data/data_600x400_orig','./data/data_600x400_orig.mgard', str(m), str(n), str(k), str(tol), str(s), str(opt)]
  subprocess.call(' '.join(cmd), shell = True)
  prep_2D_cuda = read_timing('prep_2D_cuda.csv')
  if (opt == 0):
    refactor_2D_cuda = read_timing('refactor_2D_cuda.csv')
  elif (opt == 1):
    refactor_2D_cuda = read_timing('refactor_2D_cuda_o{}.csv'.format(opt))
    refactor_2D_cuda = refactor_2D_cuda[2:]
  elif (opt == 2):
    refactor_2D_cuda = read_timing('refactor_2D_cuda_o{}.csv'.format(opt))
    refactor_2D_cuda = refactor_2D_cuda[4:]
  recompose_2D_cuda = read_timing('recompose_2D_cuda.csv')
  postp_2D_cuda = read_timing('postp_2D_cuda.csv')
  return prep_2D_cuda + refactor_2D_cuda + recompose_2D_cuda + postp_2D_cuda

def get_kernel_names(m, n, k, tol, s):
  cmd = ['./build/bin/mgard_check','./data/data_600x400_orig','./data/data_600x400_orig.mgard', str(m), str(n), str(k), str(tol), str(s)]
  subprocess.call(' '.join(cmd), shell = True)
  prep_2D = read_kernel_names('prep_2D.csv')
  refactor_2D = read_kernel_names('refactor_2D.csv')
  recompose_2D = read_kernel_names('recompose_2D.csv')
  postp_2D = read_kernel_names('postp_2D.csv')
  return prep_2D + refactor_2D + recompose_2D + postp_2D


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



warming_up_gpu=3
for i in range(warming_up_gpu):
  ret = run_cuda_version(600, 400, 1, 0.01, 0, 0)

kernel_names = get_kernel_names(600, 400, 1, 0.01, 0)


num_runs = 10
cpu_results = []
cuda_results = []
cuda_o1_results = []
cuda_o2_results = []
for i in range(num_runs):
  ret = run_cpu_version(600, 400, 1, 0.01, 0)
  cpu_results.append(ret)

  ret = run_cuda_version(600, 400, 1, 0.01, 0, 0)
  cuda_results.append(ret)

  ret = run_cuda_version(600, 400, 1, 0.01, 0, 1)
  cuda_o1_results.append(ret)

  ret = run_cuda_version(600, 400, 1, 0.01, 0, 2)
  cuda_o2_results.append(ret)

cpu_avg = np.average(np.array(cpu_results), axis=0)
print(cpu_avg)

cuda_avg = np.average(np.array(cuda_results), axis=0)
print(cuda_avg)

cuda_o1_avg = np.average(np.array(cuda_o1_results), axis=0)
print(cuda_o1_avg)

cuda_o2_avg = np.average(np.array(cuda_o2_results), axis=0)
print(cuda_o2_avg)

cpu_speedup = np.full(cpu_avg.shape, 1)
cuda_speedup = cpu_avg/cuda_avg
cuda_o1_speedup = cpu_avg/cuda_o1_avg
cuda_o2_speedup = cpu_avg/cuda_o2_avg
print(cpu_speedup)
print(cuda_speedup)

plot_cuda_speedup(kernel_names, cpu_speedup, cuda_speedup, cuda_o1_speedup, cuda_o2_speedup)


