#!/usr/bin/python
import numpy as np
import csv
import sys

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

def read_csv(filename):
  levels = read_levels(filename)
  kernels = read_kernel_names(filename)
  timing = read_timing(filename)
  return [levels, kernels, timing]

def write_csv(filename, data):
  file = open(filename, 'w')
  csv_writer = csv.writer(file)
  for i in range(len(data[0])):
    csv_writer.writerow([data[0][i], data[1][i], data[2][i]])

def get_refactor_csv_name(nrow, ncol, nfib, opt, B, num_of_queues):
  if (nfib == 1): # 2D
    if (opt == -1):
      return 'refactor_2D_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 0):
      return 'refactor_2D_cuda_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 1):
      return 'refactor_2D_cuda_cpt_l1_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 2):
      return 'refactor_2D_cuda_cpt_l2_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 3):
      return 'refactor_2D_cuda_cpt_l2_sm_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)

  else: # 3D
    if (opt == -1):
      return 'refactor_3D_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 3):
      return 'refactor_3D_cuda_cpt_l2_sm_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)

def get_recompose_csv_name(nrow, ncol, nfib, opt, B, num_of_queues):
  if (nfib == 1): # 2D
    if (opt == -1):
      return 'recompose_2D_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 0):
      return 'recompose_2D_cuda_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 1):
      return 'recompose_2D_cuda_cpt_l1_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 2):
      return 'recompose_2D_cuda_cpt_l2_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 3):
      return 'recompose_2D_cuda_cpt_l2_sm_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)

  else: # 3D
    if (opt == -1):
      return 'recompose_3D_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)
    if (opt == 3):
      return 'recompose_3D_cuda_cpt_l2_sm_{}_{}_{}_{}_{}.csv'.format(nrow, ncol, nfib, B, num_of_queues)

def aggre(n, nproc, dims, dev, platform, nrow, ncol, nfib, B, num_of_queues):  
  if (dev == 0) :
    f = platform + "/results-cpu/0-0/refactor_{}D.csv".format(dims)
    refac = read_csv(f)
    f = platform + "/results-cpu/0-0/recompose_{}D.csv".format(dims)
    recomp = read_csv(f)
  else:
    f = platform + "/results-gpu/0-0/refactor_{}D_cuda_cpt_l2_sm.csv".format(dims)
    refac = read_csv(f)
    f = platform + "/results-gpu/0-0/recompose_{}D_cuda_cpt_l2_sm.csv".format(dims)
    recomp = read_csv(f)

  refac_time = np.zeros(len(refac[2]))
  recomp_time = np.zeros(len(recomp[2]))

  for rank in range(nproc):
    refac_time_tmp = np.zeros(len(refac[2]))
    recomp_time_tmp = np.zeros(len(recomp[2]))
    for i in range(n/nproc):
      if (dev == 0):
        f =platform + "/results-cpu/{}-{}/refactor_{}D.csv".format(rank, i, dims)
        refac_time_tmp += np.array(read_csv(f)[2])
        f =platform + "/results-cpu/{}-{}/recompose_{}D.csv".format(rank, i, dims)
        recomp_time_tmp += np.array(read_csv(f)[2])
      else:
        f =platform + "/results-gpu/{}-{}/refactor_{}D_cuda_cpt_l2_sm.csv".format(rank, i, dims)
        refac_time_tmp += np.array(read_csv(f)[2])
        f =platform + "/results-gpu/{}-{}/recompose_{}D_cuda_cpt_l2_sm.csv".format(rank, i, dims)
        recomp_time_tmp += np.array(read_csv(f)[2])

      refac_time = np.maximum(refac_time_tmp, refac_time)
      recomp_time = np.maximum(recomp_time_tmp, recomp_time)

  if (dev == 0):
    output = get_refactor_csv_name(nrow, ncol, nfib, -1, B, num_of_queues)
    write_csv(platform + "/{}".format(output), [refac[0], refac[1], refac_time])
    output = get_recompose_csv_name(nrow, ncol, nfib, -1, B, num_of_queues)
    write_csv(platform + "/{}".format(output), [recomp[0], recomp[1], recomp_time])
  else:
    output = get_refactor_csv_name(nrow, ncol, nfib, 3, B, num_of_queues)
    write_csv(platform + "/{}".format(output), [refac[0], refac[1], refac_time])
    output = get_recompose_csv_name(nrow, ncol, nfib, 3, B, num_of_queues)
    write_csv(platform + "/{}".format(output), [recomp[0], recomp[1], recomp_time])




n = int(sys.argv[1])
nproc = int(sys.argv[2])
dims = int(sys.argv[3])
dev = int(sys.argv[4])
platform = sys.argv[5]
nrow = int(sys.argv[6])
ncol = int(sys.argv[7])
nfib = int(sys.argv[8])
B = int(sys.argv[9])
num_of_queues = int(sys.argv[10])

aggre(n, nproc, dims, dev, platform, nrow, ncol, nfib, B, num_of_queues)



