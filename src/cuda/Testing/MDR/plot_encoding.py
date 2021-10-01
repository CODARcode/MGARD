#!/usr/bin/python
import subprocess
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import math



SMALL_SIZE = 12
MEDIUM_SIZE = 18
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


def read_csv(filename):
  f = open(filename)
  r = csv.reader(f, delimiter=',')
  data = []
  header = 0
  for row in r:
    row_data = []
    if (header == 0):
      for token in row:
        row_data.append(float(token))
      data.append(row_data)
    else:
      header -= 1;
  return data

def plot_line(title, data, x_ticks, y_max, y_step, xlabel, ylabel, legend, output):
  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
  ax1.set_title(title)
  if (len(data[0]) > len(data[1])):
    x_idx = np.array(range(len(data[0])))
  else:
    x_idx = np.array(range(len(data[1])))

  y_idx = np.arange(0, y_max, y_step)
  
  style = ['b-s', 'g-o', 'b-v']
  ps = []
  for i in range(len(legend)):
    p, = ax1.plot(np.array(range(len(data[i]))), data[i], style[i], linewidth=4, markersize=15)
    ps.append(p)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(x_ticks)
  ax1.set_xlabel(xlabel)
  ax1.tick_params(axis='x', rotation=0)
  ax1.set_yticks(y_idx)
  ax1.set_yscale('log')
  # ax1.set_yticklabels([str(round(float(label), 2)) for label in y_idx])
  ax1.set_ylabel(ylabel)
  ax1.grid(which='major', axis='y')
  ax1.legend(tuple(ps), legend)
  plt.tight_layout()
  plt.savefig('{}.png'.format(output))


def plot_bar(title, data, x_ticks, y_max, y_step, xlabel, ylabel, legend, output):
  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
  bar_width = 0.25
  offset = 0
  ax1.set_title(title)
  if (len(data[0]) > len(data[1])):
    x_idx = np.array(range(len(data[0])))
  else:
    x_idx = np.array(range(len(data[1])))

  y_idx = np.arange(0, y_max, y_step)
  
  colors = ['b', 'g', 'r']
  ps = []
  for i in range(len(legend)):
    print np.array(range(len(data[i])))+offset
    p = ax1.bar(np.array(range(len(data[i])))+offset, data[i], color=colors[i], width=bar_width)
    ps.append(p)
    offset += bar_width

  ax1.set_xticks(x_idx + bar_width/2)
  ax1.set_xticklabels(x_ticks)
  ax1.set_xlabel(xlabel)
  ax1.tick_params(axis='x', rotation=0)
  ax1.set_yticks(y_idx)
  # ax1.set_yscale('log')
  ax1.set_yticklabels([str(round(float(label), 2)) for label in y_idx])
  ax1.set_ylabel(ylabel)
  ax1.grid(which='major', axis='y')
  ax1.legend(tuple(ps), legend)
  plt.tight_layout()
  plt.savefig('{}.png'.format(output))





def get_filename(encoding_type_bits, 
                 decoding_type_bits, 
                 n, 
                 encoding_num_bitplanes, 
                 decoding_num_bitplanes, 
                 BinaryType,
                 DataEncodingAlgorithm,
                 ErrorCollectingAlgorithm,
                 DataDecodingAlgorithm):
  filename = "encoding_perf_results/pref_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(encoding_type_bits, 
                                                 decoding_type_bits, 
                                                 n, 
                                                 encoding_num_bitplanes, 
                                                 decoding_num_bitplanes, 
                                                 BinaryType,
                                                 DataEncodingAlgorithm,
                                                 ErrorCollectingAlgorithm,
                                                 DataDecodingAlgorithm)
  print filename
  return filename;

def large_data_different_num_bitplanes(BinaryType):

  DataEncodingAlgorithm = 1
  ErrorCollectingAlgorithm = 1
  DataDecodingAlgorithm = 1

  cpu_encoding = []
  cpu_decoding = []
  gpu_encoding = []
  gpu_decoding = []

  for num_bitplane in range(1,32,1):
    csv_data = read_csv(get_filename(32, 32, 512*1024*1024, num_bitplane, num_bitplane, 
                          BinaryType, DataEncodingAlgorithm, ErrorCollectingAlgorithm, 
                          DataDecodingAlgorithm))
    cpu_encoding.append(csv_data[0][0])
    cpu_decoding.append(csv_data[0][1])
    gpu_encoding.append(csv_data[0][2])
    gpu_decoding.append(csv_data[0][3])

  print cpu_encoding
  print cpu_decoding

  print gpu_encoding
  print gpu_decoding

  x_ticks = []
  for num_bitplane in range(1,33,1):
    x_ticks.append("{}".format(num_bitplane))

  plot_line("Bitplane Encoding (Num. of Coefficients = $2^{29}$)", [cpu_encoding, gpu_encoding], x_ticks, 100, 5, 
              "Number of Encoding Bitplanes", "Throughput (GB/s)", ["CPU", "GPU"], "encoding_num_bitplanes_binarytype_{}".format(BinaryType))
  plot_line("Bitplane Decoding (Num. of Coefficients = $2^{29}$)", [cpu_decoding, gpu_decoding], x_ticks, 100, 5, 
              "Number of Decoding Bitplanes", "Throughput (GB/s)", ["CPU", "GPU"], "decoding_num_bitplanes_binarytype_{}".format(BinaryType))

def max_num_bitplane_different_data_sizes(BinaryType):

  DataEncodingAlgorithm = 1
  ErrorCollectingAlgorithm = 1
  DataDecodingAlgorithm = 1
  num_bitplane = 32

  cpu_encoding = []
  cpu_decoding = []
  gpu_encoding = []
  gpu_decoding = []

  for i in range(1,20,1):
    csv_data = read_csv(get_filename(32, 32, 2**i*1024, num_bitplane, num_bitplane, 
                          BinaryType, DataEncodingAlgorithm, ErrorCollectingAlgorithm, 
                          DataDecodingAlgorithm))
    cpu_encoding.append(csv_data[0][0])
    cpu_decoding.append(csv_data[0][1])
    gpu_encoding.append(csv_data[0][2])
    gpu_decoding.append(csv_data[0][3])

  print cpu_encoding
  print cpu_decoding

  print gpu_encoding
  print gpu_decoding

  x_ticks = []
  for i in range(10,30,1):
    x_ticks.append("$2^{" +str(i) + "}$")

  plot_line("Bitplane Encoding (Num. of bitplanes = 32)", [cpu_encoding, gpu_encoding], x_ticks, 100, 5, 
              "Number of Coefficients", "Throughput (GB/s)", ["CPU", "GPU"], "encoding_data_sizes_binarytype_{}".format(BinaryType))
  plot_line("Bitplane Decoding (Num. of bitplanes = 32)", [cpu_decoding, gpu_decoding], x_ticks, 100, 5, 
              "Number of Coefficients", "Throughput (GB/s)", ["CPU", "GPU"], "decoding_data_sizes_binarytype_{}".format(BinaryType))

def end_to_end():
  refactor = [[0, 10.8476,0], [0,2.25562,0]]
  reconstruct = [[4.32562, 3.18212, 3.06498], [0.769606, 0.304263, 0.249241]]
  plot_bar("Refactoring", refactor, [""], 15, 5, "", "Time (s)", ["CPU", "GPU"], "refactor")
  plot_bar("Progressive Reconstruction", reconstruct, ["125" ,"145", "158"], 5, 1, "PSNR", "Time (s)", ["CPU", "GPU"], "reconstuct")



# large_data_different_num_bitplanes(0)
# max_num_bitplane_different_data_sizes(0)
# large_data_different_num_bitplanes(1)
# max_num_bitplane_different_data_sizes(1)
end_to_end()

