import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 12
MEDIUM_SIZE = 18
BIGGER_SIZE = 14

# plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

Actual_error = ['$1e^{-4}$', '$1e^{-3}$', '$1e^{-2}$', '$1e^{-1}$']
Types = ['Original CPU', 'CUDA (L0)', 'CUDA (L1)', 'CUDA (L2)', 'HIP (L0)', 'HIP (L2)']
NYX = [
        [ # Compress Ratio
          [5.402201, 12.817980, 108.555351, 1886.299126], # original CPU (zstd)
          [5.393630, 11.985457, 28.640875, 31.595436], # X-CUDA (Huffman)
          [5.374151, 11.966424, 61.813310, 401.725900], # X-CUDA (Huffman + LZ4)
          [5.398262, 12.465143, 93.543284, 1259.184434], # X-CUDA (Huffman + zstd)
          [5.393630, 11.985457, 28.640875, 31.595436], # X-HIP (Huffman)
          [5.398262, 12.465143, 93.543284, 1259.184434], # X-HIP (Huffman + zstd)
        ],
        [ # Compression Speed
          [0.0546406, 0.0572094, 0.0611658, 0.0591082], # original CPU (zstd)
          [4.8974, 6.5442, 6.9776, 6.56036], # X-CUDA (Huffman)
          [2.87468, 3.85826, 4.98381, 4.81757], # X-CUDA (Huffman + LZ4)
          [2.32737, 2.56573, 3.11475, 5.30809], # X-CUDA (Huffman + zstd)
          [3.88433, 3.82859, 3.97107, 4.22841], # X-HIP (Huffman)
          [1.61152, 2.09385, 2.35168, 3.68284], # X-HIP (Huffman + zstd)
        ],
        [ # Decompression Speed
          [0.039652, 0.0454052, 0.0470239, 0.0467448], # original CPU (zstd)
          [7.19133, 8.24694, 9.5593, 10.4115], # X-CUDA (Huffman)
          [6.73909, 7.72308, 8.94575, 8.55307], # X-CUDA (Huffman + LZ4)
          [3.45459, 3.43631, 6.04759, 7.23331], # X-CUDA (Huffman + zstd)
          [4.15226, 4.15226, 6.51118, 8.34212], # X-HIP (Huffman)
          [2.59679, 2.09385, 3.88071, 6.56936], # X-HIP (Huffman + zstd)
        ]
      ]


XGC = [
        [ # Compress Ratio
          [11.252583, 26.619555, 110.304926, 2272.112982], # original CPU (zstd)
          [8.990491, 15.437648, 30.223002, 62.203022], # X-CUDA (Huffman)
          [9.165414, 17.057499, 42.389939, 497.727380], # X-CUDA (Huffman + LZ4)
          [9.606351, 20.362018, 63.232526, 1163.757468], # X-CUDA (Huffman + zstd)
          [8.990491, 15.437648, 30.223002, 62.203022], # X-HIP (Huffman)
          [9.606351, 20.362018, 63.232526, 1163.757468], # X-HIP (Huffman + zstd)
        ],
        [ # Compression Speed
          [0.056716, 0.0580109, 0.058607, 0.0589073], # original CPU (zstd)
          [6.2098, 6.11959, 6.87476, 6.78347], # X-CUDA (Huffman)
          [3.82614, 4.40076, 5.3331, 5.54344], # X-CUDA (Huffman + LZ4)
          [1.36009, 1.91809, 2.83379, 5.33747], # X-CUDA (Huffman + zstd)
          [5.18314, 5.25402, 5.26588, 5.68074], # X-HIP (Huffman)
          [2.22757, 2.01097, 2.68003, 4.47563], # X-HIP (Huffman + zstd)
        ],
        [ # Decompression Speed
          [0.0483185, 0.0522003, 0.0542078, 0.0548493], # original CPU (zstd)
          [5.72933, 5.80409, 6.08945, 6.2023], # X-CUDA (Huffman)
          [5.65653, 5.55401, 5.84085, 5.85516], # X-CUDA (Huffman + LZ4)
          [3.15508, 3.2697, 4.46122, 5.55407], # X-CUDA (Huffman + zstd)
          [8.79956, 8.79956, 8.79956, 11.7825], # X-HIP (Huffman)
          [4.7613, 4.7613, 4.7613, 9.11193], # X-HIP (Huffman + zstd)
        ]
      ]

E3SM = [
        [ # Compress Ratio
          [5.176808, 11.524835, 71.125502, 4880.775704], # original CPU (zstd)
          [3.548201, 5.652103, 12.846273, 28.434430], # X-CUDA (Huffman)
          [3.545491, 5.659909, 13.964576, 83.733901], # X-CUDA (Huffman + LZ4)
          [3.586439, 5.841004, 16.486520, 147.163732], # X-CUDA (Huffman + zstd)
          [3.548201, 5.652103, 12.846273, 28.434430], # X-HIP (Huffman)
          [3.586439, 5.841004, 16.486520, 147.163732], # X-HIP (Huffman + zstd)
        ],
        [ # Compression Speed
          [0.0516544, 0.0532094, 0.0529181, 0.053668], # original CPU (zstd)
          [3.30916, 3.19597, 3.17304, 3.27932], # X-CUDA (Huffman)
          [1.33956, 1.93694, 2.47586, 2.58751], # X-CUDA (Huffman + LZ4)
          [1.2768, 1.182, 1.11247, 2.32372], # X-CUDA (Huffman + zstd)
          [1.63604, 1.65282, 1.69286, 1.77386], # X-HIP (Huffman)
          [0.841459, 0.870178, 1.05042, 1.35576], # X-HIP (Huffman + zstd)
        ],
        [ # Decompression Speed
          [0.0397186, 0.0462312, 0.049839, 0.050244], # original CPU (zstd)
          [2.07613, 2.42887, 3.3552, 4.52448], # X-CUDA (Huffman)
          [2.37263, 2.87235, 3.56927, 5.12888], # X-CUDA (Huffman + LZ4)
          [1.17044, 1.60876, 2.29395, 3.3554], # X-CUDA (Huffman + zstd)
          [1.51667, 1.51667, 1.51667, 2.16075], # X-HIP (Huffman)
          [0.893723, 0.893723, 1.11982, 1.85127], # X-HIP (Huffman + zstd)
        ]
      ]


def plot_bar(data, title, output):
  fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
  width = 0.15
  x_idx = np.array(range(0, len(Actual_error), 1))  
  x_ticks = Actual_error

  bars = []
  for i in range(6):
    p = ax[0].bar(x_idx+width*i, data[0][i], width)
    bars.append(p)

  ax[0].set_xticks(x_idx+width*3)
  ax[0].set_xticklabels(x_ticks)
  # ax1.set_xlabel("Kernels")
  ax[0].tick_params(axis='x', rotation=0)
  # ax1.set_yticks(y_idx)
  ax[0].set_yscale('log')
  # ax[0].set_title(title)
  ax[0].set_ylabel('Compression Ratio')
  ax[0].grid(which='major', axis='y')
  # ax[0].legend(tuple(bars), Types,
  #   loc='upper left', bbox_to_anchor=(0,-0.05), ncol=3)




  # plt.tight_layout()
  # plt.savefig('{}.png'.format(output))

  bars = []
  for i in range(6):
    p = ax[1].bar(x_idx+width*i, data[1][i], width)
    bars.append(p)

  ax[1].set_xticks(x_idx+width*3)
  ax[1].set_xticklabels(x_ticks)
  # ax1.set_xlabel("Kernels")
  ax[1].tick_params(axis='x', rotation=0)
  # ax1.set_yticks(y_idx)
  ax[1].set_yscale('linear')
  ax[1].set_title(title)
  ax[1].set_ylabel('Compression Throughput (GB/s)')
  ax[1].grid(which='major', axis='y')
  # ax[1].legend(tuple(bars), Types,
  #   loc='upper left', bbox_to_anchor=(0,-0.15), ncol=2)
  # plt.tight_layout()
  # plt.savefig('{}.png'.format(output))

  bars = []
  for i in range(6):
    p = ax[2].bar(x_idx+width*i, data[2][i], width)
    bars.append(p)

  ax[2].set_xticks(x_idx+width*3)
  ax[2].set_xticklabels(x_ticks)
  # ax1.set_xlabel("Kernels")
  ax[2].tick_params(axis='x', rotation=0)
  # ax1.set_yticks(y_idx)
  ax[2].set_yscale('linear')
  # ax[2].set_title(title)
  ax[2].set_ylabel('Decompression Throughput (GB/s)')
  ax[2].grid(which='major', axis='y')
  # ax[2].legend(tuple(bars), Types,
  #   loc='upper left', bbox_to_anchor=(0,-0.15), ncol=2)

  x0, y0, width, height = 0.5, 0.05, 0, 0

  lgd = fig.legend(tuple(bars), Types, loc = 'upper center', ncol=6, bbox_to_anchor=(x0, y0, width, height))


  plt.tight_layout()
  plt.savefig('{}.png'.format(output), bbox_extra_artists=(lgd,), bbox_inches='tight')



plot_bar(NYX, "EXASKA velocity (FP32/3D/512 MB)", "NYX_CR")
# plot_bar(NYX, "Compression Throughput (GB/s)", 1, 'linear', "EXASKA velocity (FP32/3D/512 MB)", "NYX_CP")
# plot_bar(NYX, "Decompression Throughput (GB/s)", 2, 'linear', "EXASKA velocity (FP32/3D/512 MB)", "NYX_DC")

plot_bar(XGC, "XGC (FP64/5D/1.5 GB)", "XGC_CR")
# plot_bar(XGC, "Compression Throughput (GB/s)", 1, 'linear', "XGC (FP64/5D/1.5 GB)", "XGC_CP")
# plot_bar(XGC, "Decompression Throughput (GB/s)", 2, 'linear', "XGC (FP64/5D/1.5 GB)", "XGC_DC")

plot_bar(E3SM, "E3SM temperature (FP32/3D/143 MB)", "E3SM_CR")
# plot_bar(E3SM, "Compression Throughput (GB/s)", 1, 'linear', "E3SM temperature (FP32/3D/143 MB)", "E3SM_CP")
# plot_bar(E3SM, "Decompression Throughput (GB/s)", 2, 'linear', "E3SM temperature (FP32/3D/143 MB)", "E3SM_DC")