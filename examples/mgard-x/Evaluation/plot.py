import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 12
MEDIUM_SIZE = 25
BIGGER_SIZE = 14

# plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

Actual_error = ['$1e^{-4}$', '$1e^{-3}$', '$1e^{-2}$', '$1e^{-1}$']
# Types = ['CPU', 'NVIDIA V100 (High Speed, Low CR)', 'NVIDIA V100 (Mid. Speed, Mid, CR)', 'NVIDIA V100 (Low Speed, High CR)', 'AMD MI100 (High Speed, Low CR)', 'AMD MI100 (Low Speed, High CR)']
Types = ['CPU', 'NVIDIA V100 (High Speed Mode)', 'NVIDIA V100 (High Compress Mode)', 'AMD MI100 (High Speed Mode)', 'AMD MI100 (High Compress Mode)']
Color = ['teal', 'olivedrab', 'lawngreen', 'yellowgreen', 'maroon', 'lightcoral']


# Summit
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
          [0.0816041, 0.0862138, 0.0846691, 0.0855711], # original CPU (zstd)
          [3.15362, 3.23683, 3.54528, 3.53627], # X-CUDA (Huffman)
          [1.92262, 2.38614, 2.79213, 2.79624], # X-CUDA (Huffman + LZ4)
          [1.73869, 1.8509, 2.08465, 2.85483], # X-CUDA (Huffman + zstd)
          [3.88433, 3.82859, 3.97107, 4.22841], # X-HIP (Huffman)
          [1.61152, 2.09385, 2.35168, 3.68284], # X-HIP (Huffman + zstd)
        ],
        [ # Decompression Speed
          [0.0451107, 0.0625231, 0.0763279, 0.0774323], # original CPU (zstd)
          [4.74448, 5.2634, 6.40883, 6.58018], # X-CUDA (Huffman)
          [4.65419, 5.19161, 5.50165, 6.30292], # X-CUDA (Huffman + LZ4)
          [3.15769, 2.90878, 4.35225, 5.53827], # X-CUDA (Huffman + zstd)
          [4.15226, 4.15226, 6.51118, 8.34212], # X-HIP (Huffman)
          [2.59679, 2.09385, 3.88071, 6.56936], # X-HIP (Huffman + zstd)
        ]
      ]


# Workstation
# NYX = [ 
#         [ # Compress Ratio
#           [5.402201, 12.817980, 108.555351, 1886.299126], # original CPU (zstd)
#           [5.393630, 11.985457, 28.640875, 31.595436], # X-CUDA (Huffman)
#           [5.374151, 11.966424, 61.813310, 401.725900], # X-CUDA (Huffman + LZ4)
#           [5.398262, 12.465143, 93.543284, 1259.184434], # X-CUDA (Huffman + zstd)
#           [5.393630, 11.985457, 28.640875, 31.595436], # X-HIP (Huffman)
#           [5.398262, 12.465143, 93.543284, 1259.184434], # X-HIP (Huffman + zstd)
#         ],
#         [ # Compression Speed
#           [0.0546406, 0.0572094, 0.0611658, 0.0591082], # original CPU (zstd)
#           [4.8974, 6.5442, 6.9776, 6.56036], # X-CUDA (Huffman)
#           [2.87468, 3.85826, 4.98381, 4.81757], # X-CUDA (Huffman + LZ4)
#           [2.32737, 2.56573, 3.11475, 5.30809], # X-CUDA (Huffman + zstd)
#           [3.88433, 3.82859, 3.97107, 4.22841], # X-HIP (Huffman)
#           [1.61152, 2.09385, 2.35168, 3.68284], # X-HIP (Huffman + zstd)
#         ],
#         [ # Decompression Speed
#           [0.039652, 0.0454052, 0.0470239, 0.0467448], # original CPU (zstd)
#           [7.19133, 8.24694, 9.5593, 10.4115], # X-CUDA (Huffman)
#           [6.73909, 7.72308, 8.94575, 8.55307], # X-CUDA (Huffman + LZ4)
#           [3.45459, 3.43631, 6.04759, 7.23331], # X-CUDA (Huffman + zstd)
#           [4.15226, 4.15226, 6.51118, 8.34212], # X-HIP (Huffman)
#           [2.59679, 2.09385, 3.88071, 6.56936], # X-HIP (Huffman + zstd)
#         ]
#       ]

#Summit
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
          [0.0974297, 0.102583, 0.101956, 0.102627], # original CPU (zstd)
          [5.65168, 5.4199, 5.74517, 5.71633], # X-CUDA (Huffman)
          [3.62034, 4.15737, 4.78975, 4.95357], # X-CUDA (Huffman + LZ4)
          [1.30231, 1.70362, 2.58156, 4.84744], # X-CUDA (Huffman + zstd)
          [5.18314, 5.25402, 5.26588, 5.68074], # X-HIP (Huffman)
          [2.22757, 2.01097, 2.68003, 4.47563], # X-HIP (Huffman + zstd)
        ],
        [ # Decompression Speed
          [0.0609786, 0.0748752, 0.0872117, 0.0877305], # original CPU (zstd)
          [6.8038, 6.94433, 7.45138, 7.98317], # X-CUDA (Huffman)
          [6.42563, 6.69084, 6.99167, 7.81145], # X-CUDA (Huffman + LZ4)
          [3.56677, 3.49549, 4.90003, 7.16708], # X-CUDA (Huffman + zstd)
          [8.79956, 8.79956, 8.79956, 11.7825], # X-HIP (Huffman)
          [4.7613, 4.7613, 4.7613, 9.11193], # X-HIP (Huffman + zstd)
        ]
      ]

# Workstation
# XGC = [
#         [ # Compress Ratio
#           [11.252583, 26.619555, 110.304926, 2272.112982], # original CPU (zstd)
#           [8.990491, 15.437648, 30.223002, 62.203022], # X-CUDA (Huffman)
#           [9.165414, 17.057499, 42.389939, 497.727380], # X-CUDA (Huffman + LZ4)
#           [9.606351, 20.362018, 63.232526, 1163.757468], # X-CUDA (Huffman + zstd)
#           [8.990491, 15.437648, 30.223002, 62.203022], # X-HIP (Huffman)
#           [9.606351, 20.362018, 63.232526, 1163.757468], # X-HIP (Huffman + zstd)
#         ],
#         [ # Compression Speed
#           [0.056716, 0.0580109, 0.058607, 0.0589073], # original CPU (zstd)
#           [6.2098, 6.11959, 6.87476, 6.78347], # X-CUDA (Huffman)
#           [3.82614, 4.40076, 5.3331, 5.54344], # X-CUDA (Huffman + LZ4)
#           [1.36009, 1.91809, 2.83379, 5.33747], # X-CUDA (Huffman + zstd)
#           [5.18314, 5.25402, 5.26588, 5.68074], # X-HIP (Huffman)
#           [2.22757, 2.01097, 2.68003, 4.47563], # X-HIP (Huffman + zstd)
#         ],
#         [ # Decompression Speed
#           [0.0483185, 0.0522003, 0.0542078, 0.0548493], # original CPU (zstd)
#           [5.72933, 5.80409, 6.08945, 6.2023], # X-CUDA (Huffman)
#           [5.65653, 5.55401, 5.84085, 5.85516], # X-CUDA (Huffman + LZ4)
#           [3.15508, 3.2697, 4.46122, 5.55407], # X-CUDA (Huffman + zstd)
#           [8.79956, 8.79956, 8.79956, 11.7825], # X-HIP (Huffman)
#           [4.7613, 4.7613, 4.7613, 9.11193], # X-HIP (Huffman + zstd)
#         ]
#       ]

#Summit
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
          [0.0684779, 0.0721772, 0.0728979, 0.0741929], # original CPU (zstd)
          [2.47057, 2.68165, 2.72084, 2.73935], # X-CUDA (Huffman)
          [1.21504, 1.4168, 1.57178, 1.60871], # X-CUDA (Huffman + LZ4)
          [1.12718, 1.12718, 0.970576, 0.970576], # X-CUDA (Huffman + zstd)
          [1.63604, 1.65282, 1.69286, 1.77386], # X-HIP (Huffman)
          [0.841459, 0.870178, 1.05042, 1.35576], # X-HIP (Huffman + zstd)
        ],
        [ # Decompression Speed
          [0.0396557, 0.0542099, 0.0676658, 0.0675658], # original CPU (zstd)
          [1.79356, 2.03, 2.63356, 3.52791], # X-CUDA (Huffman)
          [1.12718, 1.12718, 2.58144, 3.38406], # X-CUDA (Huffman + LZ4)
          [1.17044, 1.60876, 0.970576, 0.970576], # X-CUDA (Huffman + zstd)
          [1.51667, 1.51667, 1.51667, 2.16075], # X-HIP (Huffman)
          [0.893723, 0.893723, 1.11982, 1.85127], # X-HIP (Huffman + zstd)
        ]
      ]

# Workstation
# E3SM = [
#         [ # Compress Ratio
#           [5.176808, 11.524835, 71.125502, 4880.775704], # original CPU (zstd)
#           [3.548201, 5.652103, 12.846273, 28.434430], # X-CUDA (Huffman)
#           [3.545491, 5.659909, 13.964576, 83.733901], # X-CUDA (Huffman + LZ4)
#           [3.586439, 5.841004, 16.486520, 147.163732], # X-CUDA (Huffman + zstd)
#           [3.548201, 5.652103, 12.846273, 28.434430], # X-HIP (Huffman)
#           [3.586439, 5.841004, 16.486520, 147.163732], # X-HIP (Huffman + zstd)
#         ],
#         [ # Compression Speed
#           [0.0516544, 0.0532094, 0.0529181, 0.053668], # original CPU (zstd)
#           [3.30916, 3.19597, 3.17304, 3.27932], # X-CUDA (Huffman)
#           [1.33956, 1.93694, 2.47586, 2.58751], # X-CUDA (Huffman + LZ4)
#           [1.2768, 1.182, 1.11247, 2.32372], # X-CUDA (Huffman + zstd)
#           [1.63604, 1.65282, 1.69286, 1.77386], # X-HIP (Huffman)
#           [0.841459, 0.870178, 1.05042, 1.35576], # X-HIP (Huffman + zstd)
#         ],
#         [ # Decompression Speed
#           [0.0397186, 0.0462312, 0.049839, 0.050244], # original CPU (zstd)
#           [2.07613, 2.42887, 3.3552, 4.52448], # X-CUDA (Huffman)
#           [2.37263, 2.87235, 3.56927, 5.12888], # X-CUDA (Huffman + LZ4)
#           [1.17044, 1.60876, 2.29395, 3.3554], # X-CUDA (Huffman + zstd)
#           [1.51667, 1.51667, 1.51667, 2.16075], # X-HIP (Huffman)
#           [0.893723, 0.893723, 1.11982, 1.85127], # X-HIP (Huffman + zstd)
#         ]
#       ]


def plot_bar(data, title, output, plot_legend):
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
  width = 0.15
  x_idx = np.array(range(0, len(Actual_error), 1))  
  x_ticks = Actual_error
  y_idx = np.array(range(0, 12, 1))  

  tlt = fig.suptitle(title)

  # bars = []
  # for i in range(6):
  #   p = ax[0].bar(x_idx+width*i, data[0][i], width)
  #   bars.append(p)

  # ax[0].set_xticks(x_idx+width*3)
  # ax[0].set_xticklabels(x_ticks)
  # # ax1.set_xlabel("Kernels")
  # ax[0].tick_params(axis='x', rotation=0)
  # # ax1.set_yticks(y_idx)
  # ax[0].set_yscale('log')
  # # ax[0].set_title(title)
  # ax[0].set_ylabel('Compression Ratio')
  # ax[0].grid(which='major', axis='y')

  bars = []
  offset = 0
  for i in [0, 1, 3, 4, 5]:#range(6):
    p = ax[0].bar(x_idx+offset, data[1][i], width, color = Color[i])
    offset = offset + width
    bars.append(p)

  ax[0].set_xticks(x_idx+width*3)
  ax[0].set_xticklabels(x_ticks)
  # ax1.set_xlabel("Kernels")
  ax[0].tick_params(axis='x', rotation=0)
  ax[0].set_yticks(y_idx)
  ax[0].set_yscale('linear')
  # ax[0].set_title(title)
  ax[0].set_ylabel('Compression (GB/s)')
  ax[0].grid(which='major', axis='y')

  bars = []
  offset = 0
  for i in [0, 1, 3, 4, 5]:
    p = ax[1].bar(x_idx+offset, data[2][i], width, color = Color[i])
    offset = offset + width
    bars.append(p)

  ax[1].set_xticks(x_idx+width*3)
  ax[1].set_xticklabels(x_ticks)
  # ax1.set_xlabel("Kernels")
  ax[1].tick_params(axis='x', rotation=0)
  ax[1].set_yticks(y_idx)
  ax[1].set_yscale('linear')
  # ax[1].set_title(title)
  ax[1].set_ylabel('Decompression (GB/s)')
  ax[1].grid(which='major', axis='y')


  if (plot_legend):
    x0, y0, width, height = 0, 1.4, 0, 0
    lgd = fig.legend(tuple(bars), Types, loc = 'upper left', ncol=2, bbox_to_anchor=(x0, y0, width, height))
    plt.tight_layout()
    plt.savefig('{}.png'.format(output), bbox_extra_artists=(lgd, tlt), bbox_inches='tight')
  else:
    plt.tight_layout()
    plt.savefig('{}.png'.format(output), bbox_extra_artists=(tlt,), bbox_inches='tight')


plot_bar(E3SM, "E3SM temperature (FP32/3D/143 MB)", "E3SM", True)
plot_bar(NYX, "EXASKY velocity (FP32/3D/512 MB)", "NYX", False)
plot_bar(XGC, "XGC (FP64/5D/1.5 GB)", "XGC", False)



def plot_bar_cr(title, output, plot_legend):
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
  width = 0.15
  x_idx = np.array(range(0, 3, 1))  
  x_ticks = ['E3SM', 'EXASKY', 'XGC']
  y_idx = [1, 10, 100, 1000, 10000]  

  tlt = fig.suptitle(title)

  bars = []
  offset = 0
  for i in [0, 3, 5]:
    data = [E3SM[0][i][3], NYX[0][i][3], XGC[0][i][3]]
    print(data)
    p = ax.bar(x_idx+offset, data, width, color = Color[i])
    offset = offset + width
    bars.append(p)

  ax.set_xticks(x_idx+width*1)
  ax.set_xticklabels(x_ticks)
  # .set_xlabel("Kernels")
  ax.tick_params(axis='x', rotation=0)
  
  ax.set_yscale('log')
  ax.set_yticks(y_idx)
  # ax[0].set_title(title)
  ax.set_ylabel('Compression Ratio')
  ax.grid(which='major', axis='y')

  if (plot_legend):
    x0, y0, width, height = 0, 1.2, 0, 0
    lgd = fig.legend(tuple(bars), ['CPU', 'NVIDIA V100', 'AMD MI100'], loc = 'upper left', ncol=2, bbox_to_anchor=(x0, y0, width, height))
    plt.tight_layout()
    plt.savefig('{}.png'.format(output), bbox_extra_artists=(lgd, tlt), bbox_inches='tight')
  else:
    plt.tight_layout()
    plt.savefig('{}.png'.format(output), bbox_extra_artists=(tlt,), bbox_inches='tight')


plot_bar_cr("", "E3SM_CR", True)