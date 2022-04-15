import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import numpy as np
import sys
import csv

filename = sys.argv[1]
dim1 = int(sys.argv[2])
dim0 = int(sys.argv[3])


file = open(filename)
reader = csv.reader(file)

max_error = 0.0
min_error = sys.float_info.max

for row in reader:
    max_error = max(max_error, float(row[4]))
    min_error = min(min_error, float(row[4]))

print("max_error", max_error)
print("min_error", min_error)

fig, ax = plt.subplots()

# ax.plot([0, 10],[0, 10])

x_idx = np.array(range(0, dim0, 1))
y_idx = np.array(range(0, dim1, 1))  

ax.set_xticks(x_idx)
ax.set_yticks(y_idx)


cmap = plt.get_cmap('summer', 10)
# norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
norm = matplotlib.colors.Normalize(vmin=60, vmax=60*2)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
mapper.set_array([])

file.seek(0)
for row in reader:
    start = int(row[0])
    end = int(row[1])
    width = int(row[2]) - int(row[0])
    height = int(row[3]) - int(row[1])
    error = float(row[4])
    feature = int(row[5])
    # print("error: ", error)
    if (feature == 1):
        ax.add_patch(Rectangle((start, end), width, height,
                     edgecolor = 'black',
                     facecolor = mapper.to_rgba(error),
                     fill=True,
                     lw=1))
    else:
        ax.add_patch(Rectangle((start, end), width, height,
                     edgecolor = 'none',
                     facecolor = mapper.to_rgba(error),
                     fill=False,
                     lw=1))



plt.colorbar(mapper, label="L_inf Error")
plt.show()