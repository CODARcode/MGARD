#!/usr/bin/python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def f(x):
	return (x-2)*(x-3)

idx1 = [0, 1, 2, 3, 4]
idx2 = [0, 2, 4]
idx3 = [0, 4]

correction2 = np.array([-0.5, -0.5, -0.5])
correction1 = np.array([-2, -2])

def get_org_func():
	result = [] 
	for x in idx1:
		result.append(f(x))
	return np.array(result)

org = get_org_func()

qu = org
def do_interpolantion(qu):
	result = [] 
	for i in range(len(qu)):
		if (i % 2 == 0):
			result.append(qu[i])
		else:
			result.append(0.5 * (qu[i-1] + qu[i+1]))
	return np.array(result)

def add_correction(qu, correction):
	result = []
	for i in range(len(qu)):
		if (i % 2 == 0):
			result.append(qu[i] + correction[i/2])
	return np.array(result)

after_interpolantion = do_interpolantion(qu)
after_correction = add_correction(after_interpolantion, correction2)

print(qu)
print(after_interpolantion)
print(after_correction)

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6,6))


p1, = ax1.plot(idx1, qu, color = 'gray', marker = 'o', linestyle='-', linewidth=2)
p2, = ax1.plot(idx1, after_interpolantion, color = 'gray', linestyle='--')
p3, = ax1.plot(idx2, after_correction, color = 'black', marker = 's', linestyle='-', linewidth=2)
# p4, = ax1.annotate('', xy=(1, qu[1]), xycoords='data',
# 										xytext=(1, after_interpolantion[1]), textcoords='data',
# 										arrowprops={'arrowstyle': '<->'})

ax1.annotate('', xy=(1, qu[1]), xytext=(1, after_interpolantion[1]), arrowprops=dict(arrowstyle="|-|", color='green', linewidth = 1))
ax1.annotate('', xy=(3, qu[3]), xytext=(3, after_interpolantion[3]), arrowprops=dict(arrowstyle="|-|", color='green', linewidth = 1))
# ax1.annotate('Coefficient', xy=(1, 0.5*(qu[1] + after_interpolantion[1])), xycoords='data', xytext=(5, 0), textcoords='offset points')
# ax1.annotate('Coefficient', xy=(3, 0.5*(qu[3] + after_interpolantion[3])), xycoords='data', xytext=(5, 0), textcoords='offset points', Text())
ax1.annotate(r'Coefficient', xy=(1, 0.2+0.5*(qu[1] + after_interpolantion[1])), xytext=(60,25), 
         textcoords='offset points', ha='center', va='bottom',color='blue',
         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
         arrowprops=dict(arrowstyle='->', linestyle = '-', connectionstyle='arc3,rad=-0.3', 
                            color='g'))

ax1.annotate(r'Coefficient', xy=(3, 0.2+0.5*(qu[3] + after_interpolantion[3])), xytext=(-60,25), 
         textcoords='offset points', ha='center', va='bottom',color='blue',
         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
         arrowprops=dict(arrowstyle='->', linestyle = '-', connectionstyle='arc3,rad=0.3', 
                            color='g'))


ax1.annotate('', xy=(0, qu[0]), xytext=(0, after_correction[0]), arrowprops=dict(arrowstyle="<|-", color='blue', mutation_scale=15, linewidth = 2))
ax1.annotate('', xy=(2, qu[2]), xytext=(2, after_correction[1]), arrowprops=dict(arrowstyle="<|-", color='blue', mutation_scale=15))
ax1.annotate('', xy=(4, qu[4]), xytext=(4, after_correction[2]), arrowprops=dict(arrowstyle="<|-", color='blue', mutation_scale=15))

ax1.annotate(r'Add Correction', xy=(0, 0.5*(qu[0] + after_correction[0])), xytext=(60,10), 
         textcoords='offset points', ha='center', va='bottom',color='blue',
         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
         arrowprops=dict(arrowstyle='->', linestyle = '-', connectionstyle='arc3,rad=-0.3', 
                            color='b'))
ax1.annotate(r'Add Correction', xy=(2, 0.5*(qu[2] + after_correction[1])), xytext=(-80,15), 
         textcoords='offset points', ha='center', va='bottom',color='blue',
         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
         arrowprops=dict(arrowstyle='->', linestyle = '-', connectionstyle='arc3,rad=-0.3', 
                            color='b'))

ax1.annotate(r'Add Correction', xy=(4, 0.5*(qu[4] + after_correction[2])), xytext=(-80,15), 
         textcoords='offset points', ha='center', va='bottom',color='blue',
         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
         arrowprops=dict(arrowstyle='->', linestyle = '-', connectionstyle='arc3,rad=-0.3', 
                            color='b'))




ax1.set_xticks(idx1)
ax1.set_xticklabels(idx1)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend(tuple([p1, p2, p3]), ['$Q_2u$ (Original Data)', 'Linear Interpolation', '$Q_1u$'])
plt.tight_layout()
plt.savefig('example_l2.png', bbox_inches='tight')

qu = after_correction
after_interpolantion = do_interpolantion(qu)
after_correction = add_correction(after_interpolantion, correction1)

print(qu)
print(after_interpolantion)
print(after_correction)


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6,6))

#p0, = ax1.plot(idx1, org, color = 'gray', marker = 'o', linestyle='-', linewidth=2)
p1, = ax1.plot(idx2, qu, color = 'gray', marker = 'o', linestyle='-', linewidth=2)
p2, = ax1.plot(idx2, after_interpolantion, color = 'gray', linestyle='--')
p3, = ax1.plot(idx3, after_correction, color = 'black', marker = 's', linestyle='-', linewidth=2)


ax1.annotate('', xy=(2, qu[1]), xytext=(2, after_interpolantion[1]), arrowprops=dict(arrowstyle="|-|", color='green', linewidth = 1))

ax1.annotate(r'Coefficient', xy=(2, 1+0.5*(qu[1] + after_interpolantion[1])), xytext=(80,35), 
         textcoords='offset points', ha='center', va='bottom',color='blue',
         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
         arrowprops=dict(arrowstyle='->', linestyle = '-', connectionstyle='arc3,rad=-0.3', 
                            color='g'))


ax1.annotate('', xy=(0, qu[0]), xytext=(0, after_correction[0]), arrowprops=dict(arrowstyle="<|-", color='blue', mutation_scale=15, linewidth = 2))
ax1.annotate('', xy=(4, qu[2]), xytext=(4, after_correction[1]), arrowprops=dict(arrowstyle="<|-", color='blue', mutation_scale=15))

ax1.annotate(r'Add Correction', xy=(0, 0.5*(qu[0] + after_correction[0])), xytext=(80,45), 
         textcoords='offset points', ha='center', va='bottom',color='blue',
         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
         arrowprops=dict(arrowstyle='->', linestyle = '-', connectionstyle='arc3,rad=-0.3', 
                            color='b'))
ax1.annotate(r'Add Correction', xy=(4, 0.5*(qu[2] + after_correction[1])), xytext=(-80,45), 
         textcoords='offset points', ha='center', va='bottom',color='blue',
         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
         arrowprops=dict(arrowstyle='->', linestyle = '-', connectionstyle='arc3,rad=-0.3', 
                            color='b'))


ax1.set_xticks(idx1)
ax1.set_xticklabels(idx1)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend(tuple([p1, p2, p3]), ['$Q_1u $', 'Linear Interpolation', '$Q_0u$'])
plt.tight_layout()
plt.savefig('example_l1.png', bbox_inches='tight')
