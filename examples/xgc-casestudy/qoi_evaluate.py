import numpy as np
import sys
sys.path.append('/ccs/home/gongq/andes/indir/lib/python3.7/site-packages/adios2/')
import adios2 as ad2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import os
import subprocess

def relative_abs_error(x, y):
    """
    relative L-inf error: max(|x_i - y_i|)/max(|x_i|)
    """
    assert(x.shape == y.shape)
    absv    = np.abs(x-y)
    maxv = np.max(x)
    minv = np.min(x)
    return (absv/(maxv - minv))
 
def relative_ptw_abs_error(x, y):
    """
    relative point-wise L-inf error: max(|x_i - y_i|/|x_i|)
    """
    assert(x.shape == y.shape)
    absv = np.abs(x-y)
    return (absv/x) 

def rmse_error(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2)
    return np.sqrt(mse)

def relative_rmse_error(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2)
    maxv = np.max(x)
    minv = np.min(x)
    return np.sqrt(mse)/(maxv - minv)

def compute_diff(name, x, x_):
    assert(x.shape == x_.shape)
    rel_err = relative_abs_error(x, x_)
    l2_err  = rmse_error(x, x_) 
    if (len(x.shape)==4):
        gb_L_inf = np.max(rel_err, axis=(-1,-2))
    else:
        gb_L_inf = rel_err
        
    print("{}, shape = {}: L-inf error = {}, RMSE = {}, NRMSE = {}, maxv = {}, minv = {}".format(name, x.shape, np.max(gb_L_inf), l2_err, l2_err/(np.max(x)-np.min(x)), np.max(x), np.min(x)))

exdir = '/gpfs/alpine/proj-shared/csc143/jyc/summit/xgc-deeplearning/d3d_coarse_v2/'
with ad2.open(exdir+'xgc.mesh.bp', 'r') as f:
# psi_surf: psi value of each surface
# surf_len: # of nodes of each surface
# surf_idx: list of node index of each surface
    nnodes = int(f.read('n_n', ))
    ncells = int(f.read('n_t', ))
    rz = f.read('rz')
    conn = f.read('nd_connect_list')
    psi = f.read('psi')
    nextnode = f.read('nextnode')
    epsilon = f.read('epsilon')
    node_vol = f.read('node_vol')
    node_vol_nearest = f.read('node_vol_nearest')
    psi_surf = f.read('psi_surf')
    surf_idx = f.read('surf_idx')
    surf_len = f.read('surf_len')

r = rz[:,0]
z = rz[:,1]
print (nnodes)

with ad2.open(exdir + 'restart_dir/xgc.f0.00700.bp', 'r') as f:
    f0_f = f.read('i_f')
f0_f = np.moveaxis(f0_f, 1, 2)

## module for xgc experiment: https://github.com/jychoi-hpc/xgc4py
import xgc4py
xgcexp = xgc4py.XGC(exdir)
#xgcexp = xgc4py.XGC('/gpfs/alpine/world-shared/csc143/jyc/summit/d3d_coarse_small')

res_path = 'build/'
filename = res_path + "d3d_coarse_v2_700.bp.mgard" 
with ad2.open(filename, 'r') as f:
    print(filename)
    f0_g = f.read('i_f_4d')
    
idx_zr = np.where(f0_g < 0)
f0_g[idx_zr] = 1e-4 
print(f0_f.shape, f0_g.shape)
n_phi = f0_f.shape[0]
n_vx, n_vy = f0_f.shape[-1], f0_f.shape[-2]
f0_inode1 = 0
ndata = f0_f.shape[-3]

den_f    = np.zeros([n_phi, ndata])
u_para_f = np.zeros([n_phi, ndata]) 
T_perp_f = np.zeros([n_phi, ndata]) 
T_para_f = np.zeros([n_phi, ndata]) 
n0_f     = np.zeros([n_phi, ndata]) 
T0_f     = np.zeros([n_phi, ndata])

den_g    = np.zeros([n_phi, ndata]) 
u_para_g = np.zeros([n_phi, ndata]) 
T_perp_g = np.zeros([n_phi, ndata]) 
T_para_g = np.zeros([n_phi, ndata]) 
n0_g     = np.zeros([n_phi, ndata])
T0_g     = np.zeros([n_phi, ndata])

for iphi in range(n_phi):
    den_f[iphi,], u_para_f[iphi,], T_perp_f[iphi,], T_para_f[iphi,], n0_f[iphi,], T0_f[iphi,] =\
        xgcexp.f0_diag(f0_inode1=f0_inode1, ndata=ndata, isp=1, f0_f=f0_f[iphi,:])
    den_g[iphi,], u_para_g[iphi,], T_perp_g[iphi,], T_para_g[iphi,], n0_g[iphi,], T0_g[iphi,] =\
        xgcexp.f0_diag(f0_inode1=f0_inode1, ndata=ndata, isp=1, f0_f=f0_g[iphi,:])

#print (den_g.shape, u_para_g.shape, T_perp_g.shape, T_para_g.shape, n0_g.shape, T0_g.shape) 

n0_avg, T0_avg = xgcexp.f0_avg_diag(f0_inode1, ndata, n0_f, T0_f)
n0_avg_rct, T0_avg_rct = xgcexp.f0_avg_diag(f0_inode1, ndata, n0_g, T0_g)

# compare
compute_diff("i_f", f0_f, f0_g) 
compute_diff("density_5d", den_f   , den_g) 
compute_diff("u_para_5d" , u_para_f, u_para_g) 
compute_diff("T_perp_5d" , T_perp_f, T_perp_g) 
compute_diff("T_para_5d" , T_para_f, T_para_g) 
compute_diff("n0_avg_5d", n0_avg_rct, n0_avg)
compute_diff("T0_avg_5d", T0_avg_rct, T0_avg)

