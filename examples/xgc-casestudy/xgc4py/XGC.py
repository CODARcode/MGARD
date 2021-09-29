import numpy as np import adios2 as ad2 from tqdm import tqdm

    import os import subprocess

    from math import sqrt, floor, exp try : import torch except ImportError:
    import warnings
    warnings.warn("No torch module. Disabled.")

class XGC:
    class Mesh:
        def __init__(self, expdir=''):
            fname = os.path.join(expdir, 'xgc.mesh.bp')
            print (f"Reading: {fname}")
            with ad2.open(fname, 'r') as f:
                self.nnodes = f.read('n_n').item()
                self.ncells = f.read('n_t').item()
                self.rz = f.read('rz')
                self.conn = f.read('nd_connect_list')
                self.psi = f.read('psi')
                self.psi_surf = f.read('psi_surf')
                self.surf_idx = f.read('surf_idx')
                self.surf_len = f.read('surf_len')
                self.nextnode = f.read('nextnode')

            self.r = self.rz[:,0]
            self.z = self.rz[:,1]

            if len(self.surf_len) == 0:
                print (f"==> Warning: no psi_surf/surf_len/surf_idx in {fname}")
                print (f"==> Warning: Plese check if CONVERT_GRID2 enabled.")

            bl = np.zeros_like(self.nextnode, dtype=bool)
            for i in range(len(self.surf_len)):
                n = self.surf_len[i]
                k = self.surf_idx[i,:n]-1
                for j in k:
                    bl[j] = True

            self.not_in_surf=np.arange(len(self.nextnode))[~bl]

        def surf_nodes(self, i):
            m = len(self.surf_len)
            if i < m:
                n = self.surf_len[i]
                k = self.surf_idx[i,:n]-1
            else:
                print (f"==> Warning: surf index out of range (max: {m}). Returning non surf nodes.")
                k = self.not_in_surf
            return (k)

    class F0mesh:
        def __init__(self, expdir=''):
            fname = os.path.join(expdir, 'xgc.f0.mesh.bp')
            print (f"Reading: {fname}")
            with ad2.open(fname, 'r') as f:
                self.f0_nmu = f.read('f0_nmu')
                self.f0_nvp = f.read('f0_nvp')
                self.f0_smu_max = f.read('f0_smu_max')
                self.f0_vp_max = f.read('f0_vp_max')
                self.f0_dsmu = f.read('f0_dsmu')
                self.f0_dvp = f.read('f0_dvp')
                self.f0_T_ev = f.read('f0_T_ev')
                self.f0_grid_vol_vonly = f.read('f0_grid_vol_vonly')
                self.nb_curl_nb = f.read('nb_curl_nb')

            self.sml_e_charge=1.6022E-19  ## electron charge (MKS)
            self.sml_ev2j=self.sml_e_charge

            self.ptl_e_mass_au=2E-2
            self.ptl_mass_au=2E0
            self.sml_prot_mass=1.6720E-27 ## proton mass (MKS)
            self.ptl_mass = [self.ptl_e_mass_au*self.sml_prot_mass, self.ptl_mass_au*self.sml_prot_mass]

            self.ptl_charge_eu=1.0  #! charge number
            self.ptl_e_charge_eu=-1.0
            self.ptl_charge = [self.ptl_e_charge_eu*self.sml_e_charge, self.ptl_charge_eu*self.sml_e_charge]

            ## index: imu, range: [0, f0_nmu]
            self.mu_vol = np.ones(self.f0_nmu+1)
            self.mu_vol[0] = 0.5
            self.mu_vol[-1] = 0.5

            ## index: ivp, range: [-f0_nvp, f0_nvp]
            self.vp_vol = np.ones(self.f0_nvp*2+1)
            self.vp_vol[0] = 0.5
            self.vp_vol[-1] = 0.5

#f0_smu_max = 3.0
#f0_dsmu = f0_smu_max / f0_nmu
            self.mu = (np.arange(self.f0_nmu+1, dtype=np.float128)*self.f0_dsmu)**2
            self.vp = np.arange(-self.f0_nvp, self.f0_nvp+1, dtype=np.float128)*self.f0_dvp

            ## pre-calculation for f0_diag
            isp = 1
            self.en_th = self.f0_T_ev[isp,:]*self.sml_ev2j
            self.vth2 = self.en_th/self.ptl_mass[isp]
            self.vth = np.sqrt(self.vth2)
            self.f0_grid_vol = self.f0_grid_vol_vonly[isp,:]

            _x, _y = np.meshgrid(self.mu_vol, self.vp_vol)
            self.mu_vp_vol = _x*_y

    class Grid:
        class Mat:
            """
             integer :: n,m,width
             real (8), allocatable :: value(:,:)
             integer, allocatable :: eindex(:,:),nelement(:)
            """
            n,m,width = 0,0,0
            value = None
            eindex = None
            nelement = None

            def mat_transpose_mult(self, x):
                assert self.n == len(x)
                y = np.zeros([self.m,])
                for i in range(self.n):
                    for j in range(self.nelement[i]):
                        k = self.eindex[i,j]-1
                        y[k] = y[k] + self.value[i,j] * x[i]
                return y

            def mat_mult(self, x):
                assert self.m == len(x)
                y = np.zeros([self.n,])
                for i in range(self.n):
                    for j in range(self.nelement[i]):
                        k = self.eindex[i,j]-1
                        y[i] = y[i] + self.value[i,j] * x[k]
                return y

        def __init__(self, expdir=''):
            self.cnv_2d_00 = self.Mat()
            self.cnv_00_2d = self.Mat()
            
            fname = os.path.join(expdir, 'xgc.fluxavg.bp')
            print (f"Reading: {fname}")
            with ad2.open(fname, 'r') as f:
                self.cnv_2d_00.n = f.read('nnode')
                self.cnv_2d_00.m = f.read('npsi')
                self.cnv_2d_00.width = f.read('width')
                self.cnv_2d_00.value = f.read('value')
                self.cnv_2d_00.eindex = f.read('eindex')
                self.cnv_2d_00.nelement = f.read('nelement')

                self.cnv_00_2d.n = f.read('nnode')
                self.cnv_00_2d.m = f.read('npsi')
                self.cnv_00_2d.width = f.read('width2')
                self.cnv_00_2d.value = f.read('value2')
                self.cnv_00_2d.eindex = f.read('eindex2')
                self.cnv_00_2d.nelement = f.read('nelement2')

                self.cnv_norm_1d00 = f.read('norm1d')
                self.cnv_norm_2d = f.read('norm2d')
                self.npsi = f.read('npsi').item()
    
            fname = os.path.join(expdir, 'xgc.mesh.bp')
            print (f"Reading: {fname}")
            with ad2.open(fname, 'r') as f:
                self.psi = f.read('psi')
                self.nnodes = f.read('n_n')
                self.x = f.read('rz')

            fname = os.path.join(expdir, 'xgc.grad_rz.bp')
            print (f"Reading: {fname}")
            with ad2.open(fname, 'r') as f:
                self.basis = f.read('basis')

            fname = os.path.join(expdir, 'xgc.bfield.bp')
            print (f"Reading: {fname}")
            with ad2.open(fname, 'r') as f:
                self.bfield = f.read('/node_data[0]/values')
            x = self.bfield
            y = np.sqrt(np.sum(x**2, axis=1))
            self.bfield = np.concatenate((x,y[:,np.newaxis]), axis=1)

            fname = os.path.join(expdir, 'xgc.f0.mesh.bp')
            print (f"Reading: {fname}")
            with ad2.open(fname, 'r') as f:
                self.v_gradb = f.read('v_gradb')
                self.v_curv = f.read('v_curv')
                self.nb_curl_nb = f.read('nb_curl_nb')
                self.gradpsi = f.read('gradpsi')

            fname = os.path.join(expdir, 'xgc.f0analysis.static.bp')
            if os.path.exists(fname):
                print (f"Reading: {fname}")
                with ad2.open(fname, 'r') as f:
                    self.f0_mu0_factor = f.read('f0_mu0_factor')
                    self.rgn = f.read('rgn')
            else:
                print (f"[WARN] cannot open: {fname}")
                print (f"==> Warning: f0_non_adiabatic will be incorrect.")

    class PSN:
        def __init__(self, expdir='', step=None):
            if step is None:
                return
            fname = os.path.join(expdir, 'restart_dir/xgc.f0.%05d.bp'%step)
            print (f"Reading: {fname}")
            with ad2.open(fname, 'r') as f:
                self.E_rho_ff = f.read('E_rho_ff') # (nphi,nnodes,3,2,3)
                self.pot_rho_ff = f.read('pot_rho_ff') # (nphi,nnodes,3,2)
                self.pot0 = f.read('pot0') # (nphi,nnodes)
            if len(self.E_rho_ff) == 0:
                print (f"==> Warning: no E_rho_ff/pot_rho_ff/pot0 data in {fname}")
                print (f"==> Warning: Plese check if XGC_F_COUPLING enabled.")
            

    """
      ! check if region 1 or 2
      logical function is_rgn12(r,z,psi)
        implicit none
        real (8) :: r,z,psi

        ! Use better logic for double-null configurations
        !if((psi > eq_x_psi -epsil_psi .or. &
        !     -(r-eq_x_r)*eq_x_slope + (z-eq_x_z) > 0D0) .and. -(r-eq_x2_r)*eq_x2_slope + (z-eq_x2_z) < 0D0   ) then
        !   is_rgn12=.true.
        !else
        !   is_rgn12=.false.
        !endif
        if ( (psi .le. eq_x_psi-epsil_psi .and. -(r-eq_x_r)*eq_x_slope + (z-eq_x_z) > 0D0 .and. &
              -(r-eq_x2_r)*eq_x2_slope + (z-eq_x2_z) < 0D0) .or. &
             (psi .gt. eq_x_psi-epsil_psi .and. psi .le. eq_x2_psi-epsil_psi .and. &
              -(r-eq_x2_r)*eq_x2_slope + (z-eq_x2_z) < 0D0) .or. &
             psi .gt. eq_x2_psi-epsil_psi) then
           is_rgn12=.true.
        else
           is_rgn12=.false.
        endif
      end function is_rgn12
    """
    def is_rgn12(self, r,z,psi):
        if ((psi <= self.eq_x_psi-self.epsil_psi and -(r-self.eq_x_r)*self.eq_x_slope + (z-self.eq_x_z) > 0.0 and \
              -(r-self.eq_x2_r)*self.eq_x2_slope + (z-self.eq_x2_z) < 0.0) or \
            (psi > self.eq_x_psi-self.epsil_psi and psi <= self.eq_x2_psi-self.epsil_psi and \
             -(r-self.eq_x2_r)*self.eq_x2_slope + (z-self.eq_x2_z) < 0.0) or \
            psi > self.eq_x2_psi-self.epsil_psi):
            return True
        else:
            return False

    """
    subroutine convert_grid_2_001d(grid,v2d,v1d)
      use grid_class
      implicit none
      type(grid_type), intent(in) :: grid
      real (8), intent(in) :: v2d(grid%nnode)
#ifdef CONVERT_GRID2
      real (8), intent(out)  :: v1d(grid%npsi_surf)
#else
      real (8), intent(out)  :: v1d(grid%npsi00)
#endif

      call mat_transpose_mult(grid%cnv_2d_00,v2d,v1d)
      v1d=v1d/grid%cnv_norm_1d00

    end subroutine convert_grid_2_001d
    """
    def convert_grid_2_001d(self, v2d):
        v1d = self.grid.cnv_2d_00.mat_transpose_mult(v2d)
        v1d = v1d/self.grid.cnv_norm_1d00
        return v1d

    """
    !! With CONVERT_GRID2 Off
    subroutine convert_001d_2_grid(grid,v1d,v2d)
      use eq_module
      use grid_class
      implicit none
      type(grid_type), intent(in) :: grid
      real (8), intent(in)  :: v1d(grid%npsi00)
      real (8), intent(out) :: v2d(grid%nnode)
      !
      integer :: i, ip
      real (8) :: pn, wp

      do i=1, grid%nnode
         pn=(grid%psi(i)-grid%psi00min)/grid%dpsi00
         ip=floor(pn)+1
         if(0 < ip .and. ip < grid%npsi00 .and. is_rgn12(grid%x(1,i),grid%x(2,i),grid%psi(i)) ) then
            wp=1D0 - ( pn - real(ip-1,8) )
         elseif( ip <= 0 ) then
            ip=1
            wp=1D0
         else
            ip=grid%npsi00-1
            wp=0D0
         endif

         v2d(i)=v1d(ip)*wp  + v1d(ip+1)*(1D0-wp)
      end do

    end subroutine convert_001d_2_grid

    !! With CONVERT_GRID2 On
    subroutine convert_001d_2_grid(grid,v1d,v2d)
    use eq_module
    use grid_class
    implicit none
    type(grid_type), intent(in) :: grid
    real (8), intent(in)  :: v1d(grid%npsi_surf)
    real (8), intent(out) :: v2d(grid%nnode)

    print *, "!! convert_001d_2_grid for CONVERT_GRID2"
    call mat_mult(grid%cnv_00_2d,v1d,v2d)

    end subroutine convert_001d_2_grid
    """
    def convert_001d_2_grid(self, v1d, CONVERT_GRID2=True):
        if CONVERT_GRID2:
            v2d = self.grid.cnv_00_2d.mat_mult(v1d)
            return v2d
        
        v2d = np.zeros(self.grid.nnodes)
        for i in range(self.grid.nnodes):
            pn=(self.grid.psi[i]-self.grid.psi00min)/self.grid.dpsi00
            ip=floor(pn)+1
            if(0 < ip and ip < self.grid.npsi00 and self.is_rgn12(self.grid.x[i,0],self.grid.x[i,1],self.grid.psi[i])):
                wp=1.0 - ( pn - float(ip-1) )
            elif (ip<=0):
                ip=1
                wp=1.0
            else:
                ip=self.grid.npsi00-1
                wp=0.0

            v2d[i]=v1d[ip-1]*wp  + v1d[ip]*(1.0-wp)

        return v2d

    def __init__(self, expdir='', step=None, device=None):
        self.expdir = expdir
        
        ## populate mesh
        self.mesh = self.Mesh(expdir)

        ## populate f0mesh
        self.f0mesh = self.F0mesh(expdir)
        
        ## populate grid
        self.grid = self.Grid(expdir)

        ## populate psn
        self.psn = self.PSN(expdir, step)

        ## XGC has been updated to output more values: eq_x_slope, eq_x2_psi, eq_x2_r, eq_x2_z, eq_x2_slope
        fname = os.path.join(expdir, 'xgc.equil.bp')
        print (f"Reading: {fname}")
        with ad2.open(fname, 'r') as f:
            self.eq_axis_b = f.read('eq_axis_b')
            self.eq_axis_r = f.read('eq_axis_r')
            self.eq_axis_z = f.read('eq_axis_z')

            self.eq_max_r = f.read('eq_max_r')
            self.eq_max_z = f.read('eq_max_z')
            self.eq_min_r = f.read('eq_min_r')
            self.eq_min_z = f.read('eq_min_z')

        fname = os.path.join(expdir, 'xgc.units.bp')
        print (f"Reading: {fname}")
        with ad2.open(fname, 'r') as f:
            self.eq_x_psi = f.read('eq_x_psi')
            self.eq_x_r = f.read('eq_x_r')
            self.eq_x_z = f.read('eq_x_z')
            self.eq_x_slope = f.read('eq_x_slope')

            ## Added by Jong
            self.eq_x2_psi = f.read('eq_x2_psi')
            self.eq_x2_r = f.read('eq_x2_r')
            self.eq_x2_z = f.read('eq_x2_z')
            self.eq_x2_slope = f.read('eq_x2_slope')

        if self.eq_x2_psi.ndim > 0:
            print (f"==> Warning: no eq_x2_psi/eq_x2_r/eq_x2_z/eq_x2_slope data in {fname}")
            print (f"==> Warning: f0_avg_diag will be incorrect.")

        self.epsil_psi =  1E-5

#fname = os.path.join(expdir, 'xgc.f0analysis.static.bp')
#if os.path.exists(fname) :
#print(f "Reading: {fname}")
#with ad2.open(fname, 'r') as f:
#self.sml_inpsi = f.read('sml_inpsi')
#self.sml_outpsi = f.read('sml_outpsi')

        fname = os.path.join(expdir, 'fort.input.used')
        result = subprocess.run(['/usr/bin/grep', '-a', 'SML_OUTPSI', fname], stdout=subprocess.PIPE)
        kv = result.stdout.decode().replace(' ','').replace(',','').split('\n')[0].split('=')
        self.sml_outpsi = float(kv[1])
        self.sml_outpsi = self.sml_outpsi * self.eq_x_psi
#print('sml_outpsi=', self.sml_outpsi)

        fname = os.path.join(expdir, 'fort.input.used')
        result = subprocess.run(['/usr/bin/grep', '-a', 'SML_INPSI', fname], stdout=subprocess.PIPE)
        kv = result.stdout.decode().replace(' ','').replace(',','').split('\n')[0].split('=')
        self.sml_inpsi = float(kv[1])
        self.sml_inpsi = self.sml_inpsi * self.eq_x_psi
#print('sml_inpsi=', self.sml_inpsi)

        self.sml_00_npsi = self.grid.npsi
        print ('sml_00_npsi, sml_inpsi, sml_outpsi=', self.sml_00_npsi, self.sml_inpsi, self.sml_outpsi)

        fname = os.path.join(expdir, 'fort.input.used')
        result = subprocess.run(['/usr/bin/grep', '-a', 'SML_NPHI_TOTAL', fname], stdout=subprocess.PIPE)
        kv = result.stdout.decode().replace(' ','').replace(',','').split('\n')[0].split('=')
        self.nphi = int(kv[1])
        print ('sml_nphi_total=', self.nphi)

        fname = os.path.join(expdir, 'fort.input.used')
        result = subprocess.run(['/usr/bin/grep', '-a', 'SML_GRID_NRHO', fname], stdout=subprocess.PIPE)
        kv = result.stdout.decode().replace(' ','').replace(',','').split('\n')[0].split('=')
        self.sml_grid_nrho = int(kv[1])
        print ('sml_grid_nrho=', self.sml_grid_nrho)

        fname = os.path.join(expdir, 'fort.input.used')
        result = subprocess.run(['/usr/bin/grep', '-a', 'SML_EXCLUDE_PRIVATE', fname], stdout=subprocess.PIPE)
        kv = result.stdout.decode().replace(' ','').replace(',','').split('\n')[0].split('=')
        self.sml_exclude_private = kv[1] == 'T'
        print ('sml_exclude_private=', self.sml_exclude_private)

        fname = os.path.join(expdir, 'fort.input.used')
        result = subprocess.run(['/usr/bin/grep', '-a', 'SML_RHOMAX', fname], stdout=subprocess.PIPE)
        kv = result.stdout.decode().replace(' ','').replace(',','').split('\n')[0].split('=')
        self.sml_rhomax = float(kv[1])
        print ('sml_rhomax=', self.sml_rhomax)

#!min - max of grid
        self.grid.npsi00 = self.sml_00_npsi
#self.grid.psi00min = self.sml_inpsi * self.eq_x_psi
#self.grid.psi00max = self.sml_outpsi * self.eq_x_psi
        self.grid.psi00min = self.sml_inpsi
        self.grid.psi00max = self.sml_outpsi
        self.grid.dpsi00 = (self.grid.psi00max - self.grid.psi00min)/float(self.grid.npsi00-1)

#!rho
        self.grid.nrho=self.sml_grid_nrho  #! rho indexing starts from 0
        self.grid.rhomax=self.sml_rhomax   #! rho min is 0.
        self.grid.drho = self.grid.rhomax/self.grid.nrho
        
        self.device = device
        if device is not None:
            self.to(device)

        ## untwist
        nextnode_list = list()
        init = list(range(self.mesh.nnodes))
        nextnode_list.append(init)
        for iphi in range(1,self.nphi):
            prev = nextnode_list[iphi-1]
            current = [0,]*self.mesh.nnodes
            for i in range(self.mesh.nnodes):
                current[i] = self.mesh.nextnode[prev[i]]
#print(i, prev[i], nextnode[prev[i]])
            nextnode_list.append(current)
        self.nextnode_arr = np.array(nextnode_list)

    def f0_diag_future(self, f0_inode1, ndata, isp, f0_f, progress=False, nchunk=256, max_workers=16):
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = list()
            for i in range(0, ndata, nchunk):
                n = nchunk if i+nchunk < ndata else ndata-i
                f = executor.submit(self.f0_diag, f0_inode1+i, n, isp, f0_f[i:i+n])
                futures.append(f)
            
            alist = list()
            for f in tqdm(futures, disable=not progress):
                alist.append(f.result())

            y = list(map(lambda a: np.concatenate(a), zip(*alist)))
            return y

    def f0_diag(self, f0_inode1, ndata, isp, f0_f, progress=False):
        """ 
        Input:
        f0_inode1: int
        ndata: int (f0_inode2=f0_inode1+ndata)
        isp: electron(=0) or ion(=1)
        f0mesh: (F0mesh object) shoul have the following attrs:
            f0_nmu: int
            f0_nvp: int
            f0_smu_max: float
            f0_dsmu: float
            f0_T_ev: (nsp, nnodes)
            f0_grid_vol_vonly: (nsp, nnodes)
            f0_dvp: double
        mesh: (Mesh object) shoul have the following attrs:
            nnodes: int  -- number of mesh nodes
        f0_f: (ndata, f0_nmu+1, 2*f0_nvp+1) -- f-data

        Output: 
        den: (ndata, f0_nmu+1, 2*f0_nvp+1)
        u_para: (ndata, f0_nmu+1, 2*f0_nvp+1)
        T_perp: (ndata, f0_nmu+1, 2*f0_nvp+1)
        T_para: (ndata, f0_nmu+1, 2*f0_nvp+1)
        n0: (ndata)
        T0: (ndata)

        All outputs are before performing flux-surface averaging
        """

        ## Aliases
        f0_nmu = self.f0mesh.f0_nmu
        f0_nvp = self.f0mesh.f0_nvp
        f0_smu_max = self.f0mesh.f0_smu_max
        f0_dsmu = self.f0mesh.f0_dsmu
        f0_T_ev = self.f0mesh.f0_T_ev
        f0_grid_vol_vonly = self.f0mesh.f0_grid_vol_vonly
        f0_dvp = self.f0mesh.f0_dvp    
        nnodes = self.mesh.nnodes
        mu_vol = self.f0mesh.mu_vol
        vp_vol = self.f0mesh.vp_vol
        f0_grid_vol = self.f0mesh.f0_grid_vol[f0_inode1:f0_inode1+ndata].astype(np.float128)
        mu_vp_vol = self.f0mesh.mu_vp_vol.astype(np.float128)
        mu = self.f0mesh.mu
        vp = self.f0mesh.vp
        vth = self.f0mesh.vth[f0_inode1:f0_inode1+ndata].astype(np.float128)
        vth2 = self.f0mesh.vth2[f0_inode1:f0_inode1+ndata].astype(np.float128)

        ## Check
        if f0_f.ndim == 2:
            f0_f = f0_f[np.newaxis,:]
#print(f0_f.shape, (ndata, f0_nmu + 1, f0_nvp * 2 + 1))
        assert(f0_f.shape[0] == ndata)
        assert(f0_f.shape[1] == f0_nmu+1)
        assert(f0_f.shape[2] >= f0_nvp*2+1)

        sml_e_charge=1.6022E-19  ## electron charge (MKS)
        sml_ev2j=sml_e_charge

        ptl_e_mass_au=2E-2
        ptl_mass_au=2E0
        sml_prot_mass=1.6720E-27 ## proton mass (MKS)
        ptl_mass = [ptl_e_mass_au*sml_prot_mass, ptl_mass_au*sml_prot_mass]

        ptl_charge_eu=1.0  #! charge number
        ptl_e_charge_eu=-1.0
        ptl_charge = [ptl_e_charge_eu*sml_e_charge, ptl_charge_eu*sml_e_charge]

#(2020 / 12) use pre - computed in xgc4py
# ##index : imu, range : [0, f0_nmu]
#mu_vol = np.ones(f0_nmu + 1)
#mu_vol[0] = 0.5
#mu_vol[-1] = 0.5

# ##index : ivp, range : [- f0_nvp, f0_nvp]
#vp_vol = np.ones(f0_nvp * 2 + 1)
#vp_vol[0] = 0.5
#vp_vol[-1] = 0.5

#f0_smu_max = 3.0
#f0_dsmu = f0_smu_max / f0_nmu
#mu = (np.arange(f0_nmu + 1, dtype = np.float64) * f0_dsmu) * *2
#vp = np.arange(-f0_nvp, f0_nvp + 1, dtype = np.float64) * f0_dvp

#(2020 / 12) update to use matrix - vector operations.
# 1) Density, parallel flow, and T_perp moments
        vol_ = f0_grid_vol[:,np.newaxis,np.newaxis]*mu_vp_vol[np.newaxis,:,:]
        den_ = f0_f.astype(np.float128) * vol_
        u_para_ = f0_f.astype(np.float128) * vol_ * vth[:,np.newaxis,np.newaxis] * vp[np.newaxis,np.newaxis,:]
        T_perp_ = f0_f.astype(np.float128) * vol_ * 0.5 * mu[np.newaxis,:,np.newaxis] * vth2[:,np.newaxis,np.newaxis] * ptl_mass[isp]
#print(np.sum(f0_f.astype(np.float128) * mu [np.newaxis, :, np.newaxis],       \
              axis = (1, 2))[0],                                               \
       vth2[0], ptl_mass[isp])
#print(mu)
        s_den_  = np.sum(den_, axis=(1,2), dtype=np.float128)
        u_para_ = np.sum(u_para_, axis=(1,2), dtype=np.float128)/s_den_
        T_perp_ = np.sum(T_perp_, axis=(1,2), dtype=np.float128)/s_den_/sml_e_charge
#print(T_perp_[0])

# 2) T_para moment
        upar_ = u_para_/vth
        en_   = 0.5 * (vp[np.newaxis,:].astype(np.float128) - upar_[:,np.newaxis])**2

        T_para_ = f0_f * vol_ * en_[:,np.newaxis,:] * vth2[:,np.newaxis,np.newaxis] * ptl_mass[isp]
        T_para_ = 2.0*np.sum(T_para_, axis=(1,2), dtype=np.float128)/s_den_/sml_e_charge

        n0_ = s_den_
        T0_ = (2.0*T_perp_+T_para_)/3.0
        
        return (s_den_, u_para_, T_perp_, T_para_, n0_, T0_)

        ## (2020/12) Just for a reference
# #out
#den = np.zeros((ndata, f0_nmu + 1, 2 * f0_nvp + 1))
#u_para = np.zeros((ndata, f0_nmu + 1, 2 * f0_nvp + 1))
#T_perp = np.zeros((ndata, f0_nmu + 1, 2 * f0_nvp + 1))
#T_para = np.zeros((ndata, f0_nmu + 1, 2 * f0_nvp + 1))

# #1) Density, parallel flow, and T_perp moments
#for inode in tqdm(range(0, ndata), disable = not progress):
# ##Mesh properties
#en_th = f0_T_ev[isp, f0_inode1 + inode] * sml_ev2j
#vth = np.sqrt(en_th / ptl_mass[isp])
#f0_grid_vol = f0_grid_vol_vonly[isp, f0_inode1 + inode]

#for imu in range(0, f0_nmu + 1):
#for ivp in range(0, f0_nvp * 2 + 1):
# ##Vspace properties
#vol = f0_grid_vol * mu_vol[imu] * vp_vol[ivp]
#vp = (ivp - f0_nvp) * f0_dvp
#en = 0.5 * mu[imu]

#f = f0_f[inode, imu, ivp] #f0_f(ivp, inode, imu, isp)
#den[inode, imu, ivp] = f * vol
#u_para[inode, imu, ivp] = f * vol * vp * vth
#T_perp[inode, imu, ivp] = f * vol * en * vth * *2 * ptl_mass[isp]
# #if (inode == 0) : print('imu,inode,ivp,ptl_mass,vth,en,vol=', imu, inode,   \
                           ivp, ptl_mass[isp], vth, en, vol)

#for inode in range(0, ndata):
#u_para[inode, : ] = u_para[inode, : ] / np.sum(den [inode, :])
#T_perp[inode, : ] = T_perp[inode, : ] / np.sum(den [inode, :]) / sml_e_charge

#upar = np.sum(u_para, axis = (1, 2))

# #2) T_para moment
#for inode in tqdm(range(0, ndata), disable = not progress):
# ##Mesh properties
#en_th = f0_T_ev[isp, f0_inode1 + inode] * sml_ev2j
#vth = np.sqrt(en_th / ptl_mass[isp])
#f0_grid_vol = f0_grid_vol_vonly[isp, f0_inode1 + inode]

#for imu in range(0, f0_nmu + 1):
#for ivp in range(0, f0_nvp * 2 + 1):
# ##Vspace properties
#vol = f0_grid_vol * mu_vol[imu] * vp_vol[ivp]
#vp = (ivp - f0_nvp) * f0_dvp
#en = 0.5 * (vp - upar[inode] / vth) * *2

#f = f0_f[inode, imu, ivp] #f0_f(ivp, inode, imu, isp)
#T_para[inode, imu, ivp] = f * vol * en * vth * *2 * ptl_mass[isp]

#for inode in range(0, ndata):
#T_para[inode, : ] = 2.0 * T_para[inode, : ] / np.sum(den [inode, :]) /        \
                                  sml_e_charge

#n0 = np.sum(den, axis = (1, 2))
#T0 = (2.0 * np.sum(T_perp, axis = (1, 2)) + np.sum(T_para, axis = (1, 2))) /  \
      3.0

# #3) Get the flux - surface average of n and T
# #And the toroidal averages of n, T, and u_par
# #jyc : We need all plane data to get flux - surface average.Call f0_avg_diag

#return (den, u_para, T_perp, T_para, n0, T0)

    def to(self, device):
        """
        Create Torch tensors
        """
        self.device = device
        self.torch_f0_grid_vol = torch.from_numpy(self.f0mesh.f0_grid_vol)
        self.torch_mu_vp_vol = torch.from_numpy(self.f0mesh.mu_vp_vol)
        self.torch_mu = torch.from_numpy(self.f0mesh.mu)
        self.torch_vp = torch.from_numpy(self.f0mesh.vp)
        self.torch_vth = torch.from_numpy(self.f0mesh.vth)
        self.torch_vth2 = torch.from_numpy(self.f0mesh.vth2)

        if self.torch_f0_grid_vol.device != device: self.torch_f0_grid_vol = self.torch_f0_grid_vol.to(self.device)
        if self.torch_mu_vp_vol.device != device: self.torch_mu_vp_vol = self.torch_mu_vp_vol.to(self.device)
        if self.torch_mu.device != device: self.torch_mu = self.torch_mu.to(self.device)
        if self.torch_vp.device != device: self.torch_vp = self.torch_vp.to(self.device)
        if self.torch_vth.device != device: self.torch_vth = self.torch_vth.to(self.device)
        if self.torch_vth2.device != device: self.torch_vth2 = self.torch_vth2.to(self.device)
        print ('device:', self.device)
        
    def f0_diag_torch(self, f0_inode1, ndata, isp, f0_f, progress=False):
        """
        This is a torch version of f0_diag.
        Input:
        f0_inode1: int
        ndata: int (f0_inode2=f0_inode1+ndata)
        isp: electron(=0) or ion(=1)
        f0mesh: (F0mesh object) shoul have the following attrs:
            f0_nmu: int
            f0_nvp: int
            f0_smu_max: float
            f0_dsmu: float
            f0_T_ev: (nsp, nnodes)
            f0_grid_vol_vonly: (nsp, nnodes)
            f0_dvp: double
        mesh: (Mesh object) shoul have the following attrs:
            nnodes: int  -- number of mesh nodes
        f0_f: (ndata, f0_nmu+1, 2*f0_nvp+1) -- f-data

        Output:
        den: (ndata, f0_nmu+1, 2*f0_nvp+1)
        u_para: (ndata, f0_nmu+1, 2*f0_nvp+1)
        T_perp: (ndata, f0_nmu+1, 2*f0_nvp+1)
        T_para: (ndata, f0_nmu+1, 2*f0_nvp+1)
        n0: (ndata)
        T0: (ndata)

        All outputs are before performing flux-surface averaging
        """

        ## Aliases
        device = f0_f.device
        f0_nmu = self.f0mesh.f0_nmu
        f0_nvp = self.f0mesh.f0_nvp
        f0_smu_max = self.f0mesh.f0_smu_max
        f0_dsmu = self.f0mesh.f0_dsmu
        f0_T_ev = self.f0mesh.f0_T_ev
        f0_grid_vol_vonly = self.f0mesh.f0_grid_vol_vonly
        f0_dvp = self.f0mesh.f0_dvp
        nnodes = self.mesh.nnodes
        mu_vol = self.f0mesh.mu_vol
        vp_vol = self.f0mesh.vp_vol
        
        f0_grid_vol = self.torch_f0_grid_vol[f0_inode1:f0_inode1+ndata]
        mu_vp_vol = self.torch_mu_vp_vol
        mu = self.torch_mu
        vp = self.torch_vp
        vth = self.torch_vth[f0_inode1:f0_inode1+ndata]
        vth2 = self.torch_vth2[f0_inode1:f0_inode1+ndata]

        ## Check
        if f0_f.ndim == 2:
            f0_f = f0_f[np.newaxis,:]
#print(f0_f.shape, (ndata, f0_nmu + 1, f0_nvp * 2 + 1))
        assert(f0_f.shape[0] == ndata)
        assert(f0_f.shape[1] == f0_nmu+1)
        assert(f0_f.shape[2] >= f0_nvp*2+1)

        sml_e_charge=1.6022E-19  ## electron charge (MKS)
        sml_ev2j=sml_e_charge

        ptl_e_mass_au=2E-2
        ptl_mass_au=2E0
        sml_prot_mass=1.6720E-27 ## proton mass (MKS)
        ptl_mass = [ptl_e_mass_au*sml_prot_mass, ptl_mass_au*sml_prot_mass]

        ptl_charge_eu=1.0  #! charge number
        ptl_e_charge_eu=-1.0
        ptl_charge = [ptl_e_charge_eu*sml_e_charge, ptl_charge_eu*sml_e_charge]

#(2020 / 12) use pre - computed in xgc4py
# ##index : imu, range : [0, f0_nmu]
#mu_vol = np.ones(f0_nmu + 1)
#mu_vol[0] = 0.5
#mu_vol[-1] = 0.5

# ##index : ivp, range : [- f0_nvp, f0_nvp]
#vp_vol = np.ones(f0_nvp * 2 + 1)
#vp_vol[0] = 0.5
#vp_vol[-1] = 0.5

#f0_smu_max = 3.0
#f0_dsmu = f0_smu_max / f0_nmu
#mu = (np.arange(f0_nmu + 1, dtype = np.float64) * f0_dsmu) * *2
#vp = np.arange(-f0_nvp, f0_nvp + 1, dtype = np.float64) * f0_dvp

#(2020 / 12) update to use matrix - vector operations.
# 1) Density, parallel flow, and T_perp moments
        vol_ = f0_grid_vol[:,np.newaxis,np.newaxis]*mu_vp_vol[np.newaxis,:,:]
        den_ = f0_f * vol_
        u_para_ = f0_f * vol_ * vth[:,np.newaxis,np.newaxis] * vp[np.newaxis,np.newaxis,:]
        T_perp_ = f0_f * vol_ * 0.5 * mu[np.newaxis,:,np.newaxis] * vth2[:,np.newaxis,np.newaxis] * ptl_mass[isp]

        s_den_ = torch.sum(den_, axis=(1,2))
        u_para_ = u_para_/s_den_[:,np.newaxis,np.newaxis]
        T_perp_ = T_perp_/s_den_[:,np.newaxis,np.newaxis]/sml_e_charge

# 2) T_para moment
        upar_ = torch.sum(u_para_, axis=(1,2))/vth
        en_ = 0.5 * (vp[np.newaxis,:] - upar_[:,np.newaxis])**2

        T_para_ = f0_f * vol_ * en_[:,np.newaxis,:] * vth2[:,np.newaxis,np.newaxis] * ptl_mass[isp]
        T_para_ = 2.0*T_para_/s_den_[:,np.newaxis,np.newaxis]/sml_e_charge

        n0_ = s_den_
        T0_ = (2.0*torch.sum(T_perp_, axis=(1,2))+torch.sum(T_para_, axis=(1,2)))/3.0

        return (den_, u_para_, T_perp_, T_para_, n0_, T0_)

    def get_drift_velocity(self,iphi,node,mu,vp,isp,v_th):
        """
        Input:

        Output:
        exb
        """
        v_mag,v_exb,v_pardrift,pot_rho,grad_psi_sqr = None,None,None,None,None
        E = np.zeros(3)
        wrho = np.zeros(2)

        charge=self.f0mesh.ptl_charge[isp]
        mass=self.f0mesh.ptl_mass[isp]
        b=self.grid.bfield[node,3]
        rho_mks=vp*v_th/b*mass/charge
        mu_mks=0.5*mu*self.f0mesh.f0_T_ev[isp,node]*self.f0mesh.sml_ev2j/b

        over_B2=1.0/(b**2)
        D=1.0/ ( 1.0 + rho_mks * self.grid.nb_curl_nb[node] )

        grad_psi = np.zeros(2)
        grad_psi[0]=self.grid.gradpsi[node,0] #!psi_interpol(grid%x(1,node),grid%x(2,node),1,0)
        grad_psi[1]=self.grid.gradpsi[node,1] #!psi_interpol(grid%x(1,node),grid%x(2,node),0,1)
        grad_psi_sqr=sum(grad_psi[:]**2)
        over_abs_grad_psi=1.0/sqrt(grad_psi_sqr)

        if isp >= 1:
#!Ions
            rhoi=sqrt(2.0*mass*mu_mks/b)/charge
            rhon=min(rhoi,self.grid.rhomax)/self.grid.drho
            irho=min(floor(rhon),self.grid.nrho-1)
            wrho[1]=rhon - float(irho)
            wrho[0]=1.0-wrho[1]

#pot_rho =                                                                     \
    wrho(1) * 0.5D0 *                                                          \
    (psn % pot_rho_ff(0, irho, node) + psn % pot_rho_ff(1, irho, node)) &
#+ wrho(2) * 0.5D0 *(psn % pot_rho_ff(0, irho + 1, node) +                     \
                                      psn % pot_rho_ff(1, irho + 1, node))
#E = E + wrho(1) * 0.5D0 *                                                     \
             (psn % E_rho_ff(                                                  \
                        :, 0, irho, node) +                                    \
              psn % E_rho_ff(                                                  \
                        :, 1, irho, node))
#E = E + wrho(2) * 0.5D0 *                                                     \
             (psn % E_rho_ff(                                                  \
                        :, 0, irho + 1, node) +                                \
              psn % E_rho_ff(                                                  \
                        :, 1, irho + 1, node))

            pot_rho = wrho[0]*0.5*(self.psn.pot_rho_ff[iphi,node,irho  ,0]+self.psn.pot_rho_ff[iphi,node,irho  ,1])  \
                    + wrho[1]*0.5*(self.psn.pot_rho_ff[iphi,node,irho+1,0]+self.psn.pot_rho_ff[iphi,node,irho+1,1])
            E   = E   + wrho[0]*0.5*(self.psn.E_rho_ff[iphi,node,irho  ,0,:]+self.psn.E_rho_ff[iphi,node,irho  ,1,:])
            E   = E   + wrho[1]*0.5*(self.psn.E_rho_ff[iphi,node,irho+1,0,:]+self.psn.E_rho_ff[iphi,node,irho+1,1,:])
        else:
#!Electrons
            pot_rho = 0.5*(self.psn.pot_rho_ff[iphi,node,0,0]+self.psn.pot_rho_ff[iphi,node,0,1])
            E = 0.5*(self.psn.E_rho_ff[iphi,node,0,0,:]+self.psn.E_rho_ff[iphi,node,0,1,:])

#The ExB drift
        v_exb = np.zeros(3, dtype=np.float64)
        if self.grid.basis[node]==1:
#!rh This is good if E(1 : 2) is in R, Z basis -->
            vr       = D * ( E[2]*self.grid.bfield[node,1] - E[1]*self.grid.bfield[node,2] ) * over_B2
            vz       = D * ( E[0]*self.grid.bfield[node,2] - E[2]*self.grid.bfield[node,0] ) * over_B2
            v_exb[0] = vr * grad_psi[0] + vz * grad_psi[1]
            v_exb[1] = (- vr * grad_psi[1] + vz * grad_psi[0]) * over_abs_grad_psi
            v_exb[2] = D * ( E[1]*self.grid.bfield[node,0] - E[0]*self.grid.bfield[node,1] ) * over_B2
        else:
#!rh E(1 : 2) is now E_psi and E_theta !!!-->
            v_exb[0] = -D * E[1]*self.grid.bfield[node,2]*over_B2/over_abs_grad_psi #!! -E_theta*B_phi
            v_exb[1] =  D * E[0]*self.grid.bfield[node,2]*over_B2  #!! E_psi*B_phi
            v_exb[2] = -D * E[0]*(-self.grid.bfield[node,0]*grad_psi[1]+self.grid.bfield[node,1]*grad_psi[0]) \
                          * over_B2 * over_abs_grad_psi  #!! -E_psi*B_theta

        return (v_mag,v_exb,v_pardrift,pot_rho,grad_psi_sqr)

    def f0_avg_diag(self, f0_inode1, ndata, n0_all, T0_all, CONVERT_GRID2=True):
        """ 
        Input:
        nphi: int -- total number of planes
        f0_inode1: int -- the starting index of mesh node
        ndata: int (f0_inode2=f0_inode1+ndata) -- the number of mesh node
        nnodes: int  -- number of mesh nodes
        n0_all: (nphi, ndata)
        T0_all: (nphi, ndata)

        Output: 
        n0_avg: (ndata) -- i.e., averaged over planes
        T0_avg: (ndata) -- i.e., averaged over planes

        All outputs are before performing flux-surface averaging
        """
        assert n0_all.shape == (self.nphi, ndata)
        assert T0_all.shape == (self.nphi, ndata)

        ## First, we calcuate average over planes
        n0 = np.mean(n0_all, axis=0)
        T0 = np.mean(T0_all, axis=0)

        ## n0
        n0_avg = np.zeros([self.grid.nnodes,])
        n0_avg[:] = np.nan
        n0_avg[f0_inode1:f0_inode1+ndata] = n0

        tmp00_surf = self.convert_grid_2_001d(n0_avg)
        n0_avg = self.convert_001d_2_grid(tmp00_surf, CONVERT_GRID2=CONVERT_GRID2)
        n0_avg[np.logical_or(np.isinf(n0_avg), np.isnan(n0_avg), n0_avg < 0.0)] = 1E17

        ## T0
        T0_avg = np.zeros([self.grid.nnodes,])
        T0_avg[:] = np.nan
        T0_avg[f0_inode1:f0_inode1+ndata] = T0

        tmp00_surf = self.convert_grid_2_001d(T0_avg)
        T0_avg = self.convert_001d_2_grid(tmp00_surf, CONVERT_GRID2=CONVERT_GRID2)
        T0_avg[np.logical_or(np.isinf(T0_avg), np.isnan(T0_avg), T0_avg < 0.0)] = 10E0

        return (n0_avg[f0_inode1:f0_inode1+ndata], T0_avg[f0_inode1:f0_inode1+ndata])

    def f0_non_adiabatic_future(self, iphi, f0_inode1, ndata, isp, f0_f, n0_avg, T0_avg, progress=False, nchunk=256, max_workers=16):
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = list()
            for i in range(0, ndata, nchunk):
                n = nchunk if i+nchunk < ndata else ndata-i
                f = executor.submit(self.f0_non_adiabatic, iphi, f0_inode1+i, n, isp, f0_f[:,i:i+n,:,:], n0_avg[i:i+n], T0_avg[i:i+n])
                futures.append(f)
            
            alist = list()
            for f in tqdm(futures, disable=not progress):
                alist.append(f.result())

            y = list(map(lambda a: np.concatenate(a), zip(*alist)))
            return y

    def f0_non_adiabatic(self, iphi, f0_inode1, ndata, isp, f0_f, n0_avg, T0_avg, progress=False):
        """ 
        Input:
        f0_inode1: int
        ndata: int (f0_inode2=f0_inode1+ndata)
        isp: electron(=0) or ion(=1)
        f0_f: (nphi, ndata, f0_nmu+1, 2*f0_nvp+1) -- f-data
        n0_avg: (ndata) -- i.e., averaged over planes
        T0_avg: (ndata) -- i.e., averaged over planes

        Output: 

        Calculate non-adiabatic distribution functions (DIAG_3D_F_CALC4)
        """

        ## Aliases
        f0_nmu = self.f0mesh.f0_nmu
        f0_nvp = self.f0mesh.f0_nvp
        f0_smu_max = self.f0mesh.f0_smu_max
        f0_vp_max = self.f0mesh.f0_vp_max
        f0_dsmu = self.f0mesh.f0_dsmu
        f0_T_ev = self.f0mesh.f0_T_ev
        f0_grid_vol_vonly = self.f0mesh.f0_grid_vol_vonly
        f0_dvp = self.f0mesh.f0_dvp    
        nnodes = self.mesh.nnodes

        ## Check
#if f0_f.ndim == 2:
#f0_f = f0_f[np.newaxis, : ]
#print(f0_f.shape, (ndata, f0_nmu + 1, f0_nvp * 2 + 1))
        assert(f0_f.shape[0] == self.nphi)
        assert(f0_f.shape[1] == ndata)
        assert(f0_f.shape[2] == f0_nmu+1)
        assert(f0_f.shape[3] >= f0_nvp*2+1)

        sml_e_charge=1.6022E-19  ## electron charge (MKS)
        sml_ev2j=sml_e_charge

        ptl_e_mass_au=2E-2
        ptl_mass_au=2E0
        sml_prot_mass=1.6720E-27 ## proton mass (MKS)
        ptl_mass = [ptl_e_mass_au*sml_prot_mass, ptl_mass_au*sml_prot_mass]

        ptl_charge_eu=1.0  #! charge number
        ptl_e_charge_eu=-1.0
        ptl_charge = [ptl_e_charge_eu*sml_e_charge, ptl_charge_eu*sml_e_charge]

        ## index: imu, range: [0, f0_nmu]
        mu_vol = np.ones(f0_nmu+1)
        mu_vol[0] = 0.5
        mu_vol[-1] = 0.5

        ## index: ivp, range: [-f0_nvp, f0_nvp]
        vp_vol = np.ones(f0_nvp*2+1)
        vp_vol[0] = 0.5
        vp_vol[-1] = 0.5

#f0_smu_max = 3.0
#f0_dsmu = f0_smu_max / f0_nmu
        mu = (np.arange(f0_nmu+1, dtype=np.float64)*f0_dsmu)**2
        smu = (np.arange(f0_nmu+1, dtype=np.float64)*f0_dsmu)
        smu[0] = f0_dsmu/self.grid.f0_mu0_factor

#call t_startf("DIAG_3D_F_CALC3")
#allocate(v_exb_n0(3, f0_inode1                                                \
                   : f0_inode2, 0                                              \
                   : f0_nmu, diag_1d_isp                                       \
                   : diag_1d_nsp))
#allocate(boltz_fac_n0(f0_inode1                                               \
                       : f0_inode2, 0                                          \
                       : f0_nmu, diag_1d_isp                                   \
                       : diag_1d_nsp))
#allocate(ftot_n0(-f0_nvp                                                      \
                  : f0_nvp, f0_inode1                                          \
                  : f0_inode2, 0                                               \
                  : f0_nmu, diag_1d_isp                                        \
                  : diag_1d_nsp))
#allocate(dpot_n0(f0_inode1                                                    \
                  : f0_inode2, 0                                               \
                  : f0_nmu, diag_1d_isp                                        \
                  : diag_1d_nsp),                                              \
              &
#dpot_turb(f0_inode1 : f0_inode2, 0 : f0_nmu, diag_1d_isp : diag_1d_nsp))
        _v_exb_n0 = np.zeros((self.nphi,f0_nmu+1,ndata,3))
        _boltz_fac_n0 = np.zeros((self.nphi,f0_nmu+1,ndata))
        _dpot_turb = np.zeros((self.nphi,f0_nmu+1,ndata))
        
        vp  = 0.0
        vth = 0.0
        csign = ptl_charge[isp]/sml_e_charge
        for _iphi in range(self.nphi):
            for inode in tqdm(range(0, ndata), disable=not progress):
                for imu in range(0, f0_nmu+1):
                    v_mag,v_exb,v_pardrift,pot_rho,grad_psi_sqr = self.get_drift_velocity(_iphi,f0_inode1+inode,mu[imu],vp,isp,vth)
                    _v_exb_n0[_iphi,imu,inode,:] = v_exb[:]
                    _boltz_fac_n0[_iphi,imu,inode] = exp(-csign/T0_avg[inode]*(pot_rho-self.psn.pot0[_iphi,f0_inode1+inode]))
                    _dpot_turb[_iphi,imu,inode] = pot_rho

#call t_startf("DIAG_3D_F_MPI2")
#v_exb_n0 = v_exb_n0 avg over planes
#boltz_fac_n0 = boltz_fac_n0 avg over planes
#dpot_n0 = dpot_turb avg over planes
#ftot_n0 = f0_f avg over planes
        v_exb_n0 = np.mean(_v_exb_n0, axis=0)
        boltz_fac_n0 = np.mean(_boltz_fac_n0, axis=0)
        dpot_n0 = np.mean(_dpot_turb, axis=0)
        ftot_n0 = np.mean(f0_f, axis=0)

#call t_startf("DIAG_3D_F_CALC4")
        fn_n0 = np.zeros((ndata,f0_nmu+1,f0_nvp*2+1), dtype=np.float64)
        fn_turb = np.zeros((ndata,f0_nmu+1,f0_nvp*2+1), dtype=np.float64)
        n_energy=8 
        ndiag_en=4
        en_max = 1.0/sqrt(f0_smu_max**2+f0_vp_max**2)
        csign = ptl_charge[isp]/sml_e_charge
        for inode in tqdm(range(0, ndata), disable=not progress):
            if (self.grid.psi[f0_inode1+inode] > self.sml_outpsi or self.grid.psi[f0_inode1+inode] < self.sml_inpsi or (self.grid.rgn[f0_inode1+inode]==3 and self.sml_exclude_private) ):
#print('skip:', f0_inode1 + inode, self.grid.psi[f0_inode1 + inode],           \
       self.sml_outpsi, self.sml_inpsi, self.grid.rgn[f0_inode1 + inode],      \
       self.sml_exclude_private)
                continue
            en_th = f0_T_ev[isp,f0_inode1+inode]*sml_ev2j
            vth = sqrt(en_th/ptl_mass[isp])
            f0_grid_vol = f0_grid_vol_vonly[isp,f0_inode1+inode]

            for imu in range(0, f0_nmu+1):
                for ivp in range(0, f0_nvp*2+1):
#vol = f0_grid_vol_vonly(node, isp) * mu_vol(imu) * vp_vol(ivp)
                    vol = f0_grid_vol * mu_vol[imu] * vp_vol[ivp]
                    vp = (ivp - f0_nvp) * f0_dvp
#call get_drift_velocity(grid, psn, node, mu(imu), vp, isp, vth, v_mag, v_exb, \
                         v_pardrift, pot_rho, grad_psi_sqr)
                    v_mag,v_exb,v_pardrift,pot_rho,grad_psi_sqr = self.get_drift_velocity(iphi,f0_inode1+inode,mu[imu],vp,isp,vth)

#!Adiabatic distribution function <f_M> exp(- q dphi / T) ~ <f_M>              \
    (1 - q dphi / T)
#!(It is safe to compute radial grad(B) and curvature fluxes from f_M because
#!the numerical error <v_gradB> _r != 0 is removed in get_drift_velocity.)
                    en = 0.5 * (mu[imu]+vp*vp)
                    i_en = max(0,min(round(sqrt(2.0*en)*en_max*n_energy),n_energy))
                    f0_prefac = sqrt(f0_T_ev[isp,f0_inode1+inode]/T0_avg[inode]) * n0_avg[inode]/T0_avg[inode] \
                                * exp(-f0_T_ev[isp,f0_inode1+inode]/T0_avg[inode]*en)*smu[imu]

#!Nonaxisymmetric parts of ExB velocity and Boltzmann - factor
                    v_exb_turb     = v_exb[:] - v_exb_n0[imu,inode,:]
                    boltz_fac_turb = exp(-csign/T0_avg[inode]*(pot_rho-self.psn.pot0[iphi,f0_inode1+inode])) - boltz_fac_n0[imu,inode]

#!Axisymmetric and non - axisymmetric adiabatic distribution functions
                    f_adia_n0   = f0_prefac * boltz_fac_n0[imu,inode]
                    f_adia_turb = f0_prefac * boltz_fac_turb

#!Axisymmetric and non - axisymmetric Non - adiabatic distribution functions
                    f_nonadia_n0   = ftot_n0[inode,imu,ivp]-f_adia_n0
                    f_nonadia_turb = (f0_f[iphi,inode,imu,ivp]-ftot_n0[inode,imu,ivp])-f_adia_turb

                    fn_n0[inode,imu,ivp] = f_nonadia_n0
                    fn_turb[inode,imu,ivp] = f_nonadia_turb
#if (imu == 0) and (ivp == 0) and (f0_inode1 + inode == 16000) :
#print('fn_n0:', self.grid.psi[f0_inode1 + inode], self.sml_outpsi,            \
       self.sml_inpsi, self.grid.rgn[f0_inode1 + inode],                       \
       self.sml_exclude_private)
#print('fn_n0:', imu, ivp, '=>', f_nonadia_n0, ftot_n0[inode, imu, ivp],       \
       f_adia_n0)
        
        return (fn_n0, fn_turb, np.moveaxis(boltz_fac_n0,1,0), np.moveaxis(dpot_n0,1,0), np.moveaxis(v_exb_n0,1,0))

if __name__ == "__main__":
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--expdir', help='exp directory (default: %(default)s)', default='')
    parser.add_argument('--timestep', help='timestep', type=int, default=0)
    parser.add_argument('--ndata', help='ndata', type=int)
    args = parser.parse_args()
    
    xgcexp = XGC(args.expdir, step=args.timestep)

    fname = os.path.join(args.expdir, 'restart_dir/xgc.f0.%05d.bp'%args.timestep)
    with ad2.open(fname, 'r') as f:
        i_f = f.read('i_f')
    
    nphi = i_f.shape[0]
    iphi = 0
    f0_inode1 = 0
    ndata = i_f.shape[2] if args.ndata is None else args.ndata
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("device:", device)

    fn0_all = np.zeros([nphi,ndata])
    fT0_all = np.zeros([nphi,ndata])
    for iphi in range(nphi):
        f0_f = np.moveaxis(i_f[iphi,:],1,0)
        f0_f = f0_f[f0_inode1:f0_inode1+ndata,:,:]
        den, upara, Tperp, Tpara, fn0, fT0 = \
            xgcexp.f0_diag_future(f0_inode1=f0_inode1, ndata=ndata, isp=1, f0_f=f0_f, progress=True)
        fn0_all[iphi,:] = fn0
        fT0_all[iphi,:] = fT0
    print (den.shape, upara.shape, Tperp.shape, Tpara.shape, fn0.shape, fT0.shape)
    
    fn0_avg, fT0_avg = xgcexp.f0_avg_diag(f0_inode1, ndata, fn0_all, fT0_all)
    print (fn0_avg.shape, fT0_avg.shape)

    f0_f_all = np.moveaxis(i_f[:,:,:,:],2,1)[:,f0_inode1:(f0_inode1+ndata),:,:].copy()
    fn_n0_all = np.zeros([nphi,ndata,f0_f_all.shape[2],f0_f_all.shape[3]])
    fn_turb_all = np.zeros([nphi,ndata,f0_f_all.shape[2],f0_f_all.shape[3]])
    for iphi in range(nphi):
        fn_n0, fn_turb, _, _, _ = \
            xgcexp.f0_non_adiabatic_future(iphi=iphi, f0_inode1=f0_inode1, ndata=ndata, isp=1, \
                f0_f=f0_f_all, n0_avg=fn0_avg, T0_avg=fT0_avg, progress=True)
        fn_n0_all[iphi,:] = fn_n0
        fn_turb_all[iphi,:] = fn_turb
    print (fn_n0_all.shape, fn_turb_all.shape)
    print ('fn_n0_all:', np.min(fn_n0_all),np.mean(fn_n0_all), np.max(fn_n0_all))
    print ('fn_turb_all:', np.min(fn_turb_all),np.mean(fn_turb_all), np.max(fn_turb_all))
