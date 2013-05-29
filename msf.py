from integrator import Integrator
from integrator import methods
from msf_tools import MSF_Integrator
import numpy as np


class PyMSF():
    def __init__(self, model, init_cond, tmax = 5e3, dt = 5e-4):
        self.m = model
        self.init_cond = init_cond
        self.tmax = tmax
        self.dt = dt
        self.integrator_memory_limit = 512 # MB
        self.msfi = MSF_Integrator(model)

    def floq(self, gamma, traj):
        self.msfi.compute_Floquet(gamma, self.dt, traj) 
        evals,_ = np.linalg.eig(self.msfi.X[-1])
        
        return np.sort(np.abs(evals))[::-1]

    def compute_real_MSF(self, gamma_vals, traj):
        MSF = np.zeros((self.m.ndim, len(gamma_vals)))
        for i, gamma in enumerate(gamma_vals):
            MSF[:, i] = self.floq(gamma, traj)
        
        return MSF

    def compute_trajectory(self, dev_max, dev_var, dev_thr):
        n_chunks, chunk_size_ms, last_chunk_ms = self._get_integration_chunks()
        init_cond = self.init_cond
        idx = None

        for i in range(n_chunks):
            if i == n_chunks - 1:
                integrator = Integrator(self.m, last_chunk_ms, self.dt)
            else:
                integrator = Integrator(self.m, chunk_size_ms, self.dt)

            integrator.set_initial_conditions(init_cond)
            idx = integrator.find_limit_cycle(dev_max, dev_var, dev_thr, methods.EULER)

            if idx[1] > idx[0] or i == n_chunks - 1:
                return integrator.data[:, idx[0]:idx[1]].copy()
            else:
                init_cond = integrator.data[:, -1].copy()

    def vary_par(self, par_name, n_points, par_min, par_max,
            gamma_min, gamma_max,
            dev_max = 8e-5, dev_var = 0, dev_thr = -58,
            only_traj = False):

        par_vals = np.linspace(par_min, par_max, n_points)
        gamma_vals = np.linspace(gamma_min, gamma_max, n_points)
        MSF = np.zeros((self.m.ndim, n_points, n_points))

        for i, par in enumerate(par_vals):
            setattr(self.m.par, par_name, par)
            print('Trajectory for %s = %s' % (par_name, par))
            traj = self.compute_trajectory(dev_max, dev_var, dev_thr)

            if not only_traj:
                print('\n Comutig MSF... \n')
                MSF[:, :, i] = self.compute_real_MSF(gamma_vals, traj)

        return MSF, gamma_vals, par_vals

    def _get_integration_chunks(self):
        # assuming double datatype
        ms_size_bytes = self.m.ndim * 8 / self.dt
        chunk_size_bytes = self.integrator_memory_limit * 1024 ** 2
        chunk_size_ms = chunk_size_bytes / int(ms_size_bytes)
        n_chunks, last_chunk_ms = divmod(self.tmax, chunk_size_ms)

        return int(n_chunks), chunk_size_ms, last_chunk_ms
