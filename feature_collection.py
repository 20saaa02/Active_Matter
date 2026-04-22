import numpy as np
import pandas as pd
from typing import List, Dict
import json
import os
import math
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from all_trajectories_figure import AllTrajectoriesFigure  
from particle_trajectory_fitting import ParticleTrajectoryFitting
from colors_manager import ColorManager


class FeatureCollection:
    """
    FeatureCollection is responsible for extracting descriptive features from robot trajectories.

    It combines:
    - Fourier-based trajectory approximation features
    - Particle model fitting features
    - Kinematic trajectory features 

    It also provides tools for:
    - Saving extracted features to CSV
    - Visualizing feature distributions and correlations

    Parameters
    ----------
    coord_data : Dict[int, np.ndarray]
        Dictionary mapping robot_id → trajectory array of shape (T, 2)
    colors : ColorManager, optional
        Object for consistent color mapping in visualizations
    robots_form : str, default 'circle'
        Type of robots (affects particle modeling logic)
    time_period : List[int], default [0, 200]
        Time slice of trajectory to use
    show_flag : bool, default False
        Whether to visualize intermediate results
    save_flag : bool, default False
        Whether to save plots and outputs
    """
    def __init__(self, coord_data: Dict[int, np.ndarray],   colors: ColorManager = None, robots_form: str = 'circle', time_period: List = [0,200],
                  show_flag: bool = False, save_flag: bool = False):
        self.t0, self.t1 = time_period
        self.coords = np.array([coord_data[r] for r in coord_data])[:,self.t0:self.t1]
        self.robot_ids = list(coord_data.keys())
        self.robots_form = robots_form
        self.time_period = time_period
        self.show_flag = show_flag
        self.colors=colors
        self.save_flag = save_flag
        self.save_dir = Path(robots_form) / 'figures'


        self.x, self.y = self.coords[:,:,0], self.coords[:,:,1]

        self.fitting = ParticleTrajectoryFitting(coord_data=coord_data, time_period=[self.t0, self.t1])

        self.data={}        
        self.data['robot_id'] = self.robot_ids

    def fourier_features(self):
        """
        Extract Fourier-based features from trajectories.

        This method:
        - Computes FFT of complex trajectory signal (x + i*y)
        - Selects dominant frequency components
        - Reconstructs trajectory using top-k harmonics
        - Computes reconstruction error (RMSE)
        - Computes residual standard deviation

        Features added:
        - omega_z_* : dominant angular frequencies
        - amp_z_*   : amplitudes of dominant components
        - std_x, std_y : residual standard deviation
        - rmse_comp : reconstruction error

        Optional:
        - Visualizes original vs reconstructed trajectories if show_flag=True
        """
        k=2
        N, T = self.x.shape
        t = np.arange(T)[None, None, :]
        omega_z =np.zeros((N, k*2))
        am_z = np.zeros((N, k*2))
        phi_z = np.zeros((N, k*2))
        std = np.zeros((N, 2))
        z_rec = np.zeros((N, T), dtype=complex)

        mean_x, mean_y = self.x.mean(axis=1, keepdims=True), self.y.mean(axis=1, keepdims=True)
        
        Zf = np.fft.fft(self.x-mean_x+1j*(self.y-mean_y), axis=1)
        freqs = np.fft.fftfreq(T, 1)  
        mask = freqs != 0
        freq = freqs[mask]
        Z = Zf[:,mask]
        amp = np.abs(Z)
        idx = np.argsort(np.abs(Z),axis=1)[:,::-1][:,:k*2]
        omega_z = 2*np.pi*freq[idx]
        phi_z = np.angle(np.take_along_axis(Z, idx, axis=1))
        am_z =  np.take_along_axis(amp, idx, axis=1) / T
        omega = omega_z[:, :, None]   # (N,k,1)
        phi   = phi_z[:, :, None]     # (N,k,1)
        amp   = am_z[:, :, None]      # (N,k,1)
        z_rec = np.sum(
            amp * np.exp(1j * (omega * t + phi)),
            axis=1
        )
        xz_rec = np.real(z_rec)+ mean_x
        yz_rec = np.imag(z_rec)+ mean_y
        xz_div = self.x - xz_rec
        yz_div = self.y - yz_rec
        std = np.stack([
            np.std(xz_div, axis=1),
            np.std(yz_div, axis=1)
        ], axis=1)

        for j in range(k*2):
            self.data[f'omega_z_{j+1}'] = omega_z[:,j].tolist()
        for j in range(k):
            self.data[f'amp_z_{j+1}'] = am_z[:,j].tolist()
        self.data[f'std_x'] = std[:, 0]
        self.data[f'std_y'] = std[:, 1]
        self.data['rmse_comp'] = list(np.sqrt(
            np.mean((self.x - xz_rec)**2 + (self.y - yz_rec)**2, axis=1)
        ))


        if self.show_flag:
            coord_data_rec_robot = {}
            for i, robot_id in enumerate(self.robot_ids):
                coord_data_rec_robot[robot_id] = self.coords[i]
                coord_data_rec_robot[robot_id + 200] = np.stack(
                    (xz_rec[i], yz_rec[i]),
                    axis=1
                )
            fig = AllTrajectoriesFigure(coord_data = coord_data_rec_robot, 
                                        robots_form = self.robots_form, 
                                        colors = self.colors,
                                        time_period = self.time_period)
            fig.space_trajectories()
            fig.temporal_trajectories()
        '''
        Xf = np.fft.fft(self.x - mean_x, axis=1)
        freqs = np.fft.fftfreq(T, 1)
        mask = freqs > 0
        freq = freqs[mask]
        X = Xf[:, mask]
        amp = np.abs(X)
        idx = np.argsort(amp, axis=1)[:, ::-1][:, :k]
        omega_x = 2 * np.pi * freq[idx]
        phi_x   = np.angle(np.take_along_axis(X, idx, axis=1))
        am_x    = 2 * np.take_along_axis(amp, idx, axis=1) / T
        omega = omega_x[:, :, None]   # (N,k,1)
        phi   = phi_x[:, :, None]
        amp   = am_x[:, :, None]
        x_rec = np.sum(
            amp * np.cos(omega * t + phi),
            axis=1
        )
        x_rec = x_rec + mean_x

        Yf = np.fft.fft(self.y - mean_y, axis=1)
        Y = Yf[:, mask]
        amp_y = np.abs(Y)
        idx_y = np.argsort(amp_y, axis=1)[:, ::-1][:, :k]
        omega_y = 2 * np.pi * freq[idx_y]
        phi_y   = np.angle(np.take_along_axis(Y, idx_y, axis=1))
        am_y    = 2 * np.take_along_axis(amp_y, idx_y, axis=1) / T
        omega = omega_y[:, :, None]
        phi   = phi_y[:, :, None]
        amp   = am_y[:, :, None]
        y_rec = np.sum(
            amp * np.cos(omega * t + phi),
            axis=1
        )
        y_rec = y_rec + mean_y
        '''
        
    def features_of_particle(self):
        """
        Extract features based on particle trajectory fitting.

        Uses precomputed fitting parameters (from JSON file) and simulates:
        - Angular frequencies (w, W)
        - Field-related parameters (E0, Ex0, Ey0)
        - Phase (f)
        - Velocity-related constant (B)

        Also includes:
        - RMSE of particle model fit
        - Binary label indicating successful particle-like behavior

        Features added:
        - rmse_particle
        - particle_successfull
        - w, W, f, B, E0, Ex0, Ey0
        """
        N = self.x.shape[0]
        f = np.zeros((N, 1))
        B = np.zeros((N, 1))
        E0 = np.zeros((N, 1))
        Ex0 = np.zeros((N, 1))
        Ey0 = np.zeros((N, 1))
        w = np.zeros((N, 1))
        W = np.zeros((N, 1))
        rmse = np.zeros((N, 1))
        particle_succ = np.zeros((N, 1))

        particle_parameters_file = "circle/fit_all_particle_results.json"
        with open(particle_parameters_file, "r") as file:
            particle_parameters = json.load(file)
        particle_parameters_dict = {entry["robot_id"]: entry for entry in particle_parameters}
        for i, robot_id in enumerate(self.robot_ids):
            params = particle_parameters_dict[robot_id]["params"]

            f[i] = params["f"]

            coord_data_robot = {}
            coord_data_robot[robot_id] = self.coords[i]
            self.fitting.i=i
            self.fitting.omega_sign = self.fitting.omega_from_fft()
            w[i], B[i], E0[i], Ex0[i], Ey0[i], W[i] = self.fitting.simulate([params['RE'], 
                                                                             params['RB'],
                                                                             params['f'],
                                                                             params['k'],
                                                                             params['T']], flag_features_collection=True) 

            rmse[i] = particle_parameters_dict[robot_id]["rmse"]

            if robot_id in [12, 17,19,27,41, 45, 50, 52,54,62, 66, 73, 74,77,79,83]:  #[12, 17,19,27,30, 41, 45, 50, 52,54,62, 65, 73, 74,77,79,83]:
                particle_succ[i] = 1
            else:
                particle_succ[i] = 0
        
        self.data['rmse_particle'] = rmse[:,0].tolist()
        self.data['particle_successfull'] = particle_succ[:,0].astype(int).tolist()
        self.data['w'] = list(w[:,0])
        self.data['W'] = list(W[:,0])
        self.data['f'] = list(f[:,0])
        self.data['B'] = list(B[:,0])
        self.data['E0'] = list(E0[:,0])
        self.data['Ex0'] = list(Ex0[:,0])
        self.data['Ey0'] = list(Ey0[:,0])


    def features_of_trajectory(self):
        """
        Compute kinematic features of trajectories.

        Includes:
        - Mean velocity magnitude
        - Mean curvature (absolute)
        - Curvature variance

        Computation:
        - Velocity via numerical gradient
        - Acceleration via second gradient
        - Curvature using standard 2D formula

        Features added:
        - mean_v
        - mean_cuv
        - var_cuv
        """
        N, T = self.x.shape
        mean_v = np.zeros((N, 1))
        mean_cuv = np.zeros((N, 1))
        cuv_var = np.zeros((N, 1))

        vx = np.gradient(self.x, axis=1)
        vy = np.gradient(self.y, axis=1)
        mean_v = np.mean(np.sqrt(vx**2 + vy**2), axis=1, keepdims=True)
        ax = np.gradient(vx, axis=1)
        ay = np.gradient(vy, axis=1)
        cuv = (vx*ay - vy*ax) / (vx**2 + vy**2 + 1e-8 )**1.5
        mean_cuv = np.mean(np.abs(cuv), axis=1, keepdims=True)
        cuv_var  = np.var(cuv, axis=1, keepdims=True)

        self.data['mean_v'] = mean_v.flatten().tolist()
        self.data['mean_cuv'] = mean_cuv.flatten().tolist()
        self.data['var_cuv'] = cuv_var.flatten().tolist()
        
    def feature_collection(self):
        """
        Main pipeline for feature extraction.

        Steps:
        1. Compute Fourier-based features
        2. If robots are circular:
            - Compute particle-based features
            - Compute difference between Fourier and particle RMSE
        3. Compute trajectory kinematic features
        4. Assemble all features into DataFrame
        5. Save results to CSV file

        Outputs:
        - self.df (pandas DataFrame)
        - CSV file: '{robots_form}/robots_features.csv'

        Prints:
        - Summary statistics of reconstruction errors
        """
        self.fourier_features()
        if self.robots_form == 'circle':
            self.features_of_particle()
            self.data['rmse_div'] = (np.array(self.data['rmse_comp']) -
                np.array(self.data['rmse_particle'])
            ).tolist()
            print(f'rmse_comp mean: {np.mean(self.data["rmse_comp"]):.4f}, rmse_particle mean: {np.mean(self.data["rmse_particle"]):.4f}')
        else:
            print(f'rmse_comp mean: {np.mean(self.data["rmse_comp"]):.4f}')
        self.features_of_trajectory()
        
        
        self.df = pd.DataFrame(self.data)
        print(f'Columns: {self.df.columns}')
        self.df.set_index('robot_id').to_csv(f'{self.robots_form}/robots_features.csv')

    def feature_analysis(self):
        """
        Perform exploratory analysis of extracted features.

        Includes:
        - Correlation heatmap (Pearson correlation)
        - Histograms with KDE for each feature

        Visualization:
        - Heatmap of feature correlations
        - Distribution plots for each feature

        Saving:
        - heatmap.pdf
        - histograms.pdf
        """
        corr = self.df.corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(30,27))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            cmap="twilight",
            vmin=-1,
            vmax=1
        )
        if self.save_flag:
            plt.savefig(self.save_dir/"heatmap.pdf", bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(20, 15))
        n_cols = 4
        n_rows = math.ceil(len(self.df.columns) / n_cols)

        for i, col in enumerate(self.df.columns):
            plt.subplot(n_rows, n_cols, i+1)
            sns.histplot(self.df[col], bins=30, kde=True, color='purple')
            plt.title(col)
            
        plt.tight_layout()
        if self.save_flag:
            plt.savefig(self.save_dir/"histograms.pdf", bbox_inches='tight')
        plt.show()
        

        # df.corr()['particle_successfull'].sort_values(ascending=False)
        



    
   


    












