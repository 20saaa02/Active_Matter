import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.optimize import differential_evolution
import json
import os
import math
from pathlib import Path

from all_trajectories_figure import AllTrajectoriesFigure  
from colors_manager import ColorManager


class ParticleTrajectoryFitting:
    def __init__(self, coord_data: Dict, robot_ids: List = None, maxiter: int = 0, colors: ColorManager = None, robots_form: str = 'circle', time_period: List = [0,200],
                 show_flag: bool = True):
        if robots_form != 'circle':
            raise ValueError("Invalid robots_form. Expected 'circle'.")
        self.t0, self.t1 = time_period
        self.coords = np.array([coord_data[r] for r in coord_data])[:,self.t0:self.t1]
        self.maxiter = maxiter
        self.robot_ids = robot_ids if robot_ids is not None else list(coord_data.keys())
        self.particle_parameters_dict = {}
        self.colors= colors
        
        self.show_flag = show_flag

        self.x, self.y = self.coords[:,:,0], self.coords[:,:,1]
        if (self.maxiter==0) and (robot_ids is None):
            self.coord_data_full = {}
        else:
            self.coord_data_full = None
            
        self.i = 0

    def omega_from_fft(self):
        x = self.x[self.i,:50]
        y = self.y[self.i,:50]
        dt = 1
        t = np.arange(len(x)) * dt
        Zf = np.fft.fft(x-np.mean(x)+1j*(y - np.mean(y)))
        freqs = np.fft.fftfreq(len(t), dt) 
        omega = 2 * np.pi * freqs     
        mask = omega != 0
        idx = np.argmax(np.abs(Zf[mask]))
        omega_peak = omega[mask][idx]
        return -np.sign(omega_peak)

    def simulate(self, params, show_flag=False, save_flag = False, flag_features_collection = False):
        RE, RB, f, k, T = params
        m = 60
        q = 1.2
        if  (self.robot_ids[self.i] == 12):
            x0 = self.x[self.i,:25]
            y0 = self.y[self.i,:25]
        elif (self.robot_ids[self.i] == 73) or (self.robot_ids[self.i] == 34):
            x0 = self.x[self.i,:10]
            y0 = self.y[self.i,:10]
        else:
            x0 = self.x[self.i,:5]
            y0 = self.y[self.i,:5]
        vx_all = np.gradient(x0, 1)
        vy_all = np.gradient(y0, 1)
        vx = np.mean(vx_all)
        vy = np.mean(vy_all)

        dt = 0.01
    
        x_p = np.zeros(round(T/dt)+1)
        y_p = np.zeros(round(T/dt)+1)
        x_p[0] = self.x[self.i,0]
        y_p[0] = self.y[self.i,0]

        w = self.omega_sign * np.linalg.norm([vx,vy])/RB
        B = m/q * w

        W = k * w
        
        E0 = RE * m/q * np.abs(W) * np.abs(np.sign(np.cos(f))*W + np.sign(np.sin(f))*w)
        Ex0 = E0 * np.cos(f)
        Ey0 = E0 * np.sin(f)
        fi = math.atan(np.abs(Ey0)/np.abs(Ex0))

        D = - q/m * np.sign(Ex0*Ey0) * np.linalg.norm([Ex0,Ey0])/(np.sign(Ex0)*W + np.sign(Ey0)*w)
        C = -D * np.sign(Ex0/Ey0)
        B = vy - D * np.cos(fi)
        A = vx - C * np.sin(fi)
        
        if flag_features_collection:
            return w, B, E0, Ex0, Ey0, W

        for h in range(round(T/dt)):
            Time = (h+1) * dt
            x_p[h+1] = x_p[0] + B/w + C/W*np.cos(fi) + A/w*np.sin(w*Time) - B/w*np.cos(w*Time) - C/W*np.cos(W*Time + fi)
            y_p[h+1] = y_p[0] - A/w - D/W*np.sin(fi) + A/w*np.cos(w*Time) + B/w*np.sin(w*Time) + D/W*np.sin(W*Time + fi)
        
        x_model = x_p[0 : math.floor(len(x_p)/len(self.x[self.i]))*len(self.x[self.i]) : math.floor(len(x_p)/len(self.x[self.i]))]
        y_model = y_p[0 : math.floor(len(x_p)/len(self.x[self.i]))*len(self.x[self.i]) : math.floor(len(x_p)/len(self.x[self.i]))]

        coords_model = np.stack((x_model, y_model), axis=1)
        if show_flag:
            coord_data_robot_particle = {}
            coord_data_robot_particle[self.robot_ids[self.i]] = self.coords[self.i]
            coord_data_robot_particle[self.robot_ids[self.i] + 100] = coords_model
            if self.maxiter>0:
                save_fig=False 
            else:                
                save_fig=True
            fig = AllTrajectoriesFigure(
                coord_data=coord_data_robot_particle,
                robot_ids=list(coord_data_robot_particle.keys()),
                colors = self.colors,
                time_period=[0, len(self.x[0,:])],
                save_flag=save_fig
                )
            fig.temporal_trajectories()
            fig.space_trajectories()

        if (self.maxiter==0) and (self.coord_data_full is not None):
                self.coord_data_full[self.robot_ids[self.i]] = self.coords[self.i]
                self.coord_data_full[self.robot_ids[self.i] + 100] = coords_model


        if save_flag:
            if os.path.exists("circle/particle_coordinates.npz"):
                    particle_coordinates = dict(np.load("circle/particle_coordinates.npz"))
            else:
                    particle_coordinates={}

            particle_coordinates[str(self.robot_ids[self.i])] = coords_model
            np.savez(f'circle/particle_coordinates.npz', **particle_coordinates)

        return x_model, y_model
    
    def loss(self, params):
        RE, RB, f, k, T = params
        eps = 1e-6
        if abs(k) < eps or abs(np.cos(f)) < eps or abs(np.sin(f)) < eps:
            return 1e12
        x_model, y_model = self.simulate(params)
        return np.sqrt(np.mean((self.x[self.i,:] - x_model)**2 + (self.y[self.i,:] - y_model)**2)/2)

    def fit_particle_trajectories(self):
        particle_parameters_file = "circle/fit_all_particle_results.json"
        if os.path.exists(particle_parameters_file):
            with open(particle_parameters_file, "r") as f:
                particle_parameters = json.load(f)
        else:
            particle_parameters = []

        particle_parameters_dict = {entry["robot_id"]: entry for entry in particle_parameters}

        for i, robot_id in enumerate(self.robot_ids):
            self.i = i
            self.omega_sign = self.omega_from_fft()
            bounds = [
                (10, 250),          # RE
                (10, 80),           # RB
                (-np.pi, np.pi),    # f
                (-1,1),             # k
                (100,600)           # T
            ]
            if self.maxiter>0:
                result = differential_evolution(self.loss, bounds, maxiter = self.maxiter, disp=True)
                print('Result of differential evolution:')
                print('RE, RB, f, k, T:', result.x)
                print('RMSE:',result.fun)

                new_entry = {
                    "robot_id": int(robot_id),
                    "params": {
                        "RE": float(result.x[0]),
                        "RB": float(result.x[1]),
                        "f": float(result.x[2]),
                        "k": float(result.x[3]),
                        "T": float(result.x[4])
                    },
                    "rmse": float(result.fun)
                }

                if robot_id in particle_parameters_dict:
                    if particle_parameters_dict[robot_id]["rmse"] > new_entry["rmse"]:
                        print(f"Updated results for robot_id {robot_id} with better RMSE {new_entry['rmse']} ({particle_parameters_dict[robot_id]['rmse']})")
                        particle_parameters_dict[robot_id] = new_entry
                        save_flag = True
                    else:
                        save_flag = False
                else:
                    particle_parameters_dict[robot_id] = new_entry
                    save_flag = True
                self.simulate(result.x, show_flag=self.show_flag, save_flag=save_flag)

            self.simulate([particle_parameters_dict[robot_id]["params"]['RE'], 
                    particle_parameters_dict[robot_id]["params"]['RB'], 
                    particle_parameters_dict[robot_id]["params"]['f'],
                    particle_parameters_dict[robot_id]["params"]['k'],
                    particle_parameters_dict[robot_id]["params"]['T']], show_flag=self.show_flag, save_flag=True)
            

            with open(particle_parameters_file, "w") as f:
                json.dump(list(particle_parameters_dict.values()), f, indent=4)

        if (self.maxiter == 0) and (self.coord_data_full is not None):
            fig = AllTrajectoriesFigure(
                        coord_data=self.coord_data_full,
                        colors = self.colors,
                        time_period=[0, len(self.x[0,:])],
                        save_flag=True
                        )
            fig.space_trajectories()
