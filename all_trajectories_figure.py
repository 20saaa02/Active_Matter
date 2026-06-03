import dill as pickle
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.colors as pc
from typing import List, Dict
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
from pathlib import Path
import numpy as np
from IPython.display import clear_output
import seaborn as sns
import time

from colors_manager import ColorManager


class AllTrajectoriesFigure():
    """
    Visualization class for robot trajectories.

    Purpose
    -------
    Provides multiple visualization modes for analyzing robot motion:
    - Spatial trajectories (Y(X))
    - Temporal trajectories (X(t), Y(t))
    - Animated motion (for circular robots)
    - Cluster-based trajectory visualization

    Supports:
    ---------
    - Individual robots
    - Subsets of robots
    - Robot-particle pairs
    - Clustered groups of robots

    Parameters
    ----------
    coord_data : Dict
        Dictionary mapping robot_id -> trajectory array of shape (T, 2)
    robot_ids : List, optional
        List of robot IDs to visualize (default: all)
    colors : ColorManager, optional
        Object responsible for assigning colors to robots
    robots_form : str
        Shape of robots ('circle' or 'oval')
    time_period : List[int]
        Time interval [t0, t1] for visualization
    save_flag : bool
        Whether to save generated figures
    save_dir : str
        Directory name for saving figures
    """
    def __init__(self, coord_data: Dict, robot_ids: List = None, colors: ColorManager = None, robots_form: str = 'circle', time_period: List = [0,200],
                 save_flag: bool = False, save_dir: str = 'figures'):

        self.coords = np.array([coord_data[r] for r in coord_data])
        self.x, self.y = self.coords[:,:,0], self.coords[:,:,1]
        self.robot_ids = robot_ids if robot_ids is not None else list(coord_data.keys())
        print(self.robot_ids)
        self.robots_form = robots_form

        self.colors = colors

        self.save_flag = save_flag
        if self.save_flag:
            self.save_dir = Path(robots_form) / save_dir
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.t0, self.t1 = time_period
        

        if robot_ids is None:
            cx = (self.x.max() + self.x.min()) / 2
            cy = (self.y.max() + self.y.min()) / 2
            r_px = (self.x - cx)**2 + (self.y - cy)**2
            R = np.sqrt(r_px.max())* 1.11

            self.circle_shape = dict(
                type="circle",
                xref="x",
                yref="y",
                x0=cx - R,
                y0=-cy - R,
                x1=cx + R,
                y1=-cy + R,
                line=dict(color="black", width=2, dash="dash")
            )
            self.range_x = [cx-R, cx+R]
            self.range_y = [-cy-R, -cy+R]
        else:
            self.circle_shape = None
            self.range_x = [self.x.min()+110, self.x.max()+110]
            self.range_y = [-self.y.max()-110, -self.y.min()-110]

        self.x, self.y = self.coords[:,self.t0:self.t1,0], self.coords[:,self.t0:self.t1,1]

        self.has_particles = any((r >= 100) and (r<200) for r in self.robot_ids)
        self.has_fourier = any(r >= 200 for r in self.robot_ids)

    def add_start_end(self, fig, x, y, color, showlegend: bool = False):
        """
        Adds markers for start and end points of a trajectory.
        """
        fig.add_trace(go.Scatter(
            x=[x[0]], y=[y[0]],
            mode='markers',
            marker=dict(color=color, symbol='circle', size=5),
            line=dict(color='black', width=1),
            name=f'Object starts', showlegend = showlegend #f'Start, object {robot_id}'
        ))

        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y[-1]],
            mode='markers',
            marker=dict(color=color, symbol='x', size=5),
            line=dict(color='black', width=1),
            name=f'Object ends', showlegend = showlegend
        ))
    
    def is_particle(self, robot_id):
        """
        Checks whether a given ID corresponds to a particle.
        """
        return 100 <= robot_id < 200

    def is_fourier(self, robot_id):
        """
        Checks whether a given ID corresponds to a Fourier-based trajectory.
        """
        return robot_id >= 200


    def space_trajectories(self):
        """
        Plots spatial trajectories Y(X) for selected robots.

        Features
        --------
        - Distinguishes robots, particles, and Fourier trajectories
        - Marks start and end points
        - Optionally overlays bounding circle
        - Supports saving as interactive HTML

        Output
        ------
        Plotly interactive figure
        """
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        if self.circle_shape is not None:
            fig.update_layout(
                shapes=[self.circle_shape])

        self.add_start_end(fig, [None], [None], 'black',showlegend=True )
        for i, robot_id in enumerate(self.robot_ids):
                
                if robot_id<100:
                    color = self.colors.get(robot_id)
                    fig.add_trace(go.Scatter(x=self.x[i], y=-self.y[i], mode='lines', 
                                            line=dict(color=color, width=1), name=f'Y(X) trajectory, {self.robots_form} object {robot_id}'), 
                                            row=1,col=1)
                    self.add_start_end(fig, self.x[i], -self.y[i], color)
                elif self.is_particle(robot_id):
                    color = self.colors.get(robot_id)
                    fig.add_trace(go.Scatter(x=self.x[i], y=-self.y[i], mode='lines', 
                                            line=dict(color=color, width=1, dash='dash'), name=f'Y(X) trajectory, object {robot_id} - <br>'
                                            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; particle of {self.robots_form} object {robot_id-100}'), 
                                            row=1,col=1)
                    self.add_start_end(fig, self.x[i], -self.y[i], color)
                elif self.is_fourier(robot_id):
                    color = self.colors.get(robot_id)
                    self.save_flag = False
                    fig.add_trace(go.Scatter(x=self.x[i], y=-self.y[i], mode='lines', 
                                            line=dict(color=color, width=1, dash='dashdot'), name=f'Y(X) trajectory, object {robot_id} - <br> '
                                             f'partial inverse complex Fourier <br>'
                                             f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; transform of object {robot_id-200}'),
                                            row=1,col=1)
                    self.add_start_end(fig, self.x[i], -self.y[i], color)

        fig.update_layout(
            height=700,
            width=1000,
            plot_bgcolor='white',
            font=dict(family="Inter", size=13),
            margin=dict(l=70, r=10, t=30, b=60),
            legend=dict(
                x=1,     
                y=1, 
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(0,0,0,0)'
            ), 
            xaxis=dict(
                title_text="X",
                showline=True, linewidth=1, linecolor='black',
                mirror=True, gridcolor='lightgrey'),
            yaxis=dict(
                title_text="Y(X)", 
                showline=True, linewidth=1, linecolor='black',
                mirror=True, gridcolor='lightgrey')
        )
        fig.show()

        if self.save_flag:
            s = '_'.join(map(str, self.robot_ids))
            file_path = self.save_dir / f"Y(X)_trajectories_{s}.html"
            fig.write_html(file_path)
    
    def robots_moving(self):
        """
        Creates animated visualization of robots moving over time.

        Limitations
        -----------
        - Works only for circular robots
        - Does not support particles

        Output
        ------
        Animated Plotly figure with play/pause controls
        """
        if self.has_particles:
            print('Robots with IDs greater than 100 are considered particles and will not be animated.')
            return
        if self.robots_form != 'circle':
            print('Animation is only implemented for circular robots. Please set robots_form to "circle".')
            return

        fig = go.Figure()

        if self.circle_shape is not None:
            fig.update_layout(shapes=[self.circle_shape])

        theta = np.linspace(0, 2*np.pi, 40)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        for i, robot_id in enumerate(self.robot_ids):
            color = self.colors.get(robot_id)
            circle_x = self.x[i, 0] + 110 * cos_t
            circle_y = -self.y[i, 0] + 110 * sin_t

            fig.add_trace(go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                fill='toself',
                line=dict(color=color),
                name=f'{self.robots_form} object {robot_id}'
            ))

        frames = []
        for t in range(self.t0, self.t1):
            frame_data = []
            for i in range(len(self.robot_ids)):
                circle_x = self.x[i, t-self.t0] + 110 * cos_t
                circle_y = -self.y[i, t-self.t0] + 110 * sin_t

                frame_data.append(go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode='lines',
                    fill='toself'
                ))
            frames.append(go.Frame(data=frame_data, name=str(t)))
        fig.frames = frames

        fig.update_layout(
            height=850,
            width=1000,
            plot_bgcolor='white',
            font=dict(family="Inter", size=13),
            margin=dict(l=70, r=10, t=30, b=60),
            xaxis=dict(
                title='X',
                showline=True, linewidth=1, linecolor='black',
                mirror=True, gridcolor='lightgrey',
                range=self.range_x
            ),
            yaxis=dict(
                title='Y(X)',
                showline=True, linewidth=1, linecolor='black',
                mirror=True, gridcolor='lightgrey',
                range=self.range_y
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="▶ Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 50, "redraw": True},
                                    "fromcurrent": True}]),
                    dict(label="⏸ Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
                ]
            )],
            sliders = [dict(
            active=0,
            currentvalue=dict(
                prefix="Time: ",
                font=dict(size=14)
            ),
            steps=[
                dict(
                    method="animate",
                    args=[
                        [str(t)],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0}
                        }
                    ],
                    label=str(t)
                )
                for t in range(self.t0, self.t1)
            ]
        )]
        )

        fig.show()

    def temporal_trajectories(self):
        """
        Plots temporal trajectories X(t) and Y(t).

        Features
        --------
        - Separate subplots for X(t) and Y(t)
        - Supports robots, particles, and Fourier data
        - Optional saving to HTML

        Output
        ------
        Plotly figures for each robot
        """
        t = np.arange(self.t0, self.t1)
        for i, robot_id in enumerate(self.robot_ids):
                if robot_id<100:
                    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                    color = self.colors.get(robot_id)
                    fig1.add_trace(go.Scatter(
                            x=t, y=self.x[i], mode='lines',
                            line=dict(color=color, width=1), name=f'X(t), Y(t) trajectory, {self.robots_form} object {robot_id}',
                            legendgroup='x'), 
                            row=1, col=1)
                    fig1.add_trace(go.Scatter(
                            x=t, y=self.y[i], mode='lines',
                            line=dict(color=color, width=1), showlegend=False  ), 
                            row=2, col=1)
                    legend_tracegroupgap = 305
                elif self.is_particle(robot_id):
                    if self.robots_form != 'circle': 
                        print('Particle trajectories are only implemented for circular robots. Please set robots_form to "circle".')
                        return
                    color = self.colors.get(robot_id)
                    fig1.add_trace(go.Scatter(
                            x=t, y=self.x[i], mode='lines',
                            line=dict(color=color, width=1,dash='dash'), name=f'X(t), Y(t) trajectory, object {robot_id} — <br> '
                            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; particle of {self.robots_form} object {robot_id-100}',
                            legendgroup='x'), 
                            row=1, col=1)
                    fig1.add_trace(go.Scatter(
                            x=t, y=self.y[i], mode='lines',
                            line=dict(color=color, width=1, dash='dash'),showlegend=False  ), 
                            row=2, col=1)
                    legend_tracegroupgap = 260
                elif self.is_fourier(robot_id):
                    self.save_flag = False
                    color = self.colors.get(robot_id)
                    fig1.add_trace(go.Scatter(
                            x=t, y=self.x[i], mode='lines',
                            line=dict(color=color, width=1,dash='dashdot'), name=f'X(t), Y(t) trajectory, object {robot_id} — <br>'
                            f'partial inverse complex Fourier <br>'
                            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; transform of object {robot_id-200}',
                            legendgroup='x'), 
                            row=1, col=1)
                    fig1.add_trace(go.Scatter(
                            x=t, y=self.y[i], mode='lines',
                            line=dict(color=color, width=1, dash='dashdot'), showlegend=False  ), 
                            row=2, col=1)
                    legend_tracegroupgap = 260


                    
                fig1.update_layout(
                    height=700,
                    width=1000,
                    plot_bgcolor='white',
                    font=dict(family="Inter", size=16),
                    margin=dict(l=40, r=10, t=20, b=60),
                    legend=dict(
                        x=0.70,      
                        y=1,   
                        xanchor='left',
                        yanchor='top',
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    legend_tracegroupgap=legend_tracegroupgap,
                )

                fig1.update_xaxes(
                    tickmode='array',
                    showgrid=True, mirror=True,showline=True, linewidth=1, linecolor='black',#gridcolor='lightgrey',
                    row=1, col=1
                )
                fig1.update_xaxes(
                    title_text="t",
                    tickmode='array',
                    showgrid=True, mirror=True, showline=True, linewidth=1, linecolor='black',#gridcolor='lightgrey',
                    row=2, col=1
                )
                fig1.update_yaxes(
                    title_text="X(t)",
                    row=1, col=1,
                    showline=True, linewidth=1, linecolor='black',
                    mirror=True, #gridcolor='lightgrey'
                )
                fig1.update_yaxes(
                    title_text="Y(t)", 
                    row=2, col=1,
                    showline=True, linewidth=1, linecolor='black',
                    mirror=True, #gridcolor='lightgrey'
                    )

                if self.has_particles or self.has_fourier:
                    if self.is_particle(robot_id) or self.is_fourier(robot_id):
                        fig1.show()
                else:
                    fig1.show()

                if self.save_flag:
                    if self.has_particles:
                        if self.is_particle(robot_id):
                            file_path = self.save_dir / f"X(t)_Y(t)_trajectories_{robot_id-100}_{robot_id}.html"
                            fig1.write_html(file_path)
                    else:
                        file_path = self.save_dir / f"X(t)_Y(t)_trajectories_{robot_id}.html"
                        fig1.write_html(file_path)

    def plotly_to_matplotlib(self, color):
        """
        Converts Plotly RGB color string to Matplotlib-compatible tuple.
        """
        nums = color.strip('rgb()').split(',')
        return tuple(int(n)/255 for n in nums)

    def clusters_trajectories(self, df):
        """
        Visualizes trajectories grouped by clusters.

        Features
        --------
        - Spatial trajectories colored by cluster
        - Temporal trajectories per cluster
        - Pairplot of features colored by cluster

        Output
        ------
        - Plotly trajectory plots
        - Seaborn pairplot
        """
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        self.add_start_end(fig, [None], [None], 'black', showlegend=True)
        n_clusters = df['cluster'].nunique()
        colors = pc.sample_colorscale(px.colors.cyclical.Twilight, n_clusters+2)
        colors = colors[1:-1]
        
        clusters_dict = clusters_dict = df.groupby('cluster').groups
        
        for i in clusters_dict.keys():
            color = colors[i]
            for j, robot_id in enumerate(clusters_dict[i]):
                idx = self.robot_ids.index(robot_id)
                fig.add_trace(go.Scatter(x=self.x[idx], y=-self.y[idx], mode='lines', 
                                        line=dict(color=color, width=1), name=f'Y(X) trajectories objects of cluster {i}', showlegend=(j == 0)), 
                                        row=1,col=1)
                self.add_start_end(fig, self.x[idx], -self.y[idx], color)

        fig.update_layout(
            height=700,
            width=1000,
            plot_bgcolor='white',
            font=dict(family="Inter", size=13),
            margin=dict(l=70, r=10, t=30, b=60),
            shapes=[self.circle_shape],
            legend=dict(
                x=1,     
                y=1, 
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(0,0,0,0)'
            ), 
            xaxis=dict(
                title_text="X",
                showline=True, linewidth=1, linecolor='black',
                mirror=True, gridcolor='lightgrey'),
            yaxis=dict(
                title_text="Y(X)", 
                showline=True, linewidth=1, linecolor='black',
                mirror=True, gridcolor='lightgrey')
        )
        fig.show()

        if self.save_flag:
            file_path = self.save_dir / f"Y(X)_clusters_trajectories.html"
            fig.write_html(file_path)

        t = np.arange(self.t0, self.t1)
    
        for i in clusters_dict.keys():
            fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
            for j, robot_id in enumerate(clusters_dict[i]):
                idx = self.robot_ids.index(robot_id)
                color = self.colors.get(robot_id)
                min_x, min_y = self.x[idx].min(), self.y[idx].min()
                fig1.add_trace(go.Scatter(
                        x=t, y=self.x[idx]-min_x, mode='lines',
                        line=dict(color=color, width=1), name=f'X(t), Y(t) trajectory, {self.robots_form} object {robot_id}'), 
                        row=1, col=1)
                fig1.add_trace(go.Scatter(
                        x=t, y=self.y[idx]-min_y, mode='lines',
                        line=dict(color=color, width=1),showlegend=False   ), 
                        row=2, col=1)
            
            fig1.update_layout(
                    height=700,
                    width=850,
                    plot_bgcolor='white',
                    font=dict(family="Inter", size=16),
                    margin=dict(l=40, r=10, t=20, b=60),
                    legend=dict(
                        x=1,      
                        y=1,   
                        xanchor='left',
                        yanchor='top',
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    legend_tracegroupgap=1,
                )


            fig1.update_xaxes(
                tickmode='array',
                showgrid=True, mirror=True,showline=True, linewidth=1, linecolor='black',#gridcolor='lightgrey',
                row=1, col=1
            )
            fig1.update_xaxes(
                title_text="t",
                tickmode='array',
                showgrid=True, mirror=True, showline=True, linewidth=1, linecolor='black',#gridcolor='lightgrey',
                row=2, col=1
            )
            fig1.update_yaxes(
                title_text="X(t)",
                row=1, col=1,
                showline=True, linewidth=1, linecolor='black',
                mirror=True, #gridcolor='lightgrey'
            )
            fig1.update_yaxes(
                title_text="Y(t)", 
                row=2, col=1,
                showline=True, linewidth=1, linecolor='black',
                mirror=True, #gridcolor='lightgrey'
                )

            fig1.show()

            if self.save_flag:
                file_path = self.save_dir / f"X(t)_Y(t)_trajectories_of_cluster_{i}.html"
                fig1.write_html(file_path)

        colors = [self.plotly_to_matplotlib(c) for c in colors]
        sns.pairplot(
            df,
            vars=[col for col in df.columns if col != 'cluster'],
            hue='cluster',
            palette=colors,
            plot_kws={'alpha': 0.8}
        )
        if self.save_flag:
            plt.savefig(self.save_dir/"pair_plot.pdf", bbox_inches='tight')
        plt.show()
        
        
    def pair_trajectories(self):
        t = np.arange(self.t0, self.t1)
        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        color = self.colors.get(self.robot_ids[0])
        fig1.add_trace(go.Scatter(
                            x=t, y=self.x[0], mode='lines',
                            line=dict(color=color, width=1), name=f'X(t), Y(t) trajectory, {self.robots_form} object {self.robot_ids[0]}',
                            legendgroup='x'), 
                            row=1, col=1)
        fig1.add_trace(go.Scatter(
                            x=t, y=self.y[0], mode='lines',
                            line=dict(color=color, width=1), showlegend=False  ), 
                            row=2, col=1)
        color = self.colors.get(self.robot_ids[1])
        fig1.add_trace(go.Scatter(
                            x=t, y=self.x[1], mode='lines',
                            line=dict(color=color, width=1), name=f'X(t), Y(t) trajectory, {self.robots_form} object {self.robot_ids[1]}',
                            legendgroup='x'), 
                            row=1, col=1)
        fig1.add_trace(go.Scatter(
                            x=t, y=self.y[1], mode='lines',
                            line=dict(color=color, width=1), showlegend=False  ), 
                            row=2, col=1)
        legend_tracegroupgap = 305


        fig1.update_layout(
                    height=700,
                    width=1000,
                    plot_bgcolor='white',
                    font=dict(family="Inter", size=16),
                    margin=dict(l=40, r=10, t=20, b=60),
                    legend=dict(
                        x=0.70,      
                        y=1,   
                        xanchor='left',
                        yanchor='top',
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    legend_tracegroupgap=legend_tracegroupgap,
                )

        fig1.update_xaxes(
            tickmode='array',
            showgrid=True, mirror=True,showline=True, linewidth=1, linecolor='black',#gridcolor='lightgrey',
            row=1, col=1
        )
        fig1.update_xaxes(
            title_text="t",
            tickmode='array',
            showgrid=True, mirror=True, showline=True, linewidth=1, linecolor='black',#gridcolor='lightgrey',
            row=2, col=1
        )
        fig1.update_yaxes(
            title_text="X(t)",
            row=1, col=1,
            showline=True, linewidth=1, linecolor='black',
            mirror=True, #gridcolor='lightgrey'
        )
        fig1.update_yaxes(
            title_text="Y(t)", 
            row=2, col=1,
            showline=True, linewidth=1, linecolor='black',
            mirror=True, #gridcolor='lightgrey'
            )
        
        for x_val in [32,36, 90,95, 146,152,192,198]:  
            fig1.add_vline(
                x=x_val,
                line=dict(color='gray', width=1.5, dash='dot'),
                row='all', col=1
            )

        if self.save_flag:
            file_path = self.save_dir / f"X(t)_Y(t)_trajectories_{self.robot_ids[0]}_{self.robot_ids[1]}.html"
            fig1.write_html(file_path)
        
        fig1.show()


            