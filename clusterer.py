from typing import List, Dict
from pathlib import Path
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from colors_manager import ColorManager
from all_trajectories_figure import AllTrajectoriesFigure  


class Clusterer:
    """
    Clusterer performs clustering of robots based on extracted features.

    It uses:
    - StandardScaler for feature normalization
    - KMeans for clustering

    It also provides:
    - Saving clustering results
    - Printing cluster composition
    - Visualization of clustered trajectories

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing extracted features (indexed by robot_id)
    coord_data : Dict[int, np.ndarray], optional
        Dictionary of trajectories (robot_id → trajectory)
        Required for visualization and saving cluster structures
    n_clusters : int, default 6
        Number of clusters for KMeans
    robots_form : str, default 'circle'
        Type of robots (used for saving paths and visualization)
    colors : ColorManager, optional
        Color manager for consistent visualization
    time_period : List[int], default [0, 200]
        Time range used for trajectory visualization
    save_flag : bool, default False
        Whether to save clustering results to files
    """
    def __init__(self, df, coord_data:Dict = None, n_clusters: int = 6, robots_form: str = 'circle', colors: ColorManager =  None, time_period: List = [0, 200],
                 save_flag :bool = False):
        self.df = df
        self.coord_data = coord_data
        self.n_clusters = n_clusters
        self.t0, self.t1 = time_period[0], time_period[1]
        self.colors = colors
        self.robots_form = robots_form
        self.save_flag = save_flag
        self.save_dir = Path(robots_form) 

    def cluster_creating(self):
        """
        Perform clustering of robots based on feature data.

        Steps:
        1. Normalize features using StandardScaler
        2. Apply KMeans clustering
        3. Assign cluster labels to each robot
        4. Save updated feature table to CSV
        5. Print cluster composition

        Output:
        - Updates self.df with 'cluster' column
        - Saves updated CSV: '{robots_form}/robots_features.csv'

        If save_flag=True:
        - Saves hierarchical cluster structure to JSON:
            levels_robots_ids.json with:
                - micro: individual robots
                - meso_cluster: clusters of robots
                - meso_union: all robots together
        """
        scaler = StandardScaler()
        X = self.df
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.df['cluster'] = kmeans.fit_predict(X_scaled)
        self.df.to_csv(f'{self.robots_form}/robots_features.csv')
        print(self.df.sort_values(by=['cluster', self.df.index.name]))
        clusters = sorted(self.df['cluster'].unique())
        for cluster in clusters:
            robot_ids = self.df[self.df['cluster'] == cluster].index.tolist()
            print(f'Cluster {cluster} contain robots: {robot_ids[:]}')
        if self.save_flag:
            levels_robots_ids = {}
            levels_robots_ids['micro'] = [[int(x)] for x in self.coord_data.keys()]
            levels_robots_ids['meso_cluster'] = [
                [int(x) for x in self.df[self.df['cluster'] == cluster].index.tolist()]
                for cluster in sorted(self.df['cluster'].unique())
            ]
            levels_robots_ids['meso_union'] = [[int(x) for x in self.coord_data.keys()]]
            file_path = self.save_dir/'levels_robots_ids.json'
            with open(file_path, 'w') as f:
                json.dump(levels_robots_ids, f, indent=4, default=int)

 
    def cluster_analysis(self):
        """
        Visualize clustered trajectories.

        Uses AllTrajectoriesFigure to:
        - Plot trajectories grouped by cluster
        - Show spatial and temporal structure of clusters
        """
        fig = AllTrajectoriesFigure(coord_data=self.coord_data, robots_form=self.robots_form, colors = self.colors,time_period=[self.t0, self.t1], save_flag=self.save_flag)
        fig.clusters_trajectories(self.df)
    

