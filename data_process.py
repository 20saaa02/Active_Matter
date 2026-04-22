import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import MinMaxScaler

# ==========================
# === DATA PROCESSOR CLASS ===
# ==========================
class DataProcessor:
    """
    Handles raw data extraction and normalization.

    Attributes
    ----------
    data : List
        Raw input data.
    coord_data : dict
        Dictionary storing coordinates per object.
    angle_data : dict
        Dictionary storing angle values per object.
    normalized_coord_data : dict
        Dictionary storing normalized coordinates.
    """
    def __init__(self, data: List):
        self.data = data
        self.coord_data = {}
        self.angle_data = {}
        self.normalized_coord_data = {}

    # === Data extraction ===
    def extract_data(self) -> Tuple[Dict, Dict]:
        """Extracts coordinate and angle data from raw input."""
        coord_data = {}
        angle_data = {}

        for inner_list in self.data:
            for item in inner_list:
                key = item[0]
                angle = np.deg2rad(item[1]) 
                coordinates = item[2]

                if key not in coord_data:
                    coord_data[key] = []
                    angle_data[key] = []

                coord_data[key].append(coordinates)
                angle_data[key].append(angle)

        self.coord_data = coord_data
        self.angle_data = angle_data
        return coord_data, angle_data

    # === Normalization ===
    @staticmethod
    def normalize_to_normal_distribution(data: np.ndarray) -> np.ndarray:
        """Applies Box-Cox transformation and MinMax scaling to [0,1]."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        return normalized_data, scaler

    def normalize_all_coordinates(self) -> Dict:
        """Normalizes all coordinates of all objects."""
        all_x, all_y = [], []
        
        for points in self.coord_data.values():
            points_array = np.array(points)
            all_x.extend(points_array[:, 0])
            all_y.extend(points_array[:, 1])

        x_normalized, scaler_x = self.normalize_to_normal_distribution(np.array(all_x))
        y_normalized, scaler_y = self.normalize_to_normal_distribution(np.array(all_y))

        self.scaler_x,  self.scaler_y = scaler_x, scaler_y

        idx = 0
        for obj_id, points in self.coord_data.items():
            points_array = np.array(points)
            num_points = len(points_array)
            obj_x_norm = x_normalized[idx:idx + num_points]
            obj_y_norm = y_normalized[idx:idx + num_points]
            self.normalized_coord_data[obj_id] = np.column_stack((obj_x_norm, obj_y_norm))
            idx += num_points

        return self.normalized_coord_data
    
    def transform_new_robot(self, x_new, y_new, id_new):
        x_scaled = self.scaler_x.transform(x_new.reshape(-1,1))
        y_scaled = self.scaler_y.transform(y_new.reshape(-1,1))

        key = id_new

        self.normalized_coord_data[key] = np.column_stack((x_scaled, y_scaled))

        return self.normalized_coord_data, key
