import plotly.express as px
import plotly.colors as pc
import numpy as np

class ColorManager:
    def __init__(self, coord_data, colorscale=px.colors.cyclical.Twilight):
        self.robot_ids = list(coord_data.keys())
        self.n = len(self.robot_ids)
        k = 7
        self.colors = pc.sample_colorscale(colorscale, self.n+k*2)
        self.colors = self.colors[k:-k]
        self.id_to_color = {
            rid: self.colors[i]
            for i, rid in enumerate(self.robot_ids)
        }

        self.colors_add = pc.sample_colorscale(px.colors.cyclical.IceFire, self.n+k*2)
        mid = len(self.colors_add) // 2
        self.colors_add = self.colors_add[:mid - k] + self.colors_add[mid + k:]
        self.id_to_color_add = {
            rid: self.colors_add[i]
            for i, rid in enumerate(self.robot_ids)
        }
    def get(self, robot_id):
        if robot_id<100:
            return self.id_to_color.get(robot_id, 'black')
        else:
            return self.id_to_color_add.get(robot_id%100, 'black')

    