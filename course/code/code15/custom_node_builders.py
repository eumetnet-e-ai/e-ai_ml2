# custom_node_builders.py
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from anemoi.graphs.nodes import AnemoiDatasetNodes

class AnemoiDatasetNodesWithInterpolation(AnemoiDatasetNodes):
    def __init__(self, dataset, interpolation_method="bilinear", **kwargs):
        super().__init__(dataset, **kwargs)
        self.interpolation_method = interpolation_method

    def interpolate(self, dataset, target_grid):
        """
        Interpolates the data from the lat-lon grid to the target icosahedral grid.

        :param dataset: Original lat-lon dataset
        :param target_grid: Icosahedral grid on which the data will be mapped
        :return: Interpolated data
        """
        lat_lon_coords = dataset.coords['lat'], dataset.coords['lon']
        # Create the target grid coordinates for the icosahedral graph (example)
        target_coords = target_grid  # Replace with actual target grid coordinates

        # Perform interpolation
        interpolated_data = griddata(lat_lon_coords, dataset.values, target_coords, method=self.interpolation_method)
        return interpolated_data

    def update_graph(self, graph, **kwargs):
        # Get the dataset and graph
        dataset = self.dataset
        target_grid = self.target_grid  # Assume you have the target icosahedral grid here

        # Interpolate the data to the target grid
        interpolated_data = self.interpolate(dataset, target_grid)

        # Store the interpolated data in the graph as node attributes
        graph['data']['interpolated'] = interpolated_data
        return graph
