import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def compute_color_histograms(cloud, nbins = 96, using_hsv=True, bins_range=(0, 256)):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    
    # TODO: Compute histograms
    r_hist = np.histogram(channel_1_vals, bins=nbins, range=bins_range)
    g_hist = np.histogram(channel_2_vals, bins=nbins, range=bins_range)
    b_hist = np.histogram(channel_3_vals, bins=nbins, range=bins_range)
    # TODO: Concatenate and normalize the histograms
    rgb_features = np.concatenate((r_hist[0], g_hist[0], b_hist[0])).astype(np.float64)
    normed_features = rgb_features  
    return normed_features 


def compute_normal_histograms(normal_cloud, nbins = 96, bins_range=(0, 256)):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # TODO: Compute histograms of normal values (just like with color)
    x_hist = np.histogram(norm_x_vals, bins=nbins, range=bins_range)
    y_hist = np.histogram(norm_y_vals, bins=nbins, range=bins_range)
    z_hist = np.histogram(norm_z_vals, bins=nbins, range=bins_range)
    # TODO: Concatenate and normalize the histograms
    xyz_features = np.concatenate((x_hist[0], y_hist[0], z_hist[0])).astype(np.float64)
    normed_features = xyz_features 
    return normed_features