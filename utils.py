import numpy as np
import matplotlib.pyplot as plt
import os

def visualize(x, y):
    """
    Visualize the frame pair.
    Args:
        x: the initial frame
        y: the generated frame
    Returns:
        fig: the matplotlib figure object
    """
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", 
                "#bcbd22", "#17becf"]

    def get_cell_center_geom(pointnum=64, grad=10):
        cell_center_geom = np.geomspace(1, grad, pointnum) / 2
        cell_center_geom[1:] += np.geomspace(1, grad, pointnum).cumsum()[:-1]
        total_length = np.geomspace(1, grad, pointnum).sum()
        cell_center_geom /= total_length
        return cell_center_geom
    
    xcell1, xcell2, xcell3 = 50, 92, 50  # 192
    ycell1, ycell2 = 120, 136  # 256
    
    real_x1 = -0.015 + ((-0.125) - (-0.015)) * get_cell_center_geom(xcell1, grad=20)
    real_x1 = real_x1[::-1]
    real_x2 = np.linspace(-0.015, 0.015, xcell2 + 2)[1:-1]
    real_x3 = 0.015 + ((0.125) - (0.015)) * get_cell_center_geom(xcell3, grad=20)
    real_x = np.concatenate((real_x1, real_x2, real_x3), axis=0)
    real_y1 = 0 + (0.050 - 0) * get_cell_center_geom(ycell1, grad=2)
    real_y2 = 0.050 + (0.400 - 0.050) * get_cell_center_geom(ycell2, grad=5)
    real_y = np.concatenate((real_y1, real_y2), axis=0)
    
    D = 4.57 * 1e-3
    real_x = real_x / D
    real_y = real_y / D
    
    x_slice, y_slice = np.meshgrid(real_y, real_x)
    
    if x.shape != x_slice.shape:
        x_slice = np.linspace(real_y.min(), real_y.max(), x.shape[1])
        y_slice = np.linspace(real_x.min(), real_x.max(), x.shape[0])
        x_slice, y_slice = np.meshgrid(x_slice, y_slice)
    
    vmin, vmax = 300, 1600
    fig, axes = plt.subplots(1, 2, figsize=(6, 4.5))
    titles = ['Initial Frame', 'Generated Frame']
    frames = [x, y]
    
    for i, ax in enumerate(axes):
        im = ax.pcolormesh(
            y_slice, x_slice, frames[i],
            cmap='jet', shading='gouraud', vmin=vmin, vmax=vmax
        )
        ax.set_ylim(0, 50)
        ax.set_xlim(-5, 5)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xticks([-2, 0, 2])
        ax.set_title(titles[i])
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig