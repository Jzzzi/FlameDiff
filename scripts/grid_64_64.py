import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", 
                "#bcbd22", "#17becf"]

###非等距采样, linspace为等差数列，geomspace为等比数列，均包含首尾.logspace为对数等差数列
def get_cell_center_geom(pointnum=64,grad=10):
    cell_center_geom = np.geomspace(1,grad,pointnum)/2
    cell_center_geom[1:] += np.geomspace(1,grad,pointnum).cumsum()[:-1]
    total_length = np.geomspace(1,grad,pointnum).sum()
    cell_center_geom /= total_length
    return cell_center_geom
xcell1 = 50; xcell2=92; xcell3=50 #192
#ycell1 = 120; ycell2 = 136 #256
ycell1 = 256
real_x1 = -0.015 + ((-0.125)-(-0.015))*get_cell_center_geom(xcell1,grad=20)
real_x1 = real_x1[::-1] #reserve x1 逆序
real_x2 = np.linspace(-0.015,0.015,xcell2+2)[1:-1] #要求不含首尾
real_x3 = 0.015 + ((0.125)-(0.015))*get_cell_center_geom(xcell3,grad=20)
real_x = np.concatenate((real_x1,real_x2,real_x3),axis=0)
#real_y1 = 0+(0.050-0)*get_cell_center_geom(ycell1,grad=2)
#real_y2 = 0.050+(0.400-0.050)*get_cell_center_geom(ycell2,grad=5)
#real_y = np.concatenate((real_y1,real_y2),axis=0)
real_y = 0+(0.400-0)*get_cell_center_geom(ycell1,grad=10)
#real_x: 192: R方向
#real_y: 256: H方向
##################################
#resolution = 64
real_x = real_x[::3]; real_y = real_y[::4]; 
D = 4.57*1e-3
real_x = real_x/D; real_y = real_y/D
x_slice, y_slice = np.meshgrid(real_y, real_x)
##################################

T = np.load('T.npy')
print(T.shape) #(300, 64, 64)
print(T[299])
print(x_slice.shape, y_slice.shape, T[299].shape)

vmin=300; vmax=1600
fig, axes = plt.subplots(1, 1, figsize=(3, 4.5))
#auto; gouraud

im = axes.pcolormesh(np.rot90(y_slice), np.rot90(x_slice), np.rot90(T[299]), cmap='jet', shading='gouraud', vmin=vmin, vmax=vmax) 
axes.set_ylim(0, 50)
axes.set_xlim(-5, 5)
axes.tick_params(axis='x', labelsize=10) 
axes.tick_params(axis='y', labelsize=10) 

axes.set_xticks([-2,0,2])
cbar = plt.colorbar(im)


plt.savefig('example_T.png')

