#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: magdalena
"""

import pyvista
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

import argparse
pyvista.set_plot_theme('document')

plt.rcParams["figure.figsize"] = (11.69,8.27)
plt.rcParams["figure.titlesize"] = 'small'

def hide_x_ticks(ax):
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

def find_filenames(path_to_dir, suffix=".pvd", prefix=""):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix) and filename.startswith(prefix)]

def process_pvd( n, folder, pvdfile, enable_plot = True, solidus_temp = (1933+2200)*0.5, vertical_axis = 1):
    reader = pyvista.get_reader(os.path.join(folder,pvdfile))

    # loop over time steps:
    time_range = list(range(0,len(reader.time_values),n))
    
    # make sure last time step is considered
    if len(reader.time_values)-1 not in time_range:
        time_range.append(len(reader.time_values)-1)
    
    time = []
    mp_width = []
    mp_depth = []
    keyhole_depth = []

    for i, t in enumerate(time_range): 
        reader.set_active_time_point(t)
        mesh = reader.read()[0]
        
        # evaluate melt pool width
        ls_contour = mesh.contour([0.5], mesh.get_array("heaviside"))
        temperature = ls_contour.get_array("temperature")
        

        # create temporary stack
        temp = np.column_stack((ls_contour.points, temperature))

        # sort values by x-coordinate
        temp = temp[temp[:, 0].argsort()]

        # finding out the row numbers 
        rows = []
        for idx, T in enumerate(temp[:,-1]):
            if T > solidus_temp:
                rows.append(idx)
        
        if len(rows) > 0:
            time.append(reader.time_values[t]*1e3)
            
            mp_xmin = np.min(temp[rows,0]) * 1e6
            mp_xmax = np.max(temp[rows,0])* 1e6
            mp_width.append(np.abs(mp_xmax-mp_xmin))

            solid_contour = mesh.contour([solidus_temp], mesh.get_array("temperature"))
            mp_ymin = np.min(solid_contour.points[:,vertical_axis])* 1e6 
            mp_depth.append(np.abs(mp_ymin))

            keyhole_ymin = np.min(temp[rows,vertical_axis])
            keyhole_depth.append(np.abs(keyhole_ymin)* 1e6)
        

    print("keyhole morphology")
    print(f"    -- depth {keyhole_depth[-1]}")
    print("melt pool morphology")
    print(f"    -- width {mp_width[-1]}")
    print(f"    -- depth {mp_depth[-1]}")
    np.savetxt(os.path.join(folder,"melt_pool_morphology.csv"),
               np.column_stack((time,mp_width,mp_depth,keyhole_depth)),
               header="time mp_width mp_depth keyhole_depth")
    
#    ###########################################################################
#    # save figure
#    ###########################################################################
    
    if (enable_plot):
        fig, axs = plt.subplots(2,1)

        ax = axs[0]
        ax.invert_yaxis()
        ax.plot(time, mp_depth, color="red", label="melt pool depth")
        ax.plot(time, keyhole_depth, color="red", ls="dashed", label="keyhole depth")
        
        ax = axs[1]
        ax.plot(time, mp_width, color="blue", label="melt pool width")

        for ax in axs:
            ax.grid()
            ax.set_xlim([0,time[-1]])
            ax.legend()
            ax.set_xlabel("time (ms)")
            ax.set_ylabel(r"dimension ($\mu$m)")

        fig.savefig(os.path.join(folder,"melt_pool_morphology.png"), dpi=1200)
        print("file written: {:}".format(os.path.join(folder, "melt_pool_morphology.png")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export the level set contour to a file. Execute with pvpython!')
    parser.add_argument('--folder',             type=str,
                        help='define the folder, where the existing pvd-file is located')
    parser.add_argument('--pvdfile',        type=str, required=False,
                        help='define the name of the processed pvd-file, e.g. solution.pvd')
    parser.add_argument('-n', type=int, help='Write only every n.', default=10,
                        required=False)
    parser.add_argument('-y', action='store_true', help='Set this action to automatically overwrite files.',
                        required=False)
    args = parser.parse_args()

    folder = args.folder
    if not folder:
        folder = "."
    folder = os.path.join(os.getcwd(), folder)
    pvdfile = args.pvdfile

    # find pvd file if none is given
    if not pvdfile:
        pvdfile = find_filenames(folder)
        assert len(pvdfile) == 1
        pvdfile = pvdfile[0]

    # read vertical axis from file
    json_file = find_filenames(folder, '.json')
    assert len(json_file) == 1
    json_file = os.path.join(folder,json_file[0])
    with open(json_file, 'r') as f:
        data = json.load(f)
    v_axis = int(data["base"]["dimension"])-1

    # create temp folder
    print(70*"-")
    print(f" Vertical axis determined to {v_axis} from {json_file}")
    print(
        " Start processing pvd-file: {:}".format(os.path.abspath(os.path.join(folder, pvdfile))))
    process_pvd( args.n, folder, pvdfile, vertical_axis = v_axis)

    print(" The end")
    print(70*"-")
