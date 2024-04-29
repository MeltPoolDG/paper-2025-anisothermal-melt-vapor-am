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

def find_filenames(path_to_dir, suffix=".pvd", prefix=""):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix) and filename.startswith(prefix)]

def process_pvd( n, folder, pvdfile, enable_plot = True, vertical_axis = 1):
    reader = pyvista.get_reader(os.path.join(folder,pvdfile))

    # loop over time steps:
    time_range = list(range(0,len(reader.time_values),n))
    
    # make sure last time step is considered
    if len(reader.time_values)-1 not in time_range:
        time_range.append(len(reader.time_values)-1)
    
    time = []
    solid = []
    liquid = []
    gas = []

    for i, t in enumerate(time_range): 
        reader.set_active_time_point(t)
        mesh = reader.read()[0]

        # estimate volume from bounding box
        bounds = mesh.bounds
        volume = 1
        for j in range(0,3):
            x_min = bounds[2*j]
            x_max = bounds[2*j+1]
            length = x_max - x_min
            if (length < 1e-14):
                break
            else:
                volume *= length
        
        if (i==0):
            print(f"Volume: {volume}")

        # evaluate solid phase fraction
        solid.append(mesh.integrate_data()["solid"]/volume)
        liquid.append(mesh.integrate_data()["liquid"]/volume)
        gas.append(1-solid[-1]-liquid[-1])
            
        time.append(reader.time_values[t]*1e3)
    
    np.savetxt(os.path.join(folder,"phase_fractions.csv"),
               np.column_stack((time,solid,liquid,gas)),
               header="time solid liquid gas")
    
#    ###########################################################################
#    # save figure
#    ###########################################################################
    
    if (enable_plot):
        fig, axs = plt.subplots(1,1)

        ax = axs
        ax.plot(time, solid,  label="solid")
        ax.plot(time, liquid, label="liquid")
        ax.plot(time, gas,    label="gas")
        
        ax.grid()
        ax.set_xlim([0,time[-1]])
        ax.legend()
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(r"phase fraction (-)")

        fig.savefig(os.path.join(folder,"phase_fractions.png"), dpi=1200)
        print("file written: {:}".format(os.path.join(folder, "phase_fractions.png")))

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
