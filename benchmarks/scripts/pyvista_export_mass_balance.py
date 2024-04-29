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

def process_pvd( n, folder, pvdfile, phase_densities, enable_plot = True, vertical_axis = 1):

    rho_g = float(phase_densities[0])
    rho_l = float(phase_densities[1])
    rho_s = float(phase_densities[2])
    print(r"rho_g="+f"{rho_g}")
    print(r"rho_l="+f"{rho_l}")
    print(r"rho_s="+f"{rho_s}")

    reader = pyvista.get_reader(os.path.join(folder,pvdfile))

    # loop over time steps:
    time_range = list(range(0,len(reader.time_values),n))
    
    # make sure last time step is considered
    if len(reader.time_values)-1 not in time_range:
        time_range.append(len(reader.time_values)-1)
    
    time = []
    total = []
    estimated_mass = []

    # time_step
    dt = 0

    for i, t in enumerate(time_range): 
        reader.set_active_time_point(t)
        mesh = reader.read()[0]

        # evaluate total mass
        total.append(mesh.integrate_data()["density"])
        
        # evaluate time
        time.append(reader.time_values[t]*1e3)

        # compute estimated mass loss
        ls_contour = mesh.contour([0.5], mesh.get_array("heaviside"))
        if (v_axis == 0): # 1D
            evapor_mass_flux = ls_contour["evaporative_mass_flux"][0]
        else:
            evapor_mass_flux = ls_contour.integrate_data()["evaporative_mass_flux"][0]

        if (len(time)>1):
           dt = (time[-1] - time[-2])*1e-3
           estimated_mass.append(estimated_mass[-1] - evapor_mass_flux/rho_l * (rho_l - rho_g)*dt)
        else:
            estimated_mass.append(total[0])

        print(f"{time[-1]}, {total[-1]}")

    
    np.savetxt(os.path.join(folder,"mass.csv"),
               np.column_stack((time,total, estimated_mass)),
               header="time total_mass total_mass_estimated")
    
#    ###########################################################################
#    # save figure
#    ###########################################################################
    
    if (enable_plot):
        fig, axs = plt.subplots(1,1)

        ax = axs
        ax.plot(time, total,  label="mass")
        ax.plot(time, estimated_mass, label="estimated mass")
        
        ax.grid()
        ax.set_xlim([0,time[-1]])
        ax.legend()
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(r"mass (kg)")

        fig.savefig(os.path.join(folder,"mass.png"), dpi=1200)
        print("file written: {:}".format(os.path.join(folder, "mass.png")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export the level set contour to a file. Execute with pvpython!')
    parser.add_argument('--folder',             type=str, required=True,
                        help='define the folder, where the existing pvd-file is located')
    parser.add_argument('--pvdfile',        type=str, required=False,
                        help='define the name of the processed pvd-file, e.g. solution.pvd')
    parser.add_argument('-n', type=int, help='Write only every n.', default=10,
                        required=False)
    parser.add_argument('--phase_densities', help='Phase densities in the order of gas -- liquid -- solid.', nargs='+',
                        required=True)
    parser.add_argument('-y', action='store_true', help='Set this action to automatically overwrite files.',
                        required=False)
    args = parser.parse_args()

    assert len(args.phase_densities) == 3

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
    process_pvd( args.n, folder, pvdfile, args.phase_densities, vertical_axis = v_axis)

    print(" The end")
    print(70*"-")
