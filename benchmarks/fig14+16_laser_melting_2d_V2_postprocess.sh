output_dir=output_fig14+16_laser_melting_2d_V2
python3 scripts/pyvista_export_mass_balance.py --folder $output_dir -n 10 --phase_densities 4.087 4087 4087 &
python3 scripts/pyvista_export_phase_fractions.py --folder $output_dir -n 50 &
python3 scripts/pyvista_export_melt_pool_morphology.py --folder $output_dir -n 10 &
python3 scripts/paraview_export_data_along_path.py -s 0 -0.0003 0 -e 0 0.0003 0 -r temperature pressure heaviside velocity -n 50 --folder $output_dir --max_points 200 
