mpirun -np 60 ../MeltPoolDG/build_release/meltpooldg fig1_stefans_problem_1000el.json | tee fig1_stefans_problem_1000el.output
mpirun -np 60 ../MeltPoolDG/build_release/meltpooldg fig1_stefans_problem_2000el.json | tee fig1_stefans_problem_2000el.output
mpirun -np 60 ../MeltPoolDG/build_release/meltpooldg fig1_stefans_problem_4000el.json | tee fig1_stefans_problem_4000el.output
mpirun -np 60 ../MeltPoolDG/build_release/meltpooldg fig3-6_film_boiling_2d.json | tee fig3-6_film_boiling_2d.output
mpirun -np 240 ../MeltPoolDG/build_release/meltpooldg fig7+8_film_boiling_3d.json | tee fig7+8_film_boiling_3d.output
mpirun -np 60 ../MeltPoolDG/build_release/meltpooldg fig13_laser_melting_1d.json | tee fig13_laser_melting_1d.output
mpirun -np 120 ../MeltPoolDG/build_release/meltpooldg fig14+16_laser_melting_2d_V2.json | tee fig14+16_laser_melting_2d_V2.output
mpirun -np 60 ../MeltPoolDG/build_release/meltpooldg fig15+16_laser_melting_2d_V1.json | tee fig15+16_laser_melting_2d_V1.output
mpirun -np 240 ../MeltPoolDG/build_release/meltpooldg fig17_laser_melting_3d_V1.json | tee fig17_laser_melting_3d_V1.output

