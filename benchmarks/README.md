# Section 3: 

## 3.1 One-dimensional Stefan Problem

```bash
   mpirun -np 60 ../MeltPoolDG/build_release/meltpooldg fig1_stefans_problem_1000el.json | tee fig1_stefans_problem_1000el.output
   mpirun -np 60 ../MeltPoolDG/build_release/meltpooldg fig1_stefans_problem_2000el.json | tee fig1_stefans_problem_2000el.output
   mpirun -np 60 ../MeltPoolDG/build_release/meltpooldg fig1_stefans_problem_4000el.json | tee fig1_stefans_problem_4000el.output
```

## 3.2 Film Boiling

## 3.2.1 2D Simulation

```bash
   mpirun -np 60 ../MeltPoolDG/build_release/meltpooldg fig3-6_film_boiling_2d.json | tee fig3-6_film_boiling_2d.output
   bash fig3-6_film_boiling_2d_postprocess.sh
```

## 3.2.2 3D Simulation

```bash
   mpirun -np 240 ../MeltPoolDG/build_release/meltpooldg fig7+8_film_boiling_3d.json | tee fig7+8_film_boiling_3d.output
```

# Section 5: Numerical study of melt--vapor interaction in PBF-LB/M during stationary laser illumination of a bare metal plate

## 5.1 1D simulation

```bash
   mpirun -np 60 ../MeltPoolDG/build_release/meltpooldg fig13_laser_melting_1d.json | tee fig13_laser_melting_1d.output
   bash fig13_laser_melting_1d_postprocess.sh
```

## 5.2 2D simulation

```bash
   mpirun -np 120 ../MeltPoolDG/build_release/meltpooldg fig14+16_laser_melting_2d_V2.json | tee fig14+16_laser_melting_2d_V2.output
   bash fig14+16_laser_melting_2d_V2_postprocess.sh
```

```bash
   mpirun -np 60 ../MeltPoolDG/build_release/meltpooldg fig15+16_laser_melting_2d_V1.json | tee fig15+16_laser_melting_2d_V1.output
   bash fig15+16_laser_melting_2d_V1_postprocess.sh
```

## 5.3 3D simulation

```bash
   mpirun -np 240 ../MeltPoolDG/build_release/meltpooldg fig17_laser_melting_3d_V1.json | tee fig17_laser_melting_3d_V1.output
```

