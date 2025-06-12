# A consistent diffuse-interface finite element approach to rapid melt--vapor dynamics with application to metal additive manufacturing

[![LGPLv3 License](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://opensource.org/license/lgpl-3-0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15061694.svg)](https://doi.org/10.5281/zenodo.15061694)

This repository contains information and code to reproduce the results presented in the article:
```bibtex
@article{schreterfleischhacker2025meltvapordynamics,
title = {A consistent diffuse-interface finite element approach to rapid melt–vapor dynamics with application to metal additive manufacturing},
journal = {Computer Methods in Applied Mechanics and Engineering},
volume = {442},
pages = {117985},
year = {2025},
doi = {https://doi.org/10.1016/j.cma.2025.117985},
author = {Magdalena Schreter-Fleischhacker and Nils Much and Peter Munch and Martin Kronbichler and Wolfgang A. Wall and Christoph Meier}
}
```

If you find these results useful, please cite the article mentioned above. If you use the implementations provided here, please also cite this repository as:

```bibtex
@misc{schreterfleischhacker2025meltvapordynamicsrepo,
  title={Reproducibility repository for "{A} consistent diffuse-interface finite element approach to rapid melt--vapor dynamics with application to metal additive manufacturing"},
  author={Schreter-Fleischhacker, Magdalena and Much, Nils and Munch, Peter and Kronbichler, Martin},
  year={2025},
  howpublished={\url{https://github.com/MeltPoolDG/paper-2025-anisothermal-melt-vapor-am}},
  doi={https://doi.org/10.5281/zenodo.15061694}
}
```

## Disclaimer
Everything is provided as is and without warranty. Use at your own risk! 

## Abstract
Metal additive manufacturing via laser-based powder bed fusion (PBF-LB/M) faces performance-critical challenges due to complex melt pool and vapor dynamics, often oversimplified by computational models that neglect crucial aspects, such as vapor jet formation. To address this limitation, we propose a consistent computational multi-physics mesoscale model to study melt pool dynamics, laser-induced evaporation, and vapor flow. In addition to the evaporation-induced pressure jump, we also resolve the evaporation-induced volume expansion and the resulting velocity jump at the liquid--vapor interface. We use an anisothermal incompressible Navier--Stokes solver extended by a conservative diffuse level-set framework and integrate it into a matrix-free adaptive finite element framework. To ensure accurate physical solutions despite extreme density, pressure and velocity gradients across the diffuse liquid--vapor interface, we employ consistent interface source term formulations developed in our previous work. These formulations consider projection operations to extend solution variables from the sharp liquid--vapor interface into the computational domain. Benchmark examples, including film boiling, confirm the accuracy and versatility of the model. As a key result, we demonstrate the model's ability to capture the strong coupling between melt and vapor flow dynamics in PBF-LB/M based on simulations of stationary laser illumination on a metal plate. Additionally, we show the derivation of the well-known Anisimov model and extend it to a new hybrid model. This hybrid model, together with consistent interface source term formulations, especially for the level-set transport velocity, enables PBF-LB/M simulations that combine accurate physical results with the robustness of an incompressible, diffuse-interface computational modeling framework.

## Reproducing the results

### Requirements
- linux-based system
- `boost` installed
- `mpi` installed
- `openblas` or `blas` installed
- `python3` installed

### Installation
To download the code using Git, use:
```bash
git clone git@github.com:MeltPoolDG/paper-2025-anisothermal-melt-vapor-am.git
```

If you do not have Git installed, you can obtain a `.zip` file and unpack it:
```bash
wget https://github.com/MeltPoolDG/paper-2025-anisothermal-melt-vapor-am/archive/main.zip
unzip paper-2025-anisothermal-melt-vapor-am.zip
```

To install the code execute:
```bash
cd paper-2025-anisothermal-melt-vapor-am/MeltPoolDG
bash scripts/config/install.sh
```
Then, follow the installation instructions. You can accept the default settings by pressing `Enter`.

### Running the code
The input files and postprocessing scripts for the numerical studies, presented in the article, are located in the `benchmarks` directory. Please follow the instructions provided in the [`README`](https://github.com/MeltPoolDG/paper-2025-anisothermal-melt-vapor-am/blob/main/benchmarks/README.md) file within the benchmarks folder for running the simulations.

## Authors of this repository
- Magdalena Schreter-Fleischhacker (Corresponding Author), [@mschreter](https://github.com/mschreter), Technical University of Munich
- Nils Much, [@nmuch](https://github.com/nmuch), Technical University of Munich
- Peter Munch, [@peterrum](https://github.com/peterrum), Technical University of Berlin
- Martin Kronbichler, [@kronbichler](https://github.com/kronbichler), Ruhr University Bochum

