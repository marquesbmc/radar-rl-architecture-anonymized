# RADAR: Attentional Neural Architecture with Semantic Channel Fusion

This repository contains the source code, trained neural models, training notebooks, simulation experiments, and figures used in the accompanying manuscript:

**RADAR: Attentional Architecture with Semantic-Derived Channel Fusion for Multi-Agent Environments**

> This work is currently under double-anonymized peer review. Author names,
> affiliations, and citation details are withheld from this repository until
> the review process is complete. A permanent, citable version of this
> repository (with full author and citation information) will be linked from
> the published article upon acceptance.

---

## About the RADAR Architecture

The **RADAR** (Reinforced Attention for Dynamic Agent Relations) architecture was developed to address perceptual and strategic challenges in partially observable multi-agent environments. It integrates two modules into a traditional CNN+MLP structure:

- **Cross-Channel Fusion Network (C2FN)**: performs semantic fusion of RGB channels, capturing tactical compositions and overlap information.
- **Spatial Attention via L2 and Mean Pooling (SALM)**: applies hybrid spatial attention based on local statistics (mean pooling and L2 norm), enabling contextual adaptation of perceptual focus.

The architecture is evaluated in a simulated predator-prey environment with multichannel perception, obstacles, and directional movement.

---

## Repository Structure

```
├── model/       # Trained models (.h5) for predator and prey agents
├── notebooks/   # Self-contained notebooks for training and running simulations
├── sim/         # Simulation output directories for different configurations
├── support/     # Auxiliary scripts and statistics
├── train/       # Training output logs and performance analyses
```

Repeated Double DQN results are consistently organized under
`model/Statistical Validation of Double DQN and RADAR Double/`, `train/Statistical Validation of Double DQN and RADAR Double/`, and
`sim/Statistical Validation of Double DQN and RADAR Double/`. Each set contains ten runs for the CNN-MLP and
RADAR architectures.

> Note: the `notebooks/` files are self-contained -- the environment,
> agent, and neural-network classes are defined directly in the notebook
> cells, so no separate simulator package is required to reproduce training
> or simulation. 

---

## Getting Started

1. Download or clone this repository (see the journal submission system /
   the published article for the permanent repository link).

2. Run the main notebooks:

- `notebooks/model_training.ipynb`: model training
- `notebooks/model_simulator.ipynb`: simulations with saved models

3. Analyze results:

```bash
python support/sim_statistics.py
python support/sim_visualization.py
python support/repeated_runs_statistics.py
```

These scripts process data from `sim_summary.csv` and generate performance metric charts.
The `repeated_runs_statistics.py` script reads `Mean Loss` and `Mean Reward`
from the final 50 episodes of each training log and calculates the mean and
sample standard deviation across the ten runs. The summary is saved to
`support/double_repeated_runs_summary.csv`.

---

## Results

Comparisons between the **RADAR** architecture and the classic **CNN+MLP** architecture were conducted using the following algorithms:

- DQN
- Double DQN
- Dueling DQN

### Evaluated metrics include:

- Training stability (*Loss*, *Reward*)
- Evasion/Capture effectiveness
- Collaborative effectiveness
- Spatial coverage (exploration)

Data and results are available in the `sim/` and `support/` directories.

## Environment Naming Convention

The three environments discussed in the manuscript correspond to the following:

| Environment (as named in the paper)   | Configuration code              |
|----------------------------------------|----------------------------------|
| Standard Environment                    | `sz10_s10_py10_pd5_o0.1`         |
| Dense and Constrained Environment       | `sz10_s10_py15_pd10_o0.3`        |
| Extensive and Sparse Environment        | `sz20_s10_py15_pd10_o0.05`       |

---



## Citation

A full citation will be provided here once the manuscript has completed peer
review and been assigned final publication details.

---

## License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

---

## Contact

Author contact information is withheld during double-anonymized review and
will be added here upon publication.
