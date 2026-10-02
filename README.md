<p align="center">
  <img src="docs/assets/PolyProton-logo.png" alt="PolyProton logo" width="720">
</p>

# PolyProton
This repository contains an automated workflow for constructing, equilibrating, and analyzing
sulfonated polyimide (SPI) electrolyte systems for proton conductivity studies. The workflow
is designed around Amber molecular dynamics simulations and post processing of dry, hydrated
and hydronium containing conductivity systems. The workflow is currently tailored to sulfonated
polyimide, but the code structure can be adapted to related polymer electrolyte systems by 
changing the monomer SMILES definitions, residue templates, input files, and analysis settings.

## Workflow overview
The complete workflow has two main codes:
- "PolyProton_simulation.py"
- "PolyProton_analysis.py"

The simulation workflow is organized into four stages:

1. Parameterization / structure generation
	- Builds the SPI monomer from backbone and sulfonated side-chain SMILES
	- Generates conformers using RDKit
	- Optimizes and ranks conformers based on energy using MACE (mace-off)
	- Selects either the lowest or a higher energy conformer
	- Calculates RESP charges of selected conformer using ORCA (wb97x-d4/def2-svp)
	- Builds polymer chains with fixed or mixed chain lengths
	- Creates Amber residue templates and force field files

2. Dry polymer equilibration
	- Packs the polymer chains into a periodic dry bulk cell using Packmol
	- Create Amber topology and coordinate files using LeAP
	- Runs either a 6-step or 12-step dry equilibration protocal
	- Produces the dry equilibrated structure and trajectory

3. Hydrated polymer equilibration
	- Adds water molecules at selected hydration levels (lambda values)
	- Runs hydration equilibration protocal
	- Produces hydrated structures and trajectories for each lambda value

4. Conductivity simulation
	- Creates hydronium-water systems for selected hydration levels
	- Runs conductivity production simulation (100 ns)
	- Produces trajectories

## Requirements
The simulation workflow requires the following
1. Programs:
- Amber / AmberTools >= 24
- ORCA >= 5.0.3
- Multiwfn >= 3.8
- OpenBabel >= 3.2.1
- Packmol >= 20.15.1
2. Python packages (python >= 3.8)
- ase
- rdkit
- MACE (`mace-torch`)
- numpy
- pandas
- scipy
- biopython

The analysis workflow requires the following
1. Python packages (python >= 3.8)
- numpy
- pandas
- scipy
- matplotlib
- ovito

## Main simulation settings
Most user controlled simulation settings are defined near the top of "PolyProton_simulation.py". 
The simulation workflow is intended to be run from the project root directory.
Simulation files are written under the `simulation/` folder, while analysis results are written
separately under the `analysis/` folder.
For v1.0.0, start each new calculation in a clean working directory containing the workflow scripts and the complete `input_files/` folder.
Use working and scratch paths without spaces. Automatic restart/resume support is not provided yet.

Example:
```
# Polymer / monomer settings
polymer = 'a1'

dianhydride_smiles = 'C1=CC2=C3C(=CC=C4C3=C1C(=O)OC4=O)C(=O)OC2=O'
sidechain_smiles = "OCCCS(O)(=O)=O"
benzene_smiles = "C1=CC=CC=C1"

# Conformer settings
conf_num = 50
conf_selection = "LE"	 
conf_far_fraction = 0.5
conf_prune_rms_thresh = 0.02	
temperature = 300

# Polymer system settings
chain_length = 10
num_chains = 15
mix_chains = False
mix_seed = 42
mix_chain_fraction = 0.20
pack_z_padding = 15.0
aligned = True

# Hydration / conductivity
lam_list = [2, 4, 6, 8, 10, 12]
cond_lam_list = [12]

# Equilibration protocol
dry_eq_prot = "6-step" 

# Workflow stages
run_param = True
run_dry = True
run_hyd = True
run_cond = True

# Computational resources
use_gpu = True
use_nproc = 24
use_mace_device = "cuda"

# External software / scratch paths
orca_dir = "/opt/orca"
scratch_dir = None
```

Important options:
| Setting | Meaning |
|---|---|
| `polymer` | Polymer label, such as `a1`, `a6`, or `a8`. This label is used in generated file names. |
| `dianhydride_smiles` | SMILES string of the dianhydride-backbone unit. |
| `sidechain_smiles` | SMILES string of the sulfonated side chain. |
| `benzene_smiles` | SMILES string of the benzene-backbone unit. |
| `conf_num` | Target number of conformers to generate. The actual number may be lower after RDKit pruning. |
| `conf_selection` | Selects the conformer used for parameterization. `"LE"` selects the lowest-energy conformer. `"HE"` randomly selects a conformer from the highest-energy fraction of the remaining conformers, as defined by `conf_far_fraction`. |
| `temperature` | Temperature used when calculating the reported Boltzmann populations of the optimized conformers. It does not change the `"LE"`/`"HE"` conformer-selection rule. |
| `conf_far_fraction` | Fraction of the non-lowest-energy conformers forming the high-energy selection pool when `conf_selection = "HE"`. For example, `0.5` uses the highest-energy 50% of the remaining conformers. |
| `conf_prune_rms_thresh` | RDKit conformer pruning threshold. Smaller values retain more similar conformers; `-1.0` disables pruning. |
| `chain_length` | Number of repeat units in the base polymer chain. This value is also used in generated file names. |
| `num_chains` | Number of chains packed into the bulk simulation cell. This value is also used in generated file names. |
| `mix_chains` | If `True`, generates a distribution of chain lengths around chain_length. If `False`, all chains use exactly `chain_length`. This value also affects generated file names. |
| `mix_seed` | Random seed used to generate mixed chain lengths when `mix_chains = True`. |
| `mix_chain_fraction` | Maximum fractional deviation from chain_length used for mixed-chain systems. For example, 0.20 gives chain lengths within approximately ±20% of the base chain length. |
| `pack_z_padding` | Extra z-direction padding, in Angstrom, used during initial Packmol dry-bulk construction. Larger values give more space along the chain/alignment direction during packing. |
| `aligned` | If `True`, uses aligned/compact initial packing. If `False`, uses less-aligned initial orientations. |
| `lam_list` | Hydration levels (λ) prepared sequentially during the hydration workflow. Here, λ represents the number of water molecules per sulfonic acid group. |
| `cond_lam_list` | Hydration levels selected for conductivity simulations. These must be included in `lam_list`. |
| `dry_eq_prot` | Selects the 6-step or 12-step dry equilibration protocol. |
| `run_param` | Enables or disables monomer parameterization and polymer-chain generation. |
| `run_dry` | Enables or disables dry polymer packing and dry equilibration. |
| `run_hyd` | Enables or disables hydrated polymer equilibration. |
| `run_cond`  | Enables or disables hydronium-containing conductivity simulations. |
| `use_gpu` | Controls Amber MD execution mode. If `True`, supported MD steps use `pmemd.cuda`. If `False`, all Amber MD steps use `pmemd.MPI`. When running under SLURM, `GRES` overrides this value automatically. |
| `use_nproc` | Number of CPU processes used for MACE-OFF conformer geometry optimization `device = cpu`, parallel ORCA single-point calculations and Amber `pmemd.MPI` simulations. When running under SLURM, `SLURM_NTASKS` overrides this value automatically. |
| `use_mace_device` | Device used for MACE-OFF conformer geometry optimization. Use `"cpu"` for CPU execution or `"cuda"` for GPU execution. |
| `orca_dir` | Directory containing the ORCA executables, for example `"/opt/orca"`. The workflow expects both `orca` and `orca_2mkl` to be available in this directory. |
| `scratch_dir` | Base directory used for temporary Amber calculation files. Set this to a writable local or scratch filesystem appropriate for the computing environment. |

GPU/CPU execution
```
use_gpu = True
```
When enabled, PolyProton uses a hybrid CPU/GPU execution scheme.
Dry-stage minimization and NPT steps are run with `pmemd.MPI`, while dry NVT steps are run with `pmemd.cuda`.
For hydrated and conductivity workflows, minimization is run with `pmemd.MPI`, while subsequent MD steps are run with `pmemd.cuda`.
```
use_gpu = False
```
When disabled, all Amber minimization and MD steps are run with `pmemd.MPI`.
The number of MPI processes is determined by `SLURM_NTASKS`, when available, or otherwise by `use_nproc`.

## Main analysis settings
The analysis workflow is controlled near the top of "PolyProton_analysis.py".
The following settings must match the simulation settings because they are used to reconstruct 
the expected system names and locate the correct simulation files:

```
polymer = "a1"

chain_length = 15
num_chains = 30
mix_chains = True

hyd_analysis_lam_list = [12]
cond_analysis_lam_list = [12]

run_dry_analysis = True
run_hyd_analysis = True
run_cond_analysis = True

dry_eq_prot = "6-step"
```

Therefore, when analyzing a simulation, make sure that `polymer`, `chain_length`, `num_chains`, and `mix_chains` in `PolyProton_analysis.py`
are the same as those used in `PolyProton_simulation.py`.

Important options:
| Setting | Meaning |
|---|---|
| `polymer` | Polymer label used to identify the system. Must match the simulation workflow. |
| `chain_length` | Base polymer chain length used in the simulation. Must match the simulation workflow. |
| `num_chains` | Number of polymer chains used in the simulation. Must match the simulation workflow. |
| `mix_chains` | Whether mixed chain lengths were used. Must match the simulation workflow because it changes the system name. |
| `hyd_analysis_lam_list` | Hydration levels selected for hydrated-system analysis. |
| `cond_analysis_lam_list` | Hydration levels selected for conductivity analysis. |
| `run_dry_analysis` | Enables or disables analysis of the dry polymer system. |
| `run_hyd_analysis` | Enables or disables analysis of hydrated systems. |
| `run_cond_analysis` | Enables or disables analysis of hydronium-containing conductivity systems. |
| `dry_eq_prot` | Must match the dry equilibration protocol used in the simulation. It determines which dry production trajectory is analyzed. |


## Workflow stage dependencies

The workflow stages are sequential. Hydrated equilibration requires the dry equilibration stage to be run in the same workflow execution:

`parameterization → dry equilibration → hydrated equilibration → conductivity`

When `run_hyd = True`, `run_dry` must also be enabled so that the dry equilibrated structure is passed to the hydration workflow.

Conductivity calculations require the corresponding hydrated system and parameterization files to already be available.

## Notes and limitations

Workflow limitations
- The workflow assumes that required input templates, water/hydronium PDB files, Amber input files, and force-field files are present in `input_files/`.
- The conductivity simulation is set to 100 ns, which cannot be changed right now. 
- Fitting window for MSD is set to 10-100 ns by default. This should be checked for each system to ensure that the selected time range is approximately linear.

Methodology limitations
- The current conductivity calculation is based on hydronium vehicle diffusion and Nernst-Einstein conductivity.

Notes
- The water-cluster analysis uses an O-O cutoff of 3.5 Angstrom. This cutoff should be kept consistent when comparing systems.
- The hydronium-sulfonate residence analysis uses a 4.0 Angstrom cutoff. This cutoff is a structural definition of bound/contacting hydronium and may need sensitivity testing.
- If `mix_chains = True`, the reported system name keeps the base chain length but the actual chain-length distribution is written to the workflow log.
- The actual number of generated conformers may be smaller than `conf_num` because of conformer embedding or pruning.

## License

PolyProton is distributed under the BSD 3-Clause License.
This license applies only to the PolyProton source code and documentation in this repository.
External software packages used by the workflow, including Amber/AmberTools, ORCA, Multiwfn, OpenBabel, Packmol, OVITO, MACE, RDKit, ASE, and other dependencies,
are distributed under their own licenses. Users are responsible for obtaining and using these third-party programs according to their respective license terms.

## Citation

If you use PolyProton in academic work, please cite the associated publication.
Citation information will be added after publication.
