<p align="center">
  <img src="docs/assets/PolyProton-logo.png" alt="PolyProton logo" width="720">
</p>

# PolyProton
This repository contains as automated workflow for constructing, equlibrating, and analyzing
sulfonated polyimide (SPI) electrolyte systems for proton conductivity studies. The workflow
is designed around Amber molecular dynamics simulations and post processing of dry, hydrated
and hydronium containing conductivity systems. The workflows is currently tailored to sulfonated
polyimide, but the code structure can be adapted to related polymer electrolyte systems by 
chaning the monomer SMILES definitions, residue templates, input files, and analysis settings.

## Workflow overview
The complete workflow has two main codes:
"PolyProton_simulation.py"
"PolyProton_analysis.py"

The simulation workflow is organized into four stages:

1. Parameterization / structure generation
	- Builds the SPI monomer from backbone and sulfonated side-chain SMILES
	- Generates conformers using RDKit
	- Optimizes and ranks conformers based on energy using MACE (mace-off)
	- Selects either the lowest or a higher energy conformer
	- Calculate RESP charges of selected conformer using ORCA (wb97x-d4/def2-svp)
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
- OpenBabel = 3.0.0!
- Packmol >= 20.15.1
2. Python packages (python >= 3.8)
- ase
- rdkit
- mace 
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
Most user controlled simulation settings are defined near the top of "PolyProton_simulation.py" 

Example:
```
polymer = "a1"

conf_num = 50
conf_selection = "best"          
conf_prune_rms_thresh = 0.02      

chain_length = 15
num_chains = 30
mix_chains = True
mix_seed = 42
aligned = True

lam_list = [2, 4, 6, 8, 10, 12]
cond_lam_list = [12]

dry_eq_prot = "6-step"           

run_param = True
run_dry = True
run_hyd = True
run_cond = True
```

Important options:
| Setting | Meaning |
|---|---|
| `polymer` | Polymer label, such as `a1`, `a6`, or `a8`. Will create a folder name based on this and perform simulations in it. |
| `backbone_smiles` | SMILES string of the dianhydride/backbone unit. |
| `sidechain_smiles` | SMILES string of the sulfonated side chain. |
| `conf_num` | Target number of conformers to generate. The actual number may be lower after pruning. |
| `conf_prune_rms_thresh` | RDKit conformer pruning threshold. Smaller values retain more similar conformers; `-1.0` disables pruning. |
| `conf_selection` | Selects either the best-ranked conformer or a random conformer from the high-energy/diverse conformer pool. |
| `chain_length` | Number of repeat units in the base polymer chain. |
| `num_chains` | Number of chains packed into the bulk simulation cell. |
| `mix_chains` | If `True`, randomly generates a chain-length distribution around the base chain length. |
| `aligned` | If `True`, uses aligned/compact initial packing. If `False`, uses less-aligned initial orientations. |
| `lam_list` | Hydration levels to prepare and equilibrate. |
| `cond_lam_list` | Hydration levels selected for conductivity simulations. These must be included in `lam_list`. |
| `dry_eq_prot` | Selects the 6-step or 12-step dry equilibration protocol. |
| `run_param`, `run_dry`, `run_hyd`, `run_cond` | Enable or disable each workflow stage. |