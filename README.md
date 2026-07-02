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


