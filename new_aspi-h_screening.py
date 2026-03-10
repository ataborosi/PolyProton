import numpy as np
import pandas as pd
import math
import os
import shutil
import glob
import subprocess

import ase
from ase.io import read, write
from ase.optimize import LBFGS, FIRE
from ase.constraints import FixAtoms

from rdkit import Chem
from rdkit.Chem import AllChem, rdGeometry, Draw

from Bio.PDB import PDBParser, PDBIO, Atom

from scipy.spatial.transform import Rotation as R

import warnings
warnings.simplefilter("ignore")

from mace.calculators import mace_off
nnp_calc = mace_off(model="small", device='cpu')
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile
profile = OrcaProfile(command='/opt/orca/orca')

output_2 = 'process-temp.txt'

class MonomerBuilder:
	def __init__(self, polymer, backbone_smiles, sidechain_smiles, benzene_smiles, conf_num):
		self.polymer = polymer
		self.backbone_smiles = backbone_smiles
		self.sidechain_smiles = sidechain_smiles
		self.benzene_smiles = benzene_smiles
		self.final_conf = None
		self.cids = []
		self.conf_num = conf_num
	
	def create_backbone(self):
		backbone = Chem.MolFromSmiles(self.backbone_smiles)
		for atom in backbone.GetAtoms():
			if (atom.GetSymbol() == 'O'):
				carbon_neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors() if (neighbor.GetSymbol() == 'C')]
				if len(carbon_neighbors) == 2:
					if all(len([neighbor.GetIdx() for neighbor in backbone.GetAtomWithIdx(carbon).GetNeighbors() if neighbor.GetSymbol() == 'O']) == 2 for carbon in carbon_neighbors):
						atom.SetAtomicNum(7)
		self.backbone = backbone
	
	def attach_sidechain(self):
		benzene = Chem.MolFromSmiles(self.benzene_smiles)
		sidechain = Chem.MolFromSmiles(self.sidechain_smiles)
		benside = Chem.CombineMols(benzene, sidechain)
		ed_benside = Chem.EditableMol(benside)
		ed_benside.AddBond(3, 6, Chem.BondType.SINGLE)
		benside = ed_benside.GetMol()
		
		combo = Chem.CombineMols(benside, self.backbone)
		Nidx = [atom.GetIdx() for atom in combo.GetAtoms() if atom.GetSymbol() == "N"]
		ed_combo = Chem.EditableMol(combo)
		ed_combo.AddBond(4, Nidx[0], Chem.BondType.SINGLE)
		combo = ed_combo.GetMol()
		
		final = Chem.CombineMols(benside, combo)
		Nidx = [atom.GetIdx() for atom in final.GetAtoms() if atom.GetSymbol() == "N"]
		ed_final = Chem.EditableMol(final)
		ed_final.AddBond(4, Nidx[1], Chem.BondType.SINGLE)
		final = ed_final.GetMol()
		Chem.SanitizeMol(final)
		self.final_conf = Chem.AddHs(final)
	
	def create_conformations(self):
		params = Chem.rdDistGeom.srETKDGv3()
		params.pruneRmsThresh = 0.1
		params.clearConfs=True
		params.numThreads = 0
		
		self.cids = Chem.rdDistGeom.EmbedMultipleConfs(self.final_conf, numConfs=self.conf_num, params=params)
		
		atom_map = [1, 15]
		Chem.rdMolAlign.AlignMolConformers(self.final_conf, atomIds=atom_map, maxIters=100000)
		for i, cid in enumerate(self.cids):
			Chem.MolToXYZFile(self.final_conf, f'{self.polymer}_{i}.xyz', confId=cid)

class ConformationAnalysis:
	def __init__(self, polymer, conf_num, calculator):
		self.polymer = polymer
		self.conf_num = conf_num
		self.calculator = calculator
		
	def optimize_confomer(self):
		for conf_i in range(self.conf_num):
			mol = read(f"{self.polymer}_{conf_i}.xyz")
			fix = [1, 15]
			mol.set_constraint(FixAtoms(indices=list(set(fix))))
			mol.set_calculator(self.calculator)
			
			opt = LBFGS(mol, logfile=f"{self.polymer}_{conf_i}_opt.txt")
			opt.run(fmax=0.01)
			write(f"{self.polymer}_{conf_i}_opt.xyz", mol, format='xyz')

class ConformationAnalyzer:
	def __init__ (self, polymer, conf_num, output_file, temperature):
		self.polymer = polymer
		self.conf_num = conf_num
		self.output_file = output_file
		self.temperature = temperature
		
	def analyze_conformers(self):
		dfs = []
		for conf_i in range(self.conf_num):
			file_opt = f"{polymer}_{conf_i}_opt.txt"
			with open(file_opt) as f:
				last_line = f.readlines()[-1]
				columns = last_line.split()
				E = float(columns[3].rstrip('*'))
				E = round(E, 4)
			df_temp = pd.DataFrame({'conf_i': [str(conf_i)], 'E': E})
			dfs.append(df_temp)
		
		df = pd.concat(dfs, ignore_index = True)
		
		min_E = df['E'].min()
		df['ΔE'] = df['E'] - min_E
		q = sum([math.exp(-E / (0.00008617 * self.temperature)) for E in df['ΔE']])
		df['boltzmann'] = [(math.exp(-E / (0.00008617 * self.temperature)) / q) for E in df['ΔE']]
		df = df.sort_values(by='boltzmann', ascending=False)
		df.to_csv(self.output_file, sep=' ', index=False)

class GAFF2Param:
	def __init__(self, polymer, chain_length):
		self.polymer = polymer
		self.chain_length = chain_length
		
	def orca_calculation(self, mol):
		orca_calc = ORCA(
			profile=profile,
			orcasimpleinput='wb97x-d4 def2-svp def2/j rijcosx tightscf',
			charge=0,
			mult=1,
			orcablocks='%pal nprocs 6 end'
		)
		mol.set_calculator(orca_calc)
		mol.get_potential_energy()
		for f in glob.glob("orca.*"):
			new_name = f.replace("orca.", f"{self.polymer}.", 1)
			if not os.path.exists(new_name):
				shutil.move(f, new_name)	
		orca_command = f'/opt/orca/orca_2mkl {self.polymer} -molden'
		subprocess.run(orca_command, shell=True, check=True)

	def run_multiwfn(self):
		molden_file = f'{self.polymer}.molden.input'
		multiwfn_input = f'{molden_file}\n7\n18\n10\n2\n1\ny\n0\n0\nq\n'
		with open("multiwfn_input.txt", "w") as input_file:
			input_file.write(multiwfn_input)
		multiwfn_command = f'Multiwfn < multiwfn_input.txt -set /opt/multiwfn/settings.ini'
		subprocess.run(multiwfn_command, shell=True, check=True)

	def convert_xyz_to_mol2(self, mol):
		mol2_file = f'{self.polymer}.mol2'
		obabel_command = f'obabel -i xyz {mol} -O {mol2_file}'
		subprocess.run(obabel_command, shell=True, check=True)

	def modify_mol2_file(self):
		mol2_file = f'{self.polymer}.mol2'
		with open(mol2_file, 'r') as f:
			mol2_lines = f.readlines()
		atom_start = next(i for i, line in enumerate(mol2_lines) if line.startswith('@<TRIPOS>ATOM'))
		bond_start = next(i for i, line in enumerate(mol2_lines) if line.startswith('@<TRIPOS>BOND'))
		atom_lines = mol2_lines[atom_start + 1:bond_start]
		atom_data = [line.split() for line in atom_lines]
		atom_df = pd.DataFrame(atom_data)
#		molden_chg_file = f'{self.polymer}.chg'
		molden_chg_file = f'{self.polymer}.molden.chg'
		molden_df = pd.read_csv(molden_chg_file, delim_whitespace=True, header=None)
		atom_df[7] = self.polymer
		atom_df[8] = molden_df[4].map('{:.4f}'.format)
		atom_lines_modified = [' '.join(row) + '\n' for row in atom_df.values.astype(str)]
		modified_mol2_file = f'{self.polymer}_mod.mol2'
		with open(modified_mol2_file, 'w') as f:
			f.writelines(mol2_lines[:atom_start + 1] + atom_lines_modified + mol2_lines[bond_start:])

	def run_antechamber(self):
		antechamber_command_1 = f'/opt/amber/amber24/bin/wrapped_progs/antechamber -i {self.polymer}_mod.mol2 -fi mol2 -o {self.polymer}_gaff2.mol2 -fo mol2 -at gaff2'
		subprocess.run(antechamber_command_1, shell=True, check=True)
		antechamber_command_2 = f'/opt/amber/amber24/bin/wrapped_progs/antechamber -i {self.polymer}_gaff2.mol2 -fi mol2 -o {self.polymer}.ac -fo ac'
		subprocess.run(antechamber_command_2, shell=True, check=True)
		antechamber_command_3 = f'/opt/amber/amber24/bin/wrapped_progs/parmchk2 -i {self.polymer}_gaff2.mol2 -f mol2 -o {self.polymer}_gaff2.frcmod -s 2'
		subprocess.run(antechamber_command_3, shell=True, check=True)	
		prepgen_command_1 = f'/opt/amber/amber24/bin/wrapped_progs/prepgen -i {self.polymer}.ac -o {self.polymer}_m.prepi -f prepi -m main -rn {self.polymer}'
		subprocess.run(prepgen_command_1, shell=True, check=True)
		prepgen_command_2 = f'/opt/amber/amber24/bin/wrapped_progs/prepgen -i {self.polymer}.ac -o h.prepi -f prepi -m head -rn H'
		prepgen_command_3 = f'/opt/amber/amber24/bin/wrapped_progs/prepgen -i {self.polymer}.ac -o t.prepi -f prepi -m tail -rn T'
		subprocess.run(prepgen_command_2, shell=True, check=True)
		subprocess.run(prepgen_command_3, shell=True, check=True)

	def create_polymer_chain(self):
		sequence = ['H']
		chain = self.chain_length - 2
		sequence.extend([f'{self.polymer}'] * chain)
		original_sequence = sequence + ['T']
		sequence = original_sequence.copy()
		middle_sequence = sequence[1:-1]
		sequence = ['H'] + middle_sequence + ['T']
		formatted_sequence = '{' + ' '.join(sequence) + '}'
		leap_input_filename = f'leap_input.in'
		leap_output_filename = f'leap_input.out'
		with open(leap_input_filename, 'w') as f:
			f.write('source leaprc.gaff2\n')
			f.write(f'loadamberprep h.prepi\n')
			f.write(f'loadamberprep t.prepi\n')
			f.write(f'loadamberprep {self.polymer}_m.prepi\n')
			f.write(f'loadamberparams {self.polymer}_gaff2.frcmod\n')
			f.write(f'mol = sequence {formatted_sequence}\n')
			f.write(f'savepdb mol {self.polymer}_n-{self.chain_length}.pdb\n')
			f.write(f'saveamberparm mol {self.polymer}_n-15.prmtop {self.polymer}_n-{self.chain_length}.inpcrd\n')
			f.write('quit\n')
		subprocess.run(f'tleap -f {leap_input_filename} > {leap_output_filename}', shell=True, check=True)
	
	def parameterization(self):
		mol = read(f'{self.polymer}_0_opt.xyz')
		mol_name = f'{self.polymer}_0_opt.xyz'
		with open(output_2, 'a') as f:
			print(f"parameterization started", file=f)
		self.orca_calculation(mol)
		with open(output_2, 'a') as f:
			print(f"\torca_calculation done", file=f)
		self.run_multiwfn()
		with open(output_2, 'a') as f:
			print(f"\trun_multiwfn", file=f)
		self.convert_xyz_to_mol2(mol_name)
		with open(output_2, 'a') as f:
			print(f"\tconvert_xyz_to_mol2", file=f)
		self.modify_mol2_file()
		with open(output_2, 'a') as f:
			print(f"\tmodify_mol2_file", file=f)
		self.run_antechamber()
		with open(output_2, 'a') as f:
			print(f"\trun_antechamber", file=f)
		self.create_polymer_chain()
		with open(output_2, 'a') as f:
			print(f"\tcreate_polymer_chain", file=f)

class BulkCreator:
	def __init__(self, polymer, chain_length, num_chains):
		self.polymer = polymer
		self.chain_length = chain_length
		self.num_chains = num_chains
		self.box_size_x = 0
		self.box_size_y = 0
		self.box_size_z = 0
		
	def rotate_chain(self):
		single_chain_pdb = f'{self.polymer}_n-{self.chain_length}.pdb'
		parser = PDBParser(QUIET=True)
		pdb_structure = parser.get_structure('original', single_chain_pdb)
		single_chain = read(single_chain_pdb)
		positions = single_chain.get_positions()
		center_of_mass = np.mean(positions, axis=0)
		single_chain.translate(-center_of_mass)
		cov_matrix = np.cov(positions.T)
		eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
		longest_axis = eigenvectors[:, np.argmax(eigenvalues)]
		z_axis = np.array([0, 0, 1])
		rotation_axis = np.cross(longest_axis, z_axis)
		rotation_angle = np.arccos(np.dot(longest_axis, z_axis) / (np.linalg.norm(longest_axis) * np.linalg.norm(z_axis)))
		single_chain.rotate(v=rotation_axis, a=np.degrees(rotation_angle), center='COM')
		new_positions = single_chain.get_positions()
		for i, atom in enumerate(pdb_structure.get_atoms()):
			atom.set_coord(new_positions[i])
		self.rotated_single_chain = f'{self.polymer}_n-{self.chain_length}_rot.pdb'
		io = PDBIO()
		io.set_structure(pdb_structure)
		io.save(self.rotated_single_chain)

	def box_dimension(self):
		structure = read(self.rotated_single_chain)
		positions = structure.get_positions()
		min_coords = np.min(positions, axis=0)
		max_coords = np.max(positions, axis=0)
		dim_x, dim_y, dim_z = max_coords - min_coords
		polymer_volume = dim_x * dim_y * dim_z
		total_box_volume = polymer_volume * self.num_chains
		box_height = dim_z + 30
		box_side_area = total_box_volume / box_height
		
		scale_factor = 1.65 - 0.3 * np.log2(1 + (dim_x + dim_y) / 10)
		scale_factor = max(0.5, min(1.5, scale_factor))

		self.box_size_x = np.sqrt(box_side_area) * scale_factor
		self.box_size_y = np.sqrt(box_side_area) * scale_factor
		self.box_size_z = box_height

	def packmol_generate_box(self):
		with open("packmol_input.inp", "w") as f:
			f.write(f"""
			tolerance 2.0
			filetype pdb
			add_amber_ter
			output {self.polymer}_n-{self.chain_length}x{self.num_chains}.pdb
			structure {self.rotated_single_chain}
			number {self.num_chains}
			inside box 0. 0. 0. {self.box_size_x} {self.box_size_y} {self.box_size_z}
			constrain_rotation x 0. 0.
			constrain_rotation y 0. 0.
			constrain_rotation z 0. 0.
			end structure
			""")
	
		subprocess.run('/opt/packmol/packmol-20.15.1/packmol < packmol_input.inp', shell=True)
#		subprocess.run('packmol < packmol_input.inp', shell=True)
	
	def create_bulk_phase(self):
		self.rotate_chain()
		self.box_dimension()
		self.packmol_generate_box()

#class AmberParams:
		
# Create a working directory for polymer and copy polymer connectivity cards			
polymer = 'a1'
if not os.path.exists(polymer):
	os.makedirs(polymer)
polymer_cards = ['head', 'main', 'tail']
for polymer_cards_i in polymer_cards:
	shutil.copy(polymer_cards_i, f'{polymer}')
os.chdir(polymer)

# Details of polymer
backbone_smiles = 'C1=CC2=C3C(=CC=C4C3=C1C(=O)OC4=O)C(=O)OC2=O'
sidechain_smiles="OCCCS(O)(=O)=O"
benzene_smiles="C1=CC=CC=C1"
conf_num = 2
conf_output_file = f"{polymer}_conf.txt"
temperature = 300
chain_length = 15
num_chains = 20

output = 'test.txt'

with open(output, 'a') as f:
	print(f"Processing polymer: {polymer} with backbone: {backbone_smiles}", file=f)

# Monomer building for different backbone using RDKit and creating conformers
monomer_builder = MonomerBuilder(polymer, backbone_smiles, sidechain_smiles, benzene_smiles, conf_num)
monomer_builder.create_backbone()
monomer_builder.attach_sidechain()
monomer_builder.create_conformations()		
with open(output, 'a') as f:
	print(f"\tMonomer building and confomers creation finished", file=f)

# Performing conformational analysis on the created conformers
conformation_analysis = ConformationAnalysis(polymer, conf_num, nnp_calc)
conformation_analysis.optimize_confomer()
with open(output, 'a') as f:
	print(f"\tConformation analysis finished", file=f)

# Performing the evaluation of conformers based on their total energies and Boltzmann distribution
analyzer = ConformationAnalyzer(polymer, conf_num, conf_output_file, temperature)
analyzer.analyze_conformers()	
with open(output, 'a') as f:
	print(f"\tEvaluation of conformers finished", file=f)

# Creating GAFF2 parameters for the monomer unit and creating specfic chain length single polymer 
param = GAFF2Param(polymer, chain_length)
param.parameterization()
with open(output, 'a') as f:
	print(f"\tCreating GAFF2 parameters and single polymer chain creation finished", file=f)

# Align the single polymer chain, define the box dimension based on the polymer chain, and create the bulk phase
bulk_creator = BulkCreator(polymer, chain_length, num_chains)
bulk_creator.create_bulk_phase()
with open(output, 'a') as f:
	print(f"\tAlignment of single polymer chain and bulk phase creation finished", file=f)