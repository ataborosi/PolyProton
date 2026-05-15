import numpy as np
import pandas as pd
import math
import os
import shutil
import glob
import subprocess
import getpass
import random

import ase
from ase.io import read, write
from ase.optimize import LBFGS, FIRE
from ase.constraints import FixAtoms

from rdkit import Chem
from rdkit.Chem import AllChem, rdGeometry, Draw

from Bio.PDB import PDBParser, PDBIO, Atom, NeighborSearch

from scipy.spatial.transform import Rotation as R

from collections import Counter

import warnings
warnings.simplefilter("ignore")

from mace.calculators import mace_off
nnp_calc = mace_off(model="small", device='cpu')
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile
profile = OrcaProfile(command='/opt/orca/orca')

base_dir = os.getcwd()
output = os.path.join(base_dir, 'new_aspi-h_process_temp.txt')

def get_nproc(default=4):
	return int(os.environ.get("SLURM_NTASKS", default))

def select_conformer(conf_file, conf_sel="best", far_fraction=0.5):
	df = pd.read_csv(conf_file, delim_whitespace=True)
	best_conf = int(df.iloc[0]["conf_i"])
	
	if conf_sel == "best":
		return best_conf
	
	if conf_sel == "random":
		if len(df) == 1:
			return best_conf
		
		others = df.iloc[1:].copy()
		
		if "ΔE" in others.columns and len(others) > 1:
			others = others.sort_values(by="ΔE", ascending=False).reset_index(drop=True)
			n_far = max(1, int(math.ceil(len(others) * far_fraction)))
			far_pool = others.iloc[:n_far]
		else:
			far_pool = others
			
		return int(random.choice(far_pool["conf_i"].tolist()))
		
	raise ValueError("conf_sel must be either 'best' or 'random'")

def generate_chain_lengths(chain_length, num_chains, mix_chains=False, seed=42):
	if not mix-chains:
		return [int(chain_length)] * int(num_chains)
	
	half_range = int(math.ceil(chain_length / 2))
	min_len = max(2, int(chain_length - half_range))
	max_len = int(chain_length + half_range)
	
	random.seed(seed)
	return [random.randint(min_len, max_len) for _ in range(num_chains)]

def system_name(polymer, chain_length, num_chains, mix_chains=False)
	if mix_chains:
		return f"{polymer}_mix-n-{chain_length}x{num_chains}"
	return f"{polymer}_n-{chain_length}x{num_chains}"

def format_leap_sequence(sequence, residues_per_line=20):
	lines = []
	lines.append("{")

	for i in range(0, len(sequence), residues_per_line):
		chunk = sequence[i:i + residues_per_line]
		lines.append("	  " + " ".join(chunk))

	lines.append("}")
	return "\n".join(lines)
			
class Mol2Modification:
	def __init__(self, input_mol2, output_mol2=None, remove_atom_types=None):
		self.input_mol2 = input_mol2
		self.output_mol2 = output_mol2 if output_mol2 else input_mol2.replace(".mol2", "_mod.mol2")
		self.remove_atom_types = set(t.lower() for t in (remove_atom_types if remove_atom_types else ["ho"]))
	
	def split_sections(self, lines):
		sections = {}
		current = None
		for line in lines:
			if line.startswith("@<TRIPOS>"):
				current = line.strip()
				sections[current] = []
			elif current is not None:
				sections[current].append(line)
		return sections

	def parse_atom_line(self, line):
		parts = line.split()
		return {
			"atom_id": int(parts[0]),
			"atom_name": parts[1],
			"x": float(parts[2]),
			"y": float(parts[3]),
			"z": float(parts[4]),
			"atom_type": parts[5],
			"subst_id": parts[6],
			"subst_name": parts[7],
			"charge": float(parts[8]),
		}
	
	def parse_bond_line(self, line):
		parts = line.split()
		return {
			"bond_id": int(parts[0]),
			"origin": int(parts[1]),
			"target": int(parts[2]),
			"bond_type": parts[3],
		}
	
	def format_atom(self, atom):
		return (
			f"{atom['atom_id']:>7} "
			f"{atom['atom_name']:<8}"
			f"{atom['x']:>10.4f}"
			f"{atom['y']:>10.4f}"
			f"{atom['z']:>10.4f} "
			f"{atom['atom_type']:<6}"
			f"{int(atom['subst_id']):>6} "
			f"{(atom['subst_name']):<8}"
			f"{(atom['charge']):>12.6f}\n"
		)		
	
	def format_bond(self, bond):
		return (
			f"{bond['bond_id']:>6} "
			f"{bond['origin']:>5} "
			f"{bond['target']:>5} "
			f"{bond['bond_type']}\n"
		)
	
	def remove_atoms_bonds(self):
		with open(self.input_mol2, "r") as f:
			lines = f.readlines()
			
		sections = self.split_sections(lines)
		
		atoms = [self.parse_atom_line(line) for line in sections["@<TRIPOS>ATOM"] if line.strip()]
		bonds = [self.parse_bond_line(line) for line in sections["@<TRIPOS>BOND"] if line.strip()]
		
		atoms_to_remove = {
			atom["atom_id"]
			for atom in atoms
			if atom["atom_type"].lower() in self.remove_atom_types
		}
		
		kept_atoms = [atom for atom in atoms if atom["atom_id"] not in atoms_to_remove]
		
		if "ho" in self.remove_atom_types:
			for atom in kept_atoms:
				if atom["atom_type"].lower() == "oh":
					atom["atom_type"] = "o"
		
		atom_id_map = {}
		for new_id, atom in enumerate(kept_atoms, start=1):
			atom_id_map[atom["atom_id"]] = new_id
			atom["atom_id"] = new_id
		
		kept_bonds = []
		for bond in bonds:
			if bond["origin"] in atoms_to_remove or bond["target"] in atoms_to_remove:
				continue
			bond["origin"] = atom_id_map[bond["origin"]]
			bond["target"] = atom_id_map[bond["target"]]
			kept_bonds.append(bond)
		
		for new_id, bond in enumerate(kept_bonds, start=1):
			bond["bond_id"] = new_id
		
		molecule_lines = sections.get("@<TRIPOS>MOLECULE", [])
		mol_name = molecule_lines[0]
		counts_line = f"{len(kept_atoms):>5} {len(kept_bonds):>5}	  1		0	  0\n"
		
		new_molecule_lines = [mol_name, counts_line]
		if len(molecule_lines) > 2:
			new_molecule_lines.extend(molecule_lines[2:])
		
		with open(self.output_mol2, "w") as f:
			f.write("@<TRIPOS>MOLECULE\n")
			f.writelines(new_molecule_lines)
			
			f.write("@<TRIPOS>ATOM\n")
			for atom in kept_atoms:
				f.write(self.format_atom(atom))
			
			f.write("@<TRIPOS>BOND\n")
			for bond in kept_bonds:
				f.write(self.format_bond(bond))

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
			file_opt = f"{self.polymer}_{conf_i}_opt.txt"
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
	def __init__(self, polymer, chain_length, nproc):
		self.polymer = polymer
		self.chain_length = chain_length
		self.nproc = nproc

	def remove_atoms(self, mol_in, mol_out):
		mol_in = read(mol_in)
		o_indices = [i for i, s in enumerate(mol_in.get_chemical_symbols()) if s == "O"]
		oh_h_indices = [
			h for h, s in enumerate(mol_in.get_chemical_symbols())
			if s == "H" and any(mol_in.get_distance(h, o, mic=True) < 1.2 for o in o_indices)
		]
		for i in sorted(oh_h_indices, reverse=True):
			del mol_in[i]
		write(mol_out, mol_in)
		
	def orca_calculation(self, mol, charge, mult, file_name):
		orca_calc = ORCA(
			profile=profile,
			orcasimpleinput='wb97x-d4 def2-svp def2/j rijcosx tightscf',
			charge=charge,
			mult=mult,
			orcablocks=f'%pal nprocs {self.nproc} end'
		)
		mol.set_calculator(orca_calc)
		mol.get_potential_energy()
		for f in glob.glob("orca.*"):
			new_name = f.replace("orca.", f"{file_name}.", 1)
			if not os.path.exists(new_name):
				shutil.move(f, new_name)	
		orca_command = f'/opt/orca/orca_2mkl {file_name} -molden'
		
		subprocess.run(orca_command, shell=True, check=True)

	def run_multiwfn(self, file_name):
		molden_file = f'{file_name}.molden.input'
		multiwfn_input = f'{molden_file}\n7\n18\n10\n2\n1\ny\n0\n0\nq\n'
		with open("multiwfn_input.txt", "w") as input_file:
			input_file.write(multiwfn_input)
		multiwfn_command = f'Multiwfn < multiwfn_input.txt -set /opt/multiwfn/settings.ini'
		
		subprocess.run(multiwfn_command, shell=True, check=True)

	def convert_xyz_to_mol2(self, xyz_file, file_name):
		mol2_file = f'{file_name}.mol2'
		obabel_command = f'obabel -i xyz {xyz_file} -O {mol2_file}'
		
		subprocess.run(obabel_command, shell=True, check=True)

	def modify_mol2_file(self, file_name, mod_file_name):
		mol2_file = f'{file_name}.mol2'
		with open(mol2_file, 'r') as f:
			mol2_lines = f.readlines()
		atom_start = next(i for i, line in enumerate(mol2_lines) if line.startswith('@<TRIPOS>ATOM'))
		bond_start = next(i for i, line in enumerate(mol2_lines) if line.startswith('@<TRIPOS>BOND'))
		atom_lines = mol2_lines[atom_start + 1:bond_start]
		atom_data = [line.split() for line in atom_lines]
		atom_df = pd.DataFrame(atom_data)
		chg_candidates = [f"{file_name}.molden.chg", f"{file_name}.chg"]
		molden_chg_file = None
		for f in chg_candidates:
			if os.path.exists(f):
				molden_chg_file = f
				break
		molden_df = pd.read_csv(molden_chg_file, delim_whitespace=True, header=None)
		atom_df[7] = self.polymer
		atom_df[8] = molden_df[4].map('{:.4f}'.format)
		atom_lines_modified = [' '.join(row) + '\n' for row in atom_df.values.astype(str)]
		modified_mol2_file = f'{mod_file_name}.mol2'
		with open(modified_mol2_file, 'w') as f:
			f.writelines(mol2_lines[:atom_start + 1] + atom_lines_modified + mol2_lines[bond_start:])

	def run_antechamber_v1(self):
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

	def run_antechamber_v2(self):
		antechamber_command_1 = f'/opt/amber/amber24/bin/wrapped_progs/antechamber -i {self.polymer}_so3_mod.mol2 -fi mol2 -o {self.polymer}_so3_gaff2.mol2 -fo mol2 -at gaff2'
		subprocess.run(antechamber_command_1, shell=True, check=True)
		antechamber_command_2 = f'/opt/amber/amber24/bin/wrapped_progs/antechamber -i {self.polymer}_so3_gaff2.mol2 -fi mol2 -o {self.polymer}_so3.ac -fo ac'
		subprocess.run(antechamber_command_2, shell=True, check=True)
		antechamber_command_3 = f'/opt/amber/amber24/bin/wrapped_progs/parmchk2 -i {self.polymer}_so3_gaff2.mol2 -f mol2 -o {self.polymer}_so3_gaff2.frcmod -s 2'
		subprocess.run(antechamber_command_3, shell=True, check=True)	
		prepgen_command_1 = f'/opt/amber/amber24/bin/wrapped_progs/prepgen -i {self.polymer}_so3.ac -o {self.polymer}_m_so3.prepi -f prepi -m main_so3 -rn {self.polymer}'
		subprocess.run(prepgen_command_1, shell=True, check=True)
		prepgen_command_2 = f'/opt/amber/amber24/bin/wrapped_progs/prepgen -i {self.polymer}_so3.ac -o h_so3.prepi -f prepi -m head_so3 -rn H'
		prepgen_command_3 = f'/opt/amber/amber24/bin/wrapped_progs/prepgen -i {self.polymer}_so3.ac -o t_so3.prepi -f prepi -m tail_so3 -rn T'
		
		subprocess.run(prepgen_command_2, shell=True, check=True)
		subprocess.run(prepgen_command_3, shell=True, check=True)

	def create_polymer_chain(self, chain_length):
		if chain_length is None:
			chain_length = self.chain_length
		
		sequence = ['H']
		num_middle_units = chain_length - 2
		sequence.extend([f'{self.polymer}'] * num_middle_units)
		sequence.append('T')

		formatted_sequence = '{' + ' '.join(sequence) + '}'
		leap_input_filename = f'leap_input_n-{current_chain_length}.in'
		leap_output_filename = f'leap_input_n-{current_chain_length}.out'
		
		with open(leap_input_filename, 'w') as f:
			f.write('source leaprc.gaff2\n')
			f.write(f'loadamberprep h.prepi\n')
			f.write(f'loadamberprep t.prepi\n')
			f.write(f'loadamberprep {self.polymer}_m.prepi\n')
			f.write(f'loadamberparams {self.polymer}_gaff2.frcmod\n')
			f.write(f'mol = sequence {formatted_sequence}\n')
			f.write(f'savepdb mol {self.polymer}_n-{self.chain_length}.pdb\n')
			f.write(f'saveamberparm mol {self.polymer}_n-{self.chain_length}.prmtop {self.polymer}_n-{self.chain_length}.inpcrd\n')
			f.write('quit\n')
		
		subprocess.run(f'tleap -f {leap_input_filename} > {leap_output_filename}', shell=True, check=True)
	
	def parameterization(self, selected_conf):
		read_xyz_1 = read(f'{self.polymer}_{selected_conf}_opt.xyz')
		xyz_file_1 = f'{self.polymer}_{selected_conf}_opt.xyz'
		file_name_1 = f"{self.polymer}"
		mod_file_name_1 = f"{self.polymer}_mod"
		charge_1 = 0
		mult_1 = 1
		self.orca_calculation(read_xyz_1, charge_1, mult_1, file_name_1)
		self.run_multiwfn(file_name_1)
		self.convert_xyz_to_mol2(xyz_file_1, file_name_1)
		self.modify_mol2_file(file_name_1, mod_file_name_1)
		self.run_antechamber_v1()
		self.create_polymer_chain()
		xyz_file_2 = f'{self.polymer}_{selected_conf}_opt_so3.xyz'
		charge_2 = -2
		mult_2 = 1
		self.remove_atoms(xyz_file_1, xyz_file_2)
		read_xyz_2 = read(f'{self.polymer}_{selected_conf}_opt_so3.xyz')
		file_name_2 = f"{self.polymer}_so3"
		mol_file_name_1 = f"{self.polymer}_gaff2.mol2"
		mol_file_name_2 = f"{self.polymer}_so3.mol2"
		mod_file_name_2 = f"{self.polymer}_so3_mod"
		remove_atom_types = ["ho"]
		self.orca_calculation(read_xyz_2, charge_2, mult_2, file_name_2)
		self.run_multiwfn(file_name_2)
		mol2_modifier = Mol2Modification(mol_file_name_1, mol_file_name_2, remove_atom_types)
		mol2_modifier.remove_atoms_bonds()	
		self.modify_mol2_file(file_name_2, mod_file_name_2)
		self.run_antechamber_v2()

class Dry_BulkCreator:
	def __init__(self, polymer, chain_length, num_chains, mix_chains=False, chain_lengths=None, aligned=True):
		self.polymer = polymer
		self.chain_length = chain_length
		self.num_chains = num_chains
		self.mix_chains = mix_chains
		self.chain_lengths = chain_lengths if chain_lengths else [chain_length] * num_chains
		self.aligned = aligned
		self.box_size_x = 0
		self.box_size_y = 0
		self.box_size_z = 0
		self.output_pdb = f"{system_name(self.polymer, self.chain_length, self.num_chains, self.mix_chains)}.pdb"
		
	def rotate_chain(self, input_pdb, output_pdb):
		parser = PDBParser(QUIET=True)
		pdb_structure = parser.get_structure('original', input_pdb)
		single_chain = read(input_pdb)
		positions = single_chain.get_positions()
		center_of_mass = np.mean(positions, axis=0)
		single_chain.translate(-center_of_mass)
		cov_matrix = np.cov(positions.T)
		eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
		longest_axis = eigenvectors[:, np.argmax(eigenvalues)]
		z_axis = np.array([0, 0, 1])
		rotation_axis = np.cross(longest_axis, z_axis)
		denom = np.linalg.norm(longest_axis) * np.linalg.norm(z_axis)
		if denom == 0:
			rotation_angle = 0.0
		else:
			rotation_angle = np.arccos(np.clip(np.dot(longest_axis, z_axis) / denom, -1.0, 1.0))
		single_chain.rotate(v=rotation_axis, a=np.degrees(rotation_angle), center='COM')
		new_positions = single_chain.get_positions()
		for i, atom in enumerate(pdb_structure.get_atoms()):
			atom.set_coord(new_positions[i])
		io = PDBIO()
		io.set_structure(pdb_structure)
		io.save(output_pdb)

	def box_dimension_single(self, pdb_file):
		structure = read(pdb_file)
		positions = structure.get_positions()
		min_coords = np.min(positions, axis=0)
		max_coords = np.max(positions, axis=0)
		return max_coords - min_coords

	def prepare_rotated_chains(self):
		if not self.mix_chains:
			input_pdb = f'{self.polymer}_n-{self.chain_length}.pdb'
			output_pdb = f'{self.polymer}_n-{self.chain_length}_rot.pdb'
			self.rotate_chain(input_pdb, output_pdb)
			self.rotated_single_chain = output_pdb
			return

		self.unique_lengths = sorted(set(self.chain_lengths))
		self.length_counts = Counter(self.chain_lengths)
		self.rotated_pdbs = []
		self.dims = []

		for L in self.unique_lengths:
			input_pdb = f'{self.polymer}_n-{L}.pdb'
			output_pdb = f'{self.polymer}_n-{L}_rot.pdb'
			self.rotate_chain(input_pdb, output_pdb)
			self.rotated_pdbs.append(output_pdb)
			self.dims.append(self.box_dimension_single(output_pdb))

	def box_dimension(self):
		if not self.mix_chains:
			structure = read(self.rotated_single_chain)
			positions = structure.get_positions()
			min_coords = np.min(positions, axis=0)
			max_coords = np.max(positions, axis=0)
			dim_x, dim_y, dim_z = max_coords - min_coords
			polymer_volume = dim_x * dim_y * dim_z
			total_box_volume = polymer_volume * self.num_chains
			box_height = dim_z + 30
			box_side_area = total_box_volume / box_height

			if self.aligned:
				scale_factor = 1.65 - 0.3 * np.log2(1 + (dim_x + dim_y) / 10)
				scale_factor = max(0.5, min(1.5, scale_factor))
			else:
				scale_factor = 3.0 - 0.3 * np.log2(1 + (dim_x + dim_y) / 10)
				scale_factor = max(0.5, min(3.5, scale_factor))

			self.box_size_x = np.sqrt(box_side_area) * scale_factor
			self.box_size_y = np.sqrt(box_side_area) * scale_factor
			self.box_size_z = box_height
			return

		counts = [self.length_counts[L] for L in self.unique_lengths]
		dim_xs, dim_ys, dim_zs = zip(*self.dims)

		avg_dim_x = sum(dx * k for dx, k in zip(dim_xs, counts)) / self.num_chains
		avg_dim_y = sum(dy * k for dy, k in zip(dim_ys, counts)) / self.num_chains
		max_dim_z = max(dim_zs)

		total_box_volume = sum((dx * dy * dz) * k for (dx, dy, dz), k in zip(self.dims, counts))
		box_height = max_dim_z + 30.0
		box_side_area = total_box_volume / box_height

		if self.aligned:
			scale_factor = 1.65 - 0.3 * np.log2(1 + (dim_x + dim_y) / 10)
			scale_factor = max(0.5, min(1.5, scale_factor))
		else:
			scale_factor = 3.0 - 0.3 * np.log2(1 + (dim_x + dim_y) / 10)
			scale_factor = max(0.5, min(3.5, scale_factor))

		self.box_size_x = np.sqrt(box_side_area) * scale_factor
		self.box_size_y = np.sqrt(box_side_area) * scale_factor
		self.box_size_z = box_height

    def packmol_generate_box(self):
        with open("packmol_input.inp", "w") as f:
            f.write("tolerance 2.0\n")
            f.write("filetype pdb\n")
            f.write("add_amber_ter\n")
            f.write(f"output {self.output_pdb}\n")
    
            if not self.mix_chains:
                f.write(f"structure {self.rotated_single_chain}\n")
                f.write(f"number {self.num_chains}\n")
                f.write(f"inside box 0. 0. 0. {self.box_size_x} {self.box_size_y} {self.box_size_z}\n")
                if self.aligned:
                    f.write("constrain_rotation x 0. 0.\n")
                    f.write("constrain_rotation y 0. 0.\n")
                    f.write("constrain_rotation z 0. 0.\n")
                f.write("end structure\n")
            else:
                for L, rotated_pdb in zip(self.unique_lengths, self.rotated_pdbs):
                    f.write(f"structure {rotated_pdb}\n")
                    f.write(f"number {self.length_counts[L]}\n")
                    f.write(f"inside box 0. 0. 0. {self.box_size_x} {self.box_size_y} {self.box_size_z}\n")
                    if self.aligned:
                        f.write("constrain_rotation x 0. 0.\n")
                        f.write("constrain_rotation y 0. 0.\n")
                        f.write("constrain_rotation z 0. 0.\n")
                    f.write("end structure\n")
    
        subprocess.run('/opt/packmol/packmol-20.15.1/packmol < packmol_input.inp', shell=True, check=True)
#		subprocess.run('packmol < packmol_input.inp', shell=True)
	
	def create_bulk_phase(self):
		self.prepare_rotated_chains()
		self.box_dimension()
		self.packmol_generate_box()

class Dry_AmberParams:
	def __init__(self, polymer, chain_length, num_chains, bulk_creator: Dry_BulkCreator):
		self.polymer = polymer
		self.chain_length = chain_length
		self.num_chains = num_chains
		self.mix_chains = bulk_creator.mix_chains
		self.box_size_x = bulk_creator.box_size_x
		self.box_size_y = bulk_creator.box_size_y
		self.box_size_z = bulk_creator.box_size_z
		self.system_tag = system_name(self.polymer, self.chain_length, self.num_chains, self.mix_chains)
		self.pdb_file = f"{self.system_tag}.pdb"

	def create_amber_params(self):
		with open('final_leap_input.in', 'w') as f:
			f.write(f"""
			source leaprc.gaff2
			loadamberprep h.prepi
			loadamberprep t.prepi
			loadamberprep {self.polymer}_m.prepi
			loadamberparams {self.polymer}_gaff2.frcmod
			mol = loadpdb {self.pdb_file}
			set mol box {{ {self.box_size_x} {self.box_size_y} {self.box_size_z} }}
			set default nocenter on
			saveamberparm mol {self.system_tag}.prmtop {self.system_tag}.inpcrd
			quit
			""")

		self.prmtop = f"{self.system_tag}.prmtop"
		self.inpcrd = f"{self.system_tag}.inpcrd"
		amber_pdb = f"{self.system_tag}_amber.pdb"

		subprocess.run('tleap -f final_leap_input.in > final_leap_input.out', shell=True, check=True)
		ambpdb_command = f"ambpdb -p {self.prmtop} -c {self.inpcrd} > {amber_pdb}"
		subprocess.run(ambpdb_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class Dry_MDSimulation():
	def __init__(self, nproc, output, amber_params: Dry_AmberParams, use_gpu=True):
		self.prmtop = amber_params.prmtop
		self.inpcrd = amber_params.inpcrd
		self.nproc = nproc
		self.dir1 = os.getcwd()
		self.uname = getpass.getuser()
		self.output = output
		self.nproc = nproc
		self.use_gpu = use_gpu

	def run_simulation(self, step_name, input_file, output_file, restart_in, restart_out, reference_file, additional_args=""):
		with open(self.output, 'a') as f:
			print(f"\t\tStarted {step_name} step", file=f)

		temp_dir = subprocess.check_output(['mktemp', '-d', f'/home/Calculations/{self.uname}/XXXXXX']).decode().strip()
		folder_name = f"{step_name}"
		os.makedirs(folder_name, exist_ok=True)
		
		files_to_copy = [input_file, self.prmtop, restart_in]
		for file in files_to_copy:
			if os.path.exists(file):
				shutil.copy(file, folder_name)
			else:
				raise FileNotFoundError(f"{file} not found")

		os.chdir(folder_name)
		subprocess.run(f"cp * {temp_dir}", shell=True, check=True)
		os.chdir(temp_dir)
		
		if self.use_gpu:
			if any(x in step_name for x in ["min", "npt"]):
				cmd = f"mpirun -np {self.nproc} pmemd.MPI -O -i {input_file} -o {output_file} -p {self.prmtop} -c {restart_in} -r {restart_out} -ref {reference_file} {additional_args}"
			else:
				cmd = f"pmemd.cuda -O -i {input_file} -o {output_file} -p {self.prmtop} -c {restart_in} -r {restart_out} -ref {reference_file} {additional_args} -AllowSmallBox"
		else:
			cmd = f"mpirun -np {self.nproc} pmemd.MPI -O -i {input_file} -o {output_file} -p {self.prmtop} -c {restart_in} -r {restart_out} -ref {reference_file} {additional_args}"

		result = subprocess.run(cmd, shell=True)
		if result.returncode != 0:
			raise RuntimeError(f"{step_name} step failed")

		subprocess.run(f"cp * {self.dir1}", shell=True, check=True)
		shutil.rmtree(temp_dir)

		os.chdir(self.dir1)
		shutil.rmtree(folder_name)
		with open(self.output, 'a') as f:
			print(f"\t\tFinished {step_name} step", file=f)

	def run_all_steps(self):
		steps = [
			("dry-eq_min", "dry-eq_0-min.in", "dry-eq_0-min.out", f"{self.inpcrd}", "dry-eq_0-min.ncrst", f"{self.inpcrd}"),
			("dry-eq_1-nvt", "dry-eq_1-nvt.in", "dry-eq_1-nvt.out", "dry-eq_0-min.ncrst", "dry-eq_1-nvt.ncrst", "dry-eq_0-min.ncrst", "-x dry-eq_1-nvt.nc"),
			("dry-eq_2-npt", "dry-eq_2-npt.in", "dry-eq_2-npt.out", "dry-eq_1-nvt.ncrst", "dry-eq_2-npt.ncrst", "dry-eq_1-nvt.ncrst", "-x dry-eq_2-npt.nc"),
			("dry-eq_3-nvt", "dry-eq_3-nvt.in", "dry-eq_3-nvt.out", "dry-eq_2-npt.ncrst", "dry-eq_3-nvt.ncrst", "dry-eq_2-npt.ncrst", "-x dry-eq_3-nvt.nc"),
			("dry-eq_4-npt", "dry-eq_4-npt.in", "dry-eq_4-npt.out", "dry-eq_3-nvt.ncrst", "dry-eq_4-npt.ncrst", "dry-eq_3-nvt.ncrst", "-x dry-eq_4-npt.nc"),
			("dry-eq_5-nvt", "dry-eq_5-nvt.in", "dry-eq_5-nvt.out", "dry-eq_4-npt.ncrst", "dry-eq_5-nvt.ncrst", "dry-eq_4-npt.ncrst", "-x dry-eq_5-nvt.nc"),
			("dry-eq_6-npt", "dry-eq_6-npt.in", "dry-eq_6-npt.out", "dry-eq_5-nvt.ncrst", "dry-eq_6-npt.ncrst", "dry-eq_5-nvt.ncrst", "-x dry-eq_6-npt.nc"),
			("dry-eq_7-nvt-pr", "dry-eq_7-nvt-pr.in", "dry-eq_7-nvt-pr.out", "dry-eq_6-npt.ncrst", "dry-eq_7-nvt-pr.ncrst", "dry-eq_6-npt.ncrst", "-x dry-eq_7-nvt-pr.nc"),
		]

		for step in steps:
			self.run_simulation(*step)

class Analysis():
	def merge_nc_files(self, prmtop_file, ncrst_file, pdb_file, nc_files, prefix, merged_pdb, cpptraj_file):
		ambpdb_command = f"ambpdb -p {prmtop_file} -c {ncrst_file} > {pdb_file}"
		subprocess.run(ambpdb_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		
		with open(cpptraj_file, 'w') as file:
			for nc_file in nc_files:			
				file.write(f"trajin {nc_file} 1 100 10\n")
			file.write(f"trajout {prefix} pdb multi\n")
		
		cpptraj_command = f"cpptraj -i {cpptraj_file} -p {prmtop_file}"
		subprocess.run(cpptraj_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		
		merge_command = f"ls -v {prefix}* | xargs cat > {merged_pdb}"
		subprocess.run(merge_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		
		for file in os.listdir():
			if file.startswith(prefix):
				os.remove(file)			

class Hyd_BulkCreator:
	def __init__(self, polymer, chain_length, num_chains, x, y, z, num_h2o, lam, pdb_file):
		self.polymer = polymer
		self.chain_length = chain_length
		self.num_chains = num_chains
		self.box_size_x = x
		self.box_size_y = y
		self.box_size_z = z
		self.num_h2o = num_h2o
		self.lam = lam
		self.pdb_file = pdb_file
		
	def packmol_generate_box(self):
		with open("packmol_input.inp", "w") as f:
			f.write(f"""
			tolerance 1.5
			filetype pdb
			add_amber_ter
			amber_ter_preserve
			output {self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h2o.pdb
			pbc {self.box_size_x} {self.box_size_y} {self.box_size_z}
			structure {self.pdb_file}
			number 1
			fixed 0. 0. 0. 0. 0. 0.
			end structure
			structure h2o.pdb
			number {self.num_h2o}
			end structure
			""")

		subprocess.run('/opt/packmol/packmol-20.15.1/packmol < packmol_input.inp', shell=True)

	def create_bulk_phase(self):
		self.packmol_generate_box()

class Hyd_AmberParams:
	def __init__(self, polymer, chain_length, num_chains, bulk_creator: Hyd_BulkCreator):
		self.polymer = polymer
		self.chain_length = chain_length
		self.num_chains = num_chains
		self.box_size_x = bulk_creator.box_size_x
		self.box_size_y = bulk_creator.box_size_y
		self.box_size_z = bulk_creator.box_size_z
		self.lam = bulk_creator.lam
		self.pdb_file = f"{self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h2o.pdb"

	def create_amber_params(self):
		with open("final_leap_input.in", "w") as f:
			f.write(f"""
			source leaprc.gaff2
			source leaprc.water.tip3p
			loadamberparams frcmod.tip4p
			loadamberprep h.prepi
			loadamberprep t.prepi
			loadamberprep {self.polymer}_m.prepi
			loadamberparams {self.polymer}_gaff2.frcmod
			mol = loadpdb {self.pdb_file}
			set mol box {{ {self.box_size_x} {self.box_size_y} {self.box_size_z} }}
			set default nocenter on
			saveamberparm mol {self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h2o.prmtop {self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h2o.inpcrd
			quit			
			""")

		self.prmtop = f"{self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h2o.prmtop"
		self.inpcrd = f"{self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h2o.inpcrd"
		amber_pdb = f"{self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h2o_amber.pdb"
	
		subprocess.run('tleap -f final_leap_input.in > final_leap_input.out', shell=True, check=True)
		ambpdb_command = f"ambpdb -p {self.prmtop} -c {self.inpcrd} > {amber_pdb}"
		subprocess.run(ambpdb_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class Hyd_MDSimulation():
	def __init__(self, nproc, output, amber_params: Hyd_AmberParams, use_gpu=True):
		self.prmtop = amber_params.prmtop
		self.inpcrd = amber_params.inpcrd
		self.lam = amber_params.lam
		self.nproc = nproc
		self.dir1 = os.getcwd()
		self.uname = getpass.getuser()
		self.output = output
		self.nproc = nproc
		self.use_gpu = use_gpu
		
	def run_simulation(self, step_name, input_file, output_file, restart_in, restart_out, reference_file, additional_args=""):
		with open(self.output, "a") as f:
			print(f"\t\tStarted {step_name} step", file=f)
			
		temp_dir = subprocess.check_output(['mktemp', '-d', f'/home/Calculations/{self.uname}/XXXXXX']).decode().strip()
		folder_name = f"{step_name}"
		os.makedirs(folder_name, exist_ok=True)
		
		files_to_copy = [input_file, self.prmtop, restart_in]
		for file in files_to_copy:
			if os.path.exists(file):
				shutil.copy(file, folder_name)
			else:
				raise FileNotFoundError(f"{file} not found")

		os.chdir(folder_name)
		subprocess.run(f"cp * {temp_dir}", shell=True, check=True)
		os.chdir(temp_dir)

		if self.use_gpu:
			if "min" in step_name:
				cmd = f"mpirun -np {self.nproc} pmemd.MPI -O -i {input_file} -o {output_file} -p {self.prmtop} -c {restart_in} -r {restart_out} -ref {reference_file} {additional_args}"
			else:
				cmd = f"pmemd.cuda -O -i {input_file} -o {output_file} -p {self.prmtop} -c {restart_in} -r {restart_out} -ref {reference_file} {additional_args} -AllowSmallBox"
		else:
			cmd = f"mpirun -np {self.nproc} pmemd.MPI -O -i {input_file} -o {output_file} -p {self.prmtop} -c {restart_in} -r {restart_out} -ref {reference_file} {additional_args}"

		result = subprocess.run(cmd, shell=True)
		if result.returncode != 0:
			raise RuntimeError(f"{step_name} step failed for lambda={self.lam}")

		subprocess.run(f"cp * {self.dir1}", shell=True, check=True)
		shutil.rmtree(temp_dir)

		os.chdir(self.dir1)
		shutil.rmtree(folder_name)
		with open(self.output, 'a') as f:
			print(f"\t\tFinished {step_name} step", file=f)
		os.chdir(self.dir1)

	def run_all_steps(self):
		steps = [
			("hyd-eq_min", "hyd-eq_0-min.in", "hyd-eq_0-min.out", f"{self.inpcrd}", "hyd-eq_0-min.ncrst", f"{self.inpcrd}"),
			("hyd-eq_1-nvt", "hyd-eq_1-nvt.in", "hyd-eq_1-nvt.out", "hyd-eq_0-min.ncrst", "hyd-eq_1-nvt.ncrst", "hyd-eq_0-min.ncrst", "-x hyd-eq_1-nvt.nc"),
			("hyd-eq_2-nvt", "hyd-eq_2-nvt.in", "hyd-eq_2-nvt.out", "hyd-eq_1-nvt.ncrst", "hyd-eq_2-nvt.ncrst", "hyd-eq_1-nvt.ncrst", "-x hyd-eq_2-nvt.nc"),
			("hyd-eq_3-nvt", "hyd-eq_3-nvt.in", "hyd-eq_3-nvt.out", "hyd-eq_2-nvt.ncrst", "hyd-eq_3-nvt.ncrst", "hyd-eq_2-nvt.ncrst", "-x hyd-eq_3-nvt.nc"),
			("hyd-eq_4-npt", "hyd-eq_4-npt.in", "hyd-eq_4-npt.out", "hyd-eq_3-nvt.ncrst", "hyd-eq_4-npt.ncrst", "hyd-eq_3-nvt.ncrst", "-x hyd-eq_4-npt.nc"),
			("hyd-eq_5-nvt-pr", "hyd-eq_5-nvt-pr.in", "hyd-eq_5-nvt-pr.out", "hyd-eq_4-npt.ncrst", "hyd-eq_5-nvt-pr.ncrst", "hyd-eq_4-npt.ncrst", "-x hyd-eq_5-nvt-pr.nc"),
		]

		for step in steps:
			self.run_simulation(*step)

class PDBCleaner:
	def __init__(self, input_pdb_file, output_pdb_file, chain_length):
		self.input_pdb_file = input_pdb_file
		self.output_pdb_file = output_pdb_file
		self.chain_length = chain_length
		self.hydrogens_removed_count = 0

	def is_atom_line(self, line):
		return line.startswith("ATOM") or line.startswith("HETATM")

	def resname(self, line):
		return line[17:20].strip()

	def atomname(self, line):
		return line[12:16].strip()

	def element(self, line):
		elem = line[76:78].strip()
		if elem:
			return elem
		name = self.atomname(line)
		return ''.join([c for c in name if c.isalpha()])[:1]

	def coord(self, line):
		return np.array([
			float(line[30:38]),
			float(line[38:46]),
			float(line[46:54]),
		])

	def remove_so3h_hydrogens(self):
		with open(self.input_pdb_file, "r") as f:
			lines = f.readlines()

		atom_lines = [line for line in lines if self.is_atom_line(line)]
		cryst_line = next((line for line in lines if line.startswith("CRYST1")), None)
		box_line = next((line for line in reversed(lines) if len(line.strip().split()) == 6), None)

		residues = []
		current = []
		last_key = None

		for line in atom_lines:
			key = (line[21], line[22:26], line[26], line[17:20])
			if last_key is not None and key != last_key:
				residues.append(current)
				current = []
			current.append(line)
			last_key = key

		if current:
			residues.append(current)

		remove_line_indices = set()
		hydrogens_removed = 0

		for residue in residues:
			resname = self.resname(residue[0])
			if resname in ["WAT", "HOH", "H3O"]:
				continue

			s_atoms = []
			o_atoms = []
			h_atoms = []

			for i, line in enumerate(residue):
				elem = self.element(line)
				if elem == "S":
					s_atoms.append((i, self.coord(line)))
				elif elem == "O":
					o_atoms.append((i, self.coord(line)))
				elif elem == "H":
					h_atoms.append((i, self.coord(line)))

			for s_i, s_coord in s_atoms:
				nearby_o = [
					(o_i, o_coord) for o_i, o_coord in o_atoms
					if np.linalg.norm(o_coord - s_coord) < 1.8
				]

				for o_i, o_coord in nearby_o:
					nearby_h = [
						(h_i, h_coord) for h_i, h_coord in h_atoms
						if np.linalg.norm(h_coord - o_coord) < 1.2
					]

					for h_i, h_coord in nearby_h:
						# use id of line object in full atom_lines list
						remove_line_indices.add(id(residue[h_i]))

		hydrogens_removed = len(remove_line_indices)
		self.hydrogens_removed_count = hydrogens_removed

		waters_to_remove = hydrogens_removed
		cleaned_residues = []
		removed_waters = 0

		for residue in residues:
			resname = self.resname(residue[0])

			if resname in ["WAT", "HOH"] and removed_waters < waters_to_remove:
				removed_waters += 1
				continue

			new_residue = [
				line for line in residue
				if id(line) not in remove_line_indices
			]
			cleaned_residues.append(new_residue)

		with open(self.output_pdb_file, "w") as out:
			if cryst_line:
				out.write(cryst_line)

			previous_resname = None
			polymer_res_count = 0

			for residue in cleaned_residues:
				if not residue:
					continue

				resname = self.resname(residue[0])

				if resname not in ["WAT", "HOH", "H3O"]:
					polymer_res_count += 1
					if polymer_res_count > 1 and (polymer_res_count - 1) % self.chain_length == 0:
						out.write("TER\n")

				if resname in ["WAT", "HOH", "H3O"] and previous_resname not in ["WAT", "HOH", "H3O", None]:
					out.write("TER\n")

				for line in residue:
					out.write(line)

				previous_resname = resname

			out.write("TER\nEND\n")
			if box_line:
				out.write(box_line)

class Cond_BulkCreator:
	def __init__(self, polymer, chain_length, num_chains, x, y, z, num_h3o, lam, pdb_file):
		self.polymer = polymer
		self.chain_length = chain_length
		self.num_chains = num_chains
		self.box_size_x = x
		self.box_size_y = y
		self.box_size_z = z
		self.num_h3o = num_h3o
		self.lam = lam
		self.pdb_file = pdb_file
		
	def packmol_generate_box(self):
		with open("packmol_input.inp", "w") as f:
			f.write(f"""
			tolerance 1.5
			filetype pdb
			add_amber_ter
			amber_ter_preserve
			output {self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h3o-h2o.pdb
			pbc {self.box_size_x} {self.box_size_y} {self.box_size_z}
			structure {self.pdb_file}
			number 1
			fixed 0. 0. 0. 0. 0. 0.
			end structure
			structure h3o.pdb
			number {self.num_h3o}
			end structure
			""")
	
		subprocess.run('/opt/packmol/packmol-20.15.1/packmol < packmol_input.inp', shell=True)
	
	def create_bulk_phase(self):
		self.packmol_generate_box()

class Cond_AmberParams:
	def __init__(self, polymer, chain_length, num_chains, bulk_creator: Cond_BulkCreator):
		self.polymer = polymer
		self.chain_length = chain_length
		self.num_chains = num_chains
		self.box_size_x = bulk_creator.box_size_x
		self.box_size_y = bulk_creator.box_size_y
		self.box_size_z = bulk_creator.box_size_z
		self.lam = bulk_creator.lam
		self.num_h3o = bulk_creator.num_h3o
		self.pdb_file = f"{self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h3o-h2o.pdb"
	
	def create_amber_params(self):
		with open("final_leap_input.in", "w") as f:
			f.write(f"""
			source leaprc.gaff2
			source leaprc.water.tip3p
			loadamberparams frcmod.tip4p
			loadamberprep h_so3.prepi
			loadamberprep t_so3.prepi
			loadamberprep {self.polymer}_m_so3.prepi
			loadamberparams {self.polymer}_so3_gaff2.frcmod
			loadamberprep h3o.prepi
			loadamberparams h3o.frcmod
			mol = loadpdb {self.pdb_file}
			set mol box {{ {self.box_size_x} {self.box_size_y} {self.box_size_z} }}
			set default nocenter on
			saveamberparm mol {self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h3o-h2o.prmtop {self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h3o-h2o.inpcrd
			quit
			""")
		
		self.prmtop = f"{self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h3o-h2o.prmtop"
		self.inpcrd = f"{self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h3o-h2o.inpcrd"
		amber_pdb = f"{self.polymer}_n-{self.chain_length}x{self.num_chains}_{self.lam}-h3o-h2o_amber.pdb"
	
		subprocess.run('tleap -f final_leap_input.in > final_leap_input.out', shell=True, check=True)
		ambpdb_command = f"ambpdb -p {self.prmtop} -c {self.inpcrd} > {amber_pdb}"
		subprocess.run(ambpdb_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class Cond_MDSimulation:
	def __init__(self, nproc, output, amber_params: Cond_AmberParams, use_gpu=True):
		self.prmtop = amber_params.prmtop
		self.inpcrd = amber_params.inpcrd
		self.lam = amber_params.lam
		self.nproc = nproc
		self.dir1 = os.getcwd()
		self.uname = getpass.getuser()
		self.output = output
		self.use_gpu = use_gpu
		
	def run_simulation(self, step_name, input_file, output_file, restart_in, restart_out, reference_file, additional_args=""):
		with open(self.output, "a") as f:
			print(f"\t\tStarted {step_name} step", file=f)
		
		temp_dir = subprocess.check_output(['mktemp', '-d', f'/home/Calculations/{self.uname}/XXXXXX']).decode().strip()
		folder_name = f"{step_name}"
		os.makedirs(folder_name, exist_ok=True)
		
		files_to_copy = [input_file, self.prmtop, restart_in]
		for file in files_to_copy:
			if os.path.exists(file):
				shutil.copy(file, folder_name)
			else:
				raise FileNotFoundError(f"{file} not found")
		
		os.chdir(folder_name)
		subprocess.run(f"cp * {temp_dir}", shell=True, check=True)
		os.chdir(temp_dir)
		
		if self.use_gpu:
			if "min" in step_name:
				cmd = f"mpirun -np {self.nproc} pmemd.MPI -O -i {input_file} -o {output_file} -p {self.prmtop} -c {restart_in} -r {restart_out} -ref {reference_file} {additional_args}"
			else:
				cmd = f"pmemd.cuda -O -i {input_file} -o {output_file} -p {self.prmtop} -c {restart_in} -r {restart_out} -ref {reference_file} {additional_args} -AllowSmallBox"
		else:
			cmd = f"mpirun -np {self.nproc} pmemd.MPI -O -i {input_file} -o {output_file} -p {self.prmtop} -c {restart_in} -r {restart_out} -ref {reference_file} {additional_args}"

		result = subprocess.run(cmd, shell=True)
		if result.returncode != 0:
			raise RuntimeError(f"{step_name} step failed")

		subprocess.run(f"cp * {self.dir1}", shell=True, check=True)
		shutil.rmtree(temp_dir)

		os.chdir(self.dir1)
		shutil.rmtree(folder_name)
		with open(self.output, 'a') as f:
			print(f"\t\tFinished {step_name} step", file=f)

	def run_all_steps(self):
		steps = [
			("cond_0-min", "cond_0-min.in", "cond_0-min.out", f"{self.inpcrd}", "cond_0-min.ncrst", f"{self.inpcrd}"),
			("cond_pr-nvt", "cond_pr-nvt.in", "cond_pr-nvt.out", "cond_0-min.ncrst", "cond_pr-nvt.ncrst", "cond_0-min.ncrst", "-x cond_pr-nvt.nc")
		]

		for step in steps:
			self.run_simulation(*step)
	
# Create a working directory for polymer with subfolders and copy polymer connectivity cards			
polymer = 'a1'
base_dir = os.getcwd()
polymer_dir = os.path.join(base_dir, polymer)
init_dir = os.path.join(polymer_dir, "init")
dry_eq_dir = os.path.join(polymer_dir, "dry-eq")
hyd_eq_dir = os.path.join(polymer_dir, "hyd-eq")
cond_dir = os.path.join(polymer_dir, "cond")
input_dir = os.path.join(base_dir, "input_files")

os.makedirs(polymer_dir, exist_ok=True)
os.makedirs(init_dir, exist_ok=True)
os.makedirs(dry_eq_dir, exist_ok=True)
os.makedirs(hyd_eq_dir, exist_ok=True)
os.makedirs(cond_dir, exist_ok=True)

for cards in ["head", "main", "tail", "head_so3", "main_so3", "tail_so3"]:
	shutil.copy(os.path.join(input_dir, cards), init_dir)

os.chdir(init_dir)

# Details of polymer
backbone_smiles = 'C1=CC2=C3C(=CC=C4C3=C1C(=O)OC4=O)C(=O)OC2=O'
sidechain_smiles="OCCCS(O)(=O)=O"
benzene_smiles="C1=CC=CC=C1"
conf_num = 5
conf_selection = "best" #best or random
conf_output_file = f"{polymer}_conf.txt"
temperature = 300
chain_length = 15
num_chains = 30
mix_chains = False
mix_seed = 42
chain_lengths = generate_chain_lengths(chain_length, num_chains, mix_chains, mix_seed)
system_tag = system_name(polymer, chain_length, num_chains, mix_chains)
aligned = True
nproc = get_nproc()
use_gpu = os.getenv("USE_GPU", "true").lower() == "true"

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

# Performing the evaluation of conformers based on their total energies and Boltzmann distribution, selection of conformer
analyzer = ConformationAnalyzer(polymer, conf_num, conf_output_file, temperature)
analyzer.analyze_conformers()	
with open(output, 'a') as f:
	print(f"\tEvaluation of conformers finished", file=f)

selected_conf = select_conformer(conf_output_file, conf_selection)
with open(output, 'a') as f:
	print(f"\tSelected conformer mode = {conf_selection}, conf_i = {selected_conf}", file=f)

# Creating GAFF2 parameters for the monomer unit and creating specfic chain length single polymer 
param = GAFF2Param(polymer, chain_length, nproc)
param.parameterization(selected_conf)
if mix_chains:
	for L in sorted(set(chain_lengths)):
		if L != chain_length:
			param.create_polymer_chain(chain_length=L)
with open(output, 'a') as f:
	print(f"\tGAFF2 parameters and single polymer chain creation finished", file=f)

# Align the single polymer chain, define the box dimension based on the polymer chain, and create the bulk phase
with open(output, 'a') as f:
	print(f"\tChain mode mix_chains = {mix_chains}", file=f)
	print(f"\tChain lengths = {chain_lengths}", file=f)
dry_bulk_creator = Dry_BulkCreator(polymer, chain_length, num_chains, mix_chains, chain_lengths, aligned)
dry_bulk_creator.create_bulk_phase()
with open(output, 'a') as f:
	print(f"\tBulk phase creation finished", file=f)
	print(f"\tAligned packing = {aligned}", file=f)
	
# Create amber parameters
dry_amber = Dry_AmberParams(polymer, chain_length, num_chains, bulk_creator = dry_bulk_creator)
dry_amber.create_amber_params()
with open(output, 'a') as f:
	print(f"\tAmber parameters are created for the bulk phase", file=f)

# Create a working directory for dry equilibration simulations and copy necessary files	
os.chdir(dry_eq_dir)

for params in [
	f"{system_tag}.prmtop",
	f"{system_tag}.inpcrd",
]:
	shutil.copy(os.path.join(init_dir, params), dry_eq_dir)

for inputs in [
	"dry-eq_0-min.in",
	"dry-eq_1-nvt.in",
	"dry-eq_2-npt.in",
	"dry-eq_3-nvt.in",
	"dry-eq_4-npt.in",
	"dry-eq_5-nvt.in",
	"dry-eq_6-npt.in",
	"dry-eq_7-nvt-pr.in",
]:
	shutil.copy(os.path.join(input_dir, inputs), dry_eq_dir)

# Run the dry equilibration MD simulations sequence using Amber software (pmemd.MPI & pmemd.cuda)
with open(output, 'a') as f:
	print(f"\tDry equilibration MD simulations started", file=f)
dry_md = Dry_MDSimulation(nproc, output, amber_params=dry_amber, use_gpu=use_gpu)
dry_md.run_all_steps()
with open(output, 'a') as f:
	print(f"\tDry equilibration MD simulations finished", file=f)

# Perform trajectory files merging and conversion
nc_files = [f"dry-eq_1-nvt.nc",
	f"dry-eq_2-npt.nc",
	f"dry-eq_3-nvt.nc", 
	f"dry-eq_4-npt.nc",
	f"dry-eq_5-nvt.nc",
	f"dry-eq_6-npt.nc",
	f"dry-eq_7-nvt-pr.nc",
	]
ncrst_file = f"dry-eq_7-nvt-pr.ncrst"
prmtop_file = f"{system_tag}.prmtop"
pdb_file = f"{system_tag}_dry-eq_last.pdb"
merged_pdb = f"{system_tag}_dry-eq.pdb"
cpptraj_file="cpptraj.in"
prefix=f"dry-eq_tmp"
dry_md_analysis = Analysis()
dry_md_analysis.merge_nc_files(prmtop_file, ncrst_file, pdb_file, nc_files, prefix, merged_pdb, cpptraj_file)

with open(output, 'a') as f:
	print(f"\tDry equilibration trajectory merging finished", file=f)

## Copy necessary files for hydrate equilibration
#os.chdir(hyd_eq_dir)
#
#shutil.copy(os.path.join(dry_eq_dir, pdb_file), hyd_eq_dir)
#shutil.copy(os.path.join(input_dir, "h2o.pdb"), hyd_eq_dir)
#
#for params in [
#	"h.prepi",
#	"t.prepi",
#	f"{polymer}_m.prepi",
#	f"{polymer}_gaff2.frcmod",
#]:
#	shutil.copy(os.path.join(init_dir, params), hyd_eq_dir)
#for inputs in [
#	"hyd-eq_0-min.in",
#	"hyd-eq_1-nvt.in",
#	"hyd-eq_2-nvt.in",
#	"hyd-eq_3-nvt.in",
#	"hyd-eq_4-npt.in",
#	"hyd-eq_5-nvt-pr.in",
#]:
#	shutil.copy(os.path.join(input_dir, inputs), hyd_eq_dir)
#
## Run the hydrated equilibration MD simulations sequence using Amber software (pmemd.MPI & pmemd.cuda) for different hydration levels (lambda)
#lam_list = [4, 8, 12]
#base_hyd_pdb = pdb_file
#polymer_file = read(base_hyd_pdb)
#cell = polymer_file.cell
#a, b, c = cell.lengths()
#num_S = sum(1 for atom in polymer_file if atom.symbol == "S")
#
#for i, lam in enumerate(lam_list):
#	if i > 0:
#		base_hyd_pdb = f"{polymer}_n-{chain_length}x{num_chains}_{lam_list[i-1]}-h2o_hyd-eq_last.pdb"
#		polymer_file = read(base_hyd_pdb)
#		cell = polymer_file.cell
#		a, b, c = cell.lengths()
#	
#	num_h2o = lam * num_S
#	
#	lam_dir = os.path.join(hyd_eq_dir, f"{lam}_h2o")
#	lam_init_dir = os.path.join(lam_dir, "init")
#	lam_md_dir = os.path.join(lam_dir, "md")
#	
#	os.makedirs(lam_dir, exist_ok=True)
#	os.makedirs(lam_init_dir, exist_ok=True)
#	os.makedirs(lam_md_dir, exist_ok=True)
#	
#	for fname in [
#		"h.prepi",
#		"t.prepi",
#		f"{polymer}_m.prepi",
#		f"{polymer}_gaff2.frcmod",
#		"h2o.pdb",
#		base_hyd_pdb,
#	]:
#		shutil.copy(os.path.join(hyd_eq_dir, fname), lam_dir)
#	
#	for fname in [
#		"hyd-eq_0-min.in",
#		"hyd-eq_1-nvt.in",
#		"hyd-eq_2-nvt.in",
#		"hyd-eq_3-nvt.in",
#		"hyd-eq_4-npt.in",
#		"hyd-eq_5-nvt-pr.in",
#	]:
#		shutil.copy(os.path.join(hyd_eq_dir, fname), lam_md_dir)
#
#	os.chdir(lam_dir) 
#
#	# Create the bulk phase for specific hydration level (lambda)
#	hyd_bulk_creator = Hyd_BulkCreator(polymer, chain_length, num_chains, a, b, c, num_h2o, lam, base_hyd_pdb)
#	hyd_bulk_creator.create_bulk_phase()
#	with open(output, 'a') as f:
#		print(f"\tHydration lambda={lam} bulk phase creation finished", file=f)
#
#	# Create amber parameters for specific hydration level (lambda)
#	hyd_amber = Hyd_AmberParams(polymer, chain_length, num_chains, bulk_creator = hyd_bulk_creator)
#	hyd_amber.create_amber_params()
#	with open(output, 'a') as f:
#		print(f"\tAmber parameters are created for the hydration lambda={lam} bulk phase", file=f)
#
#	for fname in [
#		f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h2o.prmtop",
#		f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h2o.inpcrd",
#		f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h2o_amber.pdb",	   
#	]:
#		shutil.move(fname, os.path.join(lam_md_dir, os.path.basename(fname)))
#	
#	for fname in os.listdir("."):
#		if os.path.isfile(fname) and not fname.endswith((".prmtop", ".inpcrd", ".pdb")):
#			shutil.move(fname, os.path.join(lam_init_dir, fname))
#	
#	os.chdir(lam_md_dir)
#	
#	# Run the dry equilibration MD simulations sequence using Amber software (pmemd.MPI & pmemd.cuda)
#	with open(output, 'a') as f:
#		print(f"\tHydration lambda={lam} equilibration MD simulations started", file=f)
#	hyd_md = Hyd_MDSimulation(nproc, output, amber_params=hyd_amber, use_gpu=use_gpu)
#	hyd_md.run_all_steps()
#	with open(output, 'a') as f:
#		print(f"\tHydration lambda={lam} equilibration MD simulations finished", file=f)
#	
#	 # Perform trajectory files merging and conversion
#	nc_files = [f"hyd-eq_1-nvt.nc",
#		f"hyd-eq_2-nvt.nc",
#		f"hyd-eq_3-nvt.nc", 
#		f"hyd-eq_4-npt.nc",
#		f"hyd-eq_5-nvt-pr.nc",
#	]
#	prmtop_file = f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h2o.prmtop"
#	ncrst_file = f"hyd-eq_5-nvt-pr.ncrst"
#	pdb_file = f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h2o_hyd-eq_last.pdb"
#	merged_pdb=f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h2o_hyd-eq.pdb"
#	cpptraj_file="cpptraj.in"
#	prefix=f"hdy-eq_tmp"
#	hyd_md_analysis = Analysis()
#	hyd_md_analysis.merge_nc_files(prmtop_file, ncrst_file, pdb_file, nc_files, prefix, merged_pdb, cpptraj_file)
#
#	shutil.copy(pdb_file, hyd_eq_dir)
#	os.chdir(hyd_eq_dir)
#
## Copy necessary files for proton conductivity calculations
#os.chdir(cond_dir)
#
#for h3o_files in [
#	"h3o.pdb",
#	"h3o.frcmod",
#	"h3o.prepi", 
#]:
#	shutil.copy(os.path.join(input_dir, h3o_files), cond_dir)
#
#for fname in [
#	"cond_0-min.in",
#	"cond_pr-nvt.in",
#]:
#	shutil.copy(os.path.join(input_dir, fname), cond_dir)
#
#for params in [
#	"h_so3.prepi",
#	"t_so3.prepi",
#	f"{polymer}_m_so3.prepi",
#	f"{polymer}_so3_gaff2.frcmod",
#]:
#	shutil.copy(os.path.join(init_dir, params), cond_dir)
#
## Run the proton conduction MD simulations sequence using Amber software (pmemd.MPI & pmemd.cuda) for different hydration levels (lambda)
#
#for lam in lam_list:	 
#	lam_cond_dir = os.path.join(cond_dir, f"{lam}_h3o-h2o")
#	lam_cond_init_dir = os.path.join(lam_cond_dir, "init")
#	lam_cond_md_dir = os.path.join(lam_cond_dir, "md")
#	
#	os.makedirs(lam_cond_dir, exist_ok=True)
#	os.makedirs(lam_cond_init_dir, exist_ok=True)
#	os.makedirs(lam_cond_md_dir, exist_ok=True)
#	
#	hyd_pdb = f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h2o_hyd-eq_last.pdb"
#	shutil.copy(os.path.join(hyd_eq_dir, hyd_pdb), lam_cond_dir)
#	
#	for fname in [
#		"h_so3.prepi",
#		"t_so3.prepi",
#		f"{polymer}_m_so3.prepi",
#		f"{polymer}_so3_gaff2.frcmod",
#		"h3o.pdb",
#		"h3o.frcmod",
#		"h3o.prepi", 
#		"cond_0-min.in",
#		"cond_pr-nvt.in"
#	]:
#		shutil.copy(os.path.join(cond_dir, fname), lam_cond_dir)
#	
#	os.chdir(lam_cond_dir)
#	hyd_pdb_file = read(hyd_pdb)
#	cell = hyd_pdb_file.cell
#	a, b, c = cell.lengths()
#	hyd_clean_pdb = f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h2o_hyd-eq_clean.pdb"
#	
#	cleaner = PDBCleaner(hyd_pdb, hyd_clean_pdb, chain_length)
#	cleaner.remove_so3h_hydrogens()
#	num_h3o = cleaner.hydrogens_removed_count
#	
#	with open(output, 'a') as f:
#		print(f"\tConductivity lambda={lam} removed SO3H hydrogens = {num_h3o} and H2O molecule = {num_h3o}", file=f)
#	
#	cond_bulk_creator = Cond_BulkCreator(polymer, chain_length, num_chains, a, b, c, num_h3o, lam, hyd_clean_pdb)
#	cond_bulk_creator.create_bulk_phase()
#	
#	with open(output, 'a') as f:
#		print(f"\tConductivity lambda={lam} bulk phase creation finished", file=f)
#	
#	cond_amber = Cond_AmberParams(polymer, chain_length, num_chains, bulk_creator=cond_bulk_creator)
#	cond_amber.create_amber_params()
#	
#	with open(output, 'a') as f:
#		print(f"\tAmber parameters are created for the conductivity lambda={lam} bulk phase", file=f)	 
#	
#	for fname in [
#		f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h3o-h2o.prmtop",
#		f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h3o-h2o.inpcrd",
#		f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h3o-h2o_amber.pdb",
#		"cond_0-min.in",
#		"cond_pr-nvt.in",
#	]:
#		shutil.move(fname, os.path.join(lam_cond_md_dir, os.path.basename(fname)))
#	
#	for fname in os.listdir("."):
#		if os.path.isfile(fname) and fname not in[
#			f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h3o-h2o.prmtop",
#			f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h3o-h2o.inpcrd",
#			f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h3o-h2o_amber.pdb",		   
#		]:
#			shutil.move(fname, os.path.join(lam_cond_init_dir, fname))
#	
#	os.chdir(lam_cond_md_dir)
#	
#	with open(output, 'a') as f:
#		print(f"\tConductivity lambda={lam} MD simulations started", file=f)
#		
#	cond_md = Cond_MDSimulation(nproc, output, amber_params=cond_amber, use_gpu=use_gpu)
#	cond_md.run_all_steps()
#	
#	with open(output, 'a') as f:
#		print(f"\tConductivity lambda={lam} MD simulations finished", file=f)
#	
#	os.chdir(base_dir)
