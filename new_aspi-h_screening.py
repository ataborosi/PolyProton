import numpy as np
import pandas as pd
import math
import os
import shutil
import glob
import subprocess
import getpass

import ase
from ase.io import read, write
from ase.optimize import LBFGS, FIRE
from ase.constraints import FixAtoms

from rdkit import Chem
from rdkit.Chem import AllChem, rdGeometry, Draw

from Bio.PDB import PDBParser, PDBIO, Atom, NeighborSearch

from scipy.spatial.transform import Rotation as R

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

class Mol2Modification:
	def __init__(self, input_mol2, output_mol2=None, remove_atom_types=None):
		self.input_mol2 = input_mol2
		self.output_mol2 = output_mol2
		self.remove_atom_types = remove_atom_types
	
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

	def modify_mol2_file_temp(self, input_mol2, output_mol2, remove_atom_types):
		modifier = Mol2Modification(input_mol2=input_mol2, output_mol2=output_mol2, remove_atom_types=remove_atom_types)
		modifier.remove_atoms_and_bonds()

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
			f.write(f'saveamberparm mol {self.polymer}_n-{self.chain_length}.prmtop {self.polymer}_n-{self.chain_length}.inpcrd\n')
			f.write('quit\n')
		subprocess.run(f'tleap -f {leap_input_filename} > {leap_output_filename}', shell=True, check=True)
	
	def parameterization(self):
		read_xyz_1 = read(f'{self.polymer}_0_opt.xyz')
		xyz_file_1 = f'{self.polymer}_0_opt.xyz'
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
		xyz_file_2 = f'{self.polymer}_0_opt_so3.xyz'
		charge_2 = -2
		mult_2 = 1
		self.remove_atoms(xyz_file_1, xyz_file_2)
		read_xyz_2 = read(f'{self.polymer}_0_opt_so3.xyz')
		file_name_2 = f"{self.polymer}_so3"
		mod_file_name_2 = f"{self.polymer}_so3_mod"
		self.orca_calculation(read_xyz_2, charge_2, mult_2, file_name_2)
		self.run_multiwfn(file_name_2)
		self.convert_xyz_to_mol2(xyz_file_2, file_name_2)
		self.modify_mol2_file(file_name_2, mod_file_name_2)
		self.run_antechamber_v2()

class Dry_BulkCreator:
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

class Dry_AmberParams:
	def __init__(self, polymer, chain_length, num_chains, bulk_creator: Dry_BulkCreator):
		self.polymer = polymer
		self.chain_length = chain_length
		self.num_chains = num_chains
		self.box_size_x = bulk_creator.box_size_x
		self.box_size_y = bulk_creator.box_size_y
		self.box_size_z = bulk_creator.box_size_z
		self.pdb_file = f"{self.polymer}_n-{self.chain_length}x{self.num_chains}.pdb"
		
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
			saveamberparm mol {self.polymer}_n-{self.chain_length}x{self.num_chains}.prmtop {self.polymer}_n-{self.chain_length}x{self.num_chains}.inpcrd
			quit
			""")
	
		self.prmtop = f"{self.polymer}_n-{self.chain_length}x{self.num_chains}.prmtop"
		self.inpcrd = f"{self.polymer}_n-{self.chain_length}x{self.num_chains}.inpcrd"
		amber_pdb = f"{self.polymer}_n-{self.chain_length}x{self.num_chains}_amber.pdb"
	
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

class Hyd_BulkCreator:
	def __init__(self, polymer, chain_length, num_chains, a, b, c, num_h2o, lam, pdb_file):
		self.polymer = polymer
		self.chain_length = chain_length
		self.num_chains = num_chains
		self.box_a = a
		self.box_b = b
		self.box_c = c
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
			pbc {self.box_a} {self.box_b} {self.box_c}
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
		self.box_size_x = bulk_creator.box_a
		self.box_size_y = bulk_creator.box_b
		self.box_size_z = bulk_creator.box_c
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
			print(f"\t\tStarted hydration level lambda = {self.lam} {step_name} step", file=f)
			
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
			print(f"\t\tFinished hydration level lambda = {self.lam} {step_name} step", file=f)
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

class PDBCleaner:
    def __init__(self, input_pdb_file, output_pdb_file, chain_length):
        self.input_pdb_file = input_pdb_file
        self.output_pdb_file = output_pdb_file
        self.chain_length = chain_length
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure('polymer', input_pdb_file)
        self.hydrogens_removed_count = 0
    
    def remove_so3h_hydroges(self):
        atoms = list(self.structure.get_atoms())
        ns = NeighborSearch(atoms)
        
        hydrogens_to_remove = set()
        
        for residue in self.structure.get_residues():
            s_atoms = [atom for atom in residue if atom.element == 'S']
            for s_atom in s_atoms:
                o_atoms = [atom for atom in ns.search(s_atom.coord, 1.8) if atom.element == 'O']
                for o_atom in o_atoms:
                    h_atoms = [atom for atom in ns.search(o_atom.coord, 1.2) if atom.element == 'H']
                    hydrogens_to_remove.update(h_atoms)
        
        for h_atom in hydrogens_to_remove:
            residue = h_atom.get_parent()
            residue.detach_child(h_atom.id)
            
        self.hydrogens_removed_count = len(hydrogens_to_remove)
        
        wat_removed = 0
        for model in self.structure:
            for chain in model:
                for residue in list(chain):
                    if residue.get_resname().strip() in ['HOH', 'WAT']:
                        if wat_removed < self.hydrogens_removed_count:
                            chain.detach_child(resideu.id)
                            wat_removed += 1
                        else:
                            break
                if wat_removed >= self.hydrogens_removed_count:
                    break
            if wat_removed >= self.hydrogens_removed_count:
                break
                    
        self.save_structure()

    def save_structure(self):
        temp_output = "temp_cleaned.pdb"
        io = PDBIO()
        io.set_structure(self.structure)
        io.save(temp_output)
        
        with open(self.input_pdb_file, 'r') as original:
            original_lines = original.readlines()
            
        cryst_line = next((line for line in original_liens if line.startswith("CRYST1")), "")
        box_line = next((line for line in reversed(original_lines) if len(line.strip().split()) == 6), "")
        
        with open(temp_output, 'r') as cleaned:
            cleaned_lines = [line for line in cleaned if line.startswith("ATOM") or line.startswith("HETATM")]
        
        final_lines = []
        if cryst_line:
            final_lines.append(cryst_line)
        
        current_res_id = None
        current_res_name = None
        res_count = 0
        
        for line in cleaned_lines:
            res_name = line[17:20].strip()
            res_id = int(line[22:26])
            
            if res_id != current_res_id:
                res_count += 1
                if res_name not in ["WAT", "HOH"] and (res_count - 1) % self.chain_length == 0 and res_count != 1:
                    final_lines.append("TER\n")
                if res_name in ["WAT", "HOH"] and current_res_name not in ["WAT", "HOH", None]:
                    final_lines.append("TER\n")
                    
            final_lines.append(line)
            current_res_id = res_id
            current_res_name = res_name
            
        final_lines.append("TER\nEND\n")
        if box_line:
            final_lines.append(box_line + '\n')
            
        with open(self.output_pdb_file, 'w') as out:
            out.writelines(final_lines)
        
        os.remove(temp_output)

class Cond_BulkCreator:
    def __init__(self, polymer, chain_length, num_chains, a, b, c, num_h3o, lam, pdb_file):
        self.polymer = polymer
        self.chain_length = chain_length
        self.num_chains = num_chains
        self.box_a = a
        self.box_b = b
        self.box_c = c
        self.num_h3o = num_h3o
        self.lam = lam
        self.pdb_file = pdb_file
        
    def packmol_generate_box(self):
	
# Create a working directory for polymer with subfolders and copy polymer connectivity cards			
polymer = 'a1'
base_dir = os.getcwd()
polymer_dir = os.path.join(base_dir, polymer)
init_dir = os.path.join(polymer_dir, "init")
dry_eq_dir = os.path.join(polymer_dir, "dry-eq")
hyd_eq_dir = os.path.join(polymer_dir, "hyd-eq")
input_dir = os.path.join(base_dir, "input_files")

os.makedirs(polymer_dir, exist_ok=True)
os.makedirs(init_dir, exist_ok=True)
os.makedirs(dry_eq_dir, exist_ok=True)
os.makedirs(hyd_eq_dir, exist_ok=True)

for cards in ["head", "main", "tail", "head_so3", "main_so3", "tail_so3"]:
	shutil.copy(os.path.join(input_dir, cards), init_dir)

os.chdir(init_dir)

# Details of polymer
backbone_smiles = 'C1=CC2=C3C(=CC=C4C3=C1C(=O)OC4=O)C(=O)OC2=O'
sidechain_smiles="OCCCS(O)(=O)=O"
benzene_smiles="C1=CC=CC=C1"
conf_num = 2
conf_output_file = f"{polymer}_conf.txt"
temperature = 300
chain_length = 15
num_chains = 20
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

# Performing the evaluation of conformers based on their total energies and Boltzmann distribution
analyzer = ConformationAnalyzer(polymer, conf_num, conf_output_file, temperature)
analyzer.analyze_conformers()	
with open(output, 'a') as f:
	print(f"\tEvaluation of conformers finished", file=f)

# Creating GAFF2 parameters for the monomer unit and creating specfic chain length single polymer 
param = GAFF2Param(polymer, chain_length, nproc)
param.parameterization()
with open(output, 'a') as f:
	print(f"\tGAFF2 parameters and single polymer chain creation finished", file=f)

# Align the single polymer chain, define the box dimension based on the polymer chain, and create the bulk phase
dry_bulk_creator = Dry_BulkCreator(polymer, chain_length, num_chains)
dry_bulk_creator.create_bulk_phase()
with open(output, 'a') as f:
	print(f"\tAlignment of single polymer chain and bulk phase creation finished", file=f)
	
# Create amber parameters
dry_amber = Dry_AmberParams(polymer, chain_length, num_chains, bulk_creator = dry_bulk_creator)
dry_amber.create_amber_params()
with open(output, 'a') as f:
	print(f"\tAmber parameters are created for the bulk phase", file=f)

# Create a working directory for dry equilibration simulations and copy necessary files	
os.chdir(dry_eq_dir)

for params in [
	f"{polymer}_n-{chain_length}x{num_chains}.prmtop",
	f"{polymer}_n-{chain_length}x{num_chains}.inpcrd",
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
prmtop_file = f"{polymer}_n-{chain_length}x{num_chains}.prmtop"
ncrst_file = f"dry-eq_7-nvt-pr.ncrst"
pdb_file = f"dry-eq_last.pdb"
merged_pdb=f"dry-eq.pdb"
cpptraj_file="cpptraj.in"
prefix=f"dry-eq_tmp"
dry_md_analysis = Analysis()
dry_md_analysis.merge_nc_files(prmtop_file, ncrst_file, pdb_file, nc_files, prefix, merged_pdb, cpptraj_file)

with open(output, 'a') as f:
	print(f"\tDry equilibration trajectory merging finished", file=f)

# Create working directory for hydrate equilibration and copy necessary files
os.chdir(hyd_eq_dir)

shutil.copy(os.path.join(dry_eq_dir, pdb_file), hyd_eq_dir)
shutil.copy(os.path.join(input_dir, "h2o.pdb"), hyd_eq_dir)

for params in [
	"h.prepi",
	"t.prepi",
	f"{polymer}_m.prepi",
	f"{polymer}_gaff2.frcmod",
]:
	shutil.copy(os.path.join(init_dir, params), hyd_eq_dir)
for inputs in [
	"hyd-eq_0-min.in",
	"hyd-eq_1-nvt.in",
	"hyd-eq_2-nvt.in",
	"hyd-eq_3-nvt.in",
	"hyd-eq_4-npt.in",
	"hyd-eq_5-nvt-pr.in",
]:
	shutil.copy(os.path.join(input_dir, inputs), hyd_eq_dir)

# Run the hydrated equilibration MD simulations sequence using Amber software (pmemd.cuda) for different hydration levels (lambda)
lam_list = [4, 8, 12]
base_hyd_pdb = pdb_file
polymer_file = read(base_hyd_pdb)
cell = polymer_file.cell
a, b, c = cell.lengths()
num_S = sum(1 for atom in polymer_file if atom.symbol == "S")

for i, lam in enumerate(lam_list):
	if i > 0:
		base_hyd_pdb = f"{polymer}_n-{chain_length}x{num_chains}_{lam_list[i-1]}-h2o.pdb"
		polymer_file = read(base_hyd_pdb)
		cell = polymer_file.cell
		a, b, c = cell.lengths()
	
	num_h2o = lam * num_S
	
	lam_dir = os.path.join(hyd_eq_dir, f"{lam}_h2o")
	lam_init_dir = os.path.join(lam_dir, "init")
	lam_md_dir = os.path.join(lam_dir, "md")
	
	os.makedirs(lam_dir, exist_ok=True)
	os.makedirs(lam_init_dir, exist_ok=True)
	os.makedirs(lam_md_dir, exist_ok=True)
	
	for fname in [
		"h.prepi",
		"t.prepi",
		f"{polymer}_m.prepi",
		f"{polymer}_gaff2.frcmod",
		"h2o.pdb",
		base_hyd_pdb,
	]:
		shutil.copy(os.path.join(hyd_eq_dir, fname), lam_dir)
	
	for fname in [
		"hyd-eq_0-min.in",
		"hyd-eq_1-nvt.in",
		"hyd-eq_2-nvt.in",
		"hyd-eq_3-nvt.in",
		"hyd-eq_4-npt.in",
		"hyd-eq_5-nvt-pr.in",
	]:
		shutil.copy(os.path.join(hyd_eq_dir, fname), lam_md_dir)

	os.chdir(lam_dir) 

	# Create the bulk phase for specific hydration level (lambda)
	hyd_bulk_creator = Hyd_BulkCreator(polymer, chain_length, num_chains, a, b, c, num_h2o, lam, base_hyd_pdb)
	hyd_bulk_creator.create_bulk_phase()
	with open(output, 'a') as f:
		print(f"\tHydration lambda={lam} bulk phase creation finished", file=f)

	# Create amber parameters for specific hydration level (lambda)
	hyd_amber = Hyd_AmberParams(polymer, chain_length, num_chains, bulk_creator = hyd_bulk_creator)
	hyd_amber.create_amber_params()
	with open(output, 'a') as f:
		print(f"\tAmber parameters are created for the hydrated bulk phase", file=f)

	for fname in [
		f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h2o.prmtop",
		f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h2o.inpcrd",
		f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h2o_amber.pdb",	   
	]:
		shutil.move(fname, os.path.join(lam_md_dir, os.path.basename(fname)))
	
	for fname in os.listdir("."):
		if os.path.isfile(fname) and not fname.endswith((".prmtop", ".inpcrd", ".pdb")):
			shutil.move(fname, os.path.join(lam_init_dir, fname))
	
	os.chdir(lam_md_dir)
	
	# Run the dry equilibration MD simulations sequence using Amber software (pmemd.MPI & pmemd.cuda)
	with open(output, 'a') as f:
		print(f"\tHydrated equilibration MD simulations started", file=f)
	hyd_md = Hyd_MDSimulation(nproc, output, amber_params=hyd_amber, use_gpu=use_gpu)
	hyd_md.run_all_steps()
	with open(output, 'a') as f:
		print(f"\tHydrated equilibration MD simulations finished", file=f)
	
	hyd_prmtop = f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h2o.prmtop"
	final_hyd_pdb = f"{polymer}_n-{chain_length}x{num_chains}_{lam}-h2o.pdb"
	subprocess.run(f"ambpdb -p {hyd_prmtop} -c hyd-eq_5-nvt-pr.ncrst > {final_hyd_pdb}", shell=True, check=True)

	shutil.copy(final_hyd_pdb, hyd_eq_dir)
	os.chdir(hyd_eq_dir)