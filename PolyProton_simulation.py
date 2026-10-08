import numpy as np
import pandas as pd
import math
import os
import shutil
import glob
import subprocess
import tempfile
import random
import warnings

import ase
from ase.io import read, write
from ase.optimize import LBFGS, FIRE
from ase.constraints import FixAtoms

from rdkit import Chem
from rdkit.Chem import AllChem, rdGeometry, Draw

from Bio.PDB import PDBParser, PDBIO, Atom, NeighborSearch

from scipy.spatial.transform import Rotation as R

from collections import Counter

warnings.simplefilter("ignore")

# ====
# Calculators / Profiles
# ====

import torch
from mace.calculators import mace_off
from contextlib import redirect_stdout, redirect_stderr
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile

# ====
# User settings
# ====

# Polymer / monomer settings
polymer = 'spi'
polymer_psmiles = ("[*:1]C=6C(=CC(N1C(C=2C=CC=4C(N(C(C3=CC=C(C1=O)C=2C3=4)=O)C=5C=C(C([*:2])=CC=5)S(O)(=O)=O)=O)=O)=CC=6)S(O)(=O)=O")

# Conformer settings
conf_num = 2
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
use_nproc = 16
use_mace_device = "cuda"

# External software / scratch paths
orca_dir = "/opt/orca"
scratch_dir = None

# ====
# Paths / Folders
# ====

base_dir = os.getcwd()
output = os.path.join(base_dir, 'PolyProton_simulation_process.txt')
command_log = os.path.join(base_dir, "PolyProton_simulation_command.txt")

simulation_dir = os.path.join(base_dir, "simulation")
init_dir = os.path.join(simulation_dir, "init")
dry_eq_dir = os.path.join(simulation_dir, "dry_eq")
hyd_eq_dir = os.path.join(simulation_dir, "hyd_eq")
cond_pr_dir = os.path.join(simulation_dir, "cond_pr")
input_dir = os.path.join(base_dir, "input_files")
dry_eq_input_dir = os.path.join(input_dir, "dry_eq", dry_eq_prot)
hyd_eq_input_dir = os.path.join(input_dir, "hyd_eq")
cond_pr_input_dir = os.path.join(input_dir, "cond_pr")

# ====
# Helper functions
# ====

def get_scratch_dir(scratch_dir):
	if scratch_dir is not None:
		resolved_dir = os.path.abspath(
			os.path.expanduser(
				os.path.expandvars(str(scratch_dir))
			)
		)
	elif os.environ.get("SLURM_TMPDIR"):
		resolved_dir = os.environ["SLURM_TMPDIR"]
	elif os.environ.get("TMPDIR"):
		resolved_dir = os.environ["TMPDIR"]
	else:
		resolved_dir = tempfile.gettempdir()

	os.makedirs(resolved_dir, exist_ok=True)

	if not os.path.isdir(resolved_dir):
		raise NotADirectoryError(
			f"Scratch path is not a directory: {resolved_dir}"
		)

	if not os.access(resolved_dir, os.W_OK):
		raise PermissionError(
			f"Scratch directory is not writable: {resolved_dir}"
		)

	return resolved_dir

def create_amber_temp_dir(scratch_dir, step_name):
	return tempfile.mkdtemp(
		prefix=f"PolyProton_{step_name}_",
		dir=scratch_dir
	)

def get_nproc(use_nproc):
	slurm_nproc = os.environ.get("SLURM_NTASKS")

	if slurm_nproc is not None:
		nproc = int(slurm_nproc)
	else:
		nproc = int(use_nproc)

	if nproc < 1:
		raise ValueError("Number of CPU processes must be >= 1")

	return nproc

def get_use_gpu(use_gpu):
	if not os.environ.get("SLURM_JOB_ID"):
		return use_gpu

	gpu_count = os.environ.get("SLURM_GPUS_ON_NODE")
	if gpu_count is not None:
		return int(gpu_count) > 0

	gpu_ids = os.environ.get("SLURM_JOB_GPUS", "").strip()
	return bool(gpu_ids)

def get_mace_calculator(device, nproc):
	device = str(device).lower()

	if device not in ["cpu", "cuda"]:
		raise ValueError(
			"use_mace_device must be either 'cpu' or 'cuda'"
		)

	if device == "cpu":
		torch.set_num_threads(nproc)

	with open(os.devnull, "w") as devnull:
		with redirect_stdout(devnull), redirect_stderr(devnull):
			nnp_calc = mace_off(
				model="small",
				device=device
			)

	return nnp_calc

def get_orca_profile(orca_dir):
	orca_binary = os.path.join(orca_dir, "orca")

	if not os.path.isfile(orca_binary):
		raise FileNotFoundError(
			f"ORCA executable not found: {orca_binary}"
		)

	return OrcaProfile(command=orca_binary)
	
def log_message(message):
	with open(output, 'a') as f:
		print(message, file=f)

def run_command(command):
#	log_message(f"Running command: {command}")

	with open(command_log, "a", encoding="utf-8") as log:
		log.write(f"\n\n$ {command}\n")
		log.flush()

		try:
			subprocess.run(command, shell=True, check=True, stdout=log, stderr=subprocess.STDOUT)
		except subprocess.CalledProcessError as exc:
			error_message = (
				f"External command failed with return code "
				f"{exc.returncode}: {command}"
			)
			log.write(f"\nERROR: {error_message}\n")
			log_message(error_message)
			raise RuntimeError(error_message) from exc

def ensure_directories():
	os.makedirs(simulation_dir, exist_ok=True)
	os.makedirs(init_dir, exist_ok=True)
	os.makedirs(dry_eq_dir, exist_ok=True)
	os.makedirs(hyd_eq_dir, exist_ok=True)
	os.makedirs(cond_pr_dir, exist_ok=True)

def select_conformer(conf_file, conf_sel="LE", far_fraction=0.5):
	df = pd.read_csv(conf_file, sep=r'\s+')
	LE_conf = int(df.iloc[0]["conf_i"])

	if conf_sel == "LE":
		return LE_conf

	if conf_sel == "HE":
		if len(df) == 1:
			return LE_conf

		others = df.iloc[1:].copy()

		if "ΔE" in others.columns and len(others) > 1:
			others = others.sort_values(by="ΔE", ascending=False).reset_index(drop=True)
			n_far = max(1, int(math.ceil(len(others) * far_fraction)))
			far_pool = others.iloc[:n_far]
		else:
			far_pool = others

		return int(random.choice(far_pool["conf_i"].tolist()))

	raise ValueError("conf_sel must be either 'LE' or 'HE'")

def get_generated_conf_indices(polymer):
	xyz_files = sorted(glob.glob(f"{polymer}_*.xyz"))
	conf_indices = []

	for xyz_file in xyz_files:
		base = os.path.splitext(os.path.basename(xyz_file))[0]

		if "_opt" in base:
			continue

		parts = base.split("_")
		if len(parts) < 2:
			continue

		try:
			conf_i = int(parts[-1])
			conf_indices.append(conf_i)
		except ValueError:
			continue

	return sorted(set(conf_indices))

def generate_chain_lengths(chain_length, num_chains, mix_chains=False, seed=42, mix_fraction=0.20):
	if not mix_chains:
		return [int(chain_length)] * int(num_chains)

	if not 0.0 < mix_fraction < 1.0:
		raise ValueError("mix_fraction must be between 0 and 1")

	max_deviation = max(1, int(round(chain_length * mix_fraction)))
	max_deviation = min(max_deviation, chain_length - 2)
	rng = random.Random(seed)
	deviations = []
	
	for _ in range(num_chains // 2):
		d = rng.randint(0, max_deviation)
		deviations.extend([-d, d])	  

	if num_chains % 2 == 1:
		deviations.append(0)

	rng.shuffle(deviations)

	chain_lengths = [chain_length + deviation for deviation in deviations]

	return chain_lengths

def system_name(polymer, chain_length, num_chains, mix_chains=False):
	if mix_chains:
		return f"{polymer}_mix-n-{chain_length}x{num_chains}"
	return f"{polymer}_n-{chain_length}x{num_chains}"

def hyd_system_name(polymer, chain_length, num_chains, lam, mix_chains=False):
	return f"{system_name(polymer, chain_length, num_chains, mix_chains)}_{lam}-h2o"

def cond_system_name(polymer, chain_length, num_chains, lam, mix_chains=False):
	return f"{system_name(polymer, chain_length, num_chains, mix_chains)}_{lam}-h3o-h2o"

def format_leap_sequence(sequence, residues_per_line=20):
	lines = []
	lines.append("{")

	for i in range(0, len(sequence), residues_per_line):
		chunk = sequence[i:i + residues_per_line]
		lines.append("	  " + " ".join(chunk))

	lines.append("}")
	return "\n".join(lines)

def ordered_chain_lengths_for_packmol(chain_length, num_chains, mix_chains=False, chain_lengths=None):
	if not mix_chains:
		return [chain_length] * num_chains

	if chain_lengths is None:
		raise ValueError("chain_lengths must be provided when mix_chains=True")

	length_counts = Counter(chain_lengths)
	ordered_lengths = []

	for L in sorted(length_counts):
		ordered_lengths.extend([L] * length_counts[L])

	return ordered_lengths

def validate_settings():
	if run_hyd and not run_dry:
		raise ValueError("run_hyd=True requires run_dry=True in the same workflow execution.")

	if conf_selection not in ["LE", "HE"]:
		raise ValueError("conf_selection must be 'LE' or 'HE'")

	if chain_length < 2:
		raise ValueError("chain_length must be >= 2")

	if num_chains < 1:
		raise ValueError("num_chains must be >= 1")
		
	if dry_eq_prot not in ["6-step", "12-step"]:
		raise ValueError("dry_eq_prot must be '6-step' or '12-step'")

	for lam in cond_lam_list:
		if lam not in lam_list:
			raise ValueError(f"cond lambda {lam} not found in lam_list")

	if int(use_nproc) < 1:
		raise ValueError("use_nproc must be >= 1")

	if str(use_mace_device).lower() not in ["cpu", "cuda"]:
		raise ValueError(
			"use_mace_device must be either 'cpu' or 'cuda'"
		)

	if not isinstance(polymer_psmiles, str) or not polymer_psmiles.strip():
		raise ValueError("polymer_psmiles must be a non-empty string")
	
	if not isinstance(polymer, str):
		raise ValueError("polymer must be a string")
	
	if len(polymer) != 3 or not polymer.isalnum():
		raise ValueError("polymer must contain exactly 3 characters, because used for PDB resideu name")
	
def write_anchor_indices(polymer, anchor_indices):
	with open(f"{polymer}_anchors.txt", "w") as f:
		for idx in anchor_indices:
			f.write(f"{idx}\n")

def read_anchor_indices(polymer):
	anchor_file = f"{polymer}_anchors.txt"
	if not os.path.exists(anchor_file):
		raise FileNotFoundError(f"Anchor index file not found: {anchor_file}")

	with open(anchor_file, "r") as f:
		indices = [int(line.strip()) for line in f if line.strip()]

	if len(indices) != 2:
		raise ValueError(f"Expected 2 anchor indices in {anchor_file}, got {indices}")

	return indices			  

def log_settings(chain_lengths, system_tag, nproc, resolved_scratch_dir, num_sulfonic_acid_groups, protonated_charge, deprotonated_charge):
	log_message(f"Processing polymer: {polymer} with polymer_psmiles: {polymer_psmiles}")
	log_message(f"\tnum_sulfonic_acid_groups = {num_sulfonic_acid_groups}")
	log_message(f"\tprotonated_charge = {protonated_charge}")
	log_message(f"\tdeprotonated_charge = {deprotonated_charge}")
	log_message(f"\tconf_selection = {conf_selection}")
	log_message(f"\tconf_num = {conf_num}")
	log_message(f"\tconf_prune_rms_thresh = {conf_prune_rms_thresh}")
	log_message(f"\tchain_length = {chain_length}")
	log_message(f"\tnum_chains = {num_chains}")
	log_message(f"\tmix_chains = {mix_chains}")
	log_message(f"\tchain_lengths = {chain_lengths}")
	log_message(f"\tmix_chain_fraction = {mix_chain_fraction}")
	log_message(f"\tminimum chain length = {min(chain_lengths)}")
	log_message(f"\tmaximum chain length = {max(chain_lengths)}")
	log_message(f"\tmean chain length = {np.mean(chain_lengths):.3f}")
	log_message(f"\ttotal repeat units = {sum(chain_lengths)}")
	log_message(f"\tpack_z_padding = {pack_z_padding}")
	log_message(f"\taligned = {aligned}")
	log_message(f"\tdry_eq_prot = {dry_eq_prot}")
	log_message(f"\tlam_list = {lam_list}")
	log_message(f"\tcond_lam_list = {cond_lam_list}")
	log_message(f"\tsystem_tag = {system_tag}")
	log_message(f"\tuse_nproc = {use_nproc}")
	log_message(f"\tnproc = {nproc}")
	log_message(f"\tuse_gpu = {use_gpu}")
	log_message(f"\tuse_mace_device = {use_mace_device}")
	log_message(f"\torca_dir = {orca_dir}")
	log_message(f"\tscratch_dir = {resolved_scratch_dir}")

def get_pdb_box_lengths(pdb_file):
	with open(pdb_file, "r") as f:
		for line in f:
			if line.startswith("CRYST1"):
				a = float(line[6:15])
				b = float(line[15:24])
				c = float(line[24:33])
				return a, b, c

	raise RuntimeError(f"CRYST1 record not found in {pdb_file}")

def count_pdb_element(pdb_file, element):
	count = 0

	with open(pdb_file, "r") as f:
		for line in f:
			if not line.startswith(("ATOM", "HETATM")):
				continue

			elem = line[76:78].strip()

			if elem == element:
				count += 1

	return count

def standardize_pdb_numbering(pdb_file):
	PDB_CHAIN_IDS = (
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"abcdefghijklmnopqrstuvwxyz"
		"0123456789"
	)

	tmp_file = f"{pdb_file}.standardize_tmp"

	atom_counter = 0
	residue_counter = 0
	last_residue_key = None

	with open(pdb_file, "r") as f_in, open(tmp_file, "w") as f_out:

		for line in f_in:

			# Reset numbering for a new MODEL
			if line.startswith("MODEL"):
				atom_counter = 0
				residue_counter = 0
				last_residue_key = None

				f_out.write(line)
				continue

			if line.startswith(("ATOM", "HETATM")):

				# Original information is only used to detect
				# when a new residue begins.
				residue_key = (
					line[21:22],   # original chain ID
					line[22:26],   # original residue number
					line[26:27],   # insertion code
					line[17:20],   # residue name
				)

				if residue_key != last_residue_key:
					residue_counter += 1
					last_residue_key = residue_key

				chain_block = (
					(residue_counter - 1) // 9999
				)

				if chain_block >= len(PDB_CHAIN_IDS):
					raise RuntimeError(
						f"Too many residues for PDB chain numbering: "
						f"{residue_counter}"
					)

				chain_id = PDB_CHAIN_IDS[chain_block]

				residue_number = (
					(residue_counter - 1) % 9999
				) + 1

				atom_counter += 1

				atom_serial = (
					(atom_counter - 1) % 99999
				) + 1

				line = (
					line[:6]
					+ f"{atom_serial:5d}"
					+ line[11:21]
					+ chain_id
					+ f"{residue_number:4d}"
					+ line[26:]
				)

			elif line.startswith("TER"):
				# TER remains the actual molecular-chain boundary.
				last_residue_key = None

			elif line.startswith(("ENDMDL", "END")):
				f_out.write(line)

				# Important for concatenated multi-frame PDB files.
				atom_counter = 0
				residue_counter = 0
				last_residue_key = None
				continue

			f_out.write(line)

	os.replace(tmp_file, pdb_file)

def get_psmiles_chemistry(polymer_psmiles):
	mol = Chem.MolFromSmiles(polymer_psmiles)
	
	so3h_pattern = Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[OX2H1]")
	matches = mol.GetSubstructMatches(so3h_pattern)
	sulfur_indices = {match[0] for match in matches}
	num_sulfonic_acid_groups = len(sulfur_indices)
	protonated_charge = int(Chem.GetFormalCharge(mol))
	deprotonated_charge = (protonated_charge - num_sulfonic_acid_groups)
	
	return (num_sulfonic_acid_groups, protonated_charge, deprotonated_charge)

# ====
# Structure & file modification classes
# ====

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

class PDBCleaner:
	def __init__(self, input_pdb_file, output_pdb_file, chain_length, num_chains, mix_chains=False, chain_lengths=None):
		self.input_pdb_file = input_pdb_file
		self.output_pdb_file = output_pdb_file
		self.chain_length = chain_length
		self.num_chains = num_chains
		self.mix_chains = mix_chains
		self.chain_lengths = chain_lengths
		self.hydrogens_removed_count = 0

		self.ordered_chain_lengths = ordered_chain_lengths_for_packmol(
			chain_length=self.chain_length,
			num_chains=self.num_chains,
			mix_chains=self.mix_chains,
			chain_lengths=self.chain_lengths
		)

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

	def polymer_chain_break_positions(self):
		break_positions = []
		cum = 0
		for L in self.ordered_chain_lengths[:-1]:
			cum += L
			break_positions.append(cum)
		return set(break_positions)

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
						remove_line_indices.add(id(residue[h_i]))

		self.hydrogens_removed_count = len(remove_line_indices)

		waters_to_remove = self.hydrogens_removed_count
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

			if new_residue:
				cleaned_residues.append(new_residue)

		polymer_breaks = self.polymer_chain_break_positions()

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
					if polymer_res_count > 1 and (polymer_res_count - 1) in polymer_breaks:
						out.write("TER\n")

				if resname in ["WAT", "HOH", "H3O"] and previous_resname not in ["WAT", "HOH", "H3O", None]:
					out.write("TER\n")

				for line in residue:
					out.write(line)

				previous_resname = resname

			out.write("TER\nEND\n")
			if box_line:
				out.write(box_line)

		standardize_pdb_numbering(self.output_pdb_file)

# ====
# Momomer & confomer classes
# ====

class MonomerBuilder:
	def __init__(self, polymer, polymer_psmiles, conf_num):
		self.polymer = polymer
		self.polymer_psmiles = polymer_psmiles
		self.final_conf = None
		self.cids = []
		self.conf_num = conf_num
		self.chain_anchor_indices = None

	def create_monomer_unit(self):
		mol = Chem.MolFromSmiles(self.polymer_psmiles)
		if mol is None:
			raise ValueError(f"Could not parse polymer_psmiles: {self.polymer_psmiles}")
		
		star_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
		if len(star_atoms) != 2:
			raise ValueError("polymer_psmiles must contain exactly" 
							"two polymerization endpoints [*]")
		
		map_numbers = [atom.GetAtomMapNum() for atom in star_atoms]
		if set(map_numbers) == {1, 2}:
			star_atoms = sorted(star_atoms, key=lambda atom: atom.GetAtomMapNum())
		elif all(map_num == 0 for map_num in map_numbers):
			star_atoms = sorted(star_atoms, key=lambda atom: atom.GetIdx())
		
		anchor_property = "_PolyProtonAnchorOrder"
		for anchor_order, star_atom in enumerate(star_atoms, start=1):
			neighbors = list(star_atom.GetNeighbors())
			anchor_atom = neighbors[0]
			anchor_atom.SetIntProp(anchor_property, anchor_order)
			
		rw_mol = Chem.RWMol(mol)
		star_indices = sorted([atom.GetIdx() for atom in star_atoms], reverse=True)
		for star_idx in star_indices:
			rw_mol.RemoveAtom(star_idx)
		mol = rw_mol.GetMol()
		Chem.SanitizeMol(mol)
		
		anchor_indices = [None, None]
		for atom in mol.GetAtoms():
			if not atom.HasProp(anchor_property):
				continue
			anchor_order = atom.GetIntProp(anchor_property)
			anchor_indices[anchor_order - 1] = atom.GetIdx()
			atom.ClearProp(anchor_property)
		
		mol = Chem.AddHs(mol)
		
		for anchor_idx in anchor_indices:
			anchor_atom = mol.GetAtomWithIdx(anchor_idx)
			h_neighbors = [nbr for nbr in anchor_atom.GetNeighbors() if nbr.GetAtomicNum() == 1]
	
		self.final_conf = mol
		self.chain_anchor_indices = anchor_indices
		
		write_anchor_indices(self.polymer, self.chain_anchor_indices)
		log_message(f"\tPolymer anchor indices = {self.chain_anchor_indices}")
	
	def create_conformations(self):
		params = Chem.rdDistGeom.srETKDGv3()
		params.pruneRmsThresh = conf_prune_rms_thresh
		params.clearConfs = True
		params.numThreads = 0
		
		self.cids = Chem.rdDistGeom.EmbedMultipleConfs(
			self.final_conf,
			numConfs=self.conf_num,
			params=params
		)
	
		actual_conf_num = len(self.cids)
	
		log_message(f"\tRequested conformers = {self.conf_num}")
		log_message(f"\tGenerated conformers = {actual_conf_num}")
		log_message(f"\tConformer pruning RMS threshold = {conf_prune_rms_thresh}")
	
		if actual_conf_num < self.conf_num:
			log_message(
				f"\tWarning: RDKit generated fewer conformers than requested "
				f"({actual_conf_num} / {self.conf_num})."
			)
	
		atom_map = self.chain_anchor_indices
		Chem.rdMolAlign.AlignMolConformers(self.final_conf, atomIds=atom_map, maxIters=100000)
	
		for i, cid in enumerate(self.cids):
			Chem.MolToXYZFile(self.final_conf, f'{self.polymer}_{i}.xyz', confId=cid)

class ConformationAnalysis:
	def __init__(self, polymer, calculator):
		self.polymer = polymer
		self.calculator = calculator
		self.fix_indices = read_anchor_indices(polymer)
		
	def optimize_confomer(self):
		conf_indices = get_generated_conf_indices(self.polymer)

		if len(conf_indices) == 0:
			raise RuntimeError(f"No generated conformer xyz files found for {self.polymer}")

		log_message(f"\tOptimizing {len(conf_indices)} generated conformers")

		for conf_i in conf_indices:
			xyz_file = f"{self.polymer}_{conf_i}.xyz"
			mol = read(xyz_file)
			mol.set_constraint(FixAtoms(indices=list(set(self.fix_indices))))
			mol.set_calculator(self.calculator)
			
			opt = LBFGS(mol, logfile=f"{self.polymer}_{conf_i}_opt.txt")
			opt.run(fmax=0.01)
			write(f"{self.polymer}_{conf_i}_opt.xyz", mol, format='xyz')

class ConformationAnalyzer:
	def __init__(self, polymer, output_file, temperature):
		self.polymer = polymer
		self.output_file = output_file
		self.temperature = temperature
		
	def analyze_conformers(self):
		opt_files = sorted(glob.glob(f"{self.polymer}_*_opt.txt"))
		dfs = []

		if len(opt_files) == 0:
			raise RuntimeError(f"No optimized conformer log files found for {self.polymer}")

		log_message(f"\tAnalyzing {len(opt_files)} optimized conformers")

		for file_opt in opt_files:
			base = os.path.splitext(os.path.basename(file_opt))[0]
			conf_i = int(base.split("_")[1])

			with open(file_opt) as f:
				last_line = f.readlines()[-1]
				columns = last_line.split()
				E = float(columns[3].rstrip('*'))
				E = round(E, 4)

			df_temp = pd.DataFrame({'conf_i': [conf_i], 'E': [E]})
			dfs.append(df_temp)
		
		df = pd.concat(dfs, ignore_index=True)
		
		min_E = df['E'].min()
		df['ΔE'] = df['E'] - min_E
		q = sum([math.exp(-E / (0.00008617 * self.temperature)) for E in df['ΔE']])
		df['boltzmann'] = [(math.exp(-E / (0.00008617 * self.temperature)) / q) for E in df['ΔE']]
		df = df.sort_values(by='boltzmann', ascending=False)
		df.to_csv(self.output_file, sep=' ', index=False)

# ====
# Parameterization classes
# ====

class GAFF2Param:
	def __init__(self, polymer, chain_length, nproc, orca_profile, orca_dir, protonated_charge, deprotonated_charge, num_sulfonic_acid_groups):
		self.polymer = polymer
		self.chain_length = chain_length
		self.nproc = nproc
		self.orca_profile = orca_profile
		self.orca_dir = orca_dir
		self.protonated_charge = protonated_charge
		self.deprotonated_charge = deprotonated_charge
		self.num_sulfonic_acid_groups = num_sulfonic_acid_groups

	def remove_atoms(self, mol_in, mol_out):
		mol = read(mol_in)
		symbols = mol.get_chemical_symbols()
	
		s_indices = [i for i, s in enumerate(symbols) if s == "S"]
		o_indices = [i for i, s in enumerate(symbols) if s == "O"]
		h_indices = [i for i, s in enumerate(symbols) if s == "H"]
	
		oh_h_indices = []
	
		for o_idx in o_indices:
			# oxygen must be bonded to sulfur
			is_o_connected_to_s = any(
				mol.get_distance(o_idx, s_idx, mic=True) < 1.9
				for s_idx in s_indices
			)
	
			if not is_o_connected_to_s:
				continue
	
			# find hydrogen bonded to this O
			for h_idx in h_indices:
				if mol.get_distance(h_idx, o_idx, mic=True) < 1.2:
					oh_h_indices.append(h_idx)
	
		oh_h_indices = sorted(
			set(oh_h_indices)
		)

		if len(oh_h_indices) != self.num_sulfonic_acid_groups:
			raise RuntimeError(
				f"Expected to remove "
				f"{self.num_sulfonic_acid_groups} "
				f"sulfonic-acid hydrogens, but found "
				f"{len(oh_h_indices)}."
			)

		for i in reversed(oh_h_indices):
			del mol[i]
	
		write(mol_out, mol)
		
	def orca_calculation(self, mol, charge, mult, file_name):
		log_message(f"\tSingle point energy calculation on selected conformer")
		orca_calc = ORCA(
			profile=self.orca_profile,
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

		orca_2mkl = os.path.join(self.orca_dir, "orca_2mkl")
		orca_command = (f'{orca_2mkl} {file_name} -molden')
		run_command(orca_command)

	def run_multiwfn(self, file_name):
		log_message(f"\tRESP charge analysis on selected conformer")
		molden_file = f'{file_name}.molden.input'
		multiwfn_input = f'{molden_file}\n7\n18\n10\n2\n1\ny\n0\n0\nq\n'
		with open("multiwfn_input.txt", "w") as input_file:
			input_file.write(multiwfn_input)
		multiwfn_command = f'Multiwfn < multiwfn_input.txt'
		
		run_command(multiwfn_command)

	def convert_xyz_to_mol2(self, xyz_file, file_name):
		mol2_file = f'{file_name}.mol2'
		obabel_command = f'obabel -i xyz {xyz_file} -O {mol2_file}'
		
		run_command(obabel_command)

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
		molden_df = pd.read_csv(molden_chg_file, sep=r'\s+', header=None)
		atom_df[7] = self.polymer
		atom_df[8] = molden_df[4].map('{:.4f}'.format)
		atom_lines_modified = [' '.join(row) + '\n' for row in atom_df.values.astype(str)]
		modified_mol2_file = f'{mod_file_name}.mol2'
		with open(modified_mol2_file, 'w') as f:
			f.writelines(mol2_lines[:atom_start + 1] + atom_lines_modified + mol2_lines[bond_start:])

	def parse_mol2_atoms_bonds(self, mol2_file):
		with open(mol2_file, "r") as f:
			lines = f.readlines()

		atoms = []
		bonds = []

		in_atom = False
		in_bond = False

		for line in lines:
			if line.startswith("@<TRIPOS>ATOM"):
				in_atom = True
				in_bond = False
				continue
			elif line.startswith("@<TRIPOS>BOND"):
				in_atom = False
				in_bond = True
				continue
			elif line.startswith("@<TRIPOS>"):
				in_atom = False
				in_bond = False
				continue

			if in_atom and line.strip():
				parts = line.split()
				atoms.append({
					"id": int(parts[0]),
					"name": parts[1],
					"type": parts[5],
				})

			if in_bond and line.strip():
				parts = line.split()
				bonds.append((int(parts[1]), int(parts[2])))

		return atoms, bonds

	def get_anchor_atom_names_and_hydrogens(self, mol2_file):
		anchor_indices = read_anchor_indices(self.polymer)	 # 0-based
		atoms, bonds = self.parse_mol2_atoms_bonds(mol2_file)

		id_to_atom = {atom["id"]: atom for atom in atoms}
		neighbors = {}

		for a, b in bonds:
			neighbors.setdefault(a, []).append(b)
			neighbors.setdefault(b, []).append(a)

		anchor_data = []

		for anchor_idx in anchor_indices:
			atom_id = anchor_idx + 1
			atom_name = id_to_atom[atom_id]["name"]

			h_neighbors = []
			for nbr in neighbors.get(atom_id, []):
				nbr_atom = id_to_atom[nbr]
				if nbr_atom["name"].startswith("H"):
					h_neighbors.append(nbr_atom["name"])

			if len(h_neighbors) != 1:
				raise RuntimeError(
					f"Expected exactly one H bonded to anchor atom {atom_name} in {mol2_file}, got {h_neighbors}"
				)

			anchor_data.append((atom_id, atom_name, h_neighbors[0]))

		head_atom_name = anchor_data[0][1]
		head_h_name = anchor_data[0][2]
		tail_atom_name = anchor_data[1][1]
		tail_h_name = anchor_data[1][2]

		return head_atom_name, head_h_name, tail_atom_name, tail_h_name

	def write_connectivity_cards(self, mol2_file, residue_charge, so3=False):
		head_atom_name, head_h_name, tail_atom_name, tail_h_name = self.get_anchor_atom_names_and_hydrogens(mol2_file)

		if so3:
			head_file = "head_so3.card"
			main_file = "main_so3.card"
			tail_file = "tail_so3.card"
		else:
			head_file = "head.card"
			main_file = "main.card"
			tail_file = "tail.card"

		with open(head_file, "w") as f:
			f.write(f"TAIL_NAME {tail_atom_name}\n")
			f.write(f"OMIT_NAME {tail_h_name}\n")
			f.write("POST_TAIL_TYPE c3\n")
			f.write(f"CHARGE {float(residue_charge):.1f}\n")

		with open(main_file, "w") as f:
			f.write(f"HEAD_NAME {head_atom_name}\n")
			f.write(f"TAIL_NAME {tail_atom_name}\n")
			f.write(f"OMIT_NAME {head_h_name}\n")
			f.write(f"OMIT_NAME {tail_h_name}\n")
			f.write("PRE_HEAD_TYPE c3\n")
			f.write("POST_TAIL_TYPE c3\n")
			f.write(f"CHARGE {float(residue_charge):.1f}\n")

		with open(tail_file, "w") as f:
			f.write(f"HEAD_NAME {head_atom_name}\n")
			f.write(f"OMIT_NAME {head_h_name}\n")
			f.write("PRE_HEAD_TYPE c3\n")
			f.write(f"CHARGE {float(residue_charge):.1f}\n")

	def run_antechamber_v1(self):
		antechamber_command_1 = f'antechamber -i {self.polymer}_mod.mol2 -fi mol2 -o {self.polymer}_gaff2.mol2 -fo mol2 -at gaff2'
		run_command(antechamber_command_1)

		antechamber_command_2 = f'antechamber -i {self.polymer}_gaff2.mol2 -fi mol2 -o {self.polymer}.ac -fo ac'
		run_command(antechamber_command_2)

		antechamber_command_3 = f'parmchk2 -i {self.polymer}_gaff2.mol2 -f mol2 -o {self.polymer}_gaff2.frcmod -s 2'
		run_command(antechamber_command_3)

		self.write_connectivity_cards(f"{self.polymer}_gaff2.mol2", residue_charge=self.protonated_charge, so3=False)

		prepgen_command_1 = f'prepgen -i {self.polymer}.ac -o {self.polymer}_m.prepi -f prepi -m main.card -rn {self.polymer}'
		prepgen_command_2 = f'prepgen -i {self.polymer}.ac -o h.prepi -f prepi -m head.card -rn H'
		prepgen_command_3 = f'prepgen -i {self.polymer}.ac -o t.prepi -f prepi -m tail.card -rn T'

		run_command(prepgen_command_1)
		run_command(prepgen_command_2)
		run_command(prepgen_command_3)

	def run_antechamber_v2(self):
		antechamber_command_1 = f'antechamber -i {self.polymer}_so3_mod.mol2 -fi mol2 -o {self.polymer}_so3_gaff2.mol2 -fo mol2 -at gaff2'
		run_command(antechamber_command_1)

		antechamber_command_2 = f'antechamber -i {self.polymer}_so3_gaff2.mol2 -fi mol2 -o {self.polymer}_so3.ac -fo ac'
		run_command(antechamber_command_2)

		antechamber_command_3 = f'parmchk2 -i {self.polymer}_so3_gaff2.mol2 -f mol2 -o {self.polymer}_so3_gaff2.frcmod -s 2'
		run_command(antechamber_command_3)

		self.write_connectivity_cards(f"{self.polymer}_so3_gaff2.mol2", residue_charge=self.deprotonated_charge, so3=True)

		prepgen_command_1 = f'prepgen -i {self.polymer}_so3.ac -o {self.polymer}_m_so3.prepi -f prepi -m main_so3.card -rn {self.polymer}'
		prepgen_command_2 = f'prepgen -i {self.polymer}_so3.ac -o h_so3.prepi -f prepi -m head_so3.card -rn H'
		prepgen_command_3 = f'prepgen -i {self.polymer}_so3.ac -o t_so3.prepi -f prepi -m tail_so3.card -rn T'

		run_command(prepgen_command_1)
		run_command(prepgen_command_2)
		run_command(prepgen_command_3)

	def create_polymer_chain(self, chain_length):
		if chain_length is None:
			chain_length = self.chain_length
		
		sequence = ['H']
		num_middle_units = chain_length - 2
		sequence.extend([f'{self.polymer}'] * num_middle_units)
		sequence.append('T')
	
		formatted_sequence = format_leap_sequence(sequence)
		leap_input_filename = f'leap_input_n-{chain_length}.in'
		leap_output_filename = f'leap_input_n-{chain_length}.out'
		
		with open(leap_input_filename, 'w') as f:
			f.write('source leaprc.gaff2\n')
			f.write(f'loadamberprep h.prepi\n')
			f.write(f'loadamberprep t.prepi\n')
			f.write(f'loadamberprep {self.polymer}_m.prepi\n')
			f.write(f'loadamberparams {self.polymer}_gaff2.frcmod\n')
			f.write(f'mol = sequence {formatted_sequence}\n')
			f.write(f'savepdb mol {self.polymer}_n-{chain_length}.pdb\n')
			f.write(f'saveamberparm mol {self.polymer}_n-{chain_length}.prmtop {self.polymer}_n-{chain_length}.inpcrd\n')
			f.write('quit\n')
		
		run_command(f'tleap -f {leap_input_filename} > {leap_output_filename}')
		
		standardize_pdb_numbering(f"{self.polymer}_n-{chain_length}.pdb")
	
	def parameterization(self, selected_conf):
		read_xyz_1 = read(f'{self.polymer}_{selected_conf}_opt.xyz')
		xyz_file_1 = f'{self.polymer}_{selected_conf}_opt.xyz'
		file_name_1 = f"{self.polymer}"
		mod_file_name_1 = f"{self.polymer}_mod"
		charge_1 = self.protonated_charge
		mult_1 = 1
		self.orca_calculation(read_xyz_1, charge_1, mult_1, file_name_1)
		self.run_multiwfn(file_name_1)
		self.convert_xyz_to_mol2(xyz_file_1, file_name_1)
		self.modify_mol2_file(file_name_1, mod_file_name_1)
		self.run_antechamber_v1()
		self.create_polymer_chain(self.chain_length)
		xyz_file_2 = f'{self.polymer}_{selected_conf}_opt_so3.xyz'
		charge_2 = self.deprotonated_charge
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

# ====
# Dry-stage classes
# ====

class Dry_BulkCreator:
	def __init__(self, polymer, chain_length, num_chains, mix_chains=False, chain_lengths=None, aligned=True, z_padding=15.0):
		self.polymer = polymer
		self.chain_length = chain_length
		self.num_chains = num_chains
		self.mix_chains = mix_chains
		self.chain_lengths = (list(chain_lengths) if chain_lengths is not None else [int(chain_length)] * int(num_chains))
		self.aligned = aligned
		self.z_padding = float(z_padding)
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
		eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
		longest_axis = eigenvectors[:, np.argmax(eigenvalues)]
		z_axis = np.array([0.0, 0.0, 1.0])
		rotation_axis = np.cross(longest_axis, z_axis)
		
		cos_angle = np.clip(np.dot(longest_axis, z_axis) / np.linalg.norm(longest_axis), -1.0, 1.0)

		rotation_angle = np.arccos(cos_angle)
		rotation_angle_deg = np.degrees(rotation_angle)	   

		axis_norm = np.linalg.norm(rotation_axis)
		
		if axis_norm > 1e-12:
			single_chain.rotate(v=rotation_axis / axis_norm, a=rotation_angle_deg, center='COM')
		elif cos_angle < 0:
			single_chain.rotate(v=[1.0, 0.0, 0.0], a=180.0, center='COM')		 
		
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
			box_height = dim_z + 2.0 * self.z_padding
			box_side_area = total_box_volume / box_height

			if self.aligned:
				scale_factor = 2 - 0.3 * np.log2(1 + (dim_x + dim_y) / 10)
				scale_factor = max(0.5, min(2, scale_factor))
			else:
				scale_factor = 4 - 0.3 * np.log2(1 + (dim_x + dim_y) / 10)
				scale_factor = max(0.5, min(4, scale_factor))

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
		box_height = max_dim_z + 2.0 * self.z_padding
		box_side_area = total_box_volume / box_height

		if self.aligned:
			scale_factor = 2 - 0.3 * np.log2(1 + (avg_dim_x + avg_dim_y) / 10)
			scale_factor = max(0.5, min(2, scale_factor))
		else:
			scale_factor = 4 - 0.3 * np.log2(1 + (avg_dim_x + avg_dim_y) / 10)
			scale_factor = max(0.5, min(4, scale_factor))

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
				f.write("connect no\n")
				f.write("end structure\n")
			else:
				z_center = 0.5 * self.box_size_z			
				for L, rotated_pdb, dimensions in zip(
					self.unique_lengths,
					self.rotated_pdbs,
					self.dims
				):
					dim_z = float(dimensions[2])			
					if self.aligned:
						z_lower = z_center - 0.5 * dim_z - self.z_padding
						z_upper = z_center + 0.5 * dim_z + self.z_padding			 
						z_lower = max(0.0, z_lower)
						z_upper = min(self.box_size_z, z_upper)
					else:
						z_lower = 0.0
						z_upper = self.box_size_z
			
					f.write(f"structure {rotated_pdb}\n")
					f.write(f"number {self.length_counts[L]}\n")
					f.write(
						f"inside box "
						f"0.0 0.0 {z_lower:.3f} "
						f"{self.box_size_x:.3f} "
						f"{self.box_size_y:.3f} "
						f"{z_upper:.3f}\n"
					)
			
					if self.aligned:
						f.write("constrain_rotation x 0. 0.\n")
						f.write("constrain_rotation y 0. 0.\n")
						f.write("constrain_rotation z 0. 0.\n")
			
					f.write("connect no\n")
					f.write("end structure\n")
	
		run_command('packmol < packmol_input.inp')
		
		standardize_pdb_numbering(self.output_pdb)
	
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

		run_command('tleap -f final_leap_input.in > final_leap_input.out')
		ambpdb_command = f"ambpdb -p {self.prmtop} -c {self.inpcrd} > {amber_pdb}"
		run_command(ambpdb_command)
		standardize_pdb_numbering(amber_pdb)

class Dry_MDSimulation():
	def __init__(self, nproc, output, amber_params: Dry_AmberParams, use_gpu=True, scratch_dir=None):
		self.prmtop = amber_params.prmtop
		self.inpcrd = amber_params.inpcrd
		self.nproc = nproc
		self.dir1 = os.getcwd()
		self.output = output
		self.nproc = nproc
		self.use_gpu = use_gpu
		self.scratch_dir = scratch_dir

	def run_simulation(self, step_name, input_file, output_file, restart_in, restart_out, reference_file, additional_args=""):
		with open(self.output, 'a') as f:
			print(f"\t\tStarted {step_name} step", file=f)

		temp_dir = create_amber_temp_dir(self.scratch_dir, step_name)
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

		run_command(cmd)
		
		subprocess.run(f"cp * {self.dir1}", shell=True, check=True)
		shutil.rmtree(temp_dir)

		os.chdir(self.dir1)
		shutil.rmtree(folder_name)
		with open(self.output, 'a') as f:
			print(f"\t\tFinished {step_name} step", file=f)

	def run_all_steps(self, protocol):
		if protocol == "6-step":
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
		elif protocol == "12-step":
			steps = [
				("dry-eq_min", "dry-eq_0-min.in", "dry-eq_0-min.out", f"{self.inpcrd}", "dry-eq_0-min.ncrst", f"{self.inpcrd}"),
				("dry-eq_1-nvt", "dry-eq_1-nvt.in", "dry-eq_1-nvt.out", "dry-eq_0-min.ncrst", "dry-eq_1-nvt.ncrst", "dry-eq_0-min.ncrst", "-x dry-eq_1-nvt.nc"),
				("dry-eq_2-npt", "dry-eq_2-npt.in", "dry-eq_2-npt.out", "dry-eq_1-nvt.ncrst", "dry-eq_2-npt.ncrst", "dry-eq_1-nvt.ncrst", "-x dry-eq_2-npt.nc"),
				("dry-eq_3-nvt", "dry-eq_3-nvt.in", "dry-eq_3-nvt.out", "dry-eq_2-npt.ncrst", "dry-eq_3-nvt.ncrst", "dry-eq_2-npt.ncrst", "-x dry-eq_3-nvt.nc"),
				("dry-eq_4-npt", "dry-eq_4-npt.in", "dry-eq_4-npt.out", "dry-eq_3-nvt.ncrst", "dry-eq_4-npt.ncrst", "dry-eq_3-nvt.ncrst", "-x dry-eq_4-npt.nc"),
				("dry-eq_5-nvt", "dry-eq_5-nvt.in", "dry-eq_5-nvt.out", "dry-eq_4-npt.ncrst", "dry-eq_5-nvt.ncrst", "dry-eq_4-npt.ncrst", "-x dry-eq_5-nvt.nc"),
				("dry-eq_6-npt", "dry-eq_6-npt.in", "dry-eq_6-npt.out", "dry-eq_5-nvt.ncrst", "dry-eq_6-npt.ncrst", "dry-eq_5-nvt.ncrst", "-x dry-eq_6-npt.nc"),
				("dry-eq_7-nvt", "dry-eq_7-nvt.in", "dry-eq_7-nvt.out", "dry-eq_6-npt.ncrst", "dry-eq_7-nvt.ncrst", "dry-eq_6-npt.ncrst", "-x dry-eq_7-nvt.nc"),
				("dry-eq_8-npt", "dry-eq_8-npt.in", "dry-eq_8-npt.out", "dry-eq_7-nvt.ncrst", "dry-eq_8-npt.ncrst", "dry-eq_7-nvt.ncrst", "-x dry-eq_8-npt.nc"),
				("dry-eq_9-nvt", "dry-eq_9-nvt.in", "dry-eq_9-nvt.out", "dry-eq_8-npt.ncrst", "dry-eq_9-nvt.ncrst", "dry-eq_8-npt.ncrst", "-x dry-eq_9-nvt.nc"),
				("dry-eq_10-npt", "dry-eq_10-npt.in", "dry-eq_10-npt.out", "dry-eq_9-nvt.ncrst", "dry-eq_10-npt.ncrst", "dry-eq_9-nvt.ncrst", "-x dry-eq_10-npt.nc"),
				("dry-eq_11-nvt", "dry-eq_11-nvt.in", "dry-eq_11-nvt.out", "dry-eq_10-npt.ncrst", "dry-eq_11-nvt.ncrst", "dry-eq_10-npt.ncrst", "-x dry-eq_11-nvt.nc"),
				("dry-eq_12-npt", "dry-eq_12-npt.in", "dry-eq_12-npt.out", "dry-eq_11-nvt.ncrst", "dry-eq_12-npt.ncrst", "dry-eq_11-nvt.ncrst", "-x dry-eq_12-npt.nc"),
				("dry-eq_13-nvt-pr", "dry-eq_13-nvt-pr.in", "dry-eq_13-nvt-pr.out", "dry-eq_12-npt.ncrst", "dry-eq_13-nvt-pr.ncrst", "dry-eq_12-npt.ncrst", "-x dry-eq_13-nvt-pr.nc"),
			]
		else:
			raise ValueError(f"Unknown dry equilibration protocol: {protocol}")

		for step in steps:
			self.run_simulation(*step)

# ====
# Hydration-stage classes
# ====

class Hyd_BulkCreator:
	def __init__(self, polymer, chain_length, num_chains, x, y, z, num_h2o, lam, pdb_file, mix_chains=False):
		self.polymer = polymer
		self.chain_length = chain_length
		self.num_chains = num_chains
		self.box_size_x = x
		self.box_size_y = y
		self.box_size_z = z
		self.num_h2o = num_h2o
		self.lam = lam
		self.pdb_file = pdb_file
		self.mix_chains = mix_chains
		self.system_tag = hyd_system_name(polymer, chain_length, num_chains, lam, mix_chains)

	def packmol_generate_box(self):
		standardize_pdb_numbering(self.pdb_file)

		with open("packmol_input.inp", "w") as f:
			f.write(f"""
			tolerance 1.5
			filetype pdb
			add_amber_ter
			amber_ter_preserve
			output {self.system_tag}.pdb
			pbc {self.box_size_x} {self.box_size_y} {self.box_size_z}
			
			structure {self.pdb_file}
			number 1
			fixed 0. 0. 0. 0. 0. 0.
			resnumbers 1
			connect no
			end structure
			
			structure h2o.pdb
			number {self.num_h2o}
			resnumbers 3
			connect no
			end structure
			""")
		run_command('packmol < packmol_input.inp')

		standardize_pdb_numbering(f"{self.system_tag}.pdb")

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
		self.mix_chains = bulk_creator.mix_chains
		self.system_tag = hyd_system_name(self.polymer, self.chain_length, self.num_chains, self.lam, self.mix_chains)
		self.pdb_file = f"{self.system_tag}.pdb"

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
			saveamberparm mol {self.system_tag}.prmtop {self.system_tag}.inpcrd
			quit			
			""")

		self.prmtop = f"{self.system_tag}.prmtop"
		self.inpcrd = f"{self.system_tag}.inpcrd"
		amber_pdb = f"{self.system_tag}_amber.pdb"
	
		run_command('tleap -f final_leap_input.in > final_leap_input.out')
		ambpdb_command = f"ambpdb -p {self.prmtop} -c {self.inpcrd} > {amber_pdb}"
		run_command(ambpdb_command)
		standardize_pdb_numbering(amber_pdb)

class Hyd_MDSimulation():
	def __init__(self, nproc, output, amber_params: Hyd_AmberParams, use_gpu=True, scratch_dir=None):
		self.prmtop = amber_params.prmtop
		self.inpcrd = amber_params.inpcrd
		self.lam = amber_params.lam
		self.nproc = nproc
		self.dir1 = os.getcwd()
		self.output = output
		self.nproc = nproc
		self.use_gpu = use_gpu
		self.scratch_dir = scratch_dir
		
	def run_simulation(self, step_name, input_file, output_file, restart_in, restart_out, reference_file, additional_args=""):
		with open(self.output, "a") as f:
			print(f"\t\tStarted {step_name} step", file=f)
			
		temp_dir = create_amber_temp_dir(self.scratch_dir, step_name)
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

		run_command(cmd)

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

# ====
# Conductivity-stage classes
# ====

class Cond_BulkCreator:
	def __init__(self, polymer, chain_length, num_chains, x, y, z, num_h3o, lam, pdb_file, mix_chains=False):
		self.polymer = polymer
		self.chain_length = chain_length
		self.num_chains = num_chains
		self.box_size_x = x
		self.box_size_y = y
		self.box_size_z = z
		self.num_h3o = num_h3o
		self.lam = lam
		self.pdb_file = pdb_file
		self.mix_chains = mix_chains
		self.system_tag = cond_system_name(polymer, chain_length, num_chains, lam, mix_chains)
		
	def packmol_generate_box(self):
		standardize_pdb_numbering(self.pdb_file)

		with open("packmol_input.inp", "w") as f:
			f.write(f"""
			tolerance 1.5
			filetype pdb
			add_amber_ter
			amber_ter_preserve
			output {self.system_tag}.pdb
			pbc {self.box_size_x} {self.box_size_y} {self.box_size_z}
			
			structure {self.pdb_file}
			number 1
			fixed 0. 0. 0. 0. 0. 0.
			resnumbers 1
			connect no
			end structure
			
			structure h3o.pdb
			number {self.num_h3o}
			resnumbers 3
			connect no
			end structure
			""")
		run_command('packmol < packmol_input.inp')

		standardize_pdb_numbering(f"{self.system_tag}.pdb")
	
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
		self.mix_chains = bulk_creator.mix_chains
		self.system_tag = cond_system_name(self.polymer, self.chain_length, self.num_chains, self.lam, self.mix_chains)
		self.pdb_file = f"{self.system_tag}.pdb"
	
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
			saveamberparm mol {self.system_tag}.prmtop {self.system_tag}.inpcrd
			quit
			""")
		
		self.prmtop = f"{self.system_tag}.prmtop"
		self.inpcrd = f"{self.system_tag}.inpcrd"
		amber_pdb = f"{self.system_tag}_amber.pdb"
	
		run_command('tleap -f final_leap_input.in > final_leap_input.out')
		ambpdb_command = f"ambpdb -p {self.prmtop} -c {self.inpcrd} > {amber_pdb}"
		run_command(ambpdb_command)
		standardize_pdb_numbering(amber_pdb)

class Cond_MDSimulation:
	def __init__(self, nproc, output, amber_params: Cond_AmberParams, use_gpu=True, scratch_dir=None):
		self.prmtop = amber_params.prmtop
		self.inpcrd = amber_params.inpcrd
		self.lam = amber_params.lam
		self.nproc = nproc
		self.dir1 = os.getcwd()
		self.output = output
		self.use_gpu = use_gpu
		self.scratch_dir = scratch_dir
		
	def run_simulation(self, step_name, input_file, output_file, restart_in, restart_out, reference_file, additional_args=""):
		with open(self.output, "a") as f:
			print(f"\t\tStarted {step_name} step", file=f)
		
		temp_dir = create_amber_temp_dir(self.scratch_dir, step_name)
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

		run_command(cmd)

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

# ====
# Analysis tools
# ====

class Analysis():
	def merge_nc_files(self, prmtop_file, ncrst_file, pdb_file, nc_files, prefix, merged_pdb, cpptraj_file):
		ambpdb_command = f"ambpdb -p {prmtop_file} -c {ncrst_file} > {pdb_file}"
		run_command(ambpdb_command)
		standardize_pdb_numbering(pdb_file)
		
		with open(cpptraj_file, 'w') as file:
			for nc_file in nc_files:			
				file.write(f"trajin {nc_file} 1 100 10\n")
			file.write(f"trajout {prefix} pdb multi\n")
		
		cpptraj_command = f"cpptraj -i {cpptraj_file} -p {prmtop_file}"
		run_command(cpptraj_command)
		
		merge_command = f"ls -v {prefix}* | xargs cat > {merged_pdb}"
		run_command(merge_command)
		standardize_pdb_numbering(merged_pdb)
				
		for file in os.listdir():
			if file.startswith(prefix):
				os.remove(file)			

# ====
# Workflow functions
# ====

def build_monomer_and_conformers():
	monomer_builder = MonomerBuilder(polymer, polymer_psmiles, conf_num)
	monomer_builder.create_monomer_unit()
	monomer_builder.create_conformations()
	log_message("\tMonomer building and conformer creation finished")

def optimize_and_rank_conformers(conf_output_file, calculator):
	conformation_analysis = ConformationAnalysis(polymer, calculator)
	conformation_analysis.optimize_confomer()
	log_message("\tConformation analysis finished")
	
	analyzer = ConformationAnalyzer(polymer, conf_output_file, temperature)
	analyzer.analyze_conformers()
	log_message("\tEvaluation of conformers finished")

	selected_conf = select_conformer(conf_output_file, conf_selection, conf_far_fraction)
	log_message(f"\tSelected conformer mode = {conf_selection}, conf_i = {selected_conf}")
	return selected_conf

def run_parameterization(selected_conf, chain_lengths, nproc, orca_profile, protonated_charge, deprotonated_charge, num_sulfonic_acid_groups):
	param = GAFF2Param(polymer, chain_length, nproc, orca_profile, orca_dir, protonated_charge, deprotonated_charge, num_sulfonic_acid_groups)
	param.parameterization(selected_conf)

	if mix_chains:
		for L in sorted(set(chain_lengths)):
			if L != chain_length:
				param.create_polymer_chain(chain_length=L)

	log_message("\tGAFF2 parameters and single polymer chain creation finished")

def run_dry_workflow(chain_lengths, system_tag, nproc, scratch_dir):
	os.chdir(init_dir)

	dry_bulk_creator = Dry_BulkCreator(
		polymer, chain_length, num_chains, mix_chains, chain_lengths, aligned, pack_z_padding
	)
	dry_bulk_creator.create_bulk_phase()

	log_message("\tBulk phase creation finished")
	log_message(f"\tAligned packing = {aligned}")

	dry_amber = Dry_AmberParams(polymer, chain_length, num_chains, bulk_creator=dry_bulk_creator)
	dry_amber.create_amber_params()
	log_message("\tAmber parameters are created for the bulk phase")

	os.chdir(dry_eq_dir)

	for params in [f"{system_tag}.prmtop", f"{system_tag}.inpcrd"]:
		shutil.copy(os.path.join(init_dir, params), dry_eq_dir)

	if dry_eq_prot == "6-step":
		dry_eq_input_files = [
			"dry-eq_0-min.in",
			"dry-eq_1-nvt.in",
			"dry-eq_2-npt.in",
			"dry-eq_3-nvt.in",
			"dry-eq_4-npt.in",
			"dry-eq_5-nvt.in",
			"dry-eq_6-npt.in",
			"dry-eq_7-nvt-pr.in",
		]
	elif dry_eq_prot == "12-step":
		dry_eq_input_files = [
			"dry-eq_0-min.in",
			"dry-eq_1-nvt.in",
			"dry-eq_2-npt.in",
			"dry-eq_3-nvt.in",
			"dry-eq_4-npt.in",
			"dry-eq_5-nvt.in",
			"dry-eq_6-npt.in",
			"dry-eq_7-nvt.in",
			"dry-eq_8-npt.in",
			"dry-eq_9-nvt.in",
			"dry-eq_10-npt.in",
			"dry-eq_11-nvt.in",
			"dry-eq_12-npt.in",
			"dry-eq_13-nvt-pr.in",
		]
	else:
		raise ValueError(f"Unknown dry equilibration protocol: {dry_eq_prot}")

	for inputs in dry_eq_input_files:
		shutil.copy(os.path.join(dry_eq_input_dir, inputs), dry_eq_dir)	   

	log_message("\tDry equilibration MD simulations started")
	dry_md = Dry_MDSimulation(nproc, output, amber_params=dry_amber, use_gpu=use_gpu, scratch_dir=scratch_dir)
	dry_md.run_all_steps(dry_eq_prot)
	log_message("\tDry equilibration MD simulations finished")

	if dry_eq_prot == "6-step":
		nc_files = [
			"dry-eq_1-nvt.nc",
			"dry-eq_2-npt.nc",
			"dry-eq_3-nvt.nc",
			"dry-eq_4-npt.nc",
			"dry-eq_5-nvt.nc",
			"dry-eq_6-npt.nc",
			"dry-eq_7-nvt-pr.nc",
		]
		ncrst_file = "dry-eq_7-nvt-pr.ncrst"
	
	elif dry_eq_prot == "12-step":
		nc_files = [
			"dry-eq_1-nvt.nc",
			"dry-eq_2-npt.nc",
			"dry-eq_3-nvt.nc",
			"dry-eq_4-npt.nc",
			"dry-eq_5-nvt.nc",
			"dry-eq_6-npt.nc",
			"dry-eq_7-nvt.nc",
			"dry-eq_8-npt.nc",
			"dry-eq_9-nvt.nc",
			"dry-eq_10-npt.nc",
			"dry-eq_11-nvt.nc",
			"dry-eq_12-npt.nc",
			"dry-eq_13-nvt-pr.nc",
		]
		ncrst_file = "dry-eq_13-nvt-pr.ncrst"
	
	else:
		raise ValueError(f"Unknown dry equilibration protocol: {dry_eq_prot}")

	prmtop_file = f"{system_tag}.prmtop"
	pdb_file = f"{system_tag}_dry-eq_last.pdb"
	merged_pdb = f"{system_tag}_dry-eq.pdb"
	cpptraj_file = "cpptraj.in"
	prefix = "dry-eq_tmp"

	dry_md_analysis = Analysis()
	dry_md_analysis.merge_nc_files(prmtop_file, ncrst_file, pdb_file, nc_files, prefix, merged_pdb, cpptraj_file)

	log_message("\tDry equilibration trajectory merging finished")
	return pdb_file

def prepare_hydration_inputs(dry_pdb_file):
	os.chdir(hyd_eq_dir)

	shutil.copy(os.path.join(dry_eq_dir, dry_pdb_file), hyd_eq_dir)
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
		shutil.copy(os.path.join(hyd_eq_input_dir, inputs), hyd_eq_dir)

def run_hydration_workflow(dry_pdb_file, nproc, scratch_dir, num_sulfonic_sites):
	os.chdir(hyd_eq_dir)

	base_hyd_pdb = dry_pdb_file
	a, b, c = get_pdb_box_lengths(base_hyd_pdb)

	for i, lam in enumerate(lam_list):
		if i > 0:
			prev_hyd_tag = hyd_system_name(polymer, chain_length, num_chains, lam_list[i-1], mix_chains)
			base_hyd_pdb = f"{prev_hyd_tag}_hyd-eq_last.pdb"
			a, b, c = get_pdb_box_lengths(base_hyd_pdb)
	
		prev_lam = 0 if i == 0 else lam_list[i - 1]
		add_lam = lam - prev_lam
		if add_lam <= 0:
			raise ValueError(f"lam_list must be strictly increasing. Previous lambda={prev_lam}, current lambda={lam}")
		num_h2o = add_lam * num_sulfonic_sites
		with open(output, 'a') as f:
			print(f"\tHydration lambda={lam}: target total H2O={lam * num_sulfonic_sites}", file=f)
		
		lam_tag = hyd_system_name(polymer, chain_length, num_chains, lam, mix_chains)
		
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
		hyd_bulk_creator = Hyd_BulkCreator(polymer, chain_length, num_chains, a, b, c, num_h2o, lam, base_hyd_pdb, mix_chains)
		hyd_bulk_creator.create_bulk_phase()
		with open(output, 'a') as f:
			print(f"\tHydration lambda={lam} bulk phase creation finished", file=f)
	
		# Create amber parameters for specific hydration level (lambda)
		hyd_amber = Hyd_AmberParams(polymer, chain_length, num_chains, bulk_creator = hyd_bulk_creator)
		hyd_amber.create_amber_params()
		with open(output, 'a') as f:
			print(f"\tAmber parameters are created for the hydration lambda={lam} bulk phase", file=f)
	
		for fname in [
			f"{lam_tag}.prmtop",
			f"{lam_tag}.inpcrd",
			f"{lam_tag}_amber.pdb",
		]:
			shutil.move(fname, os.path.join(lam_md_dir, os.path.basename(fname)))
		
		for fname in os.listdir("."):
			if os.path.isfile(fname) and not fname.endswith((".prmtop", ".inpcrd", ".pdb")):
				shutil.move(fname, os.path.join(lam_init_dir, fname))
		
		os.chdir(lam_md_dir)
		
		# Run the dry equilibration MD simulations sequence using Amber software (pmemd.MPI & pmemd.cuda)
		with open(output, 'a') as f:
			print(f"\tHydration lambda={lam} equilibration MD simulations started", file=f)
		hyd_md = Hyd_MDSimulation(nproc, output, amber_params=hyd_amber, use_gpu=use_gpu, scratch_dir=scratch_dir)
		hyd_md.run_all_steps()
		with open(output, 'a') as f:
			print(f"\tHydration lambda={lam} equilibration MD simulations finished", file=f)
		
		# Perform trajectory files merging and conversion
		nc_files = [f"hyd-eq_1-nvt.nc",
			f"hyd-eq_2-nvt.nc",
			f"hyd-eq_3-nvt.nc", 
			f"hyd-eq_4-npt.nc",
			f"hyd-eq_5-nvt-pr.nc",
		]
		prmtop_file = f"{lam_tag}.prmtop"
		ncrst_file = f"hyd-eq_5-nvt-pr.ncrst"
		pdb_file = f"{lam_tag}_hyd-eq_last.pdb"
		merged_pdb = f"{lam_tag}_hyd-eq.pdb"
		cpptraj_file="cpptraj.in"
		prefix=f"hdy-eq_tmp"
		hyd_md_analysis = Analysis()
		hyd_md_analysis.merge_nc_files(prmtop_file, ncrst_file, pdb_file, nc_files, prefix, merged_pdb, cpptraj_file)
	
		shutil.copy(pdb_file, hyd_eq_dir)
		os.chdir(hyd_eq_dir)

def prepare_conductivity_inputs():
	os.makedirs(cond_pr_dir, exist_ok=True)

	for h3o_files in [
		"h3o.pdb",
		"h3o.frcmod",
		"h3o.prepi", 
	]:
		shutil.copy(os.path.join(input_dir, h3o_files), cond_pr_dir)
	
	for fname in [
		"cond_0-min.in",
		"cond_pr-nvt.in",
	]:
		shutil.copy(os.path.join(cond_pr_input_dir, fname), cond_pr_dir)
	
	for params in [
		"h_so3.prepi",
		"t_so3.prepi",
		f"{polymer}_m_so3.prepi",
		f"{polymer}_so3_gaff2.frcmod",
	]:
		shutil.copy(os.path.join(init_dir, params), cond_pr_dir)

def run_conductivity_workflow(nproc, chain_lengths, scratch_dir):
	for lam in cond_lam_list:	 
		lam_cond_pr_dir = os.path.join(cond_pr_dir, f"{lam}_h3o-h2o")
		lam_cond_init_dir = os.path.join(lam_cond_pr_dir, "init")
		lam_cond_md_dir = os.path.join(lam_cond_pr_dir, "md")
		
		os.makedirs(lam_cond_pr_dir, exist_ok=True)
		os.makedirs(lam_cond_init_dir, exist_ok=True)
		os.makedirs(lam_cond_md_dir, exist_ok=True)
		
		hyd_tag = hyd_system_name(polymer, chain_length, num_chains, lam, mix_chains)
		hyd_pdb = f"{hyd_tag}_hyd-eq_last.pdb"
		shutil.copy(os.path.join(hyd_eq_dir, hyd_pdb), lam_cond_pr_dir)
		
		for fname in [
			"h_so3.prepi",
			"t_so3.prepi",
			f"{polymer}_m_so3.prepi",
			f"{polymer}_so3_gaff2.frcmod",
			"h3o.pdb",
			"h3o.frcmod",
			"h3o.prepi", 
			"cond_0-min.in",
			"cond_pr-nvt.in"
		]:
			shutil.copy(os.path.join(cond_pr_dir, fname), lam_cond_pr_dir)
		
		os.chdir(lam_cond_pr_dir)
		a, b, c = get_pdb_box_lengths(hyd_pdb)
		hyd_clean_pdb = f"{hyd_tag}_hyd-eq_clean.pdb"
		cleaner = PDBCleaner(hyd_pdb, hyd_clean_pdb, chain_length, num_chains, mix_chains, chain_lengths)
		cleaner.remove_so3h_hydrogens()
		
		if cleaner.hydrogens_removed_count != num_sulfonic_sites:
			raise RuntimeError(
				f"Expected {num_sulfonic_sites} "
				f"sulfonic-acid hydrogens in the polymer "
				f"system, but removed "
				f"{cleaner.hydrogens_removed_count}."
				)
		
		num_h3o = cleaner.hydrogens_removed_count
		
		with open(output, 'a') as f:
			print(f"\tConductivity lambda={lam} removed SO3H hydrogens = {num_h3o} and H2O molecule = {num_h3o}", file=f)
		
		cond_bulk_creator = Cond_BulkCreator(polymer, chain_length, num_chains, a, b, c, num_h3o, lam, hyd_clean_pdb, mix_chains=mix_chains)
		cond_bulk_creator.create_bulk_phase()
		
		with open(output, 'a') as f:
			print(f"\tConductivity lambda={lam} bulk phase creation finished", file=f)
		
		cond_amber = Cond_AmberParams(polymer, chain_length, num_chains, bulk_creator=cond_bulk_creator)
		cond_amber.create_amber_params()
	
		cond_tag = cond_system_name(polymer, chain_length, num_chains, lam, mix_chains)
		
		with open(output, 'a') as f:
			print(f"\tAmber parameters are created for the conductivity lambda={lam} bulk phase", file=f)	 
		
		for fname in [
			f"{cond_tag}.prmtop",
			f"{cond_tag}.inpcrd",
			f"{cond_tag}_amber.pdb",
			"cond_0-min.in",
			"cond_pr-nvt.in",
		]:
			shutil.move(fname, os.path.join(lam_cond_md_dir, os.path.basename(fname)))
		
		for fname in os.listdir("."):
			if os.path.isfile(fname) and fname not in[
				f"{cond_tag}.prmtop",
				f"{cond_tag}.inpcrd",
				f"{cond_tag}_amber.pdb",	   
			]:
				shutil.move(fname, os.path.join(lam_cond_init_dir, fname))
		
		os.chdir(lam_cond_md_dir)
		
		with open(output, 'a') as f:
			print(f"\tConductivity lambda={lam} MD simulations started", file=f)
			
		cond_md = Cond_MDSimulation(nproc, output, amber_params=cond_amber, use_gpu=use_gpu, scratch_dir=scratch_dir)
		cond_md.run_all_steps()
		
		with open(output, 'a') as f:
			print(f"\tConductivity lambda={lam} MD simulations finished", file=f)
		
		os.chdir(base_dir)

# ====
# Main workflow
# ====

def run_workflow():
	validate_settings()
	ensure_directories()

	os.chdir(init_dir)

	global use_gpu
	use_gpu = get_use_gpu(use_gpu)
	nproc = get_nproc(use_nproc)
	resolved_scratch_dir = get_scratch_dir(scratch_dir)
	
	chain_lengths = generate_chain_lengths(chain_length, num_chains, mix_chains, mix_seed, mix_chain_fraction)
	num_sulfonic_acid_groups, protonated_charge, deprotonated_charge = get_psmiles_chemistry(polymer_psmiles)
	num_sulfonic_sites = (num_sulfonic_acid_groups * sum(chain_lengths))
	system_tag = system_name(polymer, chain_length, num_chains, mix_chains)
	conf_output_file = f"{polymer}_conf.txt"

	log_settings(chain_lengths, system_tag, nproc, resolved_scratch_dir, num_sulfonic_acid_groups, protonated_charge, deprotonated_charge)

	if run_param:
		nnp_calc = get_mace_calculator(use_mace_device, nproc)
		orca_profile = get_orca_profile(orca_dir)
		build_monomer_and_conformers()
		selected_conf = optimize_and_rank_conformers(conf_output_file, nnp_calc)
		run_parameterization(selected_conf, chain_lengths, nproc, orca_profile, protonated_charge, deprotonated_charge, num_sulfonic_acid_groups)
		
	dry_pdb_file = None
	if run_dry:
		dry_pdb_file = run_dry_workflow(chain_lengths, system_tag, nproc, resolved_scratch_dir)

	if run_hyd and dry_pdb_file is not None:
		prepare_hydration_inputs(dry_pdb_file)
		run_hydration_workflow(dry_pdb_file, nproc, resolved_scratch_dir, num_sulfonic_sites)

	if run_cond:
		prepare_conductivity_inputs()
		run_conductivity_workflow(nproc, chain_lengths, resolved_scratch_dir)

if __name__ == "__main__":
	run_workflow()