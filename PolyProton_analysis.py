import numpy as np
import pandas as pd
import os
import shutil
import glob
import subprocess
import warnings
from scipy.spatial import cKDTree
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier

warnings.simplefilter("ignore")

# ====
# User settings
# ====

# Polymer system settings
polymer = "a1"

chain_length = 10
num_chains = 15
mix_chains = False

# Hydration / conductivity
hyd_analysis_lam_list = [12]
cond_analysis_lam_list = [12]

# Workflow stages
run_dry_analysis = True
run_hyd_analysis = True
run_cond_analysis = True

# Dry equilibration protocol
# "6-step" or "12-step"
dry_eq_prot = "6-step"

# ====
# Paths / Folders
# ====

base_dir = os.getcwd()
output = os.path.join(base_dir, "PolyProton_analysis_process.txt")

# expected folder structure:
# base_dir/
#	simulation/
#	  init/
#	  dry_eq/
#	  hyd_eq/
#	  cond_pr/
#	analysis/
#	  dry/
#	  hyd/
#	  cond/		   

simulation_dir = os.path.join(base_dir, "simulation")
dry_eq_dir = os.path.join(simulation_dir, "dry_eq")
hyd_eq_dir = os.path.join(simulation_dir, "hyd_eq")
cond_pr_dir = os.path.join(simulation_dir, "cond_pr")

analysis_dir = os.path.join(base_dir, "analysis")
dry_analysis_dir = os.path.join(analysis_dir, "dry")
hyd_analysis_dir = os.path.join(analysis_dir, "hyd")
cond_analysis_dir = os.path.join(analysis_dir, "cond")

# Dry production trajectory is selected automatically from dry_eq_prot
def dry_production_files(protocol):
	if protocol == "6-step":
		return "dry-eq_7-nvt-pr.nc", "dry-eq_7-nvt-pr.ncrst"

	if protocol == "12-step":
		return "dry-eq_13-nvt-pr.nc", "dry-eq_13-nvt-pr.ncrst"

	raise ValueError("dry_eq_prot must be '6-step' or '12-step'")

dry_traj_file, dry_restart_file = dry_production_files(dry_eq_prot)

# RDF settings for dry systems
dry_rdf_pairs = ["N-N", "S-S", "N-S"]

rdf_settings = {
	"N-N": {"cutoff": 30.0, "bins": 300},
	"S-S": {"cutoff": 15.0, "bins": 150},
	"N-S": {"cutoff": 30.0, "bins": 300},
}

# frame selection for dry pr-nvt trajectory
frame_start = 1
frame_stop = 100
frame_stride = 10
overwrite = True

# Hydration production trajectory
hyd_traj_file = "hyd-eq_5-nvt-pr.nc"
hyd_restart_file = "hyd-eq_5-nvt-pr.ncrst"

# RDF settings for hydrated systems
hyd_rdf_pairs = ["N-N", "S-S", "N-S", "S-W", "W-W"]

hyd_rdf_settings = {
	"N-N": {"cutoff": 30.0, "bins": 300},
	"S-S": {"cutoff": 15.0, "bins": 150},
	"N-S": {"cutoff": 30.0, "bins": 300},
	"S-W": {"cutoff": 15.0, "bins": 150},
	"W-W": {"cutoff": 15.0, "bins": 150},
}

# Water cluster settings
water_oo_cutoff = 3.5

# Conductivity production trajectory
cond_traj_file = "cond_pr-nvt.nc"
cond_restart_file = "cond_pr-nvt.ncrst"

# Conductivity frame selection
cond_frame_start = 1
cond_frame_stop = 10000
cond_frame_stride = 10
cond_time_between_nc_frames_ps = 10.0

# MSD fitting window
msd_fit_start_ps = 10000.0	 # 10 ns
msd_fit_end_ps = 100000.0	 # 100 ns

# Conductivity calculation
cond_temperature_K = 300.0

# Residence analysis
h3o_s_residence_cutoff = 4.0  # A

# ====
# Constants
# ====

amu_to_g = 1.66054e-24
angstrom3_to_cm3 = 1.0e-24

molecular_weights = {
	"C": 12.0107,
	"O": 15.9994,
	"S": 32.065,
	"H": 1.00794,
	"N": 14.0067,
}

excluded_non_polymer_resnames = ["WAT", "HOH", "H3O"]

elementary_charge_C = 1.602176634e-19
boltzmann_J_K = 1.380649e-23
angstrom2_per_ps_to_m2_per_s = 1.0e-8
angstrom3_to_m3 = 1.0e-30

# ====
# Helper functions
# ====

def log_message(message):
	with open(output, "a", encoding="utf-8") as f:
		print(message, file=f)

def ensure_directories():
	os.makedirs(analysis_dir, exist_ok=True)
	os.makedirs(dry_analysis_dir, exist_ok=True)
	os.makedirs(hyd_analysis_dir, exist_ok=True)
	os.makedirs(cond_analysis_dir, exist_ok=True)

def system_name(polymer, chain_length, num_chains, mix_chains=False):
	if mix_chains:
		return f"{polymer}_mix-n-{chain_length}x{num_chains}"
	return f"{polymer}_n-{chain_length}x{num_chains}"

def hyd_system_name(polymer, chain_length, num_chains, lam, mix_chains=False):
	return f"{system_name(polymer, chain_length, num_chains, mix_chains)}_{lam}-h2o"

def cond_system_name(polymer, chain_length, num_chains, lam, mix_chains=False):
	return f"{system_name(polymer, chain_length, num_chains, mix_chains)}_{lam}-h3o-h2o"

def find_prmtop(stage_dir):
	prmtop_files = sorted(glob.glob(os.path.join(stage_dir, "*.prmtop")))
	if len(prmtop_files) == 0:
		raise FileNotFoundError(f"No prmtop file found in {stage_dir}")

	if len(prmtop_files) > 1:
		log_message("Multiple prmtop files found. Using the first one:")
		for f in prmtop_files:
			log_message(f"\t{os.path.basename(f)}")

	return prmtop_files[0]

def run_command(command):
	log_message(f"Running command: {command}")
	command_log = os.path.join(analysis_dir, "analysis_command.log")
	with open(command_log, "a", encoding="utf-8") as log:
		log.write(f"\n\n$ {command}\n")
		subprocess.run(command,shell=True,check=True,stdout=log,stderr=log)

def is_atom_line(line):
	return line.startswith("ATOM") or line.startswith("HETATM")

def atom_name(line):
	return line[12:16].strip()

def residue_name(line):
	return line[17:20].strip()

def element_from_pdb_line(line):
	element = line[76:78].strip()
	if element:
		return element

	name = atom_name(line)
	letters = "".join([c for c in name if c.isalpha()])
	return letters[:1]

def load_first_model_atom_lines(pdb_file):
	lines = []
	in_model = False
	saw_model = False

	with open(pdb_file, "r") as f:
		for line in f:
			if line.startswith("MODEL"):
				in_model = True
				saw_model = True
				continue

			if line.startswith("ENDMDL") and saw_model:
				break

			if saw_model and not in_model:
				continue

			if is_atom_line(line) or line.startswith("TER"):
				lines.append(line)

			if not saw_model and line.startswith("END"):
				break

	return lines

def load_chain_line_blocks_from_first_model(pdb_file):
	lines = load_first_model_atom_lines(pdb_file)

	chains = []
	current = []

	for line in lines:
		if line.startswith("TER"):
			if len(current) > 0:
				chains.append(current)
				current = []
			continue

		if is_atom_line(line):
			resname = residue_name(line)
			if resname in excluded_non_polymer_resnames:
				continue
			current.append(line)

	if len(current) > 0:
		chains.append(current)

	return chains

def atoms_from_pdb_lines(lines):
	pdb_str = "".join(lines)
	return read(StringIO(pdb_str), format="proteindatabank")

def get_first_n_per_residue(chain_atoms, chain_lines):
	residue_to_n = {}

	for idx, line in enumerate(chain_lines):
		if not is_atom_line(line):
			continue

		if element_from_pdb_line(line) != "N":
			continue

		resid = line[22:26].strip()
		residue_to_n.setdefault(resid, []).append(idx)

	first_n_indices = []

	for resid in sorted(residue_to_n.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
		first_n_indices.append(residue_to_n[resid][0])

	return first_n_indices

def get_monomer_unit_n_atom_pairs(pdb_file):
	lines = load_first_model_atom_lines(pdb_file)

	monomer_pairs = []
	current_n_by_residue = {}
	global_atom_idx = -1

	def flush_chain(current_n_by_residue):
		if len(current_n_by_residue) == 0:
			return []

		ordered_resids = sorted(
			current_n_by_residue.keys(),
			key=lambda x: int(x) if str(x).isdigit() else str(x)
		)

		n_indices = [current_n_by_residue[r][0] for r in ordered_resids]

		pairs = []
		for i in range(len(n_indices) - 1):
			pairs.append((n_indices[i], n_indices[i + 1]))

		return pairs

	for line in lines:
		if line.startswith("TER"):
			monomer_pairs.extend(flush_chain(current_n_by_residue))
			current_n_by_residue = {}
			continue

		if not is_atom_line(line):
			continue

		global_atom_idx += 1

		resname = residue_name(line)
		if resname in excluded_non_polymer_resnames:
			continue

		if element_from_pdb_line(line) != "N":
			continue

		resid = line[22:26].strip()
		current_n_by_residue.setdefault(resid, []).append(global_atom_idx)

	monomer_pairs.extend(flush_chain(current_n_by_residue))

	if len(monomer_pairs) == 0:
		raise RuntimeError("Could not identify monomer-unit N-N atom pairs for S2 analysis.")

	return monomer_pairs
	
def nematic_order_from_unit_vectors(unit_vectors):
	if len(unit_vectors) == 0:
		return np.nan

	Q = np.zeros((3, 3), dtype=float)
	identity = np.eye(3)

	for u in unit_vectors:
		Q += 1.5 * np.outer(u, u) - 0.5 * identity

	Q /= len(unit_vectors)

	eigvals = np.linalg.eigvalsh(Q)
	S2 = np.max(eigvals)

	return float(S2)

def validate_dry_inputs(prmtop_file):
	required_files = [
		prmtop_file,
		os.path.join(dry_eq_dir, dry_traj_file),
		os.path.join(dry_eq_dir, dry_restart_file),
	]

	missing = [f for f in required_files if not os.path.exists(f)]

	if len(missing) > 0:
		raise FileNotFoundError("Missing dry analysis input files:\n" + "\n".join(missing))

def is_water_residue(resname):
	return resname in ["WAT", "HOH"]

def is_water_oxygen_line(line):
	if not is_atom_line(line):
		return False

	resname = residue_name(line)
	if not is_water_residue(resname):
		return False

	elem = element_from_pdb_line(line)
	name = atom_name(line)

	return elem == "O" or name.upper().startswith("O")

def write_pdb_with_water_oxygen_type(input_pdb, output_pdb):
	"""
	Create a PDB for OVITO RDF where water oxygen atoms are labeled as Ow.
	This allows S-Ow and Ow-Ow RDFs instead of mixing water oxygen with polymer oxygen.
	"""
	with open(input_pdb, "r") as f_in, open(output_pdb, "w") as f_out:
		for line in f_in:
			if is_water_oxygen_line(line):
				# Atom name column: 12:16
				# Element column: 76:78
				new_line = line[:12] + "Ow	" + line[16:76] + "Ow" + line[78:]
				f_out.write(new_line)
			else:
				f_out.write(line)

def get_water_oxygen_indices_from_first_model(pdb_file):
	lines = load_first_model_atom_lines(pdb_file)

	water_o_indices = []
	global_atom_idx = -1

	for line in lines:
		if not is_atom_line(line):
			continue

		global_atom_idx += 1

		if is_water_oxygen_line(line):
			water_o_indices.append(global_atom_idx)

	if len(water_o_indices) == 0:
		raise RuntimeError("No water oxygen atoms were found for water cluster analysis.")

	return water_o_indices

def union_find_cluster_sizes(num_nodes, pairs):
	parent = list(range(num_nodes))

	def find(x):
		while parent[x] != x:
			parent[x] = parent[parent[x]]
			x = parent[x]
		return x

	def union(a, b):
		ra = find(a)
		rb = find(b)
		if ra != rb:
			parent[rb] = ra

	for i, j in pairs:
		union(i, j)

	cluster_count = {}

	for i in range(num_nodes):
		root = find(i)
		cluster_count[root] = cluster_count.get(root, 0) + 1

	sizes = sorted(cluster_count.values(), reverse=True)

	return sizes

def calculate_water_cluster_sizes(positions, box_lengths, cutoff):
	if len(positions) == 0:
		return []

	# Wrap positions into the periodic box for cKDTree periodic search
	box_lengths = np.array(box_lengths, dtype=float)
	wrapped = positions % box_lengths

	tree = cKDTree(wrapped, boxsize=box_lengths)
	pairs = list(tree.query_pairs(cutoff))

	sizes = union_find_cluster_sizes(len(wrapped), pairs)

	return sizes

def get_lam_hyd_dirs(lam):
	lam_dir = os.path.join(hyd_eq_dir, f"{lam}_h2o")
	lam_md_dir = os.path.join(lam_dir, "md")
	lam_analysis_dir = os.path.join(hyd_analysis_dir, f"{lam}_h2o")

	return lam_dir, lam_md_dir, lam_analysis_dir

def validate_hyd_inputs(lam, prmtop_file):
	lam_dir, lam_md_dir, lam_analysis_dir = get_lam_hyd_dirs(lam)

	required_files = [
		prmtop_file,
		os.path.join(lam_md_dir, hyd_traj_file),
		os.path.join(lam_md_dir, hyd_restart_file),
	]

	missing = [f for f in required_files if not os.path.exists(f)]

	if len(missing) > 0:
		raise FileNotFoundError(
			f"Missing hydration analysis input files for lambda={lam}:\n" +
			"\n".join(missing)
		)

def get_lam_cond_dirs(lam):
	lam_dir = os.path.join(cond_pr_dir, f"{lam}_h3o-h2o")
	lam_md_dir = os.path.join(lam_dir, "md")
	lam_analysis_dir = os.path.join(cond_analysis_dir, f"{lam}_h3o-h2o")

	return lam_dir, lam_md_dir, lam_analysis_dir

def validate_cond_inputs(lam, prmtop_file):
	lam_dir, lam_md_dir, lam_analysis_dir = get_lam_cond_dirs(lam)

	required_files = [
		prmtop_file,
		os.path.join(lam_md_dir, cond_traj_file),
		os.path.join(lam_md_dir, cond_restart_file),
	]

	missing = [f for f in required_files if not os.path.exists(f)]

	if len(missing) > 0:
		raise FileNotFoundError(
			f"Missing conductivity analysis input files for lambda={lam}:\n" +
			"\n".join(missing)
		)

def is_h3o_residue(resname):
	return resname == "H3O"

def is_h3o_oxygen_line(line):
	if not is_atom_line(line):
		return False

	resname = residue_name(line)
	if not is_h3o_residue(resname):
		return False

	elem = element_from_pdb_line(line)
	name = atom_name(line)

	return elem == "O" or name.upper().startswith("O")

def get_h3o_oxygen_indices_from_first_model(pdb_file):
	lines = load_first_model_atom_lines(pdb_file)

	h3o_o_indices = []
	global_atom_idx = -1

	for line in lines:
		if not is_atom_line(line):
			continue

		global_atom_idx += 1

		if is_h3o_oxygen_line(line):
			h3o_o_indices.append(global_atom_idx)

	if len(h3o_o_indices) == 0:
		raise RuntimeError("No H3O oxygen atoms were found for conductivity analysis.")

	return h3o_o_indices

def get_s_atom_indices_from_first_model(pdb_file):
	lines = load_first_model_atom_lines(pdb_file)

	s_indices = []
	global_atom_idx = -1

	for line in lines:
		if not is_atom_line(line):
			continue

		global_atom_idx += 1

		resname = residue_name(line)
		if resname in excluded_non_polymer_resnames:
			continue

		if element_from_pdb_line(line) == "S":
			s_indices.append(global_atom_idx)

	if len(s_indices) == 0:
		raise RuntimeError("No S atoms were found for H3O residence analysis.")

	return s_indices

def calculate_time_origin_msd(unwrapped_positions, sample_dt_ps):
	n_frames = unwrapped_positions.shape[0]

	rows = []

	for lag in range(n_frames):
		if lag == 0:
			msd_xyz = np.array([0.0, 0.0, 0.0])
		else:
			displacements = unwrapped_positions[lag:] - unwrapped_positions[:-lag]
			msd_xyz = np.mean(displacements ** 2, axis=(0, 1))

		msd_total = float(np.sum(msd_xyz))

		rows.append({
			"lag_frame": lag,
			"time_ps": lag * sample_dt_ps,
			"time_ns": lag * sample_dt_ps / 1000.0,
			"msd_total_A2": msd_total,
			"msd_x_A2": float(msd_xyz[0]),
			"msd_y_A2": float(msd_xyz[1]),
			"msd_z_A2": float(msd_xyz[2]),
		})

	return pd.DataFrame(rows)

def fit_msd_diffusion(msd_df):
	fit_df = msd_df[
		(msd_df["time_ps"] >= msd_fit_start_ps) &
		(msd_df["time_ps"] <= msd_fit_end_ps)
	].copy()

	if len(fit_df) < 2:
		raise RuntimeError(
			"Not enough MSD points in fitting window. "
			f"Requested {msd_fit_start_ps}-{msd_fit_end_ps} ps, "
			f"but found only {len(fit_df)} points. "
			"Check cond_time_between_nc_frames_ps, cond_frame_start/stop/stride."
		)

	time = fit_df["time_ps"].values

	slope_total, intercept_total = np.polyfit(time, fit_df["msd_total_A2"].values, 1)
	slope_x, intercept_x = np.polyfit(time, fit_df["msd_x_A2"].values, 1)
	slope_y, intercept_y = np.polyfit(time, fit_df["msd_y_A2"].values, 1)
	slope_z, intercept_z = np.polyfit(time, fit_df["msd_z_A2"].values, 1)

	D_total_A2_ps = slope_total / 6.0
	D_x_A2_ps = slope_x / 2.0
	D_y_A2_ps = slope_y / 2.0
	D_z_A2_ps = slope_z / 2.0

	D_total_m2_s = D_total_A2_ps * angstrom2_per_ps_to_m2_per_s
	D_x_m2_s = D_x_A2_ps * angstrom2_per_ps_to_m2_per_s
	D_y_m2_s = D_y_A2_ps * angstrom2_per_ps_to_m2_per_s
	D_z_m2_s = D_z_A2_ps * angstrom2_per_ps_to_m2_per_s

	result = pd.DataFrame([{
		"fit_start_ps": msd_fit_start_ps,
		"fit_end_ps": msd_fit_end_ps,
		"num_fit_points": len(fit_df),
		"slope_total_A2_ps": slope_total,
		"slope_x_A2_ps": slope_x,
		"slope_y_A2_ps": slope_y,
		"slope_z_A2_ps": slope_z,
		"D_total_A2_ps": D_total_A2_ps,
		"D_x_A2_ps": D_x_A2_ps,
		"D_y_A2_ps": D_y_A2_ps,
		"D_z_A2_ps": D_z_A2_ps,
		"D_total_m2_s": D_total_m2_s,
		"D_x_m2_s": D_x_m2_s,
		"D_y_m2_s": D_y_m2_s,
		"D_z_m2_s": D_z_m2_s,
	}])

	return result

def log_general_settings():
	log_message("PolyProton analysis workflow")
	log_message(f"\tpolymer = {polymer}")
	log_message(f"\tchain_length = {chain_length}")
	log_message(f"\tnum_chains = {num_chains}")
	log_message(f"\tmix_chains = {mix_chains}")
	log_message(f"\tbase_dir = {base_dir}")
	log_message(f"\tsimulation_dir = {simulation_dir}")
	log_message(f"\tanalysis_dir = {analysis_dir}")
	log_message(f"\tframe_start = {frame_start}")
	log_message(f"\tframe_stop = {frame_stop}")
	log_message(f"\tframe_stride = {frame_stride}")

def log_dry_settings(system_tag, prmtop_file):
	log_message("Dry analysis settings")
	log_message(f"\tdry_eq_prot = {dry_eq_prot}")
	log_message(f"\tsystem_tag = {system_tag}")
	log_message(f"\tdry_eq_dir = {dry_eq_dir}")
	log_message(f"\tdry_analysis_dir = {dry_analysis_dir}")
	log_message(f"\tprmtop_file = {prmtop_file}")
	log_message(f"\tdry_traj_file = {dry_traj_file}")
	log_message(f"\tdry_restart_file = {dry_restart_file}")
	log_message(f"\tdry_rdf_pairs = {dry_rdf_pairs}")

def log_hyd_settings(lam, lam_tag, prmtop_file, lam_md_dir, lam_analysis_dir):
	log_message("Hydration analysis settings")
	log_message(f"\tlambda = {lam}")
	log_message(f"\tlam_tag = {lam_tag}")
	log_message(f"\thyd_analysis_lam_list = {hyd_analysis_lam_list}")
	log_message(f"\thyd_eq_dir = {hyd_eq_dir}")
	log_message(f"\tlam_md_dir = {lam_md_dir}")
	log_message(f"\tlam_analysis_dir = {lam_analysis_dir}")
	log_message(f"\tprmtop_file = {prmtop_file}")
	log_message(f"\thyd_traj_file = {hyd_traj_file}")
	log_message(f"\thyd_restart_file = {hyd_restart_file}")
	log_message(f"\thyd_rdf_pairs = {hyd_rdf_pairs}")
	log_message(f"\twater_oo_cutoff = {water_oo_cutoff}")

def log_cond_settings(lam, lam_tag, prmtop_file, lam_md_dir, lam_analysis_dir):
	log_message("Conductivity analysis settings")
	log_message(f"\tlambda = {lam}")
	log_message(f"\tlam_tag = {lam_tag}")
	log_message(f"\tcond_analysis_lam_list = {cond_analysis_lam_list}")
	log_message(f"\tcond_pr_dir = {cond_pr_dir}")
	log_message(f"\tlam_md_dir = {lam_md_dir}")
	log_message(f"\tlam_analysis_dir = {lam_analysis_dir}")
	log_message(f"\tprmtop_file = {prmtop_file}")
	log_message(f"\tcond_traj_file = {cond_traj_file}")
	log_message(f"\tcond_restart_file = {cond_restart_file}")
	log_message(f"\tcond_frame_start = {cond_frame_start}")
	log_message(f"\tcond_frame_stop = {cond_frame_stop}")
	log_message(f"\tcond_frame_stride = {cond_frame_stride}")
	log_message(f"\tcond_time_between_nc_frames_ps = {cond_time_between_nc_frames_ps}")
	log_message(f"\tmsd_fit_start_ps = {msd_fit_start_ps}")
	log_message(f"\tmsd_fit_end_ps = {msd_fit_end_ps}")
	log_message(f"\tcond_temperature_K = {cond_temperature_K}")
	log_message(f"\th3o_s_residence_cutoff = {h3o_s_residence_cutoff}")

def get_pdb_box_lengths(pdb_file):
	with open(pdb_file, "r") as f:
		for line in f:
			if line.startswith("CRYST1"):
				return np.array([
					float(line[6:15]),
					float(line[15:24]),
					float(line[24:33]),
				], dtype=float)

	raise RuntimeError(f"CRYST1 record not found in {pdb_file}")

def iter_pdb_frames(pdb_file):
	current_positions = []
	current_box = None
	last_box = None

	with open(pdb_file, "r") as f:
		for line in f:

			if line.startswith("CRYST1"):
				last_box = np.array([
					float(line[6:15]),
					float(line[15:24]),
					float(line[24:33]),
				], dtype=float)
				current_box = last_box.copy()
				continue

			if line.startswith("MODEL"):
				current_positions = []
				if last_box is not None:
					current_box = last_box.copy()
				continue

			if line.startswith(("ATOM", "HETATM")):
				current_positions.append([
					float(line[30:38]),
					float(line[38:46]),
					float(line[46:54]),
				])
				continue

			if line.startswith(("ENDMDL", "END")):
				if current_positions:
					yield (
						np.asarray(current_positions, dtype=float),
						None if current_box is None else current_box.copy()
					)
					current_positions = []

	if current_positions:
		yield (
			np.asarray(current_positions, dtype=float),
			None if current_box is None else current_box.copy()
		)

# ====
# Dry analysis classes
# ====

class Dry_TrajectoryPreparation:
	def __init__(self, prmtop_file):
		self.prmtop_file = prmtop_file
		self.traj_file = os.path.join(dry_eq_dir, dry_traj_file)
		self.restart_file = os.path.join(dry_eq_dir, dry_restart_file)

		self.last_pdb = os.path.join(dry_analysis_dir, "dry_last.pdb")
		self.sampled_pdb = os.path.join(dry_analysis_dir, "dry_pr-nvt_sampled.pdb")
		self.cpptraj_input = os.path.join(dry_analysis_dir, "cpptraj_dry_analysis.in")
		self.tmp_prefix = os.path.join(dry_analysis_dir, "dry_pr-nvt_tmp")

	def create_last_pdb(self):
		if os.path.exists(self.last_pdb) and not overwrite:
			log_message("\tDry last PDB already exists. Skipping ambpdb.")
			return self.last_pdb

		ambpdb_command = f"ambpdb -p {self.prmtop_file} -c {self.restart_file} > {self.last_pdb}"
		run_command(ambpdb_command)

		log_message("\tDry last PDB created")
		return self.last_pdb

	def create_sampled_pdb(self):
		if os.path.exists(self.sampled_pdb) and not overwrite:
			log_message("\tDry sampled PDB already exists. Skipping cpptraj.")
			return self.sampled_pdb

		for old_file in glob.glob(f"{self.tmp_prefix}*"):
			os.remove(old_file)

		with open(self.cpptraj_input, "w") as f:
			f.write(f"trajin {self.traj_file} {frame_start} {frame_stop} {frame_stride}\n")
			f.write("autoimage\n")
			f.write(f"trajout {self.tmp_prefix}.pdb pdb multi\n")

		cpptraj_command = f"cpptraj -i {self.cpptraj_input} -p {self.prmtop_file}"
		run_command(cpptraj_command)

		tmp_files = sorted(
			glob.glob(f"{self.tmp_prefix}.pdb*"),
			key=lambda x: int(x.split(".")[-1]) if x.split(".")[-1].isdigit() else 0
		)

		if len(tmp_files) == 0:
			raise RuntimeError(
				"cpptraj did not create sampled PDB files.\n"
				f"Expected files matching: {self.tmp_prefix}.pdb*\n"
				f"Check cpptraj input file: {self.cpptraj_input}"
			)

		with open(self.sampled_pdb, "w") as outfile:
			for pdb_file in tmp_files:
				with open(pdb_file, "r") as infile:
					shutil.copyfileobj(infile, outfile)

		for pdb_file in tmp_files:
			os.remove(pdb_file)

		log_message("\tDry sampled PDB created")
		return self.sampled_pdb

class Dry_RDFAnalysis:
	def __init__(self, sampled_pdb, system_tag):
		self.sampled_pdb = sampled_pdb
		self.system_tag = system_tag

	def calculate_rdf(self, pair):
		settings = rdf_settings[pair]

		rdf_csv = os.path.join(dry_analysis_dir, f"rdf_{pair}.csv")
		rdf_png = os.path.join(dry_analysis_dir, f"rdf_{pair}.png")

		pipeline = import_file(self.sampled_pdb)
		pipeline.modifiers.append(
			CoordinationAnalysisModifier(
				cutoff=settings["cutoff"],
				number_of_bins=settings["bins"],
				partial=True
			)
		)

		r_values = None
		y_sum = None
		num_frames = 0

		possible_pair_names = [pair, "-".join(pair.split("-")[::-1])]
		found_pair_name = None

		for frame in range(pipeline.source.num_frames):
			rdf_table = pipeline.compute(frame).tables["coordination-rdf"]

			rdf_names = list(rdf_table.y.component_names)
			table_x = np.array([row[0] for row in rdf_table.xy()], dtype=float)

			component_idx = None

			for i, name in enumerate(rdf_names):
				if name in possible_pair_names:
					component_idx = i
					found_pair_name = name
					break

			if component_idx is None:
				raise ValueError(
					f"RDF pair {pair} was not found. Available RDF components: {rdf_names}"
				)

			y = np.array(rdf_table.y[:, component_idx], dtype=float)

			if y_sum is None:
				y_sum = np.zeros_like(y)
				r_values = table_x

			y_sum += y
			num_frames += 1

		if num_frames == 0:
			raise RuntimeError(f"No frames found in sampled PDB: {self.sampled_pdb}")

		y_avg = y_sum / num_frames

		df = pd.DataFrame({
			"r_A": r_values,
			pair: y_avg
		})

		df.to_csv(rdf_csv, index=False)

		plt.figure(figsize=(7, 5))
		plt.plot(df["r_A"], df[pair])
		plt.xlabel("r [A]")
		plt.ylabel(f"g$_{{{pair}}}$(r)")
		plt.title(f"{self.system_tag}: {pair} RDF")
		plt.tight_layout()
		plt.savefig(rdf_png, dpi=300)
		plt.close()

		if found_pair_name != pair:
			log_message(f"\tRequested {pair}, OVITO used {found_pair_name}")

		log_message(f"\t{pair} RDF calculated")
		return rdf_csv

	def find_global_rdf_peak(self, rdf_csv, pair):
		df = pd.read_csv(rdf_csv)
		idx = df[pair].idxmax()

		r_peak = float(df.loc[idx, "r_A"])
		g_peak = float(df.loc[idx, pair])

		return r_peak, g_peak

	def find_rdf_peak_in_range(self, rdf_csv, pair, r_min, r_max):
		df = pd.read_csv(rdf_csv)

		filtered_df = df[
			(df["r_A"] >= r_min) &
			(df["r_A"] <= r_max)
		]

		if len(filtered_df) == 0:
			log_message(
				f"\tWarning: no {pair} RDF points found between "
				f"{r_min:.3f} and {r_max:.3f} A. Falling back to global peak."
			)
			return self.find_global_rdf_peak(rdf_csv, pair)

		idx = filtered_df[pair].idxmax()

		r_peak = float(df.loc[idx, "r_A"])
		g_peak = float(df.loc[idx, pair])

		return r_peak, g_peak

	def run_all_rdf(self, nn_mean=None, nn_std=None):
		rdf_peak_results = {}

		for pair in dry_rdf_pairs:
			rdf_csv = self.calculate_rdf(pair)

			if pair == "N-N" and nn_mean is not None and nn_std is not None:
				r_min = nn_mean - nn_std
				r_max = nn_mean + nn_std

				r_peak, g_peak = self.find_rdf_peak_in_range(
					rdf_csv,
					pair,
					r_min,
					r_max
				)

				log_message(
					f"\t{pair} RDF monomer-unit peak searched in "
					f"{r_min:.3f}–{r_max:.3f} A"
				)

			else:
				r_peak, g_peak = self.find_global_rdf_peak(rdf_csv, pair)

			rdf_peak_results[pair] = (r_peak, g_peak)

			log_message(f"\t{pair} RDF peak: r = {r_peak:.3f} A, g = {g_peak:.3f}")

		return rdf_peak_results

class Dry_DensityAnalysis:
	def __init__(self, last_pdb):
		self.last_pdb = last_pdb

	def calculate_density(self):
		atoms = read(self.last_pdb)

		volume_a3 = atoms.get_volume()
		if volume_a3 <= 0:
			return np.nan

		total_mass_amu = sum(
			molecular_weights.get(atom.symbol, 0.0)
			for atom in atoms
		)

		density = (total_mass_amu * amu_to_g) / (volume_a3 * angstrom3_to_cm3)

		log_message(f"\tDry density calculated: {density:.3f} g/cm3")
		return float(density)

class Dry_MonomerAnalysis:
	def __init__(self, last_pdb):
		self.last_pdb = last_pdb

	def calculate_monomer_nn_distance(self):
		chain_blocks = load_chain_line_blocks_from_first_model(self.last_pdb)

		all_distances = []

		for chain_lines in chain_blocks:
			chain_atoms = atoms_from_pdb_lines(chain_lines)
			first_n_indices = get_first_n_per_residue(chain_atoms, chain_lines)

			for i in range(len(first_n_indices) - 1):
				d = chain_atoms.get_distance(
					first_n_indices[i],
					first_n_indices[i + 1],
					mic=False
				)
				all_distances.append(float(d))

		if len(all_distances) == 0:
			return np.nan, np.nan

		mean_dist = float(np.mean(all_distances))
		std_dist = float(np.std(all_distances, ddof=0))

		log_message(f"\tMonomer N-N distance mean: {mean_dist:.3f} A")
		log_message(f"\tMonomer N-N distance std: {std_dist:.3f} A")

		return mean_dist, std_dist

class Dry_NematicOrderAnalysis:
	def __init__(self, sampled_pdb, system_tag):
		self.sampled_pdb = sampled_pdb
		self.system_tag = system_tag

	def calculate_s2(self):
		monomer_n_pairs = get_monomer_unit_n_atom_pairs(self.sampled_pdb)

		frames = read(self.sampled_pdb, index=":")
		if not isinstance(frames, list):
			frames = [frames]

		rows = []

		for frame_i, atoms in enumerate(frames):
			vectors = []

			for first_idx, last_idx in monomer_n_pairs:
				try:
					vec = atoms.get_distance(
						first_idx,
						last_idx,
						mic=True,
						vector=True
					)
				except Exception:
					vec = atoms.positions[last_idx] - atoms.positions[first_idx]

				norm = np.linalg.norm(vec)

				if norm > 1.0e-12:
					vectors.append(vec / norm)

			S2 = nematic_order_from_unit_vectors(vectors)

			rows.append({
				"frame": frame_i,
				"S2": S2,
				"num_monomer_unit_vectors": len(vectors)
			})

		df = pd.DataFrame(rows)

		s2_csv = os.path.join(dry_analysis_dir, "s2_nematic.csv")
		s2_png = os.path.join(dry_analysis_dir, "s2_nematic.png")

		df.to_csv(s2_csv, index=False)

		plt.figure(figsize=(7, 5))
		plt.plot(df["frame"], df["S2"], marker="o")
		plt.xlabel("sampled frame index")
		plt.ylabel("S2")
		plt.title(f"{self.system_tag}: dry monomer-unit nematic order")
		plt.tight_layout()
		plt.savefig(s2_png, dpi=300)
		plt.close()

		S2_mean = float(df["S2"].mean()) if len(df) > 0 else np.nan
		S2_std = float(df["S2"].std(ddof=0)) if len(df) > 1 else 0.0

		log_message(f"\tDry monomer-unit S2 mean: {S2_mean:.3f}")
		log_message(f"\tDry monomer-unit S2 std: {S2_std:.3f}")

		return S2_mean, S2_std

class Dry_SummaryWriter:
	def __init__(self, system_tag):
		self.system_tag = system_tag

	def write_summary(
		self,
		density,
		nn_mean,
		nn_std,
		S2_mean,
		S2_std,
		rdf_peak_results
	):
		row = {
			"system_tag": self.system_tag,
			"polymer": polymer,
			"stage": "dry",
			"chain_length": chain_length,
			"num_chains": num_chains,
			"mix_chains": mix_chains,
			"dry_eq_prot": dry_eq_prot,
			"trajectory": dry_traj_file,
			"restart": dry_restart_file,
			"frame_start": frame_start,
			"frame_stop": frame_stop,
			"frame_stride": frame_stride,
			"density_g_cm3": density,
			"S2_mean": S2_mean,
			"S2_std": S2_std,
			"N_N_monomer_distance_mean_A": nn_mean,
			"N_N_monomer_distance_std_A": nn_std,
		}

		for pair, values in rdf_peak_results.items():
			r_peak, g_peak = values
			key = pair.replace("-", "_")
			row[f"r_{key}_peak_A"] = r_peak
			row[f"g_{key}_peak"] = g_peak

		df = pd.DataFrame([row])

		summary_csv = os.path.join(dry_analysis_dir, "dry_summary.csv")
		df.to_csv(summary_csv, index=False)

		log_message("\tDry summary written")
		return df

# ====
# Hydration analysis classes
# ====

class Hyd_TrajectoryPreparation:
	def __init__(self, lam, prmtop_file):
		self.lam = lam
		self.prmtop_file = prmtop_file

		self.lam_dir, self.lam_md_dir, self.lam_analysis_dir = get_lam_hyd_dirs(lam)

		self.traj_file = os.path.join(self.lam_md_dir, hyd_traj_file)
		self.restart_file = os.path.join(self.lam_md_dir, hyd_restart_file)

		self.last_pdb = os.path.join(self.lam_analysis_dir, "hyd_last.pdb")
		self.sampled_pdb = os.path.join(self.lam_analysis_dir, "hyd_pr-nvt_sampled.pdb")
		self.rdf_sampled_pdb = os.path.join(self.lam_analysis_dir, "hyd_pr-nvt_sampled_rdf_types.pdb")

		self.cpptraj_input = os.path.join(self.lam_analysis_dir, "cpptraj_hyd_analysis.in")
		self.tmp_prefix = os.path.join(self.lam_analysis_dir, "hyd_pr-nvt_tmp")

	def create_last_pdb(self):
		if os.path.exists(self.last_pdb) and not overwrite:
			log_message(f"\tHyd lambda={self.lam} last PDB already exists. Skipping ambpdb.")
			return self.last_pdb

		ambpdb_command = f"ambpdb -p {self.prmtop_file} -c {self.restart_file} > {self.last_pdb}"
		run_command(ambpdb_command)

		log_message(f"\tHyd lambda={self.lam} last PDB created")
		return self.last_pdb

	def create_sampled_pdb(self):
		if os.path.exists(self.sampled_pdb) and not overwrite:
			log_message(f"\tHyd lambda={self.lam} sampled PDB already exists. Skipping cpptraj.")
			return self.sampled_pdb

		for old_file in glob.glob(f"{self.tmp_prefix}*"):
			os.remove(old_file)

		with open(self.cpptraj_input, "w") as f:
			f.write(f"trajin {self.traj_file} {frame_start} {frame_stop} {frame_stride}\n")
			f.write("autoimage\n")
			f.write(f"trajout {self.tmp_prefix}.pdb pdb multi\n")

		cpptraj_command = f"cpptraj -i {self.cpptraj_input} -p {self.prmtop_file}"
		run_command(cpptraj_command)

		tmp_files = sorted(
			glob.glob(f"{self.tmp_prefix}.pdb*"),
			key=lambda x: int(x.split(".")[-1]) if x.split(".")[-1].isdigit() else 0
		)

		if len(tmp_files) == 0:
			raise RuntimeError(
				"cpptraj did not create sampled hydration PDB files.\n"
				f"Expected files matching: {self.tmp_prefix}.pdb*\n"
				f"Check cpptraj input file: {self.cpptraj_input}"
			)

		with open(self.sampled_pdb, "w") as outfile:
			for pdb_file in tmp_files:
				with open(pdb_file, "r") as infile:
					shutil.copyfileobj(infile, outfile)

		for pdb_file in tmp_files:
			os.remove(pdb_file)

		log_message(f"\tHyd lambda={self.lam} sampled PDB created")
		return self.sampled_pdb

	def create_rdf_type_pdb(self):
		write_pdb_with_water_oxygen_type(self.sampled_pdb, self.rdf_sampled_pdb)
		log_message(f"\tHyd lambda={self.lam} RDF-type PDB created")
		return self.rdf_sampled_pdb

class Hyd_RDFAnalysis:
	def __init__(self, lam, sampled_pdb_for_rdf, system_tag, lam_analysis_dir):
		self.lam = lam
		self.sampled_pdb_for_rdf = sampled_pdb_for_rdf
		self.system_tag = system_tag
		self.lam_analysis_dir = lam_analysis_dir

	def calculate_rdf(self, pair):
		settings = hyd_rdf_settings[pair]

		rdf_csv = os.path.join(self.lam_analysis_dir, f"rdf_{pair}.csv")
		rdf_png = os.path.join(self.lam_analysis_dir, f"rdf_{pair}.png")

		pipeline = import_file(self.sampled_pdb_for_rdf)
		pipeline.modifiers.append(
			CoordinationAnalysisModifier(
				cutoff=settings["cutoff"],
				number_of_bins=settings["bins"],
				partial=True
			)
		)

		r_values = None
		y_sum = None
		num_frames = 0

		possible_pair_names = [pair, "-".join(pair.split("-")[::-1])]
		found_pair_name = None

		for frame in range(pipeline.source.num_frames):
			rdf_table = pipeline.compute(frame).tables["coordination-rdf"]

			rdf_names = list(rdf_table.y.component_names)
			table_x = np.array([row[0] for row in rdf_table.xy()], dtype=float)

			component_idx = None

			for i, name in enumerate(rdf_names):
				if name in possible_pair_names:
					component_idx = i
					found_pair_name = name
					break

			if component_idx is None:
				raise ValueError(
					f"Hyd lambda={self.lam}: RDF pair {pair} was not found. "
					f"Available RDF components: {rdf_names}"
				)

			y = np.array(rdf_table.y[:, component_idx], dtype=float)

			if y_sum is None:
				y_sum = np.zeros_like(y)
				r_values = table_x

			y_sum += y
			num_frames += 1

		if num_frames == 0:
			raise RuntimeError(f"No frames found in sampled PDB: {self.sampled_pdb_for_rdf}")

		y_avg = y_sum / num_frames

		df = pd.DataFrame({
			"r_A": r_values,
			pair: y_avg
		})

		df.to_csv(rdf_csv, index=False)

		plt.figure(figsize=(7, 5))
		plt.plot(df["r_A"], df[pair])
		plt.xlabel("r [A]")
		plt.ylabel(f"g$_{{{pair}}}$(r)")
		plt.title(f"{self.system_tag}: {pair} RDF")
		plt.tight_layout()
		plt.savefig(rdf_png, dpi=300)
		plt.close()

		if found_pair_name != pair:
			log_message(f"\tHyd lambda={self.lam}: requested {pair}, OVITO used {found_pair_name}")

		log_message(f"\tHyd lambda={self.lam}: {pair} RDF calculated")
		return rdf_csv

	def find_global_rdf_peak(self, rdf_csv, pair):
		df = pd.read_csv(rdf_csv)
		idx = df[pair].idxmax()

		r_peak = float(df.loc[idx, "r_A"])
		g_peak = float(df.loc[idx, pair])

		return r_peak, g_peak

	def find_rdf_peak_in_range(self, rdf_csv, pair, r_min, r_max):
		df = pd.read_csv(rdf_csv)

		filtered_df = df[
			(df["r_A"] >= r_min) &
			(df["r_A"] <= r_max)
		]

		if len(filtered_df) == 0:
			log_message(
				f"\tHyd lambda={self.lam}: warning no {pair} RDF points found between "
				f"{r_min:.3f} and {r_max:.3f} A. Falling back to global peak."
			)
			return self.find_global_rdf_peak(rdf_csv, pair)

		idx = filtered_df[pair].idxmax()

		r_peak = float(df.loc[idx, "r_A"])
		g_peak = float(df.loc[idx, pair])

		return r_peak, g_peak

	def run_all_rdf(self, nn_mean=None, nn_std=None):
		rdf_peak_results = {}

		for pair in hyd_rdf_pairs:
			rdf_csv = self.calculate_rdf(pair)

			if pair == "N-N" and nn_mean is not None and nn_std is not None:
				r_min = nn_mean - nn_std
				r_max = nn_mean + nn_std

				r_peak, g_peak = self.find_rdf_peak_in_range(
					rdf_csv,
					pair,
					r_min,
					r_max
				)

				log_message(
					f"\tHyd lambda={self.lam}: {pair} RDF monomer-unit peak searched in "
					f"{r_min:.3f}-{r_max:.3f} A"
				)

			else:
				r_peak, g_peak = self.find_global_rdf_peak(rdf_csv, pair)

			rdf_peak_results[pair] = (r_peak, g_peak)

			log_message(
				f"\tHyd lambda={self.lam}: {pair} RDF peak: "
				f"r = {r_peak:.3f} A, g = {g_peak:.3f}"
			)

		return rdf_peak_results

class Hyd_DensityAnalysis:
	def __init__(self, last_pdb):
		self.last_pdb = last_pdb

	def calculate_density(self):
		box = get_pdb_box_lengths(self.last_pdb)
		volume_a3 = float(np.prod(box))

		total_mass_amu = 0.0

		with open(self.last_pdb, "r") as f:
			for line in f:
				if not is_atom_line(line):
					continue

				element = element_from_pdb_line(line)
				total_mass_amu += molecular_weights.get(element, 0.0)

		density = (total_mass_amu * amu_to_g) / (volume_a3 * angstrom3_to_cm3)

		log_message(f"\tHyd density calculated: {density:.3f} g/cm3")
		return float(density)

class Hyd_MonomerAnalysis:
	def __init__(self, last_pdb):
		self.last_pdb = last_pdb

	def calculate_monomer_nn_distance(self):
		chain_blocks = load_chain_line_blocks_from_first_model(self.last_pdb)

		all_distances = []

		for chain_lines in chain_blocks:
			chain_atoms = atoms_from_pdb_lines(chain_lines)
			first_n_indices = get_first_n_per_residue(chain_atoms, chain_lines)

			for i in range(len(first_n_indices) - 1):
				d = chain_atoms.get_distance(
					first_n_indices[i],
					first_n_indices[i + 1],
					mic=False
				)
				all_distances.append(float(d))

		if len(all_distances) == 0:
			return np.nan, np.nan

		mean_dist = float(np.mean(all_distances))
		std_dist = float(np.std(all_distances, ddof=0))

		log_message(f"\tHyd monomer N-N distance mean: {mean_dist:.3f} A")
		log_message(f"\tHyd monomer N-N distance std: {std_dist:.3f} A")

		return mean_dist, std_dist

class Hyd_NematicOrderAnalysis:
	def __init__(self, sampled_pdb, system_tag, lam_analysis_dir):
		self.sampled_pdb = sampled_pdb
		self.system_tag = system_tag
		self.lam_analysis_dir = lam_analysis_dir

	def calculate_s2(self):
		monomer_n_pairs = get_monomer_unit_n_atom_pairs(self.sampled_pdb)

		rows = []
		
		for frame_i, (positions, box) in enumerate(
			iter_pdb_frames(self.sampled_pdb)
		):
			vectors = []
		
			for first_idx, last_idx in monomer_n_pairs:
				vec = positions[last_idx] - positions[first_idx]
		
				if box is not None:
					vec -= box * np.round(vec / box)
		
				norm = np.linalg.norm(vec)
		
				if norm > 1.0e-12:
					vectors.append(vec / norm)
		
			S2 = nematic_order_from_unit_vectors(vectors)
		
			rows.append({
				"frame": frame_i,
				"S2": S2,
				"num_monomer_unit_vectors": len(vectors)
			})

		df = pd.DataFrame(rows)

		s2_csv = os.path.join(self.lam_analysis_dir, "s2_nematic.csv")
		s2_png = os.path.join(self.lam_analysis_dir, "s2_nematic.png")

		df.to_csv(s2_csv, index=False)

		plt.figure(figsize=(7, 5))
		plt.plot(df["frame"], df["S2"], marker="o")
		plt.xlabel("sampled frame index")
		plt.ylabel("S2")
		plt.title(f"{self.system_tag}: hydrated monomer-unit nematic order")
		plt.tight_layout()
		plt.savefig(s2_png, dpi=300)
		plt.close()

		S2_mean = float(df["S2"].mean()) if len(df) > 0 else np.nan
		S2_std = float(df["S2"].std(ddof=0)) if len(df) > 1 else 0.0

		log_message(f"\tHyd monomer-unit S2 mean: {S2_mean:.3f}")
		log_message(f"\tHyd monomer-unit S2 std: {S2_std:.3f}")

		return S2_mean, S2_std

class Hyd_WaterClusterAnalysis:
	def __init__(self, sampled_pdb, lam, lam_analysis_dir):
		self.sampled_pdb = sampled_pdb
		self.lam = lam
		self.lam_analysis_dir = lam_analysis_dir

	def calculate_water_clusters(self):
		water_o_indices = get_water_oxygen_indices_from_first_model(self.sampled_pdb)

		rows = []

		for frame_i, (positions, box_lengths) in enumerate(
			iter_pdb_frames(self.sampled_pdb)
		):
			water_o_positions = positions[water_o_indices]
		
			cluster_sizes = calculate_water_cluster_sizes(
				water_o_positions,
				box_lengths,
				water_oo_cutoff
			)

			num_water = len(water_o_indices)
			num_clusters = len(cluster_sizes)

			largest_cluster_size = cluster_sizes[0] if num_clusters > 0 else 0
			second_largest_cluster_size = cluster_sizes[1] if num_clusters > 1 else 0
			average_cluster_size = float(np.mean(cluster_sizes)) if num_clusters > 0 else 0.0
			largest_cluster_fraction = largest_cluster_size / num_water if num_water > 0 else 0.0

			rows.append({
				"frame": frame_i,
				"lambda": self.lam,
				"num_water": num_water,
				"num_water_clusters": num_clusters,
				"largest_cluster_size": largest_cluster_size,
				"largest_cluster_fraction": largest_cluster_fraction,
				"second_largest_cluster_size": second_largest_cluster_size,
				"average_cluster_size": average_cluster_size,
			})

		df = pd.DataFrame(rows)

		cluster_csv = os.path.join(self.lam_analysis_dir, "water_clusters.csv")
		df.to_csv(cluster_csv, index=False)

		summary = pd.DataFrame([{
			"lambda": self.lam,
			"water_oo_cutoff_A": water_oo_cutoff,
			"num_water_mean": float(df["num_water"].mean()),
			"num_water_clusters_mean": float(df["num_water_clusters"].mean()),
			"num_water_clusters_std": float(df["num_water_clusters"].std(ddof=0)),
			"largest_water_cluster_size_mean": float(df["largest_cluster_size"].mean()),
			"largest_water_cluster_size_std": float(df["largest_cluster_size"].std(ddof=0)),
			"largest_water_cluster_fraction_mean": float(df["largest_cluster_fraction"].mean()),
			"largest_water_cluster_fraction_std": float(df["largest_cluster_fraction"].std(ddof=0)),
			"second_largest_water_cluster_size_mean": float(df["second_largest_cluster_size"].mean()),
			"average_water_cluster_size_mean": float(df["average_cluster_size"].mean()),
		}])

		summary_csv = os.path.join(self.lam_analysis_dir, "water_cluster_summary.csv")
		summary.to_csv(summary_csv, index=False)

		log_message(
			f"\tHyd lambda={self.lam}: water cluster analysis finished. "
			f"Largest fraction mean = {summary.loc[0, 'largest_water_cluster_fraction_mean']:.3f}"
		)

		return df, summary

class Hyd_SummaryWriter:
	def __init__(self, lam, system_tag, lam_analysis_dir):
		self.lam = lam
		self.system_tag = system_tag
		self.lam_analysis_dir = lam_analysis_dir

	def write_summary(
		self,
		density,
		nn_mean,
		nn_std,
		S2_mean,
		S2_std,
		rdf_peak_results,
		water_cluster_summary
	):
		row = {
			"system_tag": self.system_tag,
			"polymer": polymer,
			"stage": "hydrated",
			"lambda": self.lam,
			"chain_length": chain_length,
			"num_chains": num_chains,
			"mix_chains": mix_chains,
			"trajectory": hyd_traj_file,
			"restart": hyd_restart_file,
			"frame_start": frame_start,
			"frame_stop": frame_stop,
			"frame_stride": frame_stride,
			"density_g_cm3": density,
			"S2_mean": S2_mean,
			"S2_std": S2_std,
			"N_N_monomer_distance_mean_A": nn_mean,
			"N_N_monomer_distance_std_A": nn_std,
		}

		for pair, values in rdf_peak_results.items():
			r_peak, g_peak = values
			key = pair.replace("-", "_")
			row[f"r_{key}_peak_A"] = r_peak
			row[f"g_{key}_peak"] = g_peak

		if water_cluster_summary is not None and len(water_cluster_summary) > 0:
			for col in water_cluster_summary.columns:
				if col != "lambda":
					row[col] = water_cluster_summary.loc[0, col]

		df = pd.DataFrame([row])

		summary_csv = os.path.join(self.lam_analysis_dir, "hyd_summary.csv")
		df.to_csv(summary_csv, index=False)

		log_message(f"\tHyd lambda={self.lam}: hyd_summary.csv written")
		return df

# ====
# Conductivity analysis classes
# ====

class Cond_TrajectoryPreparation:
	def __init__(self, lam, prmtop_file):
		self.lam = lam
		self.prmtop_file = prmtop_file

		self.lam_dir, self.lam_md_dir, self.lam_analysis_dir = get_lam_cond_dirs(lam)

		self.traj_file = os.path.join(self.lam_md_dir, cond_traj_file)
		self.restart_file = os.path.join(self.lam_md_dir, cond_restart_file)

		self.last_pdb = os.path.join(self.lam_analysis_dir, "cond_last.pdb")

		# Full sampled trajectory, used for residence analysis
		self.sampled_pdb = os.path.join(self.lam_analysis_dir, "cond_pr-nvt_sampled.pdb")

		# H3O-only unwrapped trajectory, used for MSD / D / conductivity
		self.h3o_unwrapped_pdb = os.path.join(self.lam_analysis_dir, "cond_pr-nvt_h3o_unwrapped.pdb")

		self.cpptraj_input = os.path.join(self.lam_analysis_dir, "cpptraj_cond_analysis.in")
		self.cpptraj_h3o_input = os.path.join(self.lam_analysis_dir, "cpptraj_cond_h3o_unwrap.in")

		self.tmp_prefix = os.path.join(self.lam_analysis_dir, "cond_pr-nvt_tmp")
		self.h3o_tmp_prefix = os.path.join(self.lam_analysis_dir, "cond_pr-nvt_h3o_tmp")

	def create_last_pdb(self):
		if os.path.exists(self.last_pdb) and not overwrite:
			log_message(f"\tCond lambda={self.lam} last PDB already exists. Skipping ambpdb.")
			return self.last_pdb

		ambpdb_command = f"ambpdb -p {self.prmtop_file} -c {self.restart_file} > {self.last_pdb}"
		run_command(ambpdb_command)

		log_message(f"\tCond lambda={self.lam} last PDB created")
		return self.last_pdb

	def create_sampled_pdb(self):
		if os.path.exists(self.sampled_pdb) and not overwrite:
			log_message(f"\tCond lambda={self.lam} sampled PDB already exists. Skipping cpptraj.")
			return self.sampled_pdb

		for old_file in glob.glob(f"{self.tmp_prefix}*"):
			os.remove(old_file)

		with open(self.cpptraj_input, "w") as f:
			f.write(f"trajin {self.traj_file} {cond_frame_start} {cond_frame_stop} {cond_frame_stride}\n")
			f.write("autoimage\n")
			f.write(f"trajout {self.tmp_prefix}.pdb pdb multi\n")

		cpptraj_command = f"cpptraj -i {self.cpptraj_input} -p {self.prmtop_file}"
		run_command(cpptraj_command)

		tmp_files = sorted(
			glob.glob(f"{self.tmp_prefix}.pdb*"),
			key=lambda x: int(x.split(".")[-1]) if x.split(".")[-1].isdigit() else 0
		)

		if len(tmp_files) == 0:
			raise RuntimeError(
				"cpptraj did not create sampled conductivity PDB files.\n"
				f"Expected files matching: {self.tmp_prefix}.pdb*\n"
				f"Check cpptraj input file: {self.cpptraj_input}"
			)

		with open(self.sampled_pdb, "w") as outfile:
			for pdb_file in tmp_files:
				with open(pdb_file, "r") as infile:
					shutil.copyfileobj(infile, outfile)

		for pdb_file in tmp_files:
			os.remove(pdb_file)

		log_message(f"\tCond lambda={self.lam} sampled PDB created")
		return self.sampled_pdb

	def create_h3o_unwrapped_pdb(self):
		if os.path.exists(self.h3o_unwrapped_pdb) and not overwrite:
			log_message(f"\tCond lambda={self.lam} H3O unwrapped PDB already exists. Skipping cpptraj.")
			return self.h3o_unwrapped_pdb

		for old_file in glob.glob(f"{self.h3o_tmp_prefix}*"):
			os.remove(old_file)

		with open(self.cpptraj_h3o_input, "w") as f:
			f.write(f"trajin {self.traj_file} {cond_frame_start} {cond_frame_stop} {cond_frame_stride}\n")
			f.write("autoimage\n")
			f.write("strip !:H3O\n")
			f.write("unwrap :H3O\n")
			f.write(f"trajout {self.h3o_tmp_prefix}.pdb pdb multi\n")

		cpptraj_command = f"cpptraj -i {self.cpptraj_h3o_input} -p {self.prmtop_file}"
		run_command(cpptraj_command)

		tmp_files = sorted(
			glob.glob(f"{self.h3o_tmp_prefix}.pdb*"),
			key=lambda x: int(x.split(".")[-1]) if x.split(".")[-1].isdigit() else 0
		)

		if len(tmp_files) == 0:
			raise RuntimeError(
				"cpptraj did not create H3O unwrapped PDB files.\n"
				f"Expected files matching: {self.h3o_tmp_prefix}.pdb*\n"
				f"Check cpptraj input file: {self.cpptraj_h3o_input}"
			)

		with open(self.h3o_unwrapped_pdb, "w") as outfile:
			for pdb_file in tmp_files:
				with open(pdb_file, "r") as infile:
					shutil.copyfileobj(infile, outfile)

		for pdb_file in tmp_files:
			os.remove(pdb_file)

		log_message(f"\tCond lambda={self.lam} H3O unwrapped PDB created")
		return self.h3o_unwrapped_pdb

class Cond_H3OMSDAnalysis:
	def __init__(self, lam, h3o_unwrapped_pdb, lam_analysis_dir):
		self.lam = lam
		self.h3o_unwrapped_pdb = h3o_unwrapped_pdb
		self.lam_analysis_dir = lam_analysis_dir

	def load_h3o_positions(self):
		h3o_o_indices = get_h3o_oxygen_indices_from_first_model(
			self.h3o_unwrapped_pdb
		)
	
		num_h3o = len(h3o_o_indices)
	
		if num_h3o == 0:
			raise RuntimeError(
				"No H3O oxygen atoms found in H3O unwrapped PDB."
			)
	
		positions = []
		volumes = []
	
		for frame_i, (frame_positions, box) in enumerate(
			iter_pdb_frames(self.h3o_unwrapped_pdb)
		):
			if box is None:
				raise RuntimeError(
					f"No periodic box information found for "
					f"H3O frame {frame_i}."
				)
	
			if max(h3o_o_indices) >= len(frame_positions):
				raise RuntimeError(
					f"H3O atom indices do not match frame {frame_i}: "
					f"{len(frame_positions)} atoms found."
				)
	
			h3o_positions = frame_positions[h3o_o_indices]
	
			if len(h3o_positions) != num_h3o:
				raise RuntimeError(
					f"H3O count changed in frame {frame_i}: "
					f"expected {num_h3o}, found {len(h3o_positions)}."
				)
	
			positions.append(h3o_positions)
			volumes.append(float(np.prod(box)))
	
		if len(positions) == 0:
			raise RuntimeError(
				"No frames found in H3O unwrapped PDB."
			)
	
		positions = np.stack(positions, axis=0)
		volumes = np.asarray(volumes, dtype=float)
	
		return positions, volumes, num_h3o

	def calculate_msd(self):
		positions, volumes, num_h3o = self.load_h3o_positions()

		sample_dt_ps = cond_time_between_nc_frames_ps * cond_frame_stride

		# H3O positions are already unwrapped by cpptraj.
		msd_df = calculate_time_origin_msd(positions, sample_dt_ps)

		msd_csv = os.path.join(self.lam_analysis_dir, "h3o_msd.csv")
		msd_png = os.path.join(self.lam_analysis_dir, "h3o_msd.png")

		msd_df.to_csv(msd_csv, index=False)

		plt.figure(figsize=(7, 5))
		plt.plot(msd_df["time_ns"], msd_df["msd_total_A2"], label="total")
		plt.plot(msd_df["time_ns"], msd_df["msd_x_A2"], label="x")
		plt.plot(msd_df["time_ns"], msd_df["msd_y_A2"], label="y")
		plt.plot(msd_df["time_ns"], msd_df["msd_z_A2"], label="z")
		plt.xlabel("time [ns]")
		plt.ylabel("MSD [A$^2$]")
		plt.legend()
		plt.tight_layout()
		plt.savefig(msd_png, dpi=300)
		plt.close()

		diffusion_df = fit_msd_diffusion(msd_df)
		diffusion_csv = os.path.join(self.lam_analysis_dir, "h3o_diffusion.csv")
		diffusion_df.to_csv(diffusion_csv, index=False)

		msdc_df = pd.DataFrame({
			"time_ps": msd_df["time_ps"],
			"time_ns": msd_df["time_ns"],
			"MSDc_A2": msd_df["msd_z_A2"],
		})
		msdc_csv = os.path.join(self.lam_analysis_dir, "h3o_msdc.csv")
		msdc_df.to_csv(msdc_csv, index=False)

		eta_df = pd.DataFrame({
			"time_ps": msd_df["time_ps"],
			"time_ns": msd_df["time_ns"],
			"eta_c": np.where(
				msd_df["msd_total_A2"] > 0,
				msd_df["msd_z_A2"] / msd_df["msd_total_A2"],
				np.nan
			)
		})
		eta_csv = os.path.join(self.lam_analysis_dir, "h3o_directional_efficiency.csv")
		eta_df.to_csv(eta_csv, index=False)

		volume_mean_A3 = float(np.mean(volumes))

		log_message(
			f"\tCond lambda={self.lam}: H3O MSD calculated from cpptraj-unwrapped trajectory. "
			f"num_h3o = {num_h3o}, mean volume = {volume_mean_A3:.3f} A3"
		)

		return msd_df, diffusion_df, msdc_df, eta_df, num_h3o, volume_mean_A3

class Cond_ConductivityAnalysis:
	def __init__(self, lam, lam_analysis_dir):
		self.lam = lam
		self.lam_analysis_dir = lam_analysis_dir

	def calculate_conductivity(self, diffusion_df, num_h3o, volume_mean_A3):
		D_total_m2_s = float(diffusion_df.loc[0, "D_total_m2_s"])
		volume_m3 = volume_mean_A3 * angstrom3_to_m3

		sigma_S_m = (
			num_h3o *
			elementary_charge_C ** 2 *
			D_total_m2_s /
			(volume_m3 * boltzmann_J_K * cond_temperature_K)
		)

		sigma_S_cm = sigma_S_m / 100.0

		df = pd.DataFrame([{
			"lambda": self.lam,
			"num_h3o": num_h3o,
			"temperature_K": cond_temperature_K,
			"volume_A3": volume_mean_A3,
			"volume_m3": volume_m3,
			"D_total_m2_s": D_total_m2_s,
			"sigma_S_m": sigma_S_m,
			"sigma_S_cm": sigma_S_cm,
		}])

		conductivity_csv = os.path.join(self.lam_analysis_dir, "h3o_conductivity.csv")
		df.to_csv(conductivity_csv, index=False)

		log_message(
			f"\tCond lambda={self.lam}: conductivity calculated. "
			f"sigma = {sigma_S_m:.6e} S/m"
		)

		return df

class Cond_ResidenceAnalysis:
	def __init__(self, lam, sampled_pdb, lam_analysis_dir):
		self.lam = lam
		self.sampled_pdb = sampled_pdb
		self.lam_analysis_dir = lam_analysis_dir

	def calculate_residence(self):
		h3o_o_indices = get_h3o_oxygen_indices_from_first_model(self.sampled_pdb)
		s_indices = get_s_atom_indices_from_first_model(self.sampled_pdb)

		rows = []
		sample_dt_ps = cond_time_between_nc_frames_ps * cond_frame_stride

		for frame_i, (positions, box) in enumerate(
			iter_pdb_frames(self.sampled_pdb)
		):
			h3o_pos = positions[h3o_o_indices]
			s_pos = positions[s_indices]
	
			num_bound = 0
	
			for pos in h3o_pos:
				delta = s_pos - pos
				delta -= box * np.round(delta / box)
	
				distances = np.linalg.norm(delta, axis=1)
	
				if np.min(distances) <= h3o_s_residence_cutoff:
					num_bound += 1
	
			fraction_bound = (
				num_bound / len(h3o_o_indices)
				if len(h3o_o_indices) > 0
				else np.nan
			)
	
			rows.append({
				"frame": frame_i,
				"time_ps": frame_i * sample_dt_ps,
				"time_ns": frame_i * sample_dt_ps / 1000.0,
				"num_h3o": len(h3o_o_indices),
				"num_bound_h3o": num_bound,
				"fraction_bound_h3o": fraction_bound,
			})

		df = pd.DataFrame(rows)

		residence_csv = os.path.join(self.lam_analysis_dir, "h3o_residence.csv")
		residence_png = os.path.join(self.lam_analysis_dir, "h3o_residence.png")

		df.to_csv(residence_csv, index=False)

		plt.figure(figsize=(7, 5))
		plt.plot(df["time_ns"], df["fraction_bound_h3o"])
		plt.xlabel("time [ns]")
		plt.ylabel("fraction bound H3O")
		plt.tight_layout()
		plt.savefig(residence_png, dpi=300)
		plt.close()

		summary = pd.DataFrame([{
			"lambda": self.lam,
			"h3o_s_residence_cutoff_A": h3o_s_residence_cutoff,
			"h3o_bound_fraction_mean": float(df["fraction_bound_h3o"].mean()),
			"h3o_bound_fraction_std": float(df["fraction_bound_h3o"].std(ddof=0)),
			"h3o_bound_number_mean": float(df["num_bound_h3o"].mean()),
			"h3o_bound_number_std": float(df["num_bound_h3o"].std(ddof=0)),
		}])

		residence_summary_csv = os.path.join(self.lam_analysis_dir, "h3o_residence_summary.csv")
		summary.to_csv(residence_summary_csv, index=False)

		log_message(
			f"\tCond lambda={self.lam}: H3O residence calculated. "
			f"bound fraction mean = {summary.loc[0, 'h3o_bound_fraction_mean']:.3f}"
		)

		return df, summary

class Cond_SummaryWriter:
	def __init__(self, lam, system_tag, lam_analysis_dir):
		self.lam = lam
		self.system_tag = system_tag
		self.lam_analysis_dir = lam_analysis_dir

	def write_summary(
		self,
		msd_df,
		diffusion_df,
		conductivity_df,
		msdc_df,
		eta_df,
		residence_summary
	):
		row = {
			"system_tag": self.system_tag,
			"polymer": polymer,
			"stage": "conductivity",
			"lambda": self.lam,
			"chain_length": chain_length,
			"num_chains": num_chains,
			"mix_chains": mix_chains,
			"trajectory": cond_traj_file,
			"restart": cond_restart_file,
			"cond_frame_start": cond_frame_start,
			"cond_frame_stop": cond_frame_stop,
			"cond_frame_stride": cond_frame_stride,
			"cond_time_between_nc_frames_ps": cond_time_between_nc_frames_ps,
			"msd_fit_start_ps": msd_fit_start_ps,
			"msd_fit_end_ps": msd_fit_end_ps,
			"cond_temperature_K": cond_temperature_K,
		}

		for col in diffusion_df.columns:
			row[col] = diffusion_df.loc[0, col]

		for col in conductivity_df.columns:
			if col != "lambda":
				row[col] = conductivity_df.loc[0, col]

		row["MSDc_final_A2"] = float(msdc_df["MSDc_A2"].iloc[-1])
		row["eta_c_final"] = float(eta_df["eta_c"].iloc[-1])

		if residence_summary is not None and len(residence_summary) > 0:
			for col in residence_summary.columns:
				if col != "lambda":
					row[col] = residence_summary.loc[0, col]

		df = pd.DataFrame([row])

		summary_csv = os.path.join(self.lam_analysis_dir, "cond_summary.csv")
		df.to_csv(summary_csv, index=False)

		log_message(f"\tCond lambda={self.lam}: cond_summary.csv written")

		return df

# ====
# Workflow functions
# ====

def run_dry_analysis_workflow():
	system_tag = system_name(polymer, chain_length, num_chains, mix_chains)

	prmtop_file = find_prmtop(dry_eq_dir)

	validate_dry_inputs(prmtop_file)
	log_general_settings()
	log_dry_settings(system_tag, prmtop_file)

	log_message("Dry analysis workflow started")

	traj_prep = Dry_TrajectoryPreparation(prmtop_file)
	last_pdb = traj_prep.create_last_pdb()
	sampled_pdb = traj_prep.create_sampled_pdb()

	density_analysis = Dry_DensityAnalysis(last_pdb)
	density = density_analysis.calculate_density()
	monomer_analysis = Dry_MonomerAnalysis(last_pdb)
	nn_mean, nn_std = monomer_analysis.calculate_monomer_nn_distance()

	rdf_analysis = Dry_RDFAnalysis(sampled_pdb, system_tag)
	rdf_peak_results = rdf_analysis.run_all_rdf(nn_mean, nn_std)

	nematic_analysis = Dry_NematicOrderAnalysis(sampled_pdb, system_tag)
	S2_mean, S2_std = nematic_analysis.calculate_s2()

	summary_writer = Dry_SummaryWriter(system_tag)
	summary = summary_writer.write_summary(
		density,
		nn_mean,
		nn_std,
		S2_mean,
		S2_std,
		rdf_peak_results
	)

	log_message("Dry analysis workflow finished")

	return summary

def run_hyd_analysis_workflow():
	all_summaries = []

	for lam in hyd_analysis_lam_list:
		lam_tag = hyd_system_name(polymer, chain_length, num_chains, lam, mix_chains)
		lam_dir, lam_md_dir, lam_analysis_dir = get_lam_hyd_dirs(lam)

		os.makedirs(lam_analysis_dir, exist_ok=True)

		log_message(f"Hydration analysis workflow started for lambda={lam}")
		log_message(f"\tlam_tag = {lam_tag}")
		log_message(f"\tlam_md_dir = {lam_md_dir}")
		log_message(f"\tlam_analysis_dir = {lam_analysis_dir}")

		prmtop_file = find_prmtop(lam_md_dir)
		validate_hyd_inputs(lam, prmtop_file)

		log_general_settings()
		log_hyd_settings(lam, lam_tag, prmtop_file, lam_md_dir, lam_analysis_dir)

		traj_prep = Hyd_TrajectoryPreparation(lam, prmtop_file)
		last_pdb = traj_prep.create_last_pdb()
		sampled_pdb = traj_prep.create_sampled_pdb()
		rdf_type_pdb = traj_prep.create_rdf_type_pdb()

		density_analysis = Hyd_DensityAnalysis(last_pdb)
		density = density_analysis.calculate_density()

		monomer_analysis = Hyd_MonomerAnalysis(last_pdb)
		nn_mean, nn_std = monomer_analysis.calculate_monomer_nn_distance()

		rdf_analysis = Hyd_RDFAnalysis(lam, rdf_type_pdb, lam_tag, lam_analysis_dir)
		rdf_peak_results = rdf_analysis.run_all_rdf(nn_mean, nn_std)

		nematic_analysis = Hyd_NematicOrderAnalysis(sampled_pdb, lam_tag, lam_analysis_dir)
		S2_mean, S2_std = nematic_analysis.calculate_s2()

		water_cluster_analysis = Hyd_WaterClusterAnalysis(sampled_pdb, lam, lam_analysis_dir)
		water_cluster_df, water_cluster_summary = water_cluster_analysis.calculate_water_clusters()

		summary_writer = Hyd_SummaryWriter(lam, lam_tag, lam_analysis_dir)
		summary = summary_writer.write_summary(
			density,
			nn_mean,
			nn_std,
			S2_mean,
			S2_std,
			rdf_peak_results,
			water_cluster_summary
		)

		all_summaries.append(summary)

		log_message(f"Hydration analysis workflow finished for lambda={lam}")

	if len(all_summaries) > 0:
		all_summary_df = pd.concat(all_summaries, ignore_index=True)
		all_summary_csv = os.path.join(hyd_analysis_dir, "hyd_summary_all.csv")
		all_summary_df.to_csv(all_summary_csv, index=False)

		log_message("Hydration summary for all lambda values written")

		return all_summary_df

	return None

def run_cond_analysis_workflow():
	all_summaries = []

	for lam in cond_analysis_lam_list:
		lam_tag = cond_system_name(polymer, chain_length, num_chains, lam, mix_chains)
		lam_dir, lam_md_dir, lam_analysis_dir = get_lam_cond_dirs(lam)

		os.makedirs(lam_analysis_dir, exist_ok=True)

		log_message(f"Conductivity analysis workflow started for lambda={lam}")
		log_message(f"\tlam_tag = {lam_tag}")
		log_message(f"\tlam_md_dir = {lam_md_dir}")
		log_message(f"\tlam_analysis_dir = {lam_analysis_dir}")

		prmtop_file = find_prmtop(lam_md_dir)
		validate_cond_inputs(lam, prmtop_file)

		log_general_settings()
		log_cond_settings(lam, lam_tag, prmtop_file, lam_md_dir, lam_analysis_dir)

		traj_prep = Cond_TrajectoryPreparation(lam, prmtop_file)
		last_pdb = traj_prep.create_last_pdb()

		# Full sampled trajectory for H3O-S residence
		sampled_pdb = traj_prep.create_sampled_pdb()

		# H3O-only unwrapped trajectory for MSD / D / conductivity
		h3o_unwrapped_pdb = traj_prep.create_h3o_unwrapped_pdb()

		msd_analysis = Cond_H3OMSDAnalysis(lam, h3o_unwrapped_pdb, lam_analysis_dir)
		msd_df, diffusion_df, msdc_df, eta_df, num_h3o, volume_mean_A3 = msd_analysis.calculate_msd()

		conductivity_analysis = Cond_ConductivityAnalysis(lam, lam_analysis_dir)
		conductivity_df = conductivity_analysis.calculate_conductivity(
			diffusion_df,
			num_h3o,
			volume_mean_A3
		)

		residence_analysis = Cond_ResidenceAnalysis(lam, sampled_pdb, lam_analysis_dir)
		residence_df, residence_summary = residence_analysis.calculate_residence()

		summary_writer = Cond_SummaryWriter(lam, lam_tag, lam_analysis_dir)
		summary = summary_writer.write_summary(
			msd_df,
			diffusion_df,
			conductivity_df,
			msdc_df,
			eta_df,
			residence_summary
		)

		all_summaries.append(summary)

		log_message(f"Conductivity analysis workflow finished for lambda={lam}")

	if len(all_summaries) > 0:
		all_summary_df = pd.concat(all_summaries, ignore_index=True)
		all_summary_csv = os.path.join(cond_analysis_dir, "cond_summary_all.csv")
		all_summary_df.to_csv(all_summary_csv, index=False)

		log_message("Conductivity summary for all lambda values written")

		return all_summary_df

	return None

# ====
# Main workflow
# ====

def run_workflow():
	ensure_directories()

	if run_dry_analysis:
		run_dry_analysis_workflow()

	if run_hyd_analysis:
		run_hyd_analysis_workflow()

	if run_cond_analysis:
		run_cond_analysis_workflow()

if __name__ == "__main__":
	run_workflow()