import numpy as np
import pandas as pd
import os
import shutil
import glob
import subprocess
import warnings
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

polymer = "a1"

chain_length = 15
num_chains = 30
mix_chains = False
aligned = True
conf_selection = "best"
run_id = "r1"

run_dry_analysis = True
run_hyd_analysis = False
run_cond_analysis = False

# Dry equilibration protocol
# "6-step" or "12-step"
dry_eq_prot = "6-step"

# Dry production trajectory is selected automatically from dry_eq_prot
def dry_production_files(protocol):
	if protocol == "6-step":
		return "dry-eq_7-nvt-pr.nc", "dry-eq_7-nvt-pr.ncrst"

	if protocol == "12-step":
		return "dry-eq_13-nvt-pr.nc", "dry-eq_13-nvt-pr.ncrst"

	raise ValueError("dry_eq_prot must be '6-step' or '12-step'")

dry_traj_file, dry_restart_file = dry_production_files(dry_eq_prot)

# frame selection for dry pr-nvt trajectory
frame_start = 1
frame_stop = 100
frame_stride = 10
overwrite = True

# RDF settings
dry_rdf_pairs = ["N-N", "S-S", "N-S"]

rdf_settings = {
	"N-N": {"cutoff": 30.0, "bins": 300},
	"S-S": {"cutoff": 15.0, "bins": 150},
	"N-S": {"cutoff": 30.0, "bins": 300},
}

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

# ====
# Paths / Folders
# ====

base_dir = os.getcwd()

# expected folder structure:
# base_dir/
#	run/
#	  init/
#	  dry_eq/
#	  hyd_eq/
#	  cond_pr/
#	analysis/
#	  dry/

simulation_dir = os.path.join(base_dir, "run")
dry_eq_dir = os.path.join(simulation_dir, "dry_eq")
hyd_eq_dir = os.path.join(simulation_dir, "hyd_eq")
cond_pr_dir = os.path.join(simulation_dir, "cond_pr")

analysis_dir = os.path.join(base_dir, "analysis")
dry_analysis_dir = os.path.join(analysis_dir, "dry")
hyd_analysis_dir = os.path.join(analysis_dir, "hyd")
cond_analysis_dir = os.path.join(analysis_dir, "cond")

output = os.path.join(base_dir, "spi_cond_analysis_workflow_process.txt")

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

def log_settings(system_tag, prmtop_file):
	log_message("SPI conductivity analysis workflow")
	log_message(f"\tpolymer = {polymer}")
	log_message(f"\tchain_length = {chain_length}")
	log_message(f"\tnum_chains = {num_chains}")
	log_message(f"\tmix_chains = {mix_chains}")
	log_message(f"\taligned = {aligned}")
	log_message(f"\tconf_selection = {conf_selection}")
	log_message(f"\trun_id = {run_id}")
	log_message(f"\tdry_eq_prot = {dry_eq_prot}")
	log_message(f"\tsystem_tag = {system_tag}")
	log_message(f"\tbase_dir = {base_dir}")
	log_message(f"\tdry_eq_dir = {dry_eq_dir}")
	log_message(f"\tdry_analysis_dir = {dry_analysis_dir}")
	log_message(f"\tprmtop_file = {prmtop_file}")
	log_message(f"\tdry_traj_file = {dry_traj_file}")
	log_message(f"\tdry_restart_file = {dry_restart_file}")
	log_message(f"\tframe_start = {frame_start}")
	log_message(f"\tframe_stop = {frame_stop}")
	log_message(f"\tframe_stride = {frame_stride}")

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
			"aligned": aligned,
			"conf_selection": conf_selection,
			"run_id": run_id,
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
# Workflow functions
# ====

def run_dry_analysis_workflow():
	system_tag = system_name(polymer, chain_length, num_chains, mix_chains)

	prmtop_file = find_prmtop(dry_eq_dir)

	validate_dry_inputs(prmtop_file)
	log_settings(system_tag, prmtop_file)

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

# ====
# Main workflow
# ====

def run_workflow():
	ensure_directories()

	if run_dry_analysis:
		run_dry_analysis_workflow()

	if run_hyd_analysis:
		log_message("Hydrated analysis is not implemented yet")

	if run_cond_analysis:
		log_message("Conductivity analysis is not implemented yet")

if __name__ == "__main__":
	run_workflow()