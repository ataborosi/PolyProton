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

# dry pr-nvt trajectory
dry_traj_file = "dry-eq_7-nvt-pr.nc"
dry_restart_file = "dry-eq_7-nvt-pr.ncrst"

# frame selection for dry pr-nvt trajectory
frame_start = 1
frame_stop = 100
frame_stride = 10
overwrite = True

# RDF settings
dry_rdf_pairs = ["N-N", "S-S", "N-S"]

rdf_settings = {
	"N-N": {"cutoff": 30.0, "bins": 150},
	"S-S": {"cutoff": 15.0, "bins": 150},
	"N-S": {"cutoff": 30.0, "bins": 150},
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