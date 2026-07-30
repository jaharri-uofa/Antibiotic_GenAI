'''
Gen.py
From an input CSV of known GlpG inhibitors, generate the REINVENT4 TOML for TL and RL to generate novel GlpG inhibitors for further screening

Author: Jordan Harrison
'''

import importlib
'''
for pkg in ["reinvent", "tensorboard", "mols2grid", "seaborn", "ipywidgets"]:
    found = importlib.util.find_spec(pkg) is not None
    print(f"{'OK     ' if found else 'MISSING'} {pkg}")

    '''

# +
import os
import shutil
import glob
import subprocess
import argparse
import torch
import toml
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdmolops
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import reinvent
from reinvent.notebooks import load_tb_data, plot_scalars, get_image, create_mol_grid
from reinvent.scoring.transforms import ReverseSigmoid
from reinvent.scoring.transforms.sigmoids import Parameters as SigmoidParameters

import ipywidgets as widgets

# ── Molecular feature extraction ───────────────────────────────────────────────

def molecule_features(smiles: str) -> dict | None:
    """
    Given a SMILES string extract molecualr features
    Param: smiles string
    Output: dictionary of molecular features
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolecularWeight":  rdMolDescriptors.CalcExactMolWt(mol),
        "TPSA":             rdMolDescriptors.CalcTPSA(mol),
        "HBondAcceptors":   rdMolDescriptors.CalcNumHBA(mol),
        "HBondDonors":      rdMolDescriptors.CalcNumHBD(mol),
        "NumRotBond":       rdMolDescriptors.CalcNumRotatableBonds(mol),
        "NumRings":         rdMolDescriptors.CalcNumRings(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "SlogP":            rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
    }

def write_batch_file(job_name, output_file, error_file, gpu_type, mem, cpus, time, account, email):
    batch_content = f"""
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --gres={gpu_type}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --account=def-aminpour
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaharri1@ualberta.ca

jobid=$SLURM_JOB_ID

echo "Loading modules..."
module --force purge 
module load StdEnv/2023
module load openbabel/3.1.1
module load gcc/12.3
module load cmake
module load cuda/12.6
module load python/3.11.5
module load scipy-stack/2023b
module load rdkit/2024.09.6
module load python-build-bundle/2025b
echo "modules loaded."

echo "Activating virtual environment..."
source ~/reinvent4/bin/activate
export PATH=$HOME/.local/bin:$PATH
echo "Virtual environment activated."

### Stage 1: Reinforcement learning for drug like molecules
echo "Running Stage 1: Reinforcement learning for drug like molecules"
reinvent -l stage1.log stage1_RL.toml
echo "Stage 1 completed."
"""
    with open("reinvent_RL.sh", "w") as f:
        f.write(batch_content)

def main():

    # Stage 0: Setup Device

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == False:
        print("Device not detected, please check cuda installation and reload proper modules")

    parser = argparse.ArgumentParser(
            description='Build and submit a REINVENT job (with optional TL fine-tuning).'
        )
    parser.add_argument('--smiles_csv',   required=True,
                            help='SMILES file (CSV format, column name = smiles).')
    parser.add_argument('--output_toml',  default='sampling.toml',
                            help='Output RL TOML filename.')
    parser.add_argument('--slurm_script', default='submit_reinvent.sh',
                            help='Output SLURM script filename.')

    wd = os.getcwd()
    top = os.path.abspath(os.path.join(reinvent.__path__[0], ".."))
    top

    #load csv, and extract features
    df = pd.read_csv(args.smiles_csv)
    smiles = np.array(df['smiles'])

    # debug 
    print(smiles)
    # I think this is how arrays work...
    molecular_features = molecule_features(smiles)
    print (molecular_features)

    # Stage 1: Reinforcement learning for drug like molecules
    """
    Objective: establish a baseline for the RL model to create "drug like" molecules. 
    """

    prior_filename = os.path.abspath(os.path.join(reinvent.__path__[0], "..", "priors", "reinvent.prior"))
    agent_filename = prior_filename

    stage1_checkpoint = "stage1.chkpt"
    stage1_summary_csv_prefix = "stage1_RL"

    stage1_parameters = f"""
    run_type = "staged_learning"
    device = "{device}"
    tb_logdir = "tb_stage1"
    json_out_config = "_stage1.json"

    [parameters]

    prior_file = "{prior_filename}"
    agent_file = "{agent_filename}"
    summary_csv_prefix = "{stage1_summary_csv_prefix}"

    batch size = 128

    use checkpoints = false

    [learning strategy]

    type = 'dap'
    sigma = 128
    rate = 0.0001

    [[stage]]

    max score = 1.0
    max steps = 300

    chkpt_file = "{stage1_checkpoint}"

    [stage.scoring]
    type = "geometric mean"and

    [[stage.scoring.component]]
    [stage.scoring.component.custom alerts]

    [[stage.scoring.component.custom_alerts.endpoints]]
    name = "Alerts"

    [[stage.scoring.component]]
    [stage.scoring.component.QED]

    [[stage.scoring.component.QED.endpoint]]
    name = "QED"
    weight = 0.6


    [[stage.scoring.component]]
    [stage.scoring.component.NumAtomStereoCenters]

    [[stage.scoring.component.NumAtomStereoCenters.endpoint]]
    name = "Stereo"
    weight = 0.4

    transform.type = "left_step"
    transform.low = 0
    """

    stage1_toml_filename = "stage1_RL.toml"
    with open(stage1_toml_filename, "w") as f:
        f.write(stage1_parameters)

    print(f"Stage 1 RL TOML file written to {stage1_toml_filename}")

    #launch the SLURM job
    write_batch_file( 
        job_name="reinvent_RL",
        output_file="reinvent_RL.out",
        error_file="reinvent_RL.err",
        gpu_type="nvidia_h100_80gb_hbm3_1g.10gb:1",
        mem="32G",
        cpus=4,
        time="24:00:00",
        account="def-aminpour",
        email="jaharri1@ulaberta.ca"
    )

