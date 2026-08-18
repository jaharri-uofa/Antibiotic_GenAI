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

modules = "StdEnv/2023 openbabel/3.1.1 gcc/12.3 cmake cuda/12.6 python/3.11.5 scipy-stack/2023b rdkit/2024.09.6 python-build-bundle/2025b"

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

def write_batch_file(job_name, output_file, error_file, gpu_type, mem, cpus, time, account, email, stage):
    if stage not in ['RL_prep', 'TL', 'RL_gen']:
        raise ValueError("Invalid stage. Must be one of: 'RL_prep', 'TL', 'RL_gen'.")
    elif stage == 'RL_prep':
        line = f"reinvent -l {stage}.log {stage}.toml"
    elif stage == 'TL':
        line = f"reinvent -l {stage}.log {stage}.toml"
    # Figure out checkpoint selection...
    """
    elif stage == "RL_gen":
        line = f"reinvent -l {stage}.log PICKME!"
    """

    batch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --gpus={gpu_type}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --account={account}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={email}

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
echo "Modules loaded."

echo "Activating virtual environment..."
source ~/reinvent4/bin/activate
export PATH=$HOME/.local/bin:$PATH
echo "Virtual environment activated."

echo "Running REINVENT4. Stage {stage}."
{line}
    """
    
    with open(f"{stage}.sh", "w") as f:
        f.write(batch_content)

def main():

    # Reinforcement lerning on pubchem data set -> transfer learnin using GLPG inhibitors -> RL to generate novel GLPG inhibitors
    # Guard clasues in slurm script ie
        # reinvent rl (arguments?)
        # python gen.py -- smiles_csv=glpg_inhibitors.csv --stage = TL (hard code TOML file names?)
        # reinvent tl
        # python gen.py -- stage RL
        # reinvent genAI
        # python gen.py --analysis?

    parser = argparse.ArgumentParser(description='Generate REINVENT4 TOML for TL and RL to generate novel GlpG inhibitors.')

    # CSV file only required for transfer learning stage of the procedure
    parser.add_argument('--smiles_csv', required=False, help='Path to the input CSV file containing known GlpG inhibitors (column name = smiles).')

    parser.add_argument('--stage', required=True, choices=['RL_prep', 'TL', 'RL_gen'], help='Stage of the process: RL_prep, TL, or RL_gen.')

    parser.add_argument('--prior', required=False, help='Path to the prior model file.')

    parser.add_argument('--checkpoint', required=False, help='Path to prior-stage checkpoint file (used as input_model_file for TL, agent_file for RL_gen).')

    args = parser.parse_args()

    # Load necessary modules for cluster to not die
    os.system(f"module load {modules}")

    if args.stage == 'RL_prep':
        '''
        Setup Device
        1. Check for CUDA availability and set the device accordingly.
        2. Print the device being used.
        3. If CUDA is not available, print a warning message.
        '''

        if not os.path.exists("Stage_1_RL_prep"):
            os.mkdir("Stage_1_RL_prep")

        shutil.copy(args.prior, "Stage_1_RL_prep/")
        os.chdir("Stage_1_RL_prep")
        

        stage1_parameters = f"""
run_type = "staged_learning"
device = "cuda:0"
tb_logdir = "tb_stage1"
json_out_config = "_stage1.json"

[parameters]

prior_file = "{args.prior}"
agent_file = "{args.prior}"
summary_csv_prefix = "{args.stage}"

batch_size = 100

use_checkpoint = true

[learning_strategy]

type = "dap"
sigma = 128
rate = 0.0001

[[stage]]

max_score = 1.0
max_steps = 300

chkpt_file = "RL_prep.chkpt"

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]

[[stage.scoring.component.custom_alerts.endpoint]]
name = "Alerts"

params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C#C",
    "C(=[O,S])[O,S]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"
]

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
        stage1_config_filename = f"{args.stage}.toml"

        with open(stage1_config_filename, "w") as tf:
            tf.write(stage1_parameters)
        
        write_batch_file(job_name="reinvent_RL_prep",
                         output_file="reinvent_RL_prep.out",
                         error_file="reinvent_RL_prep.err", 
                         gpu_type="h100_1g.10gb:1",
                         mem="16G",
                         cpus="1",
                         time="0-04:00",
                         account="def-aminpour",
                         email="jaharri1@ualberta.ca",
                         stage="RL_prep")

        print("Submitting RL_prep job to SLURM...")
        subprocess.run(["sbatch", "RL_prep.sh"])

    elif args.stage == 'TL':
        if not args.smiles_csv:
            raise ValueError("The --smiles_csv argument is required for the TL stage.")
        if not os.path.exists("Stage_2_TL"):
            os.mkdir("Stage_2_TL")
        shutil.copy(args.checkpoint, "Stage_2_TL/")
        shutil.copy(args.smiles_csv, "Stage_2_TL/")
        os.chdir("Stage_2_TL")
        print(f"Current working directory: {os.getcwd()}")

        # Read the CSV file and extract SMILES strings
        df = pd.read_csv(args.smiles_csv)
        if 'smiles' not in df.columns:
            raise ValueError("The input CSV must contain a 'smiles' column.")
        
        smiles_list = df['smiles'].dropna().tolist()
        print(f"Extracted {len(smiles_list)} SMILES strings from {args.smiles_csv}.")

        # Split smiles into training and validation sets (80% train, 20% validation, chosen arbitrarily will check back to see if this is good)
        rng = np.random.default_rng()
        smiles_array = np.array(smiles_list)
        rng.shuffle(smiles_array)
        smiles_list = smiles_array.tolist()

        train_size = int(0.8 * len(smiles_list))
        val_size = len(smiles_list) - train_size
        print(f" Training set size: {train_size}, Validation set size: {val_size}")

        train_smiles = smiles_list[:train_size]
        val_smiles = smiles_list[train_size:]
        with open ("train_smiles.smi", "w") as file:
            for smi in train_smiles:
                file.write(f"{smi}\n")
        with open ("val_smiles.smi", "w") as file:
            for smi in val_smiles:
                file.write(f"{smi}\n")
        # Generate TOML configuration for Transfer Learning
        stage2_parameters = f"""
        run_type = "transfer_learning"
device = "cuda:0"
tb_logdir = "tb_TL"


[parameters]

num_epochs = 50
save_every_n_epochs = 2
batch_size = 100
sample_batch_size = 2000

input_model_file = "{args.checkpoint}"
output_model_file = "TL_reinvent.model"
smiles_file = "train_smiles.smi"
validation_smiles_file = "val_smiles.smi"
standardize_smiles = true
randomize_smiles = true
randomize_all_smiles = false
internal_diversity = true
"""
    
        stage2_config_filename = f"{args.stage}.toml"

        with open(stage2_config_filename, "w") as tf:
            tf.write(stage2_parameters)
        
        write_batch_file(job_name="reinvent_TL",
                         output_file="reinvent_TL.out",
                         error_file="reinvent_TL.err", 
                         gpu_type="h100_1g.10gb:1",
                         mem="16G",
                         cpus="1",
                         time="0-04:00",
                         account="def-aminpour",
                         email="jaharri1@ualberta.ca",
                         stage="TL")

        print("Submitting TL job to SLURM...")
        subprocess.run(["sbatch", "TL.sh"])

    elif args.stage == 'RL_gen':
        pass
    
    

main()