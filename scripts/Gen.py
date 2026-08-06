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
        line = f"reinvent -l {stage}.log reinvent_pubchem.toml"
    elif stage == 'TL':
        line = f"reinvent -l {stage}.log reinvent_glpg.toml"
    # Figure out checkpoint selection...
    """
    elif stage == "RL_gen":
        line = f"reinvent -l {stage}.log PICKME!"
    """

    batch_content = f"""
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --gres={gpu_type}
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

    args = parser.parse_args()

    if args.stage == 'RL_prep':
        '''
        Setup Device
        1. Check for CUDA availability and set the device accordingly.
        2. Print the device being used.
        3. If CUDA is not available, print a warning message.
        '''
        write_batch_file(job_name="reinvent_RL_prep",
                         output_file="reinvent_RL_prep.out",
                         error_file="reinvent_RL_prep.err", 
                         gpu_type="nvidia_h100_80gb_hbm3_1g.10gb:1",
                         mem="16G",
                         cpus="1",
                         time="0-04:00",
                         account="def-aminpour",
                         email="jaharri1@ualberta.ca",
                         stage="RL_prep")

        print("Submitting RL_prep job to SLURM...")
        subprocess.run(["sbatch", "RL_prep.sh"])

    elif args.stage == 'TL':
        pass

    elif args.stage == 'RL_gen':
        pass
    
    

main()