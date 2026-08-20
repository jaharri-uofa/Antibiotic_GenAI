#!/bin/bash
#SBATCH --job-name=REINVENT
#SBATCH --output=reinvent.out
#SBATCH --error=reinvent.err
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-03:00:00
#SBATCH --account=def-aminpour
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaharri1@ualberta.ca

jobid=$SLURM_JOB_ID

echo "Job ID: $jobid"

echo "Loading modules..."
module --force purge 
module load StdEnv/2023 openbabel/3.1.1 gcc/12.3 cmake cuda/12.6 python/3.11.5 scipy-stack/2023b rdkit/2024.09.6 python-build-bundle/2025b
echo "Modules loaded."

echo "Activating virtual environment..."
source ~/reinvent4/bin/activate
export PATH=$HOME/.local/bin:$PATH
echo "Virtual environment activated."

if [[ ! -e "Stage_1_RL_prep/RL_prep.chkpt" ]]; then
    echo "Running REINVENT4. Stage = Pubchem Dataset Training."
    python scripts/Gen.py --stage RL_prep --prior reinvent_pubchem.prior
    cd Stage_1_RL_prep
    reinvent -l RL_prep.log RL_prep.toml
    echo "Exit code: $?"
    cd ..
fi

if [[ -e "Stage_1_RL_prep/RL_prep.chkpt" ]]; then
    if [[ ! -e "Stage_2_TL/TL_reinvent.model.50.chkpt" ]]; then
        echo "Running REINVENT4. Stage = Transfer Learning."
        python scripts/Gen.py --stage TL --checkpoint Stage_1_RL_prep/RL_prep.chkpt --smiles_csv GlpG.csv
        cd Stage_2_TL
        reinvent -l TL.log TL.toml
        echo "Exit code: $?"
        cd ..
    fi
fi

if [[ -e "Stage_2_TL/TL_reinvent.model.50.chkpt" ]]; then
    echo "Running REINVENT4. Stage = Reinforcement Learning"
    python scripts/Gen.py --stage RL_gen --prior reinvent_pubchem.prior
    cd Stage_3_RLgen
    reinvent -l RL_gen.log RL_gen.toml
    echo "Exit code: $?"
    echo "Run Complete. Check the log files for details."
fi