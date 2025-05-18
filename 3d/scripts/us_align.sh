#!/bin/bash

#SBATCH --job-name=usalign
#SBATCH --output=out/%x_%j.out
#SBATCH --error=out/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=general

set -euo pipefail

# Set conda environment
source "/path/to/conda/etc/profile.d/conda.sh"
conda activate r3d  # RNAGym conda environment

pdb_id="$1"
asym_id="$2"
readonly QUERY="\"\`PDB ID\` == \'${pdb_id}\' and \`Asym. Chain ID\` == \'${asym_id}\'\""
readonly SCRIPT="
import pandas as pd

from cmd.split import get_split_candidates
from util.analysis import add_tm_id

mon_df, mul_df, _ = get_split_candidates()
df = pd.concat([mon_df, mul_df], axis=0).reset_index(drop=True)
print('Querying for ' + ${QUERY})
df = df.query(${QUERY})

if df.empty:
    raise ValueError(f\"Error: query returned empty DF'\")

add_tm_id(df)
print('Done!')
"

python3 -c "${SCRIPT}"

