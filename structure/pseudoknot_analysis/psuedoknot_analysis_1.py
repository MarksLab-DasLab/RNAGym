import os, sys
import pandas as pd
from arnie.bpps import bpps
from arnie.mfe import mfe
import pickle
import numpy as np
import glob
from arnie.utils import convert_bp_list_to_dotbracket, convert_dotbracket_to_bp_list

# Set DATAPATH for using RNAstructure with arnie. Either set here or in bashrc
os.environ['DATAPATH'] = '/home/groups/rhiju/cachoe/packages/RNAstructure/data_tables'

# Note:
# RNA-FM the online webserver was used: https://proj.cse.cuhk.edu.hk/rnafm/#/
# RibonanzaNet-ss was seperately run locally. The kaggle notebook is also available: https://www.kaggle.com/code/davidbtcox/ribonanzanet-ss-structure-inference/code

def read_ct_ss(ct):
    """ Function to convert .ct files to sequence and secondary structure 
    .ct or connect tables are the output format for RNA-FM
    """
    with open(ct, "r") as f:
        seq_len = 0
        seq = ""
        bp_list = []
        for idx, line in enumerate(f):
            if idx == 0:
                seq_len = int(line.split()[0])
                continue
            line = line.strip()
            line = line.split()
            base = line[1]
    
            base_idx = idx
            basepair_idx = int(line[4])
            seq += base
    
            if basepair_idx != 0:
                bp = sorted([base_idx-1, basepair_idx-1]) # 1 index to 0 index
                if bp not in bp_list:
                    bp_list.append(bp)
        ss = convert_bp_list_to_dotbracket(bp_list, seq_len)
        return seq, ss


# Read the pseudobase data
df = pd.read_csv("pseudobase_data.csv")
prediction_d = {}
for idx, row in df.iterrows():
    # Replace T with U and Y with C
    seq = row["sequence"].replace("T","U").replace("Y","C")
    ss = row["secondary_structure"]

    psuedobase_ids = row["psuedobase_ids"]

    result = {"seq":seq, "ss":ss, "pb_name":psuedobase_ids}
    for pkg in ['contrafold_2', 'vienna', 'eternafold']:
        ss_mfe = mfe(seq, package=pkg, pseudo=False)
        result[pkg] = ss_mfe
    for pkg in ['rnastructure']:
        ss_mfe = mfe(seq, package=pkg, pseudo=True)
        result[pkg] = ss_mfe
    prediction_d[idx] = result

# Read RibonanzaNet-SS predictions
ribonanza_ss_pkl = "ribonanza_ss_pred.pkl"
with open(ribonanza_ss_pkl, "rb") as handle:
    ribonanza_ss_d = pickle.load(handle)

for idx in range(len(prediction_d)):
    data = prediction_d[idx]
    seq = data["seq"]
    ss_ribonanza = ribonanza_ss_d[seq]["ss"]
    prediction_d[idx]["ribonanza"] = ss_ribonanza

# Read RNA-FM predictions
ss_rnafm_d = {}
for ct in glob.glob("./RNAFM_ct/*.ct"):
    seq_rnafm, ss_rnafm = read_ct_ss(ct)
    ss_rnafm_d[seq_rnafm] = ss_rnafm

for idx in range(len(prediction_d)):
    data = prediction_d[idx]
    seq = data["seq"]
    if seq not in ss_rnafm_d:
        print(idx, seq)
        continue
    ss_rnafm = ss_rnafm_d[seq]
    prediction_d[idx]["rna_fm"] = ss_rnafm

# Save predictions to pickle

with open('prediction_d.pkl', 'wb') as handle:
    pickle.dump(prediction_d, handle, protocol=pickle.HIGHEST_PROTOCOL)