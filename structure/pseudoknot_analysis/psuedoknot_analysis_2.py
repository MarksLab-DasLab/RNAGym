import os, sys
import pandas as pd
from arnie.bpps import bpps
from arnie.mfe import mfe
import pickle
import numpy as np
import glob
from arnie.utils import convert_bp_list_to_dotbracket, convert_dotbracket_to_bp_list

def dedupe_lists(list_of_lists):
    # Step 1: Convert each sublist to a sorted tuple
    tuple_set = {tuple(sorted(sublist)) for sublist in list_of_lists}
    
    # Step 2: Convert the set of tuples back to a list of lists
    deduped_list = [list(tup) for tup in tuple_set]
    
    return deduped_list

def detect_crossed_pairs(bp_list):
    """
    Detect crossed base pairs in a list of base pairs in RNA secondary structure.

    Args:
    bp_list (list of tuples): List of base pairs, where each tuple (i, j) represents a base pair.
    
    Returns:
    list of tuples: List of crossed base pairs.
    """
    crossed_pairs_set = set()
    crossed_pairs = []
    # Iterate through each pair of base pairs
    for i in range(len(bp_list)):
        for j in range(i+1, len(bp_list)):
            bp1 = bp_list[i]
            bp2 = bp_list[j]

            # Check if they are crossed
            if (bp1[0] < bp2[0] < bp1[1] < bp2[1]) or (bp2[0] < bp1[0] < bp2[1] < bp1[1]):
                crossed_pairs.append(bp1)
                crossed_pairs.append(bp2)
                crossed_pairs_set.add(bp1[0])
                crossed_pairs_set.add(bp1[1])
                crossed_pairs_set.add(bp2[0])
                crossed_pairs_set.add(bp2[1])
    return dedupe_lists(crossed_pairs), crossed_pairs_set

def dotbrackte2bp(structure):
    stack={'(':[],
           '[':[],
           '<':[],
           '{':[]}
    pop={')':'(',
         ']':'[',
         '>':"<",
         '}':'{'}       
    bp_list=[]
    matrix=np.zeros((len(structure),len(structure)))
    for i,s in enumerate(structure):
        if s in stack:
            stack[s].append((i,s))
        elif s in pop:
            forward_bracket=stack[pop[s]].pop()
            #bp_list.append(str(forward_bracket[0])+'-'+str(i))
            #bp_list.append([forward_bracket[0],i])
            bp_list.append([forward_bracket[0],i])

    return bp_list  


def calculate_f1_score_with_pseudoknots(true_pairs, predicted_pairs):
    true_pairs=[f"{i}-{j}" for i,j in true_pairs]
    predicted_pairs=[f"{i}-{j}" for i,j in predicted_pairs]
    
    true_pairs=set(true_pairs)
    predicted_pairs=set(predicted_pairs)

    # Calculate TP, FP, and FN
    TP = len(true_pairs.intersection(predicted_pairs))
    FP = len(predicted_pairs)-TP
    FN = len(true_pairs)-TP

    # Calculate Precision, Recall, and F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

# Load predictions from step 1
with open('prediction_d.pkl', 'rb') as handle:
    prediction_d = pickle.load(handle)

df = []
for idx in range(len(prediction_d)):
    data = prediction_d[idx]
    seq = data["seq"]
    ss = data["ss"]
    true_bp = convert_dotbracket_to_bp_list(ss, allow_pseudoknots=True)
    pb_name = data["pb_name"]
    row = [seq, ss, pb_name]
    
    for pkg in ['contrafold_2', 'rnastructure', 'vienna', 'eternafold', 'ribonanza', 'rna_fm']:
        ss_pred = data[pkg]
        row.append(ss_pred)

    for pkg in ['contrafold_2', 'rnastructure', 'vienna', 'eternafold', 'ribonanza', 'rna_fm']:
        ss_pred = data[pkg]    
        predicted_bp = convert_dotbracket_to_bp_list(ss_pred, allow_pseudoknots=True)

        # Global F1
        crossed_pairs,crossed_pairs_set=detect_crossed_pairs(true_bp)
        predicted_crossed_pairs,predicted_crossed_pairs_set=detect_crossed_pairs(predicted_bp)
        
        _,_,f1=calculate_f1_score_with_pseudoknots(true_bp, predicted_bp)

        # crossed F1
        assert(len(crossed_pairs) > 0)
        _,_,crossed_pair_f1=calculate_f1_score_with_pseudoknots(crossed_pairs, predicted_crossed_pairs)
        row.append((f1, crossed_pair_f1))
    df.append(row)


columns = ["sequence", "secondary_structure", "Pseudobase_name", "CONTRAfold", "RNAstructure", "Vienna", "EternaFold", "RibonanzaNet-SS", "RNA-FM"]
columns += ["CONTRAfold_F1_crossedF1_score", "RNAstructure_F1_crossedF1_score", "Vienna_F1_crossedF1_score", "EternaFold_F1_crossedF1_score", "RibonanzaNet-SS_F1_crossedF1_score", "RNA-FM_F1_crossedF1_score"]
df = pd.DataFrame(df, columns=columns)
df.to_csv("pseudobase_prediction.csv", index=False)
