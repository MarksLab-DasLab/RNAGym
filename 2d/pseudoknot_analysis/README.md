# RNAGym Pseudobase analysis

Contains the scripts and data for running the pseudobase analysis.

'ribonanza_ss_pred.pkl' and 'RNAFM_ct.zip' contain the predictions for RibonanzaNet-SS and RNA-FM respectively.
'psuedobase_data.csv' contains the Pseudobase dataset.

The analysis is written in two steps:
1. 'psuedoknot_analysis_1.py' : Runs arnie and reads the saved files to compile all the predictions for the Pseudobase dataset.
2. 'psuedoknot_analysis_2.py' : Calculates the F1 and crossed-F1 for all the predictions.
