# RNAGym

## Overview

RNAGym is an extensive benchmark suite for RNA fitness and structure prediction. This code repository provides unified access to all baselines leveraged in our paper, as well as to the underlying datasets used to assess their respective fitness and/or structure prediction performance.

## Baselines

### Fitness prediction
We have currently implemented the following baselines:

Model name | Model type | Reference
--- | --- | --- |
RiNALMo | RNA Language Model | Penić et al. "RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks". arXiv, 2024.
Evo | Genomic Language Model | Nguyen et al. "Sequence modeling and design from molecular to genome scale with Evo". bioRxiv, 2024.
RNA-FM | RNA Language Model | Chen et al. "Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions" arXiv, 2022. 
GenSLM | Genomic Language Model | Zvyagin et al. "GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics". The International Journal of High Performance Computing Applications, 2023.
Nucleotide Transformer | Genomic Language Model | Dalla-Torre et al. "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics". bioRxiv, 2023.

### Structure prediction

We have currently implemented the following baselines:

Model name | Model type | Reference
--- | --- | --- |
Ribonanzanet | Structure prediction | He et al. "Ribonanza: deep learning of rna structure through dual crowdsourcing". bioRxiv, 2024.
EternaFold | Structure prediction | Wayment-Steele et al. "Rna secondary structure packages evaluated and improved by high-throughput experiments." bioRxiv, 2020.
CONTRAfold | Structure prediction | Do et al. "CONTRAfold: RNA secondary structure prediction without physics-based models" Bioinformatics, 22 14:e90–8, 2006.
Vienna | Structure prediction | Gruber et al. "The vienna rna websuite." Nucleic Acids Research, 36:W70 – W74, 2008.
RNAstructure | Structure prediction | Reuter et al. "Rnastructure: software for rna secondary structure prediction and analysis". BMC Bioinformatics, 11:129 – 129, 2010.
RNA-FM | RNA Language Model | Chen et al. "Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions" arXiv, 2022. 

## Setup

The RNAGym environment may be created via conda and the provided rnagym_env.yml file as follows:
```
conda env create -f rnagym_env.yml
conda activate rnagym_env
```

For the structure prediction task, the data processing and scoring scripts expect the following folder structure:
```
structure_prediction/
├── test_data/
├── raw_data/
├── model_predictions/
└── models/
```
The content for `models`, `model_predictions`, `raw_data` and `test_data` may all be downloaded via the links in the next section.
The `data_folder` argument in the data and scoring scripts should be set to the location of the `structure_prediction` folder.

## Resources

To download and unzip the data, run the following commands for each of the data sources you would like to download, as listed in the table below. 
For example, you can download & unzip the zero-shot predictions for all baselines for all DMS substitution assays as follows:
```
curl -o rnagym_assays.zip https://marks.hms.harvard.edu/rnagym/fitness_prediction/rnagym_assays.zip
unzip rnagym_assays.zip && rm rnagym_assays.zip
```

Task | Data | Size (unzipped) | Link
--- | --- | --- | --- |
**Fitness prediction** | Processed assay data | 75MB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/rnagym_assays.zip
**Fitness prediction** | Raw assay data | 88MB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/raw_data.zip
**Fitness prediction** | Model predictions | 438MB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/model_predictions.zip
**Fitness prediction** | CV splits (supervised) | 145M | https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_CV_splits.zip
**Structure prediction** | Processed eval data | 3.2GB | https://marks.hms.harvard.edu/rnagym/structure_prediction/test_data.zip
**Structure prediction** | Raw assay data | 5.1GB | https://marks.hms.harvard.edu/rnagym/structure_prediction/raw_data.zip
**Structure prediction** | Model predictions | 34GB | https://marks.hms.harvard.edu/rnagym/structure_prediction/model_predictions.zip
**Structure prediction** | Model files | 441MB | https://marks.hms.harvard.edu/rnagym/structure_prediction/models.zip
**Structure prediction** | Additional annotations (PDB, Rfam, PseudoBase) | 29M | https://marks.hms.harvard.edu/rnagym/structure_prediction/test_sequences_annotated.zip

Model files for the structure prediction task were prepared for a linux 64-bit system. Refer to the [Arnie repo](https://github.com/DasLab/arnie) for different systems.

## Acknowledgements

Our codebase leveraged code from the following repositories to compute baselines:

Model | Repo
--- | ---
arnie | https://github.com/DasLab/arnie

## License
This project is available under the MIT license found in the LICENSE file in this GitHub repository.

## Links
- Website: https://www.rnagym.org/
