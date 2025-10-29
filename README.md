<h1 align="center">
  <img src="https://raw.githubusercontent.com/MarksLab-DasLab/RNAGym/refs/heads/main/img/RNAGym.png" alt="RNAGym Logo">

  RNAGym
</h1>

<p align="center">
  <a href="https://github.com/MarksLab-DasLab/RNAGym/stargazers"><img src="https://img.shields.io/github/stars/MarksLab-DasLab/RNAGym?style=social" alt="GitHub stars"></a>
  <a href="https://www.biorxiv.org/content/10.1101/2025.06.16.660049v1"><img src="https://img.shields.io/badge/bioRxiv-2025.06.16.660049v1-success.svg" alt="bioRxiv"></a>
  <a href="https://doi.org/10.1101/2025.06.16.660049"><img src="https://img.shields.io/badge/DOI-10.1101/2025.06.16.660049-blue" alt="DOI"></a>
  <a href="https://github.com/MarksLab-DasLab/RNAGym/blob/main/LICENSE"><img src="https://img.shields.io/github/license/MarksLab-DasLab/RNAGym" alt="License"></a>
  <a href="https://rnagym.org"><img src="https://img.shields.io/badge/website-rnagym.org-orange" alt="Website"></a>
  <a href="https://github.com/MarksLab-DasLab/RNAGym"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python"></a>
</p>

[RNAGym](https://rnagym.org) is an extensive benchmark suite and resource for RNA fitness and structure prediction. This code repository provides unified access to all baselines leveraged in our paper, as well as to the underlying datasets used to assess their respective fitness and/or structure prediction performance.

## Baselines

### Fitness prediction

We have currently implemented the following baselines (see `./fitness`):

Model name | Model type | Reference
--- | --- | --- |
Evo 1 | Genomic Language Model | Nguyen et al. "Sequence modeling and design from molecular to genome scale with Evo". Science, 2024.
Evo 2 | Genomic Language Model | Brixi et al. "Genome modeling and design across all domains of life with Evo 2". bioRxiv, 2025.
RNAErnie | Genomic Language Model | Wang et al. "Multi-purpose RNA language modelling with motif-aware pretraining and type-guided fine-tuning". Nature Machine Intelligence, 2024.
RiNALMo | RNA Language Model | Penić et al. "RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks". arXiv, 2024.
RNA-FM | RNA Language Model | Chen et al. "Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions" arXiv, 2022. 
Nucleotide Transformer | Genomic Language Model | Dalla-Torre et al. "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics". Nature Methods, 2024.
GenSLM | Genomic Language Model | Zvyagin et al. "GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics". The International Journal of High Performance Computing Applications, 2023.


### Secondary structure prediction

We have currently implemented the following baselines (see `./2d`).

Model name | Model type | Reference
--- | --- | --- |
Ribonanzanet | Structure prediction | He et al. "Ribonanza: deep learning of rna structure through dual crowdsourcing". bioRxiv, 2024.
EternaFold | Structure prediction | Wayment-Steele et al. "RNA secondary structure packages evaluated and improved by high-throughput experiments." Nature Methods, 2022.
CONTRAfold | Structure prediction | Do et al. "CONTRAfold: RNA secondary structure prediction without physics-based models" Bioinformatics, 22 14:e90–8, 2006.
Vienna | Structure prediction | Gruber et al. "The vienna rna websuite." Nucleic Acids Research, 36:W70 – W74, 2008.
RNAstructure | Structure prediction | Reuter et al. "Rnastructure: software for rna secondary structure prediction and analysis". BMC Bioinformatics, 11:129 – 129, 2010.
RNA-FM | RNA Language Model | Chen et al. "Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions" arXiv, 2022.
UFold | Structure prediction | Fu et al. "UFold: fast and accurate RNA secondary structure prediction with deep learning" Nucleic Acids Research, 2022.
Mxfold2 | Structure prediction | Sato et al. "RNA secondary structure prediction using deep learning with thermodynamic integration" Nature Methods, 2021.

### Tertiary structure prediction

We have also developed a data curation pipeline, structure datasets, and
implemented several baselines for RNA tertiary structure (see `./3d`).

Model name | Model type | Reference
--- | --- | --- |
AlphaFold3 | Structure prediction | Abramson et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3" Nature, 2024.
NuFold | Structure prediction | Kagaya et al. "NuFold: end-to-end approach for RNA tertiary structure prediction with flexible nucleobase center representation" Nature Communications, 2025.
RhoFold+ | Structure prediction | Shen et al. "Accurate RNA 3D structure prediction using a language model-based deep learning approach" Nature Methods, 2024.
RoseTTAFoldNA | Structure prediction | Baek et al. "Accurate prediction of protein–nucleic acid complexes using RoseTTAFoldNA" Nature Methods, 2023.
trRosettaRNA | Structure prediction | Wang et al. "trRosettaRNA: automated prediction of RNA 3D structure with transformer network" Nature Communications, 2023.

## Setup

The RNAGym environment may be created via conda and the provided rnagym_env.yml file as follows:
```
conda env create -f rnagym_env.yml
conda activate rnagym_env
```
For the fitness prediction task, we recommend the following folder structure:
```
fitness_prediction/
├── processed_DMS_files/
├── model_predictions/
└── model_checkpoints/
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
**Fitness prediction** | Processed assay data | 73MB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_processed_assays.zip
**Fitness prediction** | Raw assay data | 88MB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_raw_data.zip
**Fitness prediction** | Model predictions | 438MB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/model_predictions.zip
**Fitness prediction** | Alignments | 319KB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_MSAs.zip
**Fitness prediction** | 3D structures | 621KB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_assays_3D_structures.zip
**Fitness prediction** | CV splits (supervised) | 145MB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_CV_splits.zip
**2D structure prediction** | Processed eval data | 3.2GB | https://marks.hms.harvard.edu/rnagym/structure_prediction/test_data.zip
**2D Structure prediction** | Raw assay data | 5.1GB | https://marks.hms.harvard.edu/rnagym/structure_prediction/raw_data.zip
**2D structure prediction** | Model predictions | 34GB | https://marks.hms.harvard.edu/rnagym/structure_prediction/model_predictions.zip
**2D structure prediction** | Model files | 441MB | https://marks.hms.harvard.edu/rnagym/structure_prediction/models.zip
**2D structure prediction** | Training data (supervised) | 8.1GB | https://marks.hms.harvard.edu/rnagym/structure_prediction/train_data.zip
**2D structure prediction** | Additional annotations (PDB, Rfam, PseudoBase) | 29MB | https://marks.hms.harvard.edu/rnagym/structure_prediction/test_sequences_annotated.zip
**3D structure prediction** | Alignments | 55.3MB | https://marks.hms.harvard.edu/rnagym/tertiary_structure_prediction/3D_alignments.tar.xz
**3D structure prediction** | Monomer scores | 7KB | https://marks.hms.harvard.edu/rnagym/tertiary_structure_prediction/monomer.csv
**3D structure prediction** | Complex scores | 15KB | https://marks.hms.harvard.edu/rnagym/tertiary_structure_prediction/multimer.csv
**3D structure prediction** | Model predictions (PDBs) | 165MB | https://marks.hms.harvard.edu/rnagym/tertiary_structure_prediction/3D_model_outputs.tar.xz
**3D structure prediction** | Test-to-train TM scores | 150MB | https://marks.hms.harvard.edu/rnagym/tertiary_structure_prediction/3D_train_to_test_usalign.tar.xz

Model files for 2D structure prediction task were prepared for a linux 64-bit system. Refer to the [Arnie repo](https://github.com/DasLab/arnie) for different systems.

Data is also available on our
[HuggingFace](https://huggingface.co/datasets/Marks-lab/RNAgym) and
[website](https://rnagym.org).

## Acknowledgements

Our codebase leveraged code from the following repositories to compute baselines:

Model | Repo
--- | ---
arnie | https://github.com/DasLab/arnie

## License

This project is available under the MIT license found in the LICENSE file in this GitHub repository.
