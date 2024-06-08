# RNAGym

## Overview

RNAGym is an extensive benchmark suite for RNA fitness and structure prediction. This code repository provides unified access to all baselines leveraged in our paper, as well as to the underlying datasets used to assess their respective fitness and/or structure prediction performance.

## Baselines

### Fitness prediction
We have currently implemented the following baselines:

Model name | Model type | Reference
--- | --- | --- |
RiNALMo | |
Evo | |
RNA-FM | |
GenSLM | |
Nucleotide Transformer | |

### Structure prediction

We have currently implemented the following baselines:

Model name | Model type | Reference
--- | --- | --- |
Ribonanzanet | Structure prediction | He et al. "Ribonanza: deep learning of rna structure through dual crowdsourcing". bioRxiv, 2024.
EternaFold | Structure prediction | Wayment-Steele et al. "Rna secondary structure packages evaluated and improved by high-throughput experiments." bioRxiv, 2020.
CONTRAfold | Structure prediction | Do et al. "CONTRAfold: RNA secondary structure prediction without physics-based models" Bioinformatics, 22 14:e90–8, 2006.
Vienna | Structure prediction | Gruber et al. "The vienna rna websuite." Nucleic Acids Research, 36:W70 – W74, 2008.
RNAstructure | Structure prediction | Reuter et al. "Rnastructure: software for rna secondary structure prediction and analysis". BMC Bioinformatics, 11:129 – 129, 2010.

## Resources

To download and unzip the data, run the following commands for each of the data sources you would like to download, as listed in the table below. 
For example, you can download & unzip the zero-shot predictions for all baselines for all DMS substitution assays as follows:
```
curl -o zero_shot_substitutions_scores.zip https://marks.hms.harvard.edu/proteingym/zero_shot_substitutions_scores.zip
unzip zero_shot_substitutions_scores.zip && rm zero_shot_substitutions_scores.zip
```

Data | Size (unzipped) | Link
--- | --- | --- |
Fitness benchmark | 100MB | url

## Acknowledgements

Our codebase leveraged code from the following repositories to compute baselines:

Model | Repo
--- | ---
arnie | https://github.com/DasLab/arnie

## License
This project is available under the MIT license found in the LICENSE file in this GitHub repository.

## Links
- Website: https://www.rnagym.org/