# Data

<!-- TODO(MCA): Delete this file in favor of data/<modality>/README.mds -->

## Fitness prediction

| Data | Size (unzipped) | Link |
| --- | --- | --- |
| Processed assay data | 73MB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_processed_assays.zip |
| Raw assay data | 88MB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_raw_data.zip |
| Model predictions | 438MB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/model_predictions.zip |
| Alignments | 319KB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_MSAs.zip |
| 3D structures | 621KB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_assays_3D_structures.zip |
| CV splits (supervised) | 145MB | https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_CV_splits.zip |

## 2D structure prediction

| Data | Size (unzipped) | Link |
| --- | --- | --- |
| Processed eval data | 3.2GB | https://marks.hms.harvard.edu/rnagym/structure_prediction/test_data.zip |
| Raw assay data | 5.1GB | https://marks.hms.harvard.edu/rnagym/structure_prediction/raw_data.zip |
| Model predictions | 34GB | https://marks.hms.harvard.edu/rnagym/structure_prediction/model_predictions.zip |
| Model files | 441MB | https://marks.hms.harvard.edu/rnagym/structure_prediction/models.zip |
| Training data (supervised) | 8.1GB | https://marks.hms.harvard.edu/rnagym/structure_prediction/train_data.zip |
| Additional annotations (PDB, Rfam, PseudoBase) | 29MB | https://marks.hms.harvard.edu/rnagym/structure_prediction/test_sequences_annotated.zip |

Model files for 2D structure prediction task were prepared for a linux 64-bit system.

## 3D structure prediction

| Data | Size (unzipped) | Link |
| --- | --- | --- |
| Curation tables | 26MB | [`data/3d/curation/`](../data/3d/curation/) |
| Evaluation targets | 130KB | [`data/3d/rnagym_3d.parquet`](../data/3d/rnagym_3d.parquet) |
| Per-target scores | 16KB | [`data/3d/rnagym_3d_scores.parquet`](../data/3d/rnagym_3d_scores.parquet) |
| Alignments | — | [`data/3d/msa/`](../data/3d/msa/) |
| Experimental and predicted structures | 158MB | [`data/3d/structures/`](../data/3d/structures/) |
| Test-to-reference TM comparisons | 12GB | [`data/3d/usalign/`](../data/3d/usalign/) |

Data is also available on our
[HuggingFace](https://huggingface.co/datasets/Marks-lab/RNAgym) and
[website](https://rnagym.org).
