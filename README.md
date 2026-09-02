<h1 align="center">
  <a href="https://rnagym.org">
    <img src="https://raw.githubusercontent.com/MarksLab-DasLab/RNAGym/refs/heads/main/.github/resources/RNAGym.png" alt="RNAGym logo">
  </a>

  RNAGym
  <br>
  <sub><sup>
    <a href="leaderboard/fitness/">Fitness</a>&nbsp;&nbsp;·&nbsp;&nbsp;
    <a href="leaderboard/2d/">2D structure</a>&nbsp;&nbsp;·&nbsp;&nbsp;
    <a href="leaderboard/3d/">3D structure</a>
  </sup></sub>
</h1>

<p align="center">
  <a href="https://github.com/MarksLab-DasLab/RNAGym/stargazers"><img src="https://img.shields.io/github/stars/MarksLab-DasLab/RNAGym?style=social" alt="GitHub stars"></a>
  <a href="https://www.biorxiv.org/content/10.1101/2025.06.16.660049v1"><img src="https://img.shields.io/badge/bioRxiv-2025.06.16.660049v1-success.svg" alt="bioRxiv"></a>
  <a href="https://doi.org/10.1101/2025.06.16.660049"><img src="https://img.shields.io/badge/DOI-10.1101/2025.06.16.660049-blue" alt="DOI"></a>
  <a href="https://github.com/MarksLab-DasLab/RNAGym/blob/main/LICENSE"><img src="https://img.shields.io/github/license/MarksLab-DasLab/RNAGym" alt="License"></a>
  <a href="https://rnagym.org"><img src="https://img.shields.io/badge/website-rnagym.org-orange" alt="Website"></a>
  <a href="https://github.com/MarksLab-DasLab/RNAGym"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
</p>

[RNAGym][rnagym] is a benchmark suite and resource for RNA fitness and
structure prediction. This repository provides the datasets, baseline
implementations, and evaluation workflows used to compare fitness, secondary
structure, and tertiary structure prediction methods.

RNAGym is under active development. See the [releases][releases] page for
stable versions. Current results are available in the [leaderboards][leaderboards].

> **Update (v0.1.1):** We updated the [fitness leaderboard][fitness-leaderboard].
> Our [website][rnagym] will be updated shortly.

## Getting started

Download and extract the complete RNAGym v0.2 data from the repository root:

```bash
wget https://marks.hms.harvard.edu/rnagym/v0.2/data.tar.xz
tar -xJf data.tar.xz
```

Setup and reproduction instructions are provided for the [fitness][fitness],
[2D structure][s2d], and [3D structure][s3d] benchmarks.

## Contributing

We welcome contributions of models, datasets, and benchmark improvements.
Follow ongoing work, suggest additions, or report problems through the
[issue tracker][issues].

[data]: rnagym/DATA.md
[fitness]: rnagym/fitness/README.md
[fitness-leaderboard]: leaderboard/fitness/
[issues]: https://github.com/MarksLab-DasLab/RNAGym/issues/new/choose
[leaderboards]: leaderboard/
[releases]: https://github.com/MarksLab-DasLab/RNAGym/releases
[rnagym]: https://rnagym.org/
[s2d]: rnagym/s2d/README.md
[s3d]: rnagym/s3d/README.md
