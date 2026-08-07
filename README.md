<h1 align="center">
  <img src="https://raw.githubusercontent.com/MarksLab-DasLab/RNAGym/refs/heads/main/.img/RNAGym.png" alt="RNAGym Logo">

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
  <a href="https://github.com/MarksLab-DasLab/RNAGym"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python"></a>
</p>

[RNAGym](https://rnagym.org) is an extensive benchmark suite and resource for
RNA fitness and structure prediction. This code repository provides unified
access to all baselines leveraged in our paper, as well as to the underlying
datasets used to assess their respective fitness and/or structure prediction
performance.

RNAGym is under active development.  Please refer to our
[releases](https://github.com/MarksLab-DasLab/RNAGym/releases) page to track
the latest updates and stable releases. The current leaderboards can be
accessed [here](leaderboard/), while the complete list of baselines can be
viewed [here](BASELINES.md).

> **Update (v0.1.1):** We updated the [leaderboard](leaderboard/fitness/).  Our
> [website][rnagym] will be updated shortly.

## Getting started

<!-- TODO(MCA): Upload complete data files to this link -->

First, download and extract the complete RNAGym data (as described in
[DATA.md](DATA.md)), by running these commands from the repository root:

```bash
wget https://marks.hms.harvard.edu/rnagym/v0.2/data.tar.xz
tar -xJf data.tar.xz
```

For setup and reproduction details, see the respective benchmark READMEs:
[fitness](fitness/README.md), [2D structure](rnagym/s2d/README.md), and [3D
structure](rnagym/s3d/README.md).

## Contributing

We welcome community contributions, including new models and datasets. Follow
ongoing work, suggest additions, or report problems on our [issue tracker][0].

[rnagym]: https://rnagym.org/
[0]: https://github.com/MarksLab-DasLab/RNAGym/issues/new/choose
