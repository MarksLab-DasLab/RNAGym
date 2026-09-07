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
  <a href="https://github.com/MarksLab-DasLab/RNAGym/blob/main/LICENSE"><img src="https://img.shields.io/github/license/MarksLab-DasLab/RNAGym" alt="License"></a>
  <a href="https://github.com/MarksLab-DasLab/RNAGym/actions/workflows/fitness.yml"><img src="https://github.com/MarksLab-DasLab/RNAGym/actions/workflows/fitness.yml/badge.svg" alt="Fitness CI"></a>
  <a href="rnagym/fitness/README.md#quality-checks"><img src="https://img.shields.io/badge/coverage-required_80%25-blue" alt="Full fitness coverage requires 80 percent"></a>
  <a href="https://rnagym.org"><img src="https://img.shields.io/badge/website-rnagym.org-orange" alt="Website"></a>
</p>

[RNAGym][5] provides datasets, baseline models and evaluation workflows for RNA
fitness, secondary structure and tertiary structure prediction.
See the [leaderboards][3] for results and [releases][4] for stable versions.

## Getting started

Follow the setup, dataset preparation and reproduction instructions for the
[fitness][0], [2D structure][6], and [3D structure][7] benchmarks.

Model checkpoints are shared across all three benchmarks under
`/n/lw_groups/marks/ckpt/<model>`. Set `RNAGYM_CHECKPOINT_DIR` to use another
location. Model environments and source checkouts stay under each benchmark's
`.pixi/` directory.

## Contributing

Contributions of models, datasets and fixes are welcome. Use the [issue tracker][2]
to report a problem or propose a change. Run the relevant benchmark's tests and lint
before submitting a pull request.

[0]: rnagym/fitness/README.md
[2]: https://github.com/MarksLab-DasLab/RNAGym/issues/new/choose
[3]: leaderboard/
[4]: https://github.com/MarksLab-DasLab/RNAGym/releases
[5]: https://rnagym.org/
[6]: rnagym/s2d/README.md
[7]: rnagym/s3d/README.md
