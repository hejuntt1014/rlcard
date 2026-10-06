# Third-party notices

## RLCard

This project includes code from [RLCard](https://github.com/datamllab/rlcard).
The upstream project carries the MIT license in [LICENSE.md](LICENSE.md):

Copyright (c) 2019 DATA Lab at Texas A&M University.

The copyright and permission notice applies to the included upstream material.
Original source-file copyright and attribution notices are retained.

## DMC and logging components

The following source files carry Apache License 2.0 notices:

- `rlcard/agents/dmc_agent/model.py`, `context_model.py`, `trainer.py`, and `utils.py`:
  Copyright 2021 RLCard Team of Texas A&M University;
  Copyright 2021 DouZero Team of Kwai.
- `rlcard/agents/dmc_agent/file_writer.py`:
  Copyright (c) Facebook, Inc. and its affiliates.

The full license is included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
These files include modifications for the training functionality supplied by this
project and retain their source-level license notices.

Dependency packages are distributed under their respective licenses. Project
naming and repository ownership do not replace the notices attached to included
third-party source code.

## A3 evaluation reference

`evaluation/` includes game and reference-policy code supplied from the user's
A3 project. Its source boundaries and adaptations are recorded in
[evaluation/SOURCES.md](evaluation/SOURCES.md). No server configuration,
credentials or pretrained model binaries are included.
