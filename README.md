# exvis

exvis **vis**ualizes MIP instances in .mps & .lp format based on their **ex**pressions (variables and constraints) as a graph.

## Installation

```bash
pip install exvis
```

## Usage

```bash
exvis --input data/example/enlight_hard.mps.gz
```

For further options, see `exvis --help`.

## Preview

Shown here are visualizations of instances from the [MIPLIB 2017 set][MIPLIB2017].

| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" alt="enlight_hard" src="data/preview/enlight_hard.mps.gz.light.png"> enlight_hard | <img width="1604" alt="eva1aprime5x5opt" src="data/preview/eva1aprime5x5opt.mps.gz.light.png"> eva1aprime5x5opt |
|<img width="1604" alt="neos-1171737" src="data/preview/neos-1171737.mps.gz.light.png"> neos-1171737 | <img width="1604" alt="neos-1324574" src="data/preview/neos-1324574.mps.gz.dark.png"> neos-1324574 |
|<img width="1604" alt="mine-90-10" src="data/preview/mine-90-10.mps.gz.dark.png"> mine-90-10 | <img width="1604" alt="peg-solitaire-a3" src="data/preview/peg-solitaire-a3.mps.gz.dark.png"> peg-solitaire-a3 |

Many more images of files from the [MIPLIB 2017 set][MIPLIB2017] can be found [here](https://drive.google.com/drive/folders/10xykMGRfd1bMyWigQr8vPto80EeoqU_5?usp=sharing).

## Known issues

- The applied networkx functionality may eat up a lot of memory for large instances or run indefinitely.

[MIPLIB2017]: https://miplib.zib.de/
