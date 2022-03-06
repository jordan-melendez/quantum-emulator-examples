# quantum-emulator-examples

[![DOI](https://zenodo.org/badge/384679629.svg)](https://zenodo.org/badge/latestdoi/384679629)

This repository has an example of bound state calculations using subspace emulators
(aka, eigenvector continuation or a reduced basis method).

See `slides/` for the presentation.
There is source code sitting in `emulate/`, which gets called in the jupyter notebooks in `notebooks/`.

With `conda` installed, run
```bash
conda env create -f environment.yml
```
to create the `emulator-examples` environment.
It should install all the necessary packages, including the code in `emulate/`


Please feel free to raise an `Issue` here on GitHub if anything is broken or unclear.
I'd welcome a `Pull Request` to help improve it as well!
