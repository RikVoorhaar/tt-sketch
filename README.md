# TT-sketch

Fast sketching algorithms for computing Tensor Train decompositions of a variety of tensorial data.

This software implements the algorithms discussed in the preprint arXiv:2208.XXXXX.

## Installation

The `tt-sketch` package is available on [PyPI](https://pypi.python.org/pypi/tt-sketch), and can be installed
using `pip` by running 

```sh
pip install tt-sketch
```

Alternatively you can install it by first cloning this repository:
```sh
git clone git@github.com:RikVoorhaar/tt-sketch.git
cd tt-sketch
pip install .
```

## Reproducing numerical experiments
All numerical experiments in the preprint can be reproduced using the scripts starting with `plot_` in the `scripts` directory. All experiments were produced using version 1.1 of this software. The dependencies for running these scripts, as well as running the tests or building the documentation, are listed in `environment.yml`.

## Documentation
The documentation for this project lives here: [tt-sketch.readthedocs.io](https://tt-sketch.readthedocs.io).

## Credits
All code for this project is written by [Rik Voorhaar](https://rikvoorhaar.com), in a joint project with  Daniel Kressner and Bart Vandereycken. This work was supported by the Swiss National Science Foundation under [research project 192363](https://data.snf.ch/grants/grant/192363).

This software is free to use and edit. When using this software for academic purposes, please cite the following preprint:

```
@article{
    title = {Streaming tensor train approximation},
    journal = {arXiv:2208.XXXXX},
    author = {Kressner, Daniel and Vandereycken, Bart and Voorhaar, Rik},
    doi = {},
    year = {2022}, 
}
```

## Contributing
All contributions or suggestions are welcome. Feel free to open an issue with suggestions, or submit a pull request. 