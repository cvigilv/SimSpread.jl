# SimSpread.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cvigilv.github.io/SimSpread.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cvigilv.github.io/SimSpread.jl/dev/)
[![Build Status](https://github.com/cvigilv/SimSpread.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cvigilv/SimSpread.jl/actions/workflows/CI.yml?query=branch%3Amain)

<img src="/docs/src/assets/logo.png" align="right" style="padding-left:10px;" width="250"/>

## What is SimSpread.jl?

SimSpread.jl is a Julia implementation of the SimSpread formalism for link prediction.
SimSpread is a novel approach for predicting interactions between two distinct set of
nodes, query and target nodes, using a similarity measure vector between query nodes as
a meta-description in combination with the network-based inference for link prediction.

Originally developed for the prediction of pharmacological targets for a chemical compound,
this packages generalizes the method to enable the prediction of links between any pair of
nodes, e.g., user-object, reader-book, buyer-product, etc.

## Installation

The package can be installed using the `Pkg` Julia package directly from the Julia prompt with
the following:
~~~ julia
using Pkg; Pkg.add(url = "https://github.com/cvigilv/SimSpread.jl.git")
~~~

For the newest version (development version, may break):
~~~ julia
using Pkg; Pkg.add(url = "https://github.com/cvigilv/SimSpread.jl.git", rev = "develop")
~~~

## Examples
For examples, please refer to the tutorial notebooks here.

## Support

Please [open an issue](https://github.com/cvigilv/SimSpread.jl/issues/new) for support.

## Contributing

Please contribute using [Github Flow](https://guides.github.com/introduction/flow/).
Create a branch, add commits, and [open a pull request](https://github.com/cvigilv/SimSpread.jl/compare/).

## License

MIT license, refer to [LICENSE](./LICENSE) for more information.
