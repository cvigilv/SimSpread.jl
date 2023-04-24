*WARNING: this is a highly experimental package that is in its infancy. Expect
breaking changes from time to time and change in functionality provided as it
matures into its final state.*

# SimSpread.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cvigilv.github.io/SimSpread.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cvigilv.github.io/SimSpread.jl/dev/)
[![Build Status](https://github.com/cvigilv/SimSpread.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cvigilv/SimSpread.jl/actions/workflows/CI.yml?query=branch%3Amain)

<img src="/docs/src/assets/SimSpread_logo.png" align="right" style="padding-left:10px;" width="300"/>

## What is SimSpread.jl?

SimSpread.jl is a Julia implementation of the SimSpread formalism and its necessary functions
for link prediction. SimSpread is a novel approach for predicting interactions between source
and targets nodes using a similarity measure vector between source nodes as a meta-description
in combination with the network-based inference formalism for link prediction.

## Installation

In the Julia REPL:
~~~ julia
]add https://github.com/cvigilv/SimSpread.jl.git
~~~
The ] character starts the Julia package manager. Press the backspace key to return to the Julia prompt.

Alternatively, the package can be installed using the `Pkg` Julia package:
~~~ julia
using Pkg
Pkg.add("https://github.com/cvigilv/SimSpread.jl.git")
~~~

For the newest version (development version, may break):
~~~ julia
]add https://github.com/cvigilv/SimSpread.jl.git#develop
~~~
or
~~~ julia
using Pkg
Pkg.add("https://github.com/cvigilv/SimSpread.jl.git#develop")
~~~

## Examples
For examples, please refer to the tutorial notebooks here.

## Support

Please [open an issue](https://github.com/cvigilv/SimSpread.jl/issues/new) for support.

## Contributing

Please contribute using [Github Flow]
(https://guides.github.com/introduction/flow/). Create a branch, add
commits, and [open a pull request](https://github.com/cvigilv/SimSpread.jl/compare/).

## License

MIT license, refer to [LICENSE](./LICENSE) for more information.
