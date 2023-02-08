*WARNING: this is a highly experimental package that is in its infancy. Expect
breaking changes from time to time and change in functionality provided as it
matures into its final state.*

# SimSpread

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cvigilv.github.io/SimSpread.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cvigilv.github.io/SimSpread.jl/dev/)
[![Build Status](https://github.com/cvigilv/SimSpread.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cvigilv/SimSpread.jl/actions/workflows/CI.yml?query=branch%3Amain)

# Installation
In your terminal
~~~ bash
git clone git@github.com:cvigilv/SimSpread.jl.git
cd SimSpread.jl
julia
~~~
In the REPL:
~~~ julia
]dev
~~~

## Roadmap
For release 1.0 and FOSS publication, the following must be done:
- Improve function naming to better align with SimSpread language
- Add documentation
  - Docstring all functions
  - Add Documenter.jl landing page
  - Add tutorial and example used of package
- Add performance metrics:
  - AuROC
  - AuPRC
  - BEDROC
  - RIE
  - R@L
  - P@L
  - eR@L
  - eP@L
  - mean(Acc, bAcc, F1, MCC)
  - max(Acc, bAcc, F1, MCC)
- Add example data and tutorial
  - Add Yamanishi (2008) Nuclear Receptor dataset
  - Add example of prediction of compounds
  - Add performance assessment using available functions in package
- Add unit testing
  - Export docstrings tests to 'tests' directory
  - Cover all functions with unit tests
- Archive old version of package and link to this version
- Link to publications
