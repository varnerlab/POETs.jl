## Introduction
POETs.jl is a Julia package that implements the Pareto Optimal Ensemble Techniques (POETs) method for multiobjective optimization. The POETs algorithm has been published:

[Song S, Chakrabarti A, and J. Varner (2010) Identifying ensembles of signal transduction models using Pareto Optimal Ensemble Techniques (POETs). Biotechnology Journal DOI: 10.1002/biot.201000059](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3021968/)

## Installation
Within Julia, use the ``clone`` command of the package manager to download and install the POETs repository:

```
Pkg.clone("git://github.com/varnerlab/POETs.jl")
```
To use POETs in your project (following installation) simply issue the command:

```
using POETs
```
To test the POETs installation use:

```
Pkg.test("POETs")
```
which runs test examples from the ``test`` directory.