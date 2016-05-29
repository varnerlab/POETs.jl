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
which runs test examples from the ``test`` directory. Lastly, to delete the JuPOETs package use the command:

```
Pkg.rm("POETs")
```

## Sample code
We've included sample code to help you get started with JuPOETs in your project. The sample can be found in the ``sample/biochemical`` subdirectory. The sample encodes the estimation of an ensemble of biochemical model parameters from four conflicting training data sets. The driver for this sample is given in the ``run_biochemical_test.jl`` file, while the ``objective``,``neighbor``,``cooling`` and ``acceptance`` functions (required by JuPOETs) are encoded in the ``hcmem_lib.jl`` library.