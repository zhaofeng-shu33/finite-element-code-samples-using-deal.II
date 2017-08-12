finite element code samples using deal.II
========================

## Introduction
This code collects FEM codes of my summer practice, focusing mainly on solving elastic mechanics problems.

## How to build
I use Visual Studio to add include directories and lib paths. For cmd line build, CMakeLists.txt is provided.
However, you still need to set the input file name,include directories and lib paths in CMakeLists.txt.
For release build, change *CMAKE_BUILD_TYPE*  to Release in CMakeCache.txt

## Pre and Post Processing
I use Python script to deal with some complexity of pre and post processing. For purpose of visualization, [gmsh](http://www.gmsh.info) is used for msh file
and [paraview](http://www.paraview.org/) is used for vtk file.

## Acnowledgement
For info about how to use deal.II, see [http://dealii.org/](http://dealii.org/)
