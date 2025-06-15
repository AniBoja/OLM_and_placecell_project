# OLM project analysis

This repository contains code used to analyse place cell data for the paper Udakis et al 2025

### Python_placecell_analysis
This contains the python code in jupyter notebooks used to analyse placecell data

- Example data in the form of rate maps is included in .mat files in the data folder these mice have mCherry expressed in OLM cells and GCaMP6f in CA1 pyramidal neurons.
- Data can be loaded and selected based on a custom data loader class 
- Each experiment contains 10 min sessions on a particular linear track, data can be aligned to each session using the data loader

To run the python code for place cell analysis we recommend creating a new python environment as described [here](python_placecell_analysis/env_setup.md)


---

[### MATLAB_preprocessing](MATLAB_preprocessing)

This set of Live MATLAB files can be used to analyse the place cells.
Files include:
- Tracking the mouse behaviour from behaviour video 
- Processing and aligning calcium events exported from inscopix software 
- Combining location and events to calculate place cells 

Also included is examples on how to export data for further analysis using Python
