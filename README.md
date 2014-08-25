GalSim-Code
===========

This script is designed to utilize the Python modules GalSim and Tractor to simulate composite galaxies of certain redshifts and bulge-to-total ratios.  It then takes the AB magnitudes and errors of these galaxies and feeds them into a redshift analyzer software, ZEBRA.


Primary Goal
============

Galaxy flux is frequently composed of the flux from two galaxy components:  A bulge of red stars and a disk of blue stars.  Usually, when calculating the redshift of a galaxy with redshift analysis software, all of the galaxy's flux is used in the analysis.  The primary goal of this project is to find if it is more accurate to use only a galaxy's bulge flux in the analysis of redshifts instead of its total flux.


Dependencies
============

The script requires the following Python modules and their dependencies to run:

* GalSim: https://github.com/GalSim-developers/GalSim
* Tractor: https://github.com/dstndstn/tractor
* fitsio: https://github.com/esheldon/fitsio

As well as this software:
* ZEBRA: http://www.astro.ethz.ch/research/Projects/ZEBRA/


Setup & Files to Add
====================

For the current code setup, the ZEBRA and GalSim-Code directories should reside in the same directory.  Move the files contained the following directories to their designated directory in zebra-1.10:

* filters_to_add/		->	zebra-1.10/examples/filters/
* templates_to_add/	->	zebra-1.10/examples/templates/
* conf_files/		    ->	zebra-1.10/examples/ML_notImproved/
* scripts_to_add/		->	zebra-1.10/scripts/


Running Galaxy_Generator.py
===========================

In order to run Galaxy_Generator.py, simply run the script in a Python environment, and the script will execute based upon the parameters given.  The most relevant parameters to change between runs are found in initMain and initFluxesAndRedshifts, directly below the main run function.

The function initMain allows the enabling and disabling of different galaxy generation methods and the adjustment of the forced filter band, pixel scale, image size, noise iterations, tractor iterations, and Gaussian noise sigma.  "Basic" generates galaxy fluxes in each band using the GalSim method galsim.hsm.FindAdaptiveMom() on generated galaxy images.  "Forced" uses the same GalSim method to make a set of model images based on the forced filter band, and then uses those model images in tandem with images in each band to calculate each band's flux.  "Tractor" utilizes the Python module Tractor to create optimized galaxy models for each band and calculate their flux, optimizing parameters such as flux, deV-to-total ratio, position, and shape.  "Forced Tractor" is similar: It optimizes a model galaxy in the forced filter band, and then takes parameters that should be consistent across the bands (such as position and shape), initializes the galaxy models in each band with those parameters, and then only optimizes the flux and deV-to-total ratio of each band.  "Forced Tractor" is recommended for obtaining the intended results of this project.

In the function initFluxesAndRedshifts, the number of bulge-to-total ratios and redshifts can be changed, as well as the range of ratios and redshifts.  The lists of ratios and redshifts are generated using the numpy.linspace method.

When the script is run, the available redshift analysis data, such as redshift, lower-bound error, and higher-bound error, will be printed, along with deV-to-total ratios for the Tractor and Forced Tractor methods.  In addition, a list of signal-to-noise ratios will be printed.  A redshift comparison plot containing all calculated redshifts for the run will be created in the main directory.  

If Tractor or Forced Tractor is used, deV-to-total ratio vs filter band plots will also be created for each bulge-to-total ratio used in the run.  Furthermore, the redshifts generated will include those calculated from only the bulge flux or disk flux of the galaxy; these will be included in the final redshift plot.


References
==========

ZEBRA reference:
* Feldmann, Carollo, Porciani, Lilly et al., MNRAS 372, 564 (2006)
