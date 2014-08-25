GalSim-Code
===========

This script is designed to utilize the Python modules GalSim and Tractor to simulate composite galaxies of certain redshifts and bulge-to-total ratios.  It then takes the AB magnitudes and errors of these galaxies and feeds them into a redshift analyzer software, ZEBRA.  


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




References
==========

ZEBRA reference:
* Feldmann, Carollo, Porciani, Lilly et al., MNRAS 372, 564 (2006)
