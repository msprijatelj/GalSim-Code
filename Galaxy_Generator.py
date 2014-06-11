# Taken and modified from demo12.py in Galsim/examples
import sys
import os
import math
import numpy
import logging
import galsim

def main(argv):
    # Where to find and output data
    path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(path, "data/"))
    outpath = os.path.abspath(os.path.join(path, "output/"))

    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo12")

    # initialize (pseudo-)random number generator
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)

    # read in SEDs
    SED_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']
    SEDs = {}
    for SED_name in SED_names:
        SED_filename = os.path.join(datapath, '{}.sed'.format(SED_name))
        # Here we create some galsim.SED objects to hold star or galaxy spectra.  The most
        # convenient way to create realistic spectra is to read them in from a two-column ASCII
        # file, where the first column is wavelength and the second column is flux. Wavelengths in
        # the example SED files are in Angstroms, flux in flambda.  The default wavelength type for
        # galsim.SED is nanometers, however, so we need to override by specifying
        # `wave_type = 'Ang'`.
        SED = galsim.SED(SED_filename, wave_type='Ang')
        # The normalization of SEDs affects how many photons are eventually drawn into an image.
        # One way to control this normalization is to specify the flux density in photons per nm
        # at a particular wavelength.  For example, here we normalize such that the photon density
        # is 1 photon per nm at 500 nm.
        SEDs[SED_name] = SED.withFluxDensity(target_flux_density=1.0, wavelength=500)
    logger.debug('Successfully read in SEDs')

    # read in the LSST filters
    filter_names = 'ugrizy'
    filters = {}
    for filter_name in filter_names:
        filter_filename = os.path.join(datapath, 'LSST_{}.dat'.format(filter_name))
        # Here we create some galsim.Bandpass objects to represent the filters we're observing
        # through.  These include the entire imaging system throughput including the atmosphere,
        # reflective and refractive optics, filters, and the CCD quantum efficiency.  These are
        # also conveniently read in from two-column ASCII files where the first column is
        # wavelength and the second column is dimensionless flux. The example filter files have
        # units of nanometers and dimensionless throughput, which is exactly what galsim.Bandpass
        # expects, so we just specify the filename.
        filters[filter_name] = galsim.Bandpass(filter_filename)
        # For speed, we can thin out the wavelength sampling of the filter a bit.
        # In the following line, `rel_err` specifies the relative error when integrating over just
        # the filter (however, this is not necessarily the relative error when integrating over the
        # filter times an SED)
        filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)
    logger.debug('Read in filters')

    pixel_scale = 0.2 # arcseconds

    #-----------------------------------------------------------------------------------------------
    # Part B: chromatic bulge+disk galaxy

    logger.info('')
    logger.info('Starting part B: chromatic bulge+disk galaxy')
    totalFlux = 4.8
    for baseFluxRatio in xrange(0,11,1):
        fluxRatio = baseFluxRatio / 10.0
        for baseRedshift in xrange(0,16,1):
            redshift = baseRedshift / 10.0
            # make a bulge ...
            mono_bulge = galsim.DeVaucouleurs(half_light_radius=0.5)
            bulge_SED = SEDs['CWW_E_ext'].atRedshift(redshift)
            # The `*` operator can be used as a shortcut for creating a chromatic version of a GSObject:
            bulge = mono_bulge * bulge_SED
            bulge = bulge.shear(g1=0.12, g2=0.07)
            logger.debug('Created bulge component')
            # ... and a disk ...
            mono_disk = galsim.Exponential(half_light_radius=2.0)
            disk_SED = SEDs['CWW_Im_ext'].atRedshift(redshift)
            disk = mono_disk * disk_SED
            disk = disk.shear(g1=0.4, g2=0.2)
            logger.debug('Created disk component')
            # ... and then combine them.
            bulgeMultiplier, diskMultiplier = fluxRatio * totalFlux, (1 - fluxRatio) * totalFlux
            bdgal = 1.1 * (bulgeMultiplier*bulge+diskMultiplier*disk)
            PSF = galsim.Moffat(fwhm=0.6, beta=2.5)
            bdfinal = galsim.Convolve([bdgal, PSF])
            # Note that at this stage, our galaxy is chromatic but our PSF is still achromatic.  Part C)
            # below will dive into chromatic PSFs.
            logger.debug('Created bulge+disk galaxy final profile')
        
            # draw profile through LSST filters
            gaussian_noise = galsim.GaussianNoise(rng, sigma=0.02)
            for filter_name, filter_ in filters.iteritems():
                if not os.path.isdir('output_{}'.format(filter_name)):
                    os.mkdir('output_{}'.format(filter_name))
                outpath = os.path.abspath(os.path.join(path, "output_{}/".format(filter_name)))
                img = galsim.ImageF(64, 64, scale=pixel_scale)
                bdfinal.drawImage(filter_, image=img)
                img.addNoise(gaussian_noise)
                logger.debug('Created {}-band image'.format(filter_name))
                out_filename = os.path.join(outpath, 'demo12b_{}_{}_{}.fits'.format(filter_name,baseFluxRatio,baseRedshift))
                galsim.fits.write(img, out_filename)
                logger.debug('Wrote {}-band image to disk, bulge-to-total ratio = {}, redshift = {}'.format(filter_name,
                                                                                                            baseFluxRatio,
                                                                                                            baseRedshift))
                logger.info('Added flux for {}-band image: {}'.format(filter_name, img.added_flux))

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 -rgb -blue -scale limits -0.2 0.8 output/demo12b_r.fits -green -scale limits'
                +' -0.25 1.0 output/demo12b_i.fits -red -scale limits -0.25 1.0 output/demo12b_z.fits'
                +' -zoom 2 &')

if __name__ == "__main__":
    main(sys.argv)
