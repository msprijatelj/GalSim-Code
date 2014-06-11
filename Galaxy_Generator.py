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
        SED = galsim.SED(SED_filename, wave_type='Ang')
        SEDs[SED_name] = SED.withFluxDensity(target_flux_density=1.0, wavelength=500)
    logger.debug('Successfully read in SEDs')

    # read in the LSST filters
    filter_names = 'ugrizy'
    filters = {}
    for filter_name in filter_names:
        filter_filename = os.path.join(datapath, 'LSST_{}.dat'.format(filter_name))
        filters[filter_name] = galsim.Bandpass(filter_filename)
        filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)
    logger.debug('Read in filters')

    pixel_scale = 0.2 # arcseconds

    #-----------------------------------------------------------------------------------------------
    # Part B: chromatic bulge+disk galaxy

    logger.info('')
    logger.info('Starting part B: chromatic bulge+disk galaxy')
    totalFlux = 4.8
    fluxNum = 11
    redshiftMin, redshiftMax = 0.0, 1.5
    redshiftNum = int((redshiftMax - redshiftMin) * 10 + 1)
    bulgeG1, bulgeG2 = 0.12, 0.07
    diskG1, diskG2 = 0.4, 0.2
    mono_bulge_HLR, mono_disk_HLR = 0.5, 2.0
    psf_FWHM, psf_beta = 0.6, 2.5
    noiseSigma = 0.02
    # Bulge-to-total ratio can only range from 0 to 1; the amount of bulge flux cannot be greater 
    # than the total amount of flux.
    for fluxRatio in numpy.linspace(0.0,1.0,fluxNum):   
        for redshift in numpy.linspace(redshiftMin,redshiftMax,redshiftNum):
            mono_bulge = galsim.DeVaucouleurs(half_light_radius=mono_bulge_HLR)
            bulge_SED = SEDs['CWW_E_ext'].atRedshift(redshift)
            bulge = mono_bulge * bulge_SED
            bulge = bulge.shear(g1=bulgeG1, g2=bulgeG2)
            logger.debug('Created bulge component')
            mono_disk = galsim.Exponential(half_light_radius=mono_disk_HLR)
            disk_SED = SEDs['CWW_Im_ext'].atRedshift(redshift)
            disk = mono_disk * disk_SED
            disk = disk.shear(g1=diskG1, g2=diskG2)
            logger.debug('Created disk component')
            bulgeMultiplier, diskMultiplier = fluxRatio * totalFlux, (1 - fluxRatio) * totalFlux
            bdgal = 1.1 * (bulgeMultiplier*bulge+diskMultiplier*disk)
            PSF = galsim.Moffat(fwhm=psf_FWHM, beta=psf_beta)
            bdfinal = galsim.Convolve([bdgal, PSF])
            logger.debug('Created bulge+disk galaxy final profile')
        
            # draw profile through LSST filters
            gaussian_noise = galsim.GaussianNoise(rng, sigma=noiseSigma)
            for filter_name, filter_ in filters.iteritems():
                if not os.path.isdir('output_{}'.format(filter_name)):
                    os.mkdir('output_{}'.format(filter_name))
                outpath = os.path.abspath(os.path.join(path, "output_{}/".format(filter_name)))
                img = galsim.ImageF(64, 64, scale=pixel_scale)
                bdfinal.drawImage(filter_, image=img)
                img.addNoise(gaussian_noise)
                logger.debug('Created {}-band image'.format(filter_name))
                out_filename = os.path.join(outpath, 'demo12b_{}_{}_{}.fits'.format(filter_name,fluxRatio,redshift))
                galsim.fits.write(img, out_filename)
                logger.debug('Wrote {}-band image to disk, bulge-to-total ratio = {}, redshift = {}'.format(filter_name,
                                                                                                            fluxRatio,
                                                                                                            redshift))
                logger.info('Added flux for {}-band image: {}'.format(filter_name, img.added_flux))

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 -rgb -blue -scale limits -0.2 0.8 output_r/demo12b_r.fits -green -scale limits'
                +' -0.25 1.0 output_i/demo12b_i.fits -red -scale limits -0.25 1.0 output_z/demo12b_z.fits'
                +' -zoom 2 &')

if __name__ == "__main__":
    main(sys.argv)
