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
    logging.basicConfig(format="%(message)s", level=logging.INFO, 
                        stream=sys.stdout)
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
        SEDs[SED_name] = SED.withFluxDensity(target_flux_density=1.0, 
                                             wavelength=500)
    logger.debug('Successfully read in SEDs')

    # read in the LSST filters
    filter_names = 'ugrizy'
    filters = {}
    for filter_name in filter_names:
        filter_filename = os.path.join(datapath, 
                                       'LSST_{}.dat'.format(filter_name))
        filters[filter_name] = galsim.Bandpass(filter_filename)
        filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)
    logger.debug('Read in filters')

    pixel_scale = 0.2 # arcseconds

    #-----------------------------------------------------------------------------------------------

    logger.info('')
    logger.info('Starting part B: chromatic bulge+disk galaxy')
    totalFlux = 4.8
    # Iterations to complete
    fluxNum = 3
    redshiftNum = 3
    noiseIterations = 100
    # Other parameters
    fluxMin, fluxMax = 0.0, 1.0
    redshiftMin, redshiftMax = 0.2, 1.0
    bulgeG1, bulgeG2 = 0.12, 0.07
    diskG1, diskG2 = 0.4, 0.2
    mono_bulge_HLR, mono_disk_HLR = 0.5, 2.0
    psf_FWHM, psf_beta = 0.6, 2.5
    noiseSigma = 0.02
    for fluxRatio in numpy.linspace(fluxMin,fluxMax,fluxNum):   
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
            bulgeMultiplier = fluxRatio * totalFlux
            diskMultiplier = (1 - fluxRatio) * totalFlux
            bdgal = 1.1 * (bulgeMultiplier*bulge+diskMultiplier*disk)
            PSF = galsim.Moffat(fwhm=psf_FWHM, beta=psf_beta)
            bdfinal = galsim.Convolve([bdgal, PSF])
            logger.debug('Created bulge+disk galaxy final profile')
        
            # draw profile through LSST filters
            gaussian_noise = galsim.GaussianNoise(rng, sigma=noiseSigma)
            for filter_name, filter_ in filters.iteritems():
                if not os.path.isdir('output_{}'.format(filter_name)):
                    os.mkdir('output_{}'.format(filter_name))
                outDir = "output_{}/".format(filter_name)
                outpath = os.path.abspath(os.path.join(path, outDir))
                img = galsim.ImageF(64, 64, scale=pixel_scale)
                bdfinal.drawImage(filter_, image=img)
                images = []
                # Create many images with different noises, compile into a cube
                for i in xrange(noiseIterations):
                    newImg = img.copy()
                    newImg.addNoise(gaussian_noise)
                    images.append(newImg)
                totalFlux = 0
                for image in images:
                    flux = galsim.hsm.FindAdaptiveMom(image)
                    totalFlux += flux.moments_amp
                avgFlux = totalFlux/len(images)
                logger.debug('Created {}-band image'.format(filter_name))
                fitsName = 'gal_{}_{}_{}.fits'.format(filter_name,fluxRatio,
                                                      redshift)
                out_filename = os.path.join(outpath, fitsName)
                galsim.fits.writeCube(images, out_filename)
                logger.debug('Wrote {}-band image to disk, '
                             + 'bulge-to-total ratio = {}, '
                             + 'redshift = {}'.format(filter_name, fluxRatio, 
                                                      redshift))
                logger.info('Average flux for {}-band image: {}'.format(filter_name, avgFlux))                

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 -rgb -blue -scale limits -0.2 0.8 output_r/gal_r.fits -green -scale limits'
                +' -0.25 1.0 output_i/gal_i.fits -red -scale limits -0.25 1.0 output_z/gal_z.fits'
                +' -zoom 2 &')

if __name__ == "__main__":
    main(sys.argv)
