# Taken and modified from demo12.py in Galsim/examples
import sys
import os
import shutil
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy
import logging
import galsim

# Top-level function
def main(argv):
    # Where to find and output data
    path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(path, "data/"))
    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO, 
                        stream=sys.stdout)
    logger = logging.getLogger("demo12")
    # read in SEDs
    SEDs = readInSEDs(datapath)
    logger.debug('Successfully read in SEDs')
    # read in the LSST filters
    filter_names = 'ugrizy'
    filters = readInLSST(datapath,filter_names)
    logger.debug('Read in filters')
    makeGalaxies(SEDs,filters,filter_names,path,logger)
    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 -rgb -blue -scale limits -0.2 0.8 output_r/gal_r.fits -green -scale limits'
                +' -0.25 1.0 output_i/gal_i.fits -red -scale limits -0.25 1.0 output_z/gal_z.fits'
                +' -zoom 2 &')

# Initialization functions
def readInSEDs(datapath):
    SED_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']
    SEDs = {}
    for SED_name in SED_names:
        SED_filename = os.path.join(datapath, '{}.sed'.format(SED_name))
        SED = galsim.SED(SED_filename, wave_type='Ang')
        SEDs[SED_name] = SED.withFluxDensity(target_flux_density=1.0, 
                                             wavelength=500)
    return SEDs

def readInLSST(datapath, filter_names):
    filters = {}
    for filter_name in filter_names:
        filter_filename = os.path.join(datapath, 
                                       'LSST_{}.dat'.format(filter_name))
        filters[filter_name] = galsim.Bandpass(filter_filename)
        filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)
    return filters

# Galaxy generating functions
def makeGalaxies(SEDs, filters, filter_names, path, logger):
    logger.info('')
    logger.info('Starting to generate chromatic bulge+disk galaxy')
    # Iterations to complete
    fluxNum = 3
    redshiftNum = 3
    # Other parameters
    fluxMin, fluxMax = 0.0, 1.0
    redshiftMin, redshiftMax = 0.2, 1.0
    # Make the pdf file
    pp = PdfPages('FluxPlots.pdf')    
    for fluxRatio in numpy.linspace(fluxMin,fluxMax,fluxNum):   
        for redshift in numpy.linspace(redshiftMin,redshiftMax,redshiftNum):           
            bdfinal = makeGalaxy(fluxRatio, redshift, SEDs)
            logger.debug('Created bulge+disk galaxy final profile')
            # draw profile through LSST filters
            plotData = applyFilter(fluxRatio,redshift,filters,filter_names,
                                   path,bdfinal,logger)
            drawPlot(plotData,pp,fluxRatio,redshift)
    pp.close()

# Individual galaxy generator functions
def makeGalaxy(fluxRatio, redshift, SEDs):
    bulge, disk = makeBulge(redshift, SEDs), makeDisk(redshift, SEDs)  
    bdfinal = makeFinal(fluxRatio, bulge, disk)
    return bdfinal

def makeBulge(redshift, SEDs):
    bulgeG1, bulgeG2, mono_bulge_HLR = 0.12, 0.07, 0.5
    mono_bulge = galsim.DeVaucouleurs(half_light_radius=mono_bulge_HLR)
    bulge_SED = SEDs['CWW_E_ext'].atRedshift(redshift)
    bulge = mono_bulge * bulge_SED
    bulge = bulge.shear(g1=bulgeG1, g2=bulgeG2)
    return bulge

def makeDisk(redshift, SEDs):
    diskG1, diskG2, mono_disk_HLR = 0.4, 0.2, 2.0
    mono_disk = galsim.Exponential(half_light_radius=mono_disk_HLR)
    disk_SED = SEDs['CWW_Im_ext'].atRedshift(redshift)
    disk = mono_disk * disk_SED
    disk = disk.shear(g1=diskG1, g2=diskG2)
    return disk
    
def makeFinal(fluxRatio, bulge, disk):
    totalFlux = 4.8
    psf_FWHM, psf_beta = 0.6, 2.5
    bulgeMultiplier = fluxRatio * totalFlux
    diskMultiplier = (1 - fluxRatio) * totalFlux
    bdgal = 1.1 * (bulgeMultiplier*bulge+diskMultiplier*disk)
    PSF = galsim.Moffat(fwhm=psf_FWHM, beta=psf_beta)
    bdfinal = galsim.Convolve([bdgal, PSF])    
    return bdfinal

# Filter application, generation of images and data cube
def applyFilter(fluxRatio,redshift,filters,filter_names,path,bdfinal,logger):
    # initialize (pseudo-)random number generator
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)
    noiseSigma = 0.02
    avgFluxes, avgSigmas, stDevs = [], [], []
    filter_name_list = [char.upper() for char in filter_names]
    gaussian_noise = galsim.GaussianNoise(rng, sigma=noiseSigma)
    for filter_name in filter_names:
        outpath = setOutput(filter_name,path)
        avgFlux,avgSigma,stDev,successRate,images = makeCube(filters,
                                                             filter_name,
                                                             bdfinal,
                                                             gaussian_noise)
        avgFluxes.append(avgFlux), avgSigmas.append(avgSigma), stDevs.append(stDev)
        # Write the data cube
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
        logger.info('Average Sigma = {}'.format(avgSigma))                
        logger.info('Standard Deviation = {}'.format(stDev))                
        logger.info('Success Rate = {}'.format(successRate))
    plotData = zip(filter_name_list,avgFluxes,stDevs)
    return plotData

def setOutput(filter_name,path):
    outputDir = 'output_{}'.format(filter_name)
    if os.path.isdir(outputDir):
        shutil.rmtree(outputDir)
    os.mkdir(outputDir)
    outDir = "output_{}/".format(filter_name)
    outpath = os.path.abspath(os.path.join(path, outDir))
    return outpath

def makeCube(filters,filter_name,bdfinal,gaussian_noise):
    pixel_scale = 0.2 # arcseconds
    # initialize (pseudo-)random number generator
    noiseIterations = 100
    filter_ = filters[filter_name]
    img = galsim.ImageF(64, 64, scale=pixel_scale)
    bdfinal.drawImage(filter_, image=img)
    images = makeCubeImages(img, gaussian_noise, noiseIterations)
    avgFlux, avgSigma, stDev, successRate = findAvgFlux(images)
    return avgFlux,avgSigma,stDev,successRate,images

def makeCubeImages(img, gaussian_noise, noiseIterations):
    images = []
    # Create many images with different noises, compile into a cube
    for i in xrange(noiseIterations):
        newImg = img.copy()
        newImg.addNoise(gaussian_noise)
        images.append(newImg)
    return images

def findAvgFlux(images):
    fluxList = []
    sigmaList = []
    successes = 0.0
    totalAttempts = len(images)
    for image in images:
        adaptiveMom = galsim.hsm.FindAdaptiveMom(image, strict = False)
        flux, sigma = adaptiveMom.moments_amp, adaptiveMom.moments_sigma
        if not (flux < 0):
            successes += 1
            fluxList.append(flux)
            sigmaList.append(sigma)
    avgFlux = sum(fluxList)/totalAttempts
    avgSigma = sum(sigmaList)/totalAttempts
    successRate = successes/totalAttempts
    stDev = calcStDev(fluxList, avgFlux)
    return avgFlux, avgSigma, stDev, successRate

def calcStDev(entries, mean):
    devTotal = 0
    for entry in entries:
        devTotal += (mean - entry) ** 2
    stDev = math.sqrt(devTotal/len(entries))
    return stDev

# drawPlot is modified from barchart_demo.py from 
# http://matplotlib.org/mpl_examples/pylab_examples/barchart_demo.py
def drawPlot(plotData,pp,fluxRatio,redshift):
    n_groups = len(plotData)
    filter_name_list = [plotData[i][0] for i in xrange(n_groups)]
    avgFluxes = [plotData[i][1] for i in xrange(n_groups)]
    stDevs = [plotData[i][2] for i in xrange(n_groups)]
    colors = ["k","b","g","y","r","m"]
    index = numpy.arange(n_groups)
    fig, ax = plt.subplots()
    bar_width = 0.5
    opacity = 0.60
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, avgFluxes, bar_width, bottom = 0, align = "center",
                     alpha = opacity,
                     color=colors,
                     yerr=stDevs,
                     error_kw=error_config,
                     label='Flux',clip_on = True)
    plt.xlabel('Band')
    plt.ylabel('Magnitude of the Flux')
    plt.ylim(0)
    plt.title('Flux with {} B/T ratio & {} redshift'.format(fluxRatio,redshift))
    plt.xticks(index, filter_name_list)
    plt.show()
    pp.savefig()
    
if __name__ == "__main__":
    main(sys.argv)
