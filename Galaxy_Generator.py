# Taken and modified from demo12.py in Galsim/examples
import sys
import os
import shutil
import math
import matplotlib.pyplot as plt
import numpy
import logging
import galsim

# Top-level function
def main(argv):
    class Struct(): pass
    data = Struct()
    data.forced = False
    # Where to find and output data
    data.path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(data.path, "data/"))
    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO, 
                        stream=sys.stdout)
    data.logger = logger = logging.getLogger("demo12")
    # read in SEDs
    data.SEDs = readInSEDs(datapath)
    logger.debug('Successfully read in SEDs')
    # read in the LSST filters
    data.filter_names = 'ugrizy'
    data.filter_name_list = [char.upper() for char in data.filter_names]
    data.color_name_list = []
    for i in xrange(1,len(data.filter_names)):
        data.color_name_list.append("%s-%s" % (data.filter_names[i-1].upper(),
                                    data.filter_names[i].upper()))
    clearOutputs(data.filter_names)
    data.filters = readInLSST(datapath,data.filter_names)
    logger.debug('Read in filters')
    makeGalaxies(data)
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

def clearOutputs(filter_names):
    for filter_name in filter_names:
        outputDir = 'output_{}'.format(filter_name)
        if os.path.isdir(outputDir):    
            shutil.rmtree(outputDir)

def readInLSST(datapath, filter_names):
    filters = {}
    for filter_name in filter_names:
        filter_filename = os.path.join(datapath, 
                                       'LSST_{}.dat'.format(filter_name))
        filters[filter_name] = galsim.Bandpass(filter_filename)
        filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)
    return filters

# Galaxy generating functions
def makeGalaxies(data):
    data.logger.info('')
    data.logger.info('Starting to generate chromatic bulge+disk galaxy')
    # Iterations to complete
    fluxNum = 3
    redshiftNum = 3
    # Other parameters
    fluxMin, fluxMax = 0.0, 1.0
    redshiftMin, redshiftMax = 0.2, 1.0
    fluxIndex = 0
    #fig, ax = plt.subplots()
    for fluxRatio in numpy.linspace(fluxMin,fluxMax,fluxNum):   
        shiftIndex = 0
        for redshift in numpy.linspace(redshiftMin,redshiftMax,redshiftNum):           
            bdfinal = makeGalaxy(fluxRatio, redshift, data.SEDs)
            data.logger.debug('Created bulge+disk galaxy final profile')
            # draw profile through LSST filters
            applyFilter(data,fluxRatio,redshift,bdfinal)
            newPlot(data,fluxRatio,redshift,fluxIndex,shiftIndex)         
            shiftIndex += 1
        fluxIndex += 1
    figure1Setup(data)
    figure2Setup(data)

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
def applyFilter(data,fluxRatio,redshift,bdfinal):
    # initialize (pseudo-)random number generator
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)
    noiseSigma = 0.02
    avgFluxes, avgSigmas, stDevs = [], [], []
    avgColors, colorStDevs = [], []
    oldFluxList = []
    gaussian_noise = galsim.GaussianNoise(rng, sigma=noiseSigma)
    for filter_name in data.filter_names:
        # Write the data cube
        outpath = setOutput(filter_name,data.path)
        images = makeCube(data.filters,filter_name,bdfinal,gaussian_noise)
        data.logger.debug('Created {}-band image'.format(filter_name))
        fitsName = 'gal_{}_{}_{}.fits'.format(filter_name,fluxRatio,
                                              redshift)
        out_filename = os.path.join(outpath, fitsName)
        galsim.fits.writeCube(images, out_filename)
        data.logger.debug(('Wrote {}-band image to disk, '.format(filter_name))
                          + ('bulge-to-total ratio = {}, '.format(fluxRatio))
                          + ('redshift = {}'.format(redshift)))
        fluxList, sigmaList, successRate = makeFluxList(images)
        avgFlux,avgSigma,stDev = findAvgFlux(fluxList,sigmaList)
        avgFluxes.append(avgFlux), avgSigmas.append(avgSigma), stDevs.append(stDev)
        data.logger.info('Average flux for {}-band image: {}'.format(filter_name, avgFlux))
        data.logger.info('Average Sigma = {}'.format(avgSigma))                
        data.logger.info('Standard Deviation = {}'.format(stDev))  
        if oldFluxList != []:
            avgColor, colorStDev = findColors(oldFluxList, fluxList)
            avgColors.append(avgColor), colorStDevs.append(colorStDev)
            data.logger.info('Color = {}'.format(avgColor))
            data.logger.info('Color Standard Dev = {}'.format(colorStDev))
        data.logger.info('Success Rate = {}'.format(successRate))
        oldFluxList = fluxList
    data.avgFluxes, data.stDevs = avgFluxes, stDevs
    data.avgColors, data.colorStDevs = avgColors, colorStDevs

def setOutput(filter_name,path):
    outputDir = 'output_{}'.format(filter_name)
    if not os.path.isdir(outputDir):
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
    return images

def makeCubeImages(img, gaussian_noise, noiseIterations):
    images = []
    # Create many images with different noises, compile into a cube
    for i in xrange(noiseIterations):
        newImg = img.copy()
        newImg.addNoise(gaussian_noise)
        images.append(newImg)
    return images

def findAvgFlux(fluxList, sigmaList):
    avgFlux = numpy.mean(fluxList)
    avgSigma = numpy.mean(sigmaList)
    stDev = numpy.std(fluxList)
    return avgFlux, avgSigma, stDev

def makeFluxList(images):
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
        """
        else:
            fluxList.append(0)
            sigmaList.append(0)
        """
    successRate = successes/totalAttempts
    return fluxList, sigmaList, successRate

def findColors(oldFluxList, fluxList):
    colorList = []
    for i in xrange(min(len(oldFluxList),len(fluxList))):
        newColor = -2.5*math.log10(oldFluxList[i]/fluxList[i])
        colorList.append(newColor)
    avgColor = numpy.mean(colorList)
    colorStDev = numpy.std(colorList)
    return avgColor, colorStDev

def newPlot(data,fluxRatio,redshift,fluxIndex,shiftIndex):
    # colors = ["b","c","g","y","r","m"]
    colors = ["g","y","r","m"]
    shapes = ["o","^","s","p","h","D"]
    n_groups, m_groups = len(data.avgFluxes), len(data.avgColors)
    plt.figure(1)
    index, filter_name_list = range(n_groups), data.filter_name_list
    avgFluxes, stDevs = data.avgFluxes, data.stDevs
    plt.errorbar(index, avgFluxes, stDevs, None, barsabove = True, 
                 marker = "%s" % shapes[fluxIndex], linestyle = "none", 
                 mfc = "%s" % colors[shiftIndex], capsize = 10, ecolor = "k",
                 label = "{} B/T, {} redshift".format(fluxRatio,redshift))
    plt.xticks(index, filter_name_list)
    plt.figure(2)
    colorIndex, color_name_list = range(m_groups), data.color_name_list
    avgColors, colorStDevs = data.avgColors, data.colorStDevs
    plt.errorbar(colorIndex, avgColors, colorStDevs, None, barsabove = True,
                 marker = "%s" % shapes[fluxIndex], linestyle = "none",
                 mfc = "%s" % colors[shiftIndex], capsize = 10, ecolor = "k",
                 label = "{} B/T, {} redshift".format(fluxRatio,redshift))
    plt.xticks(colorIndex, color_name_list)

def figure1Setup(data):
    plt.figure(1)
    plt.xlim([-1,len(data.filter_names)])
    plt.xlabel('Band')
    plt.ylabel('Magnitude of the Flux')
    plt.title('Flux across bands at varied flux ratios and redshifts')
    plt.grid()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()
    plt.savefig("Flux_Plot.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
    
def figure2Setup(data):
    plt.figure(2)
    plt.xlim([-1,len(data.color_name_list)])
    plt.xlabel('Band')
    plt.ylabel('Color')
    plt.title('Color across bands at varied flux ratios and redshifts')
    plt.grid()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()
    plt.savefig("Color_Plot.png",bbox_extra_artists=(lgd,),bbox_inches='tight')

# flux = sum(Image*model)/sum(model**2)
# use parameter to pick between original and new model
# make Zebra-friendly output file

if __name__ == "__main__":
    main(sys.argv)
