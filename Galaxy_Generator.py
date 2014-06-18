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
    data.forced = True
    data.forcedFilter = "r"
    data.imageSize = 64
    data.pixel_scale = 0.2 # arcseconds
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
            data.bdfinal = makeGalaxy(fluxRatio, redshift, data.SEDs)
            data.logger.debug('Created bulge+disk galaxy final profile')
            # draw profile through LSST filters
            applyFilter(data,fluxRatio,redshift)
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
def applyFilter(data,fluxRatio,redshift):
    # initialize (pseudo-)random number generator
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)
    noiseSigma = 0.02
    avgFluxes, stDevs = [], []
    avgColors, colorStDevs = [], []
    oldFluxList = []
    data.gaussian_noise = galsim.GaussianNoise(rng, sigma=noiseSigma)
    if data.forced == True:
        models = makeModels(data)
    for filter_name in data.filter_names:
        # Write the data cube
        outpath = setOutput(filter_name,data.path)
        images = makeCube(data,filter_name)
        data.logger.debug('Created {}-band image'.format(filter_name))
        fitsName = 'gal_{}_{}_{}.fits'.format(filter_name,fluxRatio,
                                              redshift)
        out_filename = os.path.join(outpath, fitsName)
        galsim.fits.writeCube(images, out_filename)
        data.logger.debug(('Wrote {}-band image to disk, '.format(filter_name))
                          + ('bulge-to-total ratio = {}, '.format(fluxRatio))
                          + ('redshift = {}'.format(redshift)))
        if data.forced == False:
            fluxList, successRate = makeFluxList(images)
        else:
            fluxList, successRate = makeForcedFluxList(images,models), 1.0
        avgFlux,stDev = findAvgFlux(fluxList)
        avgFluxes.append(avgFlux), stDevs.append(stDev)
        data.logger.info('Average flux for {}-band image: {}'.format(filter_name, avgFlux))
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

def makeModels(data):
    images = makeCube(data, data.forcedFilter)
    models = []
    bounds = galsim.BoundsI(1,64,1,64)
    for image in images:
        gal_moments = galsim.hsm.FindAdaptiveMom(image)
        flux = gal_moments.moments_amp
        sigma = gal_moments.moments_sigma
        g1 = gal_moments.observed_shape.getG1()
        g2 = gal_moments.observed_shape.getG2()        
        newModel = galsim.Gaussian(flux = flux, sigma = sigma)
        newModel = newModel.shear(g1=g1,g2=g2)
        #model = galsim.Image(bounds, scale=data.pixel_scale)
        model = galsim.Image(bounds, scale=data.pixel_scale*5)
        newModel.drawImage(image = model)
        model.addNoise(data.gaussian_noise)
        #print model.added_flux, newModel.getFlux()
        file_name = os.path.join('output','test.fits')
        model.write(file_name)
        image_file_name = os.path.join('output','control.fits')
        image.write(image_file_name)
        models.append(model)
        #assert model.added_flux > 0.99*newModel.getFlux()
        #print "good flux"
        #assert True == False        
    return models

def makeCube(data,filter_name):
    # initialize (pseudo-)random number generator
    filter_ = data.filters[filter_name]
    img = galsim.ImageF(data.imageSize, data.imageSize, scale=data.pixel_scale)
    data.bdfinal.drawImage(filter_, image=img)
    images = makeCubeImages(img, data.gaussian_noise)
    return images

def makeCubeImages(img, gaussian_noise):
    images = []
    noiseIterations = 100
    # Create many images with different noises, compile into a cube
    for i in xrange(noiseIterations):
        newImg = img.copy()
        newImg.addNoise(gaussian_noise)
        images.append(newImg)
    return images

def findAvgFlux(fluxList):
    avgFlux = numpy.mean(fluxList)
    stDev = numpy.std(fluxList)
    return avgFlux, stDev

def makeFluxList(images):
    fluxList = []
    successes = 0.0
    totalAttempts = len(images)
    for image in images:
        adaptiveMom = galsim.hsm.FindAdaptiveMom(image, strict = False)
        flux = adaptiveMom.moments_amp
        if not (flux < 0):
            successes += 1
            fluxList.append(flux)
    successRate = successes/totalAttempts
    return fluxList, successRate

def makeForcedFluxList(images,models):
    fluxList = []
    totalAttempts = len(images)
    for attempt in xrange(totalAttempts):
        numerator = 0
        denominator = 0
        for x in xrange(1,images[attempt].bounds.xmax+1):
            for y in xrange(1,images[attempt].bounds.ymax+1):
                numerator += (images[attempt].at(x,y))*(models[attempt].at(x,y))
                denominator += (models[attempt].at(x,y))**2
        flux = (numerator/denominator)
        fluxList.append(flux)
    return fluxList

def findColors(oldFluxList, fluxList):
    colorList = []
    for i in xrange(min(len(oldFluxList),len(fluxList))):
        if (oldFluxList[i] < 0 or fluxList[i] < 0) and (oldFluxList[i] > 0 or fluxList[i] > 0):
            newColor = -2.5*math.log10(abs(oldFluxList[i]/fluxList[i]))
        else: 
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
    if data.forced == False:
        plt.title('Flux across bands at varied flux ratios and redshifts')
    else:
        plt.title('Flux across bands at varied flux ratios and redshifts; '
                  +'forced fit at {}-band'.format(data.forcedFilter))
    plt.grid()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()
    plt.savefig("Flux_Plot.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
    
def figure2Setup(data):
    plt.figure(2)
    plt.xlim([-1,len(data.color_name_list)])
    plt.xlabel('Band')
    plt.ylabel('Color')
    if data.forced == False:
        plt.title('Color across bands at varied flux ratios and redshifts')
    else:
        plt.title('Color across bands at varied flux ratios and redshifts; '
                  +'forced fit at {}-band'.format(data.forcedFilter))
    plt.grid()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()
    plt.savefig("Color_Plot.png",bbox_extra_artists=(lgd,),bbox_inches='tight')

# make Zebra-friendly output file

if __name__ == "__main__":
    main(sys.argv)
