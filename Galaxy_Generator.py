# Taken and modified from demo12.py in Galsim/examples
import sys
import os
import shutil
import copy
import math
import matplotlib.pyplot as plt
import numpy
import logging
import galsim

# Top-level function
def main(argv):
    class Struct(): pass
    data = Struct()
    # Enable/disable forced photometry, select forced band
    data.forced = False
    data.forcedFilter = "r"
    data.refBand = "r"
    data.refMag = 20
    # Establish basic image parameters
    data.imageSize = 64
    data.pixel_scale = 0.2 # arcseconds
    data.noiseIterations = 100

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
    if os.path.exists("gal_catalog.cat"):
        os.remove("gal_catalog.cat")
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
        filters[filter_name] = filters[filter_name].withZeropoint("AB", effective_diameter=640.0, exptime=15.0)
        filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)
    return filters

# Galaxy generating functions
def makeGalaxies(data):
    data.logger.info('')
    data.logger.info('Starting to generate chromatic bulge+disk galaxy')
    # Iterations to complete
    fluxNum = 2
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
    figure3Setup(data)

# Individual galaxy generator functions
def makeGalaxy(fluxRatio, redshift, SEDs):
    # Construct the bulge, disk, and then the final profile from the two
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
    # initialize (pseudo-)random number generator and noise
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)
    noiseSigma = 0.02
    data.gaussian_noise = galsim.GaussianNoise(rng, sigma=noiseSigma)
    filter_names = data.filter_names
    # Initialize data storage lists
    avgFluxes, stDevs = [], []
    avgColors, colorStDevs = [], []
    colorDict = {}
    avgMags, magStDevs = [], []
    oldFluxList = []
    # Create list of models if forced photometry is enabled
    if data.forced == True:
        models = makeModels(data)
    for i in xrange(len(filter_names)):
        # Write the data cube
        outpath = setOutput(filter_names[i],data.path)
        images = makeCube(data,filter_names[i])
        data.logger.debug('Created {}-band image'.format(filter_names[i]))
        fitsName = 'gal_{}_{}_{}.fits'.format(filter_names[i],fluxRatio,
                                              redshift)
        out_filename = os.path.join(outpath, fitsName)
        galsim.fits.writeCube(images, out_filename)
        data.logger.debug(('Wrote {}-band image to disk, '.format(filter_names[i]))
                          + ('bulge-to-total ratio = {}, '.format(fluxRatio))
                          + ('redshift = {}'.format(redshift)))
        # Make lists of flux using the enabled method
        if data.forced == False:
            fluxList, successRate = makeFluxList(images)
        else:
            fluxList, successRate = makeForcedFluxList(images,models), None
        # Use the list of fluxes to find all other relevant data
        avgFlux,stDev = findAvgStDev(fluxList)
        avgFluxes.append(avgFlux), stDevs.append(stDev)
        data.logger.info('Average flux for {}-band image: {}'.format(filter_names[i], avgFlux))
        data.logger.info('Standard Deviation = {}'.format(stDev))  
        # Calculate colors using existing flux data
        if oldFluxList != []:
            colorList = findColors(oldFluxList, fluxList)
            avgColor, colorStDev = findAvgStDev(colorList)
            avgColors.append(avgColor), colorStDevs.append(colorStDev)
            data.logger.info('Color = {}'.format(avgColor))
            data.logger.info('Color Standard Dev = {}'.format(colorStDev))
            key = "%s%s" % (filter_names[i-1], filter_names[i])
            colorDict[key] = colorList
        data.logger.info('Success Rate = {}'.format(successRate))
        # Update old flux list for next color calculation
        oldFluxList = fluxList
    # Using accumulated color lists, generate the magnitudes
    magLists = makeMagLists(data,colorDict)
    # embed acquired information in the data structure
    avgMags,magStDevs = listAvgStDev(magLists)
    data.avgFluxes, data.stDevs = avgFluxes, stDevs
    data.avgColors, data.colorStDevs = avgColors, colorStDevs
    data.avgMags, data.magStDevs = avgMags, magStDevs
    makeCatalog(data, redshift)

def setOutput(filter_name,path):
    # Check if output directory exists, and create it if it does not
    outputDir = 'output_{}'.format(filter_name)
    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)
    # Return target output path for file
    outDir = "output_{}/".format(filter_name)
    outpath = os.path.abspath(os.path.join(path, outDir))
    return outpath

def makeModels(data):
    # Make base images off which to model
    images = makeCube(data, data.forcedFilter)
    # Establish relevant model list and image bounds
    models = []
    bounds = galsim.BoundsI(1,data.imageSize,1,data.imageSize)
    for image in images:
        gal_moments = galsim.hsm.FindAdaptiveMom(image, strict = False)
        # If .FindAdaptiveMom was successful, make the model image
        if gal_moments.moments_status == 0:
            flux = gal_moments.moments_amp
            # Resize sigma based on pixel scale of the image
            sigma = gal_moments.moments_sigma * data.pixel_scale
            g1 = gal_moments.observed_shape.getG1()
            g2 = gal_moments.observed_shape.getG2()
            # Create new model using the Gaussian information above
            newModel = galsim.Gaussian(flux = flux, sigma = sigma)
            newModel = newModel.shear(g1=g1,g2=g2)
            model = galsim.Image(bounds, scale=data.pixel_scale) 
            newModel.drawImage(image = model)
            model.addNoise(data.gaussian_noise)
            file_name = os.path.join('output','test.fits')
            model.write(file_name)
            models.append(model)
    return models

def makeCube(data,filter_name):
    # Acquire filter, then apply to the base image
    filter_ = data.filters[filter_name]
    img = galsim.ImageF(data.imageSize, data.imageSize, scale=data.pixel_scale)
    data.bdfinal.drawImage(filter_, image=img)
    # Create an image cube from the base image
    images = makeCubeImages(data, img)
    return images

def makeCubeImages(data, img):
    images = []
    gaussian_noise = data.gaussian_noise
    noiseIterations = data.noiseIterations
    # Create many images with different noises, compile into a cube
    for i in xrange(noiseIterations):
        newImg = img.copy()
        newImg.addNoise(gaussian_noise)
        images.append(newImg)
    return images

def findAvgStDev(inputList):
    calcList = []
    # Remove all None-type objects
    for i in xrange(len(inputList)):
        if inputList[i] != None: calcList.append(inputList[i])
    avg = numpy.mean(calcList)
    stDev = numpy.std(calcList)
    return avg, stDev

def listAvgStDev(inputLists):
    meanList, stDevList = [], []
    calcLists = copy.deepcopy(inputLists)
    for i in xrange(len(calcLists)):
        tempList = []
        for j in xrange(len(calcLists[i])):
            if calcLists[i][j] != None: tempList.append(calcLists[i][j])
        meanList.append(numpy.mean(tempList))
        stDevList.append(numpy.std(tempList))
    return meanList, stDevList
        
def makeFluxList(images):
    fluxList = []
    successes = 0.0
    totalAttempts = len(images)
    for image in images:
        adaptiveMom = galsim.hsm.FindAdaptiveMom(image, strict = False)
        flux = adaptiveMom.moments_amp
        # Record if successful
        if adaptiveMom.moments_status == 0:
            successes += 1
            fluxList.append(flux)
        else:
            fluxList.append(None)
    successRate = successes/totalAttempts
    return fluxList, successRate

def makeForcedFluxList(images,models):
    fluxList = []
    totalAttempts = min(len(images),len(models))
    for attempt in xrange(totalAttempts):
        numerator = 0
        denominator = 0
        # Use flux = sum(I_ij * M_ij)/sum(M_ij ** 2) to find flux
        for x in xrange(1,images[attempt].bounds.xmax+1):
            for y in xrange(1,images[attempt].bounds.ymax+1):
                numerator+=(images[attempt].at(x,y))*(models[attempt].at(x,y))
                denominator += (models[attempt].at(x,y))**2
        flux = (numerator/denominator)
        fluxList.append(flux)
    return fluxList

def findColors(oldFluxList, fluxList):
    colorList = []
    for i in xrange(min(len(oldFluxList),len(fluxList))):
        if ((oldFluxList[i] == None) or (fluxList[i] == None)):
            colorList.append(None)
        else:
            newColor = -2.5*math.log10(oldFluxList[i]/fluxList[i])
            colorList.append(newColor)
    return colorList

def makeMagLists(data, colorDict):
    filter_names = data.filter_names
    startIndex = filter_names.index(data.refBand)
    # Generate empty list of magnitudes, fill with given magnitude to start
    magLists = [[] for char in filter_names]
    magLists[startIndex] = [data.refMag for i in xrange(data.noiseIterations)]
    # color = -2.5*log10(flux1/flux2) = mag1 - mag2
    # Find the magnitudes for earlier bands
    if startIndex > 0:
        for i in xrange(startIndex, 0 , -1):
            key = "%s%s" % (filter_names[i-1],filter_names[i])
            length = min(len(colorDict[key]), len(magLists[i]))
            newMags = []
            for k in xrange(length):
                if (colorDict[key][k] == None or magLists[i][k] == None):
                    # Retain list length, but do not record meaningful data
                    newMags.append(None)
                else:
                    newMags.append(colorDict[key][k]+magLists[i][k])
            magLists[i-1] = newMags
    # Find the magnitudes for later bands
    if startIndex < (len(filter_names) - 1):
        for i in xrange(startIndex, len(filter_names)-1):
            key = "%s%s" % (filter_names[i],filter_names[i+1])
            length = min(len(colorDict[key]), len(magLists[i]))
            newMags = []
            for k in xrange(length):
                if (colorDict[key][k] == None or magLists[i][k] == None):
                    # Retain list length, but do not record meaningful data
                    newMags.append(None)
                else:
                    newMags.append(magLists[i][k]-colorDict[key][k])
            magLists[i+1] = newMags
    return magLists

def newPlot(data,fluxRatio,redshift,fluxIndex,shiftIndex):
    # colors = ["b","c","g","y","r","m"]
    colors = ["g","y","r","m"]
    shapes = ["o","^","s","p","h","D"]
    n_groups, m_groups = len(data.avgFluxes), len(data.avgColors)
    # Start with the flux plot
    plt.figure(1)
    index, filter_name_list = range(n_groups), data.filter_name_list
    avgFluxes, stDevs = data.avgFluxes, data.stDevs
    plt.errorbar(index, avgFluxes, stDevs, None, barsabove = True, 
                 marker = "%s" % shapes[fluxIndex], linestyle = "none", 
                 mfc = "%s" % colors[shiftIndex], capsize = 10, ecolor = "k",
                 label = "{} B/T, {} redshift".format(fluxRatio,redshift))
    plt.xticks(index, filter_name_list)
    # Alternate to the color plot
    plt.figure(2)
    colorIndex, color_name_list = range(m_groups), data.color_name_list
    avgColors, colorStDevs = data.avgColors, data.colorStDevs
    plt.errorbar(colorIndex, avgColors, colorStDevs, None, barsabove = True,
                 marker = "%s" % shapes[fluxIndex], linestyle = "none",
                 mfc = "%s" % colors[shiftIndex], capsize = 10, ecolor = "k",
                 label = "{} B/T, {} redshift".format(fluxRatio,redshift))
    plt.xticks(colorIndex, color_name_list)
    # Alternate to the magnitude plot
    plt.figure(3)
    index, filter_name_list = range(n_groups), data.filter_name_list
    avgMags, magStDevs = data.avgMags, data.magStDevs
    plt.errorbar(index, avgMags, magStDevs, None, barsabove = True, 
                 marker = "%s" % shapes[fluxIndex], linestyle = "none", 
                 mfc = "%s" % colors[shiftIndex], capsize = 10, ecolor = "k",
                 label = "{} B/T, {} redshift".format(fluxRatio,redshift))
    plt.xticks(index, filter_name_list)

def figure1Setup(data):
    # Swap focus to flux plot, add finishing touches to plot
    plt.figure(1)
    plt.xlim([-1,len(data.filter_names)])
    plt.xlabel('Band')
    plt.grid()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()
    if data.forced == False:
        plt.title('Flux across bands at varied flux ratios and redshifts')
        plt.ylabel('Magnitude of the Flux (Arbitrary Units)')
        plt.savefig("Flux_Plot.png",bbox_extra_artists=(lgd,),bbox_inches='tight')
    else:
        plt.title('Flux across bands at varied flux ratios and redshifts; '
                  +'forced fit at {}-band'.format(data.forcedFilter))
        plt.ylabel('Ratio of Flux in band to Flux in {}-band'.format(data.forcedFilter))
        saveName = "Flux_Plot-Forced_{}.png".format(data.forcedFilter)
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')
    
def figure2Setup(data):
    # Swap focus to color plot, add finishing touches to plot
    plt.figure(2)
    plt.xlim([-1,len(data.color_name_list)])
    plt.xlabel('Band')
    plt.ylabel('AB Color')
    plt.grid()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()
    if data.forced == False:
        plt.title('Color across bands at varied flux ratios and redshifts')
        plt.savefig("Color_Plot.png",bbox_extra_artists=(lgd,),bbox_inches='tight')

    else:
        plt.title('Color across bands at varied flux ratios and redshifts; '
                  +'forced fit at {}-band'.format(data.forcedFilter))
        saveName = "Color_Plot-Forced_{}.png".format(data.forcedFilter)
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')

def figure3Setup(data):
    # Swap focus to color plot, add finishing touches to plot
    plt.figure(3)
    plt.xlim([-1,len(data.filter_names)])
    plt.xlabel('Band')
    plt.ylabel('AB Magnitude')
    plt.grid()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()
    if data.forced == False:
        plt.title('Mag across bands at varied flux ratios and redshifts,'
                  + ' base mag {} {}'.format(data.refBand, data.refMag))
        saveName = "Mag_Plot_{}_{}.png".format(data.refBand,data.refMag)
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')
    else:
        plt.title('Mag across bands at varied flux ratios and redshifts,'
                  + ' base mag {} {}; '.format(data.refBand, data.refMag)
                  +'forced fit at {}-band'.format(data.forcedFilter))
        saveName = "Mag_Plot_{}_{}-Forced_{}.png".format(data.refBand,data.refMag, data.forcedFilter)
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')

# make Zebra-friendly output file
# writeFile taken from 15-112 class notes:
# http://www.kosbie.net/cmu/spring-13/15-112/handouts/fileWebIO.py
def writeFile(filename, contents, mode="wt"):
    fout = None
    try:
        fout = open(filename, mode)
        fout.write(contents)
    finally:
        if (fout != None): fout.close()
    return True

def makeCatalog(data, redshift):
    avgMags, magStDevs = data.avgMags, data.magStDevs
    fluxData = zip(avgMags, magStDevs)
    contents = ""
    for i in xrange(len(fluxData)):
        for j in xrange(len(fluxData[0])):
            contents += (str(fluxData[i][j]) + " ")
    contents += ("\n")
    if not os.path.exists("gal_catalog.cat"):
        writeFile("gal_catalog.cat", contents)
    else:
        writeFile("gal_catalog.cat", contents, "a")


def getNewMagList(data, fluxList, filter_name):
    zeroPoint = data.filters[filter_name].zeropoint
    magList = []
    for flux in fluxList:
        if flux == None:  magList.append(None)
        else:
            magList.append(-2.5 * numpy.log10(flux) + zeroPoint)
    return magList
    
def getNewColorList(data, oldMagList, magList):
    colorList = []
    length = min(len(oldMagList),len(magList))
    for i in xrange(length):
        colorList.append(oldMagList[i] - magList[i])
    return colorList

if __name__ == "__main__":
    main(sys.argv)
