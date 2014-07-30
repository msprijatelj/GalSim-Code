# Taken and modified from demo12.py in Galsim/examples
import sys
import os
import shutil
import copy
import math
import string
import matplotlib.pyplot as plt
import numpy
import logging
import subprocess
import galsim
import fitsio
from tractor import *
from tractor.galaxy import *

# Top-level function
def main(argv):
    class Struct(): pass
    data = Struct()
    # Enable/disable forced photometry, select forced band
    data.useTractor = False
    if data.useTractor == False:
        data.forced = False
    data.forcedFilter = "r"
    #data.refBand = "r"
    #data.refMag = 20
    # Establish basic image parameters
    data.imageSize = 64
    data.pixel_scale = 0.2 # arcseconds
    data.noiseIterations = 5 if data.useTractor == True else 100
    data.noiseSigma = 0.1
    # Iterations to complete
    fluxNum = 4
    redshiftNum = 3
    # Other parameters
    fluxMin, fluxMax = 0.0, 1.0
    redshiftMin, redshiftMax = 0.2, 1.0
    data.ratios = numpy.linspace(fluxMin,fluxMax,fluxNum)
    data.redshifts = numpy.linspace(redshiftMin,redshiftMax,redshiftNum)
    # Where to find and output data
    data.path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(data.path, "data/"))
    data.zebrapath = "/Users/michaelprijatelj/Research/zebra-1.10/"
    data.catpath = data.zebrapath + "examples/ML_notImproved/gal_catalog.cat"
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
    if os.path.exists(data.catpath):
        os.remove(data.catpath)
    makeGalaxies(data)

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
    for name in filter_names:
        filename = os.path.join(datapath, 'LSST_{}.dat'.format(name))
        filters[name] = galsim.Bandpass(filename)
        filters[name] = filters[name].withZeropoint("AB", 640.0, 15.0)
        filters[name] = filters[name].thin(rel_err=1e-4)
    return filters

# Galaxy generating functions
def makeGalaxies(data):
    data.logger.info('')
    data.logger.info('Starting to generate chromatic bulge+disk galaxy')
    fluxIndex = 0
    data.allFluxes = []
    data.allMags = []
    data.fractions = []
    data.fracs = []
    for fluxRatio in data.ratios:   
        shiftIndex = 0
        for redshift in data.redshifts:           
            data.bdfinal = makeGalaxy(fluxRatio, redshift, data.SEDs)
            data.logger.debug('Created bulge+disk galaxy final profile')
            # draw profile through LSST filters
            applyFilter(data,fluxRatio,redshift)
            newPlot(data,fluxRatio,redshift,fluxIndex,shiftIndex)
            data.allFluxes.append(data.avgFluxes)
            data.allMags.append(data.avgMags)
            shiftIndex += 1
        fluxIndex += 1
    figure1Setup(data)
    if data.useTractor == False:
        figure2Setup(data)
    figure3Setup(data)
    outputRedshifts, loError, hiError = runZebraScript(data)
    print np.array(outputRedshifts), np.array(loError), np.array(hiError)
    print "\n",np.array(data.fractions)
    print "\n",np.array(data.fracs)
    makeRedshiftPlot(data, outputRedshifts, loError, hiError)
    print "B/T ratios across bands:\n",btRatiosAcrossBands(data)

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
    noiseSigma = data.noiseSigma
    data.gaussian_noise = galsim.GaussianNoise(rng, sigma=noiseSigma)
    filter_names = data.filter_names
    # Initialize data storage lists
    data.avgFluxes, data.stDevs = [], []
    data.avgColors, data.colorStDevs = [], []
    data.avgMags, data.magStDevs = [], []
    oldMagList = []
    # Create list of models if forced photometry is enabled
    if data.useTractor == False and data.forced == True:
        models, successRate = makeModels(data)
        data.logger.info('Success Rate = {}'.format(successRate))
    for i in xrange(len(filter_names)):
        # Write the data cube
        images = imageSetup(data, fluxRatio, redshift, filter_names[i])
        # Make lists of flux using the enabled method
        if data.useTractor == False:
            if data.forced == False:
                fluxList, successRate = makeFluxList(images)
                data.logger.info('Success Rate = {}'.format(successRate))
            elif data.forced == True:
                fluxList = makeForcedFluxList(data,images,models)
            # Use the list of fluxes to find all other relevant data
            getAvgFluxAndMags(data,fluxList,filter_names[i]) 
            # Calculate colors using existing flux data
            if oldMagList != []:
                getColors(data, oldMagList)
            # Update old flux list for next color calculation
            oldMagList = data.magList
    if data.useTractor == True:
        package = makeTractorFluxList(data, fluxRatio, redshift)
        data.avgFluxes,data.stDevs,data.avgMags,data.magStDevs,fracs = package
        data.fracs.append(fracs)
        print data.avgFluxes,data.stDevs,data.avgMags,data.magStDevs,data.fracs
    # Using accumulated color lists, generate the magnitudes
    # embed acquired information in the data structure
    makeCatalog(data, redshift)

def imageSetup(data, fluxRatio, redshift, filter_name):
    outpath = setOutput(filter_name,data.path)
    images = makeCube(data,filter_name)
    data.logger.debug('Created {}-band image'.format(filter_name))
    fitsName = "gal_%s_%0.2f_%0.2f.fits"%(filter_name,fluxRatio,redshift)
    out_filename = os.path.join(outpath, fitsName)
    galsim.fits.writeCube(images, out_filename)
    data.logger.debug(('Wrote %s-band image to disk, '%(filter_name))
                      + ('bulge-to-total ratio = {}, '.format(fluxRatio))
                      + ('redshift = {}'.format(redshift)))
    return images

def getAvgFluxAndMags(data,fluxList,filter_name):
    avgFlux,stDev = findAvgStDev(fluxList)
    data.avgFluxes.append(avgFlux), data.stDevs.append(stDev)
    data.logger.info('Average flux for %s-band: %s'%(filter_name, avgFlux))
    data.logger.info('Standard Deviation = {}'.format(stDev))  
    data.magList = makeMagList(data, fluxList, filter_name)
    avgMag, magStDev = findAvgStDev(data.magList)
    data.avgMags.append(avgMag), data.magStDevs.append(magStDev)
    data.logger.info('Average mag for %s-band: %s'%(filter_name, avgMag))
    data.logger.info('Mag Standard Deviation = {}'.format(magStDev))

def getColors(data, oldMagList):
    colorList = makeColorList(oldMagList, data.magList)
    avgColor, colorStDev = findAvgStDev(colorList)
    data.avgColors.append(avgColor), data.colorStDevs.append(colorStDev)
    data.logger.info('Color = {}'.format(avgColor))
    data.logger.info('Color Standard Dev = {}'.format(colorStDev))

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
    modelFluxList = []
    bounds = galsim.BoundsI(1,data.imageSize,1,data.imageSize)
    totalAttempts = len(images)
    successes = 0.0
    for image in images:
        gal_moments = galsim.hsm.FindAdaptiveMom(image, strict = False)
        # If .FindAdaptiveMom was successful, make the model image
        if gal_moments.moments_status == 0:
            successes += 1
            flux = gal_moments.moments_amp
            modelFluxList.append(flux)
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
            #file_name = os.path.join('output','test.fits')
            #model.write(file_name)
            models.append(model)
        else: modelFluxList.append(None)
    data.modelFluxList = modelFluxList
    return models, successes/totalAttempts

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

def findAvgStDev(inputList, floor=0.0):
    calcList = []
    # Remove all None-type objects
    for i in xrange(len(inputList)):
        if inputList[i] != None: calcList.append(inputList[i])
    avg = numpy.mean(calcList)
    stDev = max(numpy.std(calcList), floor)
    return avg, stDev
        
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

def makeForcedFluxList(data,images,models):
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
        if data.modelFluxList[attempt] != None:
            flux = (numerator/denominator)*data.modelFluxList[attempt]
        else: flux = None
        fluxList.append(flux)
    return fluxList
    
def makeMagList(data, fluxList, filter_name):
    # Magnitude = zero point magnitude - 2.5 * log10(flux)
    zeropoint_mag = data.filters[filter_name].zeropoint
    magList = []
    for flux in fluxList:
        if (flux == None) or (flux <= 0):  magList.append(None)
        else:
            magList.append(-2.5 * numpy.log10(flux) + zeropoint_mag)
    return magList
    
def makeColorList(oldMagList, magList):
    # Use differences between magnitudes to calculate color
    colorList = []
    length = min(len(oldMagList),len(magList))
    for i in xrange(length):
        if oldMagList[i] != None and magList[i] != None:
            colorList.append(oldMagList[i] - magList[i])
        else: colorList.append(None)
    return colorList

def newPlot(data,fluxRatio,redshift,fluxIndex,shiftIndex):
    # colors = ["b","c","g","y","r","m"]
    colors = ["g","y","r","m"]
    color = colors[shiftIndex]
    shapes = ["o","^","s","v","h","p"]
    n_groups  = len(data.avgFluxes)
    # Start with the flux plot
    plt.figure(1)
    index, filter_name_list = range(n_groups), data.filter_name_list
    avgFluxes, stDevs = data.avgFluxes, data.stDevs
    plt.errorbar(index, avgFluxes, stDevs, None, barsabove = True, 
                 marker = "%s" % shapes[fluxIndex], linestyle = "none", 
                 mfc = "%s" % color,capsize = 10,ecolor = "%s" % color,
                 label = "%0.2f B/T, %0.2f redshift" % (fluxRatio,redshift))
    plt.xticks(index, filter_name_list)
    # Alternate to the color plot
    if data.useTractor == False:
        m_groups = len(data.avgColors)
        plt.figure(2)
        colorIndex, color_name_list = range(m_groups), data.color_name_list
        avgColors, colorStDevs = data.avgColors, data.colorStDevs
        plt.errorbar(colorIndex, avgColors, colorStDevs, None, barsabove=True,
                     marker = "%s" % shapes[fluxIndex], linestyle = "none",
                     mfc = "%s" % color,capsize = 10,ecolor = "%s" % color,
                     label = "%0.2f B/T, %0.2f redshift"%(fluxRatio,redshift))
        plt.xticks(colorIndex, color_name_list)
    # Alternate to the magnitude plot
    plt.figure(3)
    index, filter_name_list = range(n_groups), data.filter_name_list
    avgMags, magStDevs = data.avgMags, data.magStDevs
    plt.errorbar(index, avgMags, magStDevs, None, barsabove = True, 
                 marker = "%s" % shapes[fluxIndex], linestyle = "none", 
                 mfc = "%s" % color,capsize = 10,ecolor = "%s" % color,
                 label = "%0.2f B/T, %0.2f redshift" % (fluxRatio,redshift))
    plt.xticks(index, filter_name_list)

def figure1Setup(data):
    # Swap focus to flux plot, add finishing touches to plot
    plt.figure(1)
    plt.xlim([-1,len(data.filter_names)])
    plt.xlabel('Band')
    plt.grid()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()
    plt.ylabel('Magnitude of the Flux (Arbitrary Units)')
    if data.useTractor == True:
        plt.title('Flux across bands at varied flux ratios and redshifts, Tractor')
        saveName = "Flux_Plot-Tractor.png"
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')
    elif data.forced == False:
        plt.title('Flux across bands at varied flux ratios and redshifts')
        saveName = "Flux_Plot.png"
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')
    else:
        plt.title('Flux across bands at varied flux ratios and redshifts; '
                  +'forced fit at {}-band'.format(data.forcedFilter))
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
        saveName = "Color_Plot.png"
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')

    else:
        plt.title('Color across bands at varied flux ratios and redshifts; '
                  +'forced fit at {}-band'.format(data.forcedFilter))
        saveName = "Color_Plot-Forced_{}.png".format(data.forcedFilter)
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')

def figure3Setup(data):
    # Swap focus to magnitude plot, add finishing touches to plot
    plt.figure(3)
    plt.xlim([-1,len(data.filter_names)])
    plt.ylim(24,32) # FIXME - Limits are questionable
    plt.xlabel('Band')
    plt.ylabel('AB Magnitude')
    plt.grid()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()
    if data.useTractor == True:
        plt.title('Mag across bands at varied flux ratios and redshifts, Tractor')
        saveName = "Mag_Plot-Tractor.png"
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')
    elif data.forced == False:
        plt.title('Mag across bands at varied flux ratios and redshifts')
        saveName = "Mag_Plot.png"
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')
    else:
        plt.title('Mag across bands at varied flux ratios and redshifts,'
                  +'forced fit at {}-band'.format(data.forcedFilter))
        saveName = "Mag_Plot-Forced_{}.png".format(data.forcedFilter)
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')

def makeRedshiftPlot(data, outputRedshifts, loError, hiError):
    outShifts = copy.deepcopy(outputRedshifts)
    colors = ["b","c","g","y","r","m"]
    ratios, redshifts = data.ratios, data.redshifts
    plt.figure(4)
    errors = [copy.deepcopy(loError), copy.deepcopy(hiError)]
    fluxIndex = 0
    for fluxIndex, fluxRatio in enumerate(ratios):
        color = colors[fluxIndex]
        usedShifts, usedErrors = [], [[],[]]
        for shift in redshifts:
            usedShifts.append(outShifts.pop(0))
            usedErrors[0].append(errors[0].pop(0))
            usedErrors[1].append(errors[1].pop(0))
        plt.errorbar(redshifts, usedShifts, usedErrors, None, barsabove = True,
                     marker = "o", linestyle = "none", 
                     mfc = "%s" %color, capsize = 10, ecolor = "%s"%color,
                     label = "%0.2f B/T" % (fluxRatio))
    redshiftLine = linspace(0,2)
    plt.plot(redshiftLine, redshiftLine, linewidth=1,
             label = "Expected Redshifts")
    figure4Setup(data)
    
def figure4Setup(data):
    plt.figure(4)
    plt.xlim([0,2])
    plt.ylim(0)
    plt.xlabel('Input Redshifts')
    plt.ylabel('Output Redshifts')
    plt.grid()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()
    if data.useTractor == True:
        plt.title("Measured Redshifts vs. Actual Redshifts, Tractor")
        saveName = "Redshifts-Tractor.png"
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')
    elif data.forced == False:
        plt.title('Measured Redshifts vs. Actual Redshifts')
        saveName = "Redshifts.png"
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')
    else:
        plt.title("Measured Redshifts vs. Actual Redshifts; "
                  +'forced fit at {}-band'.format(data.forcedFilter))
        saveName = "Redshifts-Forced_{}.png".format(data.forcedFilter)
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')
        
# make Zebra-friendly output file
# readFile and writeFile taken from CMU 15-112 class notes:
# http://www.kosbie.net/cmu/spring-13/15-112/handouts/fileWebIO.py
def readFile(filename, mode="rt"):
    # rt stands for "read text"
    fin = contents = None
    try:
        fin = open(filename, mode)
        contents = fin.read()
    finally:
        if (fin != None): fin.close()
    return contents

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
    if not os.path.exists(data.catpath):
        writeFile(data.catpath, contents)
    else:
        writeFile(data.catpath, contents, "a")

def runZebraScript(data):
    os.chdir("../zebra-1.10/scripts/")
    subprocess.Popen(['./callzebra_ML_notImproved_2']).wait()
    datPath = '../examples/ML_notImproved/ML.dat'
    (rs, loE, hiE) = readRedshifts(datPath)
    os.chdir(data.path)
    return rs, loE, hiE

def readRedshifts(filename):
    extractedList = extractData(readFile(filename))
    redshifts, loErrors, hiErrors = [], [], []
    for line in extractedList:
        (items, isWhitespace) = ([""], True)
        extractionIndex = charIndex = 0
        while charIndex < len(line):
            char = line[charIndex]
            if isWhitespace == False:
                if char == " ": 
                    isWhitespace = True
                    items.append("")
                    extractionIndex += 1
                else: items[extractionIndex] += char
            else:
                if char != " ":
                    isWhitespace = False
                    items[extractionIndex] += char
            charIndex += 1
        redshifts.append(float(items[2]))
        # Errors could be from columns 21 to 28 (alternating lo/hi)   
        loErrors.append(abs(float(items[20]) - float(items[2])))
        hiErrors.append(abs(float(items[21]) - float(items[2])))
    return redshifts, loErrors, hiErrors
    
def extractData(contents):
    contentList = string.split(contents,"\n")[:-1]
    extractedList = []
    for i in xrange(len(contentList)):
        if contentList[i][0] != "#":  extractedList.append(contentList[i])
    return extractedList


# Tractor functions, modified from Tractor documentation by Dustin Lang:
# http://thetractor.org/doc/galsim.html
def makeTractorFluxList(data, fluxRatio, redshift):
    bands = data.filter_names
    pixnoise = data.noiseSigma
    psf_sigma = 1.5
    nepochs = data.noiseIterations
    tims = []
    for band in bands:
        filename = ("output_%s/gal_%s_%0.2f_%0.2f.fits" % (band,band,fluxRatio,
                    redshift))
        print 'Band', band, 'Reading', filename
        cube,hdr = fitsio.read(filename, header = True)    
        print 'Read', cube.shape
        pixscale = hdr['GS_SCALE']
        print 'Pixel scale:', pixscale, 'arcsec/pix'
        nims,h,w = cube.shape
        assert(nims == nepochs)    
        zeropoint = data.filters[band].zeropoint
        for k in range(nims):
            image = cube[k,:,:]
            photocalScale = NanoMaggies.zeropointToScale(zeropoint)
            tim = Image(data=image, invvar=np.ones_like(image) / pixnoise**2,
                        photocal=LinearPhotoCal(photocalScale, band=band),
                        wcs=NullWCS(pixscale=pixscale),
                        psf=NCircularGaussianPSF([psf_sigma], [1.0]))
            tims.append(tim)
    galaxy = makeTractorGalaxy(bands, w, h)
    """
    # Plot images
    ima = dict(interpolation='nearest', origin='lower', cmap='gray',
               vmin=-5.*pixnoise, vmax=20.*pixnoise)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92)
    plt.clf()
    for i,band in enumerate(bands):
        for e in range(nepochs):
            plt.subplot(nepochs, len(bands), e*len(bands) + i +1)
            plt.imshow(tims[nepochs*i + e].getImage(), **ima)
            plt.xticks([]); plt.yticks([])
            plt.title('%s #%i' % (band, e+1))
    plt.suptitle('Images')
    plt.savefig('%0.2f_%0.2f.png'%(fluxRatio,redshift))
    """
    tractor = optimizeTractor(Tractor(tims,[galaxy]))
    fluxList,fluxEList,magList,magEList,fracs=getFluxesAndMags(bands,tractor)
    fraction = getFraction(tractor)
    data.fractions.append(fraction)
    return fluxList, fluxEList, magList, magEList, fracs
            
def makeTractorGalaxy(bands, w, h):   
    galaxy = FixedCompositeGalaxy(PixPos(w/2, h/2),
                                  NanoMaggies(**dict([(band, 10.) for band in bands])), 
                                  0.5,
                                  EllipseESoft(0., 0., 0.),
                                  EllipseESoft(0., 0., 0.))
    return galaxy
    
def optimizeTractor(tractor):
    # Freeze all image calibration parameters
    tractor.freezeParam('images')
    # Take several linearized least squares steps
    for i in range(20):
        dlnp,X,alpha = tractor.optimize(shared_params=False)
        print 'dlnp', dlnp
        if dlnp < 1e-3:
            break
    return tractor    

def getFluxesAndMags(bands, tractor):
    bandFluxes, bandFluxErrors = [], []
    bandMags, bandMagErrors = [], []
    fractions = []
    for band in bands:
        tractor.freezeAllRecursive()
        tractor.catalog[0].thawParam('fracDev')
        tractor.thawPathsTo(band)
        tractor.catalog[0].pos.freezeParams('y')
        bandVar = tractor.optimize(variance=True, just_variance=True, shared_params=False)
        bandFlux = tractor.getParams()
        bandFluxes.append(bandFlux[0])
        bandFluxErrors.append(bandVar[0])
        fractions.append(bandFlux[1])
        # convert to mag and mag error, assuming zeropoint of 22.5:
        if bandFlux[0] <= 0:
            bandMag = [99]
            bandMagError = [0]
        else:
            npFlux, invvar = np.array([bandFlux[0]]), 1./np.array([bandVar[0]])
            bandMag,bandMagError=NanoMaggies.fluxErrorsToMagErrors(npFlux,
                                                                   invvar)
        bandMags += list(bandMag)
        bandMagErrors += list(bandMagError)
    return bandFluxes,bandFluxErrors,bandMags,bandMagErrors,fractions
    
def getFraction(tractor):
    tractor.freezeAllRecursive()
    tractor.thawPathsTo('fracDev')
    print "hi"
    fraction = tractor.getParams()
    print fraction
    return fraction[0]



def btRatiosAcrossBands(data):
    ratios = {}
    for i in xrange(len(data.filter_names)):
        bulge = data.allFluxes[-1][i]
        disk = data.allFluxes[0][i]
        ratio = bulge/(bulge+disk)
        ratios[data.filter_names[i]]=ratio
    return ratios


if __name__ == "__main__":
    main(sys.argv)
