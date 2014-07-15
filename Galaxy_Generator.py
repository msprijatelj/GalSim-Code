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
from tractor import *


# Top-level function
def main(argv):
    class Struct(): pass
    data = Struct()
    # Enable/disable forced photometry, select forced band
    data.forced = True
    data.forcedFilter = "r"
    data.refBand = "r"
    data.refMag = 20
    # Establish basic image parameters
    data.imageSize = 64
    data.pixel_scale = 0.2 # arcseconds
    data.noiseIterations = 100
    # Iterations to complete
    fluxNum = 2
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
    logger.info('You can display the output in ds9 with a command line that '+
                'looks something like:')
    logger.info('ds9 -rgb -blue -scale limits -0.2 0.8 output_r/gal_r.fits '
                +'-green -scale limits -0.25 1.0 output_i/gal_i.fits '
                +'-red -scale limits -0.25 1.0 output_z/gal_z.fits'
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
    for fluxRatio in data.ratios:   
        shiftIndex = 0
        for redshift in data.redshifts:           
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
    outputRedshifts, loError, hiError = runZebraScript(data)
    print outputRedshifts
    makeRedshiftPlot(data, outputRedshifts, loError, hiError)

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
    noiseSigma = 0.1
    data.gaussian_noise = galsim.GaussianNoise(rng, sigma=noiseSigma)
    filter_names = data.filter_names
    # Initialize data storage lists
    avgFluxes, stDevs = [], []
    avgColors, colorStDevs = [], []
    colorDict = {}
    avgMags, magStDevs = [], []
    oldMagList = []
    # Create list of models if forced photometry is enabled
    if data.forced == True:
        models, successRate = makeModels(data)
        data.logger.info('Success Rate = {}'.format(successRate))
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
            data.logger.info('Success Rate = {}'.format(successRate))
        else:
            fluxList = makeForcedFluxList(data,images,models)
        # Use the list of fluxes to find all other relevant data
        avgFlux,stDev = findAvgStDev(fluxList)
        avgFluxes.append(avgFlux), stDevs.append(stDev)
        data.logger.info('Average flux for {}-band image: {}'.format(filter_names[i], avgFlux))
        data.logger.info('Standard Deviation = {}'.format(stDev))  
        magList = makeMagList(data, fluxList, filter_names[i])
        avgMag, magStDev = findAvgStDev(magList)
        avgMags.append(avgMag), magStDevs.append(magStDev)
        data.logger.info('Average mag for {}-band image: {}'.format(filter_names[i], avgMag))
        data.logger.info('Mag Standard Deviation = {}'.format(magStDev))  
        # Calculate colors using existing flux data
        if oldMagList != []:
            colorList = makeColorList(oldMagList, magList)
            avgColor, colorStDev = findAvgStDev(colorList)
            avgColors.append(avgColor), colorStDevs.append(colorStDev)
            data.logger.info('Color = {}'.format(avgColor))
            data.logger.info('Color Standard Dev = {}'.format(colorStDev))
            key = "%s%s" % (filter_names[i-1], filter_names[i])
            colorDict[key] = colorList
        # Update old flux list for next color calculation
        oldMagList = magList
    # Using accumulated color lists, generate the magnitudes
    # embed acquired information in the data structure
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
        plt.ylabel('Magnitude of the Flux (Arbitrary Units)'.format(data.forcedFilter))
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
    # Swap focus to magnitude plot, add finishing touches to plot
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
        saveName = "Mag_Plot_{}_{}-Forced_{}.png".format(data.refBand,
                                                         data.refMag, 
                                                         data.forcedFilter)
        plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')

def makeRedshiftPlot(data, outputRedshifts, loError, hiError):
    outShifts = copy.deepcopy(outputRedshifts)
    colors = ["b","c","g","y","r","m"]
    ratios, redshifts = data.ratios, data.redshifts
    plt.figure(4)
    errors = [copy.deepcopy(loError), copy.deepcopy(hiError)]
    fluxIndex = 0
    for fluxIndex, fluxRatio in enumerate(ratios):
        usedShifts, usedErrors = [], [[],[]]
        for shift in redshifts:
            usedShifts.append(outShifts.pop(0))
            usedErrors[0].append(errors[0].pop(0))
            usedErrors[1].append(errors[1].pop(0))
        plt.errorbar(redshifts, usedShifts, usedErrors, None, barsabove = True, 
                     marker = "o", linestyle = "none", 
                     mfc = "%s" %colors[fluxIndex], capsize = 10, ecolor = "k",
                     label = "{} B/T".format(fluxRatio))
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
    if data.forced == False:
        plt.title('Measured Redshifts vs. Actual Redshifts')
        plt.savefig("Redshifts.png",bbox_extra_artists=(lgd,),bbox_inches='tight')
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
    os.chdir(data.zebrapath + "scripts/")
    subprocess.Popen(['./callzebra_ML_notImproved_2']).wait()
    (rs, loE, hiE) = readRedshifts(data.zebrapath+'examples/ML_notImproved/ML.dat')
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

if __name__ == "__main__":
    main(sys.argv)
