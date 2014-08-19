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

##FIXME - Make everything run and compare in one run of the code

###############################################################################
##FIXME - Top-level function and initialization
###############################################################################

def main(argv):
    class Struct(): pass
    data = Struct()
    initMain(data)
    initFluxesAndRedshifts(data)
    initSEDsAndFilters(data)
    removeCatalogs(data)
    makeGalaxies(data)

def initMain(data):
    # Enable mode of operation
    data.basic = False
    data.forced = False
    data.Tractor = False
    data.forcedTractor = True
    data.forcedFilter = "r"
    # Establish basic image parameters
    data.pixel_scale = 0.2 # arcseconds
    data.imageSize = 64/data.pixel_scale*0.2
    data.noiseIterations = 100
    data.tractorIterations = 8
    data.noiseSigma = 1e-15

def initFluxesAndRedshifts(data):
    # Iterations to complete
    fluxNum = 1
    redshiftNum = 1
    # Other parameters
    fluxMin, fluxMax = 0.67, 0.67
    redshiftMin, redshiftMax = 0.6, 0.6
    data.ratios = numpy.linspace(fluxMin,fluxMax,fluxNum)
    data.redshifts = numpy.linspace(redshiftMin,redshiftMax,redshiftNum)

def initSEDsAndFilters(data):
    # Where to find and output data
    data.path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(data.path, "data/"))
    data.catpath = "../zebra-1.10/examples/ML_notImproved/"
    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO, 
                        stream=sys.stdout)
    data.logger = logger = logging.getLogger("Galaxy Generator")
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

def removeCatalogs(data):
    removeCatalog(data, "gal_catalog.cat")
    removeCatalog(data, "gal_for_catalog.cat")
    removeCatalog(data, "gal_trac_catalog.cat")
    removeCatalog(data, "gal_b_trac_catalog.cat")
    removeCatalog(data, "gal_d_trac_catalog.cat")
    removeCatalog(data, "gal_for_trac_catalog.cat")
    removeCatalog(data, "gal_b_for_trac_catalog.cat")
    removeCatalog(data, "gal_d_for_trac_catalog.cat")

def removeCatalog(data, catName):
    catPath = data.catpath+catName
    if os.path.exists(catPath):
        os.remove(catPath)

def readInLSST(datapath, filter_names):
    filters = {}
    for name in filter_names:
        filename = os.path.join(datapath, 'LSST_{}.dat'.format(name))
        filters[name] = galsim.Bandpass(filename)
        filters[name] = filters[name].withZeropoint("AB", 640.0, 15.0)
        filters[name] = filters[name].thin(rel_err=1e-4)
    return filters

###############################################################################
##FIXME - Galaxy generating functions
###############################################################################

def makeGalaxies(data):
    data.logger.info('')
    data.logger.info('Starting to generate chromatic bulge+disk galaxy')
    fluxIndex = 0
    # Collect lists of all fluxes, magnitudes, and deV-to-total ratios
    initMakeGalaxies(data)
    # Iterate over all ratios and redshifts, keep track of indices for plots
    for fluxRatio in data.ratios:   
        shiftIndex = 0
        for redshift in data.redshifts:           
            data.bdfinal=makeGalaxy(data,fluxRatio,redshift)
            data.logger.debug('Created bulge+disk galaxy final profile')
            # draw profile through LSST filters
            applyFilter(data,fluxRatio,redshift)
            #newPlot(data,fluxRatio,redshift,fluxIndex,shiftIndex)
            shiftIndex += 1
        fluxIndex += 1
    """    
    figure1Setup(data)
    if data.useTractor == False:
        figure2Setup(data)
    figure3Setup(data)
    """
    basic, forced, tractor, forcedTractor = runAllZebraScripts(data)
    print "Basic Redshifts and Errors:\n", basic
    print "Forced Redshifts and Errors:\n", forced
    print "Tractor Redshifts and Errors:\n", tractor
    print "Tractor deV-to-total ratios:\n", data.allFractions
    print "Forced Tractor Redshifts and Errors:\n", forcedTractor
    print "Forced Tractor deV-to-total ratios:\n", data.allForcedFractions
    
    
def initMakeGalaxies(data):
    data.basicScript = 'callzebra_ML_notImproved_basic'
    data.forScript = 'callzebra_ML_notImproved_forced'
    data.tracScript = 'callzebra_ML_notImproved_tractor'
    data.bTracScript = 'callzebra_ML_notImproved_b_tractor'
    data.dTracScript = 'callzebra_ML_notImproved_d_tractor'
    data.forTracScript = 'callzebra_ML_notImproved_forced_tractor'
    data.bForTracScript = 'callzebra_ML_notImproved_b_forced_tractor'
    data.dForTracScript = 'callzebra_ML_notImproved_d_forced_tractor'
    data.allFractions, data.allForcedFractions = [], []

# Individual galaxy generator functions
def makeGalaxy(data, fluxRatio, redshift):
    SEDs = data.SEDs
    # Construct the bulge, disk, and then the final profile from the two
    bulge, disk = makeBulge(redshift, SEDs), makeDisk(redshift, SEDs)  
    bdfinal = makeFinal(data, fluxRatio, bulge, disk)
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
    
def makeFinal(data, fluxRatio, bulge, disk):
    totalFlux = 4.8
    psf_FWHM, psf_beta = 0.6, 2.5
    bulgeMultiplier = fluxRatio * totalFlux
    diskMultiplier = (1 - fluxRatio) * totalFlux
    bdgal = 1.1 * (bulgeMultiplier*bulge+diskMultiplier*disk)
    PSF = galsim.Moffat(fwhm=psf_FWHM, beta=psf_beta)
    #PSF = galsim.Gaussian(fwhm=psf_FWHM)
    img = galsim.ImageF(data.imageSize, data.imageSize, scale=data.pixel_scale)
    psfImage = PSF.drawImage(image = img)
    psfImage.write("PSF.fits")
    bdfinal = galsim.Convolve([bdgal, PSF])    
    return bdfinal


###############################################################################
##FIXME - Filter application and different magnitude calculation methods
###############################################################################

def applyFilter(data,fluxRatio,redshift):
    # initialize (pseudo-)random number generator and noise
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)
    noiseSigma = data.noiseSigma
    data.gaussian_noise = galsim.GaussianNoise(rng, sigma=noiseSigma)
    filter_names = data.filter_names
    # Initialize data storage lists
    initFilterLists(data)
    # Create list of models
    models = makeModels(data)
    for i in xrange(len(filter_names)):
        # Write the data cube
        images = imageSetup(data, fluxRatio, redshift, filter_names[i])
        # Make lists of flux using the enabled method
        if data.basic: getBasicFluxMag(data, images, filter_names[i])
        if data.forced: getForcedFluxMag(data, images, models, filter_names[i])
    if data.Tractor: makeTractorFluxList(data, fluxRatio, redshift)
    if data.forcedTractor: makeForcedTractorFluxList(data, fluxRatio, redshift)
    # Generate the magnitudes, place in ZEBRA catalog
    makeAllCatalogs(data)

def initFilterLists(data):
    data.fluxes, data.stDevs = [], []
    data.forFluxes, data.forStDevs = [], []
    data.tracFluxes, data.tracStDevs = [], []
    data.bTracFluxes, data.bTracStDevs = [], []
    data.dTracFluxes, data.dTracStDevs = [], []
    data.forTracFluxes, data.forTracStDevs = [], []
    data.bForTracFluxes, data.bForTracStDevs = [], []
    data.dForTracFluxes, data.dForTracStDevs = [], []
    data.mags, data.magDevs = [], []
    data.forMags, data.forMagDevs = [], []
    data.tracMags, data.tracMagDevs = [], []
    data.bTracMags, data.bTracMagDevs = [], []
    data.dTracMags, data.dTracMagDevs = [], []
    data.forTracMags, data.forTracMagDevs = [], []
    data.bForTracMags, data.bForTracMagDevs = [], []
    data.dForTracMags, data.dForTracMagDevs = [], []
    data.colors, data.colorStDevs = [], []
    data.forColors, data.forColorStDevs = [], []
    data.oldMagList, data.oldForMagList = [], []
    data.fractions = []

def makeAllCatalogs(data):
    makeCatalog(data, data.mags, data.magDevs, "gal_catalog.cat")
    makeCatalog(data, data.forMags, data.forMagDevs, "gal_for_catalog.cat")
    makeCatalog(data, data.tracMags, data.tracMagDevs, "gal_trac_catalog.cat")
    makeCatalog(data,data.bTracMags,data.bTracMagDevs,"gal_b_trac_catalog.cat")
    makeCatalog(data,data.dTracMags,data.dTracMagDevs,"gal_d_trac_catalog.cat")
    makeCatalog(data, data.forTracMags, data.forTracMagDevs, 
                "gal_for_trac_catalog.cat")
    makeCatalog(data, data.bForTracMags, data.bForTracMagDevs, 
                "gal_b_for_trac_catalog.cat")
    makeCatalog(data, data.dForTracMags, data.dForTracMagDevs, 
                "gal_d_for_trac_catalog.cat")

def getBasicFluxMag(data, images, filter_name):
    print "\n"+"Using Basic FindAdaptiveMoment:"+"\n"
    fluxList, successRate = makeFluxList(images)
    avgFlux, stDev, avgMag, magStDev = getAvgFluxMags(data,fluxList,
                                                      filter_name)
    data.fluxes.append(avgFlux), data.stDevs.append(stDev)
    data.mags.append(avgMag), data.magDevs.append(magStDev)
    if data.oldMagList != []:
        color, colorStDev = getColors(data, data.oldMagList)
        data.colors.append(color), data.colorStDevs.append(colorStDev)
    # Update old flux list for next color calculation
    data.oldMagList = data.magList

def getForcedFluxMag(data, images, models, filter_name):
    print "\n"+"Using Forced Fit:"+"\n"
    fluxList = makeForcedFluxList(data,images,models)
    avgFlux, stDev, avgMag, magStDev = getAvgFluxMags(data,fluxList,
                                                      filter_name)
    data.forFluxes.append(avgFlux), data.forStDevs.append(stDev)
    data.forMags.append(avgMag), data.forMagDevs.append(magStDev)
    if data.oldForMagList != []:
        color, colorStDev = getColors(data, data.oldForMagList)
        data.forColors.append(color), data.forColorStDevs.append(colorStDev)
    # Update old flux list for next color calculation
    data.oldForMagList = data.magList

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

def getAvgFluxMags(data,fluxList,filter_name):
    # Get average flux, average magnitude, and their respective standard devs
    avgFlux,stDev = findAvgStDev(fluxList)
    data.logger.info('Average flux for %s-band: %s'%(filter_name, avgFlux))
    data.logger.info('Standard Deviation = {}'.format(stDev))  
    data.magList = makeMagList(data, fluxList, filter_name)
    avgMag, magStDev = findAvgStDev(data.magList)
    data.logger.info('Average mag for %s-band: %s'%(filter_name, avgMag))
    data.logger.info('Mag Standard Deviation = {}'.format(magStDev))
    return avgFlux,stDev, avgMag, magStDev

def getColors(data, oldMagList):
    # Use old and current magnitude lists to find colors
    colorList = makeColorList(oldMagList, data.magList)
    avgColor, colorStDev = findAvgStDev(colorList)
    data.logger.info('Color = {}'.format(avgColor))
    data.logger.info('Color Standard Dev = {}'.format(colorStDev))
    return avgColor, colorStDev

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
    bounds = galsim.BoundsI(1,int(data.imageSize),1,int(data.imageSize))
    for image in images:
        gal_moments = galsim.hsm.FindAdaptiveMom(image, strict = False)
        # If .FindAdaptiveMom was successful, make the model image
        if gal_moments.moments_status == 0:
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
            models.append(model)
        else: modelFluxList.append(None)
    data.modelFluxList = modelFluxList
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

def findAvgStDev(inputList, floor=0.0):
    calcList = []
    # Remove all None-type objects
    for i in xrange(len(inputList)):
        if inputList[i] != None: calcList.append(inputList[i])
    avg = numpy.mean(calcList)
    stDev = max(numpy.std(calcList), floor)
    return avg, stDev
        
def makeFluxList(images):
    # Use GalSim FindAdaptiveMoment method to fit a Gaussian to the galaxy and
    # compute the flux from the fit.
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
    # Use current images and created models to calculate flux
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

###############################################################################
##FIXME - Tractor-based flux and magnitude calculation
###############################################################################

# Tractor is written by Dustin Lang, documentation found here: 
# http://thetractor.org/doc/index.html
# Tractor functions, modified from Tractor documentation by Dustin Lang:
# http://thetractor.org/doc/galsim.html
def makeTractorFluxList(data, fluxRatio, redshift):
    print "\n"+"Using Tractor:"+"\n"
    bands = data.filter_names
    for band in bands:
        # Make an optimized Tractor object and get its flux and magnitude
        getFluxBulgeDisk(data, band, fluxRatio, redshift)
    data.allFractions.append(data.fractions)

def makeOptimizedTractor(data, band, fluxRatio, redshift):
    # Get a list of Tractor images and their widths/heights
    tims, w, h = makeTractorImages(data, band, fluxRatio, redshift)
    # Make a rudimentary galaxy model using band given and image dimensions
    galaxy = makeTractorGalaxy(band, w, h)
    # Put tractor images and galaxy model into Tractor object and optimize
    tractor = optimizeTractor(Tractor(tims,[galaxy]))
    return tractor

def getFluxBulgeDisk(data, band, fluxRatio, redshift):
    # Get a list of Tractor images and their widths/heights
    tims, w, h = makeTractorImages(data, band, fluxRatio, redshift)
    # Put tractor images and galaxy model into Tractor object and optimize
    fluxes, mags, fracs = [], [], []
    bulgeFluxes, diskFluxes, bulgeMags, diskMags = [], [], [], []
    for tim in tims:
        # Make a rudimentary galaxy model using band given and image dimensions
        galaxy = makeTractorGalaxy(band, w, h)
        package = getTractorFluxAndMag(data, band, tim, galaxy)
        (flux, bulgeFlux, diskFlux, mag, bulgeMag, diskMag, frac) = package
        fluxes.append(flux), mags.append(mag), fracs.append(frac)
        bulgeFluxes.append(bulgeFlux), diskFluxes.append(diskFlux)
        bulgeMags.append(bulgeMag), diskMags.append(diskMag)
    collectFluxes(data, fluxes, bulgeFluxes, diskFluxes) 
    collectMags(data, mags, bulgeMags, diskMags)
    frac, fracDev = findAvgStDev(fracs)
    data.fractions.append(frac)

def getTractorFluxAndMag(data, band, tim, galaxy):
    zeropoint_mag = data.filters[band].zeropoint
    tractor = optimizeTractor(Tractor([tim],[galaxy]))
    flux, mag = getFluxesAndMags(data,band,tractor)
    frac = galaxy.fracDev.getClippedValue()
    (bulgeFlux, diskFlux) = (flux*frac, flux*(1-frac))
    bulgeMag = -2.5 * numpy.log10(bulgeFlux) + zeropoint_mag
    diskMag = -2.5 * numpy.log10(diskFlux) + zeropoint_mag
    return (flux, bulgeFlux, diskFlux, mag, bulgeMag, diskMag, frac)

def collectFluxes(data, fluxes, bulgeFluxes, diskFluxes):
    flux, stDev = findAvgStDev(fluxes)
    bulgeFlux, bulgeStDev = findAvgStDev(bulgeFluxes)
    diskFlux, diskStDev = findAvgStDev(diskFluxes)
    data.tracFluxes.append(flux), data.tracStDevs.append(stDev)
    data.bTracFluxes.append(bulgeFlux), data.bTracStDevs.append(bulgeStDev)
    data.dTracFluxes.append(diskFlux), data.dTracStDevs.append(diskStDev)

def collectMags(data, mags, bulgeMags, diskMags):
    mag, magDev = findAvgStDev(mags)
    bulgeMag, bulgeMagDev = findAvgStDev(bulgeMags)
    diskMag, diskMagDev = findAvgStDev(diskMags)
    data.tracMags.append(mag), data.tracMagDevs.append(magDev)
    data.bTracMags.append(bulgeMag), data.bTracMagDevs.append(bulgeMagDev)
    data.dTracMags.append(diskMag), data.dTracMagDevs.append(diskMagDev)

def makeTractorImages(data, band, fluxRatio, redshift):
    pixnoise = data.noiseSigma
    nepochs = data.tractorIterations
    tims = []
    # Get images to convert to Tractor images
    cube, pixscale = getBandImages(band,fluxRatio,redshift)
    nims,h,w = cube.shape
    zeropoint = data.filters[band].zeropoint
    psfCube = fitsio.read("PSF.fits")
    psf = GaussianMixturePSF.fromStamp(psfCube)
    for k in range(nepochs):
        image = cube[k,:,:]
        # Scale fluxes/magnitudes for the appropriate zero-point magnitude
        photocalScale = NanoMaggies.zeropointToScale(zeropoint)
        tim = Image(data=image, invvar=np.ones_like(image) / pixnoise**2,
                    photocal=LinearPhotoCal(photocalScale, band=band),
                    wcs=NullWCS(pixscale=pixscale),
                    psf=psf)
        tims.append(tim)
    return tims, w, h

def getBandImages(band, fluxRatio,redshift):
    # Read in GalSim FITS files, make cube of images using fitsio
    filename = ("output_%s/gal_%s_%0.2f_%0.2f.fits" % (band,band,fluxRatio,
                redshift))
    print 'Band', band, 'Reading', filename
    cube,hdr = fitsio.read(filename, header = True)    
    print 'Read', cube.shape
    pixscale = hdr['GS_SCALE']
    print 'Pixel scale:', pixscale, 'arcsec/pix'
    return cube, pixscale

def makeForcedTractorFluxList(data, fluxRatio, redshift):
    print "\n"+"Using Forced Tractor:"+"\n"
    bands = data.filter_names
    # Make optimized Tractor object using the forced filter band
    forTractor=makeOptimizedTractor(data,data.forcedFilter,fluxRatio,redshift)
    for band in bands:
        # Make new tractor objects with parameters from forced Tractor object
        tims, w, h = makeTractorImages(data, band, fluxRatio, redshift)
        fluxes, mags, fracs = [], [], []
        bulgeFluxes, diskFluxes, bulgeMags, diskMags = [], [], [], []
        for tim in tims:
            galaxy = prepareGalaxyWithTractor(forTractor, band, w, h)
            # Optimize Tractor object, get fluxes and magnitudes
            package = getTractorFluxAndMag(data, band, tim, galaxy)
            (flux, bulgeFlux, diskFlux, mag, bulgeMag, diskMag, frac) = package
            fluxes.append(flux), mags.append(mag), fracs.append(frac)
            bulgeFluxes.append(bulgeFlux), diskFluxes.append(diskFlux)
            bulgeMags.append(bulgeMag), diskMags.append(diskMag)
        collectForcedFluxes(data, fluxes, bulgeFluxes, diskFluxes)
        collectForcedMags(data, mags, bulgeMags, diskMags)
        data.fractions.append(frac)
    data.allForcedFractions.append(data.fractions)

def prepareGalaxyWithTractor(tractor, band, w, h):
    galaxy = makeTractorGalaxy(band, w, h)
    galaxy.pos = tractor.catalog[0].pos
    galaxy.shapeExp = tractor.catalog[0].shapeExp
    galaxy.shapeDev = tractor.catalog[0].shapeDev
    return galaxy

def collectForcedFluxes(data, fluxes, bulgeFluxes, diskFluxes):
    flux, stDev = findAvgStDev(fluxes)
    bulgeFlux, bulgeStDev = findAvgStDev(bulgeFluxes)
    diskFlux, diskStDev = findAvgStDev(diskFluxes)
    data.forTracFluxes.append(flux)
    data.forTracStDevs.append(stDev)
    data.bForTracFluxes.append(bulgeFlux)
    data.bForTracStDevs.append(bulgeStDev)
    data.dForTracFluxes.append(diskFlux)
    data.dForTracStDevs.append(diskStDev)

def collectForcedMags(data, mags, bulgeMags, diskMags):
    mag, magDev = findAvgStDev(mags)
    bulgeMag, bulgeMagDev = findAvgStDev(bulgeMags)
    diskMag, diskMagDev = findAvgStDev(diskMags)
    data.forTracMags.append(mag)
    data.forTracMagDevs.append(magDev)
    data.bForTracMags.append(bulgeMag)
    data.bForTracMagDevs.append(bulgeMagDev)
    data.dForTracMags.append(diskMag)
    data.dForTracMagDevs.append(diskMagDev)

def makeTractorGalaxy(band, w, h): 
    # Make a basic FixedCompositeGalaxy Tractor source
    galaxy = FixedCompositeGalaxy(PixPos(w/2, h/2),
                                  NanoMaggies(**dict([(band, 10.)])), 
                                  SoftenedFracDev(0.5),
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
    
def optimizeForcedTractor(tractor, band):
    # Freeze all image calibration parameters
    tractor.freezeParam('images')
    # Freeze all non-band and non-fracDev parameters
    tractor.freezeAllRecursive()
    tractor.catalog[0].thawParam('fracDev')
    tractor.thawPathsTo(band)
    tractor.catalog[0].pos.freezeParams('y')
    # Take several linearized least squares steps
    for i in range(20):
        dlnp,X,alpha = tractor.optimize(shared_params=False)
        print 'dlnp', dlnp
        if dlnp < 1e-3:
            break
    return tractor 

def getFluxesAndMags(data, band, tractor):
    # Isolate desired band and fracDev
    tractor.freezeAllRecursive()
    tractor.catalog[0].thawParam('fracDev')
    tractor.thawPathsTo(band)
    tractor.catalog[0].pos.freezeParams('y')
    # Get variance and flux of optimized Tractor object
    bandVar = tractor.optimize(variance=True, just_variance=True, 
                               shared_params=False)
    bandFlux = tractor.getParams()
    bandFlux[0]=bandFlux[0]#/data.pixel_scale**2
    # If flux is negative, tell ZEBRA to ignore the magnitude by making it 99
    if bandFlux[0] <= 0:
        bandMag = [99]
        bandMagError = [0]
    else:
        # Get magnitudes from the flux and variance
        npFlux = np.array([bandFlux[0]])
        invvar = 1./np.array([bandVar[0]])
        bandMag,bandMagError=NanoMaggies.fluxErrorsToMagErrors(npFlux,invvar)
    bright = tractor.catalog[0].getBrightness()
    flux = tractor.getImage(0).getPhotoCal().brightnessToCounts(bright)
    mag = bandMag[0]
    return flux, mag

###############################################################################
##FIXME - Plot-generating functions
###############################################################################

def newPlot(data,fluxRatio,redshift,fluxIndex,shiftIndex):
    # Make plots for fluxes, colors, and magnitudes
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
        plt.title('Flux across bands for flux ratios and redshifts, Tractor')
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
    plt.close()
    
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
    plt.close()

def figure3Setup(data):
    # Swap focus to magnitude plot, add finishing touches to plot
    plt.figure(3)
    plt.xlim([-1,len(data.filter_names)])
    plt.ylim(ymax = 30) # FIXME - Limits are questionable
    plt.xlabel('Band')
    plt.ylabel('AB Magnitude')
    plt.grid()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()
    if data.useTractor == True:
        plt.title('Mag across bands for flux ratios and redshifts, Tractor')
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
    plt.close()

def makeRedshiftPlot(data, outputRedshifts, loError, hiError, mark, name):
    # Create plot for redshifts, plot output vs expected
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
                     marker = mark, linestyle = "none", 
                     mfc = "%s" %color, capsize = 10, ecolor = "%s"%color,
                     label = "%0.2f B/T, %s" % (fluxRatio, name))

def figure4Setup(data):
    # Finallizing redshift plot
    plt.figure(4)
    redshiftLine = linspace(0,2)
    plt.plot(redshiftLine,redshiftLine,linewidth=1,label="Expected Redshifts")
    plt.xlim([0,2])
    plt.ylim(0)
    plt.xlabel('Input Redshifts')
    plt.ylabel('Output Redshifts')
    plt.grid()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()
    plt.title("Measured Redshifts vs. Actual Redshifts, All Methods")
    saveName = "Redshifts_all.png"
    plt.savefig(saveName,bbox_extra_artists=(lgd,),bbox_inches='tight')
        
###############################################################################
##FIXME - ZEBRA read/write files
###############################################################################

# make ZEBRA-friendly output file
# readFile and writeFile taken from CMU 15-112 class notes:
# http://www.kosbie.net/cmu/spring-13/15-112/handouts/fileWebIO.py
# Zurich Extragalactic Bayesian Redshift Analyzer (ZEBRA) used to calculate
# redshifts: Feldmann, Carollo, Porciani, Lilly et al., MNRAS 372, 564 (2006)
# http://adsabs.harvard.edu/abs/2006MNRAS.372..565F
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

def makeCatalog(data, avgMags, magStDevs, catName):
    # Write magnitude data to a ZEBRA-friendly catalog file
    catPath = data.catpath + catName
    fluxData = zip(avgMags, magStDevs)
    contents = ""
    for i in xrange(len(fluxData)):
        for j in xrange(len(fluxData[0])):
            contents += (str(fluxData[i][j]) + " ")
    contents += ("\n")
    if not os.path.exists(catPath):
        writeFile(catPath, contents)
    else:
        writeFile(catPath, contents, "a")

def runAllZebraScripts(data):
    basicOutput = forcedOutput = tractorOutput = forcedTractorOutput = None
    if data.basic:
        basicOutput = makeBasicRedshiftPlots(data)
    if data.forced:
        forcedOutput = makeForcedRedshiftPlots(data)
    if data.Tractor:
        tractorOutput = makeTractorRedshiftPlots(data)
    if data.forcedTractor:
        forcedTractorOutput = makeForcedTractorRedshiftPlots(data)  
    figure4Setup(data)
    return basicOutput, forcedOutput, tractorOutput, forcedTractorOutput

def makeBasicRedshiftPlots(data):
    outShifts, loE, hiE = runZebraScript(data, data.basicScript)
    makeRedshiftPlot(data, outShifts, loE, hiE, "o",
                     "Basic")
    print np.array(outShifts), np.array(loE), np.array(hiE)
    return [[outShifts, loE, hiE]]

def makeForcedRedshiftPlots(data):
    forOutShifts, forLoE, forHiE = runZebraScript(data, data.forScript)
    makeRedshiftPlot(data, forOutShifts, forLoE, forHiE, "^",
                     "Forced")
    print np.array(forOutShifts), np.array(forLoE), np.array(forHiE)
    return [[forOutShifts, forLoE, forHiE]]

def makeTractorRedshiftPlots(data):
    tracOutShifts, tracLoE, tracHiE = runZebraScript(data, data.tracScript)
    makeRedshiftPlot(data, tracOutShifts, tracLoE, tracHiE, "s",
                     "Tractor, All Flux")
    print np.array(tracOutShifts), np.array(tracLoE), np.array(tracHiE)
    
    bTracOutShifts, bTracLoE, bTracHiE = runZebraScript(data,data.bTracScript)
    makeRedshiftPlot(data, bTracOutShifts, bTracLoE, bTracHiE, "*",
                     "Tractor, Bulge Flux")        
    print np.array(bTracOutShifts), np.array(bTracLoE), np.array(bTracHiE)

    dTracOutShifts, dTracLoE, dTracHiE = runZebraScript(data,data.dTracScript)
    makeRedshiftPlot(data, dTracOutShifts, dTracLoE, dTracHiE, "+",
                     "Tractor, Disk Flux")        
    print np.array(dTracOutShifts), np.array(dTracLoE), np.array(dTracHiE)
    return [[tracOutShifts, tracLoE, tracHiE],
            [bTracOutShifts, bTracLoE, bTracHiE],
            [dTracOutShifts, dTracLoE, dTracHiE]]

def makeForcedTractorRedshiftPlots(data):
    forTrOutShifts,forTrLoE,forTrHiE=runZebraScript(data, data.forTracScript)
    makeRedshiftPlot(data, forTrOutShifts,forTrLoE,forTrHiE, "v",
                     "Forced Tractor, All Flux")
    print np.array(forTrOutShifts),np.array(forTrLoE),np.array(forTrHiE)
    bForTrShifts,bForTrLoE,bForTrHiE=runZebraScript(data,data.bForTracScript)
    makeRedshiftPlot(data, bForTrShifts,bForTrLoE,bForTrHiE, "x",
                     "Forced Tractor, Bulge Flux")
    print np.array(bForTrShifts),np.array(bForTrLoE),np.array(bForTrHiE)
    dForTrShifts,dForTrLoE,dForTrHiE=runZebraScript(data, data.dForTracScript)
    makeRedshiftPlot(data, dForTrShifts,dForTrLoE,dForTrHiE, "d",
                     "Forced Tractor, Disk Flux")
    print np.array(dForTrShifts),np.array(dForTrLoE),np.array(dForTrHiE)
    return [[forTrOutShifts, forTrLoE, forTrHiE],
            [bForTrShifts, bForTrLoE, bForTrHiE],
            [dForTrShifts, dForTrLoE, dForTrHiE]]

def runZebraScript(data, scriptName):
    # Move to directory where ZEBRA scripts are located, run script
    os.chdir("../zebra-1.10/scripts/")
    subprocess.Popen(['./' + scriptName]).wait()
    datPath = '../examples/ML_notImproved/ML.dat'
    # Grab redshifts from the ZEBRA output ML.dat file
    (rs, loE, hiE) = readRedshifts(datPath)
    os.chdir(data.path)
    return rs, loE, hiE

def readRedshifts(filename):
    # Extract redshifts and errors from given file
    extractedList = extractData(readFile(filename))
    redshifts, loErrors, hiErrors = [], [], []
    for line in extractedList:
        items = readOutputFromLine(line)
        redshifts.append(float(items[2]))
        # Errors could be from columns 21 to 28 (alternating lo/hi)   
        loErrors.append(abs(float(items[20]) - float(items[2])))
        hiErrors.append(abs(float(items[21]) - float(items[2])))
    return redshifts, loErrors, hiErrors

def readOutputFromLine(line):
    (items, isWhitespace) = ([""], True)
    extractionIndex = charIndex = 0
    # Set up system that ignores whitespace and takes consecutive non-
    # whitespace characters as a single list element, convert to floats
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
    return items

def extractData(contents):
    # Take all relevant lines (i.e. those without '#') and store them in a list
    # of lines.
    contentList = string.split(contents,"\n")[:-1]
    extractedList = []
    for i in xrange(len(contentList)):
        if contentList[i][0] != "#":  extractedList.append(contentList[i])
    return extractedList
    
###############################################################################
##FIXME - Extra test functions
###############################################################################
       
def btRatiosAcrossBands(data):
    # Compare pure bulge and pure disk galaxy fluxes to get B/T ratios
    ratios = {}
    for i in xrange(len(data.filter_names)):
        bulge = data.allFluxes[-1][i]
        disk = data.allFluxes[0][i]
        ratio = bulge/(bulge+disk)
        ratios[data.filter_names[i]]=ratio
    return ratios

def getArrayFlux(data,images):
    # Test function getting fluxes from images by summing all pixel values
    fluxList = []
    for image in images:
        fluxList.append(image.array.sum())
    return fluxList


if __name__ == "__main__":
    main(sys.argv)
