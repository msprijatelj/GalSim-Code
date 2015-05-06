import os,numpy,subprocess,shutil,copy
import matplotlib.pyplot as plt
import scipy.optimize as so

filter_dir='filters'
sed_dir='seds/VIMOS_ADAPT_iter2'
realdata_dir='realdata'
dataname='cosmos_s14a0b_matched_calib.dat'

#use fracdev method?
#change this everytime you change fracdev!!
makeNewData=False

#make extra contour plots with data overlaid
overlay=False

filter_dir='filters'
sed_dir='seds/VIMOS_ADAPT_iter2'
realdata_dir='realdata'
dataname='cosmos_s14a0b_matched_calib.dat'

global_colors=["r","k","lightblue","darkred","olive","chartreuse","orange"]


common='flux_cmodel'

def getUsableGalaxies():
	usables=[]
	datadicts=getDataDicts(realdata_dir,dataname)
	for galdata in datadicts:
		if isUsable(galdata):
			usables.append(relevantData(galdata))
	return usables

def getDataDicts(dir,dataname):
	keys=getKeys(dir)
	valss=getValss(dir,dataname)
	datas=map(lambda vals: dict(zip(keys,vals)),valss)	
	return datas

def getKeys(dir):
	keyfile=os.path.join(dir,'keys.txt')
	keyfile=open(keyfile)
	keys=keyfile.read().split('\n')
	keyfile.close()
	return keys

def getValss(dir,dataname):
	valfile=os.path.join(dir,dataname)
	#valfile=open(valfile)
	#valss=valfile.read().split('\n')
	#valfile.close()
	#the last element of the data file is just an empty line at the bottom...	
	#valss.pop(-1)
	#the second to last element in each galaxy's datapoint is an empty string because there are
	#inexplicably two spaces between the last two columns instead of one... for every entry. DUMB
	#valss=map(lambda vals: map(lambda val: eval(val),vals.split(' ')[:-2]+[vals.split(' ')[-1]]),valss)
	valss=numpy.loadtxt(valfile)
	return valss

def isUsable(galdata):
	if not galdata['classification_extendedness']:
		return False	

	for fracdev in map(lambda band: band+'_fracdev','i'):
		if galdata[fracdev] < 1/3.0:
			return False
	return True

def relevantData(galdata):
	keys=['30-band_photo-z']+(map(lambda band: band+common,'grizy')+map(lambda band: band+common+'_err','grizy')+
		map(lambda band: band+'_fracdev','grizy')+map(lambda band: band+common+'_err','grizy'))
	vals=map(lambda key: galdata[key],keys)
	return dict(zip(keys,vals))

def writeCat(usables,outputpath,lambdafn):
	try:
		out=open(outputpath,"w+")
		out.write('#')
		
		for band in 'grizy':
			out.write('{}*{} | {} | '.format(band+common,band+'_fracdev',band+common+'_err'))
		out.write('30-band_photo-z\n')
		lines=makeLines(usables,lambdafn)
		for line in lines:
			out.write(line)
	except:
		raise
	finally:
		out.close()

def makeLines(usables,lambdafn):
	lines=[]
	for gal in usables:
		line=''
		for band in 'grizy':
			line+='{} {} '.format(lambdafn((gal,band)),gal[band+common+'_err'])
		line+='{}\n'.format(gal['30-band_photo-z'])
		lines+=[line]
	return lines	

def checkSEDs(sedspath,base):
	print "checking SEDs"
	for i in range(31):
		print 'checking',base%i	
		checkSED(sedspath,base%i)

def checkSED(sedspath,sedfile):
	sed=numpy.loadtxt(os.path.join(sedspath,sedfile))	
	comp=sed[0][0]
	for i,row in enumerate(sed):
		if '25' in sedfile:
			print row[0]-comp
		if row[0] < comp:
			print i,row
			assert False
		comp=row[0]

def makeData(fracdev):
	subprocess.Popen('/home/agurvich/S14/GalSim-Code/'+'callzebra'+'_fracdev'*fracdev,close_fds=True).wait()

def plotThing(grid,xlists,ylists,labels,formts=["^","v","*","<","x",">"],offset=0):
		lineList=[]	
		thresh=500
		for i in xrange(len(ylists)):
			label=labels[i]
			xs=xlists[i]
			ys=ylists[i]
			assert len(xs)==len(ys)
			color=global_colors[i]
			lineList.append(grid.plot(map(lambda snr: snr * (1+ offset*min(.04,0.02*i))*(len(ylists)<=4) + (snr + .05*i*min(xs)*offset)*(len(ylists)>4) ,
				xs),ys,formts[i],color=color,label=label,alpha=0.15)[0])
		#axis=plt.gca()
		#axis.set_xlim(min(xlist)*.9,max(xlist)*1.1)
		return lineList

def nameAxes(ax,title,xname,yname,logflag=(0,0),make_legend=0,verty=0,
	subtitle=None,off_legend=0,axesKeys={'numpoints':1}):
	if yname!=None:
		if verty:
			ax.set_ylabel(yname,fontsize=16)
		else:
			ax.set_ylabel(yname)
	if xname!=None:
		ax.set_xlabel(xname)
	if logflag[0]:
		ax.set_xscale('log')
	if logflag[1] :
		ax.set_yscale('log')
	if title!=None:
		ax.set_title(title)
	if subtitle:
		ax.text(.01,.04,subtitle,transform=ax.transAxes,
		verticalalignment='center',horizontalalignment='left')
	if make_legend:
		if off_legend:
			return ax.legend(loc=0,bbox_to_anchor=(1.02,1),fontsize=12,**axesKeys)
		else:
			ax.legend(fontsize=12,loc=0,**axesKeys)

	return ax.get_legend_handles_labels()

def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level
 
def density_contour(xdata, ydata, nbins_x, nbins_y, ax=None, **contour_kwargs):
    """ Create a density contour plot.

    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """
 
    H, xedges, yedges = numpy.histogram2d(xdata, ydata, bins=(nbins_x,nbins_y), normed=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))
 
    pdf = (H*(x_bin_sizes*y_bin_sizes))
 
    one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99))
    levels = [one_sigma, two_sigma, three_sigma]
 
    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T
 
    if ax == None:
        contour = plt.contourf(X, Y, Z,**contour_kwargs)
    else:
        contour = ax.contour(X, Y, Z, levels=levels, origin="lower", **contour_kwargs)
 
    return contour

def density_plot(grid,xs,ys,nbins,nlevels=0,log=0):
	eps=10.0**-6
	maxx=max(max(xs),max(ys))
	minn=min(min(xs),min(ys))
	dz=(maxx-minn)/nbins
	xsp=copy.copy(xs)
	ysp=copy.copy(ys)
	total=0
	xzs=numpy.arange(minn,maxx+dz,dz)
	yzs=numpy.arange(minn,maxx+dz,dz)
	countss=[[0 for z in xrange(len(xzs))] for z in xrange(len(yzs))]
	while len(xsp)!=0:
		found=0
		assert len(xsp)==len(ysp)
		x=xsp.pop(0)
		y=ysp.pop(0)
		for iyz,yz in enumerate(yzs):
			for ixz,xz in enumerate(xzs):
				#if ((xz-round(xz,2))**2)**0.5 <= eps:
					#xz=round(xz,2)
				if xz-dz/2.0 < x <= xz+dz/2.0+.00001 and yz - dz/2.0 < y <= yz + dz/2.0:
					found=1
					countss[iyz][ixz]+=1
					total+=1
					break
			if found:
				break
					
		if not found:
			print x,y,dz/2.0
			print xzs
			print yzs
			assert False
	print 'total',total,'xs',len(xs)				
	try:
		assert len(xs)==total
	except:
		print len(xsp),len(ysp)
		print len(xs),total
		assert False
	maxx=0
	for count in countss:
		if max(count)>maxx:
			maxx=max(count)
		#print count
	#maxx=max(max(countss))
	if type(nlevels)==type(1):
		if log: 
			levels=[0]
			maxlog=numpy.log10(maxx)
			for i in numpy.arange(maxlog/nlevels,maxlog*(1+1.0/nlevels),maxlog/nlevels):
				levels+=[10**i]
		else:
			levels=numpy.arange(0.0,maxx*(1+1.0/nlevels),maxx/nlevels)
	else:
		assert type(nlevels)==type([1.0])
		levels=nlevels
	return grid.contourf(xzs,yzs,countss,levels=levels,origin='lower'),levels


def threematched((us_fracdev,dicts_fracdev),(us,dicts)):
	#assert data validity
	assert len(us_fracdev)==len(us)
	assert len(dicts_fracdev)==len(dicts)
	assert us_fracdev==us
	return dicts,dicts_fracdev,us

def filter(threematched,icutoff=1/3.0):
	threematched1=([],[],[])
	assert len(threematched1)==len(threematched)
	assert len(threematched[0])==len(threematched[1])==len(threematched[2])
	while len(threematched[2])!=0:
		d0=threematched[0].pop(0)
		d1=threematched[1].pop(0)
		d2=threematched[2].pop(0)
		if d2['i_fracdev']<icutoff:
			pass
		else:
			threematched1[0].append(d0)
			threematched1[1].append(d1)
			threematched1[2].append(d2)
	return threematched1

def run(fracdev):	
	datapath=os.path.join(realdata_dir,'zebra_out'+'_fracdev'*fracdev+'.dat')

	usables=getUsableGalaxies()
	#writeCat(usables,'conf_files/cosmos_s14_catalog.cat',lambda (gal,band): gal[band+common])
	#print usables[0],len(usables)
	#checkSEDs(sed_dir,"sed%.2d.out")

	if makeNewData:
		writeCat(usables,'conf_files/'+'fracdev_'*fracdev+'cosmos_s14_catalog.cat',lambda (gal,band): gal[band+common]*(gal[band+'_fracdev']**fracdev))
		makeData(fracdev)

	datas=numpy.loadtxt(datapath)

	keys=['z_cat','usedFilters','z_phot','T_phot','a','chi^2','M_B','D_L','z_photz','T_photz','az','chi^2z','M_Bz', 'D_Lz','z_phott','T_phott','at','chi^2t','M_Bt','D_Lt','z_ph2_L0.68','z_ph2_H0.68','z_ph2_L0.95','z_ph2_H0.95', 'z_ph_L0.68','z_ph_H0.68','z_ph_L0.95','z_ph_H0.95','z_phz_L0.68','z_phz_H0.68','z_phz_L0.95','z_phz_H0.95','z_pht_L0.68','z_pht_H0.68', 'z_pht_L0.95','z_pht_H0.95','T_ph2_L0.68','T_ph2_H0.68','T_ph2_L0.95','T_ph2_H0.95','T_ph_L0.68','T_ph_H0.68','T_ph_L0.95','T_ph_H0.95', 'T_phz_L0.68','T_phz_H0.68','T_phz_L0.95','T_phz_H0.95','T_pht_L0.68','T_pht_H0.68','T_pht_L0.95','T_pht_H0.95']

	datadicts=map(lambda data: dict(zip(keys,data)) ,datas)

	#matched=zip(usables,datadicts)
	return usables,datadicts




##############################################################################

def makePlots(icutoff):
	matched_fracdev=run(True)
	matched=run(False)


	savedir='plots%.3f'%icutoff #+'_fracdev'*fracdev
	threematched=threematched(matched_fracdev,matched)
	threematched = filter(threematched,icutoff)
	try:
		shutil.rmtree(savedir)
	except:
		print "haven't run %.3f yet" % icutoff
		pass

	os.mkdir(savedir)

	temps=list(set(map(lambda datadict: datadict['T_photz'],threematched[0])))


	summ=0
	for i in xrange(len(temps)/10):
		temps_to_use=temps[10*i:10*(i+1)]
		#popped=0
		xs,ys,ys_fracdev=[],[],[]
		for j,data_point in enumerate(zip(*threematched)):
			if data_point[0]['T_photz'] in temps_to_use:
				tuplepair=data_point
				#tuplepair=matched.pop(j-popped)
				#popped+=1
				xs+=[tuplepair[2]['30-band_photo-z']]
				ys+=[tuplepair[0]['z_photz']]
				ys_fracdev+=[tuplepair[1]['z_photz']]

		fig=plt.subplots(2,sharex=True,sharey=True)[0]
		axes=fig.get_axes()
		grid=axes[0]
		contour,levels=density_plot(grid,xs,ys,50,15,log=1)
		nameAxes(grid,'Z_cat vs z_photz '*0+'%s pts used: %d icutoff: %.3f' %(0*temps_to_use,len(ys),icutoff),'Z_cat','z_photz')
		grid=axes[1]
		contour,levels=density_plot(grid,xs,ys_fracdev,50,levels,log=1)
		nameAxes(grid,'Z_cat vs z_photz '*0+'%s%s pts used: %d icutoff: %.3f' %(0*temps_to_use,' _fracdev',len(ys_fracdev),icutoff),'Z_cat','z_photz')
		if overlay:
			plotThing(plt.gca(),[xs,[0,5]],[ys,[0,5]],['label','eye guide y=x'],formts=['.','--'])
		grid.set_xlim([0,5.0])

		fig.subplots_adjust(right=0.8)
		plt.colorbar(contour,cax=fig.add_axes([0.85,0.15,0.05,0.7]))#,orientation='horizontal')
		plt.savefig(os.path.join(savedir,'zplot_temps%d.png'%i))
		#contour=density_contour(xs, ys, 10, 10)
		#plt.clabel(contour,inline=1,fontsize=10)
		plt.show()
		plt.close()
		summ+=len(xs)

	#try:
		#assert len(matched)==summ
	#except:
		#print "Didn't catch them all"
		#print "Missed:",len(matched)
		#print "For instance, the first one had a template of:"
		#print set(map(lambda match: match[1]['T_photz'],matched))
		#print matched[2][1]['T_photz']
