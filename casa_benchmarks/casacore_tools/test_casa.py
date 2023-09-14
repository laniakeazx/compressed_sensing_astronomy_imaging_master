# import os

# print("installing pre-requisite packages...")
# os.system("apt-get install libgfortran3")
#
# print("installing casa...")
# os.system("pip install --index-url https://casa-pip.nrao.edu:443/repository/pypi-group/simple casatasks==6.2.0.124")
# os.system("pip install --index-url https://casa-pip.nrao.edu:443/repository/pypi-group/simple casadata")
# print("complete")


from casatools import simulator, image, table, coordsys, measures, componentlist, quanta, ctsys
from casatasks import tclean, ft, imhead, listobs, exportfits, flagdata, bandpass, applycal
from casatasks.private import simutil

import os
import pylab as pl
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


# Instantiate all the required tools
sm = simulator()
ia = image()
tb = table()
cs = coordsys()
me = measures()
qa = quanta()
cl = componentlist()
mysu = simutil.simutil()


def makeMSFrame(msname = 'sim_data.ms'):
    """
    Construct an empty Measurement Set that has the desired observation setup.
    """

    os.system('rm -rf '+msname)

    ## Open the simulator
    sm.open(ms=msname);

    ## Read/create an antenna configuration.
    ## Canned antenna config text files are located here : /home/casa/data/trunk/alma/simmos/*cfg
    antennalist = os.path.join( ctsys.resolve("alma/simmos") ,"vla.d.cfg")

    ## Fictitious telescopes can be simulated by specifying x, y, z, d, an, telname, antpos.
    ##     x,y,z are locations in meters in ITRF (Earth centered) coordinates.
    ##     d, an are lists of antenna diameter and name.
    ##     telname and obspos are the name and coordinates of the observatory.
    (x,y,z,d,an,an2,telname, obspos) = mysu.readantenna(antennalist)

    ## Set the antenna configuration
    sm.setconfig(telescopename=telname,
                     x=x,
                     y=y,
                     z=z,
                     dishdiameter=d,
                     mount=['alt-az'],
                     antname=an,
                     coordsystem='global',
                     referencelocation=me.observatory(telname));

    ## Set the polarization mode (this goes to the FEED subtable)
    sm.setfeed(mode='perfect R L', pol=['']);

    ## Set the spectral window and polarization (one data-description-id).
    ## Call multiple times with different names for multiple SPWs or pol setups.
    sm.setspwindow(spwname="LBand",
                   freq='1.0GHz',
                   deltafreq='0.1GHz',
                   freqresolution='0.2GHz',
                   nchannels=10,
                   stokes='RR LL');

    ## Setup source/field information (i.e. where the observation phase center is)
    ## Call multiple times for different pointings or source locations.
    sm.setfield( sourcename="fake",
                 sourcedirection=me.direction(rf='J2000', v0='19h59m28.5s',v1='+40d44m01.5s'));

    ## Set shadow/elevation limits (if you care). These set flags.
    sm.setlimits(shadowlimit=0.01, elevationlimit='1deg');

    ## Leave autocorrelations out of the MS.
    sm.setauto(autocorrwt=0.0);

    ## Set the integration time, and the convention to use for timerange specification
    ## Note : It is convenient to pick the hourangle mode as all times specified in sm.observe()
    ##        will be relative to when the source transits.
    sm.settimes(integrationtime='2000s',
                usehourangle=True,
                referencetime=me.epoch('UTC','2019/10/4/00:00:00'));

    ## Construct MS metadata and UVW values for one scan and ddid
    ## Call multiple times for multiple scans.
    ## Call this with different sourcenames (fields) and spw/pol settings as defined above.
    ## Timesteps will be defined in intervals of 'integrationtime', between starttime and stoptime.
    sm.observe(sourcename="fake",
               spwname='LBand',
               starttime='-5.0h',
               stoptime='+5.0h');

    ## Close the simulator
    sm.close()

    ## Unflag everything (unless you care about elevation/shadow flags)
    flagdata(vis=msname,mode='unflag')


def plotData(msname='out_8h.ms', myplot='uv'):
    """
    Options : myplot='uv'
              myplot='data_spectrum'
    """
    from matplotlib.collections import LineCollection
    tb.open(msname)

    # UV coverage plot
    if myplot=='uv':
        pl.figure(figsize=(5,5),dpi=300)
        pl.clf()
        uvw = tb.getcol('UVW')

        pl.plot( uvw[0], uvw[1], ',',color='steelblue',markersize=0.01)
        pl.plot( -uvw[0], -uvw[1], ',',color='steelblue',markersize=0.01)

        pl.xlabel('U(m)', fontsize=10)
        pl.ylabel('V(m)', fontsize=10)
        pl.title('UV Coverage(Start time 08:00:00)',fontsize=10)
        # pl.semilogy()
        # pl.semilogx()
        pl.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        pl.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        # pl.xscale('log')
        # pl.yscale('log')
        pl.subplots_adjust(left=0.105, right=0.990, bottom=0.105, top=0.945)
        # pl.subplots_adjust(left=0.13, right=0.905, bottom=0.15, top=0.90)
    # Spectrum of chosen column. Make a linecollection out of each row in the MS.
    if myplot=='data_spectrum' or myplot=='corr_spectrum' or myplot=='resdata_spectrum'  or myplot=='rescorr_spectrum' or myplot=='model_spectrum':
        dats=None
        if myplot=='data_spectrum':
            dats = tb.getcol('DATA')
        if myplot=='corr_spectrum':
            dats = tb.getcol('CORRECTED_DATA')
        if myplot=='resdata_spectrum':
            dats = tb.getcol('DATA') - tb.getcol('MODEL_DATA')
        if myplot=='rescorr_spectrum':
            dats = tb.getcol('CORRECTED_DATA') - tb.getcol('MODEL_DATA')
        if myplot=='model_spectrum':
            dats = tb.getcol('MODEL_DATA')

        xs = np.zeros((dats.shape[2],dats.shape[1]),'int')
        for chan in range(0,dats.shape[1]):
            xs[:,chan] = chan

        npl = dats.shape[0]
        fig, ax = pl.subplots(1,npl,figsize=(10,4))

        for pol in range(0,dats.shape[0]):
            x = xs
            y = np.abs(dats[pol,:,:]).T
            data = np.stack(( x,y ), axis=2)
            ax[pol].add_collection(LineCollection(data))
            ax[pol].set_title(myplot + ' \n pol '+str(pol))
            ax[pol].set_xlim(x.min(), x.max())
            ax[pol].set_ylim(y.min(), y.max())
        # pl.show()

    tb.close()
    pl.show()


makeMSFrame()
plotData(myplot='uv')