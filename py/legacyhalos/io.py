"""
legacyhalos.io
==============

Code to read and write the various legacyhalos files.

"""
import os, pdb
import numpy as np
import numpy.ma as ma

import fitsio
import astropy.units as u
from astropy.table import Table, Column
from astrometry.util.fits import fits_table

def legacyhalos_dir():
    if 'LEGACYHALOS_DIR' not in os.environ:
        print('Required ${LEGACYHALOS_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LEGACYHALOS_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def legacyhalos_data_dir():
    if 'LEGACYHALOS_DATA_DIR' not in os.environ:
        print('Required ${LEGACYHALOS_DATA_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LEGACYHALOS_DATA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def legacyhalos_html_dir():
    if 'LEGACYHALOS_HTML_DIR' not in os.environ:
        print('Required ${LEGACYHALOS_HTML_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LEGACYHALOS_HTML_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

# build out the FITS header
def legacyhalos_header(hdr=None):
    """Build a header with code versions, etc.

    """
    import subprocess
    from astropy.io import fits
    import pydl
    import legacyhalos

    if hdr is None:
        hdr = fits.header.Header()

    try:
        cmd = 'cd {} && git describe --always'.format(os.path.dirname(legacyhalos.__file__))
        ver = subprocess.check_output(cmd, shell=True, universal_newlines=True).strip()
        hdr['LEGHALOV'] = (ver, 'legacyhalos git version')
    except:
        print('WARNING: Problem writing legacyhalos version to the header.')

    depvers, headers = [], []
    for name, pkg in [('pydl', pydl)]:
        hdr[name] = (pkg.__version__, '{} version'.format(name))

    return hdr
    
def _missing_files_one(args):
    """Wrapper for the multiprocessing."""
    return missing_files_one(*args)

def missing_files_one(checkfile, dependsfile, clobber):
#def missing_files_one(galaxy, galaxydir, filesuffix, dependson, clobber):
    #checkfile = os.path.join(galaxydir, '{}{}'.format(galaxy, filesuffix))
    #print('missing_files_one: ', checkfile)
    #print(checkfile, dependsfile, clobber)
    from pathlib import Path
    #from glob import glob
    #if os.path.isfile(checkfile) and clobber is False:
    #checkfile = glob(checkfile)
    #if len(checkfile) > 0:
    #    checkfile = checkfile[0]
    #else:
    #    checkfile = '_'
    if Path(checkfile).exists() and clobber is False:
        # Is the stage that this stage depends on done, too?
        #print(checkfile, dependsfile, clobber)
        if dependsfile is None:
            return 'done'
        else:
            if Path(dependsfile).exists():
            #if os.path.isfile(dependsfile):
                return 'done'
            else:
                return 'todo'
    else:
        #print('missing_files_one: ', checkfile)
        # Did this object fail?
        if checkfile[-6:] == 'isdone':
            failfile = checkfile[:-6]+'isfail'
            if Path(failfile).exists():
            #if os.path.isfile(failfile):
                if clobber is False:
                    return 'fail'
                else:
                    os.remove(failfile)
                    return 'todo'
            else:
                return 'todo'
            #if dependsfile is None:
            #    return 'todo'
            #else:
            #    if os.path.isfile(dependsfile):
            #        return 'todo'
            #    else:
            #        return 'todo'
        else:
            if dependsfile is not None:
                if os.path.isfile(dependsfile):
                    return 'todo'
                else:
                    print('Missing depends file {}'.format(dependsfile))
                    return 'fail'
            else:
                return 'todo'

        return 'todo'
            
def get_run(onegal, racolumn='RA', deccolumn='DEC'):
    """Get the run based on a simple declination cut."""
    if onegal[deccolumn] > 32.375:
        if onegal[racolumn] < 45 or onegal[racolumn] > 315:
            run = 'south'
        else:
            run = 'north'
    else:
        run = 'south'
    return run

# ellipsefit data model
def _get_ellipse_datamodel(sbthresh, apertures, bands=['g', 'r', 'z'],
                           add_datamodel_cols=None, copy_mw_transmission=False):
    cols = [
        ('bands', None),
        ('refband', None),
        ('refpixscale', u.arcsec / u.pixel),
        ('success', None),
        ('fitgeometry', None),
        ('input_ellipse', None),
        ('largeshift', None),
        ('small_tractor', None),

        ('x0_moment', u.pixel),
        ('y0_moment', u.pixel),
        ('ra_moment', u.degree),
        ('dec_moment', u.degree),
        ('sma_moment', u.arcsec),
        ('majoraxis', u.pixel), # in the reference band
        ('pa_moment', u.degree),
        ('ba_moment', None),
        ('eps_moment', None),
        #('theta_moment', u.degree),
        ('maxsma', u.pixel),

        ('integrmode', None),
        ('sclip', None),
        ('nclip', None),

        #('psfsigma_g', u.pixel),
        #('psfsigma_r', u.pixel),
        #('psfsigma_z', u.pixel),

        ('psfsize_g', u.arcsec),
        #('psfsize_min_g', u.arcsec),
        #('psfsize_max_g', u.arcsec),
        ('psfsize_r', u.arcsec),
        #('psfsize_min_r', u.arcsec),
        #('psfsize_max_r', u.arcsec),
        ('psfsize_z', u.arcsec),
        #('psfsize_min_z', u.arcsec),
        #('psfsize_max_z', u.arcsec),

        ('psfdepth_g', u.mag),
        #('psfdepth_min_g', u.mag),
        #('psfdepth_max_g', u.mag),
        ('psfdepth_r', u.mag),
        #('psfdepth_min_r', u.mag),
        #('psfdepth_max_r', u.mag),
        ('psfdepth_z', u.mag),
        #('psfdepth_min_z', u.mag),
        #('psfdepth_max_z', u.mag),

        ('refband_width', u.pixel),
        ('refband_height', u.pixel)]

    if copy_mw_transmission:
        cols.append(('ebv', u.mag))

    for band in bands:
        if copy_mw_transmission:
            cols.append(('mw_transmission_{}'.format(band.lower()), None))
        cols.append(('sma_{}'.format(band.lower()), u.pixel))
        cols.append(('intens_{}'.format(band.lower()), 'nanomaggies arcsec-2'))#1e-9*u.maggy/u.arcsec**2))
        cols.append(('intens_err_{}'.format(band.lower()), 'nanomaggies arcsec-2'))#1e-9*u.maggy/u.arcsec**2))
        cols.append(('eps_{}'.format(band.lower()), None))
        cols.append(('eps_err_{}'.format(band.lower()), None))
        cols.append(('pa_{}'.format(band.lower()), u.degree))
        cols.append(('pa_err_{}'.format(band.lower()), u.degree))
        cols.append(('x0_{}'.format(band.lower()), u.pixel))
        cols.append(('x0_err_{}'.format(band.lower()), u.pixel))
        cols.append(('y0_{}'.format(band.lower()), u.pixel))
        cols.append(('y0_err_{}'.format(band.lower()), u.pixel))
        cols.append(('a3_{}'.format(band.lower()), None)) # units?
        cols.append(('a3_err_{}'.format(band.lower()), None))
        cols.append(('a4_{}'.format(band.lower()), None))
        cols.append(('a4_err_{}'.format(band.lower()), None))
        cols.append(('rms_{}'.format(band.lower()), 'nanomaggies arcsec-2'))#1e-9*u.maggy/u.arcsec**2))
        cols.append(('pix_stddev_{}'.format(band.lower()), 'nanomaggies arcsec-2'))#1e-9*u.maggy/u.arcsec**2))
        cols.append(('stop_code_{}'.format(band.lower()), None))
        cols.append(('ndata_{}'.format(band.lower()), None))
        cols.append(('nflag_{}'.format(band.lower()), None))
        cols.append(('niter_{}'.format(band.lower()), None))        

    for thresh in sbthresh:
        cols.append(('sma_sb{:0g}'.format(thresh), u.arcsec))
    for thresh in sbthresh:
        cols.append(('sma_ivar_sb{:0g}'.format(thresh), 1/u.arcsec**2))
    for band in bands:
        for thresh in sbthresh:
            cols.append(('flux_sb{:0g}_{}'.format(thresh, band.lower()), 'nanomaggies'))#1e-9*u.maggy))
        for thresh in sbthresh:
            cols.append(('flux_ivar_sb{:0g}_{}'.format(thresh, band.lower()), 'nanomaggies-2'))#1e18/u.maggy**2))
        for thresh in sbthresh:
            cols.append(('fracmasked_sb{:0g}_{}'.format(thresh, band.lower()), None))

    for iap, ap in enumerate(apertures):
        cols.append(('sma_ap{:02d}'.format(iap+1), u.arcsec))
    for band in bands:
        for iap, ap in enumerate(apertures):
            cols.append(('flux_ap{:02d}_{}'.format(iap+1, band.lower()), 'nanomaggies'))#1e-9*u.maggy))
        for iap, ap in enumerate(apertures):
            cols.append(('flux_ivar_ap{:02d}_{}'.format(iap+1, band.lower()), 'nanomaggies-2'))#1e18/u.maggy**2))
        for iap, ap in enumerate(apertures):
            cols.append(('fracmasked_ap{:02d}_{}'.format(iap+1, band.lower()), None))

    for band in bands:
        cols.append(('cog_sma_{}'.format(band.lower()), u.arcsec))
        cols.append(('cog_flux_{}'.format(band.lower()), 'nanomaggies'))#1e-9*u.maggy))
        cols.append(('cog_flux_ivar_{}'.format(band.lower()), 'nanomaggies-2'))#1e18/u.maggy**2))

    for band in bands:
        cols.append(('cog_mtot_{}'.format(band.lower()), u.mag))
        cols.append(('cog_mtot_ivar_{}'.format(band.lower()), 1/u.mag**2))
        cols.append(('cog_m0_{}'.format(band.lower()), u.mag))
        cols.append(('cog_m0_ivar_{}'.format(band.lower()), 1/u.mag**2))
        cols.append(('cog_alpha1_{}'.format(band.lower()), None))
        cols.append(('cog_alpha1_ivar_{}'.format(band.lower()), None))
        cols.append(('cog_alpha2_{}'.format(band.lower()), None))
        cols.append(('cog_alpha2_ivar_{}'.format(band.lower()), None))
        cols.append(('cog_chi2_{}'.format(band.lower()), None))
        cols.append(('cog_sma50_{}'.format(band.lower()), u.arcsec))

    if add_datamodel_cols is not None:
        cols = cols + add_datamodel_cols

    return cols

def get_ellipsefit_filename(galaxy, galaxydir, filesuffix='', galaxy_id=''):
    
    if type(galaxy_id) is not str:
        galaxy_id = str(galaxy_id)

    if galaxy_id.strip() == '':
        galid = ''
    else:
        galid = '-{}'.format(galaxy_id)
        
    if filesuffix.strip() == '':
        fsuff = ''
    else:
        fsuff = '-{}'.format(filesuffix)
        
    # ellipsefitfile = os.path.join(galaxydir, '{}{}-ellipse{}.fits'.format(galaxy, fsuff, galid))
    ellipsefitfile = os.path.join(galaxydir, '{}{}{}-ellipse.fits'.format(galaxy, fsuff, galid))

    return ellipsefitfile

def write_ellipsefit(galaxy, galaxydir, ellipsefit, filesuffix='', galaxy_id='',
                     galaxyinfo=None, refband='r', bands=['g', 'r', 'z'],
                     add_datamodel_cols=None, sbthresh=None, apertures=None,
                     copy_mw_transmission=False, verbose=False):
    """Write out a FITS file based on the output of
    legacyhalos.ellipse.ellipse_multiband..

    ellipsefit - input dictionary

    """
    from astropy.io import fits
    from astropy.table import Table

    ellipsefitfile = get_ellipsefit_filename(galaxy, galaxydir, filesuffix=filesuffix, galaxy_id=galaxy_id)

    if sbthresh is None:
        from legacyhalos.ellipse import REF_SBTHRESH as sbthresh
    
    if apertures is None:
        from legacyhalos.ellipse import REF_APERTURES as apertures
    
    # Turn the ellipsefit dictionary into a FITS table, starting with the
    # galaxyinfo dictionary (if provided).
    out = Table()
    if galaxyinfo:
        for key in galaxyinfo.keys():
            data = galaxyinfo[key][0]
            if np.isscalar(data):
                data = np.atleast_1d(data)
            else:
                data = np.atleast_2d(data)
            unit = galaxyinfo[key][1] # add units
            col = Column(name=key, data=data, dtype=data.dtype, unit=unit)
            #if type(unit) is str:
            #else:
            #    #data *= unit
            #    #data = u.Quantity(value=data, unit=unit, dtype=data.dtype)
            #    col = Column(name=key, data=data, dtype=data.dtype)
            out.add_column(col)

    # First, unpack the nested dictionaries.
    datadict = {}
    for key in ellipsefit.keys():
        #if type(ellipsefit[key]) is dict: # obsolete
        #    for key2 in ellipsefit[key].keys():
        #        datadict['{}_{}'.format(key, key2)] = ellipsefit[key][key2]
        #else:
        #    datadict[key] = ellipsefit[key]
        datadict[key] = ellipsefit[key]
    del ellipsefit

    # Add to the data table
    datakeys = datadict.keys()
    for key, unit in _get_ellipse_datamodel(sbthresh, apertures, bands=bands, add_datamodel_cols=add_datamodel_cols,
                                            copy_mw_transmission=copy_mw_transmission):
        if key not in datakeys:
            print("Key {} not in datadict!".format(key))
            print(datadict.keys())
            raise ValueError('Data model change -- no column {} for galaxy {}!'.format(key, galaxy))
        data = datadict[key]
        if np.isscalar(data):# or len(np.array(data)) > 1:
            data = np.atleast_1d(data)
        #elif len(data) == 0:
        #    data = np.atleast_1d(data)
        else:
            data = np.atleast_2d(data)
        #if type(unit) is not str:
        #    data = u.Quantity(value=data, unit=unit, dtype=data.dtype)
        #col = Column(name=key, data=data)
        col = Column(name=key, data=data, dtype=data.dtype, unit=unit)
        #if 'z_cog' in key:
        #    print(key)
        #    pdb.set_trace()
        out.add_column(col)

    if np.logical_not(np.all(np.isin([*datakeys], out.colnames))):
        raise ValueError('Data model change -- non-documented columns have been added to ellipsefit dictionary!')

    # uppercase!
    for col in out.colnames:
        out.rename_column(col, col.upper())

    hdr = legacyhalos_header()

    #for col in out.colnames:
    #    print(col, out[col])

    hdu = fits.convenience.table_to_hdu(out)
    hdu.header['EXTNAME'] = 'ELLIPSE'
    hdu.header.update(hdr)
    hdu.add_checksum()

    hdu0 = fits.PrimaryHDU()
    hdu0.header['EXTNAME'] = 'PRIMARY'
    hx = fits.HDUList([hdu0, hdu])

    if verbose:
        print('Writing {}'.format(ellipsefitfile))
    tmpfile = ellipsefitfile+'.tmp'
    hx.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, ellipsefitfile)
    #hx.writeto(ellipsefitfile, overwrite=True, checksum=True)

    #out.write(ellipsefitfile, overwrite=True)
    #fitsio.write(ellipsefitfile, out.as_array(), extname='ELLIPSE', header=hdr, clobber=True)

def read_ellipsefit(galaxy, galaxydir, filesuffix='', galaxy_id='', verbose=True,
                    asTable=False, ellipsefitfile=None):
    """Read the output of write_ellipsefit. Convert the astropy Table into a
    dictionary so we can use a bunch of legacy code.

    """
    if galaxy_id.strip() == '':
        galid = ''
    else:
        galid = '-{}'.format(galaxy_id)
    if filesuffix.strip() == '':
        fsuff = ''
    else:
        fsuff = '-{}'.format(filesuffix)

    if ellipsefitfile is None:
        ellipsefitfile = os.path.join(galaxydir, '{}{}{}-ellipse.fits'.format(galaxy, fsuff, galid))
        
    if os.path.isfile(ellipsefitfile):
        data = Table(fitsio.read(ellipsefitfile))

        # Optionally convert (back!) into a dictionary.
        if asTable:
            return data
        ellipsefit = {}
        for key in data.colnames:
            val = data[key][0]
            #val = data[key].tolist()[0]
            #if np.logical_not(np.isscalar(val)) and len(val) > 0:
            #    val = np.array(val, dtype=data[key].dtype)
            ellipsefit[key.lower()] = val # lowercase!
            #ellipsefit[key.lower()] = np.array(val, dtype=data[key].dtype)
    else:
        if verbose:
            print('Ellipse fit file {} not found!'.format(ellipsefitfile))
        if asTable:
            ellipsefit = Table()
        else:
            ellipsefit = dict()

    return ellipsefit

def get_mask_filename(galaxy, galaxydir, filesuffix='', galaxy_id=''):
    """Return the name of the source-mask file written by write_mask (the naming
    convention mirrors get_ellipsefit_filename).

    """
    if type(galaxy_id) is not str:
        galaxy_id = str(galaxy_id)

    if galaxy_id.strip() == '':
        galid = ''
    else:
        galid = '-{}'.format(galaxy_id)

    if filesuffix.strip() == '':
        fsuff = ''
    else:
        fsuff = '-{}'.format(filesuffix)

    maskfile = os.path.join(galaxydir, '{}{}{}-mask.fits.fz'.format(galaxy, fsuff, galid))

    return maskfile

# Header keywords (and keyword prefixes) copied from the coadd image headers
# into the source-mask file so that each mask carries its own WCS.
_MASK_WCS_KEYS = ('WCSAXES', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                  'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2',
                  'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'LONPOLE', 'LATPOLE',
                  'EQUINOX', 'RADESYS', 'RADECSYS', 'MJD_MEAN', 'FILTER', 'MAGZERO')
_MASK_WCS_PREFIXES = ('PV1_', 'PV2_')

def _header_records(hdr):
    """Iterate over the (name, value, comment) records of a fitsio.FITSHDR or an
    astropy.io.fits.Header object.

    """
    if hdr is None:
        return
    if hasattr(hdr, 'records'): # fitsio.FITSHDR
        for rec in hdr.records():
            yield rec['name'], rec['value'], rec.get('comment', '')
    elif hasattr(hdr, 'cards'): # astropy.io.fits.Header
        for card in hdr.cards:
            yield card.keyword, card.value, card.comment
    else: # dictionary
        for key in hdr.keys():
            yield key, hdr[key], ''

def _add_header_record(hdr, name, value, comment=''):
    """Add a keyword to a fitsio.FITSHDR object, converting numpy scalars to
    native Python types.

    """
    if isinstance(value, np.generic):
        value = value.item()
    hdr.add_record({'name': name, 'value': value, 'comment': comment})

def write_mask(galaxy, galaxydir, data, igal=0, galaxy_id='', filesuffix='',
               galaxyinfo=None, verbose=False):
    """Write out the multiband source mask built by the project-specific
    read_multiband method (e.g., legacyhalos.hsc.read_multiband and
    legacyhalos.hsc._build_multiband_mask) for the galaxy of interest.

    The output is a multi-extension, fpack-compressed FITS file (see
    get_mask_filename) with one uint8 image extension per band (EXTNAME is the
    upper-case band name) in which 1 = masked (pixels excluded from
    ellipse-fitting: bright stars and the other maskbits, pixels with no
    coverage, the image border, significant model residuals, and all the
    satellite sources) and 0 = unmasked. The pixels of the galaxy of interest
    are never masked. The moment-derived geometry of the galaxy is written to
    the primary header and the WCS of each band is copied from the coadd image
    headers.

    data - dictionary returned by the project-specific read_multiband method
    igal - index of the galaxy of interest (in data['galaxy_indx'])
    galaxy_id - add a unique ID number to the output filename
    galaxyinfo - optional dictionary of (value, comment) tuples to add to the
      primary header (see, e.g., legacyhalos.hsc.read_multiband)

    """
    maskfile = get_mask_filename(galaxy, galaxydir, filesuffix=filesuffix, galaxy_id=galaxy_id)

    bands, refband = data['bands'], data['refband']
    refpixscale = float(data['refpixscale'])

    # Primary header: code versions, galaxy identity, and geometry.
    hdr0 = fitsio.FITSHDR()
    for name, value, comment in _header_records(legacyhalos_header()):
        _add_header_record(hdr0, name, value, comment)

    _add_header_record(hdr0, 'GALAXY', str(galaxy), 'galaxy name')
    _add_header_record(hdr0, 'GALAXYID', str(galaxy_id), 'galaxy ID')
    _add_header_record(hdr0, 'FILESUFF', str(filesuffix), 'file suffix')
    _add_header_record(hdr0, 'BANDS', ','.join(bands), 'bands (one image extension per band)')
    _add_header_record(hdr0, 'REFBAND', refband, 'reference band')
    _add_header_record(hdr0, 'PIXSCALE', refpixscale, 'reference-band pixel scale [arcsec/pixel]')
    _add_header_record(hdr0, 'MASKVAL', 1, 'value of masked (excluded) pixels')

    if 'mge' in data.keys():
        mge = data['mge'][igal]
        # Note that (x,y) are switched between MGE and photutils (see
        # legacyhalos.ellipse.ellipsefit_multiband).
        for key, name, comment in (
                ('ra', 'RA', 'Tractor right ascension [deg]'),
                ('dec', 'DEC', 'Tractor declination [deg]'),
                ('bx', 'BX', 'Tractor x position [pixel]'),
                ('by', 'BY', 'Tractor y position [pixel]'),
                ('ra_moment', 'RA_MOM', 'moment-derived right ascension [deg]'),
                ('dec_moment', 'DEC_MOM', 'moment-derived declination [deg]'),
                ('ymed', 'X0_MOM', 'moment-derived x position [pixel]'),
                ('xmed', 'Y0_MOM', 'moment-derived y position [pixel]'),
                ('ypeak', 'XPEAK', 'x position of the peak [pixel]'),
                ('xpeak', 'YPEAK', 'y position of the peak [pixel]'),
                ('pa', 'PA_MOM', 'moment-derived position angle [deg]'),
                ('eps', 'EPS_MOM', 'moment-derived ellipticity (1-b/a)'),
                ('majoraxis', 'MAJAXIS', 'moment-derived semi-major axis [pixel]'),
                ('largeshift', 'LRGSHIFT', 'large centroid shift; Tractor geometry used'),
                ('small_tractor', 'SMALLTRC', 'Tractor half-light radius < 1 arcsec')):
            if key in mge.keys():
                _add_header_record(hdr0, name, mge[key], comment)
        if 'majoraxis' in mge.keys():
            _add_header_record(hdr0, 'SMA_MOM', float(mge['majoraxis']) * refpixscale,
                               'moment-derived semi-major axis [arcsec]')
        if 'eps' in mge.keys():
            _add_header_record(hdr0, 'BA_MOM', 1.0 - float(mge['eps']),
                               'moment-derived minor-to-major axis ratio')

    if galaxyinfo:
        for key in galaxyinfo.keys():
            value, comment = galaxyinfo[key]
            _add_header_record(hdr0, key.upper(), value, str(comment))

    if verbose:
        print('Writing {}'.format(maskfile))
    tmpfile = maskfile + '.tmp'
    with fitsio.FITS(tmpfile, 'rw', clobber=True) as ff:
        ff.write(None, header=hdr0)
        for filt in bands:
            img = data['{}_masked'.format(filt.lower())][igal]
            mask = ma.getmaskarray(img).astype(np.uint8) # 1 = masked

            hdr = fitsio.FITSHDR()
            # Copy the WCS from the coadd header, if available.
            for name, value, comment in _header_records(data.get('{}_header'.format(filt.lower()))):
                if name in _MASK_WCS_KEYS or name.startswith(_MASK_WCS_PREFIXES):
                    _add_header_record(hdr, name, value, comment)
            if 'filt2pixscale' in data.keys():
                pixscale = float(data['filt2pixscale'][filt])
            else:
                pixscale = refpixscale
            _add_header_record(hdr, 'GALAXY', str(galaxy), 'galaxy name')
            _add_header_record(hdr, 'GALAXYID', str(galaxy_id), 'galaxy ID')
            _add_header_record(hdr, 'BAND', filt, 'band')
            _add_header_record(hdr, 'PIXSCALE', pixscale, 'pixel scale [arcsec/pixel]')
            _add_header_record(hdr, 'MASKVAL', 1, 'value of masked (excluded) pixels')

            ff.write(mask, extname=filt.upper(), header=hdr, compress='rice')

    os.rename(tmpfile, maskfile)

    return maskfile

def read_mask(galaxy, galaxydir, filesuffix='', galaxy_id='', verbose=True, maskfile=None):
    """Read the output of write_mask.

    Returns a dictionary with one boolean array per band (True = masked) plus
    the list of bands ('bands'), the primary header ('header'), and the header
    of each image extension ('{band}_header').

    """
    if maskfile is None:
        maskfile = get_mask_filename(galaxy, galaxydir, filesuffix=filesuffix, galaxy_id=galaxy_id)

    if not os.path.isfile(maskfile):
        if verbose:
            print('File {} not found!'.format(maskfile))
        return dict()

    mask = dict()
    with fitsio.FITS(maskfile) as ff:
        hdr0 = ff[0].read_header()
        bands = [band.strip() for band in hdr0['BANDS'].split(',')]
        mask['bands'] = bands
        mask['header'] = hdr0
        for band in bands:
            mask[band] = ff[band.upper()].read().astype(bool)
            mask['{}_header'.format(band)] = ff[band.upper()].read_header()

    return mask

def write_sersic(galaxy, galaxydir, sersic, modeltype='single', verbose=False):
    """Pickle a dictionary of photutils.isophote.isophote.IsophoteList objects (see,
    e.g., ellipse.fit_multiband).

    """
    sersicfile = os.path.join(galaxydir, '{}-sersic-{}.p'.format(galaxy, modeltype))
    if verbose:
        print('Writing {}'.format(sersicfile))
    with open(sersicfile, 'wb') as ell:
        pickle.dump(sersic, ell, protocol=2)

def read_sersic(galaxy, galaxydir, modeltype='single', verbose=True):
    """Read the output of write_sersic."""

    sersicfile = os.path.join(galaxydir, '{}-sersic-{}.p'.format(galaxy, modeltype))
    try:
        with open(sersicfile, 'rb') as ell:
            sersic = pickle.load(ell)
    except:
        #raise IOError
        if verbose:
            print('File {} not found!'.format(sersicfile))
        sersic = dict()

    return sersic

def write_sbprofile(sbprofile, smascale, sbfile):
    """Write a (previously derived) surface brightness profile as a simple ASCII
    file, for use on a webpage.

    """
    data = np.array( [
        sbprofile['sma'],
        sbprofile['sma'] * smascale,
        sbprofile['mu_g'],
        sbprofile['mu_r'],
        sbprofile['mu_z'],
        sbprofile['mu_g_err'],
        sbprofile['mu_r_err'],
        sbprofile['mu_z_err']
        ] ).T

    fixnan = np.isnan(data)
    if np.sum(fixnan) > 0:
        data[fixnan] = -999
        
    np.savetxt(sbfile, data, fmt='%.6f')
    #with open(sbfile, 'wb') as sb:
    #    sb.write('# Yo\n')
    #pdb.set_trace()

    print('Wrote {}'.format(sbfile))

def _get_psfsize_and_depth(tractor, bands, pixscale, incenter=False):
    """Support function for read_multiband. Compute the average PSF size (in arcsec)
    and depth (in 5-sigma AB mags) in each bandpass based on the Tractor
    catalog.

    """
    out = {}

    # Optionally choose sources in the center of the field.
    H = np.max(tractor.bx) - np.min(tractor.bx)
    W = np.max(tractor.by) - np.min(tractor.by)
    if incenter:
        dH = 0.1 * H
        these = np.where((tractor.bx >= int(H / 2 - dH)) * (tractor.bx <= int(H / 2 + dH)) *
                         (tractor.by >= int(H / 2 - dH)) * (tractor.by <= int(H / 2 + dH)))[0]
    else:
        #these = np.where(tractor.get(psfdepthcol) > 0)[0]
        these = np.arange(len(tractor))
    
    # Get the average PSF size and depth in each bandpass.
    for filt in bands:
        psfsizecol = 'psfsize_{}'.format(filt.lower())
        psfdepthcol = 'psfdepth_{}'.format(filt.lower())
        if psfsizecol in tractor.columns():
            good = np.where(tractor.get(psfsizecol)[these] > 0)[0]
            if len(good) == 0:
                print('  No good measurements of the PSF size in band {}!'.format(filt))
                out['psfsigma_{}'.format(filt.lower())] = np.float32(0.0)
                out['psfsize_{}'.format(filt.lower())] = np.float32(0.0)
            else:
                # Get the PSF size and image depth.
                psfsize = tractor.get(psfsizecol)[these][good]   # [FWHM, arcsec]
                psfsigma = psfsize / np.sqrt(8 * np.log(2)) / pixscale # [sigma, pixels]

                out['psfsigma_{}'.format(filt.lower())] = np.median(psfsigma).astype('f4') 
                out['psfsize_{}'.format(filt.lower())] = np.median(psfsize).astype('f4') 
            
        if psfsizecol in tractor.columns():
            good = np.where(tractor.get(psfdepthcol)[these] > 0)[0]
            if len(good) == 0:
                print('  No good measurements of the PSF depth in band {}!'.format(filt))
                out['psfdepth_{}'.format(filt.lower())] = np.float32(0.0)
            else:
                psfdepth = tractor.get(psfdepthcol)[these][good] # [AB mag, 5-sigma]
                out['psfdepth_{}'.format(filt.lower())] = (22.5-2.5*np.log10(1/np.sqrt(np.median(psfdepth)))).astype('f4') 
        
    return out

def _read_image_data(data, filt2imfile, starmask=None, allmask=None,
                     fill_value=0.0, filt2pixscale=None, verbose=False):
    """Helper function for the project-specific read_multiband method.

    Read the multi-band images and inverse variance images and pack them into a
    dictionary. Also create an initial pixel-level mask and handle images with
    different pixel scales (e.g., GALEX and WISE images).

    """
    from astropy.stats import sigma_clipped_stats
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.filters import gaussian_filter
    from skimage.transform import resize

    from tractor.psf import PixelizedPSF
    from tractor.tractortime import TAITime
    from astrometry.util.util import Tan
    from legacypipe.survey import LegacySurveyWcs, ConstantFitsWcs

    bands, refband = data['bands'], data['refband']

    #refhdr = fitsio.read_header(filt2imfile[refband]['image'], ext=1)
    #refsz = (refhdr['NAXIS1'], refhdr['NAXIS2'])

    vega2ab = {'W1': 2.699, 'W2': 3.339, 'W3': 5.174, 'W4': 6.620}

    # Loop on each filter and return the masked data.
    residual_mask = None
    for filt in bands:
        # Read the data and initialize the mask with the inverse variance image,
        # if available.
        if verbose:
            print('Reading {}'.format(filt2imfile[filt]['image']))
            print('Reading {}'.format(filt2imfile[filt]['model']))
        image = fitsio.read(filt2imfile[filt]['image'])
        hdr = fitsio.read_header(filt2imfile[filt]['image'], ext=1)
        model = fitsio.read(filt2imfile[filt]['model'])

        # add the header to the data dictionary
        data['{}_header'.format(filt.lower())] = hdr

        # Initialize the mask based on the inverse variance
        if 'invvar' in filt2imfile[filt].keys():
            if verbose:
                print('Reading {}'.format(filt2imfile[filt]['invvar']))
            invvar = fitsio.read(filt2imfile[filt]['invvar'])
            mask = invvar <= 0 # True-->bad, False-->good
        else:
            invvar = None
            mask = np.zeros_like(image).astype(bool)

        # convert WISE images from Vega nanomaggies to AB nanomaggies
        # https://www.legacysurvey.org/dr9/description/#photometry
        if filt.lower() == 'w1' or filt.lower() == 'w2' or filt.lower() == 'w3' or filt.lower() == 'w4':
            image *= 10**(-0.4*vega2ab[filt])
            model *= 10**(-0.4*vega2ab[filt])
            if invvar is not None:
                invvar /= (10**(-0.4*vega2ab[filt]))**2
            
        sz = image.shape

        # GALEX, unWISE need to be resized. Never resize allmask, if present.
        if starmask is not None:
            if starmask.shape == sz:
                doresize = False
            else:
                doresize = True

        # Retrieve the PSF and WCS.
        if filt == refband:
            HH, WW = sz
            data['refband_width'] = WW
            data['refband_height'] = HH
            
        if verbose:
            print('Reading {}'.format(filt2imfile[filt]['psf']))
        psfimg = fitsio.read(filt2imfile[filt]['psf'])
        psfimg /= psfimg.sum()
        data['{}_psf'.format(filt.lower())] = PixelizedPSF(psfimg)

        wcs = Tan(filt2imfile[filt]['image'], 1)
        if 'MJD_MEAN' in hdr:
            mjd_tai = hdr['MJD_MEAN'] # [TAI]
            wcs = LegacySurveyWcs(wcs, TAITime(None, mjd=mjd_tai))
        else:
            wcs = ConstantFitsWcs(wcs)
        data['{}_wcs'.format(filt.lower())] = wcs

        # Add in the starmask, resizing if necessary for this image/pixel
        # scale. Never resize allmask (it's only for the optical).
        if starmask is not None:
            if doresize:
                _starmask = resize(starmask, mask.shape, mode='edge', anti_aliasing=False) > 0
                mask = np.logical_or(mask, _starmask)
            else:
                mask = np.logical_or(mask, starmask)
                if allmask is not None:
                    mask = np.logical_or(mask, allmask)

        # Flag significant residual pixels after subtracting *all* the models
        # (we will restore the pixels of the galaxies of interest later). Only
        # consider the optical (grz) bands here.
        resid = gaussian_filter(image - model, 2.0)
        _, _, sig = sigma_clipped_stats(resid, sigma=3.0)
        data['{}_sigma'.format(filt.lower())] = sig
        if residual_mask is None:
            residual_mask = np.abs(resid) > 5*sig
        else:
            _residual_mask = np.abs(resid) > 5*sig
            # In grz, use a cumulative residual mask. In UV/IR use an
            # individual-band mask.
            if doresize:
                pass
                #residual_mask = resize(_residual_mask, residual_mask.shape, mode='reflect')
            else:
                residual_mask = np.logical_or(residual_mask, _residual_mask)

        ## Dilate the mask, mask out a 10% border, and pack into a dictionary.
        mask = binary_dilation(mask, iterations=2)
        edge = int(0.02*sz[0])
        mask[:edge, :] = True
        mask[:, :edge] = True
        mask[:, sz[0]-edge:] = True
        mask[sz[0]-edge:, :] = True
        data[filt] = ma.masked_array(image, mask) # [nanomaggies]
        ma.set_fill_value(data[filt], fill_value)

        #if filt == 'W1':
        #    import matplotlib.pyplot as plt
        #    plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-mask-{}.png'.format(filt))
        #    pdb.set_trace()
        
        if invvar is not None:
            var = np.zeros_like(invvar)
            ok = invvar > 0
            var[ok] = 1 / invvar[ok]
            data['{}_var_'.format(filt.lower())] = var # [nanomaggies**2]
            #data['{}_var'.format(filt.lower())] = var / thispixscale**4 # [nanomaggies**2/arcsec**4]
            if np.any(invvar < 0):
                print('Warning! Negative pixels in the {}-band inverse variance map!'.format(filt))
                #pdb.set_trace()

    data['residual_mask'] = residual_mask
    if starmask is not None:
        data['starmask'] = starmask
    if allmask is not None:
        data['almask'] = allmask

    return data
