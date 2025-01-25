"""
legacyhalos.ellipse
===================

Code to do ellipse fitting on the residual coadds.
"""
import os, pdb
import time, warnings
import numpy as np
#import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import astropy.modeling

from photutils.isophote import (EllipseGeometry, Ellipse, EllipseSample,
                                Isophote, IsophoteList)
from photutils.isophote.sample import CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter

import legacyhalos.io

REF_SBTHRESH = [22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26] # surface brightness thresholds

def _get_r0():
    r0 = 10.0 # [arcsec]
    return r0

def cog_model(radius, mtot, m0, alpha1, alpha2):
    r0 = _get_r0()
    #return mtot - m0 * np.expm1(-alpha1*((radius / r0)**(-alpha2)))
    return mtot + m0 * np.log1p(alpha1*(radius/10.0)**(-alpha2))

def cog_dofit(sma, mag, mag_err, bounds=None):
    try:
        popt, _ = curve_fit(cog_model, sma, mag, sigma=mag_err,
                            bounds=bounds, max_nfev=10000)
    except RuntimeError:
        popt = None
        chisq = 1e6
    else:
        chisq = (((cog_model(sma, *popt) - mag) / mag_err) ** 2).sum()
        
    return popt, chisq

class CogModel(astropy.modeling.Fittable1DModel):
    """Class to empirically model the curve of growth.

    radius in arcsec
    r0 - constant scale factor (10)

    m(r) = mtot + mcen * (1-exp**(-alpha1*(radius/r0)**(-alpha2))
    """
    mtot = astropy.modeling.Parameter(default=20.0, bounds=(1, 30)) # integrated magnitude (r-->infty)
    m0 = astropy.modeling.Parameter(default=10.0, bounds=(1, 30)) # central magnitude (r=0)
    alpha1 = astropy.modeling.Parameter(default=0.3, bounds=(1e-3, 5)) # scale factor 1
    alpha2 = astropy.modeling.Parameter(default=0.5, bounds=(1e-3, 5)) # scale factor 2

    def __init__(self, mtot=mtot.default, m0=m0.default,
                 alpha1=alpha1.default, alpha2=alpha2.default):
        super(CogModel, self).__init__(mtot, m0, alpha1, alpha2)

        self.r0 = 10 # scale factor [arcsec]
        
    def evaluate(self, radius, mtot, m0, alpha1, alpha2):
        """Evaluate the COG model."""
        model = mtot + m0 * (1 - np.exp(-alpha1*(radius/self.r0)**(-alpha2)))
        return model
   
def _apphot_one(args):
    """Wrapper function for the multiprocessing."""
    return apphot_one(*args)

def apphot_one(img, mask, theta, x0, y0, aa, bb, pixscale, variance=False, iscircle=False):
    """Perform aperture photometry in a single elliptical annulus.

    """
    from photutils import EllipticalAperture, CircularAperture, aperture_photometry

    if iscircle:
        aperture = CircularAperture((x0, y0), aa)
    else:
        aperture = EllipticalAperture((x0, y0), aa, bb, theta)
        
    # Integrate the data to get the total surface brightness (in
    # nanomaggies/arcsec2) and the mask to get the fractional area.
    
    #area = (aperture_photometry(~mask*1, aperture, mask=mask, method='exact'))['aperture_sum'].data * pixscale**2 # [arcsec**2]
    mu_flux = (aperture_photometry(img, aperture, mask=mask, method='exact'))['aperture_sum'].data # [nanomaggies/arcsec2]
    if variance:
        apphot = np.sqrt(mu_flux) * pixscale**2 # [nanomaggies]
    else:
        apphot = mu_flux * pixscale**2 # [nanomaggies]
    return apphot

def ellipse_cog(bands, data, refellipsefit, pixscalefactor,
                pixscale, igal=0, pool=None, seed=1,
                sbthresh=REF_SBTHRESH):
    """Measure the curve of growth (CoG) by performing elliptical aperture
    photometry.

    maxsma in pixels
    pixscalefactor - assumed to be constant for all bandpasses!

    """
    import numpy.ma as ma
    import astropy.table
    from astropy.utils.exceptions import AstropyUserWarning
    from scipy import integrate
    from scipy.interpolate import interp1d

    rand = np.random.RandomState(seed)
    
    deltaa = 1.0 # pixel spacing

    #theta, eps = refellipsefit['geometry'].pa, refellipsefit['geometry'].eps
    theta, eps = np.radians(refellipsefit['pa']-90), refellipsefit['eps']
    refband = refellipsefit['refband']
    #maxsma = refellipsefit['maxsma']

    results = {}

    # Build the SB profile and measure the radius (in arcsec) at which mu
    # crosses a few different thresholds like 25 mag/arcsec, etc.
    sbprofile = ellipse_sbprofile(refellipsefit)

    #print('We should measure these radii from the extinction-corrected photometry!')
    for sbcut in sbthresh:
        if sbprofile['mu_r'].max() < sbcut or sbprofile['mu_r'].min() > sbcut:
            print('Insufficient profile to measure the radius at {:.1f} mag/arcsec2!'.format(sbcut))
            results['radius_sb{:0g}'.format(sbcut)] = np.float32(-1.0)
            results['radius_sb{:0g}_err'.format(sbcut)] = np.float32(-1.0)
            continue

        rr = (sbprofile['sma_r'] * pixscale)**0.25 # [arcsec]
        sb = sbprofile['mu_r'] - sbcut
        sberr = sbprofile['muerr_r']
        keep = np.where((sb > -1) * (sb < 1))[0]
        if len(keep) < 5:
            keep = np.where((sb > -2) * (sb < 2))[0]
            if len(keep) < 5:
                print('Insufficient profile to measure the radius at {:.1f} mag/arcsec2!'.format(sbcut))
                results['radius_sb{:0g}'.format(sbcut)] = np.float32(-1.0)
                results['radius_sb{:0g}_err'.format(sbcut)] = np.float32(-1.0)
                continue

        # Monte Carlo to get the radius
        rcut = []
        for ii in np.arange(20):
            sbfit = rand.normal(sb[keep], sberr[keep])
            coeff = np.polyfit(sbfit, rr[keep], 1)
            rcut.append((np.polyval(coeff, 0))**4)
        meanrcut, sigrcut = np.mean(rcut), np.std(rcut)
        #print(rcut, meanrcut, sigrcut)

        #plt.clf() ; plt.plot((rr[keep])**4, sb[keep]) ; plt.axvline(x=meanrcut) ; plt.savefig('junk.png')
        #plt.clf() ; plt.plot(rr, sb+sbcut) ; plt.axvline(x=meanrcut**0.25) ; plt.axhline(y=sbcut) ; plt.xlim(2, 2.6) ; plt.savefig('junk.png')
        #pdb.set_trace()
            
        #try:
        #    rcut = interp1d()(sbcut) # [arcsec]
        #except:
        #    print('Warning: extrapolating r({:0g})!'.format(sbcut))
        #    rcut = interp1d(sbprofile['mu_r'], sbprofile['sma_r'] * pixscale, fill_value='extrapolate')(sbcut) # [arcsec]
        results['radius_sb{:0g}'.format(sbcut)] = np.float32(meanrcut)
        results['radius_sb{:0g}_err'.format(sbcut)] = np.float32(sigrcut)

    for filt in bands:
        img = ma.getdata(data['{}_masked'.format(filt)][igal]) # [nanomaggies/arcsec2]
        mask = ma.getmask(data['{}_masked'.format(filt)][igal])

        deltaa_filt = deltaa * pixscalefactor

        if filt in refellipsefit['bands']:
            if len(sbprofile['sma_{}'.format(filt)]) == 0: # can happen with partial coverage in other bands
                maxsma = sbprofile['sma_{}'.format(refband)].max()
            else:
                maxsma = sbprofile['sma_{}'.format(filt)].max()        # [pixels]
            #minsma = 3 * refellipsefit['psfsigma_{}'.format(filt)] # [pixels]
        else:
            maxsma = sbprofile['sma_{}'.format(refband)].max()        # [pixels]
            #minsma = 3 * refellipsefit['psfsigma_{}'.format(refband)] # [pixels]
            
        sma = np.arange(deltaa_filt, maxsma * pixscalefactor, deltaa_filt)
        smb = sma * eps
        if eps == 0:
            iscircle = True
        else:
            iscircle = False
        
        x0 = refellipsefit['x0'] * pixscalefactor
        y0 = refellipsefit['y0'] * pixscalefactor

        #im = np.log10(img) ; im[mask] = 0 ; plt.clf() ; plt.imshow(im, origin='lower') ; plt.scatter(y0, x0, s=50, color='red') ; plt.savefig('junk.png')
        #pdb.set_trace()

        with np.errstate(all='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                cogflux = pool.map(_apphot_one, [(img, mask, theta, x0, y0, aa, bb, pixscale, False, iscircle)
                                                for aa, bb in zip(sma, smb)])
                if len(cogflux) > 0:
                    cogflux = np.hstack(cogflux)
                else:
                    cogflux = np.array([0.0])

                if '{}_var'.format(filt) in data.keys():
                    var = data['{}_var'.format(filt)][igal] # [nanomaggies**2/arcsec**4]
                    cogferr = pool.map(_apphot_one, [(var, mask, theta, x0, y0, aa, bb, pixscale, True, iscircle)
                                                    for aa, bb in zip(sma, smb)])
                    if len(cogferr) > 0:
                        cogferr = np.hstack(cogferr)
                    else:
                        cogferr = np.array([0.0])
                else:
                    cogferr = None

        # Aperture fluxes can be negative (or nan?) sometimes--
        with warnings.catch_warnings():
            if cogferr is not None:
                ok = (cogflux > 0) * (cogferr > 0) * np.isfinite(cogflux) * np.isfinite(cogferr)
            else:
                ok = (cogflux > 0) * np.isfinite(cogflux)
                cogmagerr = np.ones(len(cogmag))

        #results['cog_smaunit'] = 'arcsec'
        
        if np.count_nonzero(ok) < 10:
            print('Warning: Too few {}-band pixels to fit the curve of growth; skipping.'.format(filt))
            results['{}_cog_sma'.format(filt)] = np.float32(-1) # np.array([])
            results['{}_cog_mag'.format(filt)] = np.float32(-1) # np.array([])
            results['{}_cog_magerr'.format(filt)] = np.float32(-1) # np.array([])
            # old data model
            #results['{}_cog_params'.format(filt)] = {'mtot': np.float32(-1),
            #                                          'm0': np.float32(-1),
            #                                          'alpha1': np.float32(-1),
            #                                          'alpha2': np.float32(-1),
            #                                          'chi2': np.float32(1e6)}
            results['{}_cog_params_mtot'.format(filt)] = np.float32(-1)
            results['{}_cog_params_m0'.format(filt)] = np.float32(-1)
            results['{}_cog_params_alpha1'.format(filt)] = np.float32(-1)
            results['{}_cog_params_alpha2'.format(filt)] = np.float32(-1)
            results['{}_cog_params_chi2'.format(filt)] = np.float32(1e6)
            results['{}_cog_r50'.format(filt)] = np.float32(-1)
            for sbcut in sbthresh:
                results['{}_mag_sb{:0g}'.format(filt, sbcut)] = np.float32(-1)
                results['{}_mag_sb{:0g}_err'.format(filt, sbcut)] = np.float32(-1)
            continue

        sma_arcsec = sma[ok] * pixscale             # [arcsec]
        cogmag = 22.5 - 2.5 * np.log10(cogflux[ok]) # [mag]
        if cogferr is not None:
            cogmagerr = 2.5 * cogferr[ok] / cogflux[ok] / np.log(10)

        results['{}_cog_sma'.format(filt)] = np.float32(sma_arcsec)
        results['{}_cog_mag'.format(filt)] = np.float32(cogmag)
        results['{}_cog_magerr'.format(filt)] = np.float32(cogmagerr)

        #print('Modeling the curve of growth.')
        n_params = 4
        popt = chisq = popt_simple = chisq_simple = None
        if len(sma_arcsec) >= n_params:
            bounds = ([cogmag[-1]-1.0, 0, 0, 0], np.inf)
            #bounds = ([cogmag[-1]-0.5, 2.5, 0, 0], np.inf)
            #bounds = (0, np.inf)
            (mtot, m0, alpha1, alpha2), minchi2 = cog_dofit(sma_arcsec, cogmag, cogmagerr, bounds=bounds)
            if minchi2 < 1e6:
                print('{} CoG modeling succeeded with a chi^2 minimum of {:.2f}'.format(filt, minchi2))
                results['{}_cog_params_mtot'.format(filt)] = np.float32(mtot)
                results['{}_cog_params_m0'.format(filt)] = np.float32(m0)
                results['{}_cog_params_alpha1'.format(filt)] = np.float32(alpha1)
                results['{}_cog_params_alpha2'.format(filt)] = np.float32(alpha2)
                results['{}_cog_params_chi2'.format(filt)] = np.float32(minchi2)

                if (m0 != 0) * (alpha1 != 0.0) * (alpha2 != 0.0):
                    #half_light_sma = (- np.log(1.0 - np.log10(2.0) * 2.5 / m0) / alpha1)**(-1.0/alpha2) * _get_r0() # [arcsec]
                    half_light_sma = ((np.expm1(np.log10(2.0)*2.5/m0)) / alpha1)**(-1.0 / alpha2) * _get_r0() # [arcsec]
                    #half_light_radius = half_light_sma * np.sqrt(1 - ellipse['eps']) # circularized
                    results['{}_cog_r50'.format(filt)] = half_light_sma
        
        ##################################################
        # begin old COG modeling code
        
        #def get_chi2(bestfit):
        #    sbmodel = bestfit(self.radius, self.wave)
        #    chi2 = np.sum( (self.sb - sbmodel)**2 / self.sberr**2 ) / dof
        #
        #cogfitter = astropy.modeling.fitting.LevMarLSQFitter()
        #cogmodel = CogModel()
        #
        #nball = 10
        #
        ## perturb the parameter values
        #nparams = len(cogmodel.parameters)
        #dof = len(cogmag) - nparams
        #
        #params = np.repeat(cogmodel.parameters, nball).reshape(nparams, nball)
        #for ii, pp in enumerate(cogmodel.param_names):
        #    pinfo = getattr(cogmodel, pp)
        #    if pinfo.bounds[0] is not None:
        #        scale = 0.2 * pinfo.default
        #        params[ii, :] += rand.normal(scale=scale, size=nball)
        #        toosmall = np.where( params[ii, :] < pinfo.bounds[0] )[0]
        #        if len(toosmall) > 0:
        #            params[ii, toosmall] = pinfo.default
        #        toobig = np.where( params[ii, :] > pinfo.bounds[1] )[0]
        #        if len(toobig) > 0:
        #            params[ii, toobig] = pinfo.default
        #    else:
        #        params[ii, :] += rand.normal(scale=0.2 * pinfo.default, size=nball)
        #        
        ## perform the fit nball times
        #chi2fail = 1e6
        #with warnings.catch_warnings():
        #    warnings.simplefilter('ignore')
        #    chi2 = np.zeros(nball) + chi2fail
        #    for jj in range(nball):
        #        cogmodel.parameters = params[:, jj]
        #        # Fit up until the curve of growth turns over, but no less than
        #        # the second moment of the light distribution! Pretty fragile..
        #        these = np.where(np.diff(cogmag) < 0)[0]
        #        if len(these) > 5: # this is bad if we don't have at least 5 points!
        #            if sma_arcsec[these[0]] < (refellipsefit['majoraxis'] * pixscale * pixscalefactor):
        #                these = np.where(sma_arcsec < refellipsefit['majoraxis'] * pixscale * pixscalefactor)[0]
        #            if len(these) > 5: # can happen in corner cases (e.g., PGC155984)
        #                ballfit = cogfitter(cogmodel, sma_arcsec[these], cogmag[these],
        #                                    maxiter=100, weights=1/cogmagerr[these])
        #                bestfit = ballfit(sma_arcsec)
        #
        #                chi2[jj] = np.sum( (cogmag - bestfit)**2 / cogmagerr**2 ) / dof
        #                if cogfitter.fit_info['param_cov'] is None: # failed
        #                    if False:
        #                        print(jj, cogfitter.fit_info['message'], chi2[jj])
        #                else:
        #                    params[:, jj] = ballfit.parameters # update
        #
        ## if at least one fit succeeded, re-evaluate the model at the chi2
        ## minimum.
        #good = chi2 < chi2fail
        #if np.sum(good) == 0:
        #    print('{} CoG modeling failed.'.format(filt))
        #    #result.update({'fit_message': cogfitter.fit_info['message']})
        #    #return result
        #mindx = np.argmin(chi2)
        #minchi2 = chi2[mindx]
        #cogmodel.parameters = params[:, mindx]
        #P = cogfitter(cogmodel, sma_arcsec, cogmag, weights=1/cogmagerr, maxiter=100)
        #print('{} CoG modeling succeeded with a chi^2 minimum of {:.2f}'.format(filt, minchi2))
        #
        ##P = cogfitter(cogmodel, sma_arcsec, cogmag, weights=1/cogmagerr)
        ##results['{}_cog_params'.format(filt)] = {'mtot': np.float32(P.mtot.value),
        ##                                         'm0': np.float32(P.m0.value),
        ##                                         'alpha1': np.float32(P.alpha1.value),
        ##                                         'alpha2': np.float32(P.alpha2.value),
        ##                                         'chi2': np.float32(minchi2)}
        #results['{}_cog_params_mtot'.format(filt)] = np.float32(P.mtot.value)
        #results['{}_cog_params_m0'.format(filt)] = np.float32(P.m0.value)
        #results['{}_cog_params_alpha1'.format(filt)] = np.float32(P.alpha1.value)
        #results['{}_cog_params_alpha2'.format(filt)] = np.float32(P.alpha2.value)
        #results['{}_cog_params_chi2'.format(filt)] = np.float32(minchi2)
        #
        # end old COG modeling code
        ##################################################

        #print('Measuring integrated magnitudes to different radii.')
        sb = ellipse_sbprofile(refellipsefit, linear=True)
        radkeys = ['radius_sb{:0g}'.format(sbcut) for sbcut in sbthresh]
        for radkey in radkeys:
            magkey = radkey.replace('radius_', '{}_mag_'.format(filt))
            magerrkey = '{}_err'.format(magkey)
            
            smamax = results[radkey] # semi-major axis
            if smamax > 0 and smamax < np.max(sma_arcsec):
                rmax = smamax * np.sqrt(1 - refellipsefit['eps']) # [circularized radius, arcsec]

                rr = sb['radius_{}'.format(filt)]    # [circularized radius, arcsec]
                yy = sb['mu_{}'.format(filt)]        # [surface brightness, nanomaggies/arcsec**2]
                yyerr = sb['muerr_{}'.format(filt)] # [surface brightness, nanomaggies/arcsec**2]
                try:
                    #print(filt, rr.max(), rmax)
                    yy_rmax = interp1d(rr, yy)(rmax) # can fail if rmax < np.min(sma_arcsec)
                    yyerr_rmax = interp1d(rr, yyerr)(rmax)

                    # append the maximum radius to the end of the array
                    keep = np.where(rr < rmax)[0]
                    _rr = np.hstack((rr[keep], rmax))
                    _yy = np.hstack((yy[keep], yy_rmax))
                    _yyerr = np.hstack((yyerr[keep], yyerr_rmax))

                    flux = 2 * np.pi * integrate.simps(x=_rr, y=_rr*_yy)
                    fvar = integrate.simps(x=_rr, y=_rr*_yyerr**2)
                    if flux > 0 and fvar > 0:
                        ferr = 2 * np.pi * np.sqrt(fvar)
                        results[magkey] = np.float32(22.5 - 2.5 * np.log10(flux))
                        results[magerrkey] = np.float32(2.5 * ferr / flux / np.log(10))
                    else:
                        results[magkey] = np.float32(-1.0)
                        results[magerrkey] = np.float32(-1.0)
                except:
                    results[magkey] = np.float32(-1.0)
                    results[magerrkey] = np.float32(-1.0)
            else:
                results[magkey] = np.float32(-1.0)
                results[magerrkey] = np.float32(-1.0)
                
    #pdb.set_trace()
        
    #    print(filt, P)
    #    default_model = cogmodel.evaluate(radius, P.mtot.default, P.m0.default, P.alpha1.default, P.alpha2.default)
    #    bestfit_model = cogmodel.evaluate(radius, P.mtot.value, P.m0.value, P.alpha1.value, P.alpha2.value)
    #    plt.scatter(radius, cog, label='Data {}'.format(filt), s=10)
    #    plt.plot(radius, bestfit_model, label='Best Fit {}'.format(filt), color='k', lw=1, alpha=0.5)
    #plt.legend(fontsize=10)
    #plt.ylim(15, 30)
    #plt.savefig('junk.png')
    #pdb.set_trace()

    #fig, ax = plt.subplots()
    #for filt in bands:
    #    ax.plot(results['apphot_sma_{}'.format(filt)],
    #            22.5-2.5*np.log10(results['apphot_mag_{}'.format(filt)]), label=filt)
    #ax.set_ylim(30, 5)
    #ax.legend(loc='lower right')
    #plt.show()
    #pdb.set_trace()

    return results

def _unmask_center(img):
    # https://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array
    import numpy.ma as ma
    nn = img.shape[0]
    x0, y0 = geometry.x0, geometry.y0
    rad = geometry.sma # [pixels]
    yy, xx = np.ogrid[-x0:nn-x0, -y0:nn-y0]
    img.mask[xx**2 + yy**2 <= rad**2] = ma.nomask
    return img

def _unpack_isofit(ellipsefit, filt, isofit, failed=False):
    """Unpack the IsophotList objects into a dictionary because the resulting pickle
    files are huge.

    https://photutils.readthedocs.io/en/stable/api/photutils.isophote.IsophoteList.html#photutils.isophote.IsophoteList

    """
    if failed:
        ellipsefit.update({
            '{}_sma'.format(filt): np.array([-1]).astype(np.int16),
            '{}_intens'.format(filt): np.array([-1]).astype('f4'),
            '{}_intens_err'.format(filt): np.array([-1]).astype('f4'),
            '{}_eps'.format(filt): np.array([-1]).astype('f4'),
            '{}_eps_err'.format(filt): np.array([-1]).astype('f4'),
            '{}_pa'.format(filt): np.array([-1]).astype('f4'),
            '{}_pa_err'.format(filt): np.array([-1]).astype('f4'),
            '{}_x0'.format(filt): np.array([-1]).astype('f4'),
            '{}_x0_err'.format(filt): np.array([-1]).astype('f4'),
            '{}_y0'.format(filt): np.array([-1]).astype('f4'),
            '{}_y0_err'.format(filt): np.array([-1]).astype('f4'),
            '{}_a3'.format(filt): np.array([-1]).astype('f4'),
            '{}_a3_err'.format(filt): np.array([-1]).astype('f4'),
            '{}_a4'.format(filt): np.array([-1]).astype('f4'),
            '{}_a4_err'.format(filt): np.array([-1]).astype('f4'),
            '{}_rms'.format(filt): np.array([-1]).astype('f4'),
            '{}_pix_stddev'.format(filt): np.array([-1]).astype('f4'),
            '{}_stop_code'.format(filt): np.array([-1]).astype(np.int16),
            '{}_ndata'.format(filt): np.array([-1]).astype(np.int16), 
            '{}_nflag'.format(filt): np.array([-1]).astype(np.int16), 
            '{}_niter'.format(filt): np.array([-1]).astype(np.int16)})
    else:
        ellipsefit.update({
            '{}_sma'.format(filt): isofit.sma.astype(np.int16),
            '{}_intens'.format(filt): isofit.intens.astype('f4'),
            '{}_intens_err'.format(filt): isofit.int_err.astype('f4'),
            '{}_eps'.format(filt): isofit.eps.astype('f4'),
            '{}_eps_err'.format(filt): isofit.ellip_err.astype('f4'),
            '{}_pa'.format(filt): isofit.pa.astype('f4'),
            '{}_pa_err'.format(filt): isofit.pa_err.astype('f4'),
            '{}_x0'.format(filt): isofit.x0.astype('f4'),
            '{}_x0_err'.format(filt): isofit.x0_err.astype('f4'),
            '{}_y0'.format(filt): isofit.y0.astype('f4'),
            '{}_y0_err'.format(filt): isofit.y0_err.astype('f4'),
            '{}_a3'.format(filt): isofit.a3.astype('f4'),
            '{}_a3_err'.format(filt): isofit.a3_err.astype('f4'),
            '{}_a4'.format(filt): isofit.a4.astype('f4'),
            '{}_a4_err'.format(filt): isofit.a4_err.astype('f4'),
            '{}_rms'.format(filt): isofit.rms.astype('f4'),
            '{}_pix_stddev'.format(filt): isofit.pix_stddev.astype('f4'),
            '{}_stop_code'.format(filt): isofit.stop_code.astype(np.int16),
            '{}_ndata'.format(filt): isofit.ndata.astype(np.int16),
            '{}_nflag'.format(filt): isofit.nflag.astype(np.int16),
            '{}_niter'.format(filt): isofit.niter.astype(np.int16)})
        
    return ellipsefit

def _integrate_isophot_one(args):
    """Wrapper function for the multiprocessing."""
    return integrate_isophot_one(*args)

def integrate_isophot_one(img, sma, theta, eps, x0, y0, pixscalefactor,
                          integrmode, sclip, nclip):
    """Integrate the ellipse profile at a single semi-major axis.

    theta in radians

    """
    import copy
    #g = iso.sample.geometry # fixed geometry
    #g = copy.deepcopy(iso.sample.geometry) # fixed geometry
    g = EllipseGeometry(x0=x0, y0=y0, eps=eps, sma=sma, pa=theta)

    # Use the same integration mode and clipping parameters.
    # The central pixel is a special case:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if g.sma == 0.0:
            gcen = copy.deepcopy(g)
            gcen.sma = 0.0
            gcen.eps = 0.0
            gcen.pa = 0.0
            censamp = CentralEllipseSample(img, 0.0, geometry=gcen,
                                           integrmode=integrmode, sclip=sclip, nclip=nclip)
            out = CentralEllipseFitter(censamp).fit()
        else:
            g.sma *= pixscalefactor
            g.x0 *= pixscalefactor
            g.y0 *= pixscalefactor

            sample = EllipseSample(img, sma=g.sma, geometry=g, integrmode=integrmode,
                                   sclip=sclip, nclip=nclip)
            sample.update(fixed_parameters=True)
            #print(filt, g.sma, sample.mean)

            # Create an Isophote instance with the sample.
            out = Isophote(sample, 0, True, 0)
        
    return out

def ellipse_sbprofile(ellipsefit, minerr=0.0, snrmin=2.0, sma_not_radius=False,
                      cut_on_cog=False, sdss=False, linear=False):
    """Convert ellipse-fitting results to a magnitude, color, and surface brightness
    profiles.

    linear - stay in linear (nanomaggies/arcsec2) units (i.e., don't convert to
      mag/arcsec2) and do not compute colors; used by legacyhalos.integrate

    sma_not_radius - if True, then store the semi-major axis in the 'radius' key
      (converted to arcsec) rather than the circularized radius

    cut_on_cog - if True, limit the sma to where we have successfully measured
      the curve of growth

    """
    sbprofile = dict()
    bands = ellipsefit['bands']
    if 'refpixscale' in ellipsefit.keys():
        pixscale = ellipsefit['refpixscale']
    else:
        pixscale = ellipsefit['pixscale']
    eps = ellipsefit['eps']
    if 'redshift' in ellipsefit.keys():
        sbprofile['redshift'] = ellipsefit['redshift']    
            
    for filt in bands:
        psfkey = 'psfsize_{}'.format(filt)
        sbprofile[psfkey] = ellipsefit[psfkey]

    sbprofile['minerr'] = minerr
    sbprofile['smaunit'] = 'pixels'
    sbprofile['radiusunit'] = 'arcsec'

    # semi-major axis and circularized radius
    #sbprofile['sma'] = ellipsefit[bands[0]].sma * pixscale # [arcsec]

    for filt in bands:
        #area = ellipsefit[filt].sarea[indx] * pixscale**2

        sma = np.atleast_1d(ellipsefit['{}_sma'.format(filt)])   # semi-major axis [pixels]
        sb = np.atleast_1d(ellipsefit['{}_intens'.format(filt)]) # [nanomaggies/arcsec2]
        sberr = np.atleast_1d(np.sqrt(ellipsefit['{}_intens_err'.format(filt)]**2 + (0.4 * np.log(10) * sb * minerr)**2))
            
        if sma_not_radius:
            radius = sma * pixscale # [arcsec]
        else:
            radius = sma * np.sqrt(1 - eps) * pixscale # circularized radius [arcsec]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if linear:
                keep = np.isfinite(sb)
            else:
                keep = np.isfinite(sb) * ((sb / sberr) > snrmin)
            if cut_on_cog:
                keep *= (ellipsefit['{}_sma'.format(filt)] * pixscale) <= np.max(ellipsefit['{}_cog_sma'.format(filt)])
            keep = np.where(keep)[0]
                
            sbprofile['{}_keep'.format(filt)] = keep

        if len(keep) == 0 or sma[0] == -1:
            sbprofile['sma_{}'.format(filt)] = np.array([-1.0]).astype('f4')    # [pixels]
            sbprofile['radius_{}'.format(filt)] = np.array([-1.0]).astype('f4') # [arcsec]
            sbprofile['mu_{}'.format(filt)] = np.array([-1.0]).astype('f4')     # [nanomaggies/arcsec2]
            sbprofile['muerr_{}'.format(filt)] = np.array([-1.0]).astype('f4')  # [nanomaggies/arcsec2]
        else:
            sbprofile['sma_{}'.format(filt)] = sma[keep]       # [pixels]
            sbprofile['radius_{}'.format(filt)] = radius[keep] # [arcsec]
            if linear:
                sbprofile['mu_{}'.format(filt)] = sb[keep] # [nanomaggies/arcsec2]
                sbprofile['muerr_{}'.format(filt)] = sberr[keep] # [nanomaggies/arcsec2]
                continue
            else:
                sbprofile['mu_{}'.format(filt)] = 22.5 - 2.5 * np.log10(sb[keep]) # [mag/arcsec2]
                sbprofile['muerr_{}'.format(filt)] = 2.5 * sberr[keep] / sb[keep] / np.log(10) # [mag/arcsec2]

        #sbprofile[filt] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens)
        #sbprofile['mu_{}_err'.format(filt)] = 2.5 * ellipsefit[filt].int_err / \
        #  ellipsefit[filt].intens / np.log(10)
        #sbprofile['mu_{}_err'.format(filt)] = np.sqrt(sbprofile['mu_{}_err'.format(filt)]**2 + minerr**2)

        # Just for the plot use a minimum uncertainty
        #sbprofile['{}_err'.format(filt)][sbprofile['{}_err'.format(filt)] < minerr] = minerr

    if 'g' in bands and 'r' in bands and 'z' in bands:
        radius_gr, indx_g, indx_r = np.intersect1d(sbprofile['radius_g'], sbprofile['radius_r'], return_indices=True)
        sbprofile['gr'] = sbprofile['mu_g'][indx_g] - sbprofile['mu_r'][indx_r]
        sbprofile['gr_err'] = np.sqrt(sbprofile['muerr_g'][indx_g]**2 + sbprofile['muerr_r'][indx_r]**2)
        sbprofile['radius_gr'] = radius_gr

        radius_rz, indx_r, indx_z = np.intersect1d(sbprofile['radius_r'], sbprofile['radius_z'], return_indices=True)
        sbprofile['rz'] = sbprofile['mu_r'][indx_r] - sbprofile['mu_z'][indx_z]
        sbprofile['rz_err'] = np.sqrt(sbprofile['muerr_r'][indx_r]**2 + sbprofile['muerr_z'][indx_z]**2)
        sbprofile['radius_rz'] = radius_rz
        
    # SDSS
    if sdss and 'g' in bands and 'r' in bands and 'i' in bands:
        radius_gr, indx_g, indx_r = np.intersect1d(sbprofile['radius_g'], sbprofile['radius_r'], return_indices=True)
        sbprofile['gr'] = sbprofile['mu_g'][indx_g] - sbprofile['mu_r'][indx_r]
        sbprofile['gr_err'] = np.sqrt(sbprofile['muerr_g'][indx_g]**2 + sbprofile['muerr_r'][indx_r]**2)
        sbprofile['radius_gr'] = radius_gr

        radius_ri, indx_r, indx_i = np.intersect1d(sbprofile['radius_r'], sbprofile['radius_i'], return_indices=True)
        sbprofile['ri'] = sbprofile['mu_r'][indx_r] - sbprofile['mu_i'][indx_i]
        sbprofile['ri_err'] = np.sqrt(sbprofile['muerr_r'][indx_r]**2 + sbprofile['muerr_i'][indx_i]**2)
        sbprofile['radius_ri'] = radius_ri
        
    # Just for the plot use a minimum uncertainty
    #sbprofile['gr_err'][sbprofile['gr_err'] < minerr] = minerr
    #sbprofile['rz_err'][sbprofile['rz_err'] < minerr] = minerr

    # # Add the effective wavelength of each bandpass, although this needs to take
    # # into account the DECaLS vs BASS/MzLS filter curves.
    # from speclite import filters
    # filt = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z', 'wise2010-W1', 'wise2010-W2')
    # for ii, band in enumerate(('g', 'r', 'z', 'W1', 'W2')):
    #     sbprofile.update({'{}_wave_eff'.format(band): filt.effective_wavelengths[ii].value})

    return sbprofile

def _fitgeometry_refband(ellipsefit, geometry0, majoraxis, refband='r', verbose=False,
                         integrmode='median', sclip=3, nclip=2):
    """Support routine for ellipsefit_multiband. Optionally use photutils to fit for
    the ellipse geometry as a function of semi-major axis.

    """
    smamax = majoraxis # inner, outer radius
    #smamax = 1.5*majoraxis
    smamin = ellipsefit['psfsize_{}'.format(refband)] / ellipsefit['refpixscale']

    if smamin > majoraxis:
        print('Warning! this galaxy is smaller than three times the seeing FWHM!')
        
    t0 = time.time()
    print('Finding the mean geometry using the reference {}-band image...'.format(refband), end='')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        factor = np.arange(1.0, 6, 0.5) # (1, 2, 3, 3.5, 4, 4.5, 5, 10)
        for ii, fac in enumerate(factor): # try a few different starting sma0
            sma0 = smamin*fac
            try:
                iso0 = ellipse0.fit_image(sma0, integrmode=integrmode, sclip=sclip, nclip=nclip)
            except:
                iso0 = []
                sma0 = smamin
            if len(iso0) > 0:
                break
    print('...took {:.3f} sec'.format(time.time()-t0))

    if len(iso0) == 0:
        print('Initial ellipse-fitting failed.')
    else:
        # Try to determine the mean fitted geometry, for diagnostic purposes,
        # masking out outliers and the inner part of the galaxy where seeing
        # dominates.
        good = (iso0.sma > smamin) * (iso0.stop_code <= 4)
        #good = ~sigma_clip(iso0.pa, sigma=3).mask
        #good = (iso0.sma > smamin) * (iso0.stop_code <= 4) * ~sigma_clip(iso0.pa, sigma=3).mask
        #good = (iso0.sma > 3 * ellipsefit['psfsigma_{}'.format(refband)]) * ~sigma_clip(iso0.pa, sigma=3).mask
        #good = (iso0.stop_code < 4) * ~sigma_clip(iso0.pa, sigma=3).mask

        ngood = np.sum(good)
        if ngood == 0:
            print('Too few good measurements to get ellipse geometry!')
        else:
            ellipsefit['success'] = True
            ellipsefit['init_smamin'] = iso0.sma[good].min()
            ellipsefit['init_smamax'] = iso0.sma[good].max()

            ellipsefit['x0_median'] = np.mean(iso0.x0[good])
            ellipsefit['y0_median'] = np.mean(iso0.y0[good])
            ellipsefit['x0_err'] = np.std(iso0.x0[good]) / np.sqrt(ngood)
            ellipsefit['y0_err'] = np.std(iso0.y0[good]) / np.sqrt(ngood)

            ellipsefit['pa'] = (np.degrees(np.mean(iso0.pa[good]))+90) % 180
            ellipsefit['pa_err'] = np.degrees(np.std(iso0.pa[good])) / np.sqrt(ngood)
            ellipsefit['eps'] = np.mean(iso0.eps[good])
            ellipsefit['eps_err'] = np.std(iso0.eps[good]) / np.sqrt(ngood)

            if verbose:
                print(' x0 = {:.3f}+/-{:.3f} (initial={:.3f})'.format(
                    ellipsefit['x0_median'], ellipsefit['x0_err'], ellipsefit['x0']))
                print(' y0 = {:.3f}+/-{:.3f} (initial={:.3f})'.format(
                    ellipsefit['y0_median'], ellipsefit['y0_err'], ellipsefit['y0']))
                print(' PA = {:.3f}+/-{:.3f} (initial={:.3f})'.format(
                    ellipsefit['pa'], ellipsefit['pa_err'], np.degrees(geometry0.pa)+90))
                print(' eps = {:.3f}+/-{:.3f} (initial={:.3f})'.format(
                    ellipsefit['eps'], ellipsefit['eps_err'], geometry0.eps))

    return ellipsefit

def ellipsefit_multiband(galaxy, galaxydir, data, igal=0, galaxy_id='',
                         refband='r', nproc=1, 
                         integrmode='median', nclip=3, sclip=3,
                         maxsma=None, logsma=True, delta_logsma=5.0, delta_sma=1.0,
                         sbthresh=REF_SBTHRESH,
                         galaxyinfo=None, input_ellipse=None, fitgeometry=False,
                         nowrite=False, verbose=False):
    """Multi-band ellipse-fitting, broadly based on--
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    Some, but not all hooks for fitgeometry=True are in here, so user beware.

    galaxyinfo - additional dictionary to append to the output file

    galaxy_id - add a unique ID number to the output filename (via
      io.write_ellipsefit).

    """
    import multiprocessing

    bands, refband, refpixscale = data['bands'], data['refband'], data['refpixscale']
    filesuffix = data['filesuffix']

    if galaxyinfo is not None:
        galaxyinfo = np.atleast_1d(galaxyinfo)
        assert(len(galaxyinfo)==len(data['mge']))
    
    # If fitgeometry=True then fit for the geometry as a function of semimajor
    # axis, otherwise (the default) use the mean geometry of the galaxy to
    # extract the surface-brightness profile.
    if fitgeometry:
        maxrit = None
    else:
        maxrit = -1

    # Initialize the output dictionary, starting from the galaxy geometry in the
    # 'data' dictionary.
    ellipsefit = dict()
    ellipsefit['integrmode'] = integrmode
    ellipsefit['sclip'] = np.int16(sclip)
    ellipsefit['nclip'] = np.int16(nclip)
    ellipsefit['fitgeometry'] = fitgeometry

    if input_ellipse:
        ellipsefit['input_ellipse'] = True
    else:
        ellipsefit['input_ellipse'] = False

    # This is fragile, but copy over a specific set of keys from the data dictionary--
    copykeys = ['bands', 'refband', 'refpixscale',
                'refband_width', 'refband_height',
                #'psfsigma_g', 'psfsigma_r', 'psfsigma_z',
                'psfsize_g', #'psfsize_min_g', 'psfsize_max_g',
                'psfdepth_g', #'psfdepth_min_g', 'psfdepth_max_g', 
                'psfsize_r', #'psfsize_min_r', 'psfsize_max_r',
                'psfdepth_r', #'psfdepth_min_r', 'psfdepth_max_r',
                'psfsize_z', #'psfsize_min_z', 'psfsize_max_z',
                'psfdepth_z'] #'psfdepth_min_z', 'psfdepth_max_z']
    for key in copykeys:
        if key in data.keys():
            ellipsefit[key] = data[key]

    img = data['{}_masked'.format(refband)][igal]
    mge = data['mge'][igal]

    # Fix the center to be the peak (pixel) values. Could also use bx,by here
    # from Tractor.  Also initialize the geometry with the moment-derived
    # values.  Note that (x,y) are switched between MGE and photutils!!
    for key in ['largeshift', 'eps', 'pa', 'theta', 'majoraxis', 'ra_x0', 'dec_y0',
                'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_z']:
        ellipsefit[key] = mge[key]
    for mgekey, ellkey in zip(['ymed', 'xmed'], ['x0', 'y0']):
        ellipsefit[ellkey] = mge[mgekey]

    majoraxis = mge['majoraxis'] # [pixel]

    # Get the mean geometry of the system by ellipse-fitting the inner part and
    # taking the mean values of everything.

    # http://photutils.readthedocs.io/en/stable/isophote_faq.html#isophote-faq
    # Note: position angle in photutils is measured counter-clockwise from the
    # x-axis, while .pa in MGE measured counter-clockwise from the y-axis.
    geometry0 = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'],
                                eps=ellipsefit['eps'], sma=0.5*majoraxis, 
                                pa=np.radians(ellipsefit['pa']-90))
    ellipse0 = Ellipse(img, geometry=geometry0)
    #import matplotlib.pyplot as plt
    #plt.imshow(img, origin='lower') ; plt.scatter(ellipsefit['y0'], ellipsefit['x0'], s=50, color='red') ; plt.savefig('junk.png')
    #pdb.set_trace()

    if fitgeometry:
        ellipsefit = _fitgeometry_refband(ellipsefit, geometry0, majoraxis, refband,
                                          integrmode=integrmode, sclip=sclip, nclip=nclip,
                                          verbose=verbose)
    
    # Re-initialize the EllipseGeometry object, optionally using an external set
    # of ellipticity parameters.
    if input_ellipse:
        print('Using input ellipse parameters.')
        ellipsefit['input_ellipse'] = True
        input_eps, input_pa = input_ellipse['eps'], input_ellipse['pa'] % 180
        geometry = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'],
                                   eps=input_eps, sma=majoraxis, 
                                   pa=np.radians(input_pa-90))
    else:
        # Note: we use the MGE, not fitted geometry here because it's more
        # reliable based on visual inspection.
        geometry = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'],
                                   eps=ellipsefit['eps'], sma=majoraxis, 
                                   pa=np.radians(ellipsefit['pa']-90))

    geometry_cen = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'],
                                   eps=0.0, sma=0.0, pa=0.0)
    #ellipsefit['geometry'] = geometry # can't save an object in an .asdf file
    ellipse = Ellipse(img, geometry=geometry)

    # Integrate to the edge [pixels].
    if maxsma is None:
        maxsma = 0.95 * (data['refband_width']/2) / np.cos(geometry.pa % (np.pi/4))
    ellipsefit['maxsma'] = np.float32(maxsma) # [pixels]

    if logsma:
        #https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
        def _mylogspace(limit, n):
            result = [1]
            if n > 1:  # just a check to avoid ZeroDivisionError
                ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
            while len(result) < n:
                next_value = result[-1]*ratio
                if next_value - result[-1] >= 1:
                    # safe zone. next_value will be a different integer
                    result.append(next_value)
                else:
                    # problem! same integer. we need to find next_value by artificially incrementing previous value
                    result.append(result[-1]+1)
                    # recalculate the ratio so that the remaining values will scale correctly
                    ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
            # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
            return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.int32)

        # this algorithm can fail if there are too few points
        nsma = np.ceil(maxsma / delta_logsma).astype('int')
        sma = _mylogspace(maxsma, nsma).astype('f4')
        #sma = np.hstack((0, np.logspace(0, np.ceil(np.log10(maxsma)).astype('int'), nsma, dtype=np.int))).astype('f4')
        print('  maxsma={:.2f} pix, delta_logsma={:.1f} log-pix, nsma={}'.format(maxsma, delta_logsma, len(sma)))
    else:
        sma = np.arange(0, np.ceil(maxsma), delta_sma).astype('f4')
        #ellipsefit['sma'] = np.arange(np.ceil(maxsma)).astype('f4')
        print('  maxsma={:.2f} pix, delta_sma={:.1f} pix, nsma={}'.format(maxsma, delta_sma, len(sma)))

    # this assert will fail when integrating the curve of growth using
    # integrate.simps because the x-axis values have to be unique.
    assert(len(np.unique(sma)) == len(sma))

    nbox = 3
    box = np.arange(nbox)-nbox // 2
    
    # Now get the surface brightness profile.  Need some more code for this to
    # work with fitgeometry=True...
    pool = multiprocessing.Pool(nproc)

    tall = time.time()
    for filt in bands:
        print('Fitting {}-band took...'.format(filt), end='')
        img = data['{}_masked'.format(filt)][igal]

        # Loop on the reference band isophotes.
        t0 = time.time()
        #isobandfit = pool.map(_integrate_isophot_one, [(iso, img, pixscalefactor, integrmode, sclip, nclip)

        # In extreme cases, and despite my best effort in io.read_multiband, the
        # image at the central position of the galaxy can end up masked, which
        # always points to a deeper issue with the data (e.g., bleed trail,
        # extremely bright star, etc.). Capture that corner case here.
        imasked, val = False, []
        for xb in box:
            for yb in box:
                val.append(img.mask[int(xb+ellipsefit['x0']), int(yb+ellipsefit['y0'])])
        if np.any(val):
            imasked = True

        if imasked:
        #if img.mask[np.int(ellipsefit['x0']), np.int(ellipsefit['y0'])]:
            print(' Central pixel is masked; resorting to extreme measures!')
            ellipsefit = _unpack_isofit(ellipsefit, filt, None, failed=True)
        else:
            isobandfit = pool.map(_integrate_isophot_one, [(
                img, _sma, ellipsefit['pa'], ellipsefit['eps'], ellipsefit['x0'],
                ellipsefit['y0'], 1.0, integrmode, sclip, nclip) for _sma in sma])
            ellipsefit = _unpack_isofit(ellipsefit, filt, IsophoteList(isobandfit))
        
        print('...{:.3f} sec'.format(time.time() - t0))
    print('Time for all images = {:.3f} min'.format((time.time()-tall)/60))

    ellipsefit['success'] = True

    # Perform elliptical aperture photometry--
    print('Performing elliptical aperture photometry.')
    t0 = time.time()
    cog = ellipse_cog(bands, data, ellipsefit, 1.0, refpixscale,
                      igal=igal, pool=pool, sbthresh=sbthresh)
    ellipsefit.update(cog)
    del cog
    print('Time = {:.3f} min'.format( (time.time() - t0) / 60))

    pool.close()

    # Write out
    if not nowrite:
        if galaxyinfo is None:
            outgalaxyinfo = None
        else:
            outgalaxyinfo = galaxyinfo[igal]
            ellipsefit.update(galaxyinfo[igal])

        legacyhalos.io.write_ellipsefit(galaxy, galaxydir, ellipsefit,
                                        galaxy_id=galaxy_id,
                                        galaxyinfo=outgalaxyinfo,
                                        refband=refband,
                                        sbthresh=sbthresh,
                                        verbose=True,
                                        filesuffix=filesuffix)

    return ellipsefit

def legacyhalos_ellipse(galaxy, galaxydir, data, galaxyinfo=None,
                        pixscale=0.262, nproc=1, refband='r',
                        bands=['g', 'r', 'z'], integrmode='median',
                        nclip=3, sclip=3, sbthresh=REF_SBTHRESH,
                        delta_sma=1.0, delta_logsma=5, maxsma=None, logsma=True,
                        input_ellipse=None, fitgeometry=False,
                        verbose=False, debug=False):
                        
    """Top-level wrapper script to do ellipse-fitting on a single galaxy.

    fitgeometry - fit for the ellipse parameters (do not use the mean values
      from MGE).

    """
    if bool(data):
        if data['missingdata']:
            if os.path.isfile(os.path.join(galaxydir, '{}-{}-coadds.isdone'.format(galaxy, data['filesuffix']))):
                return 1
            else:
                return 0

        if data['failed']: # all galaxies dropped
            return 1

        if 'galaxy_id' in data.keys():
            galaxy_id = np.atleast_1d(data['galaxy_id'])
        else:
            galaxy_id = ['']
            
        for igal, galid in enumerate(galaxy_id):
            ellipsefit = ellipsefit_multiband(galaxy, galaxydir, data,
                                              galaxyinfo=galaxyinfo,
                                              igal=igal, galaxy_id=str(galid),
                                              delta_logsma=delta_logsma, maxsma=maxsma,
                                              delta_sma=delta_sma, logsma=logsma,
                                              refband=refband, nproc=nproc, sbthresh=sbthresh,
                                              integrmode=integrmode, nclip=nclip, sclip=sclip,                                           
                                              input_ellipse=input_ellipse, 
                                              verbose=verbose, fitgeometry=False)
        return 1
    else:
        # An object can get here if it's a "known" failure, e.g., if the object
        # falls off the edge of the footprint (and therefore it will never have
        # coadds).
        if os.path.isfile(os.path.join(galaxydir, '{}-{}-coadds.isdone'.format(galaxy, 'custom'))):
            return 1
        else:
            return 0
