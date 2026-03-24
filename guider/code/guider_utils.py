"""
Guider and WCS Utility Code for LSST Camera

This module provides utilities for working with LSST Camera guider data, including:
- Coordinate transformations between CCD and DVCS (Detector View Coordinate System) views
- ROI (Region of Interest) bounding box creation and manipulation
- Guider stamp extraction and processing
- Star catalog matching and tracking
- Visualization tools for guider data

The module integrates with the LSST Science Pipelines and provides utilities for
analyzing guide star data from the LSST Camera's guider CCDs.
"""

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import Counter

# LSST Butler
from lsst.resources import ResourcePath
from lsst.daf.butler import Butler
from lsst.summit.utils.efdUtils import makeEfdClient, getEfdData
import lsst.summit.utils.butlerUtils as butlerUtils
from lsst.summit.utils.butlerUtils import getExpRecordFromDataId

# EFD Client
client = makeEfdClient()

# Core LSST Camera code
import lsst.afw.math as afwMath
from lsst.afw import cameraGeom
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms
from lsst.obs.lsst import LsstCam

camera = LsstCam.getCamera()
det_nums = {det.getName(): i for i, det in enumerate(camera)}

# AstroPy code
from astropy.coordinates import Angle as AAngle
import astropy.units as u
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord, angular_separation
from astropy.table import Table, join, vstack

# LSST code for Stamps
from lsst.meas.algorithms.stamps import Stamp, Stamps
from lsst.afw.image import MaskedImageF, ImageF
from lsst.obs.base import createInitialSkyWcsFromBoresight

# LSST geometry
import lsst.geom
from lsst.geom import (Point2I, Box2I, Extent2I, Point2D, Box2D, Extent2D,
                       Angle, degrees, AffineTransform, SpherePoint)

# ROI code
from lsst.ts.observatory.control.utils.extras.guider_roi import GuiderROIs  # noqa: F401

# Code from Summit Utils
from lsst.summit.utils.guiders.transformation import stampToCcd, ampToCcdView, pixelToFocal
from lsst.summit.utils.guiders.reading import GuiderReader
from lsst.summit.utils.guiders.tracking import GuiderStarTracker, GuiderStarTrackerConfig
from lsst.summit.utils.guiders.plotting import GuiderPlotter
from lsst.summit.utils.guiders.metrics import GuiderMetricsBuilder


# ==============================================================================
# Coordinate Transformation Functions
# ==============================================================================

def mk_rot(det_nquarter):
    """
    Create a rotation transformation matrix for detector orientation.
    
    This function creates an affine transformation that rotates coordinates by
    90-degree increments (quarters) to account for detector orientation in the
    focal plane.
    
    Parameters
    ----------
    det_nquarter : int
        Number of 90-degree rotations needed to align the detector with DVCS.
        Valid values are 0, 1, 2, or 3 (modulo 4 is applied internally).
    
    Returns
    -------
    frotation : lsst.geom.AffineTransform
        An affine transformation representing the rotation.
    
    Notes
    -----
    The rotation matrices correspond to:
    - 0 quarters: identity (no rotation)
    - 1 quarter: 90° counterclockwise
    - 2 quarters: 180° rotation
    - 3 quarters: 270° counterclockwise (90° clockwise)
    
    Examples
    --------
    >>> rot_transform = mk_rot(1)  # 90° rotation
    >>> point = Point2D(1.0, 0.0)
    >>> rotated_point = rot_transform(point)
    """
    # Define rotation matrices for each 90-degree increment
    rot = {
        0: np.array([[1., 0.], [0., 1.]]),   # 0°
        1: np.array([[0., -1], [1., 0.]]),   # 90°
        2: np.array([[-1., 0.], [0., -1.]]), # 180°
        3: np.array([[0., 1.], [-1., 0.]])   # 270°
    }
    
    nq = np.mod(det_nquarter, 4)
    frotation = AffineTransform(rot[nq])
    
    return frotation


def mk_ccd_to_dvcs(llpt_ccd, det_nquarter):
    """
    Create transformations between CCD pixel coordinates and DVCS coordinates.
    
    This function creates forward and backward transformations suitable for
    guider stamps, converting between CCD pixel coordinates (with origin at
    lower left) and DVCS view (oriented correctly in the focal plane).
    
    Parameters
    ----------
    llpt_ccd : lsst.geom.Extent2D
        The lower-left point of the stamp in CCD pixel coordinates.
    det_nquarter : int
        Number of 90-degree quarters needed to rotate from CCD view to DVCS view.
    
    Returns
    -------
    forwards : lsst.geom.AffineTransform
        Transform from CCD pixel coordinates to stamp pixel coordinates in DVCS view.
    backwards : lsst.geom.AffineTransform
        Transform from stamp pixel coordinates in DVCS view to CCD pixel coordinates.
    
    Examples
    --------
    >>> detector = camera[189]
    >>> llpt_ccd = Extent2D(100., 200.)
    >>> ft, bt = mk_ccd_to_dvcs(llpt_ccd, detector.getOrientation().getNQuarter())
    >>> 
    >>> # Convert sky coordinate to stamp position in DVCS view
    >>> pt_ccd = wcs.skyToPixel(sky_coord)
    >>> pt_stamp_dvcs = ft(pt_ccd)
    >>> 
    >>> # Convert stamp position back to sky coordinate
    >>> pt_ccd = bt(pt_stamp_dvcs)
    >>> sky_coord = wcs.pixelToSky(pt_ccd)
    
    Notes
    -----
    The transformation sequence is: translate to origin, then rotate.
    The inverse transformation applies rotation inverse, then translation.
    """
    # Define rotation matrices
    rot = {
        0: np.array([[1., 0.], [0., 1.]]),
        1: np.array([[0., -1], [1., 0.]]),
        2: np.array([[-1., 0.], [0., -1.]]),
        3: np.array([[0., 1.], [-1., 0.]])
    }
    
    # Define inverse rotation matrices (transpose for orthogonal matrices)
    irot = {i: rot[i].transpose() for i in range(4)}
    
    nq = np.mod(det_nquarter, 4)
    
    # Forward transformation: translate, then rotate
    ftranslation = AffineTransform(-llpt_ccd)
    frotation = AffineTransform(rot[nq])
    forwards = frotation * ftranslation  # Note: ordering is second * first
    
    # Backward transformation: inverse rotate, then translate back
    btranslation = AffineTransform(llpt_ccd)
    brotation = AffineTransform(irot[nq])
    backwards = btranslation * brotation
    
    return forwards, backwards


# ==============================================================================
# ROI (Region of Interest) Functions
# ==============================================================================

def mk_roi_bboxes(md, camera):
    """
    Create bounding boxes for a guider stamp in both CCD and DVCS views.
    
    This function extracts ROI information from metadata and creates bounding
    boxes in both CCD pixel coordinates and DVCS (rotated focal plane) coordinates.
    
    Parameters
    ----------
    md : dict-like
        Metadata from one guider CCD exposure, containing ROI information including:
        ROICOL, ROIROW, ROICOLS, ROIROWS, and detector/amplifier identifiers.
    camera : lsst.afw.cameraGeom.Camera
        Camera object (typically LsstCam).
    
    Returns
    -------
    ccd_view_bbox : lsst.geom.Box2I
        Bounding box for the stamp in full CCD coordinates (CCD view).
    dvcs_view_bbox : lsst.geom.Box2I
        Bounding box for the stamp rotated to DVCS (focal plane) view.
    
    Notes
    -----
    The function accounts for amplifier flips (RawFlipX, RawFlipY) when
    determining the actual corners of the ROI in CCD coordinates.
    """
    # Extract ROI information from metadata
    roiCol = md['ROICOL']
    roiRow = md['ROIROW']
    roiCols = md['ROICOLS']
    roiRows = md['ROIROWS']
    
    # Get detector and amplifier
    detector, ampName = get_detector_amp(md, camera)
    amp = detector[ampName]
    
    # Get corner0 of the ROI location in CCD coordinates
    lct = LsstCameraTransforms(camera, detector.getName())
    corner0_CCDX, corner0_CCDY = lct.ampPixelToCcdPixel(roiCol, roiRow, ampName)
    corner0_CCDX = int(corner0_CCDX)
    corner0_CCDY = int(corner0_CCDY)
    
    # Get opposite corner (corner2) of the ROI
    # The location depends on amplifier flip settings
    if amp.getRawFlipX():
        corner2_CCDX = corner0_CCDX - roiCols
    else:
        corner2_CCDX = corner0_CCDX + roiCols
    
    if amp.getRawFlipY():
        corner2_CCDY = corner0_CCDY - roiRows
    else:
        corner2_CCDY = corner0_CCDY + roiRows
    
    # Create CCD view bounding box with lower-left point at smallest x,y
    ll_x = min(corner0_CCDX, corner2_CCDX)
    ur_x = max(corner0_CCDX, corner2_CCDX)
    ll_y = min(corner0_CCDY, corner2_CCDY)
    ur_y = max(corner0_CCDY, corner2_CCDY)
    ll_CCD = Point2D(ll_x, ll_y)
    ur_CCD = Point2D(ur_x, ur_y)
    ccd_view_bbox = Box2I(Box2D(ll_CCD, ur_CCD))
    
    # Create DVCS view bounding box by rotating the CCD corners
    frot = mk_rot(detector.getOrientation().getNQuarter())
    ll_rot = frot(ll_CCD)
    ur_rot = frot(ur_CCD)
    ll_x_dvcs = min(ll_rot.getX(), ur_rot.getX())
    ur_x_dvcs = max(ll_rot.getX(), ur_rot.getX())
    ll_y_dvcs = min(ll_rot.getY(), ur_rot.getY())
    ur_y_dvcs = max(ll_rot.getY(), ur_rot.getY())
    ll_DVCS = Point2D(ll_x_dvcs, ll_y_dvcs)
    ur_DVCS = Point2D(ur_x_dvcs, ur_y_dvcs)
    dvcs_view_bbox = Box2I(Box2D(ll_DVCS, ur_DVCS))
    
    return ccd_view_bbox, dvcs_view_bbox


def get_detector_amp(md, camera):
    """
    Extract detector and amplifier information from metadata.
    
    Parameters
    ----------
    md : dict-like
        Metadata containing RAFTBAY, CCDSLOT, OBSID, and ROISEG fields.
    camera : lsst.afw.cameraGeom.Camera
        Camera object (typically LsstCam).
    
    Returns
    -------
    detector : lsst.afw.cameraGeom.Detector
        The detector object.
    ampName : str
        The amplifier name (e.g., 'C10', 'C15').
    
    Notes
    -----
    The observation ID (OBSID) format is expected to be compatible with
    dayObs and seqNum extraction.
    """
    raftBay = md['RAFTBAY']
    ccdSlot = md['CCDSLOT']
    obsId = md['OBSID']
    dayObs = int(obsId[5:13])
    seqNum = int(obsId[14:])
    
    segment = md['ROISEG']
    ampName = 'C' + segment[7:]
    detName = raftBay + '_' + ccdSlot
    detector = camera[detName]
    
    return detector, ampName


def convert_roi(roi, detector, ampName, camera, view='dvcs'):
    """
    Convert ROI image array between CCD and DVCS views.
    
    This function transforms a guider ROI image from amplifier coordinates
    to either CCD view (standard CCD pixel coordinates) or DVCS view
    (rotated to match focal plane orientation).
    
    Parameters
    ----------
    roi : numpy.ndarray
        ROI image array in amplifier coordinates.
    detector : lsst.afw.cameraGeom.Detector
        The detector object.
    ampName : str
        The amplifier name (e.g., 'C10').
    camera : lsst.afw.cameraGeom.Camera
        Camera object (not used in current implementation but kept for API consistency).
    view : {'dvcs', 'ccd'}, optional
        Target coordinate system:
        - 'dvcs': Detector View Coordinate System (focal plane orientation)
        - 'ccd': CCD pixel coordinates
        Default is 'dvcs'.
    
    Returns
    -------
    imf : lsst.afw.image.ImageF
        Image in the requested view with the same pixel values but
        potentially rotated/flipped.
    
    Notes
    -----
    The transformation sequence is:
    1. Convert from amplifier view to CCD view (handles flips)
    2. Rotate by detector's getNQuarter() to get DVCS view
    
    Examples
    --------
    >>> roi_array = guider_raw.image.array
    >>> roi_dvcs = convert_roi(roi_array, detector, 'C10', camera, view='dvcs')
    """
    # Convert image from amplifier view to CCD view
    roi_ccdview = ampToCcdView(roi, detector, ampName)
    
    # Convert image to DVCS view by rotating based on detector orientation
    roi_dvcsview = np.rot90(roi_ccdview, -detector.getOrientation().getNQuarter())
    
    # Create output ImageF with same dimensions as input
    ny, nx = roi.shape
    imf = ImageF(nx, ny, 0.)
    
    # Populate output image based on requested view
    if view == 'dvcs':
        imf.array[:] = roi_dvcsview
    elif view == 'ccd':
        imf.array[:] = roi_ccdview
    
    return imf


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_guiders(camera, stamps_dict, istamp, plo=10.0, phi=99.0, 
                 biastype='col', view='dvcs', title='Guiders', 
                 filename=None, figsize=(7, 7)):
    """
    Plot guider stamp images in their focal plane layout.
    
    This function creates a mosaic plot showing all guider CCDs arranged in
    their physical positions in the focal plane, with proper orientation.
    
    Parameters
    ----------
    camera : lsst.afw.cameraGeom.Camera
        Camera object (typically LsstCam).
    stamps_dict : dict
        Dictionary mapping detector IDs to Stamps objects.
    istamp : int
        Index of the stamp to display from each Stamps object.
    plo, phi : float, optional
        Lower and upper percentiles for image display scaling. Default: 10.0, 99.0.
    biastype : {'col', 'scalar'}, optional
        Type of bias subtraction:
        - 'col': subtract median along columns (or rows in DVCS if rotated)
        - 'scalar': subtract overall median
        Default is 'col'.
    view : {'dvcs', 'ccd'}, optional
        Coordinate system for display. Default is 'dvcs'.
    title : str, optional
        Figure title. Default is 'Guiders'.
    filename : str or None, optional
        If provided, save figure to this file. Default is None.
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default is (7, 7).
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axs : dict
        Dictionary of axes objects keyed by detector name.
    
    Notes
    -----
    The layout matches the physical arrangement of guider CCDs in the
    LSST Camera focal plane:
    
        .      R40_SG1  R44_SG0  .
        R40_SG0    .        .    R44_SG1
        R00_SG1    .        .    R04_SG0
        .      R00_SG0  R04_SG1  .
    
    Examples
    --------
    >>> fig, axs = plot_guiders(camera, stamps_dict, istamp=0, 
    ...                          title='Guide Stars - Exposure 123456')
    """
    # Define the focal plane layout
    layout = [
        [".", "R40_SG1", "R44_SG0", "."],
        ["R40_SG0", ".", ".", "R44_SG1"],
        ["R00_SG1", ".", ".", "R04_SG0"],
        [".", "R00_SG0", "R04_SG1", "."],
    ]
    
    fig, axs = plt.subplot_mosaic(layout, figsize=figsize)
    
    for idet in stamps_dict.keys():
        detector = camera[idet]
        detname = detector.getName()
        
        # Extract image data
        img = stamps_dict[idet].getMaskedImages()[istamp].getImage().getArray()
        
        # Apply bias subtraction
        if biastype == 'col':
            # Determine axis for column/row median based on view and orientation
            if view == 'dvcs':
                nq = detector.getOrientation().getNQuarter()
                # If rotated by odd quarters, rows become columns
                rowcol_axis = 1 if (nq % 2 == 1) else 0
            else:
                rowcol_axis = 0
            
            img_median = np.median(img, axis=rowcol_axis, keepdims=True)
        else:
            img_median = np.median(img)
        
        # Subtract bias and get display range
        img_isr = img - img_median
        lo, hi = np.nanpercentile(img_isr, [plo, phi])
        
        # Display image
        im = axs[detname].imshow(img_isr, origin='lower', cmap='Greys', 
                                  vmin=lo, vmax=hi)
        
        # Clean up axis ticks and labels
        axs[detname].set_yticklabels([])
        axs[detname].set_xticklabels([])
        axs[detname].set_xticks([])
        axs[detname].set_xticks([], minor=True)
        axs[detname].set_yticks([])
        axs[detname].set_yticks([], minor=True)
    
    fig.suptitle(title)
    fig.tight_layout()
    
    if filename is not None:
        fig.savefig(filename)
    
    return fig, axs


# ==============================================================================
# WCS and Pointing Offset Functions
# ==============================================================================

def get_science_wcs(butler, seqNum, dayObs, idetList):
    """
    Get initial and preliminary visit WCS for science detectors.
    
    Parameters
    ----------
    butler : lsst.daf.butler.Butler
        Butler object for data access.
    seqNum : int
        Sequence number.
    dayObs : int
        Day of observation in YYYYMMDD format.
    idetList : list of int
        List of detector IDs.
    
    Returns
    -------
    init_wcs : dict
        Initial WCS for each detector, keyed by detector ID.
    pvisit_wcs : dict
        Preliminary visit WCS for each detector, keyed by detector ID.
    
    Examples
    --------
    >>> idetList = [94, 95, 96]  # Example science detector IDs
    >>> init_wcs, pvisit_wcs = get_science_wcs(butler, 123, 20240115, idetList)
    """
    init_wcs = {}
    pvisit_wcs = {}
    
    for idet in tqdm(idetList):
        imdataId = {'instrument': 'LSSTCam', 'detector': idet, 
                    'day_obs': dayObs, 'seq_num': seqNum}
        init_wcs[idet] = butler.get('raw.wcs', imdataId)
        pvisit_wcs[idet] = butler.get('preliminary_visit_image.wcs', imdataId)
    
    return init_wcs, pvisit_wcs


def calc_pointing_offset(camera, init_wcs, pvisit_wcs, idet):
    """
    Calculate pointing offset between initial and preliminary visit WCS.
    
    This function computes the offset in pixels and arcseconds between the
    initial boresight WCS and the preliminary visit WCS for a given detector.
    
    Parameters
    ----------
    camera : lsst.afw.cameraGeom.Camera
        Camera object.
    init_wcs : dict
        Initial WCS keyed by detector ID.
    pvisit_wcs : dict
        Preliminary visit WCS keyed by detector ID.
    idet : int
        Detector ID.
    
    Returns
    -------
    delta_X : float
        Offset in X direction (pixels).
    delta_Y : float
        Offset in Y direction (pixels).
    delta_ra : float
        Offset in RA direction (arcseconds).
    delta_dec : float
        Offset in Dec direction (arcseconds).
    init_pixscale : float
        Initial pixel scale (arcseconds/pixel).
    pvisit_pixscale : float
        Preliminary visit pixel scale (arcseconds/pixel).
    delta_rot : astropy.coordinates.Angle
        Rotation offset between WCS solutions.
    
    Examples
    --------
    >>> delta_X, delta_Y, delta_ra, delta_dec, _, _, delta_rot = \
    ...     calc_pointing_offset(camera, init_wcs, pvisit_wcs, 94)
    """
    detector = camera[idet]
    bbox = detector.getBBox()
    center = bbox.getCenter()
    x0, y0 = detector.getBBox().getCenter()
    
    # Find ra,dec at detector's center using pvisit_wcs, then find the 
    # corresponding pixel from init_wcs
    radec_center = pvisit_wcs[idet].pixelToSky(center)
    init_center = init_wcs[idet].skyToPixel(radec_center)
    delta_center = center - init_center  # in units of pixels
    delta_X = delta_center.getX()
    delta_Y = delta_center.getY()
    
    # Using pvisit_wcs, find the delta_RA and delta_Dec between the 
    # init_wcs and the pvisit_wcs
    offset = Point2D(center.getX() + delta_X, center.getY() + delta_Y)
    radec_offset = pvisit_wcs[idet].pixelToSky(offset)
    delta_ra = (radec_offset.getRa() - radec_center.getRa()).asArcseconds()
    delta_dec = (radec_offset.getDec() - radec_center.getDec()).asArcseconds()
    
    # Get pixel scale from pvisit_wcs
    init_pixscale = init_wcs[idet].getPixelScale(Point2D(x0, y0)).asArcseconds()
    pvisit_pixscale = pvisit_wcs[idet].getPixelScale(Point2D(x0, y0)).asArcseconds()
    
    # Get delta rotation
    awcs = init_wcs[idet]
    bwcs = pvisit_wcs[idet]
    delta_rot = AAngle(awcs.getRelativeRotationToWcs(bwcs).asDegrees() * u.deg)
    delta_rot = delta_rot.wrap_at('180d')
    
    return delta_X, delta_Y, delta_ra, delta_dec, init_pixscale, pvisit_pixscale, delta_rot


# ==============================================================================
# Star Catalog and Tracking Functions
# ==============================================================================

def get_guider_data(butler, groi, dayObs, seqNum):
    """
    Get comprehensive guider data for a single exposure.
    
    This function retrieves and combines:
    1. Expected ROI locations from the catalog (based on ra, dec, sky_angle)
    2. Actual ROI locations from the exposure metadata (from GuiderReader)
    3. Detected stars from GuiderStarTracker (filtered to stamp==10)
    
    The output table has exactly 8 rows (one per guider detector), with NaN
    values in star columns for detectors where no star was detected.
    
    Parameters
    ----------
    butler : lsst.daf.butler.Butler
        Butler for data access.
    groi : GuiderROIs
        GuiderROIs object for catalog queries.
    dayObs : int
        Observation day in YYYYMMDD format.
    seqNum : int
        Sequence number.
    
    Returns
    -------
    combined_table : astropy.table.Table or None
        Combined table containing:
        - Catalog star information (RA, Dec, magnitude, etc.)
        - Expected ROI location (from catalog)
        - Actual ROI location (ampNameH, ampxLLH, ampyLLH, issplit)
        - Detected star information for stamp==10 (centroid, flux, etc.)
        Always contains 8 rows (one per guider), with NaN for missing detections.
        Returns None if GuiderReader fails to retrieve data.
    
    Notes
    -----
    The table combines three data sources:
    1. selected_stars: Expected guide stars from catalog (8 entries)
    2. ROI metadata: Actual ROI placement from exposure headers (8 entries)
    3. Detected stars: Stars found by GuiderStarTracker, filtered to stamp==10
    
    The join is done with join_type='left' so all 8 guiders appear in the output
    even if some have no detected stars.
    
    Examples
    --------
    >>> result = get_guider_data(butler, groi, 20240115, 123)
    >>> if result is not None:
    ...     print(f"Table has {len(result)} rows (should be 8)")
    ...     # Check which detectors had detections
    ...     has_detection = ~np.isnan(result['x']) if 'x' in result.colnames else []
    ...     print(f"Detected stars in {has_detection.sum()} guiders")
    """
    # Get guider data using GuiderReader
    reader = GuiderReader(butler, view="dvcs")
    try:
        guiderData = reader.get(dayObs=dayObs, seqNum=seqNum, doSubtractMedian=True)
    except Exception as e:
        # If GuiderReader.get fails, return None
        return None
    
    # Get exposure information
    expId = dayObs * 100000 + seqNum
    dataId = {'exposure': expId, 'instrument': 'LSSTCam'}
    expRecord = getExpRecordFromDataId(butler, dataId)
    ra = expRecord.tracking_ra
    dec = expRecord.tracking_dec
    skyang = expRecord.sky_angle
    physical_filter = expRecord.physical_filter
    band = physical_filter[0:1]
    
    # Get WCS for detector 94 to calculate pointing offsets
    try:
        init_wcs, pvisit_wcs = get_science_wcs(butler, seqNum, dayObs, [94])
    except Exception as e:
        # If WCS retrieval fails, set to None
        init_wcs = None
        pvisit_wcs = None
    
    # Get expected ROI locations from catalog
    roi_spec, selected_stars = groi.get_guider_rois(
        ra=ra,
        dec=dec,
        sky_angle=skyang,
        roi_size=400,
        roi_time=200,
        band=band,
        use_guider=True,
        use_science=False,
        use_wavefront=False,
    )
    
    # Add corrected Gaia magnitude to selected_stars
    if len(selected_stars) > 0:
        selected_stars['gaia_G_corr'] = selected_stars['gaia_G'] + selected_stars['delta_mag']
    
    # Extract actual ROI location information from GuiderReader metadata
    vars = ['seqNum', 'ccdName', 'ampNameH', 'ampxLLH', 'ampyLLH', 'issplit']
    dfdict = {var: [] for var in vars}
    
    # Loop over all detectors in camera and check if they are guiders
    for detector in camera:
        if detector.getType() == cameraGeom.DetectorType.GUIDER:
            detName = detector.getName()
            
            # Try to get metadata from GuiderReader data
            try:
                # Access detector data from guiderData
                aStamp = guiderData[detName][0]  # use Stamp=0
                if aStamp is not None and aStamp.metadata is not None:
                    mdata = aStamp.metadata
                    llX = mdata['ROICOL']
                    llY = mdata['ROIROW']
                    seg = mdata['ROISEG']
                    issplit = mdata['ROISPLIT']
                    
                    dfdict['ccdName'].append(detName)
                    dfdict['ampNameH'].append(f"C{int(seg[7:]):02d}")
                    dfdict['ampxLLH'].append(llX)
                    dfdict['ampyLLH'].append(llY)
                    dfdict['seqNum'].append(seqNum)
                    dfdict['issplit'].append(issplit)
                else:
                    raise KeyError("Detector data or metadata not available")
                    
            except (KeyError, TypeError, AttributeError):
                # Handle missing or invalid metadata
                dfdict['ccdName'].append(detName)
                dfdict['ampNameH'].append("CC")
                dfdict['ampxLLH'].append(-1)
                dfdict['ampyLLH'].append(-1)
                dfdict['seqNum'].append(seqNum)
                dfdict['issplit'].append(False)
    
    tabROI = Table(dfdict)
    
    # Get detected stars from GuiderStarTracker
    config = GuiderStarTrackerConfig()
    starTracker = GuiderStarTracker(guiderData, config)
    stars = starTracker.trackGuiderStars(refCatalog=None)
    stars_ap = Table.from_pandas(stars)
    
    # Filter stars_ap to only include stamp==10
    if len(stars_ap) > 0:
        stars_ap = stars_ap[stars_ap['stamp'] == 10]
        
        # Convert timestamp to MJD
        timecol = stars_ap['timestamp'].tolist()
        timecol_mjd = [atime.mjd if atime is not None else np.nan for atime in timecol]
        stars_ap['mjd'] = timecol_mjd
        
        # Remove the timestamp column
        stars_ap.remove_column('timestamp')
    
    # Join tables together
    # First join catalog stars with ROI locations (both have 8 entries)
    if len(selected_stars) > 0 and len(tabROI) > 0:
        combined = join(selected_stars, tabROI, keys='ccdName')
        
        # Then left join with detected stars (filtered to stamp==10)
        # This keeps all 8 guiders, with NaN for those without detections
        if len(stars_ap) > 0:
            combined = join(combined, stars_ap, 
                          keys_left='ccdName', keys_right='detector',
                          join_type='left')
        else:
            # No stars detected, but still return the combined catalog + ROI table
            pass
    else:
        combined = None
    
    # Add pointing offset information from WCS
    if combined is not None and init_wcs is not None and pvisit_wcs is not None:
        # Calculate pointing offsets using detector 94
        try:
            delta_X, delta_Y, delta_ra, delta_dec, init_pixscale, pvisit_pixscale, delta_rot = \
                calc_pointing_offset(camera, init_wcs, pvisit_wcs, 94)
            
            # Add these as columns (same value for all rows in this exposure)
            combined['delta_x'] = delta_X
            combined['delta_y'] = delta_Y
            combined['delta_rot'] = delta_rot.arcsec  # Convert to arcseconds
        except Exception as e:
            # If pointing offset calculation fails, add NaN columns
            combined['delta_x'] = np.nan
            combined['delta_y'] = np.nan
            combined['delta_rot'] = np.nan
    elif combined is not None:
        # WCS not available, add NaN columns
        combined['delta_x'] = np.nan
        combined['delta_y'] = np.nan
        combined['delta_rot'] = np.nan
    
    # Set float64 print format to 2 significant figures
    if combined is not None:
        for col in combined.colnames:
            if combined[col].dtype == np.float64:
                combined[col].info.format = '.2g'
    
    return combined


def mk_roi_location_table(dayObs, seqNumlo, seqNumhi):
    """
    Create a table of ROI and star data across multiple exposures.
    
    This is a convenience wrapper around get_guider_data that collects
    comprehensive guider information for a range of exposures.
    
    Parameters
    ----------
    dayObs : int
        Observation day in YYYYMMDD format.
    seqNumlo : int
        Starting sequence number (inclusive).
    seqNumhi : int
        Ending sequence number (inclusive).
    
    Returns
    -------
    combined_table : astropy.table.Table or None
        Combined table with columns including:
        - Star catalog information (RA, Dec, magnitude, etc.)
        - Expected ROI location from catalog
        - Actual ROI location (ccdName, ampName, pixel coordinates)
        - Detected star measurements (centroid, flux, etc.)
        - Sequence number and split information
        Returns None if no valid data found.
    
    Examples
    --------
    >>> table = mk_roi_location_table(20240115, 100, 150)
    >>> if table is not None:
    ...     print(f"Found {len(table)} guide star measurements")
    """
    # Initialize GuiderROIs and Butler
    butler = butlerUtils.makeDefaultButler("LSSTCam")
    
    repo_name = "LSSTCam"
    catalog_dataset = "guider_roi_monster_guide_catalog"
    vignetting_dataset = "guider_roi_vignetting_correction"
    collection = "guider_roi_data"
    groi = GuiderROIs(
        catalog_name=catalog_dataset,
        vignetting_dataset=vignetting_dataset,
        collection=collection,
        repo_name=repo_name,
    )
    
    # Process each exposure
    tables = []
    for seqNum in tqdm(range(seqNumlo, seqNumhi + 1)):
        result = get_guider_data(butler, groi, dayObs, seqNum)
        
        if result is not None:
            tables.append(result)
    
    # Stack all tables
    if len(tables) > 0:
        combined_table = vstack(tables)
        return combined_table
    else:
        return None


def getmany_guiderstarcat(dayObs, seqNumLo, seqNumHi):
    """
    Get comprehensive guider data for multiple exposures.
    
    This is a convenience function that calls get_guider_data for a range
    of sequence numbers and collects the results.
    
    Parameters
    ----------
    dayObs : int
        Observation day in YYYYMMDD format.
    seqNumLo : int
        Starting sequence number (inclusive).
    seqNumHi : int
        Ending sequence number (exclusive).
    
    Returns
    -------
    combined_table : astropy.table.Table or None
        Combined table from all exposures containing catalog stars, ROI 
        locations, and detected star measurements. Returns None if no
        valid data found.
    
    Examples
    --------
    >>> result = getmany_guiderstarcat(20240115, 100, 150)
    >>> if result is not None:
    ...     print(f"Processed {len(result)} guide star measurements")
    ...     # Count exposures
    ...     n_exposures = len(set(result['expid']))
    ...     print(f"From {n_exposures} exposures")
    """
    butler = butlerUtils.makeDefaultButler("LSSTCam")
    
    # Initialize GuiderROIs
    repo_name = "LSSTCam"
    catalog_dataset = "guider_roi_monster_guide_catalog"
    vignetting_dataset = "guider_roi_vignetting_correction"
    collection = "guider_roi_data"
    groi = GuiderROIs(
        catalog_name=catalog_dataset,
        vignetting_dataset=vignetting_dataset,
        collection=collection,
        repo_name=repo_name,
    )
    
    all_tables = []
    
    for seqNum in tqdm(range(seqNumLo, seqNumHi)):
        result = get_guider_data(butler, groi, dayObs, seqNum)
        
        if result is not None:
            all_tables.append(result)
    
    # Stack all tables
    if len(all_tables) > 0:
        combined_table = vstack(all_tables)
        return combined_table
    else:
        return None
