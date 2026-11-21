"""Batoid optical models, defocus configurations and wavefront Zernikes for simulated donuts.

``optics`` loads a batoid model and danish obscuration file by path, builds the ts_wep
`Instrument` of a defocus configuration (camera, detector or camera plus M2 shifted
along z), makes the field grids and computes in-focus and defocused Zernikes.

Import as ``from rubinwork.donut_sim import optics``.
"""
