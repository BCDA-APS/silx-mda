# New matplotlib colormaps by Nathaniel J. Smith, Stefan van der Walt,
# and (in the case of viridis) Eric Firing.
#
# This file and the colormaps in it are released under the CC0 license /
# public domain dedication. We would appreciate credit if you use or
# redistribute these colormaps, but do not impose any legal restrictions.
#
# To the extent possible under law, the persons who associated CC0 with
# mpl-colormaps have waived all copyright and related or neighboring rights
# to mpl-colormaps.
#
# You should have received a copy of the CC0 legalcode along with this
# work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
"""Matplotlib's new colormaps"""

import numpy
import logging
from matplotlib.colors import ListedColormap
import matplotlib.colors
import matplotlib.cm
import silx.resources

_logger = logging.getLogger(__name__)

_AVAILABLE_AS_RESOURCE = ('magma', 'inferno', 'plasma', 'viridis')
"""List available colormap name as resources"""

_AVAILABLE_AS_BUILTINS = ('gray', 'reversed gray',
                          'temperature', 'red', 'green', 'blue')
"""List of colormaps available through built-in declarations"""

_CMAPS = {}
"""Cache colormaps"""


@property
def magma():
    return getColormap('magma')


@property
def inferno():
    return getColormap('inferno')


@property
def plasma():
    return getColormap('plasma')


@property
def viridis():
    return getColormap('viridis')


def getColormap(name):
    """Returns matplotlib colormap corresponding to given name

    :param str name: The name of the colormap
    :return: The corresponding colormap
    :rtype: matplolib.colors.Colormap
    """
    if not _CMAPS:  # Lazy initialization of own colormaps
        cdict = {'red': ((0.0, 0.0, 0.0),
                         (1.0, 1.0, 1.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 0.0, 0.0),
                          (1.0, 0.0, 0.0))}
        _CMAPS['red'] = matplotlib.colors.LinearSegmentedColormap(
            'red', cdict, 256)

        cdict = {'red': ((0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (1.0, 1.0, 1.0)),
                 'blue': ((0.0, 0.0, 0.0),
                          (1.0, 0.0, 0.0))}
        _CMAPS['green'] = matplotlib.colors.LinearSegmentedColormap(
            'green', cdict, 256)

        cdict = {'red': ((0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 0.0, 0.0),
                          (1.0, 1.0, 1.0))}
        _CMAPS['blue'] = matplotlib.colors.LinearSegmentedColormap(
            'blue', cdict, 256)

        # Temperature as defined in spslut
        cdict = {'red': ((0.0, 0.0, 0.0),
                         (0.5, 0.0, 0.0),
                         (0.75, 1.0, 1.0),
                         (1.0, 1.0, 1.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (0.25, 1.0, 1.0),
                           (0.75, 1.0, 1.0),
                           (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 1.0, 1.0),
                          (0.25, 1.0, 1.0),
                          (0.5, 0.0, 0.0),
                          (1.0, 0.0, 0.0))}
        # but limited to 256 colors for a faster display (of the colorbar)
        _CMAPS['temperature'] = \
            matplotlib.colors.LinearSegmentedColormap(
                'temperature', cdict, 256)

        # reversed gray
        cdict = {'red': ((0.0, 1.0, 1.0),
                         (1.0, 0.0, 0.0)),
                 'green': ((0.0, 1.0, 1.0),
                           (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 1.0, 1.0),
                          (1.0, 0.0, 0.0))}

        _CMAPS['reversed gray'] = \
            matplotlib.colors.LinearSegmentedColormap(
                'yerg', cdict, 256)

    if name in _CMAPS:
        return _CMAPS[name]
    elif name in _AVAILABLE_AS_RESOURCE:
        filename = silx.resources.resource_filename("gui/colormaps/%s.npy" % name)
        data = numpy.load(filename)
        lut = ListedColormap(data, name=name)
        _CMAPS[name] = lut
        return lut
    else:
        # matplotlib built-in
        return matplotlib.cm.get_cmap(name)


def getScalarMappable(colormap, data=None):
    """Returns matplotlib ScalarMappable corresponding to colormap

    :param :class:`.Colormap` colormap: The colormap to convert
    :param numpy.ndarray data:
        The data on which the colormap is applied.
        If provided, it is used to compute autoscale.
    :return: matplotlib object corresponding to colormap
    :rtype: matplotlib.cm.ScalarMappable
    """
    assert colormap is not None

    if colormap.getName() is not None:
        cmap = getColormap(colormap.getName())

    else:  # No name, use custom colors
        if colormap.getColorMapLUT() is None:
            raise ValueError(
                'addImage: colormap no name nor list of colors.')
        colors = colormap.getColorMapLUT()
        assert len(colors.shape) == 2
        assert colors.shape[-1] in (3, 4)
        if colors.dtype == numpy.uint8:
            # Convert to float in [0., 1.]
            colors = colors.astype(numpy.float32) / 255.
        cmap = matplotlib.colors.ListedColormap(colors)

    if colormap.getNormalization().startswith('log'):
        vmin, vmax = None, None
        if not colormap.isAutoscale():
            if colormap.getVMin() > 0.:
                vmin = colormap.getVMin()
            if colormap.getVMax() > 0.:
                vmax = colormap.getVMax()

            if vmin is None or vmax is None:
                _logger.warning('Log colormap with negative bounds, ' +
                                'changing bounds to positive ones.')
            elif vmin > vmax:
                _logger.warning('Colormap bounds are inverted.')
                vmin, vmax = vmax, vmin

        # Set unset/negative bounds to positive bounds
        if (vmin is None or vmax is None) and data is not None:
            finiteData = data[numpy.isfinite(data)]
            posData = finiteData[finiteData > 0]
            if vmax is None:
                # 1. as an ultimate fallback
                vmax = posData.max() if posData.size > 0 else 1.
            if vmin is None:
                vmin = posData.min() if posData.size > 0 else vmax
            if vmin > vmax:
                vmin = vmax

        norm = matplotlib.colors.LogNorm(vmin, vmax)

    else:  # Linear normalization
        if colormap.isAutoscale():
            if data is None:
                vmin, vmax = None, None
            else:
                finiteData = data[numpy.isfinite(data)]
                vmin = finiteData.min()
                vmax = finiteData.max()
        else:
            vmin = colormap.getVMin()
            vmax = colormap.getVMax()
            if vmin > vmax:
                _logger.warning('Colormap bounds are inverted.')
                vmin, vmax = vmax, vmin

        norm = matplotlib.colors.Normalize(vmin, vmax)

    return matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)


def applyColormapToData(data,
                        colormap):
    """Apply a colormap to the data and returns the RGBA image

    This supports data of any dimensions (not only of dimension 2).
    The returned array will have one more dimension (with 4 entries)
    than the input data to store the RGBA channels
    corresponding to each bin in the array.

    :param numpy.ndarray data: The data to convert.
    :param :class:`.Colormap`: The colormap to apply
    """
    # Debian 7 specific support
    # No transparent colormap with matplotlib < 1.2.0
    # Add support for transparent colormap for uint8 data with
    # colormap with 256 colors, linear norm, [0, 255] range
    if matplotlib.__version__ < '1.2.0':
        if (colormap.getName() is None and
                colormap.getColorMapLUT() is not None):
            colors = colormap.getColorMapLUT(copy=False)
            if (colors.shape[-1] == 4 and
                    not numpy.all(numpy.equal(colors[3], 255))):
                # This is a transparent colormap
                if (colors.shape == (256, 4) and
                        colormap.getNormalization() == 'linear' and
                        not colormap.isAutoscale() and
                        colormap.getVMin() == 0 and
                        colormap.getVMax() == 255 and
                        data.dtype == numpy.uint8):
                    # Supported case, convert data to RGBA
                    return colors[data.reshape(-1)].reshape(
                        data.shape + (4,))
                else:
                    _logger.warning(
                        'matplotlib %s does not support transparent '
                        'colormap.', matplotlib.__version__)

    scalarMappable = getScalarMappable(colormap, data)
    rgbaImage = scalarMappable.to_rgba(data, bytes=True)

    return rgbaImage


def getSupportedColormaps():
    """Get the supported colormap names as a tuple of str.
    """
    colormaps = set(matplotlib.cm.datad.keys())
    colormaps.update(_AVAILABLE_AS_BUILTINS)
    colormaps.update(_AVAILABLE_AS_RESOURCE)
    return tuple(sorted(colormaps))
