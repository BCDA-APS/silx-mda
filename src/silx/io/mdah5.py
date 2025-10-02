# /*##########################################################################
# Copyright (C) 2016-2023 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ############################################################################*/
"""This module provides functions to read MDA files as an HDF5-like object.

    >>> import silx.io.mdah5
    >>> f = silx.io.mdah5.File("scan.mda")

.. note:: This module has a dependency on the `mda` library,
    which is not a mandatory dependency for `silx`.

"""

import logging
import os
import numpy

from . import commonh5
from silx import version as silx_version

_logger = logging.getLogger(__name__)

# Import the MDA reader
try:
    # Try to import from the local simple_mda.py file first
    import sys
    import os

    # Try standard import
    from silx.synApps_mdalib import mda

    _MDA_AVAILABLE = True
except ImportError:
    _MDA_AVAILABLE = False
    _logger.warning("MDA module not available. MDA file support disabled.")


class MDADataset(commonh5.LazyLoadableDataset):
    """Expose MDA detector/positioner data as an HDF5-like dataset."""

    def __init__(self, name, data, attrs=None, parent=None):
        """Constructor

        :param str name: Name of the dataset
        :param data: MDA data (list or numpy array)
        :param dict attrs: Attributes dictionary
        :param parent: Parent group
        """
        super().__init__(name, parent=parent)
        self._data = data
        self._attrs = attrs or {}

    def _get_data(self):
        """Return the actual data"""
        return self._data

    @property
    def attrs(self):
        """Return attributes dictionary"""
        return self._attrs

    @property
    def shape(self):
        """Return shape of the data"""
        if hasattr(self._data, "shape"):
            return self._data.shape
        elif isinstance(self._data, list):
            # Handle nested lists (multi-dimensional data)
            if len(self._data) > 0 and isinstance(self._data[0], list):
                # Check for deeper nesting (3D+ data)
                if len(self._data[0]) > 0 and isinstance(self._data[0][0], list):
                    # 3D data: [depth][height][width]
                    return (len(self._data), len(self._data[0]), len(self._data[0][0]))
                else:
                    # 2D data: [height][width]
                    return (len(self._data), len(self._data[0]))
            else:
                return (len(self._data),)
        else:
            return (1,)

    @property
    def dtype(self):
        """Return data type as numpy dtype"""
        import numpy as np

        # Convert data to numpy array if it's a list to get proper dtype
        if isinstance(self._data, list):
            try:
                data = np.array(self._data, dtype=np.float64)
                return data.dtype
            except (ValueError, TypeError):
                # If direct conversion fails, try to convert each row
                data = np.array([np.array(row, dtype=np.float64) for row in self._data])
                return data.dtype
        elif hasattr(self._data, "dtype"):
            return self._data.dtype
        else:
            # Convert Python type to numpy dtype
            return np.dtype(type(self._data))

    @property
    def size(self):
        """Return total number of elements in the dataset"""
        if hasattr(self._data, "size"):
            return self._data.size
        elif isinstance(self._data, list):
            # Calculate size for nested lists
            if len(self._data) > 0 and isinstance(self._data[0], list):
                return len(self._data) * len(self._data[0])
            else:
                return len(self._data)
        else:
            return 1

    @property
    def compression(self):
        """Return compression information (MDA files are not compressed)"""
        return None

    @property
    def ndim(self):
        """Return number of dimensions"""
        return len(self.shape)

    @property
    def nbytes(self):
        """Return number of bytes"""
        return self.size * self.dtype.itemsize

    @property
    def chunks(self):
        """Return chunk information (MDA files are not chunked)"""
        return None

    @property
    def compression_opts(self):
        """Return compression options (MDA files are not compressed)"""
        return None

    @property
    def maxshape(self):
        """Return maximum shape (same as shape for MDA data)"""
        return self.shape

    @property
    def fillvalue(self):
        """Return fill value (not applicable for MDA data)"""
        return None

    @property
    def fletcher32(self):
        """Return fletcher32 checksum flag (not applicable for MDA data)"""
        return False

    @property
    def shuffle(self):
        """Return shuffle filter flag (not applicable for MDA data)"""
        return False

    @property
    def scaleoffset(self):
        """Return scaleoffset filter (not applicable for MDA data)"""
        return None

    @property
    def track_times(self):
        """Return track times flag (not applicable for MDA data)"""
        return False

    @property
    def track_order(self):
        """Return track order flag (not applicable for MDA data)"""
        return False

    @property
    def dims(self):
        """Return dimension scales (empty for MDA datasets)"""

        # Create a mock dims object like h5py
        class MockDims:
            def __init__(self, dataset):
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset.shape)

            def __getitem__(self, key):
                return []

            def __iter__(self):
                return iter([] for _ in range(len(self.dataset.shape)))

        return MockDims(self)

    @property
    def is_scale(self):
        """Return whether this dataset is a dimension scale"""
        return False

    @property
    def ref(self):
        """Return HDF5 object reference (not supported for MDA)"""

        # Create a mock reference
        class MockRef:
            pass

        return MockRef()

    @property
    def regionref(self):
        """Return region reference proxy (not supported for MDA)"""

        # Create a mock region reference proxy
        class MockRegionRef:
            pass

        return MockRegionRef()

    @property
    def external(self):
        """Return external file information (MDA files are not external)"""
        return None

    @property
    def is_virtual(self):
        """Return whether this is a virtual dataset (MDA files are not virtual)"""
        return False

    def __bool__(self):
        """Return boolean value for the dataset"""
        try:
            data = self[()]
            if hasattr(data, "size"):
                # For numpy arrays, return True if not empty
                return data.size > 0
            else:
                # For scalar values
                return bool(data)
        except:
            # If we can't access the data, assume it exists
            return True

    @property
    def virtual_sources(self):
        """Return virtual sources (MDA files are not virtual)"""
        return None

    @property
    def value(self):
        """Return the data value (alias for data access)"""
        return self[()]

    @property
    def filename(self):
        """Return the filename of the file containing this dataset"""
        # Simple approach: return the filename from the parent file if available
        if hasattr(self, "parent") and self.parent is not None:
            if hasattr(self.parent, "filename"):
                return self.parent.filename
        return "mda_0001.mda"  # Fallback filename

    @property
    def id(self):
        """Return a unique identifier for this dataset"""
        return id(self)

    def __getattr__(self, name):
        """Catch-all for any missing attributes to prevent AttributeError"""
        # Log the missing attribute for debugging
        import traceback

        # Try to provide some common missing attributes
        if name == "attrs":
            return {}
        elif name == "dtype":
            return self.dtype
        elif name == "shape":
            return self.shape
        elif name == "size":
            return self.size
        elif name == "ndim":
            return self.ndim
        elif name == "nbytes":
            return self.nbytes
        elif name == "compression":
            return self.compression
        elif name == "chunks":
            return self.chunks
        elif name == "compression_opts":
            return self.compression_opts
        elif name == "external":
            return self.external
        elif name == "is_virtual":
            return self.is_virtual
        elif name == "virtual_sources":
            return self.virtual_sources
        elif name == "value":
            return self.value
        elif name == "filename":
            return self.filename
        elif name == "id":
            return self.id
        elif name == "file":
            return self.file
        elif name == "name":
            return self.name
        elif name == "maxshape":
            return self.maxshape
        elif name == "fillvalue":
            return self.fillvalue
        elif name == "fletcher32":
            return self.fletcher32
        elif name == "shuffle":
            return self.shuffle
        elif name == "scaleoffset":
            return self.scaleoffset
        elif name == "track_times":
            return self.track_times
        elif name == "track_order":
            return self.track_order
        elif name == "dims":
            return self.dims
        elif name == "is_scale":
            return self.is_scale
        elif name == "ref":
            return self.ref
        elif name == "regionref":
            return self.regionref

        # For any other missing attributes, raise AttributeError to be more explicit
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @property
    def h5_class(self):
        """Returns the HDF5 class which is mimicked by this class."""
        from silx.io.utils import H5Type

        return H5Type.DATASET

    @property
    def h5Class(self):
        """Returns the HDF5 class which is mimicked by this class (GUI compatibility)."""
        from silx.io.utils import H5Type

        return H5Type.DATASET

    @property
    def name(self):
        """Return the name of the dataset"""
        name = super().name
        if name is None:
            # Fallback to basename if name is None
            return self.basename
        return name

    @name.setter
    def name(self, value):
        """Set the name of the dataset (for GUI compatibility)"""
        # For GUI compatibility, we need to allow setting the name
        # This is a hack to make the GUI work
        pass

    def __getitem__(self, key):
        """Access data elements, compatible with HDF5 dataset access"""
        import numpy as np

        # Convert data to numpy array if it's a list
        if isinstance(self._data, list):
            # Ensure proper numeric conversion for 2D data
            try:
                data = np.array(self._data, dtype=np.float64)
            except (ValueError, TypeError):
                # If direct conversion fails, try to convert each row
                data = np.array([np.array(row, dtype=np.float64) for row in self._data])
        else:
            data = self._data

        # Ensure data is contiguous for plotting
        if hasattr(data, "flags") and not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)

        # Handle different key types
        if key == ():  # Empty tuple - return all data
            return data
        elif key is Ellipsis:  # Ellipsis - return all data
            return data
        elif isinstance(key, str):
            # If key is a string, it's probably a path - return None or raise KeyError
            raise KeyError(f"'{key}' not found in dataset '{self.name}'")
        else:
            try:
                # Use numpy indexing
                return data[key]
            except (IndexError, TypeError) as e:
                # If indexing fails, it might be an invalid key type
                raise KeyError(f"Invalid key '{key}' for dataset '{self.name}': {e}")

    def __len__(self):
        """Return the length of the first dimension"""
        if hasattr(self._data, "__len__"):
            return len(self._data)
        else:
            return 1

    def __array__(self, dtype=None):
        """Convert dataset to numpy array for numpy operations"""
        import numpy as np

        # Get the data using __getitem__ which already handles conversion
        data = self[()]

        # Convert to requested dtype if specified
        if dtype is not None:
            return np.asarray(data, dtype=dtype)
        return np.asarray(data)

    def read_direct(self, dest, source_sel=None, dest_sel=None):
        """Read data directly into a destination array (compatible with HDF5)"""
        import numpy as np

        # Convert data to numpy array if it's a list
        if isinstance(self._data, list):
            # Ensure proper numeric conversion for 2D data
            try:
                data = np.array(self._data, dtype=np.float64)
            except (ValueError, TypeError):
                # If direct conversion fails, try to convert each row
                data = np.array([np.array(row, dtype=np.float64) for row in self._data])
        else:
            data = self._data

        # If no source selection, return all data
        if source_sel is None:
            if dest_sel is None:
                dest[:] = data
            else:
                dest[dest_sel] = data[dest_sel]
        else:
            if dest_sel is None:
                dest[:] = data[source_sel]
            else:
                dest[dest_sel] = data[source_sel]

    @property
    def file(self):
        """Return the file object (for GUI compatibility)"""
        # Navigate up to find the file object
        current = self
        while current is not None:
            if isinstance(current, MDAFile):
                return current
            current = getattr(current, "parent", None)
        return None


class MDAGroup(commonh5.Group):
    """Expose MDA scan dimension as an HDF5-like group."""

    def __init__(self, name, scan_dim, parent=None):
        """Constructor

        :param str name: Name of the group
        :param scan_dim: MDA scan dimension object
        :param parent: Parent group
        """
        super().__init__(name, parent=parent)
        self._scan_dim = scan_dim
        self._children = {}
        self._attrs_dict = {}  # Custom attributes dictionary
        self._load_children()

    def _load_children(self):
        """Load child groups and datasets"""
        # Skip if no scan dimension (e.g., for environment group)
        if self._scan_dim is None:
            return

        # Add positioners as datasets
        for i, pos in enumerate(self._scan_dim.p):
            dataset_name = f"positioner_{i:02d}_{pos.fieldName}"
            attrs = {
                "name": pos.name,
                "description": pos.desc,
                "unit": pos.unit,
                "step_mode": pos.step_mode,
                "readback_name": pos.readback_name,
                "readback_description": pos.readback_desc,
                "readback_unit": pos.readback_unit,
            }
            dataset = MDADataset(dataset_name, pos.data, attrs, parent=self)
            self._children[dataset_name] = dataset

        # Add detectors as datasets with axis information
        for i, det in enumerate(self._scan_dim.d):
            dataset_name = f"detector_{i:02d}_{det.fieldName}"
            attrs = {
                "name": det.name,
                "description": det.desc,
                "unit": det.unit,
            }

            # Add axis information for proper plotting
            if len(self._scan_dim.p) > 0 and len(det.data) > 0:
                # Determine data dimensionality
                data_dim = 1
                if isinstance(det.data, list) and len(det.data) > 0:
                    if isinstance(det.data[0], list):
                        data_dim = 2
                        if len(det.data[0]) > 0 and isinstance(det.data[0][0], list):
                            data_dim = 3

                # Add axis information
                if data_dim == 1:
                    attrs["interpretation"] = "spectrum"
                    # For 1D data, add the positioner as an axis
                    if len(self._scan_dim.p) > 0:
                        pos = self._scan_dim.p[0]
                        axis_name = f"axis_{pos.number:02d}_{pos.fieldName}"
                        attrs["axes"] = axis_name
                elif data_dim == 2:
                    attrs["interpretation"] = "image"
                    # For 2D data, add both positioners as axes
                    if len(self._scan_dim.p) >= 2:
                        axes_list = []
                        for j in range(2):
                            if j < len(self._scan_dim.p):
                                pos = self._scan_dim.p[j]
                                axis_name = f"axis_{pos.number:02d}_{pos.fieldName}"
                                axes_list.append(axis_name)
                            else:
                                axes_list.append(".")
                        attrs["axes"] = axes_list
                elif data_dim == 3:
                    attrs["interpretation"] = "image"
                    # For 3D data, add all positioners as axes
                    if len(self._scan_dim.p) >= 3:
                        axes_list = []
                        for j in range(3):
                            if j < len(self._scan_dim.p):
                                pos = self._scan_dim.p[j]
                                axis_name = f"axis_{pos.number:02d}_{pos.fieldName}"
                                axes_list.append(axis_name)
                            else:
                                axes_list.append(".")
                        attrs["axes"] = axes_list

            dataset = MDADataset(dataset_name, det.data, attrs, parent=self)
            self._children[dataset_name] = dataset

        # Create axis datasets for positioners
        for i, pos in enumerate(self._scan_dim.p):
            axis_name = f"axis_{pos.number:02d}_{pos.fieldName}"
            axis_attrs = {
                "name": pos.name,
                "description": pos.desc,
                "unit": pos.unit,
            }

            # For 2D data, we need to create 1D axis datasets
            # The positioner data should be 1D, not 2D
            if len(pos.data) > 0 and isinstance(pos.data[0], list):
                # This is 2D positioner data, we need to extract the 1D coordinate
                # For 2D scans, we need to find the coordinate that varies along each dimension
                # Check if all rows are identical (1D scan) or if rows vary (2D scan)
                first_row = pos.data[0]
                all_rows_same = all(row == first_row for row in pos.data)

                if all_rows_same:
                    # This is a 1D scan - use the first row as the coordinate
                    axis_data = first_row
                else:
                    # This is a 2D scan - extract the coordinate that varies
                    # For the first axis, we want the coordinate that varies along rows
                    # For the second axis, we want the coordinate that varies along columns
                    if i == 0:  # First axis - extract first column (varies along rows)
                        axis_data = [row[0] for row in pos.data]
                    else:  # Second axis - extract first row (varies along columns)
                        axis_data = pos.data[0]
            else:
                # This is already 1D data
                axis_data = pos.data

            axis_dataset = MDADataset(axis_name, axis_data, axis_attrs, parent=self)
            self._children[axis_name] = axis_dataset

        # Create NXdata groups for each detector with proper axis mapping
        for i, det in enumerate(self._scan_dim.d):
            if len(self._scan_dim.p) > 0 and len(det.data) > 0:
                # Determine data dimensionality
                data_dim = 1
                if isinstance(det.data, list) and len(det.data) > 0:
                    if isinstance(det.data[0], list):
                        data_dim = 2
                        if len(det.data[0]) > 0 and isinstance(det.data[0][0], list):
                            data_dim = 3

                # Create NXdata group for this detector
                nxdata_name = f"nxdata_{i:02d}_{det.fieldName}"
                nxdata_group = MDAGroup(nxdata_name, None, parent=self)

                # Add signal dataset
                signal_attrs = {
                    "name": det.name,
                    "description": det.desc,
                    "unit": det.unit,
                }
                if data_dim == 1:
                    signal_attrs["interpretation"] = "spectrum"
                elif data_dim == 2:
                    signal_attrs["interpretation"] = "image"
                elif data_dim == 3:
                    signal_attrs["interpretation"] = "image"

                signal_dataset = MDADataset(
                    "signal", det.data, signal_attrs, parent=nxdata_group
                )
                nxdata_group._children["signal"] = signal_dataset

                # Add axis datasets (1D only for NXdata)
                axes_names = []
                for j in range(data_dim):
                    if j < len(self._scan_dim.p):
                        # For 2D data, we need to use positioners from the correct dimensions
                        # Y-axis (j=0): use dimension_1 positioner (282-582)
                        # X-axis (j=1): use dimension_2 positioner (1.92-2.08)
                        if data_dim == 2:
                            # Get the root file to access other dimensions
                            root_file = self
                            while (
                                hasattr(root_file, "parent")
                                and root_file.parent is not None
                            ):
                                root_file = root_file.parent

                            if j == 0:
                                # Use dimension_1 positioner for Y-axis
                                try:
                                    dim1 = root_file["dimension_1"]
                                    if len(dim1._scan_dim.p) > 0:
                                        pos = dim1._scan_dim.p[
                                            0
                                        ]  # Use first positioner from dimension_1
                                    else:
                                        pos = self._scan_dim.p[j]  # Fallback
                                except:
                                    pos = self._scan_dim.p[j]  # Fallback
                            else:
                                # Use first positioner for X-axis (P1 with range 1.920-2.080)
                                if len(self._scan_dim.p) > 0:
                                    pos = self._scan_dim.p[
                                        0
                                    ]  # Use first positioner (P1)
                                else:
                                    pos = self._scan_dim.p[j]  # Fallback
                        else:
                            pos = self._scan_dim.p[j]
                        axis_name = f"axis_{j}"
                        axes_names.append(axis_name)

                        # Create 1D axis data for NXdata
                        if len(pos.data) > 0 and isinstance(pos.data[0], list):
                            # This is 2D positioner data, extract 1D coordinate
                            # Check if all rows are identical (1D scan) or if rows vary (2D scan)
                            first_row = pos.data[0]
                            all_rows_same = all(row == first_row for row in pos.data)

                            if all_rows_same:
                                # This is a 1D scan - use the first row as the coordinate
                                axis_data = first_row
                            else:
                                # This is a 2D scan - extract the coordinate that varies
                                if (
                                    j == 0
                                ):  # First axis - extract first column (varies along rows)
                                    axis_data = [row[0] for row in pos.data]
                                else:  # Second axis - extract first row (varies along columns)
                                    axis_data = pos.data[0]
                        else:
                            # This is already 1D data
                            axis_data = pos.data

                        # Ensure axis data is linear for proper calibration
                        if len(axis_data) > 1:
                            import numpy as np

                            axis_array = np.asarray(axis_data)
                            # Create a linear version if the data is not perfectly linear
                            if not np.allclose(
                                np.diff(axis_array), np.diff(axis_array)[0], rtol=1e-6
                            ):
                                axis_data = np.linspace(
                                    axis_array.min(), axis_array.max(), len(axis_array)
                                )

                        axis_attrs = {
                            "name": pos.name,
                            "description": pos.desc,
                            "unit": pos.unit,
                        }
                        axis_dataset = MDADataset(
                            axis_name, axis_data, axis_attrs, parent=nxdata_group
                        )
                        nxdata_group._children[axis_name] = axis_dataset
                    else:
                        axes_names.append(".")

                # Set NXdata attributes
                nxdata_group._attrs_dict = {
                    "NX_class": "NXdata",
                    "signal": "signal",
                    "axes": axes_names,
                }

                self._children[nxdata_name] = nxdata_group

        # Set the first NXdata group as the default
        if len(self._scan_dim.d) > 0:
            first_detector = self._scan_dim.d[0]
            default_nxdata_name = f"nxdata_00_{first_detector.fieldName}"
            if default_nxdata_name in self._children:
                self._attrs_dict["default"] = default_nxdata_name

        # Add metadata as attributes (only if scan_dim exists)
        if self._scan_dim is not None:
            for key, value in {
                "scan_name": self._scan_dim.name,
                "scan_time": self._scan_dim.time,
                "rank": self._scan_dim.rank,
                "dimension": self._scan_dim.dim,
                "npts": self._scan_dim.npts,
                "curr_pt": self._scan_dim.curr_pt,
                "num_positioners": self._scan_dim.np,
                "num_detectors": self._scan_dim.nd,
            }.items():
                self._attrs_dict[key] = value

    def _get_items(self):
        """Returns the child items as a name-node dictionary."""
        return self._children

    def __getitem__(self, key):
        """Get child by name, supporting full paths"""
        # Handle full paths like "/dimension_1/detector_07_D08"
        if key.startswith("/"):
            # Remove leading slash and split path
            path_parts = key[1:].split("/")
            if len(path_parts) == 1:
                # Direct child access
                return self._children[path_parts[0]]
            else:
                # Navigate through the path
                current = self
                for part in path_parts:
                    current = current[part]
                return current
        else:
            # Direct key access
            if key in self._children:
                return self._children[key]
            else:
                raise KeyError(f"'{key}' not found in group '{self.name}'")

    def get(self, key, default=None, getclass=False, getlink=False):
        """Return child object by key, compatible with HDF5 get method"""

        # Handle full paths like "/dimension_1/detector_04_D05"
        if key.startswith("/"):
            # Remove leading slash and split path
            path_parts = key[1:].split("/")
            if len(path_parts) == 1:
                # Direct child access
                result = self._children.get(path_parts[0], default)
            else:
                # Navigate through the path, but start from the root if needed
                # If the first part of the path matches our name, skip it
                if path_parts[0] == self.basename:
                    # We're already at the right level, skip the first part
                    path_parts = path_parts[1:]
                    current = self
                else:
                    # Start from the root
                    current = self
                    while hasattr(current, "parent") and current.parent is not None:
                        current = current.parent

                # Navigate through the remaining path
                for part in path_parts:
                    if hasattr(current, "__getitem__"):
                        current = current[part]
                    else:
                        result = default
                        break
                else:
                    result = current
        else:
            # Direct key access
            if getlink:
                # For now, just return the object (we don't support links)
                result = self._children.get(key, default)
            else:
                result = self._children.get(key, default)

        # Handle getlink and getclass parameters
        if getlink and getclass and result is not None:
            # When both getlink=True and getclass=True, return the h5py.HardLink class
            import h5py

            return h5py.HardLink
        elif getlink and result is not None:
            # When only getlink=True, return an h5py.HardLink instance
            import h5py

            return h5py.HardLink()
        elif getclass and result is not None:
            # When only getclass=True, return the actual Python class
            return result.__class__

        return result

    def keys(self):
        """Return list of child names"""
        return list(self._children.keys())

    def values(self):
        """Return list of child objects"""
        return list(self._children.values())

    def items(self):
        """Return list of (name, object) pairs"""
        return list(self._children.items())

    def __iter__(self):
        """Iterate over child names"""
        return iter(self._children.keys())

    def __len__(self):
        """Return number of children"""
        return len(self._children)

    @property
    def attrs(self):
        """Return attributes dictionary"""
        return self._attrs_dict

    @property
    def h5_class(self):
        """Returns the HDF5 class which is mimicked by this class."""
        from silx.io.utils import H5Type

        return H5Type.GROUP

    @property
    def h5Class(self):
        """Returns the HDF5 class which is mimicked by this class (GUI compatibility)."""
        from silx.io.utils import H5Type

        return H5Type.GROUP

    @property
    def name(self):
        """Return the name of the group"""
        name = super().name
        if name is None:
            # Fallback to basename if name is None
            return self.basename
        return name

    @name.setter
    def name(self, value):
        """Set the name of the group (for GUI compatibility)"""
        # For GUI compatibility, we need to allow setting the name
        # This is a hack to make the GUI work
        pass

    @property
    def file(self):
        """Return the file object (for GUI compatibility)"""
        # Navigate up to find the file object
        current = self
        while current is not None:
            if isinstance(current, MDAFile):
                return current
            current = getattr(current, "parent", None)
        return None


class MDAFile(commonh5.File):
    """Expose MDA file as an HDF5-like file object."""

    def __init__(self, filename, mode="r"):
        """Constructor

        :param str filename: Path to MDA file
        :param str mode: File mode (only 'r' supported)
        """
        if not _MDA_AVAILABLE:
            raise ImportError("MDA module not available")

        if mode != "r":
            raise ValueError("MDA files are read-only")

        if not os.path.isfile(filename):
            raise FileNotFoundError(f"MDA file not found: {filename}")

        super().__init__(filename, mode=mode)

        # Load MDA data
        self._mda_data = mda.readMDA(filename, maxdim=4, verbose=0)
        if not self._mda_data:
            raise ValueError(f"Failed to read MDA file: {filename}")

        self._children = {}
        self._attrs_dict = {}  # Custom attributes dictionary
        self._load_structure()

    def _load_structure(self):
        """Load the MDA file structure as HDF5-like groups and datasets"""
        # Add file-level metadata
        for key, value in {
            "filename": self._mda_data[0]["filename"],
            "version": self._mda_data[0]["version"],
            "scan_number": self._mda_data[0]["scan_number"],
            "rank": self._mda_data[0]["rank"],
            "dimensions": self._mda_data[0]["dimensions"],
            "acquired_dimensions": self._mda_data[0]["acquired_dimensions"],
            "isRegular": self._mda_data[0]["isRegular"],
        }.items():
            self._attrs_dict[key] = value

        # Add scan environment variables as a group
        env_group = MDAGroup("scan_environment", None, parent=self)
        for key, value in self._mda_data[0].items():
            if key not in [
                "filename",
                "version",
                "scan_number",
                "rank",
                "dimensions",
                "acquired_dimensions",
                "isRegular",
                "ourKeys",
            ]:
                # Create dataset for each environment variable
                if isinstance(value, tuple) and len(value) == 5:
                    desc, unit, val, epics_type, count = value
                    attrs = {
                        "description": desc,
                        "unit": unit,
                        "epics_type": epics_type,
                        "count": count,
                    }
                    # Ensure key is a string, not bytes
                    key_str = key if isinstance(key, str) else key.decode("utf-8")
                    dataset = MDADataset(key_str, val, attrs, parent=env_group)
                    env_group._children[key_str] = dataset
        self._children["scan_environment"] = env_group

        # Add scan dimensions as groups
        for i in range(1, len(self._mda_data)):
            scan_dim = self._mda_data[i]
            group_name = f"dimension_{i}"
            group = MDAGroup(group_name, scan_dim, parent=self)
            self._children[group_name] = group

    def __getitem__(self, key):
        """Get child by name, supporting full paths"""
        # Handle full paths like "/dimension_1/detector_07_D08"
        if key.startswith("/"):
            # Remove leading slash and split path
            path_parts = key[1:].split("/")
            if len(path_parts) == 1:
                # Direct child access
                if path_parts[0] in self._children:
                    return self._children[path_parts[0]]
                else:
                    raise KeyError(
                        f"'{path_parts[0]}' not found in file '{self.filename}'"
                    )
            else:
                # Navigate through the path
                current = self
                for part in path_parts:
                    if hasattr(current, "__getitem__"):
                        current = current[part]
                    else:
                        raise KeyError(f"'{part}' not found in path '{key}'")
                return current
        else:
            # Direct key access
            if key in self._children:
                return self._children[key]
            else:
                raise KeyError(f"'{key}' not found in file '{self.filename}'")

    def get(self, key, default=None, getclass=False, getlink=False):
        """Return child object by key, compatible with HDF5 get method"""

        # Handle full paths like "/dimension_1/detector_04_D05"
        if key.startswith("/"):
            # Remove leading slash and split path
            path_parts = key[1:].split("/")
            if len(path_parts) == 1:
                # Direct child access
                result = self._children.get(path_parts[0], default)
            else:
                # Navigate through the path, but start from the root if needed
                # If the first part of the path matches our name, skip it
                if path_parts[0] == self.basename:
                    # We're already at the right level, skip the first part
                    path_parts = path_parts[1:]
                    current = self
                else:
                    # Start from the root
                    current = self
                    while hasattr(current, "parent") and current.parent is not None:
                        current = current.parent

                # Navigate through the remaining path
                for part in path_parts:
                    if hasattr(current, "__getitem__"):
                        current = current[part]
                    else:
                        result = default
                        break
                else:
                    result = current
        else:
            # Direct key access
            if getlink:
                # For now, just return the object (we don't support links)
                result = self._children.get(key, default)
            else:
                result = self._children.get(key, default)

        # Handle getlink and getclass parameters
        if getlink and getclass and result is not None:
            # When both getlink=True and getclass=True, return the h5py.HardLink class
            import h5py

            return h5py.HardLink
        elif getlink and result is not None:
            # When only getlink=True, return an h5py.HardLink instance
            import h5py

            return h5py.HardLink()
        elif getclass and result is not None:
            # When only getclass=True, return the actual Python class
            return result.__class__

        return result

    def keys(self):
        """Return list of child names"""
        return list(self._children.keys())

    def values(self):
        """Return list of child objects"""
        return list(self._children.values())

    def items(self):
        """Return list of (name, object) pairs"""
        return list(self._children.items())

    def __iter__(self):
        """Iterate over child names"""
        return iter(self._children.keys())

    def __len__(self):
        """Return number of children"""
        return len(self._children)

    @property
    def attrs(self):
        """Return attributes dictionary"""
        return self._attrs_dict

    @property
    def h5_class(self):
        """Returns the HDF5 class which is mimicked by this class."""
        from silx.io.utils import H5Type

        return H5Type.FILE

    @property
    def h5Class(self):
        """Returns the HDF5 class which is mimicked by this class (GUI compatibility)."""
        from silx.io.utils import H5Type

        return H5Type.FILE

    @property
    def name(self):
        """Return the name of the file"""
        name = super().name
        if name is None:
            # Fallback to basename if name is None
            return self.basename
        return name

    @name.setter
    def name(self, value):
        """Set the name of the file (for GUI compatibility)"""
        # For GUI compatibility, we need to allow setting the name
        # This is a hack to make the GUI work
        pass

    def close(self):
        """Close the file"""
        self._mda_data = None
        self._children = {}
        self._attrs_dict = {}


def File(filename, mode="r"):
    """Open an MDA file as an HDF5-like object.

    :param str filename: Path to MDA file
    :param str mode: File mode (only 'r' supported)
    :returns: MDAFile object
    """
    return MDAFile(filename, mode)


def supported_extensions():
    """Returns all extensions supported by MDA.

    :returns: A set containing extensions like "*.mda".
    :rtype: Set[str]
    """
    if not _MDA_AVAILABLE:
        return set()
    return {"*.mda"}
