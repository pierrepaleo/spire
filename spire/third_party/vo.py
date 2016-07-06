# Copyright 2014 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2014 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

# Author: Mark Basham <scientificsoftware@diamond.ac.uk>

#
# Version extracted from SAVU and modified for standalone working
# https://github.com/DiamondLightSource/Savu
#



import scipy.ndimage as ndi
import math
import logging
import numpy as np
import pyfftw.interfaces.scipy_fftpack as fft
import scipy.ndimage.filters as filter



class VoCentering(object):
    """
    A plugin to calculate the centre of rotation using the Vo Method

    :param ratio: The ratio between the size of object and FOV of \
        the camera. Default: 2.0.
    :param row_drop: Drop lines around vertical center of the \
        mask. Default: 20.
    :param search_radius: Use for fine searching. Default: 3.
    :param step: Step of fine searching. Default: 0.2.
    :param downsample: The step length over the rotation axis. Default: 1.
    :param preview: A slice list of required frames. Default: [].
    :param no_clean: Do not clean up potential outliers. Default: True.
    :param datasets_to_populate: A list of datasets which require this \
        information. Default: [].
    :param out_datasets: The default names. Default: ['cor_raw','cor_fit'].
    :param poly_degree: The polynomial degree of the fit \
        (1 or 0 = gradient or no gradient). Default: 0.
    :param start_pixel: The approximate centre. If value is None, take the \
        value from .nxs file else set to image centre. Default: None.
    :param search_area: Search area from horizontal approximate centre of the \
        image. Default: (-50, 50).
    """

    def __init__(self, params=None):
        if params is None: params = []
        default_params = {
            'ratio': 2.0,
            'row_drop': 20,
            'search_radius': 3,
            'step': 0.2,
            'downsample': 1,
            'preview': [],
            'no_clean': True,
            'datasets_to_populate': [],
            'our_datasets': ['cor_raw', 'cor_fit'],
            'poly_degree': 0,
            'start_pixel': None,
            'search_area': (-50, 50)
            }

        self.parameters = {}
        for par_name, par_val in default_params.items():
            if par_name in params:
                self.parameters[par_name] = params[par_name]
            else:
                self.parameters[par_name] = par_val

        #
        logr = logging.getLogger()
        logr.setLevel(10)
        #

    def _create_mask(self, Nrow, Ncol, obj_radius):
        du = 1.0/Ncol
        dv = (Nrow-1.0)/(Nrow*2.0*math.pi)
        cen_row = np.ceil(Nrow/2)-1
        cen_col = np.ceil(Ncol/2)-1
        drop = self.parameters['row_drop']
        mask = np.zeros((Nrow, Ncol), dtype=np.float32)
        for i in range(Nrow):
            num1 = \
                np.round(((i-cen_row)*dv/obj_radius)/du)
            (p1, p2) = \
                np.clip(np.sort((-num1+cen_col, num1+cen_col)), 0, Ncol-1)
            mask[i, p1:p2+1] = np.ones(p2-p1+1, dtype=np.float32)
        if drop < cen_row:
            mask[cen_row-drop:cen_row+drop+1, :] = \
                np.zeros((2*drop + 1, Ncol), dtype=np.float32)
        return mask


    def _get_start_shift(self, centre):
        if self.parameters['start_pixel'] is not None:
            shift = centre - self.parameters['start_pixel']
        else:
            shift = 0
        return int(shift)


    def _coarse_search(self, sino):
        # search minsearch to maxsearch in 1 pixel steps
        smin, smax = self.parameters['search_area']
        logging.debug("SMIN and SMAX %d %d", smin, smax)
        (Nrow, Ncol) = sino.shape
        centre_fliplr = (Ncol - 1.0)/2.0
        # check angles here to determine if a sinogram should be chopped off.
        # Copy the sinogram and flip left right, the purpose is to make a full
        # [0;2Pi] sinogram
        sino2 = np.fliplr(sino[1:])
        # This image is used for compensating the shift of sino2
        compensateimage = np.zeros((Nrow-1, Ncol), dtype=np.float32)
        # Start coarse search in which the shift step is 1
        compensateimage[:] = sino[-1]
        start_shift = self._get_start_shift(centre_fliplr)*2
        list_shift = np.arange(smin, smax + 1)*2 - start_shift
        logging.debug("%s", list_shift)
        list_metric = np.zeros(len(list_shift), dtype=np.float32)
        mask = self._create_mask(2*Nrow-1, Ncol,
                                 0.5*self.parameters['ratio']*Ncol)

        count = 0
        for i in list_shift:
            logging.debug("list shift %d", i)
            sino2a = np.roll(sino2, i, axis=1)
            if i >= 0:
                sino2a[:, 0:i] = compensateimage[:, 0:i]
            else:
                sino2a[:, i:] = compensateimage[:, i:]
            list_metric[count] = np.sum(
                np.abs(fft.fftshift(fft.fft2(np.vstack((sino, sino2a)))))*mask)
            count += 1
        minpos = np.argmin(list_metric)
        rot_centre = centre_fliplr + list_shift[minpos]/2.0
        return rot_centre, list_metric


    def _fine_search(self, sino, raw_cor):
        (Nrow, Ncol) = sino.shape
        centerfliplr = (Ncol + 1.0)/2.0-1.0
        # Use to shift the sino2 to the raw CoR
        shiftsino = np.int16(2*(raw_cor-centerfliplr))
        sino2 = np.roll(np.fliplr(sino[1:]), shiftsino, axis=1)
        lefttake = 0
        righttake = Ncol-1
        search_rad = self.parameters['search_radius']
        if raw_cor <= centerfliplr:
            lefttake = np.ceil(search_rad+1)
            righttake = np.floor(2*raw_cor-search_rad-1)
        else:
            lefttake = np.ceil(raw_cor-(Ncol-1-raw_cor)+search_rad+1)
            righttake = np.floor(Ncol-1-search_rad-1)
        Ncol1 = righttake-lefttake + 1
        mask = self._create_mask(2*Nrow-1, Ncol1,
                                 0.5*self.parameters['ratio']*Ncol)
        numshift = np.int16((2*search_rad+1.0)/self.parameters['step'])
        listshift = np.linspace(-search_rad, search_rad, num=numshift)
        listmetric = np.zeros(len(listshift), dtype=np.float32)
        num1 = 0
        for i in listshift:
            logging.debug("list shift %d", i)
            sino2a = ndi.interpolation.shift(sino2, (0, i), prefilter=False)
            sinojoin = np.vstack((sino, sino2a))
            listmetric[num1] = np.sum(np.abs(fft.fftshift(
                fft.fft2(sinojoin[:, lefttake:righttake + 1])))*mask)
            num1 = num1 + 1
        minpos = np.argmin(listmetric)
        rotcenter = raw_cor + listshift[minpos]/2.0
        return rotcenter, listmetric

    def filter_frames(self, data):
        # if data is greater than a certain size
        # data = data[0][::self.parameters['step']]
        # Reducing noise by smooth filtering, it's important
        logging.debug("performing gaussian filtering")
        sino = filter.gaussian_filter(data[0], sigma=(3, 1))
        logging.debug("performing coarse search")
        (raw_cor, raw_metric) = self._coarse_search(sino)
        logging.debug("performing fine search")
        (cor, listmetric) = self._fine_search(sino, raw_cor)
        logging.debug("%d %d", raw_cor, cor)
        return [np.array([cor]), np.array([cor])] # FIXME: should be raw_cor

    def post_process(self):
        # do some curve fitting here
        in_datasets, out_datasets = self.get_datasets()

        cor_raw = np.squeeze(out_datasets[0].data[...])
        # special case of one cor_raw value (i.e. only one sinogram)
        if not cor_raw.shape:
            # add to metadata
            cor_raw = out_datasets[0].data[...]
            self.populate_meta_data('cor_raw', cor_raw)
            self.populate_meta_data('centre_of_rotation', cor_raw)
            return

        cor_fit = np.squeeze(out_datasets[1].data[...])
        fit = np.zeros(cor_fit.shape)
        fit[:] = np.mean(cor_fit)
        cor_fit = fit

        self.populate_meta_data('cor_raw', cor_raw)
        self.populate_meta_data('centre_of_rotation', cor_fit)

    def populate_meta_data(self, key, value):
        datasets = self.parameters['datasets_to_populate']
        in_meta_data = self.get_in_meta_data()[0]
        in_meta_data.set_meta_data(key, value)
        for name in datasets:
            self.exp.index['in_data'][name].meta_data.set_meta_data(key, value)


    def setup(self):

        return;





        self.exp.log(self.name + " Start")

        # set up the output dataset that is created by the plugin
        in_dataset, out_dataset = self.get_datasets()

        self.orig_full_shape = in_dataset[0].get_shape()

        # if preview parameters exist then use these
        # else get the size of the data
        # get n processes and take 4 different sets of 5 from the data if this is feasible based on the data size.
        # calculate the slice list here and determine if it is feasible, else apply to max(n_processes, data_size)

        # reduce the data as per data_subset parameter
        in_dataset[0].get_preview().set_preview(self.parameters['preview'],
                                                revert=self.orig_full_shape)

        in_pData, out_pData = self.get_plugin_datasets()
        in_pData[0].plugin_data_setup('SINOGRAM', self.get_max_frames())
        # copy all required information from in_dataset[0]
        fullData = in_dataset[0]

        slice_dirs = np.array(in_pData[0].get_slice_directions())
        new_shape = (np.prod(np.array(fullData.get_shape())[slice_dirs]), 1)
        self.orig_shape = \
            (np.prod(np.array(self.orig_full_shape)[slice_dirs]), 1)

        out_dataset[0].create_dataset(shape=new_shape,
                                      axis_labels=['x.pixels', 'y.pixels'],
                                      remove=True)
        out_dataset[0].add_pattern("METADATA", core_dir=(1,), slice_dir=(0,))

        out_dataset[1].create_dataset(shape=self.orig_shape,
                                      axis_labels=['x.pixels', 'y.pixels'],
                                      remove=True)
        out_dataset[1].add_pattern("METADATA", core_dir=(1,), slice_dir=(0,))

        out_pData[0].plugin_data_setup('METADATA', self.get_max_frames())
        out_pData[1].plugin_data_setup('METADATA', self.get_max_frames())

        self.exp.log(self.name + " End")

    def nOutput_datasets(self):
        return 2

    def get_max_frames(self):
        """
        This filter processes 1 frame at a time

         :returns:  1
        """
        return 1

    def get_citation_information(self):
        cite_info = CitationInformation()
        cite_info.description = \
            ("The center of rotation for this reconstruction was calculated " +
             "automatically using the method described in this work")
        cite_info.bibtex = \
            ("@article{vo2014reliable,\n" +
             "title={Reliable method for calculating the center of rotation " +
             "in parallel-beam tomography},\n" +
             "author={Vo, Nghia T and Drakopoulos, Michael and Atwood, " +
             "Robert C and Reinhard, Christina},\n" +
             "journal={Optics Express},\n" +
             "volume={22},\n" +
             "number={16},\n" +
             "pages={19078--19086},\n" +
             "year={2014},\n" +
             "publisher={Optical Society of America}\n" +
             "}")
        cite_info.endnote = \
            ("%0 Journal Article\n" +
             "%T Reliable method for calculating the center of rotation in " +
             "parallel-beam tomography\n" +
             "%A Vo, Nghia T\n" +
             "%A Drakopoulos, Michael\n" +
             "%A Atwood, Robert C\n" +
             "%A Reinhard, Christina\n" +
             "%J Optics Express\n" +
             "%V 22\n" +
             "%N 16\n" +
             "%P 19078-19086\n" +
             "%@ 1094-4087\n" +
             "%D 2014\n" +
             "%I Optical Society of America")
        cite_info.doi = "http://dx.doi.org/10.1364/OE.22.019078"
        return cite_info





# -

