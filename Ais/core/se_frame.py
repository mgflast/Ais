from itertools import count
import numpy as np
import mrcfile
from Ais.core.opengl_classes import *
import datetime
import Ais.core.config as cfg
import Ais.core.settings as settings
from Ais.core.background_process import BackgroundProcess
from skimage import measure
from scipy.ndimage import label, binary_dilation
from Ais.core.util import coords_from_star
from Ais.core.normalization import global_stats as compute_global_stats


class SEFrame:
    idgen = count(0)

    HISTOGRAM_BINS = 40

    def __init__(self, path):
        uid_counter = next(SEFrame.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+"000") + uid_counter
        self.path = path
        self.scns_path = "n/a"
        self._norm_cache = None   # (path, center, scale)
        if os.path.exists(self.path):
            self.title = os.path.splitext(os.path.basename(path))[0]
            self.n_slices = 0
            self.current_slice = -1
            self.slice_changed = False
            self.data = None
            self.rendered_data = None
            self.includes_map = False
            self.map = None
            self.features = list()
            self.feature_counter = 0
            self.active_feature = None
            self.height, self.width = mrcfile.mmap(self.path, mode="r", permissive=True).data.shape[1:3]
            self.pixel_size = mrcfile.open(self.path, header_only=True, permissive=True).voxel_size.x / 10.0
            if self.pixel_size == 0.0:
                self.pixel_size = 1.0
            self.transform = Transform()
            self.clem_frame = None
            self.clem_frame_path = None
            self.overlay = None
            self.texture = None
            self.quad_va = None
            self.border_va = None
            self.interpolate = False
            self.alpha = 1.0
            self.filters = list()
            self.invert = False
            self.crop = False
            self.crop_roi = [0, 0, self.width, self.height]
            self.autocontrast = True
            self.sample = True
            self.export = False
            self.pick = False
            self.export_bottom = 0
            self.export_top = None
            self.hist_vals = list()
            self.hist_bins = list()
            self.corner_positions_local = []
            self.set_slice(0, False)
            self.setup_opengl_objects()
            self.contrast_lims = [0, 512.0]
            self.compute_autocontrast()
            self.compute_histogram()
            self.set_slice(self.n_slices // 2, True)

    def __getstate(self):
        state = self.__dict__.copy()
        return state

    def __setstate(self, state):
        self.__dict__.update(state)

    def setup_opengl_objects(self):
        if not cfg.glfw_initialized:
            return
        self.texture = Texture(format="r32f")
        self.texture.update(self.data.astype(np.float32))
        self.quad_va = VertexArray()
        self.border_va = VertexArray(attribute_format="xy")
        self.generate_va()
        self.interpolate = not self.interpolate
        self.toggle_interpolation()

    def toggle_interpolation(self):
        if not cfg.glfw_initialized:
            return
        self.interpolate = not self.interpolate
        if self.interpolate:
            self.texture.set_linear_mipmap_interpolation()
        else:
            self.texture.set_no_interpolation()

    def generate_va(self):
        # set up the quad vertex array
        w, h = self.width * 0.5, self.height * 0.5
        vertex_attributes = list()
        indices = list()
        n = 4
        for i in range(n):
            for j in range(n):
                x = ((2 * i / (n - 1)) - 1) * w
                y = ((2 * j / (n - 1)) - 1) * h
                u = 0.5 + x / w / 2
                v = 0.5 + y / h / 2
                vertex_attributes += [x, y, 0.0, u, v]

        for i in range(n - 1):
            for j in range(n - 1):
                idx = i * n + j
                indices += [idx, idx + 1, idx + n, idx + n, idx + 1, idx + n + 1]

        self.corner_positions_local = [[-w, h], [-w, -h], [w, -h], [w, h]]
        self.quad_va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))

        # set up the border vertex array
        vertex_attributes = [-w, h,
                             w, h,
                             w, -h,
                             -w, -h]
        indices = [0, 1, 1, 2, 2, 3, 3, 0]
        self.border_va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))

    def set_slice(self, requested_slice, update_texture=True, reset=False):
        if requested_slice == self.current_slice and not reset:
            return
        else:
            if not self.includes_map and not os.path.isfile(self.path):
                print(f"Parent .mrc file at {self.path} can no longer be found!")
                return
            self.slice_changed = True
            if self.includes_map:
                mrc = self.map
            else:
                mrc = mrcfile.mmap(self.path, mode="r", permissive=True)
            self.n_slices = mrc.data.shape[0]
            if self.export_top is None:
                self.export_top = self.n_slices
            requested_slice = min([max([requested_slice, 0]), self.n_slices - 1])
            if mrc.data.dtype == np.dtype('int8'):
                self.data = mrc.data[requested_slice, :, :].astype(np.uint8).astype(np.float32)
            else:
                self.data = mrc.data[requested_slice, :, :].astype(np.float32)
            self.rendered_data = self.data
        self.current_slice = requested_slice
        for s in self.features:
            s.set_slice(self.current_slice)
        if update_texture:
            self.update_image_texture()

    def get_slice(self, requested_slice=None, as_float=True, model_depth=None):
        if requested_slice is None:
            requested_slice = self.current_slice
        requested_slice = min([max([requested_slice, 0]), self.n_slices - 1])
        if self.includes_map:
            mrc = self.map
        else:
            mrc = mrcfile.mmap(self.path, mode="r", permissive=True)
        self.n_slices = mrc.data.shape[0]
        if self.export_top is None:
            self.export_top = self.n_slices
        if model_depth is None:
            out_data = mrc.data[requested_slice, :, :]
        else:
            stack_indices = np.clip(np.arange(requested_slice - model_depth // 2, requested_slice - model_depth // 2 + model_depth), 0, self.n_slices - 1)
            out_data = mrc.data[stack_indices, :, :]
        if as_float:
            target_type_dict = {np.float32: float, float: float, np.dtype('int8'): np.dtype('uint8'), np.dtype('int16'): np.dtype('float32')}
            if out_data.dtype not in target_type_dict:
                target_type = float
            else:
                target_type = target_type_dict[out_data.dtype]
            out_data = np.array(out_data.astype(target_type, copy=False), dtype=float)
        return out_data

    def global_norm_stats(self):
        # whole-tomogram (center, scale), from 32 slices; cached, recomputed on path change
        if self._norm_cache is None or self._norm_cache[0] != self.path:
            zidx = np.linspace(0, self.n_slices - 1, min(32, self.n_slices)).astype(int)
            slabs = np.stack([self.get_slice(int(z)) for z in zidx])
            center, scale = compute_global_stats(slabs)
            self._norm_cache = (self.path, center, scale)
        return self._norm_cache[1], self._norm_cache[2]

    def get_roi_indices(self):
        """Returns a tuple of tuples (x_indices, y_indices), where x_indices is (x_start, x_stop)"""
        y_indices = (self.height - self.crop_roi[3], self.height - self.crop_roi[1])
        x_indices = (self.crop_roi[0], self.crop_roi[2])
        return y_indices, x_indices

    def update_image_texture(self):
        if not cfg.glfw_initialized:
            return
        self.texture.update(self.rendered_data.astype(np.float32))

    def update_model_matrix(self):
        self.transform.scale = self.pixel_size
        self.transform.compute_matrix()
        for i in range(4):
            vec = np.matrix([*self.corner_positions_local[i], 0.0, 1.0]).T
            corner_pos = (self.transform.matrix * vec)[0:2]
            self.corner_positions_local[i] = [float(corner_pos[0]), float(corner_pos[1])]

    def world_to_pixel_coordinate(self, world_coordinate):
        vec = np.matrix([world_coordinate[0], world_coordinate[1], 0.0, 1.0]).T
        invmat = np.linalg.inv(self.transform.matrix)
        out_vec = invmat * vec
        pixel_coordinate = [int(out_vec[0, 0] + self.width / 2), int(out_vec[1, 0] + self.height / 2)]
        return pixel_coordinate

    def compute_autocontrast(self, saturation=None, pxd=None):
        data = self.data if pxd is None else pxd
        #s = data.shape
        #data = data[int(s[0]/3):int(s[0]*2/3), int(s[1]/3):int(s[1]*2/3)]
        saturation_pct = settings.autocontrast_saturation
        if saturation:
            saturation_pct = saturation
        if min(data.shape) > Filter.M * 4:
            subsample = data[Filter.M:-Filter.M:settings.autocontrast_subsample, Filter.M:-Filter.M:settings.autocontrast_subsample]
        else:
            subsample = data[::settings.autocontrast_subsample, ::settings.autocontrast_subsample]
        n = subsample.shape[0] * subsample.shape[1]
        sorted_pixelvals = np.sort(subsample.flatten())

        min_idx = min([int(saturation_pct / 100.0 * n), n - 1])
        max_idx = max([int((1.0 - saturation_pct / 100.0) * n), 0])
        self.contrast_lims[0] = sorted_pixelvals[min_idx]
        self.contrast_lims[1] = sorted_pixelvals[max_idx]

    def compute_histogram(self, pxd=None):
        data = self.data if pxd is None else pxd
        # ignore very bright pixels
        subsample = data[Filter.M:-Filter.M:settings.autocontrast_subsample, Filter.M:-Filter.M:settings.autocontrast_subsample]
        mean = np.mean(subsample)
        std = np.std(subsample)
        self.hist_vals, self.hist_bins = np.histogram(subsample[subsample < (mean + 20 * std)], bins=SEFrame.HISTOGRAM_BINS)

        self.hist_vals = self.hist_vals.astype('float32')
        self.hist_bins = self.hist_bins.astype('float32')
        self.hist_vals = np.log(self.hist_vals + 1)

    @staticmethod
    def load(path):
        import pickle
        try:
            with open(path, 'rb') as pf:
                se_frame = pickle.load(pf)
                se_frame.on_load()
            return se_frame
        except Exception as e:
            print(f'Error loading SEFrame from {path}: {e}')

    def on_load(self):
        uid_counter = next(SEFrame.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        self.setup_opengl_objects()
        for f in self.features:
            f.on_load()
        if self.overlay is not None:
            self.overlay.setup_opengl_objects()

    def set_overlay(self, pxd, parent_clem_frame, overlay_update_function):
        self.overlay = Overlay(pxd, self)
        self.overlay.set_parent_clem_frame(parent_clem_frame)
        self.overlay.set_update_function(overlay_update_function)

    def include_map(self):
        self.includes_map = True
        self.map = mrcfile.open(self.path, permissive=True)

    def __reduce__(self):
        state = self.__dict__.copy()
        to_remove = ['rendered_data', 'clem_frame']
        for key in to_remove:
            if key in state:
                del state[key]
        return self.__class__, (self.path,), state

    def __eq__(self, other):
        if isinstance(other, SEFrame):
            return self.uid == other.uid
        return False


class Filter:
    TYPES = ["Gaussian blur", "Offset Gaussian blur","Box blur", "Sobel vertical", "Sobel horizontal"]
    PARAMETER_NAME = ["sigma", "sigma", "box", None, None]
    M = 16

    def __init__(self, filter_type):
        self.type = filter_type  # integer, corresponding to an index in the Filter.TYPES list
        self.k1 = np.zeros((Filter.M * 2 + 1, 1), dtype=np.float32)
        self.k2 = np.zeros((Filter.M * 2 + 1, 1), dtype=np.float32)
        self.enabled = True
        self.ssbo1 = -1
        self.ssbo2 = -1
        self.param = 1.0
        self.strength = 1.0
        self.fill_kernel()

    def upload_buffer(self):
        if self.ssbo1 != -1:
            glDeleteBuffers(2, [self.ssbo1, self.ssbo2])
        self.ssbo1 = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo1)
        glBufferData(GL_SHADER_STORAGE_BUFFER, (Filter.M * 2 + 1) * 4, self.k1.flatten(), GL_STATIC_READ)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        self.ssbo2 = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo2)
        glBufferData(GL_SHADER_STORAGE_BUFFER, (Filter.M * 2 + 1) * 4, self.k2.flatten(), GL_STATIC_READ)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)


    def bind(self, horizontal=True):
        if horizontal:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo1)
        else:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo2)

    def unbind(self):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def fill_kernel(self):
        if self.type == 0:
            self.k1 = np.exp(-np.linspace(-Filter.M, Filter.M, 2*Filter.M+1)**2 / self.param**2, dtype=np.float32)
            self.k1 /= np.sum(self.k1, dtype=np.float32)
            self.k2 = self.k1
        if self.type == 1:
            self.k1 = np.exp(-np.linspace(-Filter.M, Filter.M, 2 * Filter.M + 1) ** 2 / self.param ** 2, dtype=np.float32)
            self.k1 /= np.sum(self.k1, dtype=np.float32)
            self.k1 -= np.mean(self.k1)
            self.k2 = self.k1
        if self.type == 2:
            m = min([int(self.param), 7])
            self.k1 = np.asarray([0] * (Filter.M - 1 - m) + [1] * m + [1] + [1] * m + [0] * (Filter.M - 1 - m)) / (2 * m + 1)
            self.k2 = self.k1
        if self.type == 3:
            self.k1 = np.asarray([0] * (Filter.M - 1) + [1, 2, 1] + [0] * (Filter.M - 1))
            self.k2 = np.asarray([0] * (Filter.M - 1) + [1, 0, -1] + [0] * (Filter.M - 1))
        if self.type == 4:
            self.k1 = np.asarray([0] * (Filter.M - 1) + [1, 0, -1] + [0] * (Filter.M - 1))
            self.k2 = np.asarray([0] * (Filter.M - 1) + [1, 2, 1] + [0] * (Filter.M - 1))

        self.k1 = np.asarray(self.k1, dtype=np.float32)
        self.k2 = np.asarray(self.k2, dtype=np.float32)
        self.upload_buffer()


class Overlay:
    idgen = count(0)

    def __init__(self, pxd, parent_se_frame):
        self.uid = next(Overlay.idgen)
        self.pxd = pxd
        self.texture = Texture(format="rgba32f")
        self.texture.update(pxd)
        self.texture.set_linear_interpolation()
        self.clem_frame = None
        self.clem_frame_path = ''
        self.se_frame = parent_se_frame
        self.update_function = None

    def set_parent_clem_frame(self, clem_frame):
        self.clem_frame = clem_frame
        self.clem_frame_path = clem_frame.path

    def set_update_function(self, fn):
        self.update_function = fn

    def update(self):
        self.pxd = self.update_function(self.clem_frame)
        self.texture.update(self.pxd)

    def setup_opengl_objects(self):
        self.texture = Texture(format="rgba32f")
        self.texture.update(self.pxd)

    @property
    def size(self):
        return self.pxd.shape

    def __reduce__(self):
        state = self.__dict__.copy()
        to_remove = ['clem_frame', 'update_function']
        for key in to_remove:
            if key in state:
                del state[key]
        return self.__class__, (self.pxd, self.se_frame, ), state


class Segmentation:
    idgen = count(0)

    DEFAULT_COLOURS = [(66 / 255, 214 / 255, 164 / 255),
                       (255 / 255, 243 / 255, 0 / 255),
                       (255 / 255, 104 / 255, 0 / 255),
                       (255 / 255, 13 / 255, 0 / 255),
                       (174 / 255, 0 / 255, 255 / 255),
                       (21 / 255, 0 / 255, 255 / 255),
                       (0 / 255, 136 / 255, 266 / 255),
                       (0 / 255, 247 / 255, 255 / 255),
                       (0 / 255, 255 / 255, 0 / 255)]

    __debug_easymode_annotate_filaments_previous_coordinate = None

    def __init__(self, parent_frame, title):
        uid_counter = next(Segmentation.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        self.parent = parent_frame
        self.parent.active_feature = self
        self.parent.feature_counter += 1
        self.width = self.parent.width
        self.height = self.parent.height
        self.title = title
        self.colour = Segmentation.DEFAULT_COLOURS[(self.parent.feature_counter - 1) % len(Segmentation.DEFAULT_COLOURS)]
        self.alpha = 1.0
        self.hide = False
        self.contour = False
        self.expanded = False
        self.collapsed = False   # GUI: feature panel folded down to its title row
        self.brush_size = 10.0
        self.show_boxes = True
        self.magic = False
        self.magic_strength = 50.0
        self.box_size = 128
        self.box_size_nm = self.box_size * self.parent.pixel_size
        self.slices = dict()
        self.boxes = dict()
        self.n_boxes = 0
        self.edited_slices = list()
        self.current_slice = -1
        self.data = None
        self.texture = Texture(format="r32f")
        self.texture.update(None, self.width, self.height)
        self.texture.set_linear_interpolation()
        self.set_slice(self.parent.current_slice)

    def on_load(self):
        if not hasattr(self, "collapsed"):   # attribute added later - older saved datasets lack it
            self.collapsed = False
        uid_counter = next(Segmentation.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        self.texture = Texture(format="r32f")
        self.texture.set_linear_interpolation()
        cslice = self.current_slice
        self.current_slice = -1
        self.set_slice(cslice)

    def set_box_size(self, box_size_px):
        self.box_size = box_size_px
        self.box_size_nm = self.box_size * self.parent.pixel_size

    def add_box(self, pixel_coordinates):
        if self.current_slice not in self.boxes:
            self.boxes[self.current_slice] = list()
        self.request_draw_in_current_slice()
        self.boxes[self.current_slice].append(pixel_coordinates)
        self.n_boxes += 1

    def remove_box(self, pixel_coordinate):
        box_list = self.boxes[self.current_slice]
        x = pixel_coordinate[0]
        y = pixel_coordinate[1]
        d = np.inf
        idx = None
        for i in range(len(box_list)):
            _d = (box_list[i][0]-x)**2 + (box_list[i][1]-y)**2
            if _d < d:
                d = _d
                idx = i
        if idx is not None:
            self.boxes[self.current_slice].pop(idx)
            self.n_boxes -= 1

    def set_slice(self, requested_slice):
        if requested_slice == self.current_slice:
            return
        self.current_slice = requested_slice
        if requested_slice in self.slices:
            self.data = self.slices[requested_slice]
            if self.data is None:
                self.texture.update(self.data, self.width, self.height)
            else:
                self.texture.update(self.data)
        else:
            self.slices[requested_slice] = None
            if requested_slice not in self.boxes:
                self.boxes[requested_slice] = list()
            self.data = None
            self.texture.update(self.data, self.width, self.height)

    def remove_slice(self, requested_slice):
        if requested_slice in self.edited_slices:
            self.edited_slices.remove(requested_slice)
            self.slices.pop(requested_slice)
            if self.current_slice == requested_slice:
                self.data *= 0
                self.texture.update(self.data, self.width, self.height)
        if requested_slice in self.boxes:
            self.n_boxes -= len(self.boxes[requested_slice])
            self.boxes[requested_slice] = list()

    def request_draw_in_current_slice(self):
        if self.current_slice in self.slices:
            if self.slices[self.current_slice] is None:
                self.slices[self.current_slice] = np.zeros((self.height, self.width), dtype=np.uint8)
                self.data = self.slices[self.current_slice]
                self.texture.update(self.data, self.width, self.height)
                self.edited_slices.append(self.current_slice)
        else:
            self.slices[self.current_slice] = np.zeros((self.height, self.width), dtype=np.uint8)
            self.data = self.slices[self.current_slice]
            self.texture.update(self.data, self.width, self.height)
            self.edited_slices.append(self.current_slice)

    def save_particle_positions(self):
        fpath = os.path.splitext(self.parent.path)[0] + "_" + self.title + "_particles.txt"
        try:
            with open(fpath, 'w') as f:
                for z in self.boxes:
                    box_list = self.boxes[z]
                    if len(box_list) > 0:
                        for box in box_list:
                            f.write(f"{box[0]}\t{box[1]}\t{z}\n")
            print(f"Coordinates saved to: {fpath}")
        except Exception as e:
            cfg.set_error(e, "Could not save particle positions, see details below.")

    def save_current_slice(self):
        fpath = os.path.splitext(self.parent.path)[0] + "_" + self.title + f"_slice_{self.current_slice}.mrc"
        try:
            with mrcfile.new(fpath, overwrite=True) as mrc:
                image = self.parent.get_slice(as_float=False)
                annotation = np.zeros_like(image) if self.data is None else self.data.astype(image.dtype)
                mrc.set_data(np.array([image, annotation]))
                mrc.voxel_size = self.parent.pixel_size * 10.0
            print(f"Slice saved to: {fpath}")
        except Exception as e:
            cfg.set_error(e, "Could not save current slice, see details below.")

    def save_volume(self, path=None):
        fpath = os.path.splitext(self.parent.path)[0] + "__" + self.title + f"_manual.mrc" if path is None else path
        try:
            with mrcfile.new(fpath, overwrite=True) as outf:
                vol = np.zeros((self.parent.n_slices, self.parent.height, self.parent.width), dtype=np.uint8)
                for i in self.edited_slices:
                    vol[i] = self.slices[i]
                outf.set_data(vol * 255)
                outf.voxel_size = self.parent.pixel_size * 10.0
            print(f"Volume saved to: {fpath}")
        except Exception as e:
            print(e)

    def set_slice_ndarray(self, pxd, slice_number):
        self.set_slice(slice_number)
        if slice_number not in self.edited_slices:
            self.edited_slices.append(slice_number)
        self.slices[slice_number] = (pxd > 0).astype(np.uint8)
        self.data = self.slices[slice_number]
        self.texture.update(self.data, self.width, self.height)

    def import_slice(self, path, threshold=128):
        data = mrcfile.mmap(path).data[self.current_slice, :, :]
        if data.shape != (self.height, self.width):
            raise Exception("Selected .mrc does not have the same dimensions as this Feature's parent .mrc")
        self.slices[self.current_slice] = (data > threshold).astype(np.uint8)
        if self.current_slice not in self.edited_slices:
            self.edited_slices.append(self.current_slice)
        self.boxes[self.current_slice] = list()
        self.data = self.slices[self.current_slice]
        self.texture.update(self.data, self.width, self.height)

    def import_mrc(self, path, threshold=128):
        data = mrcfile.read(path)
        if data.shape != (self.parent.n_slices, self.height, self.width):
            raise Exception("Selected .mrc does not have the same dimensions as this Feature's parent .mrc")
        self.slices = dict()
        self.edited_slices = list()
        for i in range(data.shape[0]):
            self.edited_slices.append(i)
            self.slices[i] = (data[i, :, :] > threshold).astype(np.uint8)
            self.boxes[i] = list()
        self.data = self.slices[self.current_slice]
        self.texture.update(self.data, self.width, self.height)


class Transform:
    def __init__(self):
        self.translation = np.array([0.0, 0.0])
        self.rotation = 0.0
        self.scale = 1.0
        self.matrix = np.identity(4)
        self.matrix_no_scale = np.identity(4)

    def compute_matrix(self):
        scale_mat = np.identity(4) * self.scale
        scale_mat[3, 3] = 1

        rotation_mat = np.identity(4)
        _cos = np.cos(self.rotation / 180.0 * np.pi)
        _sin = np.sin(self.rotation / 180.0 * np.pi)
        rotation_mat[0, 0] = _cos
        rotation_mat[1, 0] = _sin
        rotation_mat[0, 1] = -_sin
        rotation_mat[1, 1] = _cos

        translation_mat = np.identity(4)
        translation_mat[0, 3] = self.translation[0]
        translation_mat[1, 3] = self.translation[1]

        self.matrix = np.matmul(translation_mat, np.matmul(rotation_mat, scale_mat))
        self.matrix_no_scale = np.matmul(translation_mat, rotation_mat)

    def __add__(self, other):
        out = Transform()
        out.translation[0] = self.translation[0] + other.translation[0]
        out.translation[1] = self.translation[1] + other.translation[1]
        out.rotation = self.rotation + other.rotation
        return out

    def __str__(self):
        return f"Transform with translation = {self.translation[0], self.translation[1]}, scale = {self.scale}, rotation = {self.rotation}"

    def __sub__(self, other):
        out = Transform()
        out.translation[0] = self.translation[0] - other.translation[0]
        out.translation[1] = self.translation[1] - other.translation[1]
        out.rotation = self.rotation - other.rotation
        return out


class SurfaceModel:
    idgen = count(0)
    COLOURS = dict()

    DEFAULT_COLOURS = [(66 / 255, 214 / 255, 164 / 255),
                       (255 / 255, 243 / 255, 0 / 255),
                       (255 / 255, 104 / 255, 0 / 255),
                       (255 / 255, 13 / 255, 0 / 255),
                       (174 / 255, 0 / 255, 255 / 255),
                       (21 / 255, 0 / 255, 255 / 255),
                       (0 / 255, 136 / 255, 266 / 255),
                       (0 / 255, 247 / 255, 255 / 255),
                       (0 / 255, 255 / 255, 0 / 255)]
    DEFAULT_COLOURS_IDX = 0

    def __init__(self, path, pixel_size, no_gpu=False):
        self.uid = next(SurfaceModel.idgen)
        self.path = path
        self.title = os.path.splitext(os.path.basename(self.path))[0]
        self.title = self.title[2+self.title.rfind("__"):]
        self.colour = (0.0, 0.0, 0.0)
        self.vertices = list()
        self.indices = list()
        self.no_gpu = no_gpu

        # check whether there are any models or features with the same name, if so, give it that color:
        self.set_colour()
        self.data = None
        self.size_mb = os.path.getsize(self.path) * 1e-6
        self.binned_data = None

        self.blobs = dict()
        self.level = 128
        self.last_level = 0
        self.dust = 1.0
        self.bin = 2
        header = mrcfile.open(self.path, header_only=True, mode='r').header
        n_voxels = header.nx * header.ny * header.nz
        self.bin = max(int(np.round(n_voxels**0.5 / 4e3)), 1)
        self.latest_bin = -1
        self.hide = False
        self.alpha = 1.0
        self.process = None
        self.pixel_size = pixel_size
        self.center_xyz = np.array([header.nx, header.ny, header.nz])
        self.initialized = False

        self.particles = list()
        self.particle_size = 0.0
        self.particle_colour = (0.0, 0.0, 0.0)
        self.find_coordinates()

        # if a feature that corresponds to this SurfaceModel exists in the feature library, edit settings accordingly.
        for feature in cfg.feature_library + list(cfg.feature_library_session.values()):
            if self.title == feature.title and feature.use:
                self.level = feature.level
                self.dust = feature.dust
                self.alpha = feature.render_alpha
                self.hide = feature.hide

    def find_coordinates(self):
        # is there a coordinate file?
        self.particles = list()
        star_file = os.path.basename(os.path.splitext(self.path)[0]) + "_coords.star"
        for d in cfg.settings["SEARCH_DIRECTORIES"] + [os.path.dirname(self.path)]:
            coordinate_file = os.path.join(d, star_file)
            if os.path.exists(coordinate_file):
                self.load_coordinates(coordinate_file)


    def load_coordinates(self, coordinate_file):
        try:
            coordinates = coords_from_star(coordinate_file)
            if isinstance(coordinates, list):
                self.particles = coordinates
                print(f"\t\tloading coordinates for SurfaceModel object with path {self.path}")
        except Exception as e:
            cfg.set_error(e, f'Could not load coordinates from {coordinate_file}')

    def set_colour(self):
        for feature in cfg.feature_library:
            SurfaceModel.COLOURS[feature.title] = feature.colour
        if self.title in SurfaceModel.COLOURS:
            self.colour = SurfaceModel.COLOURS[self.title]
            return
        for frame in cfg.se_frames:
            for feature in frame.features:
                if self.title in feature.title:
                    self.colour = feature.colour
                    return
        for model in cfg.se_models:
            if self.title in model.title:
                self.colour = model.colour
                return
        self.colour = SurfaceModel.DEFAULT_COLOURS[SurfaceModel.DEFAULT_COLOURS_IDX % len(SurfaceModel.DEFAULT_COLOURS)]
        SurfaceModel.DEFAULT_COLOURS_IDX += 1
        SurfaceModel.COLOURS[self.title] = self.colour

    def hide_dust(self):
        try:
            for i in self.blobs:
                self.blobs[i].hide = self.blobs[i].volume < self.dust
        except Exception as e:
            print("Error in hide_dust - maybe SurfaceModel and its thread lost synchronisation.", e)

    def generate_model(self):
        self.initialized = True
        if self.process is None:
            self.process = BackgroundProcess(self._generate_model, (), name=f"{self.title} generate model (SurfaceModel)")
            self.process.start()

    def _generate_model(self, process):
        if self.data is None:
            self.data = mrcfile.read(self.path)
            MARGIN = 1
            self.data[:MARGIN, :, :] = 0
            self.data[-MARGIN:, :, :] = 0
            self.data[:, :MARGIN, :] = 0
            self.data[:, -MARGIN:, :] = 0
            self.data[:, :, :MARGIN] = 0
            self.data[:, :, -MARGIN:] = 0
            if self.data.dtype == np.float32:
                self.data *= 255.0
            if self.data.dtype == np.int8:
                self.data = self.data.astype(np.float32) * 255.0 / 127.0

        if self.latest_bin != self.bin and self.bin != 1:
            self.latest_bin = self.bin
            self.binned_data = SurfaceModel.bin_data(self.data, self.bin)
        data = self.data if self.bin == 1 else self.binned_data
        origin = 0.5 * np.array(self.data.shape) * self.pixel_size
        process.set_progress(0.05)
        new_blobs = dict()

        # 1: scipy.ndimage.label
        labels, N = label(data >= self.level)
        process.set_progress(0.1)

        # 2: convert nonzero blobs to SurfaceBlob objects.
        Z, Y, X = np.nonzero(labels)
        for i in range(len(Z)):
            z = Z[i]
            y = Y[i]
            x = X[i]
            l = labels[z, y, x]
            if l not in new_blobs:
                new_blobs[l] = SurfaceModelBlob(data, self.level, self.pixel_size * self.bin, origin, no_gpu=self.no_gpu)
            new_blobs[l].x.append(x)
            new_blobs[l].y.append(y)
            new_blobs[l].z.append(z)

        process.set_progress(0.2)

        # 3: upload surface blobs one by one.
        for i in new_blobs:
            try:
                new_blobs[i].compute_mesh()
            except Exception as e:
                pass
            process.set_progress(0.2 + 0.699 * ((i + 1) / len(new_blobs)))

        for i in self.blobs:
            self.blobs[i].delete()
        self.blobs = new_blobs
        self.hide_dust()

        process.set_progress(1.0)

    def paint_particles(self, path_to_tsv):
        data = list()
        with open(path_to_tsv, 'r') as f:
            for line in f.readlines():
                values = line.split("\t")
                data.append([int(values[0]), int(values[1]), int(values[2]), float(values[3]), float(values[4]), float(values[5])])
        data = np.array(data)
        for blob in self.blobs.values():
            x_range = (np.amin(blob.x), np.amax(blob.x))
            y_range = (np.amin(blob.y), np.amax(blob.y))
            z_range = (np.amin(blob.z), np.amax(blob.z))
            for j in range(data.shape[0]):
                x = data[j, 0] / self.bin
                y = data[j, 1] / self.bin
                z = data[j, 2] / self.bin
                if x_range[0] < x < x_range[1] and y_range[0] < y < y_range[1] and z_range[0] < z < z_range[1]:
                    blob.colour = (data[j, 3], data[j, 4], data[j, 5])
                    blob.painted = True
                    break

    @staticmethod
    def bin_data(data, b):
        z, y, x = data.shape
        data = data[:z//b * b, :y//b * b, :x//b * b]
        data = data.reshape((z // b, b, y//b, b, x//b, b)).mean(5).mean(3).mean(1)
        data[0, :, :] = 0
        data[-1, :, :] = 0
        data[:, 0, :] = 0
        data[:, -1, :] = 0
        data[:, :, 0] = 0
        data[:, :, -1] = 0
        return data

    def delete(self):
        for i in self.blobs:
            self.blobs[i].delete()

    def on_update(self):
        for i in self.blobs:
            self.blobs[i].update_if_necessary()

    def save_as_obj(self, path, force_save=False):
        if self.hide or not self.initialized:
            return
        with open(path, 'w') as f:
            f.write(f"# colour {self.colour[0]:.3f} {self.colour[1]:.3f} {self.colour[2]:.3f}\n")
            f_idx = 1
            for b in self.blobs.values():
                if b.hide:
                    continue
                n = len(b.vertices)
                print(f"Adding {n} positions and normals")
                for i in range(n):
                    xyz = b.vertices[i, :]
                    f.write(f"v {xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}\n")
                for i in range(n):
                    nxnynz = b.normals[i, :]
                    f.write(f"vn {nxnynz[0]:.4f} {nxnynz[1]:.4f} {nxnynz[2]:.4f}\n")
                n = b.indices.shape[0]
                for i in range(int(n / 3)):
                    v0 = b.indices[3 * i + 0] + f_idx
                    v1 = b.indices[3 * i + 1] + f_idx
                    v2 = b.indices[3 * i + 2] + f_idx
                    f.write(f"f {v0}//{v0} {v1}//{v1} {v2}//{v2}\n")
                f_idx += (1 + np.max(b.indices))
                print(f"Incrementing f_idx by {np.max(b.indices)}")
        return path


    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.uid == other.uid
        return False


class SurfaceModelBlob:
    def __init__(self, data, level, pixel_size, origin, no_gpu=False):
        self.data = data
        self.level = level
        self.pixel_size = pixel_size
        self.origin = origin
        self.no_gpu = no_gpu
        self.x = list()
        self.y = list()
        self.z = list()
        self.volume = 0
        self.indices = list()
        self.vertices = list()
        self.normals = list()
        self.vao_data = list()
        self.painted = False
        self.colour = (1.0, 0.0, 1.0)
        if not no_gpu:
            self.va = VertexArray(attribute_format="xyznxnynz")
        self.va_requires_update = False
        self.complete = False
        self.hide = False

    def compute_mesh(self):
        self.volume = len(self.x) * self.pixel_size**3
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)

        rx = (np.amin(self.x), np.amax(self.x)+2)
        ry = (np.amin(self.y), np.amax(self.y)+2)
        rz = (np.amin(self.z), np.amax(self.z)+2)
        box = np.zeros((1 + rz[1]-rz[0] + 1, 1 + ry[1]-ry[0] + 1, 1 + rx[1]-rx[0] + 1))
        box[1:-1, 1:-1, 1:-1] = self.data[rz[0]:rz[1], ry[0]:ry[1], rx[0]:rx[1]]
        mask = np.zeros((1 + rz[1]-rz[0] + 1, 1 + ry[1]-ry[0] + 1, 1 + rx[1]-rx[0] + 1), dtype=bool)

        mx = self.x - rx[0] + 1
        my = self.y - ry[0] + 1
        mz = self.z - rz[0] + 1
        for x, y, z in zip(mx, my, mz):
            mask[z, y, x] = True
        mask = binary_dilation(mask, iterations=2)
        box *= mask
        vertices, faces, normals, _ = measure.marching_cubes(box, level=self.level)
        vertices += np.array([rz[0], ry[0], rx[0]])
        self.vertices = vertices[:, [2, 1, 0]]
        self.normals = normals[:, [2, 1, 0]]

        self.vertices *= self.pixel_size
        self.vertices -= np.array([self.origin[2], self.origin[1], self.origin[0]])
        self.vao_data = np.hstack((self.vertices, self.normals)).flatten()
        self.indices = faces.flatten()
        self.va_requires_update = True

    def update_if_necessary(self):
        if self.va_requires_update and not self.no_gpu:
            self.va.update(VertexBuffer(self.vao_data), IndexBuffer(self.indices, long=True))
            self.va_requires_update = False
            self.complete = True

    def delete(self):
        if self.va.initialized:
            # free GPU buffers/VAO to avoid leaking VRAM on regeneration. glDelete* becomes a
            # null wrapper at interpreter shutdown, so guard with bool() to avoid NullFunctionError.
            if bool(glDeleteBuffers):
                if self.va.vertexBuffer is not None:
                    glDeleteBuffers(1, [self.va.vertexBuffer.vertexBufferObject])
                if self.va.indexBuffer is not None:
                    glDeleteBuffers(1, [self.va.indexBuffer.indexBufferObject])
            if bool(glDeleteVertexArrays) and self.va.vertexArrayObject is not None:
                glDeleteVertexArrays(1, [self.va.vertexArrayObject])
            self.va.initialized = False
