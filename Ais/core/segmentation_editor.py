import glfw
import imgui
from PIL import Image
from copy import copy
from tkinter import filedialog
import dill as pickle
from Ais.core.se_model import *
from Ais.core.se_frame import *
import Ais.core.se_scnt as se_scnt
import Ais.core.widgets as widgets
from Ais.core.util import clamp, bin_mrc
from Ais.core import progression
import pyperclip
import os
import math
import subprocess
import tempfile
import shutil
from time import sleep
from Ais.core.util import pick_particles, icosphere_va
EMBEDDED = False
try:
    import scNodes.core.config as scn_cfg
    EMBEDDED = True
except ImportError:
    pass


# TODO: fix error in Render -> pick in GUI
# TODO: when choosing a session feature in annotation context menu, brush size doesnt get set.

class SegmentationEditor:
    CAMERA_ZOOM_STEP = 0.1
    CAMERA_MAX_ZOOM = 100.0
    DEFAULT_HORIZONTAL_FOV_WIDTH = 1000
    DEFAULT_ZOOM = 1.0  # adjusted in init
    DEFAULT_WORLD_PIXEL_SIZE = 1.0  # adjusted on init

    # GUI params
    MAIN_WINDOW_WIDTH = 330
    FEATURE_PANEL_HEIGHT = 104
    INFO_HISTOGRAM_HEIGHT = 70
    SLICER_WINDOW_VERTICAL_OFFSET = 30
    SLICER_WINDOW_WIDTH = 700
    ACTIVE_SLICES_CHILD_HEIGHT = 140
    PROGRESS_BAR_HEIGHT = 8
    MODEL_PANEL_HEIGHT_TRAINING = 158
    MODEL_PANEL_HEIGHT_PREDICTION = 145
    MODEL_PANEL_HEIGHT_LOGIC = 115

    TOOLTIP_APPEAR_DELAY = 1.0
    TOOLTIP_HOVERED_TIMER = 0.0
    TOOLTIP_HOVERED_START_TIME = 0.0

    renderer = None

    BLEND_MODES = dict()  # blend mode template: (glBlendFunc ARG1, ... ARG2, glBlendEquation ARG1, glsl_side_blend_mode_code)
    BLEND_MODES["Sum"] = (GL_SRC_ALPHA, GL_DST_ALPHA, GL_FUNC_ADD, 0)
    BLEND_MODES["Colourize"] = (GL_DST_COLOR, GL_DST_ALPHA, GL_FUNC_ADD, 1)
    BLEND_MODES["Mask"] = (GL_ZERO, GL_SRC_ALPHA, GL_FUNC_ADD, 2)
    BLEND_MODES["Overlay"] = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_FUNC_ADD, 3)
    BLEND_MODES_3D = dict()  # blend mode template: (glBlendFunc ARG1, ... ARG2, glBlendEquation ARG1, glsl_side_blend_mode_code)
    BLEND_MODES_3D["Halo"] = (GL_SRC_ALPHA, GL_DST_ALPHA, GL_FUNC_ADD, 0)
    BLEND_MODES_3D["Threshold"] = (GL_SRC_ALPHA, GL_DST_ALPHA, GL_FUNC_ADD, 1)
    BLEND_MODES_LIST = list(BLEND_MODES.keys())
    BLEND_MODES_LIST_3D = list(BLEND_MODES_3D.keys())

    pick_tab_index_datasets_segs = False
    VIEW_3D_PIVOT_SPEED = 0.3
    VIEW_3D_MOVE_SPEED = 100.0
    PICKING_FRAME_ALPHA = 1.0
    SELECTED_RENDER_STYLE = 1
    RENDER_STYLES = ["Cartoon", "Phong", "Flat", "Misc."]
    RENDER_BOX = False
    RENDER_PARTICLES_XRAY = False
    RENDER_SILHOUETTES = True
    RENDER_SILHOUETTES_THRESHOLD = 0.1
    RENDER_SILHOUETTES_ALPHA = 0.15
    RENDER_CLEAR_COLOUR = cfg.COLOUR_WINDOW_BACKGROUND[:3]
    RENDER_LIGHT_COLOUR = (1.0, 1.0, 1.0)
    VIEW_REQUIRES_UPDATE = True

    FRAME_TEXTURE_REQUIRES_UPDATE = False
    OVERLAY_ALPHA = 1.0
    OVERLAY_INTENSITY = 1.0
    OVERLAY_BLEND_MODE = 0
    OVERLAY_BLEND_MODE_3D = 0

    LIGHT_SPOT = None
    LIGHT_AMBIENT_STRENGTH = 0.5

    DATASETS_WINDOW_EXPANDED = False
    DATASETS_WINDOW_EXPANDED_HEIGHT = 500
    DATASETS_EXPORT_PANEL_EXPANDED = False
    DATASETS_EXPORT_PANEL_EXPANDED_HEIGHT = 300
    DATASETS_EXPORT_SELECT_ALL = False

    DATASETS_PICK_PANEL_EXPANDED = False
    DATASETS_PICK_PANEL_EXPANDED_HEIGHT = 300
    DATASETS_PICK_SELECT_ALL = False

    EXTRACT_THRESHOLD = 128
    EXTRACT_MIN_WEIGHT = 1000
    EXTRACT_MIN_SPACING = 10.0  # Angstrom
    EXTRACT_STAR_FILE = True
    EXTRACT_SELECTED_FEATURE_TITLE = ""
    EXTRACT_ALL = True
    queued_exports = list()
    export_dir = ""

    trainset_apix = 10.0
    seg_folder = ""

    SHOW_BOOT_SPRITE = True
    ICON = Image.open(os.path.join(cfg.root, "icons", "LOGO_Pom_128.png"))

    FEATURE_IMPORT_MRC_THRESHOLD = 128
    PATH_VIEWER_MISSING_DICT = dict()
    PATH_VIEWER_OPEN = False
    PATH_VIEWER_OPEN_FIND = ""
    PATH_VIEWER_OPEN_REPLACE = ""
    SHOW_IMGUI_DEBUG = False

    MENU_BAR_END_X = 400.0          # x where the main menu bar items end (party buttons anchor here)
    MENU_BAR_H = 22.0
    FEATURE_LIB_OPEN = False
    FEATURE_LIB_OPEN_INCLUDE_SESSION = False
    FEATURE_LIB_ANNOTATION = True   # whether the feature library panel shows the annotation (True) or the rendering (False) settings.
    FEATURE_LIB_HIDE_DEACTIVATED_FEATURES = False
    FEATURE_LIB_HIGHLIGHT_TEXT = ''
    FEATURE_LIB_SORT_MODE = 0  # 0 = Alphabetically, 1 = By Hue
    FEATURE_LIB_RENAME_TEXT = ''  # edit buffer for renaming the active sub-library.
    FEATURE_LIB_RENAME_TARGET = None  # which library FEATURE_LIB_RENAME_TEXT applies to; reset when the active library changes.

    POM_SYNCHRONIZE_INTERVAL = 0.5  # seconds
    POM_SYNCHRONIZE_TIMER = 0

    FORCE_SELECT_TAB = None
    FORCE_SELECT_TAB_NEXT = None  # like FORCE_SELECT_TAB, but applied on the *next* frame. Needed when a tab switch is requested from within a tab's own content (e.g. 'copy output to annotation'), which runs after that frame's Annotation tab item has already been drawn.

    def __init__(self, window, imgui_context, imgui_impl):
        cfg.start_log()
        self.window = window
        self.window.clear_color = cfg.COLOUR_WINDOW_BACKGROUND
        self.window.make_current()
        self.imgui_context = imgui_context
        self.imgui_implementation = imgui_impl

        self.camera = Camera()
        self.camera3d = Camera3D()
        SegmentationEditor.LIGHT_SPOT = Light3D()
        SegmentationEditor.DEFAULT_ZOOM = cfg.window_height / SegmentationEditor.DEFAULT_HORIZONTAL_FOV_WIDTH  # now it is DEFAULT_HORIZONTAL_FOV_WIDTH
        SegmentationEditor.DEFAULT_WORLD_PIXEL_SIZE = 1.0 / SegmentationEditor.DEFAULT_ZOOM
        self.camera.zoom = SegmentationEditor.DEFAULT_ZOOM
        SegmentationEditor.renderer = Renderer()
        self.filters = list()
        self.active_tab = "Segmentation"

        # training dataset params
        self.all_feature_names = list()
        self.feature_colour_dict = dict()
        self.trainset_feature_selection = dict()
        self.trainset_selection = list()
        self.trainset_num_boxes_positive = 0
        self.trainset_num_boxes_negative = 0
        self.trainset_boxsize = 64
        self.trainset_boxdepth = 1
        self.show_trainset_boxes = False
        self.active_trainset_exports = list()

        # drop files
        self.incoming_files = list()

        # export & extract
        self.export_limit_range = True
        self.export_overlays = False
        self.export_batch_size = 1

        # crop handles
        self.crop_handles = list()

        # picking / 3d renders
        self.pick_box_va = VertexArray(attribute_format="xyz")  # render box lines
        self.pick_box_quad_va = VertexArray(attribute_format="xyz")  # render box faces
        self.pick_box_va.update(VertexBuffer([0.0, 0.0]), IndexBuffer([]))
        self.pick_box_quad_va.update(VertexBuffer([0.0, 0.0]), IndexBuffer([]))
        self.pick_overlay_3d = True

        for i in range(4):
            self.crop_handles.append(WorldSpaceIcon(i))

        if True:
            icon_dir = os.path.join(cfg.root, "icons")

            self.icon_close = Texture(format="rgba32f")
            pxd_icon_close = np.asarray(Image.open(os.path.join(icon_dir, "icon_close_256.png"))).astype(np.float32) / 255.0
            self.icon_close.update(pxd_icon_close)
            self.icon_close.set_linear_interpolation()

            self.icon_stop = Texture(format="rgba32f")
            pxd_icon_stop = np.asarray(Image.open(os.path.join(icon_dir, "icon_stop_256.png"))).astype(np.float32) / 255.0
            self.icon_stop.update(pxd_icon_stop)
            self.icon_stop.set_linear_interpolation()

            self.icon_obj = Texture(format="rgba32f")
            self.icon_blender = Texture(format="rgba32f")
            self.icon_chimerax = Texture(format="rgba32f")
            pxd = np.asarray(Image.open(os.path.join(icon_dir, "icon_obj_256.png"))).astype(np.float32) / 255.0
            self.icon_obj.update(pxd)
            pxd = np.asarray(Image.open(os.path.join(icon_dir, "icon_blender_256.png"))).astype(np.float32) / 255.0
            self.icon_blender.update(pxd)
            pxd = np.asarray(Image.open(os.path.join(icon_dir, "icon_chimerax_256.png"))).astype(np.float32) / 255.0
            self.icon_chimerax.update(pxd)
            self.icon_obj.set_linear_interpolation()
            self.icon_chimerax.set_linear_interpolation()
            self.icon_blender.set_linear_interpolation()

            self.boot_sprite_texture = Texture(format="rgba32f")
            pxd = np.asarray(Image.open(os.path.join(icon_dir, "ais_boot_sprite.png"))).astype(np.float32) / 255.0
            self.boot_sprite_texture.update(pxd)
            self.boot_sprite_width, self.boot_sprite_height = pxd.shape[0:2]
            self.boot_sprite_texture.set_linear_interpolation()

            self.progression_icon_texture = Texture(format="rgba32f")
            pxd = np.asarray(Image.open(os.path.join(icon_dir, "ais_boot_sprite.png"))).astype(np.float32) / 255.0
            self.progression_icon_texture.update(pxd)
            self.progression_icon_texture.set_linear_interpolation()
            progression.set_panel_icon(self.progression_icon_texture.renderer_id)

            # Bake a large font for crisp level-up text; falls back to upscaling if unavailable.
            _font_path = progression.big_font_path()
            if _font_path is not None:
                try:
                    _big_font = imgui.get_io().fonts.add_font_from_file_ttf(_font_path, progression.BIG_FONT_PX)
                    self.imgui_implementation.refresh_font_texture()
                    progression.set_big_font(_big_font)
                except Exception as e:
                    cfg.set_error(e, "Could not load progression display font; using fallback.")

    @staticmethod
    def set_active_dataset(dataset):
        SegmentationEditor.pick_tab_index_datasets_segs = True
        cfg.se_active_frame = dataset
        SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
        cfg.se_active_frame.slice_changed = True
        if len(cfg.se_frames) == 1:
            SegmentationEditor.trainset_apix = cfg.se_active_frame.pixel_size * 10.0
            SegmentationEditor.seg_folder = os.path.dirname(cfg.se_active_frame.path)
        SegmentationEditor.renderer.fbo1.set_size(dataset.width, dataset.height)
        SegmentationEditor.renderer.fbo2.set_size(dataset.width, dataset.height)
        SegmentationEditor.renderer.fbo3.set_size(dataset.width, dataset.height)
        if dataset.interpolate:
            SegmentationEditor.renderer.fbo1.texture.set_linear_interpolation()
            SegmentationEditor.renderer.fbo2.texture.set_linear_interpolation()
        else:
            SegmentationEditor.renderer.fbo1.texture.set_no_interpolation()
            SegmentationEditor.renderer.fbo2.texture.set_no_interpolation()

    def on_update(self):
        imgui.set_current_context(self.imgui_context)
        imgui.CONFIG_DOCKING_ENABLE = True  # maayyyybe not?

        self.window.make_current()
        self.window.set_full_viewport()
        if self.window.focused:
            self.imgui_implementation.process_inputs()

        if EMBEDDED:
            if not imgui.get_io().want_capture_keyboard and imgui.is_key_pressed(glfw.KEY_TAB):
                if SegmentationEditor.is_shift_down():
                    scn_cfg.active_editor = (scn_cfg.active_editor - 1) % len(scn_cfg.editors)
                else:
                    scn_cfg.active_editor = (scn_cfg.active_editor + 1) % len(scn_cfg.editors)

        for filepath in self.window.dropped_files:
            if os.path.splitext(filepath)[-1] == cfg.filetype_semodel:
                self.load_model(filepath)
                SegmentationEditor.FORCE_SELECT_TAB = 1
            else:
                self.import_dataset(filepath)

        if cfg.settings["POM_SYNCHRONIZE"]:
            SegmentationEditor.POM_SYNCHRONIZE_TIMER += self.window.delta_time
            if SegmentationEditor.POM_SYNCHRONIZE_TIMER > SegmentationEditor.POM_SYNCHRONIZE_INTERVAL:
                try:
                    self.pom_synchronize()
                except Exception as e:
                    cfg.set_error(e, f"Error during a Pom synchronization call.")
                SegmentationEditor.POM_SYNCHRONIZE_TIMER = 0

        self.window.on_update()

        if self.window.window_size_changed:
            cfg.window_width = self.window.width
            cfg.window_height = self.window.height
            self.camera.set_projection_matrix(cfg.window_width, cfg.window_height)
            self.camera3d.set_projection_matrix(cfg.window_width, cfg.window_height)

        if self.active_tab in ["Export", "Models"] and cfg.se_active_frame is not None and cfg.se_active_frame.slice_changed:
            emissions = list()
            absorbing_models = list()
            # launch the models
            for model in cfg.se_models:
                model.set_slice(cfg.se_active_frame.get_slice(model_depth=model.model_depth), cfg.se_active_frame.pixel_size, cfg.se_active_frame.get_roi_indices(), cfg.se_active_frame.data.shape)
                if model.active and model.compiled:
                    # applying a model to a new slice: XP orb from the model's panel, and a background shape
                    progression.award_inference(model.title, model.colour, screen_xy=getattr(model, "_panel_xy", None))
                    progression.background_spawn(tuple(model.colour), throttle=True, siblings=[tuple(mm.colour) for mm in cfg.se_models if mm is not model])
                if model.emit:
                    emissions.append(model.data)
                if model.absorb:
                    absorbing_models.append(model)
            # have models compete for pixels
            if len(emissions) > 0 and len(absorbing_models) > 0:
                # have models compete
                stacked_array = np.stack(emissions)
                maxima = np.max(stacked_array, axis=0)
                for model in absorbing_models:
                    model.data[model.data < maxima] = 0
            # apply interactions
            for model in cfg.se_models:
                for interaction in model.interactions:
                    interaction.apply(cfg.se_active_frame.pixel_size)
                model.update_texture()
            cfg.se_active_frame.slice_changed = False

        imgui.get_io().display_size = self.window.width, self.window.height
        imgui.new_frame()

        if SegmentationEditor.queued_exports:
            if SegmentationEditor.queued_exports[0].process.progress >= 1.0:
                SegmentationEditor.queued_exports.pop(0)
            if SegmentationEditor.queued_exports:
                if SegmentationEditor.queued_exports[0].process.progress == 0.0:
                    SegmentationEditor.queued_exports[0].start()

        # GUI calls

        if cfg.se_active_frame is not None and cfg.se_active_frame.title == "New template":
            cfg.se_active_frame.transform.translation[0] *= 0.95
            cfg.se_active_frame.transform.translation[1] *= 0.95
        if not self.window.is_minimized():
            self.camera_control()
            self.camera.on_update()
            self.camera3d.on_update()
            # living background: GL pre-pass, drawn behind the tomogram
            _bg = None
            if not cfg.settings.get("PROGRESSION_HIDE", True):
                _bg = progression.background_frame(self.window.delta_time, self.window.width, self.window.height, self.camera, self.window.cursor_pos, imgui.is_mouse_down(0))
            if _bg is not None:
                SegmentationEditor.renderer.render_background(_bg[0], _bg[1], _bg[2], (self.window.width, self.window.height), _bg[3])
            self.gui_main()
            SegmentationEditor.renderer.render_draw_list(self.camera)
            self.input()

            _prog_hidden = cfg.settings.get("PROGRESSION_HIDE", True)
            progression.set_cursor_pos(self.window.cursor_pos[0], self.window.cursor_pos[1])
            progression.tick_particles(self.window.delta_time)
            if not _prog_hidden:
                progression.render_xp_hud(self.window.width, self.window.height)
                progression.render_level_up(self.window.width, self.window.height)
                progression.render_skills_panel()
                progression.draw_particles(self.camera, self.window.height)
            self.render_party_buttons()
            progression.maybe_save()

        imgui.render()
        self.imgui_implementation.render(imgui.get_draw_data())
        imgui.CONFIG_DOCKING_ENABLE = False

    def input(self):
        if self.active_tab not in ["Segmentation", "Render"] and cfg.se_active_frame is not None and cfg.se_active_frame.crop:
            SegmentationEditor.renderer.render_crop_handles(cfg.se_active_frame, self.camera, self.crop_handles)
        if imgui.get_io().want_capture_keyboard:
            return
        if cfg.se_active_frame is not None:
            sef = cfg.se_active_frame
            if imgui.is_key_pressed(glfw.KEY_C):
                sef.crop = not sef.crop
                if not sef.crop:
                    sef.crop_roi = [0, 0, sef.width, sef.height]
            if imgui.is_key_pressed(glfw.KEY_S) and SegmentationEditor.is_ctrl_down():
                SegmentationEditor.save_dataset(dialog=False)
            if imgui.is_key_pressed(glfw.KEY_I):
                if SegmentationEditor.is_shift_down():
                    sef.interpolate = not sef.interpolate
                    if sef.interpolate:
                        sef.texture.set_linear_interpolation()
                        SegmentationEditor.renderer.fbo1.texture.set_linear_interpolation()
                        SegmentationEditor.renderer.fbo2.texture.set_linear_interpolation()
                    else:
                        sef.texture.set_no_interpolation()
                        SegmentationEditor.renderer.fbo1.texture.set_no_interpolation()
                        SegmentationEditor.renderer.fbo2.texture.set_no_interpolation()
                else:
                    sef.invert = not sef.invert
            if imgui.is_key_pressed(glfw.KEY_O):
                SegmentationEditor.OVERLAY_ALPHA = float(not bool(SegmentationEditor.OVERLAY_ALPHA))
            if imgui.is_key_pressed(glfw.KEY_A) and not SegmentationEditor.is_ctrl_down():
                sef.autocontrast = not sef.autocontrast
                if sef.autocontrast:
                    sef.compute_autocontrast()
            if imgui.is_key_pressed(glfw.KEY_Z) and self.filters:
                # Toggle all filters on/off; works whether or not the Filters panel is open.
                any_enabled = any(flt.enabled for flt in self.filters)
                for flt in self.filters:
                    flt.enabled = not any_enabled
                SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
            if imgui.is_key_pressed(glfw.KEY_SPACE):
                self.camera = Camera()
                self.camera3d = Camera3D()
                self.camera.zoom = SegmentationEditor.DEFAULT_ZOOM
                SegmentationEditor.VIEW_REQUIRES_UPDATE = True
            if self.active_tab == "Segmentation" and cfg.se_active_frame.active_feature is not None:
                if imgui.is_key_pressed(glfw.KEY_X):
                    cfg.se_active_frame.active_feature.hide = not cfg.se_active_frame.active_feature.hide
                if imgui.is_key_pressed(glfw.KEY_F):
                    if not SegmentationEditor.is_shift_down():
                        cfg.se_active_frame.active_feature.magic = not cfg.se_active_frame.active_feature.magic
                    else:
                        cfg.se_active_frame.active_feature.contour = not cfg.se_active_frame.active_feature.contour
                if cfg.se_active_frame.active_feature.magic:
                    if imgui.is_key_pressed(glfw.KEY_MINUS):
                        cfg.se_active_frame.active_feature.magic_strength -= 5.0
                        cfg.se_active_frame.active_feature.magic_strength = max([5.0, cfg.se_active_frame.active_feature.magic_strength])
                    elif imgui.is_key_pressed(glfw.KEY_EQUAL):
                        cfg.se_active_frame.active_feature.magic_strength += 5.0
                        cfg.se_active_frame.active_feature.magic_strength = min([100.0, cfg.se_active_frame.active_feature.magic_strength])


        # key input
        active_frame = cfg.se_active_frame
        if active_frame in cfg.se_frames:
            if imgui.is_key_pressed(glfw.KEY_UP):
                idx = cfg.se_frames.index(active_frame) - 1
                idx = max(0, idx)
                SegmentationEditor.set_active_dataset(cfg.se_frames[idx])
            elif imgui.is_key_pressed(glfw.KEY_DOWN):
                idx = cfg.se_frames.index(active_frame) + 1
                idx = min(idx, len(cfg.se_frames) - 1)
                SegmentationEditor.set_active_dataset(cfg.se_frames[idx])
            elif imgui.is_key_pressed(glfw.KEY_LEFT, True) and not SegmentationEditor.is_ctrl_down():
                active_frame.set_slice(active_frame.current_slice - (5 if SegmentationEditor.is_shift_down() else 1))
                SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
            elif imgui.is_key_pressed(glfw.KEY_RIGHT, True) and not SegmentationEditor.is_ctrl_down():
                active_frame.set_slice(active_frame.current_slice + (5 if SegmentationEditor.is_shift_down() else 1))
                SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
        active_feature = None
        if active_frame is not None:
            active_feature = cfg.se_active_frame.active_feature

        # Key inputs that affect the active feature:
        if active_frame is not None:
            if not SegmentationEditor.is_shift_down() and not SegmentationEditor.is_ctrl_down() and active_frame is not None:
                if self.window.scroll_delta[1] != 0.0:
                    idx = int(active_frame.current_slice - self.window.scroll_delta[1])
                    idx = idx % active_frame.n_slices
                    active_frame.set_slice(idx)
                    SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True

        if self.active_tab == "Segmentation":
            if active_feature is not None:
                if SegmentationEditor.is_ctrl_down() and active_feature is not None:
                    active_feature.brush_size += 0.5 * self.window.scroll_delta[1]
                    active_feature.brush_size = max([1, active_feature.brush_size])
                if imgui.is_key_pressed(glfw.KEY_S) and not imgui.is_key_down(glfw.KEY_LEFT_CONTROL):
                    idx = 0 if active_feature not in active_frame.features else active_frame.features.index(active_feature)
                    idx = (idx + 1) % len(active_frame.features)
                    cfg.se_active_frame.active_feature = cfg.se_active_frame.features[idx]
                elif imgui.is_key_pressed(glfw.KEY_W):
                    idx = 0 if active_feature not in active_frame.features else active_frame.features.index(active_feature)
                    idx = (idx - 1) % len(active_frame.features)
                    cfg.se_active_frame.active_feature = cfg.se_active_frame.features[idx]

            # Drawing / mouse input
            if active_feature is not None and not imgui.get_io().want_capture_mouse:
                cursor_world_position = self.camera.cursor_to_world_position(self.window.cursor_pos)
                pixel_coordinate = active_feature.parent.world_to_pixel_coordinate(cursor_world_position)
                _pframe = active_feature.parent
                _inside_frame = (0 <= pixel_coordinate[0] < _pframe.width and 0 <= pixel_coordinate[1] < _pframe.height)

                if not SegmentationEditor.is_shift_down():
                    _cx, _cy = self.window.cursor_pos[0], self.window.cursor_pos[1]
                    _wx, _wy = cursor_world_position[0], cursor_world_position[1]
                    _ftr_color = tuple(active_feature.colour)
                    _skill_lvl = progression.get_profile().skill_level(active_feature.title)
                    _brush_radius_world = active_feature.brush_size * active_feature.parent.pixel_size
                    if imgui.is_mouse_down(0):
                        active_feature.hide = False
                        if active_feature.magic:
                            try:
                                Brush.apply_magic(active_feature, active_feature.parent.rendered_data, pixel_coordinate)
                            except Exception as e:
                                pass  # defensive guard around the flood-fill brush; the edge-of-image case is handled in apply_magic.
                        else:
                            Brush.apply_circular(active_feature, pixel_coordinate, True)
                        if _inside_frame:   # no reward for painting off the tomogram
                            progression.award(skill=active_feature.title, xp=1, color=_ftr_color, rate_limit=True, cursor_pos=(_cx, _cy), orb_radius=_brush_radius_world * self.camera.zoom)
                            progression.emit_brush_trail(_wx, _wy, _brush_radius_world, _ftr_color, skill_level=_skill_lvl)
                            progression.background_spawn(_ftr_color)   # drop a background shape in the tool's colour
                    elif imgui.is_mouse_down(1):
                        Brush.apply_circular(active_feature, pixel_coordinate, False)
                else:
                    if not SegmentationEditor.is_ctrl_down():
                        if imgui.is_mouse_clicked(0):
                            active_feature.add_box(pixel_coordinate)
                            if _inside_frame:   # no reward for boxing off the tomogram
                                _skill_lvl_box = progression.get_profile().skill_level(active_feature.title)
                                progression.award(skill=active_feature.title, xp=5, color=tuple(active_feature.colour), cursor_pos=(self.window.cursor_pos[0], self.window.cursor_pos[1]), orb_radius=active_feature.box_size_nm * self.camera.zoom * 0.5)
                                progression.emit_box_burst(cursor_world_position[0], cursor_world_position[1], active_feature.box_size_nm, tuple(active_feature.colour), skill_level=_skill_lvl_box)
                                progression.background_pulse("box", tuple(active_feature.colour))
                        elif imgui.is_mouse_clicked(1):
                            active_feature.remove_box(pixel_coordinate)
            if cfg.se_active_frame and SegmentationEditor.is_shift_down() and SegmentationEditor.is_ctrl_down():
                if imgui.is_mouse_clicked(0):
                    cursor_world_position = self.camera.cursor_to_world_position(self.window.cursor_pos)
                    pixel_coordinate = cfg.se_active_frame.world_to_pixel_coordinate(cursor_world_position)
                    try:
                        self.pom_pick_template(pixel_coordinate, cursor_world_position)
                    except Exception as e:
                        cfg.set_error(e, "Could not extract subtomogram at chosen position.")
        elif self.active_tab != "Render" and active_frame is not None and active_frame.crop and not imgui.get_io().want_capture_mouse:
            if not self.crop_handles[0].moving_entire_roi:
                any_handle_active = False
                for h in self.crop_handles:
                    if not h.active and h.is_hovered(self.camera, self.window.cursor_pos) and imgui.is_mouse_clicked(0):
                        h.active = True
                        any_handle_active = False
                    if h.active:
                        any_handle_active = True
                        world_pos = self.camera.cursor_to_world_position(self.window.cursor_pos)
                        pixel_pos = active_frame.world_to_pixel_coordinate(world_pos)
                        h.affect_crop(pixel_pos)
                        if not imgui.is_mouse_down(0):
                            h.active = False
                            h.convert_crop_roi_to_integers()
                            cfg.se_active_frame.slice_changed = True  # not really, but this flag is also used to trigger a model update.
                if not any_handle_active and imgui.is_mouse_clicked(0): ## TODO: only when mouse is clicked within ROI!
                    self.crop_handles[0].moving_entire_roi = True
                    self.crop_handles[0].convert_crop_roi_to_integers()
            else:
                if not imgui.is_mouse_down(0):
                    self.crop_handles[0].moving_entire_roi = False
                    self.crop_handles[0].convert_crop_roi_to_integers()
                    cfg.se_active_frame.slice_changed = True  # not really, but this flag is also used to trigger a model update.
                else:
                    world_pos_old = self.camera.cursor_to_world_position(self.window.cursor_pos_previous_frame)
                    world_pos = self.camera.cursor_to_world_position(self.window.cursor_pos)
                    world_delta = np.array([-(world_pos_old[0] - world_pos[0]), world_pos_old[1] - world_pos[1]]) / cfg.se_active_frame.pixel_size
                    self.crop_handles[0].move_crop_roi(world_delta[0], world_delta[1])

    def import_dataset(self, filename):
        # TODO: upon import, if scNodes, if has overlay and overlay.clem_frame.path found in any CLEMFrame's path, link CLEMFrame and SEFrame s.t. overlay can be updated.
        SegmentationEditor.SHOW_BOOT_SPRITE = False
        print(f'Importing {filename}')
        if isinstance(filename, str):
            filename = (filename, )
        if not isinstance(filename, tuple):
            return
        if not os.path.exists(filename[0]):
            print(f'Could not find file {filename[0]}')
            return
        for f in filename:
            try:
                _, ext = os.path.splitext(f)
                if ext == ".mrc":
                    cfg.se_frames.append(SEFrame(f))
                    SegmentationEditor.set_active_dataset(cfg.se_frames[-1])
                    self.parse_available_features()
                elif ext == cfg.filetype_segmentation:
                    with open(f, 'rb') as pickle_file:
                        seframe = pickle.load(pickle_file)
                        seframe.on_load()
                        seframe.slice_changed = False
                        cfg.se_frames.append(seframe)
                        SegmentationEditor.set_active_dataset(cfg.se_frames[-1])
                        if seframe.includes_map:
                            seframe.path = f[:-len(cfg.filetype_segmentation)]+".mrc"  # 'virtual' file path, pointing at the location of the .scns file but ending with .mrc s.t. model output regex doesn't get messed up
                        seframe.scns_path = f
                    self.parse_available_features()
                if ext in (".mrc", cfg.filetype_segmentation):
                    cfg.push_recent_dataset(f)
            except Exception as e:
                cfg.set_error(e, f"Error importing dataset {f}, see details below:")

    def pom_synchronize(self):
        # check command dir, if file then read it and follow the command.
        cmd_path = os.path.join(cfg.settings["POM_COMMAND_DIR"], "pom_to_ais.cmd")

        if os.path.exists(cmd_path):
            with open(cmd_path, 'r') as f:
                lines = f.readlines()
                print(lines)
            os.remove(cmd_path)

            for line in lines:
                line = line.split("\n")[0]
                bars = line.split("\t")
                # only one command for now:
                #   open <filepath> slice <n>
                if bars[0] == "open":
                    if os.path.exists('C:/Users/Mart Last'):  # (mart specific hack 260408)
                        bars[1] = bars[1].replace('/cephfs/mlast', 'Z:\\')
                        print(f"trigger: substitute '/cephfs/mlast' > 'Z:\\'")
                    filetype = os.path.splitext(bars[1])[-1]
                    if filetype in ['.mrc', cfg.filetype_segmentation]:
                        dataset_already_imported = False
                        tomo_name = os.path.splitext(os.path.basename(bars[1]))[0]
                        for f in cfg.se_frames:
                            if tomo_name == os.path.splitext(os.path.basename(f.path))[0]:
                                SegmentationEditor.set_active_dataset(f)
                                if "slice" in bars:
                                    cfg.se_active_frame.set_slice(int(bars[3]))
                                dataset_already_imported = True
                        if not dataset_already_imported:
                            self.import_dataset(bars[1])
                            if "slice" in bars:
                                cfg.se_frames[-1].set_slice(int(bars[3]))
                    elif filetype in [cfg.filetype_semodel]:
                        SegmentationEditor.load_model(bars[1])

            self.window.bring_to_front()

    def pom_pick_template(self, coordinate, world_position=np.zeros(3)):
        volume = mrcfile.mmap(cfg.se_active_frame.path).data
        j = cfg.se_active_frame.current_slice
        k = coordinate[1]
        l = coordinate[0]
        s = cfg.settings["POM_TEMPLATE_PX_SIZE"] // 2
        template_volume = volume[j-s:j+s, k-s:k+s, l-s:l+s]
        template_temp_path = os.path.join(os.path.expanduser("~"), ".Ais", "temp.mrc")
        print(template_volume.shape)
        if template_volume.shape != (s*2, s*2, s*2):
            raise Exception(f"Not enough data for template of size {s*2}**3")
        with mrcfile.new(template_temp_path, overwrite=True) as f:
            f.set_data(template_volume)
            f.voxel_size = cfg.se_active_frame.pixel_size * 10.0
        parent_se_frame = cfg.se_active_frame
        self.import_dataset(template_temp_path)
        cfg.se_active_frame.title = "New template"
        cfg.se_active_frame.invert = parent_se_frame.invert
        cfg.se_active_frame.autocontrast = parent_se_frame.autocontrast
        cfg.se_active_frame.contrast_lims = parent_se_frame.contrast_lims
        cfg.se_active_frame.interpolate = parent_se_frame.interpolate
        cfg.se_active_frame.include_map()
        cfg.se_active_frame.features.append(Segmentation(cfg.se_active_frame, f"Template mask"))
        cfg.se_active_frame.features[-1].contour = True
        cfg.se_active_frame.features[-1].colour = (1.0, 1.0, 1.0)
        cfg.se_active_frame.transform.translation = [world_position[0] + cfg.se_active_frame.transform.translation[0], world_position[1] + cfg.se_active_frame.transform.translation[1]]

    @staticmethod
    def _push_slider_grab(colour):
        # In dark mode, colour a feature/model's slider grab handles with its own colour. Returns pushed count.
        if not cfg.settings.get("DARK_MODE", False):
            return 0
        c = tuple(float(x) for x in colour[:3])
        hi = (min(1.0, c[0] + 0.25), min(1.0, c[1] + 0.25), min(1.0, c[2] + 0.25))
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *c, 1.0)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *hi, 1.0)
        return 2

    @staticmethod
    def save_dataset(dialog=False):
        try:
            default_name = cfg.se_active_frame.path[:-4]
            if dialog:
                filename = filedialog.asksaveasfilename(
                    filetypes=[("Ais segmentation", f"{cfg.filetype_segmentation}")],
                    initialfile=default_name)
            else:
                filename = default_name
                if hasattr(cfg.se_active_frame, "scns_path"):
                    if cfg.se_active_frame.scns_path != "n/a":
                        filename = cfg.se_active_frame.scns_path
            if filename != '':
                if filename[-len(cfg.filetype_segmentation):] != cfg.filetype_segmentation:
                    filename += cfg.filetype_segmentation
                if hasattr(cfg.se_active_frame, "scns_path"):
                    cfg.se_active_frame.scns_path = filename
                with open(filename, 'wb') as pickle_file:
                    pickle.dump(cfg.se_active_frame, pickle_file)
                    print(f"Saved {filename}")
                cfg.push_recent_dataset(filename)   # a saved .scns joins the recent datasets (and evicts its .mrc twin)


        except Exception as e:
            cfg.set_error(e, "Could not save dialog - see details below.")

    def gui_main(self):
        if True:
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *cfg.COLOUR_PANEL_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT)
            imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *cfg.COLOUR_TITLE_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *cfg.COLOUR_TITLE_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *cfg.COLOUR_TITLE_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *cfg.COLOUR_TITLE_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *cfg.COLOUR_TITLE_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_FRAME_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *cfg.COLOUR_FRAME_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_FRAME_ACTIVE)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *cfg.COLOUR_FRAME_ACTIVE)
            imgui.push_style_color(imgui.COLOR_BUTTON, *cfg.COLOUR_FRAME_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *cfg.COLOUR_FRAME_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *cfg.COLOUR_FRAME_EXTRA_DARK)
            imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *cfg.COLOUR_FRAME_DARK)
            imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *cfg.COLOUR_FRAME_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB, *cfg.COLOUR_FRAME_DARK)
            imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB_HOVERED, *cfg.COLOUR_FRAME_DARK)
            imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB_ACTIVE, *cfg.COLOUR_FRAME_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_SCROLLBAR_BACKGROUND, *cfg.COLOUR_FRAME_EXTRA_DARK)
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *cfg.COLOUR_TEXT)
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, *cfg.COLOUR_TEXT)
            imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_CHECK_MARK)
            imgui.push_style_color(imgui.COLOR_MENUBAR_BACKGROUND, *cfg.COLOUR_MAIN_MENU_BAR)
            imgui.push_style_color(imgui.COLOR_HEADER, *cfg.COLOUR_HEADER)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *cfg.COLOUR_HEADER_HOVERED)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *cfg.COLOUR_HEADER_ACTIVE)
            imgui.push_style_color(imgui.COLOR_TAB, *cfg.COLOUR_HEADER_ACTIVE)
            imgui.push_style_color(imgui.COLOR_TAB_ACTIVE, *cfg.COLOUR_HEADER)
            imgui.push_style_color(imgui.COLOR_TAB_HOVERED, *cfg.COLOUR_HEADER)
            imgui.push_style_color(imgui.COLOR_DRAG_DROP_TARGET, *cfg.COLOUR_DROP_TARGET)
            imgui.push_style_color(imgui.COLOR_SCROLLBAR_BACKGROUND, *cfg.COLOUR_WINDOW_BACKGROUND[:3], 0.0)
            imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, cfg.WINDOW_ROUNDING)

        def boot_sprite():
            if SegmentationEditor.SHOW_BOOT_SPRITE:
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *cfg.COLOUR_WINDOW_BACKGROUND[0:3], 0.0)
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *cfg.COLOUR_WINDOW_BACKGROUND[0:3], 0.0)
                imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT)

                _w = self.boot_sprite_width * 0.325
                _h = self.boot_sprite_height * 0.325
                imgui.set_next_window_position(SegmentationEditor.MAIN_WINDOW_WIDTH + (cfg.window_width - SegmentationEditor.MAIN_WINDOW_WIDTH) / 2.0 - (_w / 2.0), (cfg.window_height - _h) / 2.0 - 25)
                self.show_boot_img = imgui.begin("##boot_sprite", True,imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_BACKGROUND | imgui.WINDOW_NO_SCROLLBAR)[1]
                imgui.image(self.boot_sprite_texture.renderer_id, _w, _h)
                _boot_caption = f"{cfg.app_name} {cfg.version}"
                imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (_w - imgui.calc_text_size(_boot_caption)[0]) / 2.0)
                imgui.text(_boot_caption)
                imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *cfg.COLOUR_WINDOW_BACKGROUND)
                if imgui.begin_popup_context_window():
                    imgui.text(f"Welcome to {cfg.app_name}!")
                    imgui.text(f"version {cfg.version}\nsource: github.com/mgflast/Ais\nmanual: mgflast.github.io/Ais")
                    imgui.end_popup()
                if self.window.focused and imgui.is_mouse_clicked(glfw.MOUSE_BUTTON_LEFT) and not imgui.is_window_hovered():
                    SegmentationEditor.SHOW_BOOT_SPRITE = False
                imgui.pop_style_color(1)
                imgui.end()
                imgui.pop_style_color(3)

        def shared_gui():
            if imgui.collapsing_header("Datasets", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                panel_height = 120 if not SegmentationEditor.DATASETS_WINDOW_EXPANDED else SegmentationEditor.DATASETS_WINDOW_EXPANDED_HEIGHT
                imgui.begin_child("available_datasets", 0.0, panel_height, True, imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
                for s in cfg.se_frames:
                    imgui.push_id(f"se{s.uid}")

                    base_title = s.title.split(".")[0]
                    _change, _selected = imgui.selectable(base_title[-17:] + f" - {s.pixel_size * 10.0:.2f} A/pix ", cfg.se_active_frame == s)
                    if imgui.begin_popup_context_item("##datasetContext"):
                        if imgui.menu_item("Unlink dataset")[0]:
                            SegmentationEditor.pick_tab_index_datasets_segs = True
                            cfg.se_frames.remove(s)
                            if cfg.se_active_frame == s:
                                cfg.se_active_frame = None
                            self.parse_available_features()
                        if imgui.menu_item("Relink dataset")[0]:
                            selected_file = filedialog.askopenfilename(filetypes=[("mrcfile", ".mrc")])
                            if isinstance(selected_file, str) and os.path.exists(selected_file):
                                s.path = selected_file
                                s.title = os.path.splitext(os.path.basename(s.path))[0]
                        if imgui.menu_item("Copy path to .mrc")[0]:
                            pyperclip.copy(s.path)
                        if s.overlay is not None and s.overlay.update_function is not None and imgui.menu_item("Update overlay")[0]:
                            s.overlay.update()
                        if s.overlay is not None and imgui.menu_item("Export overlay (.tif)")[0]:
                            filename = filedialog.asksaveasfilename()
                            if filename != "":
                                try:
                                    tifffile.imwrite(os.path.splitext(filename)[0]+".tif", s.overlay.pxd[:, :, 0:3])
                                except Exception as e:
                                    cfg.set_error(e, "Could not export overlay as .tif - see below:")

                        if imgui.begin_menu("Generate binned version"):
                            path = s.path
                            for i in [2, 3, 4, 8]:
                                if imgui.menu_item(f"bin {i}")[0]:
                                    bpath = bin_mrc(path, i)
                                    self.import_dataset(bpath)
                            imgui.end_menu()
                        if imgui.begin_menu("Overrule pixel size"):
                            imgui.set_next_item_width(100)
                            pxs_ang = s.pixel_size * 10.0
                            _, pxs_ang = imgui.input_float("##Appix", s.pixel_size, 0.0, 0.0, format = f"{pxs_ang:.2f} A/pixel")
                            if _:
                                s.pixel_size = pxs_ang / 10.0
                            imgui.end_menu()
                        if imgui.menu_item("Flip")[0]:
                            with mrcfile.new(os.path.splitext(s.path)[0]+'_flipped.mrc', overwrite=True) as f:
                                volume = mrcfile.read(s.path)
                                volume = np.swapaxes(volume, 0, 1)
                                f.set_data(volume)
                                f.voxel_size = s.pixel_size * 10.0
                            self.import_dataset(os.path.splitext(s.path)[0]+'_flipped.mrc')
                        imgui.end_popup()
                    self.tooltip(f"{s.title}\nPixel size: {s.pixel_size * 10.0:.4f} Angstrom\nLocation: {s.path}")
                    if _change and _selected:
                        SegmentationEditor.set_active_dataset(s)
                        for model in cfg.se_models:
                            model.reset_textures()
                        SegmentationEditor.trainset_apix = s.pixel_size * 10.0

                    for f in s.features:
                        imgui.same_line(spacing=1)
                        if imgui.color_button(f"##{f.uid}", f.colour[0], f.colour[1], f.colour[2], 1, imgui.COLOR_EDIT_NO_TOOLTIP, 6, 13):
                            s.active_feature = f

                    imgui.pop_id()
                imgui.end_child()
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_CHECK_MARK)
                _, SegmentationEditor.DATASETS_WINDOW_EXPANDED = imgui.checkbox("expand", SegmentationEditor.DATASETS_WINDOW_EXPANDED)
                imgui.pop_style_var(3)
                imgui.pop_style_color(1)
            if imgui.collapsing_header("Filters", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                sef = cfg.se_active_frame
                if sef is not None:
                    _cw = imgui.get_content_region_available_width()
                    imgui.plot_histogram("##hist", sef.hist_vals,
                                         graph_size=(_cw, SegmentationEditor.INFO_HISTOGRAM_HEIGHT))
                    imgui.push_item_width(_cw)

                    imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                    imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                    imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                    imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                    imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_TEXT)

                    _minc, sef.contrast_lims[0] = imgui.slider_float("min", sef.contrast_lims[0], sef.hist_bins[0], sef.hist_bins[-1], format='min %.2f')
                    _maxc, sef.contrast_lims[1] = imgui.slider_float("max", sef.contrast_lims[1], sef.hist_bins[0], sef.hist_bins[-1], format='max %.2f')
                    if self.active_tab == "Render":
                        _, SegmentationEditor.PICKING_FRAME_ALPHA = imgui.slider_float("alpha", SegmentationEditor.PICKING_FRAME_ALPHA, 0.0, 1.0, format="alpha %.1f")
                    if _minc or _maxc:
                        sef.autocontrast = False
                    imgui.pop_item_width()
                    _c, sef.invert = imgui.checkbox("inverted", sef.invert)
                    imgui.same_line(spacing=16)
                    _c, sef.autocontrast = imgui.checkbox("auto", sef.autocontrast)
                    if _c and sef.autocontrast:
                        SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
                    imgui.same_line(spacing=16)
                    _c, sef.interpolate = imgui.checkbox("interpolate", sef.interpolate)
                    imgui.same_line(spacing=16)
                    if _c:
                        if sef.interpolate:
                            sef.texture.set_linear_interpolation()
                            SegmentationEditor.renderer.fbo1.texture.set_linear_interpolation()
                            SegmentationEditor.renderer.fbo2.texture.set_linear_interpolation()
                        else:
                            sef.texture.set_no_interpolation()
                            SegmentationEditor.renderer.fbo1.texture.set_no_interpolation()
                            SegmentationEditor.renderer.fbo2.texture.set_no_interpolation()
                    if self.active_tab in ["Segmentation", "Render"]:
                        imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT_DISABLED)
                        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_TEXT_DISABLED)
                    _c, sef.crop = imgui.checkbox("crop", sef.crop)
                    if _c and not sef.crop:
                        sef.crop_roi = [0, 0, sef.width, sef.height]
                    if self.active_tab in ["Segmentation", "Render"]:
                        imgui.pop_style_color(2)
                    imgui.separator()
                    fidx = 0
                    for ftr in self.filters:
                        fidx += 1
                        imgui.push_id(f"filter{fidx}")
                        cw = imgui.get_content_region_available_width()

                        # Filter type selection combo
                        imgui.set_next_item_width(cw - 60)
                        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (5, 2))
                        _c, ftrtype = imgui.combo("##filtertype", ftr.type, Filter.TYPES)
                        if _c:
                            self.filters[self.filters.index(ftr)] = Filter(ftrtype)
                            SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
                        imgui.same_line()
                        _, ftr.enabled = imgui.checkbox("##enabled", ftr.enabled)
                        if _:
                            SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
                        # Delete button
                        imgui.same_line()
                        if imgui.image_button(self.icon_close.renderer_id, 13, 13):
                            self.filters.remove(ftr)
                            SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
                        imgui.pop_style_var(1)
                        # Parameter and strength sliders
                        imgui.push_item_width(cw)
                        if Filter.PARAMETER_NAME[ftr.type] is not None:
                            _c, ftr.param = imgui.slider_float("##param", ftr.param, 0.1, 10.0, format=f"{Filter.PARAMETER_NAME[ftr.type]}: {ftr.param:.1f}")
                            if _c:
                                ftr.fill_kernel()
                                SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
                        _, ftr.strength = imgui.slider_float("##strength", ftr.strength, -1.0, 1.0, format=f"weight: {ftr.strength:.2f}")
                        if _:
                            SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
                        imgui.pop_item_width()

                        imgui.pop_id()
                    imgui.set_next_item_width(imgui.get_content_region_available_width())
                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (5, 2))
                    _c, new_filter_type = imgui.combo("##filtertype", 0, ["Add filter"] + Filter.TYPES)
                    if _c and not new_filter_type == 0:
                        self.filters.append(Filter(new_filter_type - 1))
                        SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
                    if cfg.se_active_frame.overlay is not None:
                        imgui.separator()
                        cw = imgui.get_content_region_available_width()
                        imgui.set_next_item_width(cw - 120)
                        _, SegmentationEditor.OVERLAY_ALPHA = imgui.slider_float("##alphaslider_se", SegmentationEditor.OVERLAY_ALPHA, 0.0, 1.0, format=f"overlay alpha = {SegmentationEditor.OVERLAY_ALPHA:.2f}")
                        imgui.same_line()
                        imgui.set_next_item_width(110)
                        if self.active_tab != "Render":
                            _, SegmentationEditor.OVERLAY_BLEND_MODE = imgui.combo("##overlayblending", SegmentationEditor.OVERLAY_BLEND_MODE, SegmentationEditor.BLEND_MODES_LIST)
                        else:
                            if self.pick_overlay_3d:
                                _, SegmentationEditor.OVERLAY_BLEND_MODE_3D = imgui.combo("##overlayblending", SegmentationEditor.OVERLAY_BLEND_MODE_3D, SegmentationEditor.BLEND_MODES_LIST_3D)
                            else:
                                _, SegmentationEditor.OVERLAY_BLEND_MODE = imgui.combo("##overlayblending", SegmentationEditor.OVERLAY_BLEND_MODE, SegmentationEditor.BLEND_MODES_LIST)
                            SegmentationEditor.VIEW_REQUIRES_UPDATE |= _
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                            _, self.pick_overlay_3d = imgui.checkbox("3D", self.pick_overlay_3d)
                            imgui.pop_style_var()
                            SegmentationEditor.VIEW_REQUIRES_UPDATE |= _
                            imgui.same_line()
                            imgui.set_next_item_width(imgui.get_content_region_available_width())
                            _, SegmentationEditor.OVERLAY_INTENSITY = imgui.slider_float("##inensityslider_se", SegmentationEditor.OVERLAY_INTENSITY, 0.0, 10.0, format=f"overlay intensity = {SegmentationEditor.OVERLAY_INTENSITY:.1f}")
                            SegmentationEditor.VIEW_REQUIRES_UPDATE |= _

                    imgui.pop_style_var(6)
                    imgui.pop_style_color(1)

        def segmentation_tab():

            def features_panel():
                if imgui.collapsing_header("Features", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    if cfg.se_active_frame is None:
                        return
                    features = cfg.se_active_frame.features
                    for f in features:
                        pop_active_colour = False
                        if cfg.se_active_frame.active_feature == f:
                            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *cfg.COLOUR_FRAME_ACTIVE)
                            pop_active_colour = True
                        imgui.begin_child(f"##feat_{f.uid}", 0.0, SegmentationEditor.FEATURE_PANEL_HEIGHT + 16 * f.magic, True)
                        cw = imgui.get_content_region_available_width()

                        # Colour picker
                        _, f.colour = imgui.color_edit3(f.title, *f.colour[:3], imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                        if _:
                            self.parse_available_features()
                        # Title
                        imgui.same_line()
                        imgui.set_next_item_width(cw - 45)
                        _, f.title = imgui.input_text("##title", f.title, 256, imgui.INPUT_TEXT_NO_HORIZONTAL_SCROLL | imgui.INPUT_TEXT_AUTO_SELECT_ALL)
                        if _:
                            self.parse_available_features()
                        imgui.same_line(spacing=0)
                        imgui.push_style_color(imgui.COLOR_BUTTON, *cfg.COLOUR_FRAME_BACKGROUND)
                        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 1, 1, 1, 0.1)
                        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 1, 1, 1, 0.2)
                        imgui.push_style_color(imgui.COLOR_TEXT, 0.45, 0.45, 0.42, 1.0)
                        if imgui.arrow_button("##feature_dropdown", imgui.DIRECTION_DOWN):
                            imgui.open_popup("##feature_ctx")
                        imgui.pop_style_color(4)
                        self._gui_feature_title_context_menu(f, "##feature_ctx")

                        # Alpha slider and brush size
                        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                        imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                        imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                        imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                        imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_CHECK_MARK)
                        imgui.push_item_width(cw - 40)
                        pxs = cfg.se_active_frame.pixel_size
                        _nsc = SegmentationEditor._push_slider_grab(f.colour)
                        _, f.brush_size = imgui.slider_float("brush", f.brush_size, 1.0, 25.0 / pxs, format=f"{f.brush_size:.1f} px / {f.brush_size * pxs:.1f} nm ")
                        if f.magic:
                            _, f.magic_strength = imgui.slider_float("flood", f.magic_strength, 1.0, 100.0, format=f"{f.magic_strength:.1f}%% sensitivity")
                        _, f.alpha = imgui.slider_float("alpha", f.alpha, 0.0, 1.0, format="%.2f")
                        _, f.box_size = imgui.slider_int("boxes", f.box_size, 8, 256, format=f"{f.box_size} pixel")
                        f.box_size = int(np.round(f.box_size / 32) * 32)
                        if _nsc:
                            imgui.pop_style_color(_nsc)

                        imgui.pop_item_width()
                        if _:
                            f.set_box_size(f.box_size)
                        # Show / fill checkboxes
                        _, show = imgui.checkbox("show", not f.hide)
                        f.hide = not show
                        imgui.same_line()
                        _, fill = imgui.checkbox("fill", not f.contour)
                        f.contour = not fill
                        imgui.same_line()
                        _, hide_boxes = imgui.checkbox("hide boxes", not f.show_boxes)
                        imgui.same_line()
                        _, f.magic = imgui.checkbox("flood##checkbox", f.magic)
                        f.show_boxes = not hide_boxes
                        f.contour = not fill
                        imgui.same_line()
                        imgui.pop_style_color(1)
                        # delete feature button
                        imgui.same_line(position=cw - 20)
                        delete_feature = False
                        if imgui.image_button(self.icon_close.renderer_id, 13, 13):
                            delete_feature = True
                        imgui.same_line(position=cw - 20)

                        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *cfg.COLOUR_TRANSPARENT)
                        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *cfg.COLOUR_TRANSPARENT)
                        imgui.push_style_color(imgui.COLOR_HEADER, *cfg.COLOUR_TRANSPARENT)

                        if imgui.begin_menu("##asdc"):
                            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *cfg.COLOUR_HEADER_HOVERED)
                            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *cfg.COLOUR_HEADER_ACTIVE)
                            imgui.push_style_color(imgui.COLOR_HEADER, *cfg.COLOUR_HEADER)
                            n_boxes = sum([len(boxlist) for boxlist in f.boxes.values()])
                            imgui.text(f"Active slices ({len(f.edited_slices)}, {n_boxes} boxes)")

                            imgui.begin_child("active_slices", 200, SegmentationEditor.ACTIVE_SLICES_CHILD_HEIGHT, True)
                            cw = imgui.get_content_region_available_width()
                            for i in f.edited_slices:
                                imgui.push_id(f"{f.uid}{i}")

                                slice_clicked, _ = imgui.selectable(f"Slice {i} ({len(f.boxes[i])} boxes)", f.current_slice == i, width=cw - 23)
                                if imgui.is_item_hovered():
                                    f.parent.set_slice(i)
                                    SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
                                if slice_clicked:
                                    imgui.close_current_popup()
                                imgui.same_line(position=cw - 5)
                                if imgui.image_button(self.icon_close.renderer_id, 13, 13):
                                    f.remove_slice(i)
                                imgui.pop_id()
                            imgui.end_child()
                            imgui.text("Export:")
                            cw = imgui.get_content_region_available_width()
                            if imgui.button("coords", (cw - 15) / 3, 15):
                                f.save_particle_positions()
                            imgui.same_line(spacing=5)
                            if imgui.button("slice", (cw - 15) / 3, 15):
                                f.save_current_slice()
                            imgui.same_line(spacing=5)
                            if imgui.button("volume", (cw - 15) / 3, 15):
                                f.save_volume()

                            # Import
                            imgui.push_id(f"{f.uid}_import")
                            imgui.text("Import:")
                            imgui.push_item_width((cw - 15) / 3)
                            _, SegmentationEditor.FEATURE_IMPORT_MRC_THRESHOLD = imgui.slider_int("##fthslint", SegmentationEditor.FEATURE_IMPORT_MRC_THRESHOLD, 0, 255)
                            imgui.same_line(spacing=5)
                            if imgui.button("slice", (cw - 15) / 3, 15):
                                print(1)
                                path = filedialog.askopenfilename(filetypes=[("mrcfile", f".mrc")])
                                if path != "":
                                    try:
                                        f.import_slice(path, threshold=SegmentationEditor.FEATURE_IMPORT_MRC_THRESHOLD)
                                    except Exception as e:
                                        cfg.set_error(e, f"Could not initialize Feature using mrc file {path}")
                            imgui.same_line(spacing=5)
                            if imgui.button("volume", (cw - 15) / 3, 15):
                                print(0)
                                path = filedialog.askopenfilename(filetypes=[("mrcfile", f".mrc")])
                                if path != "":
                                    try:
                                        f.import_mrc(path, threshold=SegmentationEditor.FEATURE_IMPORT_MRC_THRESHOLD)
                                    except Exception as e:
                                        cfg.set_error(e, f"Could not initialize Feature using mrc file {path}")
                            imgui.pop_style_color(3)
                            imgui.pop_id()
                            imgui.end_menu()

                        imgui.pop_style_color(3)

                        imgui.pop_style_var(5)
                        if pop_active_colour:
                            imgui.pop_style_color(1)

                        if delete_feature:
                            cfg.se_active_frame.features.remove(f)
                            if cfg.se_active_frame.active_feature == f:
                                cfg.se_active_frame.active_feature = None

                        if imgui.is_window_hovered() and imgui.is_mouse_clicked(0):
                            cfg.se_active_frame.active_feature = f


                        imgui.end_child()

                    # 'Add feature' button
                    cw = imgui.get_content_region_available_width()
                    imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)

                    imgui.new_line()
                    imgui.same_line(spacing=(cw - 120) / 2)
                    if imgui.button("Add feature", 120, 23):
                        cfg.se_active_frame.features.append(Segmentation(cfg.se_active_frame, f"Unnamed feature {len(cfg.se_active_frame.features)+1}"))
                        self.parse_available_features()
                    imgui.pop_style_var(1)


            features_panel()

        def models_tab():
            def calculate_number_of_boxes():
                self.trainset_num_boxes_positive = 0
                self.trainset_num_boxes_negative = 0
                for s in cfg.se_frames:
                    if s.sample:
                        for f in s.features:
                            if self.trainset_feature_selection[f.title] == 1:
                                self.trainset_num_boxes_positive += f.n_boxes
                            elif self.trainset_feature_selection[f.title] == -1:
                                self.trainset_num_boxes_negative += f.n_boxes


            if imgui.collapsing_header("Create a training set", None)[0]:
                self.show_trainset_boxes = True
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 2))

                imgui.text("Features of interest")
                imgui.begin_child("select_features", 0.0, 1 + len(self.all_feature_names) * 21, False, imgui.WINDOW_NO_SCROLLBAR)
                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_NEUTRAL_LIGHT)
                cw = imgui.get_content_region_available_width()
                imgui.push_item_width(cw)
                for fname in self.all_feature_names:
                    val = self.trainset_feature_selection[fname]
                    if val == 1:
                        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *cfg.COLOUR_POSITIVE)
                        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *cfg.COLOUR_POSITIVE)
                    elif val == 0:
                        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *cfg.COLOUR_NEUTRAL)
                        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *cfg.COLOUR_NEUTRAL)
                    elif val == -1:
                        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *cfg.COLOUR_NEGATIVE)
                        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *cfg.COLOUR_NEGATIVE)
                    _, self.trainset_feature_selection[fname] = imgui.slider_int(f"##{fname}", val, -1, 1, format=f"{fname}")
                    imgui.pop_style_color(2)
                imgui.pop_style_color(1)
                imgui.pop_style_var(1)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                imgui.end_child()

                imgui.text("Datasets to sample")
                imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_CHECK_MARK)
                imgui.begin_child("datasets_to_sample", 0.0, min([120, 10 + len(cfg.se_frames)*20]), True)
                for s in cfg.se_frames:
                    imgui.push_id(f"{s.uid}")
                    _, s.sample = imgui.checkbox(s.title, s.sample)
                    if _:
                        self.parse_available_features()
                    imgui.pop_id()
                imgui.end_child()
                imgui.pop_style_color()


                imgui.text("Set parameters")
                imgui.begin_child("params", 0.0, 84, True)
                cw = imgui.get_content_region_available_width()

                imgui.push_item_width(cw)
                _, self.trainset_boxsize = imgui.slider_int("##box size", self.trainset_boxsize, 8, 256, format=f"box size: {int(self.trainset_boxsize)} pixel")
                self.trainset_boxsize = np.round(self.trainset_boxsize / 32) * 32
                _, self.trainset_boxdepth = imgui.slider_int("##box depth", self.trainset_boxdepth, 1, 33, format=f"box depth: {int(self.trainset_boxdepth)} slice"+("s" if self.trainset_boxdepth>1 else ""))
                self.trainset_boxdepth += 1 if self.trainset_boxdepth % 2 == 0 else 0
                #_, SegmentationEditor.trainset_apix = imgui.slider_float("A/pix", SegmentationEditor.trainset_apix, 1.0, 20.0, format=f"{SegmentationEditor.trainset_apix:.2f}")
                imgui.pop_item_width()
                calculate_number_of_boxes()
                imgui.text(f"Positive samples: {self.trainset_num_boxes_positive}")
                imgui.text(f"Negative samples: {self.trainset_num_boxes_negative}")
                imgui.end_child()

                # progress bars
                for process in self.active_trainset_exports:
                    self._gui_background_process_progress_bar(process)
                    if process.progress == 1.0:
                        self.active_trainset_exports.remove(process)

                # 'Generate set' button
                if widgets.centred_button("Generate set", 120, 23):
                    self.launch_create_training_set()
                imgui.pop_style_var(5)
            else:
                self.show_trainset_boxes = False
            if imgui.collapsing_header("Models", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                for m in cfg.se_models:

                    # SET PANEL ID AND HEIGHT
                    imgui.push_id(f"SEModel_{m.uid}_main")
                    panel_height = 0
                    if m.active_tab == 0:
                        panel_height = SegmentationEditor.MODEL_PANEL_HEIGHT_TRAINING
                    elif m.active_tab == 1:
                        panel_height = SegmentationEditor.MODEL_PANEL_HEIGHT_PREDICTION - 16 if cfg.settings["TILED_MODE"] == 0 else SegmentationEditor.MODEL_PANEL_HEIGHT_PREDICTION
                    elif m.active_tab == 2:
                        panel_height = SegmentationEditor.MODEL_PANEL_HEIGHT_LOGIC + 57 * len(m.interactions) - 20 * (len(cfg.se_models) < 2)
                    panel_height += 10 if m.background_process_train is not None else 0
                    imgui.begin_child(f"SEModel_{m.uid}", 0.0, panel_height, True, imgui.WINDOW_NO_SCROLLBAR)
                    cw = imgui.get_content_region_available_width()
                    _mgrab = SegmentationEditor._push_slider_grab(m.colour)
                    
                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                    if m.background_process_train is None:
                        # delete button
                        if imgui.image_button(self.icon_close.renderer_id, 19, 19):
                            cfg.se_models.remove(m)
                            if cfg.se_active_model == m:
                                cfg.se_active_model = None
                            m.delete()
                    else:
                        if imgui.image_button(self.icon_stop.renderer_id, 19, 19):
                            m.background_process_train.stop()

                    imgui.pop_style_var(1)

                    # WINDOW SHARED CONTENT: CLOSE, COLOUR PICKER, TITLE, INFO STRING / ARCHITECTURE SELECTION, AND CONTEXT MENU
                    if imgui.begin_popup_context_window():
                        if imgui.begin_menu("override inference pixel size"):
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (2, 0))
                            imgui.set_next_item_width(100)
                            _, m.apix = imgui.input_float("##pxs", m.apix, 0.0, 0.0, '%.2f A/px')
                            if _:
                                m.update_info()
                            imgui.pop_style_var()
                            imgui.end_menu()
                        if imgui.begin_menu("copy output to annotation"):
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                            for feature in cfg.se_active_frame.features:
                                rgb = feature.colour
                                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 7)
                                imgui.color_button(f"##clrbutton{feature.title}", rgb[0], rgb[1], rgb[2], 1.0, 0, 14, 14)
                                imgui.pop_style_var(1)
                                imgui.same_line(spacing=5)
                                if imgui.selectable(feature.title)[1]:
                                    slice_data = (m.data > m.threshold) * 255
                                    feature.set_slice_ndarray(slice_data, cfg.se_active_frame.current_slice)
                                    SegmentationEditor.FORCE_SELECT_TAB_NEXT = 0  # deferred: we are inside the Models tab content, past this frame's Annotation tab item
                                    progression.award(skill=feature.title, xp=15, color=tuple(feature.colour), cursor_pos=(self.window.cursor_pos[0], self.window.cursor_pos[1]))
                                    progression.background_spawn(tuple(feature.colour), throttle=False, count=3)
                            imgui.pop_style_var(1)
                            imgui.end_menu()
                        imgui.end_popup()
                    imgui.same_line()
                    _, m.colour = imgui.color_edit3(m.title, *m.colour[:3], imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                    _swatch_min = imgui.get_item_rect_min()   # inference XP orbs fly from the centre of the colour swatch
                    _swatch_max = imgui.get_item_rect_max()
                    m._panel_xy = (0.5 * (_swatch_min[0] + _swatch_max[0]), 0.5 * (_swatch_min[1] + _swatch_max[1]))
                    # Title
                    imgui.same_line()
                    imgui.set_next_item_width(cw - 74)
                    _, m.title = imgui.input_text("##title", m.title, 256, imgui.INPUT_TEXT_NO_HORIZONTAL_SCROLL | imgui.INPUT_TEXT_AUTO_SELECT_ALL)
                    imgui.same_line(spacing=0)
                    imgui.push_style_color(imgui.COLOR_BUTTON, *cfg.COLOUR_FRAME_BACKGROUND)
                    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 1, 1, 1, 0.1)
                    imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 1, 1, 1, 0.2)
                    imgui.push_style_color(imgui.COLOR_TEXT, 0.45, 0.45, 0.42, 1.0)
                    if imgui.arrow_button("##model_dropdown", imgui.DIRECTION_DOWN):
                        imgui.open_popup("##model_ctx")
                    imgui.pop_style_color(4)
                    self._gui_feature_title_context_menu(m, "##model_ctx")
                    # Model selection
                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (5, 2))
                    imgui.align_text_to_frame_padding()

                    if m.compiled:
                        imgui.text(m.info)
                    else:
                        imgui.text("Model:")
                        imgui.same_line()
                        imgui.set_next_item_width(cw - 51)
                        _, m.model_enum = imgui.combo("##model_type", m.model_enum, SEModel.AVAILABLE_MODELS)
                    imgui.pop_style_var()


                    # TAB CONTENT: TRAINING / PREDICTION / INTERACTIONS
                    if imgui.begin_tab_bar("##tabs"):
                        if imgui.begin_tab_item(" Training ")[0]:
                            m.active_tab = 0
                            imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (3, 3))
                            # Train data selection
                            cw = imgui.get_content_region_available_width()
                            imgui.set_next_item_width(cw - 65)
                            _, m.train_data_path = imgui.input_text("##training_data", m.train_data_path, 256)
                            imgui.pop_style_var()
                            imgui.same_line()
                            if imgui.button("browse", 56, 19):
                                selected_file = filedialog.askopenfilename(filetypes=[("Ais training data", f"{cfg.filetype_traindata}")])
                                if selected_file:
                                    m.train_data_path = selected_file

                            # Training parameters
                            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                            imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                            imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))

                            imgui.set_next_item_width(cw)
                            _, m.epochs = imgui.slider_int("##epochs", m.epochs, 1, 50, f"{m.epochs} epoch"+("s" if m.epochs>1 else ""))
                            imgui.push_item_width((cw - 7) / 2)
                            #_, m.excess_negative = imgui.slider_int("##excessnegative", m.excess_negative, 0, 100, f"+{m.excess_negative}%% negatives")
                            _, m.batch_size = imgui.slider_int("##batchs", m.batch_size, 1, 128, f"{m.batch_size} batch size")
                            imgui.same_line()
                            _, m.n_copies = imgui.slider_int("##copies", m.n_copies, 1, 10, f"{m.n_copies} copies")
                            imgui.pop_item_width()
                            imgui.pop_style_var(1)
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 2))

                            # Load, save, train buttons.
                            block_buttons = False
                            if not m.background_process_train is None:
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_NEUTRAL_LIGHT)
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *cfg.COLOUR_NEUTRAL_LIGHT)
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_NEUTRAL_LIGHT)
                                imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_NEUTRAL)
                                block_buttons = True
                            if imgui.button("load", (cw - 16) / 3, 20):
                                if not block_buttons:
                                    model_path = filedialog.askopenfilename(filetypes=[("Ais model", f"{cfg.filetype_semodel}")])
                                    if model_path != "":
                                        m.load(model_path)
                                        cfg.push_recent("RECENT_MODELS", model_path)
                            imgui.same_line(spacing=8)
                            block_save_button = False
                            if m.model is None:
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_NEUTRAL_LIGHT)
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *cfg.COLOUR_NEUTRAL_LIGHT)
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_NEUTRAL_LIGHT)
                                imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_NEUTRAL)
                                block_save_button = True
                            if imgui.button("save", (cw - 16) / 3, 20):
                                if not block_buttons and not block_save_button:
                                    proposed_filename = f"{m.apix:.2f}_{m.box_size}_{m.loss:.4f}_{m.title}"
                                    try:
                                        model_path = filedialog.asksaveasfilename(filetypes=[("Ais model", f"{cfg.filetype_semodel}")], initialfile=proposed_filename)
                                        if model_path != "" and type(model_path) == str:
                                            if model_path[-len(cfg.filetype_semodel):] != cfg.filetype_semodel:
                                                model_path += cfg.filetype_semodel
                                            m.save(model_path, None if cfg.se_active_frame is None else cfg.se_active_frame.get_slice(model_depth=m.model_depth))
                                    except Exception as e:
                                        cfg.set_error(e, "Could not save model. See details below.")
                            if block_save_button:
                                imgui.pop_style_color(4)
                            imgui.same_line(spacing=8)
                            if imgui.button("train", (cw - 16) / 3, 20):
                                if not block_buttons:
                                    m.bcprms["VALIDATION_SPLIT"] = int(cfg.settings["VALIDATION_SPLIT"]) / 100.0
                                    m.train()
                            if block_buttons:
                                imgui.pop_style_color(4)

                            imgui.pop_style_var(5)
                            imgui.end_tab_item()

                        if imgui.begin_tab_item(" Prediction ")[0]:
                            m.active_tab = 1
                            # Checkboxes and sliders
                            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                            imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                            imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                            imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                            imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_TEXT)

                            imgui.push_item_width(imgui.get_content_region_available_width())
                            _, m.alpha = imgui.slider_float("##alpha", m.alpha, 0.0, 1.0, format=f"{m.alpha:.2f} alpha")
                            if cfg.settings["TILED_MODE"] == 1:
                                _, m.overlap = imgui.slider_float("##overlap", m.overlap, 0.0, 0.67, format=f"{m.overlap:.2f} overlap")
                                m.overlap = max(0.0, m.overlap)
                            _, m.threshold = imgui.slider_float("##thershold", m.threshold, 0.0, 1.0, format=f"{m.threshold:.2f} threshold")
                            imgui.pop_item_width()

                            _, m.active = imgui.checkbox("active   ", m.active)
                            if _ and m.active:
                                if cfg.se_active_frame is not None:
                                    cfg.se_active_frame.slice_changed = True
                            imgui.same_line()
                            _, m.blend = imgui.checkbox("blend   ", m.blend)
                            imgui.same_line()
                            _, m.show = imgui.checkbox("show     ", m.show)
                            imgui.same_line()
                            if imgui.button("save", 41, 14) and m.data is not None:
                                path = filedialog.asksaveasfilename(filetypes=[("mrcfile", ".mrc")], initialfile=os.path.basename(cfg.se_active_frame.path) + "_" + m.get_model_title() + f"_slice_{cfg.se_active_frame.current_slice}")
                                if path != "":
                                    if path[-4:] != ".mrc":
                                        path += ".mrc"
                                    with mrcfile.new(path, overwrite=True) as mrc:
                                        pxd = np.clip(m.data * 255, 0, 255).astype(np.uint8).squeeze()
                                        mrc.set_data(pxd)
                                        mrc.voxel_size = cfg.se_active_frame.pixel_size * 10.0
                            imgui.pop_style_var(5)
                            imgui.pop_style_color(1)
                            imgui.end_tab_item()

                        if imgui.begin_tab_item(" Interactions ")[0]:
                            m.active_tab = 2

                            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                            imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                            imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                            imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                            imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_TEXT)

                            imgui.text("Model competition:  ")
                            imgui.same_line()
                            _, m.emit = imgui.checkbox(" emit ", m.emit)
                            self.tooltip("When checked, this model emits prediction values, meaning it will affect\n"
                                         "absorbing models by nullifying their prediction wherever the emitting   \n"
                                         "model's prediction value is higher than that of an absorber.")
                            imgui.same_line()
                            _, m.absorb = imgui.checkbox(" absorb ", m.absorb)
                            self.tooltip("When checked, this model absorbs predictions by other models, meaning\n"
                                         "its output is nullified wherever there is any emitting model that \n"
                                         "predicts a higher value.")

                            # parse available models
                            available_partner_models = list()
                            available_partner_model_names = list()
                            for partner in cfg.se_models:
                                if partner != m:
                                    available_partner_models.append(partner)
                                    available_partner_model_names.append(partner.title)

                            # interaction GUI
                            cw = imgui.get_content_region_available_width()
                            for interaction in m.interactions:
                                delete_this_inter = False
                                if interaction.partner not in cfg.se_models:
                                    m.interactions.remove(interaction)
                                    continue
                                if interaction not in ModelInteraction.all:
                                    m.interactions.remove(interaction)
                                    continue
                                imgui.push_id(f"modelinteraction{interaction.uid}")

                                # model index
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, 1.0, 1.0, 1.0, 1.0)
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, 1.0, 1.0, 1.0, 1.0)
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, 1.0, 1.0, 1.0, 1.0)
                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (5, 1))
                                imgui.push_item_width(18)
                                m_idx = ModelInteraction.all.index(interaction)
                                _, new_idx = imgui.input_int("##interactionindex", m_idx+1, 0, 0)
                                if _:
                                    new_idx -= 1
                                    new_idx = min(max(0, new_idx), len(ModelInteraction.all) - 1)
                                    ModelInteraction.all[new_idx], ModelInteraction.all[m_idx] = ModelInteraction.all[m_idx], ModelInteraction.all[new_idx]
                                imgui.pop_style_color(3)
                                imgui.pop_style_var(1)

                                # select model to interact with
                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (8, 1))
                                imgui.set_next_item_width((cw - 18-9-15) // 2)
                                imgui.same_line(spacing=3)
                                _, interaction.type = imgui.combo("##type", interaction.type, ModelInteraction.TYPES)
                                if _:
                                    if cfg.se_active_frame is not None:
                                        cfg.se_active_frame.slice_changed = True
                                # select interaction type
                                p_idx = available_partner_model_names.index(interaction.partner.title)
                                imgui.same_line(spacing=3)
                                imgui.set_next_item_width((cw - 18-9-15) // 2)
                                imgui.push_style_color(imgui.COLOR_BUTTON, *interaction.partner.colour)
                                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *interaction.partner.colour)
                                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *interaction.partner.colour)
                                _, p_idx = imgui.combo("##partner_model", p_idx, available_partner_model_names)
                                if _:
                                    if cfg.se_active_frame is not None:
                                        cfg.se_active_frame.slice_changed = True
                                imgui.pop_style_color(3)
                                imgui.pop_style_var(1)
                                interaction.partner = available_partner_models[p_idx]

                                imgui.same_line(position = cw-7)
                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                                if imgui.image_button(self.icon_close.renderer_id, 15, 15):
                                    delete_this_inter = True
                                imgui.pop_style_var(1)
                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (3, 0))
                                imgui.set_next_item_width(cw)
                                _, interaction.radius = imgui.slider_float("##radius", interaction.radius, 0.0, 30.0,
                                                                           f"radius = {interaction.radius:.1f} nm")
                                if _:
                                    if cfg.se_active_frame is not None:
                                        cfg.se_active_frame.slice_changed = True
                                imgui.set_next_item_width(cw)
                                _, interaction.threshold = imgui.slider_float("##threshold", interaction.threshold, 0.0, 1.0,
                                                                           f"{interaction.partner.title} threshold = {interaction.threshold:.2f}")
                                if _:
                                    interaction.partner.threshold = interaction.threshold
                                    if cfg.se_active_frame is not None:
                                        cfg.se_active_frame.slice_changed = True
                                self.tooltip("Before applying interactions, the partner model's segmentation is thresholded at this value.")
                                imgui.pop_style_var(1)
                                imgui.separator()
                                imgui.pop_id()
                                if delete_this_inter:
                                    m.interactions.remove(interaction)
                                    ModelInteraction.all.remove(interaction)

                            # Add interaction
                            if len(available_partner_models) >= 1:
                                imgui.new_line()
                                imgui.same_line(spacing = (cw - 30) // 2)
                                if imgui.button("+", 30, 18):
                                    m.interactions.append(ModelInteraction(m, available_partner_models[0]))


                            imgui.pop_style_var(5)
                            imgui.pop_style_color(1)
                            imgui.end_tab_item()
                        imgui.end_tab_bar()

                    if m.background_process_train is not None:
                        self._gui_background_process_progress_bar(m.background_process_train, colour=(*m.colour, 1.0))
                        # trickle XP orbs off the growing training bar toward the HUD
                        _tb0, _tb1 = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                        _tp = min(1.0, m.background_process_train.progress)
                        progression.award_training(m.title, m.colour,
                                                   (_tb0[0] + (_tb1[0] - _tb0[0]) * _tp, (_tb0[1] + _tb1[1]) * 0.5))
                        if m.background_process_train.progress >= 1.0:
                            m.background_process_train = None
                    if m.background_process_apply is not None:
                        self._gui_background_process_progress_bar(m.background_process_apply)
                        if m.background_process_apply.progress >= 1.0:
                            m.background_process_apply = None

                    if imgui.is_window_hovered() and imgui.is_mouse_clicked(0):
                        cfg.se_active_model = m

                    if _mgrab:
                        imgui.pop_style_color(_mgrab)
                    imgui.end_child()
                    imgui.pop_id()
                cw = imgui.get_content_region_available_width()
                imgui.new_line()
                imgui.same_line(spacing=(cw - 120) / 2)
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                if imgui.button("Add model", 120, 23):
                    cfg.se_models.append(SEModel())
                imgui.pop_style_var(3)

        def export_tab():
            if imgui.collapsing_header("Export volumes", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                imgui.text("Models to include")
                n_available_models = sum([m.compiled for m in cfg.se_models])
                c_height = (1 if n_available_models == 0 else 12) + n_available_models * 17
                imgui.begin_child("models_included", 0.0, c_height, True)
                for m in cfg.se_models:
                    if not m.compiled:
                        continue
                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *m.colour)
                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *m.colour)
                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *m.colour)
                    imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_CHECK_MARK)
                    _, m.export = imgui.checkbox(m.title + " " + m.info_short, m.export)
                    if _:
                        m.active = m.export
                        m.show = m.export
                    imgui.pop_style_color(4)
                imgui.end_child()

                imgui.text("Datasets to process")

                imgui.same_line(position = imgui.get_content_region_available_width() - 97)
                _, SegmentationEditor.DATASETS_EXPORT_PANEL_EXPANDED = imgui.checkbox("expand",
                                                                                      SegmentationEditor.DATASETS_EXPORT_PANEL_EXPANDED)
                imgui.same_line(spacing=5)
                _, SegmentationEditor.DATASETS_EXPORT_SELECT_ALL = imgui.checkbox("all",
                                                                                  SegmentationEditor.DATASETS_EXPORT_SELECT_ALL)
                if _:
                    for s in cfg.se_frames:
                        s.export = SegmentationEditor.DATASETS_EXPORT_SELECT_ALL
                imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_CHECK_MARK)
                c_height = min([120, (1 if len(cfg.se_frames) == 0 else 9) + len(cfg.se_frames) * 21])
                if SegmentationEditor.DATASETS_EXPORT_PANEL_EXPANDED:
                    c_height = SegmentationEditor.DATASETS_EXPORT_PANEL_EXPANDED_HEIGHT
                imgui.begin_child("datasets_to_sample", 0.0, c_height, True)
                for s in cfg.se_frames:
                    imgui.push_id(f"{s.uid}")
                    if s == cfg.se_active_frame:
                        imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT_ACTIVE)
                        _, s.export = imgui.checkbox(s.title, s.export)
                        imgui.pop_style_color()
                    else:
                        _, s.export = imgui.checkbox(s.title, s.export)
                    if _:
                        self.parse_available_features()
                    imgui.pop_id()
                imgui.end_child()


                imgui.pop_style_color()

                imgui.text("Export settings")
                imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                imgui.begin_child("export_settings", 0.0, 53.0, True)
                _, SegmentationEditor.export_dir = widgets.select_directory("browse", SegmentationEditor.export_dir)
                imgui.set_next_item_width(imgui.get_content_region_available_width())
                _, self.export_limit_range = imgui.checkbox(" limit range ", self.export_limit_range)
                #imgui.same_line(spacing=62)
                #_, self.export_overlays = imgui.checkbox(" export overlays", self.export_overlays)
                imgui.end_child()


                if widgets.centred_button("Start export", 120, 23):
                    self.launch_export_volumes()

                # export progress:
                if SegmentationEditor.queued_exports:
                    qe = SegmentationEditor.queued_exports[0]
                    imgui.spacing()
                    if isinstance(qe, QueuedExport):
                        imgui.text(f"Processing '{qe.title}':")
                    else:
                        imgui.text(f"Extracting {os.path.splitext(os.path.basename(qe.path))[0]}")


                    for i, qe in enumerate(SegmentationEditor.queued_exports):
                        if i != 0:
                            imgui.text(qe.title)
                            imgui.same_line()
                        imgui.push_id(f"{qe.tag}_cancel")
                        colour = (*qe.colour, 1.0) if i == 0 else (0.0, 0.0, 0.0, 0.0)
                        cancel = self._gui_background_process_progress_bar(qe.process, colour, cancellable=True, transparent_background=(i!=0))
                        imgui.pop_id()
                        if cancel:
                            SegmentationEditor.queued_exports[i].stop()
                            SegmentationEditor.queued_exports.pop(i)
                            break
                        if i == 0 and len(SegmentationEditor.queued_exports) > 1:
                            imgui.spacing()
                            imgui.text(f"Queue:")


                imgui.pop_style_var(4)

        def picking_tab():
            if imgui.collapsing_header("Volumes", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                imgui.text("Segmentation (.mrc) directory:")
                src_folder_changed, SegmentationEditor.seg_folder = widgets.select_directory("...", SegmentationEditor.seg_folder)
                self.tooltip("Select the location in which to look for segmentations that belong to the\n"
                             "imported datasets. Right-click to set the path to the location of the ac-\n"
                             "tive dataset.")
                if imgui.is_item_clicked(1):
                    src_folder_changed = True
                    SegmentationEditor.seg_folder = os.path.dirname(cfg.se_active_frame.path)
                self.tooltip("Specify the directory in which to look for segmentations for the active dataset.")
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_CHECK_MARK)

                # list the segmentations found for the currently selected dataset.
                def update_picking_tab_for_new_active_frame():
                    SegmentationEditor.VIEW_REQUIRES_UPDATE = True
                    # delete old surface models
                    for s in cfg.se_surface_models:
                        s.delete()
                    cfg.se_surface_models = list()
                    # index new surface models - find corresponding segmentation .mrc's
                    if cfg.se_active_frame is None:
                        self.pick_box_va.update(VertexBuffer([0.0, 0.0]), IndexBuffer([]))
                        self.pick_box_quad_va.update(VertexBuffer([0.0, 0.0]), IndexBuffer([]))
                        return
                    se_frame = cfg.se_active_frame
                    SegmentationEditor.pick_tab_index_datasets_segs = False
                    # find segmentations in the selected dataset's folder.


                    files = []
                    directories = cfg.settings["SEARCH_DIRECTORIES"] + [SegmentationEditor.seg_folder]
                    print(f'Searching for segmentations related to {se_frame.path} in:')
                    for d in directories:
                        files += glob.glob(os.path.join(d, os.path.basename(os.path.splitext(se_frame.path)[0]) + "__*.mrc"))
                        print(f"\t", os.path.join(d, os.path.basename(os.path.splitext(se_frame.path)[0]) + "__*.mrc"))
                    for f in sorted(files):
                        print(f'\t{f}')
                        cfg.se_surface_models.append(SurfaceModel(f, se_frame.pixel_size))
                    # set the size of the volume spanning box
                    w, h, d = se_frame.width / 2, se_frame.height / 2, se_frame.n_slices / 2
                    w *= se_frame.pixel_size
                    h *= se_frame.pixel_size
                    d *= se_frame.pixel_size
                    vertices = [-w, h, d,
                                w, h, d,
                                w, -h, d,
                                -w, -h, d,
                                -w, h, -d,
                                w, h, -d,
                                w, -h, -d,
                                -w, -h, -d]
                    line_indices = [0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 1, 5, 2, 6, 3, 7, 4, 5, 5, 6, 6, 7, 7, 4]
                    quad_indices = [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 0, 4, 7, 7, 3, 0, 5, 1, 2, 2, 6, 5, 4, 0, 1, 1, 5, 4, 3, 7, 6, 6, 2, 3]
                    self.pick_box_va.update(VertexBuffer(vertices), IndexBuffer(line_indices))
                    self.pick_box_quad_va.update(VertexBuffer(vertices), IndexBuffer(quad_indices))


                if SegmentationEditor.pick_tab_index_datasets_segs or src_folder_changed:
                    update_picking_tab_for_new_active_frame()

                s_to_remove = list()
                for s in cfg.se_surface_models:
                    # automatically gnerate the model if WAIT_TO_RENDER is False and no other surface model is currently being generated
                    if not s.hide and not s.initialized and cfg.settings["WAIT_TO_RENDER"] is False:
                        any_model_being_generated = any([el.process is not None for el in cfg.se_surface_models])
                        if not any_model_being_generated:
                            s.generate_model()

                    req_gen_models = False
                    s.on_update()
                    if s.process is not None:
                        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *s.colour, (1.0 - s.process.progress) * 0.3)
                        imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT_DISABLED)
                    else:
                        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *s.colour, 0.0)
                        imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT)
                    imgui.begin_child(f"{s.title}_surfm{s.uid}", 0.0, 82 + (15 if s.particles else 0), True)
                    cw = imgui.get_content_region_available_width()
                    _sgrab = SegmentationEditor._push_slider_grab(s.colour)
                    _, s.colour = imgui.color_edit3(s.title, *s.colour[:3], imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)

                    imgui.same_line()
                    imgui.text(f"{s.title} ({s.size_mb:.1f} mb)")
                    imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                    imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                    imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)

                    imgui.same_line(position = cw - 40)
                    _, s.hide = imgui.checkbox("hide  ", s.hide)
                    # progress bar
                    if s.process is not None:
                        if s.process.progress == 1.0:
                            s.process = None

                    imgui.push_item_width(cw)
                    original_level = s.level
                    _, s.level = imgui.slider_int("##level", s.level, 1, 256, f"level = {s.level}")
                    if s.process is not None:
                        s.level = original_level
                        _ = False
                    if _ and original_level != s.level:
                        req_gen_models = True

                    _, s.dust = imgui.slider_float("##dust", s.dust, 1.0, 1000000.0, f"dust < {s.dust:.1f} nm³", imgui.SLIDER_FLAGS_LOGARITHMIC)
                    if _:
                        s.hide_dust()
                    imgui.pop_item_width()
                    if s.particles:
                        imgui.push_item_width(cw - 22)
                        _, s.particle_size = imgui.slider_float("##particle", s.particle_size, 0.0, 25.0, "hide particles" if s.particle_size == 0 else f"particle size = {s.particle_size:.1f} nm")
                        imgui.same_line()
                        _, s.particle_colour = imgui.color_edit3(s.title+"pclr", *s.particle_colour[:3], imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                        imgui.pop_item_width()
                    imgui.push_item_width((cw - 7) / 2)
                    _, s.alpha = imgui.slider_float("##alpha", s.alpha, 0, 1.0, f"alpha = {s.alpha:.1f}")
                    imgui.same_line()
                    original_bin = s.bin
                    _, s.bin = imgui.slider_int("##bin", s.bin, 1, 8, f"bin {s.bin}")
                    if _ and original_bin != s.bin:
                        req_gen_models = True


                    if req_gen_models:
                        s.generate_model()

                    imgui.pop_item_width()
                    imgui.pop_style_var(3)
                    imgui.pop_style_color(2)
                    if imgui.begin_popup_context_window():
                        if imgui.begin_menu("Extract coordinates"):
                            self.extract_coordinates_menu(s)
                            imgui.end_menu()
                        if imgui.begin_menu("Extract mesh"):
                            self.extract_meshes_menu(s)
                            imgui.end_menu()
                        imgui.spacing()
                        imgui.separator()
                        imgui.spacing()
                        if imgui.menu_item("Paint particles")[0]:
                            filepath = filedialog.askopenfilename(filetypes=[("Ais paint info", ".tsv")])
                            if os.path.exists(filepath):
                                try:
                                    s.paint_particles(filepath)
                                except Exception as e:
                                    cfg.set_error(e, "Could not paint particles - see error:\n")
                        if imgui.menu_item("Load particles")[0]:
                            filepath = filedialog.askopenfilename(filetypes=[("Particle star file", ".star")])
                            if os.path.exists(filepath):
                                s.load_coordinates(filepath)
                        if imgui.begin_menu("Override voxel size"):
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (2, 0))
                            imgui.set_next_item_width(100)
                            _, s.pixel_size = imgui.input_float("##pxs", s.pixel_size, 0.0, 0.0, '%.2f nm/px')
                            imgui.same_line()
                            if imgui.button("x2", 20, 16):
                                s.pixel_size *= 2
                                s.generate_model()
                            imgui.same_line()
                            if imgui.button("apply", 50, 16):
                                s.generate_model()


                            imgui.pop_style_var(1)
                            imgui.end_menu()
                        if imgui.menu_item("Invert")[0]:
                            s.data = 255 - s.data
                            s.latest_bin = -1
                            s.generate_model()
                        if imgui.menu_item("Remove volume")[0]:
                            s_to_remove.append(s)
                        imgui.end_popup()
                    if _sgrab:
                        imgui.pop_style_color(_sgrab)
                    imgui.end_child()

                imgui.pop_style_var(3)
                imgui.pop_style_color(1)

                for s in s_to_remove:
                    s.delete()
                    cfg.se_surface_models.remove(s)

                if widgets.centred_button("Manual import", 115, 24):
                    filepaths = filedialog.askopenfilenames(filetypes=[("Segmented volume", ".mrc")])
                    print(filepaths)
                    if filepaths != ():
                        for f in filepaths:
                            try:
                                cfg.se_surface_models.append(SurfaceModel(f, 3.0 if cfg.se_active_frame is None else cfg.se_active_frame.pixel_size))
                            except Exception as e:
                                cfg.set_error(e, f"Could not import {f} as SurfaceModel - see details below")

            SegmentationEditor.LIGHT_SPOT.compute_vec(dyaw=-self.camera3d.yaw)
            if imgui.collapsing_header("Graphics settings", None)[0]:
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (2, 2))
                cw = imgui.get_content_region_available_width()
                imgui.align_text_to_frame_padding()
                imgui.text("Volume render style:")
                imgui.same_line()
                imgui.push_item_width(cw - 147)
                _, SegmentationEditor.SELECTED_RENDER_STYLE = imgui.combo("##Graphics style", SegmentationEditor.SELECTED_RENDER_STYLE, SegmentationEditor.RENDER_STYLES)
                imgui.pop_item_width()
                _, SegmentationEditor.RENDER_CLEAR_COLOUR = imgui.color_edit3("##clrclr", *SegmentationEditor.RENDER_CLEAR_COLOUR[:3], imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                imgui.same_line()
                imgui.align_text_to_frame_padding()
                imgui.text(" Background colour     ")
                if SegmentationEditor.SELECTED_RENDER_STYLE in [0, 1]:
                    imgui.same_line()
                    _, SegmentationEditor.LIGHT_SPOT.colour = imgui.color_edit3("##lightclr", *SegmentationEditor.LIGHT_SPOT.colour[:3], imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                    imgui.same_line()
                    imgui.align_text_to_frame_padding()
                    imgui.text(" Light colour")

                    imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                    imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 10)
                    imgui.push_item_width(cw)
                    _, SegmentationEditor.LIGHT_AMBIENT_STRENGTH = imgui.slider_float("##ambient lighting", SegmentationEditor.LIGHT_AMBIENT_STRENGTH, 0.0, 1.0, f"ambient strength = %.2f")
                    _, SegmentationEditor.LIGHT_SPOT.strength = imgui.slider_float("##spot lighting", SegmentationEditor.LIGHT_SPOT.strength, 0.0, 1.0, f"spot strength = %.2f")
                    _yaw, SegmentationEditor.LIGHT_SPOT.yaw = imgui.drag_float("##yaw", SegmentationEditor.LIGHT_SPOT.yaw, 0.5, format=f"spot yaw = %.1f")
                    _pitch, SegmentationEditor.LIGHT_SPOT.pitch = imgui.slider_float("##pitch", SegmentationEditor.LIGHT_SPOT.pitch, -90.0, 90.0, format=f"spot pitch = %.1f")
                    imgui.pop_item_width()
                    imgui.pop_style_var(3)

                imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_CHECK_MARK)
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 10)

                _, SegmentationEditor.RENDER_BOX = imgui.checkbox(" render bounding box   ", SegmentationEditor.RENDER_BOX)

                imgui.same_line()
                _render_frame = SegmentationEditor.PICKING_FRAME_ALPHA != 0.0
                _r, _render_frame = imgui.checkbox("render frame", _render_frame)
                if _r:
                    SegmentationEditor.PICKING_FRAME_ALPHA = float(_render_frame)
                _, SegmentationEditor.RENDER_PARTICLES_XRAY = imgui.checkbox(" particle x-ray        ", SegmentationEditor.RENDER_PARTICLES_XRAY)
                imgui.same_line()
                _, SegmentationEditor.RENDER_SILHOUETTES = imgui.checkbox("draw edges", SegmentationEditor.RENDER_SILHOUETTES)
                if SegmentationEditor.RENDER_SILHOUETTES:
                    imgui.set_next_item_width(cw)
                    _, SegmentationEditor.RENDER_SILHOUETTES_THRESHOLD = imgui.slider_float("##edge threshold", SegmentationEditor.RENDER_SILHOUETTES_THRESHOLD, 0.01, 10.0, f"threshold = %.2f")
                    imgui.set_next_item_width(cw)
                    _, SegmentationEditor.RENDER_SILHOUETTES_ALPHA = imgui.slider_float("##edge alpha", SegmentationEditor.RENDER_SILHOUETTES_ALPHA, 0.0, 1.0, f"contrast = %.2f")
                imgui.pop_style_var(4)
                imgui.pop_style_color(1)
                imgui.pop_style_var(1)

            if imgui.collapsing_header("Export 3D scene", None)[0]:
                cw = imgui.get_content_region_available_width()
                spacing = 10
                s = (cw - 2 * spacing * 2 - 3) / 3
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 100)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *cfg.COLOUR_FRAME_DARK)
                imgui.push_style_color(imgui.COLOR_BORDER, *cfg.COLOUR_FRAME_DARK)
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_FRAME_EXTRA_DARK)
                if imgui.image_button(self.icon_obj.renderer_id, s, s):
                    SegmentationEditor.save_surface_models_as_objs()
                imgui.same_line(spacing=spacing)
                if imgui.image_button(self.icon_blender.renderer_id, s, s):
                    obj_paths = SegmentationEditor.save_surface_models_as_objs()
                    # then open Blender.
                    SegmentationEditor.open_objs_in_blender(obj_paths)
                if imgui.begin_popup_context_window():
                    if imgui.menu_item("Set path to blender.exe")[0]:
                        path = filedialog.askopenfilename(filetypes=[("Blender executable", ".exe .sh")])
                        if path != "":
                            cfg.edit_setting("BLENDER_EXE", path)
                    imgui.end_popup()
                imgui.same_line(spacing=spacing)
                if imgui.image_button(self.icon_chimerax.renderer_id, s, s):
                    models = list()
                    for m in cfg.se_surface_models:
                        if m.initialized and not m.hide:
                            models.append(m)
                    SegmentationEditor.open_in_chimerax(models)
                if imgui.begin_popup_context_window():
                    if imgui.menu_item("Set path to ChimeraX.exe")[0]:
                        path = filedialog.askopenfilename(filetypes=[("ChimeraX executable", ".exe .sh")])
                        if path != "":
                            cfg.edit_setting("CHIMERAX_EXE", path)
                    imgui.end_popup()
                imgui.pop_style_var(2)
                imgui.pop_style_color(3)

        def menu_bar():
            imgui.push_style_color(imgui.COLOR_MENUBAR_BACKGROUND, *cfg.COLOUR_MAIN_MENU_BAR)
            imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_MAIN_MENU_BAR_TEXT)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *cfg.COLOUR_MAIN_MENU_BAR_HILIGHT)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *cfg.COLOUR_MAIN_MENU_BAR_HILIGHT)
            imgui.push_style_color(imgui.COLOR_HEADER, *cfg.COLOUR_MAIN_MENU_BAR_HILIGHT)
            imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *cfg.COLOUR_MENU_WINDOW_BACKGROUND)
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (2.0, 2.0))

            if imgui.core.begin_main_menu_bar():
                if imgui.begin_menu("File"):
                    if imgui.menu_item("Import datasets")[0]:
                        try:
                            filename = filedialog.askopenfilenames(filetypes=[("Ais segmentable", f".mrc {cfg.filetype_segmentation}"), (".scns", f"{cfg.filetype_segmentation}"), (".mrc", ".mrc")])
                            if filename != '':
                                self.import_dataset(filename)
                        except Exception as e:
                            cfg.set_error(e, "Could not import dataset, see details below.")
                    if imgui.menu_item("Import model")[0]:
                        try:
                            filename = filedialog.askopenfilename(filetypes=[("Ais model", f"{cfg.filetype_semodel}")])
                            if filename != '':
                                self.load_model(filename)
                        except Exception as e:
                            cfg.set_error(e, "Could not import model, see details below.")
                    if imgui.menu_item("Import model group")[0]:
                        try:
                            filename = filedialog.askopenfilename(filetypes=[("Ais model group", cfg.filetype_semodel_group)])
                            if filename != '':
                                SegmentationEditor.load_model_group(filename)
                        except Exception as e:
                            cfg.set_error(e, "Could not import model group, see details below.")
                    if imgui.begin_menu("Open recent dataset"):
                        _recents = cfg.dedup_recent_datasets([p for p in (cfg.settings.get("RECENT_DATASETS") or []) if os.path.exists(p)])
                        if not _recents:
                            imgui.menu_item("(none)", None, False, False)
                        for _p in _recents:
                            if imgui.menu_item(os.path.basename(_p))[0]:
                                try:
                                    self.import_dataset(_p)
                                except Exception as e:
                                    cfg.set_error(e, "Could not open recent dataset, see details below.")
                            if imgui.is_item_hovered():
                                imgui.set_tooltip(_p)
                        imgui.end_menu()
                    if imgui.begin_menu("Open recent model"):
                        _recents = [p for p in (cfg.settings.get("RECENT_MODELS") or []) if os.path.exists(p)]
                        if not _recents:
                            imgui.menu_item("(none)", None, False, False)
                        for _p in _recents:
                            if imgui.menu_item(os.path.basename(_p))[0]:
                                try:
                                    SegmentationEditor.load_model(_p)
                                except Exception as e:
                                    cfg.set_error(e, "Could not open recent model, see details below.")
                            if imgui.is_item_hovered():
                                imgui.set_tooltip(_p)
                        imgui.end_menu()
                    imgui.separator()
                    if imgui.menu_item("Save dataset")[0]:
                        SegmentationEditor.save_dataset(dialog=False)
                    if imgui.menu_item("Save dataset as ")[0]:
                        SegmentationEditor.save_dataset(dialog=True)
                    if imgui.menu_item("Save dataset w. map")[0]:
                        try:
                            filename = filedialog.asksaveasfilename(filetypes=[("Ais segmentation", f"{cfg.filetype_segmentation}")], initialfile = os.path.basename(cfg.se_active_frame.path)[:-4])
                            if filename != '':
                                if filename[-len(cfg.filetype_segmentation):] != cfg.filetype_segmentation:
                                    filename += cfg.filetype_segmentation
                                cfg.se_active_frame.include_map()
                                with open(filename, 'wb') as pickle_file:
                                    pickle.dump(cfg.se_active_frame, pickle_file)
                        except Exception as e:
                            cfg.set_error(e, "Could not save dataset (including map), see details below.")
                    if imgui.menu_item("Save model group")[0]:
                        try:
                            filename = filedialog.asksaveasfilename(filetypes=[("Ais model group", f"{cfg.filetype_semodel_group}")])
                            if filename != '':
                                if filename[-len(cfg.filetype_semodel_group):] != cfg.filetype_semodel_group:
                                    filename += cfg.filetype_semodel_group
                                SegmentationEditor.save_model_group(filename)
                        except Exception as e:
                            cfg.set_error(e, "Could not save model group, see details below.")
                    imgui.separator()
                    if imgui.menu_item("Unlink all datasets")[0]:
                        cfg.se_frames = list()
                        cfg.se_active_frame = None
                        self.parse_available_features()
                    # if imgui.menu_item("Export validation slice")[0]:
                    #     try:
                    #         filename = filedialog.asksaveasfilename(filetypes=[("tifffile", ".tiff")])
                    #         if filename != '':
                    #             if filename[-5:] != '.tiff':
                    #                 filename += '.tiff'
                    #             tifffile.imwrite(filename, cfg.se_active_frame.data.astype(np.float32))
                    #     except Exception as e:
                    #         cfg.set_error(e, "Could not export current slice as .tiff, see details below.")
                    # imgui.separator()
                    imgui.text(f'version {cfg.version}')
                    imgui.end_menu()

                if EMBEDDED and imgui.begin_menu("Editor"):
                    for i in range(len(scn_cfg.editors)):
                        select, _ = imgui.menu_item(scn_cfg.editors[i], None, False)
                        if select:
                            scn_cfg.active_editor = i
                    imgui.end_menu()
                if imgui.begin_menu("Settings"):
                    if imgui.begin_menu("Model settings"):
                        if imgui.begin_menu("Model library"):
                            imgui.text(os.path.join(os.path.dirname(cfg.settings_path), "models"))
                            imgui.separator()
                            custom_models = glob.glob(os.path.join(os.path.dirname(cfg.settings_path), "models", "*.py"))
                            for m in custom_models:
                                if imgui.begin_menu(os.path.splitext(os.path.basename(m))[0]):
                                    if imgui.menu_item("Reload")[0]:
                                        SEModel.load_models()
                                    if imgui.menu_item("Delete")[0]:
                                        os.remove(m)
                                    imgui.end_menu()
                            if imgui.menu_item("Install a model")[0]:
                                try:
                                    path = filedialog.askopenfilename(filetypes=[("Ais model (.py)", ".py")])
                                    if path != "":
                                        user_library = os.path.join(os.path.dirname(cfg.settings_path), "models")
                                        shutil.copy(path, os.path.join(user_library, os.path.basename(path)))
                                        SEModel.load_models()
                                except Exception as e:
                                    cfg.set_error(e, "Something went wrong adding a model to the library - see below.")
                            imgui.end_menu()

                        if imgui.begin_menu("Validation splits"):
                            split_setting = cfg.settings["VALIDATION_SPLIT"]
                            split_options = {'no split': 0, 'split 10%': 10, 'split 20%': 20, 'split 50%': 50}
                            for k in split_options:
                                if imgui.menu_item(k, None, split_setting == split_options[k])[0]:
                                    cfg.edit_setting("VALIDATION_SPLIT", split_options[k])
                            imgui.end_menu()

                        if imgui.begin_menu("Learning rate"):
                            rate_setting = cfg.settings["LEARNING_RATE"]
                            rate_options = {'5.0e-3': 5.0e-3, '1.0e-3 (default)': 1.0e-3, '5.0e-4': 5.0e-4, '1.0e-4': 1.0e-4, '5e.0-5': 5.0e-5}
                            for k in rate_options:
                                if imgui.menu_item(k, None, rate_setting == rate_options[k])[0]:
                                    cfg.edit_setting("LEARNING_RATE", rate_options[k])
                            if imgui.begin_menu("Custom"):
                                _, custom_rate = imgui.input_float("##lrate", rate_setting,  0.0, 0.0, '%.6f')
                                if _:
                                    cfg.edit_setting("LEARNING_RATE", custom_rate)
                                imgui.end_menu()
                            imgui.end_menu()

                        if imgui.begin_menu("Processing strategy"):
                            if imgui.menu_item("full image", None, cfg.settings["TILED_MODE"] == 0)[0]:
                                cfg.edit_setting("TILED_MODE", 0)
                            if imgui.menu_item("tiled (legacy)", None, cfg.settings["TILED_MODE"] == 1)[0]:
                                cfg.edit_setting("TILED_MODE", 1)
                            if cfg.settings["TILED_MODE"] == 1:
                                if imgui.begin_menu("Tile overlap mode"):
                                    if imgui.menu_item("best", None, cfg.settings["OVERLAP_MODE"] == 1)[0]:
                                        cfg.edit_setting("OVERLAP_MODE", 1)
                                    self.tooltip("When processing a slice, the input image is tiled into e.g. 64 x 64 boxes,\n"
                                                 "the tiles processed by the neural network, and the data then detiled back\n"
                                                 "into image shape. Overlap between tiles (the 'overlap' setting in the mo-\n"
                                                 "del parameters) is handled either by averaging the overlapping regions of\n"
                                                 "and image, or by retaining predictions closest to the center of a box,\n"
                                                 "where model predictions are typically the best quality.")
                                    if imgui.menu_item("average", None, cfg.settings["OVERLAP_MODE"] == 0)[0]:
                                        cfg.edit_setting("OVERLAP_MODE", 0)
                                    self.tooltip("When processing a slice, the input image is tiled into e.g. 64 x 64 boxes,\n"
                                                 "the tiles processed by the neural network, and the data then detiled back\n"
                                                 "into image shape. Overlap between tiles (the 'overlap' setting in the mo-\n"
                                                 "del parameters) is handled either by averaging the overlapping regions of\n"
                                                 "and image, or by retaining predictions closest to the center of a box,\n"
                                                 "where model predictions are typically the best quality.")
                                    imgui.end_menu()
                            imgui.end_menu()
                        if imgui.begin_menu("TTA multiplicity"):
                            for i in range(1, 9):
                                if imgui.menu_item(f'{i}', None, cfg.settings["TEST_TIME_AUGMENTATIONS"] == i)[0]:
                                    cfg.edit_setting("TEST_TIME_AUGMENTATIONS", i)
                            imgui.end_menu()

                        if imgui.menu_item("Trim edges", None, cfg.settings["TRIM_EDGES"] == 1)[0]:
                            cfg.edit_setting("TRIM_EDGES", 0 if cfg.settings["TRIM_EDGES"] == 1 else 1)
                        self.tooltip("When active, the margins of output segmentations are forced to zero.\n"
                                     "The size of the margin equals half the model's box size.")

                        if imgui.menu_item("Rescale for inference", selected=cfg.settings["INFERENCE_ALLOW_RESCALING"])[0]:
                            cfg.edit_setting("INFERENCE_ALLOW_RESCALING", not cfg.settings["INFERENCE_ALLOW_RESCALING"])
                        self.tooltip("When True, input data is rescaled to match the model's pixel size during inference. ")

                        if imgui.menu_item("use Ais v2.0 augmentations", None, cfg.settings["EXTRA_AUGMENTATIONS"])[0]:
                            cfg.edit_setting("EXTRA_AUGMENTATIONS", not cfg.settings["EXTRA_AUGMENTATIONS"])
                        self.tooltip("Use scaling, blurring, noise, and contrast augmentations in training.")
                        imgui.end_menu()

                    if imgui.begin_menu("3rd party applications"):
                        if imgui.begin_menu("Blender"):
                            blender_path = cfg.settings["BLENDER_EXE"]

                            if not os.path.exists(blender_path):
                                imgui.push_style_color(imgui.COLOR_TEXT, 0.8, 0.0, 0.0, 1.0)
                                imgui.text("Blender .exe not found!")
                                imgui.pop_style_color()
                            else:
                                imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 0.0, 0.8, 1.0)
                                imgui.text(cfg.settings["BLENDER_EXE"])
                                imgui.pop_style_color()
                            imgui.separator()
                            if imgui.menu_item("Set path to .exe")[0]:
                                path = filedialog.askopenfilename(filetypes=[("Blender executable", ".exe")])
                                if isinstance(path, str) and path != "":
                                    cfg.edit_setting("BLENDER_EXE", path)
                            if os.path.exists(blender_path) and imgui.menu_item("Launch")[0]:
                                subprocess.Popen([cfg.settings["BLENDER_EXE"]])

                            imgui.end_menu()
                        if imgui.begin_menu("ChimeraX"):
                            chimerax_path = cfg.settings["CHIMERAX_EXE"]

                            if not os.path.exists(chimerax_path):
                                imgui.push_style_color(imgui.COLOR_TEXT, 0.8, 0.0, 0.0, 1.0)
                                imgui.text("ChimeraX .exe not found!")
                                imgui.pop_style_color()
                            else:
                                imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 0.0, 0.8, 1.0)
                                imgui.text(cfg.settings["CHIMERAX_EXE"])
                                imgui.pop_style_color()
                            imgui.separator()
                            if imgui.menu_item("Set path to .exe")[0]:
                                path = filedialog.askopenfilename(filetypes=[("ChimeraX executable", ".exe")])
                                if isinstance(path, str) and path != "":
                                    cfg.edit_setting("CHIMERAX_EXE", path)

                            if os.path.exists(chimerax_path) and imgui.menu_item("Launch")[0]:
                                subprocess.Popen([cfg.settings["CHIMERAX_EXE"]])
                            imgui.end_menu()
                        imgui.end_menu()

                    if imgui.begin_menu("File manager"):
                        if imgui.menu_item("Open file manager")[0]:
                            SegmentationEditor.PATH_VIEWER_OPEN = True
                        # if imgui.begin_menu("Quick-save directory"):
                        #     if imgui.menu_item("same as tomogram", selected=cfg.settings["QUICK_SAVE_DIRECTORY"]=="")[0]:
                        #         cfg.edit_setting("QUICK_SAVE_DIRECTORY", "")
                        #     if imgui.menu_item("custom", selected=cfg.settings["QUICK_SAVE_DIRECTORY"]!="")[0]:
                        #         custom_dir = filedialog.askdirectory()
                        #         if isinstance(custom_dir, str):
                        #             cfg.edit_setting("QUICK_SAVE_DIRECTORY", custom_dir)
                        #     if cfg.settings["QUICK_SAVE_DIRECTORY"] != "":
                        #         imgui.separator()
                        #         imgui.text(cfg.settings["QUICK_SAVE_DIRECTORY"])
                        #     imgui.end_menu()
                        imgui.end_menu()

                    if imgui.begin_menu("Rendering"):
                        if imgui.menu_item("Recompile shaders")[0]:
                            self.renderer.recompile_shaders()
                        if imgui.menu_item("Wait to render", selected=cfg.settings["WAIT_TO_RENDER"])[0]:
                            cfg.edit_setting("WAIT_TO_RENDER", not cfg.settings["WAIT_TO_RENDER"])
                        self.tooltip("In the Rendering tab, segmentations are rendered immediately  if 'wait to render' is not set.\n"
                                     "Else, rendering is triggered only when any settings (threshold, dust size, etc.) are edited.")
                        if imgui.menu_item("Dark mode", None, cfg.settings["DARK_MODE"])[0]:
                            cfg.edit_setting("DARK_MODE", not cfg.settings["DARK_MODE"])
                            cfg.set_theme(cfg.settings["DARK_MODE"])
                        if imgui.begin_menu("Camera3D"):
                            changed_yaw, new_yaw = imgui.input_float("Yaw", self.camera3d.yaw, 5.0, 20.0)
                            changed_pitch, new_pitch = imgui.input_float("Pitch", self.camera3d.pitch, 5.0, 20.0)
                            if changed_yaw or changed_pitch:
                                self.camera3d.set_euler(new_yaw, new_pitch)
                                self.camera3d.on_update()
                            imgui.end_menu()
                        if imgui.begin_menu("Search directories"):
                            for d in cfg.settings["SEARCH_DIRECTORIES"]:
                                if imgui.begin_menu(d):
                                    if imgui.menu_item("Remove")[0]:
                                        cfg.settings["SEARCH_DIRECTORIES"].remove(d)
                                        cfg.edit_setting("SEARCH_DIRECTORIES", cfg.settings["SEARCH_DIRECTORIES"])
                                    imgui.end_menu()
                            if imgui.menu_item("Add")[0]:
                                d = filedialog.askdirectory()
                                if isinstance(d, str) and d != "":
                                    cfg.settings["SEARCH_DIRECTORIES"].append(d)
                                    cfg.edit_setting("SEARCH_DIRECTORIES", cfg.settings["SEARCH_DIRECTORIES"])
                            imgui.end_menu()
                        imgui.end_menu()

                    if imgui.begin_menu("Feature library"):
                        if imgui.menu_item("Open library")[0]:
                            SegmentationEditor.FEATURE_LIB_OPEN = True
                            self.parse_available_features()
                        if imgui.begin_menu("Active library"):
                            for lib_name in list(cfg.feature_libraries.keys()):
                                if imgui.menu_item(lib_name, None, lib_name == cfg.active_feature_library)[0]:
                                    cfg.set_active_feature_library(lib_name)
                                    self.parse_available_features()
                            imgui.end_menu()
                        imgui.end_menu()

                    if imgui.begin_menu("Pom"):
                        if imgui.menu_item("Synchronize Ais & Pom", None, cfg.settings["POM_SYNCHRONIZE"])[0]:
                            cfg.edit_setting("POM_SYNCHRONIZE", not cfg.settings["POM_SYNCHRONIZE"])
                        if imgui.begin_menu("Templates"):
                            imgui.align_text_to_frame_padding()
                            imgui.text("size (px):")
                            imgui.same_line()
                            imgui.set_next_item_width(24)
                            _, pom_template_size = imgui.input_int("##size (px)", cfg.settings["POM_TEMPLATE_PX_SIZE"], 0, 0)
                            if pom_template_size > 10:
                                pom_template_size = pom_template_size // 2 * 2
                            if _:
                                cfg.edit_setting("POM_TEMPLATE_PX_SIZE", pom_template_size)
                            if imgui.menu_item("Export masked template")[0]:
                                filename = filedialog.asksaveasfilename(filetypes = [(".mrc", ".mrc")], initialfile=f"new_template.mrc")
                                if filename:
                                    # Save volume itself
                                    f_path = os.path.splitext(filename)[0]
                                    se_f = cfg.se_active_frame
                                    if se_f.includes_map:
                                        with mrcfile.new(f_path + ".mrc", overwrite=True) as f:
                                            f.set_data(se_f.map.data)
                                            f.voxel_size = se_f.pixel_size * 10.0
                                    else:
                                        shutil.copy(se_f.path, f_path+".mrc")

                                    # If exists, save template mask
                                    se_mask = None
                                    for feature in se_f.features:
                                        if feature.title == "Template mask":
                                            se_mask = feature
                                    if se_mask is not None:
                                        se_mask.save_volume(f_path+"_mask.mrc")



                            imgui.end_menu()
                        if cfg.settings["POM_SYNCHRONIZE"]:
                            imgui.separator()
                            if cfg.settings["POM_COMMAND_DIR"] == "" or not os.path.exists(cfg.settings["POM_COMMAND_DIR"]):
                                imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.5, 0.5, 1.0)
                                imgui.text("Command directory not set.")
                                imgui.pop_style_color()
                            else:
                                imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 0.0, 0.8, 1.0)
                                imgui.text(cfg.settings["POM_COMMAND_DIR"])
                                imgui.pop_style_color()
                            if imgui.menu_item("Set command directory")[0]:
                                command_directory = filedialog.askdirectory()
                                if command_directory != "":
                                    cfg.edit_setting("POM_COMMAND_DIR", command_directory)
                        imgui.end_menu()

                    if imgui.begin_menu("Developer"):
                        if imgui.menu_item("Show ImGui debug window", None, SegmentationEditor.SHOW_IMGUI_DEBUG)[0]:
                            SegmentationEditor.SHOW_IMGUI_DEBUG = not SegmentationEditor.SHOW_IMGUI_DEBUG
                        # if imgui.menu_item("(debug) se_model bin2", selected=cfg.settings["DEBUG_PREPROC_BIN_2"])[0]:
                        #     cfg.edit_setting("DEBUG_PREPROC_BIN_2", not cfg.settings["DEBUG_PREPROC_BIN_2"])
                        #     cfg.edit_setting("DEBUG_PREPROC_BIN_3", False)
                        #     cfg.edit_setting("DEBUG_PREPROC_BIN_5", False)
                        # if imgui.menu_item("(debug) se_model bin3", selected=cfg.settings["DEBUG_PREPROC_BIN_3"])[0]:
                        #     cfg.edit_setting("DEBUG_PREPROC_BIN_3", not cfg.settings["DEBUG_PREPROC_BIN_3"])
                        #     cfg.edit_setting("DEBUG_PREPROC_BIN_2", False)
                        #     cfg.edit_setting("DEBUG_PREPROC_BIN_5", False)
                        # if imgui.menu_item("(debug) se_model bin5", selected=cfg.settings["DEBUG_PREPROC_BIN_5"])[0]:
                        #     cfg.edit_setting("DEBUG_PREPROC_BIN_5", not cfg.settings["DEBUG_PREPROC_BIN_5"])
                        #     cfg.edit_setting("DEBUG_PREPROC_BIN_2", False)
                        #     cfg.edit_setting("DEBUG_PREPROC_BIN_3", False)
                        if imgui.menu_item("(debug) FLAG_A", selected=cfg.settings["DEBUG_A"])[0]:
                            cfg.edit_setting("DEBUG_A", not cfg.settings["DEBUG_A"])
                        if imgui.menu_item("(debug) FLAG_B", selected=cfg.settings["DEBUG_B"])[0]:
                            cfg.edit_setting("DEBUG_B", not cfg.settings["DEBUG_B"])
                        if imgui.menu_item("(debug) FLAG_C", selected=cfg.settings["DEBUG_C"])[0]:
                            cfg.edit_setting("DEBUG_C", not cfg.settings["DEBUG_C"])
                        imgui.end_menu()

                    imgui.end_menu()

                if imgui.begin_menu("Controls"):
                    imgui.text(cfg.controls_info_text)
                    imgui.end_menu()

                # Party mode lives in its own top-level menu, shown only while
                # party mode is on (the sparkle glass button toggles it).
                if not cfg.settings.get("PROGRESSION_HIDE", True):
                    if imgui.begin_menu("Party mode"):
                        if imgui.begin_menu("Background"):
                            _eqbg = progression.equipped_background_id()
                            for _bid, _bname in progression.background_choices():
                                if imgui.menu_item(_bname, None, _bid == _eqbg)[0]:
                                    progression.equip_background(_bid)
                            imgui.end_menu()
                        if imgui.begin_menu("Effects"):
                            for _pk, _pl in progression.EFFECT_TOGGLES:
                                _cur = bool(cfg.settings.get(_pk, True))
                                if imgui.menu_item(_pl, None, _cur)[0]:
                                    cfg.edit_setting(_pk, not _cur)
                            imgui.end_menu()
                        if imgui.menu_item("Levels", None, progression.is_skills_open())[0]:
                            progression.toggle_skills_panel()
                        if imgui.menu_item("Turn off party mode")[0]:
                            cfg.edit_setting("PROGRESSION_HIDE", True)
                        imgui.end_menu()
                SegmentationEditor.MENU_BAR_END_X = imgui.get_cursor_screen_pos()[0]
                SegmentationEditor.MENU_BAR_H = imgui.get_window_height()
                imgui.end_main_menu_bar()

            imgui.pop_style_color(6)
            imgui.pop_style_var(1)

        def slicer_window():
            if cfg.se_active_frame is None:
                return

            imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, cfg.CE_WIDGET_ROUNDING)
            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, cfg.CE_WIDGET_ROUNDING)
            imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1.0)

            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_PANEL_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *cfg.COLOUR_PANEL_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_PANEL_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *cfg.COLOUR_FRAME_EXTRA_DARK[0:3], 0.8)
            imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *cfg.COLOUR_FRAME_EXTRA_DARK[0:3], 0.8)
            imgui.set_next_window_size(SegmentationEditor.SLICER_WINDOW_WIDTH, 0.0)
            window_x_pos = SegmentationEditor.MAIN_WINDOW_WIDTH + (self.window.width - SegmentationEditor.MAIN_WINDOW_WIDTH - SegmentationEditor.SLICER_WINDOW_WIDTH) / 2

            export_mode = self.export_limit_range and self.active_tab == "Export"
            vertical_offset = self.window.height - SegmentationEditor.SLICER_WINDOW_VERTICAL_OFFSET - (50 if export_mode else 23)
            imgui.set_next_window_position(window_x_pos, vertical_offset)
            imgui.begin("##slicer", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_BACKGROUND)

            cw = imgui.get_content_region_available_width()
            imgui.push_item_width(cw)

            frame = cfg.se_active_frame
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_FRAME_BACKGROUND[0:3], 0.7)
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_FRAME_BACKGROUND[0:3], 0.7)
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *cfg.COLOUR_FRAME_BACKGROUND[0:3], 0.7)
            imgui.push_style_color(imgui.COLOR_BORDER, 0.3, 0.3, 0.3, 1.0)

            if export_mode:
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                _, frame.export_top = imgui.slider_int("##export_top", frame.export_top, 1, frame.n_slices, format=f"export stop {frame.export_top + 1}")
                imgui.pop_style_var(1)
                if _:
                    frame.set_slice(frame.export_top)
                    SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
                    frame.export_bottom = min([frame.export_bottom, frame.export_top - 1])
                    frame.export_bottom = max([frame.export_bottom, 1])
                origin = imgui.get_window_position()
                y = imgui.get_cursor_screen_pos()[1]
                draw_list = imgui.get_background_draw_list()
                left = 8 + origin[0] + cw * frame.export_bottom / frame.n_slices
                right = 8 + origin[0] + cw * frame.export_top / frame.n_slices
                draw_list.add_rect_filled(left, y, right, y + 18, imgui.get_color_u32_rgba(*cfg.COLOUR_POSITIVE), 10)

            _, requested_slice = imgui.slider_int("##slicer_slider", frame.current_slice, 0, frame.n_slices, format=f"slice {1+frame.current_slice}/{frame.n_slices}")
            if _:
                frame.set_slice(requested_slice)
                SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
                SegmentationEditor.VIEW_REQUIRES_UPDATE = True
            if export_mode:
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                _, frame.export_bottom = imgui.slider_int("##export_bottom", frame.export_bottom, 0, frame.n_slices - 1, format=f"export start {frame.export_bottom + 1}")
                imgui.pop_style_var(1)
                if _:
                    frame.set_slice(frame.export_bottom)
                    SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE |= True
                    frame.export_top = max([frame.export_top, frame.export_bottom + 1])
                    frame.export_top = min([frame.export_top, frame.n_slices])

            # zoom button
            cw = imgui.get_content_region_available_width()
            imgui.set_cursor_pos_x(cw - 22)
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (-2, -2))

            if imgui.button("-", 11, 11):
                if self.active_tab != "Render":
                    self.camera.zoom *= (1.0 - SegmentationEditor.CAMERA_ZOOM_STEP)
                else:
                    self.camera3d.distance += SegmentationEditor.VIEW_3D_MOVE_SPEED
            imgui.same_line()
            if imgui.button("+", 11, 11):
                if self.active_tab != "Render":
                    self.camera.zoom *= (1.0 + SegmentationEditor.CAMERA_ZOOM_STEP)
                else:
                    self.camera3d.distance -= SegmentationEditor.VIEW_3D_MOVE_SPEED

            imgui.pop_style_var(1)
            imgui.pop_style_color(4)
            imgui.pop_item_width()
            imgui.end()

            imgui.pop_style_var(3)
            imgui.pop_style_color(5)

        def popup_windows():
            if SegmentationEditor.PATH_VIEWER_OPEN:
                window_width = 900
                window_max_height = 450
                imgui.set_next_window_position(cfg.window_width // 2 - window_width, 17, imgui.APPEARING)
                imgui.set_next_window_size_constraints((window_width, 250), (window_width, window_max_height))
                _, SegmentationEditor.PATH_VIEWER_OPEN = imgui.begin("Path viewer", True, imgui.WINDOW_NO_SCROLLBAR)
                w_col1 = 93
                w_col3 = 380

                imgui.push_style_color(imgui.COLOR_TABLE_HEADER_BACKGROUND, *cfg.COLOUR_HEADER)
                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, 1.0, 1.0, 1.0, 0.0)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                with imgui.begin_table("pvctable", 3, imgui.TABLE_RESIZABLE | imgui.TABLE_COLUMN_NO_SORT | imgui.TABLE_BORDERS_INNER | imgui.TABLE_SCROLL_Y, 0.0, 200.0):
                    imgui.table_setup_column("ID", imgui.TABLE_COLUMN_WIDTH_FIXED, w_col1)

                    imgui.table_setup_column(".mrc path", 0)
                    imgui.table_setup_column(".scns path", 0)
                    imgui.table_setup_scroll_freeze(0, 1)
                    imgui.table_headers_row()

                    for s in cfg.se_frames:
                        imgui.table_next_row()
                        imgui.table_next_column()
                        h_id_str = hex(s.uid)
                        if len(h_id_str) > 8:
                            h_id_str = h_id_str[8:]
                        imgui.set_next_item_width(imgui.get_column_width())
                        imgui.input_text(f"##{s.uid}col1", h_id_str, 1024, imgui.INPUT_TEXT_READ_ONLY)
                        if imgui.is_item_clicked():
                            SegmentationEditor.set_active_dataset(s)
                        imgui.table_next_column()
                        mrc_found = False
                        # TODO: the highlight and if os.path.exists thing: don't do it every frame, only when changes might have occurred.
                        highlight = SegmentationEditor.PATH_VIEWER_OPEN_FIND in s.path if SegmentationEditor.PATH_VIEWER_OPEN_FIND != "" else False
                        if s.path not in SegmentationEditor.PATH_VIEWER_MISSING_DICT:
                            SegmentationEditor.PATH_VIEWER_MISSING_DICT[s.path] = os.path.exists(s.path)
                        mrc_found = SegmentationEditor.PATH_VIEWER_MISSING_DICT[s.path]
                        if not mrc_found:
                            imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_NEGATIVE)
                        if highlight:
                            imgui.table_set_background_color(imgui.TABLE_BACKGROUND_TARGET_CELL_BG, imgui.color_convert_float4_to_u32(*cfg.COLOUR_HIGHLIGHT), 1)
                        imgui.set_next_item_width(imgui.get_column_width())
                        imgui.input_text(f"##{s.uid}col2", s.path, 1024, imgui.INPUT_TEXT_READ_ONLY)

                        if not mrc_found:
                            imgui.pop_style_color(1)

                        imgui.table_next_column()
                        imgui.set_next_item_width(w_col3 - 2)
                        if hasattr(s, "scns_path"):
                            imgui.input_text(f"##{s.uid}col3", s.scns_path, 1024, imgui.INPUT_TEXT_READ_ONLY)
                        else:
                            imgui.input_text(f"##{s.uid}col3", "n/a", 1024, imgui.INPUT_TEXT_READ_ONLY)
                imgui.pop_style_color(2)
                imgui.pop_style_var(1)

                imgui.align_text_to_frame_padding()
                imgui.text("Find:")
                imgui.same_line(position=70)

                imgui.set_next_item_width(imgui.get_content_region_available_width())
                _, SegmentationEditor.PATH_VIEWER_OPEN_FIND = imgui.input_text("##pathviewfind", SegmentationEditor.PATH_VIEWER_OPEN_FIND, 1024)
                imgui.align_text_to_frame_padding()
                imgui.text("Replace:")
                imgui.same_line(position=70)
                imgui.set_next_item_width(imgui.get_content_region_available_width())
                _, SegmentationEditor.PATH_VIEWER_OPEN_REPLACE = imgui.input_text("##pathviewrepl", SegmentationEditor.PATH_VIEWER_OPEN_REPLACE, 1024)
                imgui.new_line()
                imgui.same_line(position = imgui.get_content_region_available_width() - 163)

                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                if imgui.button("Apply to .mrc paths", 170):
                    for s in cfg.se_frames:
                        if SegmentationEditor.PATH_VIEWER_OPEN_FIND in s.path:
                            s.path = s.path.replace(SegmentationEditor.PATH_VIEWER_OPEN_FIND, SegmentationEditor.PATH_VIEWER_OPEN_REPLACE)
                imgui.pop_style_var(1)
                imgui.end()

            if SegmentationEditor.SHOW_IMGUI_DEBUG:
                SegmentationEditor.SHOW_IMGUI_DEBUG = imgui.show_demo_window(closable=True)

            if SegmentationEditor.FEATURE_LIB_OPEN:
                window_width = 950
                panel_height = 104
                imgui.set_next_window_position(cfg.window_width // 2 - window_width // 2, 60, imgui.APPEARING)
                imgui.set_next_window_size_constraints((800, 250), (1960, 1960))
                imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, cfg.COLOUR_WINDOW_BACKGROUND[0], cfg.COLOUR_WINDOW_BACKGROUND[1], cfg.COLOUR_WINDOW_BACKGROUND[2], 1.0)
                imgui.push_style_color(imgui.COLOR_RESIZE_GRIP, *cfg.COLOUR_WINDOW_BACKGROUND)
                _, SegmentationEditor.FEATURE_LIB_OPEN = imgui.begin("Feature library", True, imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_COLLAPSE)
                if SegmentationEditor.FEATURE_LIB_RENAME_TARGET != cfg.active_feature_library:
                    SegmentationEditor.FEATURE_LIB_RENAME_TARGET = cfg.active_feature_library
                    SegmentationEditor.FEATURE_LIB_RENAME_TEXT = cfg.active_feature_library
                imgui.set_cursor_pos_x(12)
                imgui.align_text_to_frame_padding()
                imgui.text("Library:")
                imgui.same_line()
                imgui.set_next_item_width(180)
                _lib_names = list(cfg.feature_libraries.keys())
                _lib_changed, _lib_idx = imgui.combo("##active_flib", _lib_names.index(cfg.active_feature_library), _lib_names)
                if _lib_changed and _lib_names[_lib_idx] != cfg.active_feature_library:
                    cfg.set_active_feature_library(_lib_names[_lib_idx])
                    self.parse_available_features()
                imgui.same_line(spacing=12)
                imgui.set_next_item_width(180)
                _rename_enter, SegmentationEditor.FEATURE_LIB_RENAME_TEXT = imgui.input_text("##rename_flib", SegmentationEditor.FEATURE_LIB_RENAME_TEXT, 128, imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                imgui.same_line()
                if imgui.button("rename", 55, 0) or _rename_enter:
                    if cfg.rename_feature_library(cfg.active_feature_library, SegmentationEditor.FEATURE_LIB_RENAME_TEXT):
                        SegmentationEditor.FEATURE_LIB_RENAME_TARGET = cfg.active_feature_library
                        SegmentationEditor.FEATURE_LIB_RENAME_TEXT = cfg.active_feature_library
                imgui.same_line()
                if imgui.button("new", 45, 0):
                    cfg.add_feature_library()
                    self.parse_available_features()
                if len(cfg.feature_libraries) > 1:
                    imgui.same_line()
                    if imgui.button("delete", 55, 0):
                        imgui.open_popup("##delete_flib")
                    if imgui.begin_popup("##delete_flib"):
                        imgui.text(f"Delete library '{cfg.active_feature_library}'?")
                        if imgui.button("confirm", 65, 0):
                            cfg.delete_feature_library(cfg.active_feature_library)
                            self.parse_available_features()
                            imgui.close_current_popup()
                        imgui.same_line()
                        if imgui.button("cancel", 65, 0):
                            imgui.close_current_popup()
                        imgui.end_popup()
                imgui.same_line(position=imgui.get_content_region_available_width() - 181)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                _, SegmentationEditor.FEATURE_LIB_OPEN_INCLUDE_SESSION = imgui.checkbox("include session features", SegmentationEditor.FEATURE_LIB_OPEN_INCLUDE_SESSION)
                imgui.pop_style_var(2)
                if len(cfg.feature_library) == 0:
                    imgui.push_style_color(imgui.COLOR_TABLE_BORDER_STRONG, cfg.COLOUR_WINDOW_BACKGROUND[0] * 0.8, cfg.COLOUR_WINDOW_BACKGROUND[1]* 0.8, cfg.COLOUR_WINDOW_BACKGROUND[2]* 0.8, 1.0)
                    imgui.push_style_color(imgui.COLOR_TABLE_BORDER_LIGHT, cfg.COLOUR_WINDOW_BACKGROUND[0] * 0.8, cfg.COLOUR_WINDOW_BACKGROUND[1]* 0.8, cfg.COLOUR_WINDOW_BACKGROUND[2]* 0.8, 1.0)
                else:
                    imgui.push_style_color(imgui.COLOR_TABLE_BORDER_STRONG, cfg.COLOUR_WINDOW_BACKGROUND[0], cfg.COLOUR_WINDOW_BACKGROUND[1], cfg.COLOUR_WINDOW_BACKGROUND[2], 1.0)
                    imgui.push_style_color(imgui.COLOR_TABLE_BORDER_LIGHT, cfg.COLOUR_WINDOW_BACKGROUND[0], cfg.COLOUR_WINDOW_BACKGROUND[1], cfg.COLOUR_WINDOW_BACKGROUND[2], 1.0)

                j = 0
                _table_n_columns = imgui.get_window_width() // 220
                visible_features = cfg.feature_library if not SegmentationEditor.FEATURE_LIB_HIDE_DEACTIVATED_FEATURES else [f for f in cfg.feature_library if f.use]
                visible_features = [f for f in visible_features if SegmentationEditor.FEATURE_LIB_HIGHLIGHT_TEXT in f.title]
                with imgui.begin_table("flib_table", _table_n_columns, imgui.TABLE_COLUMN_NO_SORT | imgui.TABLE_SCROLL_Y | imgui.TABLE_NO_BORDERS_IN_BODY | imgui.TABLE_BORDERS_OUTER, outer_size_height = imgui.get_window_height() - 100):

                    for j, feature in enumerate(visible_features):
                        if j % _table_n_columns == 0:
                            imgui.table_next_row()
                        imgui.table_next_column()
                        if feature.use:
                            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *feature.colour, 0.057)
                            imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT)
                        else:
                            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.0, 0.0, 0.0, 0.037)
                            imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT_DISABLED)
                        if SegmentationEditor.FEATURE_LIB_HIGHLIGHT_TEXT != '' and SegmentationEditor.FEATURE_LIB_HIGHLIGHT_TEXT in feature.title:
                            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_FLIB_HIGHLIGHT)

                        with imgui.begin_child(f"flib_element{j}", imgui.get_column_width(), panel_height, border=True):

                            imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)

                            _, feature.colour = imgui.color_edit3("##clrftrlib", feature.colour[0], feature.colour[1],feature.colour[2],imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                            imgui.same_line()
                            imgui.set_next_item_width(imgui.get_content_region_available_width())
                            _, feature.title = imgui.input_text("##titleftrlib", feature.title, 128)
                            if _:
                                SegmentationEditor.FEATURE_LIB_HIGHLIGHT_TEXT = ''



                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                            imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                            imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)

                            imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_CHECK_MARK)

                            if SegmentationEditor.FEATURE_LIB_ANNOTATION:
                                imgui.push_item_width(imgui.get_content_region_available_width() - 38)
                                _, feature.brush_size = imgui.slider_float("brush", feature.brush_size, 1.0, 25.0, format=f"{feature.brush_size:.1f} nm")
                                _, feature.box_size = imgui.slider_int("boxes", feature.box_size, 32, 256, format=f"{feature.box_size} pixel")
                                feature.box_size = int(np.round(feature.box_size / 32) * 32)
                                _, feature.alpha = imgui.slider_float("alpha", feature.alpha, 0.0, 1.0, format="%.2f")
                                imgui.pop_item_width()
                                sort_requested, sort_by = imgui.checkbox("sort", cfg.FeatureLibraryFeature.SORT_TITLE == feature.title)
                                if sort_requested:
                                    if cfg.FeatureLibraryFeature.SORT_TITLE == feature.title:
                                        cfg.FeatureLibraryFeature.SORT_TITLE = "\n"
                                    else:
                                        cfg.FeatureLibraryFeature.SORT_TITLE = feature.title
                                        cfg.sort_frames_by_feature(feature.title)
                            else:
                                imgui.push_item_width(imgui.get_content_region_available_width())
                                _, feature.level = imgui.slider_int("##level", feature.level, 0, 256, f"level = {feature.level:.1f} ")
                                _, feature.dust = imgui.slider_float("##dust ", feature.dust, 1.0, 1000000.0, f"dust < {feature.dust:.1f} nm³", imgui.SLIDER_FLAGS_LOGARITHMIC)
                                _, feature.render_alpha = imgui.slider_float("##alpha", feature.render_alpha, 0.0, 1.0, f"alpha = {feature.render_alpha:.1f}")
                                imgui.pop_item_width()
                                _, feature.hide = imgui.checkbox("hide", feature.hide)


                            imgui.same_line()
                            _, feature.use = imgui.checkbox("use", feature.use)
                            imgui.same_line(position = imgui.get_content_region_available_width() - 5)
                            if imgui.image_button(self.icon_close.renderer_id, 13, 13):
                                cfg.feature_library.remove(feature)
                            imgui.pop_style_var(5)
                            imgui.pop_style_color(1)
                        imgui.pop_style_color(2)
                        if SegmentationEditor.FEATURE_LIB_HIGHLIGHT_TEXT != '' and SegmentationEditor.FEATURE_LIB_HIGHLIGHT_TEXT in feature.title:
                            imgui.pop_style_color(1)
                        imgui.spacing()

                    j+=1

                    if j % _table_n_columns == 0:
                        imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.push_style_color(imgui.COLOR_BORDER, 0.0, 0.0, 0.0, 0.1)
                    with imgui.begin_child(f"flib_element{j}", imgui.get_column_width(), panel_height, border=True):

                        imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                        imgui.push_style_color(imgui.COLOR_BORDER, 0.43, 0.43, 0.5, 0.5)
                        imgui.set_cursor_pos_y(panel_height // 2 - 20)
                        if widgets.centred_button("+", 40, 40, rounding=20):
                            cfg.feature_library.append(cfg.FeatureLibraryFeature())
                        imgui.pop_style_var(1)
                        imgui.pop_style_color(1)
                    imgui.pop_style_color(1)

                    if SegmentationEditor.FEATURE_LIB_OPEN_INCLUDE_SESSION:
                        for j, feature in enumerate(cfg.feature_library_session.values(), start=j+1):
                            if j % _table_n_columns == 0:
                                imgui.table_next_row()
                            imgui.table_next_column()

                            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *cfg.COLOUR_SESSION_FEATURE)
                            imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT)

                            imgui.push_style_color(imgui.COLOR_BORDER, *cfg.COLOUR_SESSION_FEATURE_BORDER)
                            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_FRAME_BACKGROUND_LIGHT)
                            if SegmentationEditor.FEATURE_LIB_HIGHLIGHT_TEXT != '' and SegmentationEditor.FEATURE_LIB_HIGHLIGHT_TEXT in feature.title:
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_FLIB_HIGHLIGHT)
                            with imgui.begin_child(f"flib_element{j}", imgui.get_column_width(), panel_height, border=True):

                                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)

                                _, feature.colour = imgui.color_edit3("##clr", feature.colour[0], feature.colour[1],feature.colour[2],imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                                imgui.same_line()
                                imgui.set_next_item_width(imgui.get_content_region_available_width())
                                _, feature.title = imgui.input_text("##title", feature.title, 128, imgui.INPUT_TEXT_READ_ONLY)

                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                                imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                                imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)

                                imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_CHECK_MARK)

                                if SegmentationEditor.FEATURE_LIB_ANNOTATION:
                                    imgui.push_item_width(imgui.get_content_region_available_width() - 38)
                                    _, feature.brush_size = imgui.slider_float("brush", feature.brush_size, 1.0, 25.0, format=f"{feature.brush_size:.1f} nm")
                                    _, feature.box_size = imgui.slider_int("boxes", feature.box_size, 8, 128, format=f"{feature.box_size} pixel")
                                    _, feature.alpha = imgui.slider_float("alpha", feature.alpha, 0.0, 1.0, format="%.2f")
                                    imgui.pop_item_width()
                                    sort_requested, sort_by = imgui.checkbox("sort", cfg.FeatureLibraryFeature.SORT_TITLE == feature.title)
                                    if sort_requested:
                                        if cfg.FeatureLibraryFeature.SORT_TITLE == feature.title:
                                            cfg.FeatureLibraryFeature.SORT_TITLE = ""
                                        else:
                                            cfg.FeatureLibraryFeature.SORT_TITLE = feature.title
                                            cfg.sort_frames_by_feature(feature.title)
                                else:
                                    imgui.push_item_width(imgui.get_content_region_available_width())
                                    _, feature.level = imgui.slider_int("##level", feature.level, 0, 256, f"level = {feature.level:.1f} nm")
                                    _, feature.dust = imgui.slider_float("##dust ", feature.dust, 1.0, 1000000.0, f"dust < {feature.dust:.1f} nm³", imgui.SLIDER_FLAGS_LOGARITHMIC)
                                    _, feature.render_alpha = imgui.slider_float("##alpha", feature.render_alpha, 0.0, 1.0, f"alpha = {feature.render_alpha:.1f}")
                                    imgui.pop_item_width()
                                    _, feature.hide = imgui.checkbox("hide", feature.hide)

                                imgui.same_line()
                                if imgui.button("save", 36, 14):
                                    cfg.feature_library.append(feature)
                                    self.parse_available_features()
                                imgui.pop_style_var(5)
                                imgui.pop_style_color(1)
                            imgui.pop_style_color(4)
                            if SegmentationEditor.FEATURE_LIB_HIGHLIGHT_TEXT != '' and SegmentationEditor.FEATURE_LIB_HIGHLIGHT_TEXT in feature.title:
                                imgui.pop_style_color(1)
                            imgui.spacing()

                imgui.pop_style_color(2)

                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)

                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))

                all_active = not False in [f.use for f in cfg.feature_library]
                enable_all, _ = imgui.checkbox("activate all", all_active)
                if _ and enable_all:
                    for f in cfg.feature_library:
                        f.use = True
                imgui.same_line()
                none_active = not True in [f.use for f in cfg.feature_library]
                disable_all, _ = imgui.checkbox("disable all", none_active)
                if _ and disable_all:
                    for f in cfg.feature_library:
                        f.use = False
                imgui.same_line()
                _, SegmentationEditor.FEATURE_LIB_HIDE_DEACTIVATED_FEATURES = imgui.checkbox("hide deactivated features", SegmentationEditor.FEATURE_LIB_HIDE_DEACTIVATED_FEATURES)



                imgui.same_line(position=imgui.get_content_region_available_width() - 270)


                imgui.align_text_to_frame_padding()
                imgui.text("settings for:")
                imgui.same_line()
                _, SegmentationEditor.FEATURE_LIB_ANNOTATION = imgui.checkbox("annotation",SegmentationEditor.FEATURE_LIB_ANNOTATION)
                imgui.same_line()
                toggled, _ = imgui.checkbox("rendering", not SegmentationEditor.FEATURE_LIB_ANNOTATION)
                if toggled:
                    SegmentationEditor.FEATURE_LIB_ANNOTATION = not SegmentationEditor.FEATURE_LIB_ANNOTATION
                imgui.pop_style_var(1)

                imgui.align_text_to_frame_padding()
                imgui.text('Search:')
                imgui.set_next_item_width(180)
                imgui.same_line()
                _, SegmentationEditor.FEATURE_LIB_HIGHLIGHT_TEXT = imgui.input_text('##highlight', SegmentationEditor.FEATURE_LIB_HIGHLIGHT_TEXT, 128)
                imgui.same_line()
                imgui.text('Sort:')
                imgui.same_line()
                imgui.set_next_item_width(180)
                sort_changed, SegmentationEditor.FEATURE_LIB_SORT_MODE = imgui.combo('##sort_mode', SegmentationEditor.FEATURE_LIB_SORT_MODE, ['Alphabetically', 'By Hue'])
                if sort_changed:
                    if SegmentationEditor.FEATURE_LIB_SORT_MODE == 0:
                        cfg.feature_library.sort(key=lambda f: f.title.lower())
                    elif SegmentationEditor.FEATURE_LIB_SORT_MODE == 1:
                        import colorsys
                        cfg.feature_library.sort(key=lambda f: colorsys.rgb_to_hsv(*f.colour)[0])
                imgui.same_line(position=imgui.get_content_region_available_width() - 175)

                if imgui.button("reset", 55, 25):
                    cfg.reload_feature_libraries()
                    self.parse_available_features()
                imgui.same_line()
                if imgui.button("apply", 55, 25):
                    cfg.apply_feature_library()
                    self.parse_available_features()
                imgui.same_line()
                if imgui.button("save", 55, 25):
                    cfg.save_feature_library()
                    self.parse_available_features()
                imgui.pop_style_var(2)

                imgui.end()
                imgui.pop_style_color(2)

        menu_bar()

        if cfg.se_active_frame is not None:
            if self.active_tab != "Render":
                pxd = SegmentationEditor.renderer.render_filtered_frame(cfg.se_active_frame, self.camera, self.window, self.filters, emphasize_roi=self.active_tab not in ["Segmentation", "Render"])
                if pxd is not None:
                    cfg.se_active_frame.rendered_data = pxd
                    cfg.se_active_frame.update_image_texture()
                    cfg.se_active_frame.compute_histogram(pxd)
                    if cfg.se_active_frame.autocontrast:
                        cfg.se_active_frame.compute_autocontrast(None, pxd)
                SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE = False
            if self.active_tab == "Segmentation":
                SegmentationEditor.renderer.render_segmentations(cfg.se_active_frame, self.camera)

                # render drawing ROI indicator
                active_feature = cfg.se_active_frame.active_feature
                if active_feature is not None:
                    radius = active_feature.brush_size * active_feature.parent.pixel_size
                    world_position = self.camera.cursor_to_world_position(self.window.cursor_pos)
                    if not SegmentationEditor.is_shift_down():
                        SegmentationEditor.renderer.add_circle(world_position, radius, active_feature.colour)
                    elif not SegmentationEditor.is_ctrl_down():
                        SegmentationEditor.renderer.add_square(world_position, active_feature.box_size_nm, active_feature.colour)

                # Pom template picking indicator
                if cfg.se_active_frame is not None and SegmentationEditor.is_shift_down() and SegmentationEditor.is_ctrl_down():
                    world_position = self.camera.cursor_to_world_position(self.window.cursor_pos)
                    SegmentationEditor.renderer.add_square(world_position, cfg.se_active_frame.pixel_size * cfg.settings["POM_TEMPLATE_PX_SIZE"], (1.0, 1.0, 1.0))
                for f in cfg.se_active_frame.features:
                    frame_xy = f.parent.transform.translation
                    if f.show_boxes and not f.hide and f.current_slice in f.boxes:
                        for box in f.boxes[f.current_slice]:
                            box_x_pos = frame_xy[0] + (box[0] - f.parent.width / 2) * f.parent.pixel_size
                            box_y_pos = frame_xy[1] + (box[1] - f.parent.height / 2) * f.parent.pixel_size
                            SegmentationEditor.renderer.add_square((box_x_pos, box_y_pos), f.box_size_nm, f.colour)
            if self.active_tab in ["Models", "Export"]:
                SegmentationEditor.renderer.render_models(cfg.se_active_frame, self.camera)
            if self.active_tab == "Models" and self.show_trainset_boxes:
                for f in cfg.se_active_frame.features:
                    frame_xy = f.parent.transform.translation
                    if f.current_slice in f.boxes and f.title in self.trainset_feature_selection:
                        sval = self.trainset_feature_selection[f.title]
                        if sval == 0.0:
                            continue
                        clr = cfg.COLOUR_POSITIVE if sval == 1.0 else cfg.COLOUR_NEGATIVE
                        for box in f.boxes[f.current_slice]:
                            box_x_pos = frame_xy[0] + (box[0] - f.parent.width / 2) * f.parent.pixel_size
                            box_y_pos = frame_xy[1] + (box[1] - f.parent.height / 2) * f.parent.pixel_size
                            #box_size = SegmentationEditor.trainset_apix * self.trainset_boxsize / 10.0
                            box_size = f.parent.pixel_size * self.trainset_boxsize
                            SegmentationEditor.renderer.add_square((box_x_pos, box_y_pos), box_size, clr)
            if self.active_tab != "Render":
                overlay_blend_mode = SegmentationEditor.BLEND_MODES[SegmentationEditor.BLEND_MODES_LIST[SegmentationEditor.OVERLAY_BLEND_MODE]]
                SegmentationEditor.renderer.render_overlay(cfg.se_active_frame, self.camera, overlay_blend_mode, SegmentationEditor.OVERLAY_ALPHA)
            else:
                if SegmentationEditor.RENDER_BOX:
                    # Render 3D box around the volume
                    self.renderer.render_line_va(self.pick_box_va, self.camera3d)

                # Render lines along edges of frame
                self.renderer.render_frame_border(cfg.se_active_frame, self.camera3d)

                # Render surface models in 3D
                if not imgui.is_key_down(glfw.KEY_Q):
                    self.renderer.render_surface_models(cfg.se_surface_models, self.camera3d, SegmentationEditor.LIGHT_AMBIENT_STRENGTH, SegmentationEditor.LIGHT_SPOT, window_size=(self.window.width, self.window.height))

                # Render the frame
                pxd = SegmentationEditor.renderer.render_filtered_frame(cfg.se_active_frame, self.camera, self.window, self.filters, camera3d=self.camera3d)
                if pxd is not None:
                    cfg.se_active_frame.rendered_data = pxd
                    cfg.se_active_frame.update_image_texture()
                    cfg.se_active_frame.compute_histogram(pxd)
                    if cfg.se_active_frame.autocontrast:
                        cfg.se_active_frame.compute_autocontrast(None, pxd)
                SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE = False

                if not imgui.is_key_down(glfw.KEY_Q):
                    self.renderer.render_surface_model_particles(cfg.se_surface_models, self.camera3d)

                # Render overlay, if possible:
                if cfg.se_active_frame.overlay is not None:
                    if SegmentationEditor.VIEW_REQUIRES_UPDATE:
                        self.renderer.ray_trace_overlay((self.window.width, self.window.height), cfg.se_active_frame, self.camera3d, self.pick_box_quad_va)
                    if self.pick_overlay_3d:
                        self.renderer.render_overlay_3d(SegmentationEditor.OVERLAY_ALPHA, SegmentationEditor.OVERLAY_INTENSITY)
                    else:
                        overlay_blend_mode = SegmentationEditor.BLEND_MODES[SegmentationEditor.BLEND_MODES_LIST[SegmentationEditor.OVERLAY_BLEND_MODE]]
                        self.renderer.render_overlay(cfg.se_active_frame, self.camera3d, overlay_blend_mode, SegmentationEditor.OVERLAY_ALPHA)
        else:
            if self.active_tab == "Render":
                # Render surface models in 3D
                if not imgui.is_key_down(glfw.KEY_Q):
                    self.renderer.render_surface_models(cfg.se_surface_models, self.camera3d, SegmentationEditor.LIGHT_AMBIENT_STRENGTH, SegmentationEditor.LIGHT_SPOT, window_size=(self.window.width, self.window.height))

        if not imgui.get_io().want_capture_keyboard:
            if imgui.is_key_pressed(glfw.KEY_1):
                SegmentationEditor.FORCE_SELECT_TAB = 0
            if imgui.is_key_pressed(glfw.KEY_2):
                SegmentationEditor.FORCE_SELECT_TAB = 1
            if imgui.is_key_pressed(glfw.KEY_3):
                SegmentationEditor.FORCE_SELECT_TAB = 2
            if imgui.is_key_pressed(glfw.KEY_4):
                SegmentationEditor.FORCE_SELECT_TAB = 3

        # MAIN WINDOW
        imgui.set_next_window_position(0, 17, imgui.ONCE)
        imgui.set_next_window_size(SegmentationEditor.MAIN_WINDOW_WIDTH, self.window.height - 17)

        # slightly more transparent than other panels so the background shows
        # through (most visibly in the empty area below the controls)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *cfg.COLOUR_PANEL_BACKGROUND[:3], 0.85)
        imgui.begin("##se_main", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        shared_gui()
        imgui.spacing()
        imgui.spacing()

        # a tab switch requested from inside a tab's content (e.g. 'copy output to annotation') can only take
        # effect on the following frame, once the Annotation tab item is drawn again with FORCE_SELECT_TAB set.
        if SegmentationEditor.FORCE_SELECT_TAB_NEXT is not None:
            SegmentationEditor.FORCE_SELECT_TAB = SegmentationEditor.FORCE_SELECT_TAB_NEXT
            SegmentationEditor.FORCE_SELECT_TAB_NEXT = None

        if imgui.begin_tab_bar("##tabs"):
            self.window.clear_color = cfg.COLOUR_WINDOW_BACKGROUND
            if imgui.begin_tab_item("  Annotation ", flags=imgui.TAB_ITEM_SET_SELECTED if SegmentationEditor.FORCE_SELECT_TAB == 0 else 0)[0]:
                segmentation_tab()
                self.active_tab = "Segmentation"   # used to be labelled Segmentation then changed to 'Annotation'. Did not change the active_tab enum value though.
                imgui.end_tab_item()
            if imgui.begin_tab_item(" Models ", flags=imgui.TAB_ITEM_SET_SELECTED if SegmentationEditor.FORCE_SELECT_TAB == 1 else 0)[0]:
                if self.active_tab != "Models":
                    self.parse_available_features()
                self.active_tab = "Models"
                models_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item(" Export ", flags=imgui.TAB_ITEM_SET_SELECTED if SegmentationEditor.FORCE_SELECT_TAB == 2 else 0)[0]:
                self.active_tab = "Export"
                export_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item(" Render  ", flags=imgui.TAB_ITEM_SET_SELECTED if SegmentationEditor.FORCE_SELECT_TAB == 3 else 0)[0]:
                self.window.clear_color = [*SegmentationEditor.RENDER_CLEAR_COLOUR, 1.0]
                self.active_tab = "Render"
                picking_tab()
                imgui.end_tab_item()

            SegmentationEditor.FORCE_SELECT_TAB = None
            imgui.end_tab_bar()

        imgui.end()
        imgui.pop_style_color(1)

        slicer_window()
        self._warning_window()
        boot_sprite()
        popup_windows()
        imgui.pop_style_color(32)
        imgui.pop_style_var(1)

    def _warning_window(self):
        def ww_context_menu():
            imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *cfg.COLOUR_MENU_WINDOW_BACKGROUND)
            if imgui.begin_popup_context_window():
                cfg.error_window_active = True
                copy_error, _ = imgui.menu_item("Copy to clipboard")
                if copy_error:
                    pyperclip.copy(cfg.error_msg)
                copy_path_to_log, _ = imgui.menu_item("Copy path to scNodes.log")
                if copy_path_to_log:
                    pyperclip.copy(os.path.abspath(cfg.logpath))
                imgui.end_popup()
            imgui.pop_style_color(1)
        ## Error message
        if cfg.error_msg is not None:
            if cfg.error_new:
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *cfg.COLOUR_ERROR_WINDOW_HEADER_NEW)
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *cfg.COLOUR_ERROR_WINDOW_HEADER_NEW)
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *cfg.COLOUR_ERROR_WINDOW_HEADER_NEW)
            else:
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *cfg.COLOUR_ERROR_WINDOW_HEADER)
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *cfg.COLOUR_ERROR_WINDOW_HEADER)
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *cfg.COLOUR_ERROR_WINDOW_HEADER)
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *cfg.COLOUR_ERROR_WINDOW_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_ERROR_WINDOW_TEXT)
            imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 3.0)
            imgui.set_next_window_size(self.window.width, cfg.ERROR_WINDOW_HEIGHT)
            window_vertical_offset = 0 if cfg.error_window_active else cfg.ERROR_WINDOW_HEIGHT - 19
            imgui.set_next_window_position(0, self.window.height - cfg.ERROR_WINDOW_HEIGHT + window_vertical_offset)
            _, stay_open = imgui.begin("Notification", True, imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE)
            if imgui.is_window_focused():
                cfg.error_window_active = True
                if self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.PRESS):
                    cfg.error_new = False
            else:
                cfg.error_window_active = False
            if not cfg.error_logged:
                cfg.write_to_log(cfg.error_msg)
                cfg.error_logged = True
            imgui.text(cfg.error_msg)
            ww_context_menu()
            imgui.end()
            if not stay_open:
                cfg.error_msg = None
                cfg.error_new = True
            imgui.pop_style_color(5)
            imgui.pop_style_var(1)

    def parse_available_features(self):
        # parse_available_features is called whenever a list of all session features is required.
        # one example: in Feature box, right-click the title to get a list of all present features, to copy their settings
        # another: in generating trainign datasets, to present the list of available features to sample boxes from.
        # 240909: adding feature library functionality.

        self.all_feature_names = list()
        self.feature_colour_dict = dict()
        cfg.feature_library_session = dict()

        feature_lib_titles = [f.title for f in cfg.feature_library]
        for feature in cfg.feature_library:
            if feature.use:
                self.feature_colour_dict[feature.title] = feature.colour
        for sef in cfg.se_frames:
            for ftr in sef.features:
                if sef.sample:
                    if ftr.title not in self.all_feature_names:
                        self.all_feature_names.append(ftr.title)
                    if ftr.title not in self.trainset_feature_selection:
                        self.trainset_feature_selection[ftr.title] = 0.0
                if ftr.title not in self.feature_colour_dict and not "Unnamed feature" in ftr.title:
                    self.feature_colour_dict[ftr.title] = ftr.colour
                if ftr.title not in feature_lib_titles:
                    session_feature = cfg.FeatureLibraryFeature()
                    cfg.feature_library_session[ftr.title] = session_feature
                    session_feature.colour = ftr.colour
                    session_feature.box_size = ftr.box_size
                    session_feature.alpha = ftr.alpha
                    session_feature.title = ftr.title

        to_pop = list()

        # Sanitize the selection of features to sample from. If a selected key is no longer available, remove it.
        for key in self.trainset_feature_selection.keys():
            if key not in self.all_feature_names:
                to_pop.append(key)
        for key in to_pop:
            self.trainset_feature_selection.pop(key)

    def render_party_buttons(self):
        # Three frosted-glass circular buttons at the top-left, just right of the
        # main menu bar: toggle Party mode, toggle dark mode, open the feature
        # library. No labels.
        D, G = 40.0, 10.0
        win_w, win_h = 3.0 * D + 2.0 * G, D
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
        imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (0, 0))
        imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0.0)
        # top-left, just right of the main menu bar items but never over the left panel
        win_x = max(SegmentationEditor.MENU_BAR_END_X + 14.0, SegmentationEditor.MAIN_WINDOW_WIDTH + 14.0)
        imgui.set_next_window_position(win_x, SegmentationEditor.MENU_BAR_H + 14.0, imgui.ALWAYS)
        imgui.set_next_window_size(win_w, win_h)
        imgui.begin("##party_buttons", False,
                    imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE
                    | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS
                    | imgui.WINDOW_NO_FOCUS_ON_APPEARING | imgui.WINDOW_NO_NAV | imgui.WINDOW_NO_BACKGROUND)
        party_on = not cfg.settings.get("PROGRESSION_HIDE", True)
        _px, _py = imgui.get_cursor_screen_pos()
        if self._glass_circle_button("##party_toggle", D, "sparkle", party_on, (0.98, 0.78, 0.30)):
            new_hidden = not cfg.settings["PROGRESSION_HIDE"]
            cfg.edit_setting("PROGRESSION_HIDE", new_hidden)
            if not new_hidden:   # turning party ON -> celebrate (only drawn while visible)
                progression.particles.emit_confetti_burst(_px + D * 0.5, _py + D * 0.5)
        self.tooltip("Party mode: " + ("on" if party_on else "off"))
        imgui.same_line(spacing=G)
        dark_on = cfg.settings.get("DARK_MODE", False)
        if self._glass_circle_button("##dark_toggle", D, "moon", dark_on, (0.62, 0.72, 1.0)):
            cfg.edit_setting("DARK_MODE", not cfg.settings["DARK_MODE"])
            cfg.set_theme(cfg.settings["DARK_MODE"])
        self.tooltip("Dark mode: " + ("on" if dark_on else "off"))
        imgui.same_line(spacing=G)
        if self._glass_circle_button("##flib_toggle", D, "library", SegmentationEditor.FEATURE_LIB_OPEN, (0.92, 0.26, 0.26)):
            SegmentationEditor.FEATURE_LIB_OPEN = not SegmentationEditor.FEATURE_LIB_OPEN
            if SegmentationEditor.FEATURE_LIB_OPEN:
                self.parse_available_features()
        self.tooltip("Feature library")
        imgui.end()
        imgui.pop_style_var(3)

    def _glass_circle_button(self, label, D, icon, active, accent):
        dl = imgui.get_window_draw_list()
        p0 = imgui.get_cursor_screen_pos()
        clicked = imgui.invisible_button(label, D, D)
        hovered = imgui.is_item_hovered()
        cx, cy, r = p0[0] + D * 0.5, p0[1] + D * 0.5, D * 0.5
        # frosted glass: translucent light fill, top sheen, bright rim
        fill_a = (0.20 if active else 0.11) + (0.07 if hovered else 0.0)
        dl.add_circle_filled(cx, cy, r, imgui.get_color_u32_rgba(0.90, 0.93, 0.98, fill_a), 40)
        dl.add_circle_filled(cx, cy - r * 0.30, r * 0.60, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.05 + (0.05 if hovered else 0.0)), 28)
        dl.add_circle(cx, cy, r - 0.5, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.45 if hovered else 0.28), 40, 1.5)
        if active:
            dl.add_circle(cx, cy, r - 2.5, imgui.get_color_u32_rgba(accent[0], accent[1], accent[2], 0.85), 40, 2.0)
        icol = imgui.get_color_u32_rgba(accent[0], accent[1], accent[2], 1.0) if active else imgui.get_color_u32_rgba(0.82, 0.82, 0.86, 0.9)
        self._draw_party_icon(dl, icon, cx, cy, r, icol, active)
        return clicked

    @staticmethod
    def _draw_party_icon(dl, icon, cx, cy, r, col, active=False):
        if icon == "sparkle":
            def star(ox, oy, s, wv):
                dl.add_quad_filled(cx + ox, cy + oy - s, cx + ox + wv, cy + oy, cx + ox, cy + oy + s, cx + ox - wv, cy + oy, col)
                dl.add_quad_filled(cx + ox - s, cy + oy, cx + ox, cy + oy - wv, cx + ox + s, cy + oy, cx + ox, cy + oy + wv, col)
            star(0.0, 0.0, r * 0.52, r * 0.15)
            star(r * 0.44, -r * 0.40, r * 0.20, r * 0.06)
        elif icon == "moon":  # half-filled circle: a theme / dark-mode toggle
            ri = r * 0.44
            dl.add_circle(cx, cy, ri, col, 28, 1.6)
            prev = None
            for i in range(13):
                a = math.pi * 0.5 + math.pi * (i / 12.0)   # bottom -> left -> top = left half
                pt = (cx + math.cos(a) * ri, cy + math.sin(a) * ri)
                if prev is not None:
                    dl.add_triangle_filled(cx, cy, prev[0], prev[1], pt[0], pt[1], col)
                prev = pt
        else:  # "library" - a 2x2 grid of rounded tiles
            g, gap = r * 0.17, r * 0.07
            step = g + gap
            if active:
                # when the library is open, the four tiles pop in confetti colours
                tiles = [(-1, -1, imgui.get_color_u32_rgba(0.20, 0.82, 0.38, 1.0)),   # green
                         ( 1, -1, imgui.get_color_u32_rgba(1.00, 0.85, 0.15, 1.0)),   # yellow
                         ( 1,  1, imgui.get_color_u32_rgba(0.30, 0.60, 1.00, 1.0)),   # blue-ish
                         (-1,  1, imgui.get_color_u32_rgba(0.95, 0.25, 0.85, 1.0))]   # magenta
            else:
                tiles = [(sx, sy, col) for sx in (-1, 1) for sy in (-1, 1)]
            for sx, sy, tcol in tiles:
                ox, oy = sx * step, sy * step
                dl.add_rect_filled(cx + ox - g, cy + oy - g, cx + ox + g, cy + oy + g, tcol, 2.0, imgui.DRAW_ROUND_CORNERS_ALL)

    def _gui_background_process_progress_bar(self, process, colour=cfg.COLOUR_POSITIVE, cancellable=False, height=None, transparent_background=False):
        height = SegmentationEditor.PROGRESS_BAR_HEIGHT if height is None else height
        cw = imgui.get_content_region_available_width()
        cancel_button_w = SegmentationEditor.PROGRESS_BAR_HEIGHT + 5 if cancellable else 0
        origin = imgui.get_window_position()
        y = imgui.get_cursor_screen_pos()[1]
        drawlist = imgui.get_window_draw_list()
        drawlist.add_rect_filled(8 + origin[0], y,
                                 8 + origin[0] + max(0, cw - cancel_button_w),
                                 y + height,
                                 imgui.get_color_u32_rgba(*cfg.COLOUR_NEUTRAL) if not transparent_background else imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.0))
        drawlist.add_rect_filled(8 + origin[0], y,
                                 8 + origin[0] + max(0, cw * min([1.0, process.progress]) - cancel_button_w),
                                 y + height,
                                 imgui.get_color_u32_rgba(*colour))
        imgui.dummy(cw - cancel_button_w, height)
        retval = False
        if cancellable:
            imgui.same_line(spacing=5)
            retval = imgui.image_button(self.icon_close.renderer_id, SegmentationEditor.PROGRESS_BAR_HEIGHT, SegmentationEditor.PROGRESS_BAR_HEIGHT)
        return retval

    def _gui_feature_title_context_menu(self, feature_or_model, popup_id=None):
        items = list(self.feature_colour_dict.items())
        max_rows = 16
        n_cols = max(1, (len(items) + max_rows - 1) // max_rows)
        col_width = 200
        if n_cols > 1:
            imgui.set_next_window_content_size(col_width * n_cols, 0)
        if popup_id is not None:
            opened = imgui.begin_popup(popup_id)
        else:
            opened = imgui.begin_popup_context_item()
        if opened:
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
            if n_cols > 1:
                imgui.columns(n_cols, "##feature_ctx_cols", border=False)
                for c in range(n_cols):
                    imgui.set_column_width(c, col_width)
            for i, (t, rgb) in enumerate(items):
                if n_cols > 1 and i > 0 and i % max_rows == 0:
                    imgui.next_column()
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 7)
                imgui.color_button(f"##clrbutton{t}", rgb[0], rgb[1], rgb[2], 1.0, 0, 14, 14)
                imgui.pop_style_var(1)
                imgui.same_line(spacing=5)
                imgui.selectable(t)
                if imgui.is_item_hovered():
                    feature_or_model.title = t
                    feature_or_model.colour = self.feature_colour_dict[t]
                    if isinstance(feature_or_model, Segmentation):
                        flib_titles = [f.title for f in cfg.feature_library]
                        if t in flib_titles:
                            library_feature = cfg.feature_library[flib_titles.index(t)]
                            feature_or_model.brush_size = library_feature.brush_size / (feature_or_model.parent.pixel_size * 2.0)
                            feature_or_model.set_box_size(library_feature.box_size)
                            feature_or_model.alpha = library_feature.alpha
                imgui.spacing()
            if n_cols > 1:
                # Jump to last column for the Feature library menu
                last_col = n_cols - 1
                current_col = min(len(items) - 1, len(items)) // max_rows
                while current_col < last_col:
                    imgui.next_column()
                    current_col += 1

            if imgui.begin_menu("Feature library"):
                if imgui.menu_item("disable saved features")[0]:
                    for feature in cfg.feature_library:
                        feature.use = False
                    self.parse_available_features()
                if imgui.menu_item("open feature library")[0]:
                    SegmentationEditor.FEATURE_LIB_OPEN = True
                imgui.end_menu()
            if n_cols > 1:
                imgui.columns(1)
            imgui.pop_style_var(1)
            imgui.end_popup()

    @staticmethod
    def extract_coordinates_menu(feature):
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *feature.colour)
        imgui.text("Extract coordinates:")
        imgui.spacing()
        _, SegmentationEditor.EXTRACT_ALL = imgui.checkbox("all datasets", SegmentationEditor.EXTRACT_ALL)
        _, extract_single = imgui.checkbox("this dataset", not SegmentationEditor.EXTRACT_ALL)
        if _:
            SegmentationEditor.EXTRACT_ALL = not extract_single
        imgui.spacing()

        imgui.text("Minimum particle spacing:")
        cw = imgui.get_content_region_available_width()
        imgui.set_next_item_width(cw)
        _, SegmentationEditor.EXTRACT_MIN_SPACING = imgui.slider_float("##min_dist", SegmentationEditor.EXTRACT_MIN_SPACING, 0.1, 25.0, format = f"{SegmentationEditor.EXTRACT_MIN_SPACING:.1f} nm")

        imgui.spacing()
        imgui.text("Output format:")
        _, SegmentationEditor.EXTRACT_STAR_FILE = imgui.checkbox(".star", SegmentationEditor.EXTRACT_STAR_FILE)
        _, extract_tsv = imgui.checkbox(".tsv", not SegmentationEditor.EXTRACT_STAR_FILE)
        if _:
            SegmentationEditor.EXTRACT_STAR_FILE = not extract_tsv

        imgui.spacing()
        if widgets.centred_button("start", 50, 20):
            SegmentationEditor.launch_export_coordinates(feature, mesh=False, star_format=SegmentationEditor.EXTRACT_STAR_FILE)
            imgui.close_current_popup()

        imgui.pop_style_color()

    @staticmethod
    def extract_meshes_menu(feature):
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *feature.colour)
        imgui.text("Extract mesh:            ")
        imgui.spacing()
        _, SegmentationEditor.EXTRACT_ALL = imgui.checkbox("all datasets", SegmentationEditor.EXTRACT_ALL)
        _, extract_single = imgui.checkbox("this dataset", not SegmentationEditor.EXTRACT_ALL)
        if _:
            SegmentationEditor.EXTRACT_ALL = not extract_single

        imgui.spacing()
        if widgets.centred_button("start", 50, 20):
            SegmentationEditor.launch_export_coordinates(feature, mesh=True)
            imgui.close_current_popup()

        imgui.pop_style_color()

    @staticmethod
    def tooltip(text):
        if imgui.is_item_hovered() and not imgui.is_mouse_down(0):
            if SegmentationEditor.TOOLTIP_HOVERED_TIMER == 0.0:
                SegmentationEditor.TOOLTIP_HOVERED_START_TIME = time.time()
                SegmentationEditor.TOOLTIP_HOVERED_TIMER = 0.001  # add a fake 1 ms to get bypass of this if clause in the next iter.
            elif SegmentationEditor.TOOLTIP_HOVERED_TIMER > SegmentationEditor.TOOLTIP_APPEAR_DELAY:
                imgui.set_tooltip(text)
            else:
                SegmentationEditor.TOOLTIP_HOVERED_TIMER = time.time() - SegmentationEditor.TOOLTIP_HOVERED_START_TIME
        if not imgui.is_any_item_hovered():
            SegmentationEditor.TOOLTIP_HOVERED_TIMER = 0.0

    @staticmethod
    def launch_export_coordinates(feature, mesh=False, star_format=True):
        try:
            if SegmentationEditor.EXTRACT_ALL:
                datasets = glob.glob(os.path.join(SegmentationEditor.seg_folder, f'*__{feature.title}.mrc'))
            else:
                se_frame = cfg.se_active_frame
                print(os.path.join(SegmentationEditor.seg_folder, f'{se_frame.title}__{feature.title}.mrc'))
                datasets = glob.glob(os.path.join(SegmentationEditor.seg_folder, f'{se_frame.title}__{feature.title}.mrc'))[0]

            # TODO: remove datasets that aren't open in Ais

            if isinstance(datasets, str):
                datasets = [datasets]
            if not datasets:
                print("No datasets selected")
                return

            print(f"Exporting {'coordinates' if not mesh else 'meshes'} for the following datasets:\n")
            for d in datasets:
                print("\t"+d)
            print()

            if not mesh:
                for d in datasets:
                    SegmentationEditor.queued_exports.append(QueuedExtract(d, threshold=feature.level, min_size=feature.dust, min_spacing=SegmentationEditor.EXTRACT_MIN_SPACING, save_dir=SegmentationEditor.seg_folder, binning=feature.bin, star_format=star_format))
                    SegmentationEditor.queued_exports[-1].colour = feature.colour
            else:
                for d in datasets:
                    SegmentationEditor.queued_exports.append(QueuedMeshExtract(d, threshold=feature.level, min_size=feature.dust, save_dir=SegmentationEditor.seg_folder, binning=feature.bin, pixel_size=feature.pixel_size))
                    SegmentationEditor.queued_exports[-1].colour = feature.colour
            if SegmentationEditor.queued_exports:
                if SegmentationEditor.queued_exports[0].process.progress == 0.0:
                    SegmentationEditor.queued_exports[0].start()



        except Exception as e:
            cfg.set_error(e, "Could not export coordinates - see details below:")

    def launch_export_volumes(self):
        try:
            datasets = [d for d in cfg.se_frames if d.export]
            models = [m for m in cfg.se_models if m.export]
            if not datasets:
                return
            if not os.path.isdir(SegmentationEditor.export_dir):
                os.makedirs(SegmentationEditor.export_dir)

            for d in datasets:
                SegmentationEditor.queued_exports.append(QueuedExport(SegmentationEditor.export_dir, d, models, self.export_batch_size, self.export_overlays))

            if SegmentationEditor.queued_exports:
                if SegmentationEditor.queued_exports[0].process.progress == 0.0:
                    SegmentationEditor.queued_exports[0].start()
        except Exception as e:
            cfg.set_error(e, "Could not export volumes - see details below:")

    def launch_create_training_set(self):
        positive_feature_names = list()
        negative_feature_names = list()
        for f in self.all_feature_names:
            if self.trainset_feature_selection[f] == 1:
                positive_feature_names.append(f)
            elif self.trainset_feature_selection[f] == -1:
                negative_feature_names.append(f)
        if len(positive_feature_names) == 0:
            return

        path = filedialog.asksaveasfilename(filetypes=[("Ais training data", cfg.filetype_traindata)], initialfile=f"{int(self.trainset_boxsize)}x{int(self.trainset_boxsize)}x{int(self.trainset_boxdepth)}_{positive_feature_names[0]}")
        if not isinstance(path, str) or path == '':
            return
        if path[-len(cfg.filetype_traindata):] != cfg.filetype_traindata:
            path += cfg.filetype_traindata

        datasets_to_sample = list()
        for dataset in cfg.se_frames:
            if dataset.sample:
                datasets_to_sample.append(dataset)

        n_boxes = self.trainset_num_boxes_positive + self.trainset_num_boxes_negative
        if n_boxes == 0:
            return
        args = (path, n_boxes, positive_feature_names, negative_feature_names, datasets_to_sample, self.trainset_boxsize, self.trainset_boxdepth, SegmentationEditor.trainset_apix)

        process = BackgroundProcess(self._create_training_set, args)
        process.start()
        self.active_trainset_exports.append(process)

    def _create_training_set(self, path, n_boxes, positives, negatives, datasets, boxsize, boxdepth, apix, process):
        # This routes through the same extraction/writing code as the `ais extract` CLI
        # (Ais.core.se_scnt), so the GUI and CLI produce identical new-format .scnt files.
        try:
            counter = {'n': 0}

            def on_box():
                counter['n'] += 1
                if n_boxes:
                    process.set_progress(min(0.99, counter['n'] / n_boxes))

            samples = []
            for d in datasets:
                if not os.path.exists(d.path):
                    continue
                tomo_stem = os.path.splitext(os.path.basename(d.path))[0]
                mrcf = mrcfile.mmap(d.path, mode="r", permissive=True)
                flavour_data = {se_scnt.DEFAULT_ANNOTATED_FLAVOUR: mrcf.data}
                try:
                    for f in d.features:
                        if f.title in positives:
                            is_negative = False
                        elif f.title in negatives:
                            is_negative = True
                        else:
                            continue
                        samples.extend(se_scnt.extract_feature_samples(
                            tomo_stem, flavour_data, se_scnt.DEFAULT_ANNOTATED_FLAVOUR, f,
                            boxsize, boxdepth, binning=1, is_negative=is_negative,
                            tomo_path=d.path, on_box=on_box))
                finally:
                    try:
                        mrcf.close()
                    except Exception:
                        pass

            if not samples:
                process.set_progress(1.0)
                return

            se_scnt.write_training_set(path, samples, apix=apix, features=list(positives))
            process.set_progress(1.0)
        except Exception as e:
            cfg.set_error(e, "Could not create training set - see details below:")
            process.set_progress(1.0)

    @staticmethod
    def seframe_from_clemframe(clemframe):
        if not os.path.exists(clemframe.path):
            cfg.set_error(Exception(f"No file exists at path {clemframe.path}"), "Could not convert CLEMFrame to SEFrame - see below.")
            return
        new_se_frame = SEFrame(clemframe.path)
        new_se_frame.clem_frame = clemframe
        new_se_frame.clem_frame_path = clemframe.path
        new_se_frame.autocontrast = False
        new_se_frame.title = clemframe.title
        new_se_frame.contrast_lims = clemframe.contrast_lims
        new_se_frame.set_slice(clemframe.current_slice)

        cfg.se_frames.append(new_se_frame)
        SegmentationEditor.set_active_dataset(cfg.se_frames[-1])
        return cfg.se_frames[-1]

    @staticmethod
    def load_model_group(path):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract the .tar.gz archive to the temporary directory
                with tarfile.open(path, 'r') as tar:
                    tar.extractall(path=temp_dir)

                # Assuming the first file is always the interactions JSON,
                # and the rest are model files
                files = os.listdir(temp_dir)
                json_path = [f for f in files if f.endswith('_interactions.json')][0]
                model_files = [f for f in files if f != json_path]

                # Load models
                for m_file in model_files:
                    model_path = os.path.join(temp_dir, m_file)
                    # Assuming SEModel has a load method that accepts a file path
                    model = SEModel()
                    model.load(model_path)
                    cfg.se_models.append(model)

                # Load interactions
                with open(os.path.join(temp_dir, json_path), 'r') as f:
                    interaction_dict_list = json.load(f)
                    for d in interaction_dict_list:
                        ModelInteraction.from_dict(d)

        except Exception as e:
            cfg.set_error(e, "Error loading model group, see details below.")

    @staticmethod
    def load_model(path):
        try:
            model = SEModel()
            model.load(path)
            cfg.se_models.append(model)
            cfg.push_recent("RECENT_MODELS", path)
        except Exception as e:
            cfg.set_error(e, "Error loading model, see details below.")

    @staticmethod
    def save_model_group(path):
        try:
            group_name = os.path.splitext(os.path.basename(path))[0]

            # Create a temporary directory to hold the files before archiving
            with tempfile.TemporaryDirectory() as temp_dir:
                model_paths = list()
                i = 0
                for m in cfg.se_models:
                    m_path = os.path.join(temp_dir, group_name + f"_model_{i}" + cfg.filetype_semodel)
                    m.save(m_path)  # Save model directly to the temporary directory
                    model_paths.append(m_path)
                    i += 1

                # Save interactions as json in the temporary directory
                interaction_dict_list = list()
                for interaction in ModelInteraction.all:
                    interaction_dict_list.append(interaction.as_dict())
                json_path = os.path.join(temp_dir, f"{group_name}_interactions.json")
                with open(json_path, 'w') as outfile:
                    json.dump(interaction_dict_list, outfile, indent=2)

                # Create a .tar.gz archive and add all the model and JSON files
                with tarfile.open(path, 'w') as tar:
                    for file_path in model_paths + [json_path]:
                        tar.add(file_path, arcname=os.path.basename(file_path))

        except Exception as e:
            cfg.set_error(e, "Error saving model group, see details below")

    @staticmethod
    def save_surface_models_as_objs():
        obj_path = os.path.dirname(cfg.se_active_frame.path)
        base_name = os.path.splitext(os.path.basename(cfg.se_active_frame.path))[0]
        paths = list()
        for m in cfg.se_surface_models:
            try:
                m_path = m.save_as_obj(path=os.path.join(obj_path, f"{base_name}__{m.title}.obj"))
                if m_path is not None:
                    paths.append(m_path)
            except Exception as e:
                cfg.set_error(e, f"Error saving {base_name} {m.title} as .obj.")
        return paths

    @staticmethod
    def _write_launch_script(template_name, substitutions):
        # fill the template's data lines (matched by prefix) and write to a temp file rather than
        # editing the packaged template in place - the install dir is often read-only (clusters).
        with open(os.path.join(cfg.root, "core", template_name), "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            for prefix, value in substitutions.items():
                if line.startswith(prefix):
                    lines[i] = f"{prefix}= {value}\n"
                    break
        fd, script_path = tempfile.mkstemp(prefix="ais_", suffix=f"_{template_name}")
        with os.fdopen(fd, "w") as f:
            f.writelines(lines)
        return script_path

    @staticmethod
    def open_objs_in_blender(paths):
        try:
            script_path = SegmentationEditor._write_launch_script("open_in_blender.py", {"obj_paths ": paths})
            subprocess.Popen([cfg.settings["BLENDER_EXE"], "--python", script_path])
        except Exception as e:
            cfg.set_error(e, "Could not open models in Blender - is the path to the Blender executable set? See Settings -> 3rd Party Applications -> Blender ")

    @staticmethod
    def open_in_chimerax(surface_models):
        try:
            paths = list()
            level = list()
            colour = list()
            dust = list()
            for m in surface_models:
                paths.append(m.path)
                level.append(m.level)
                colour.append(m.colour)
                dust.append(m.dust)
            script_path = SegmentationEditor._write_launch_script("open_in_chimerax.py", {
                "paths ": paths,
                "level ": level,
                "colour ": colour,
                "dust ": dust,
                "bgclr ": list(SegmentationEditor.RENDER_CLEAR_COLOUR),
            })
            subprocess.Popen([cfg.settings["CHIMERAX_EXE"], script_path])
        except Exception as e:
            cfg.set_error(e, "Could not open volumes in ChimeraX - is the path to the ChimeraX executable set? See Settings -> 3rd Party Applications -> ChimeraX.")

    def camera_control(self):
        if imgui.get_io().want_capture_mouse or imgui.get_io().want_capture_keyboard:
            pass
        elif self.active_tab != "Render":
            if self.window.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE):
                delta_cursor = self.window.cursor_delta
                self.camera.position[0] += delta_cursor[0] / self.camera.zoom
                self.camera.position[1] -= delta_cursor[1] / self.camera.zoom
            if SegmentationEditor.is_shift_down() and self.window.scroll_delta[1] != 0.0:
                # zoom toward the cursor: keep the world point under the cursor fixed
                self.camera.on_update()
                world_before = self.camera.cursor_to_world_position(self.window.cursor_pos)
                self.camera.zoom *= (1.0 + self.window.scroll_delta[1] * SegmentationEditor.CAMERA_ZOOM_STEP)
                self.camera.zoom = min([self.camera.zoom, SegmentationEditor.CAMERA_MAX_ZOOM])
                self.camera.on_update()
                world_after = self.camera.cursor_to_world_position(self.window.cursor_pos)
                self.camera.position[0] += world_after[0] - world_before[0]
                self.camera.position[1] += world_after[1] - world_before[1]
        else:
            if SegmentationEditor.is_shift_down():
                if self.window.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE):
                    # ChimeraX-style tumble: drag near the centre turns the scene in 3D,
                    # drag near the edges rolls about the view axis.
                    self.camera3d.arcball_rotate(self.window.cursor_pos_previous_frame, self.window.cursor_pos, self.window.width, self.window.height, SegmentationEditor.VIEW_3D_PIVOT_SPEED * 1.5)
                    if self.window.cursor_delta[0] != 0 or self.window.cursor_delta[1] != 0:
                        SegmentationEditor.VIEW_REQUIRES_UPDATE = True
                self.camera3d.distance -= self.window.scroll_delta[1] * SegmentationEditor.VIEW_3D_MOVE_SPEED
                if self.window.scroll_delta[1] != 0:
                    SegmentationEditor.VIEW_REQUIRES_UPDATE = True
                self.camera3d.distance = max(10.0, self.camera3d.distance)
            elif SegmentationEditor.is_ctrl_down():
                rotated = False
                if imgui.is_key_pressed(glfw.KEY_LEFT, True):
                    self.camera3d.rotate_world(self.camera3d.get_up(), -5)
                    rotated = True
                elif imgui.is_key_pressed(glfw.KEY_RIGHT, True):
                    self.camera3d.rotate_world(self.camera3d.get_up(), 5)
                    rotated = True
                if imgui.is_key_pressed(glfw.KEY_UP, True):
                    self.camera3d.rotate_world(self.camera3d.get_right(), -5)
                    rotated = True
                elif imgui.is_key_pressed(glfw.KEY_DOWN, True):
                    self.camera3d.rotate_world(self.camera3d.get_right(), 5)
                    rotated = True
                if rotated:
                    SegmentationEditor.VIEW_REQUIRES_UPDATE = True
            elif self.window.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE):
                cdx, cdy = self.window.cursor_delta[0], self.window.cursor_delta[1]
                # pixels -> world units at the focus plane (perspective, 60 deg vertical fov). Move the
                # focus opposite the cursor in x and with it in y (screen y is flipped) so the grabbed
                # point stays under the cursor.
                scale = 2.0 * self.camera3d.distance * np.tan(np.radians(30.0)) / max(1, self.window.height)
                world_delta = self.camera3d.cursor_delta_to_world_delta((-cdx * scale, cdy * scale))
                self.camera3d.focus[0] += world_delta[0]
                self.camera3d.focus[1] += world_delta[1]
                self.camera3d.focus[2] += world_delta[2]
                if cdx != 0 or cdy != 0:
                    SegmentationEditor.VIEW_REQUIRES_UPDATE = True

    def end_frame(self):
        self.window.end_frame()

    @staticmethod
    def set_log_path(path):
        cfg.log_path = path
        cfg.start_log()

    @staticmethod
    def force_not_embedded():
        global EMBEDDED
        EMBEDDED = False


    @staticmethod
    def is_shift_down():
        return imgui.is_key_down(glfw.KEY_LEFT_SHIFT) or imgui.is_key_down(glfw.KEY_RIGHT_SHIFT)

    @staticmethod
    def is_ctrl_down():
        return imgui.is_key_down(glfw.KEY_LEFT_CONTROL) or imgui.is_key_down(glfw.KEY_RIGHT_CONTROL)


class Brush:
    circular_roi = np.zeros(1, dtype=np.uint8)
    circular_roi_radius = -1
    magic_roi = np.zeros(1, dtype=np.uint8)
    magic_roi_radius = -1

    @staticmethod
    def set_circular_roi_radius(radius):
        # radius is a float; the grid half-size is ceil(radius) but the inclusion test
        # uses the float radius, so fractional sizes give distinct masks
        if Brush.circular_roi_radius == radius:
            return
        Brush.circular_roi_radius = radius
        R = int(np.ceil(radius))
        Brush.circular_roi = np.zeros((2*R+1, 2*R+1), dtype=np.uint8)
        r2 = radius**2
        for x in range(0, 2*R+1):
            for y in range(0, 2*R+1):
                if ((x-R)**2 + (y-R)**2) < r2:
                    Brush.circular_roi[x, y] = True

    @staticmethod
    def set_magic_roi_radius(radius):
        if Brush.magic_roi_radius == radius:
            return
        Brush.magic_roi_radius = radius
        R = int(np.ceil(radius))
        Brush.magic_roi = np.zeros((2 * R + 1, 2 * R + 1), dtype=np.uint8)
        r2 = radius ** 2
        for x in range(0, 2 * R + 1):
            for y in range(0, 2 * R + 1):
                if ((x - R) ** 2 + (y - R) ** 2) < r2:
                    Brush.magic_roi[x, y] = True

    @staticmethod
    def apply_circular(segmentation, center_coordinates, val=True):
        # check if the current slice already exists; if not, make it.
        segmentation.request_draw_in_current_slice()
        radius = float(segmentation.brush_size)
        r = int(np.ceil(radius))   # integer grid half-size, matches the mask array
        center_coordinates[0], center_coordinates[1] = center_coordinates[1], center_coordinates[0]
        Brush.set_circular_roi_radius(radius)

        x = [center_coordinates[0] - r, center_coordinates[0] + r + 1]
        y = [center_coordinates[1] - r, center_coordinates[1] + r + 1]
        rx = [0, 2 * r + 1]
        ry = [0, 2 * r + 1]
        if x[0] > segmentation.height or x[1] < 0 or y[0] > segmentation.width or y[1] < 0:
            return
        if x[0] < 0:
            rx[0] -= x[0]
            x[0] = 0
        if y[0] < 0:
            ry[0] -= y[0]
            y[0] = 0
        if x[1] > segmentation.height:
            rx[1] -= (x[1] - segmentation.height)
            x[1] = segmentation.height
        if y[1] > segmentation.width:
            ry[1] -= (y[1] - segmentation.width)
            y[1] = segmentation.width

        if val:
            segmentation.data[x[0]:x[1], y[0]:y[1]] += Brush.circular_roi[rx[0]:rx[1], ry[0]:ry[1]]
        else:
            segmentation.data[x[0]:x[1], y[0]:y[1]] *= (np.uint8(1.0) - Brush.circular_roi[rx[0]:rx[1], ry[0]:ry[1]])

        segmentation.data[x[0]:x[1], y[0]:y[1]] = np.clip(segmentation.data[x[0]:x[1], y[0]:y[1]], 0, 1)
        segmentation.texture.update_subimage(segmentation.data[x[0]:x[1], y[0]:y[1]], y[0], x[0])

    @staticmethod
    def apply_magic(segmentation, image, center_coordinates):
        segmentation.request_draw_in_current_slice()
        radius = float(segmentation.brush_size)
        r = int(np.ceil(radius))   # integer grid half-size, matches the mask array
        center_coordinates[0], center_coordinates[1] = center_coordinates[1], center_coordinates[0]
        Brush.set_magic_roi_radius(radius)

        # set up the ROI coordinates and image coordinates of the region to sample from
        x, y = center_coordinates[0], center_coordinates[1]
        if x > segmentation.height or x < 0 or y > segmentation.width or y < 0:
            return


        mu = image[x, y]
        value_range = [mu * (1.0 - (100.0 - (50 + 0.5 * segmentation.magic_strength)) / 100.0), mu * (1.0 + (100.0 - (50 + 0.5 * segmentation.magic_strength)) / 100.0)]
        if value_range[0] > value_range[1]:
            value_range[0], value_range[1] = value_range[1], value_range[0]
        # set up the ROI coordinates and image coordinates of the region to draw in
        r = r
        x = [center_coordinates[0] - r, center_coordinates[0] + r + 1]
        y = [center_coordinates[1] - r, center_coordinates[1] + r + 1]
        rx = [0, 2 * r + 1]
        ry = [0, 2 * r + 1]

        if x[0] > segmentation.height or x[1] < 0 or y[0] > segmentation.width or y[1] < 0:
            return
        if x[0] < 0:
            rx[0] -= x[0]
            x[0] = 0
        if y[0] < 0:
            ry[0] -= y[0]
            y[0] = 0
        if x[1] > segmentation.height:
            rx[1] -= (x[1] - segmentation.height)
            x[1] = segmentation.height
        if y[1] > segmentation.width:
            ry[1] -= (y[1] - segmentation.width)
            y[1] = segmentation.width

        # mask in full 2r+1 ROI coordinates so edge-clamping the sampled region can't shift the
        # indices (the earlier clamped-size mask threw IndexError when clamped at the top/left edge).
        size = 2 * r + 1
        contiguous_mask = np.zeros((size, size), dtype=np.uint8)
        for _x, _rx in zip(range(x[0], x[1]), range(rx[0], rx[1])):
            for _y, _ry in zip(range(y[0], y[1]), range(ry[0], ry[1])):
                if Brush.magic_roi[_rx, _ry] and value_range[0] < image[_x, _y] < value_range[1]:
                    contiguous_mask[_rx, _ry] = 1

        stack = [(r, r)]  # the brush centre is always at (r, r) in ROI coordinates
        while stack:
            mx, my = stack.pop()
            if contiguous_mask[mx, my] == 1:
                contiguous_mask[mx, my] = 2
                if mx + 1 < size:
                    stack.append((mx + 1, my))
                if mx - 1 >= 0:
                    stack.append((mx - 1, my))
                if my + 1 < size:
                    stack.append((mx, my + 1))
                if my - 1 >= 0:
                    stack.append((mx, my - 1))
        contiguous_mask = (contiguous_mask == 2)[rx[0]:rx[1], ry[0]:ry[1]]
        segmentation.data[x[0]:x[1], y[0]:y[1]] += contiguous_mask
        segmentation.data[x[0]:x[1], y[0]:y[1]] = np.clip(segmentation.data[x[0]:x[1], y[0]:y[1]], 0, 1)
        segmentation.texture.update_subimage(segmentation.data[x[0]:x[1], y[0]:y[1]], y[0], x[0])


class Renderer:
    def __init__(self):
        self.quad_shader = Shader(os.path.join(cfg.root, "shaders", "se_quad_shader.glsl"))
        self.b_segmentation_shader = Shader(os.path.join(cfg.root, "shaders", "se_binary_segmentation_shader.glsl"))
        self.f_segmentation_shader = Shader(os.path.join(cfg.root, "shaders", "se_float_segmentation_shader.glsl"))
        self.overlay_shader = Shader(os.path.join(cfg.root, "shaders", "se_overlay_shader.glsl"))
        self.border_shader = Shader(os.path.join(cfg.root, "shaders", "se_border_shader.glsl"))
        self.kernel_filter = Shader(os.path.join(cfg.root, "shaders", "se_compute_kernel_filter.glsl"))
        self.mix_filtered = Shader(os.path.join(cfg.root, "shaders", "se_compute_mix.glsl"))
        self.line_shader = Shader(os.path.join(cfg.root, "shaders", "ce_line_shader.glsl"))
        self.icon_shader = Shader(os.path.join(cfg.root, "shaders", "se_icon_shader.glsl"))
        self.surface_model_shader = Shader(os.path.join(cfg.root, "shaders", "se_surface_model_shader.glsl"))
        self.line_3d_shader = Shader(os.path.join(cfg.root, "shaders", "se_line_3d_shader.glsl"))
        self.quad_3d_shader = Shader(os.path.join(cfg.root, "shaders", "se_quad_3d_shader.glsl"))
        self.depth_mask_shader = Shader(os.path.join(cfg.root, "shaders", "se_depth_mask_shader.glsl"))
        self.ray_trace_shader = Shader(os.path.join(cfg.root, "shaders", "se_overlay_ray_trace_shader.glsl"))
        self.overlay_blend_shader = Shader(os.path.join(cfg.root, "shaders", "se_overlay_blend_shader.glsl"))
        self.particle_shader = Shader(os.path.join(cfg.root, "shaders", "se_particle_shader.glsl"))
        self.edge_shader = Shader(os.path.join(cfg.root, "shaders", "se_depth_edge_detect.glsl"))
        try:
            self.background_blob_shader = Shader(os.path.join(cfg.root, "shaders", "se_background_blobs.glsl"))
        except Exception as _e:
            print(f"Background shader failed to load: {_e}")
            self.background_blob_shader = None
        self.line_list = list()
        self.line_list_s = list()
        self.line_va = VertexArray(None, None, attribute_format="xyrgb")
        self.fbo1 = FrameBuffer(100, 100)
        self.fbo2 = FrameBuffer(100, 100)
        self.fbo3 = FrameBuffer(100, 100)
        self.ray_trace_fbo_a = FrameBuffer(100, 100)
        self.ray_trace_fbo_b = FrameBuffer(100, 100)
        self.img_fbo = FrameBuffer()
        self.img_fbo_size = [0.0, 0.0]
        self.ndc_screen_va = VertexArray(attribute_format="xy")
        self.ndc_screen_va.update(VertexBuffer([-1, -1, 1, -1, 1, 1, -1, 1]), IndexBuffer([0, 1, 2, 0, 2, 3]))
        self.ray_trace_fbo_size = [0.0, 0.0]
        vertices, indices = icosphere_va()
        self.particle_va = VertexArray(VertexBuffer(vertices), IndexBuffer(indices), attribute_format="xyz")

    def recompile_shaders(self):  # for debugging
        try:
            self.quad_shader = Shader(os.path.join(cfg.root, "shaders", "se_quad_shader.glsl"))
            self.b_segmentation_shader = Shader(os.path.join(cfg.root, "shaders", "se_binary_segmentation_shader.glsl"))
            self.f_segmentation_shader = Shader(os.path.join(cfg.root, "shaders", "se_float_segmentation_shader.glsl"))
            self.overlay_shader = Shader(os.path.join(cfg.root, "shaders", "se_overlay_shader.glsl"))
            self.border_shader = Shader(os.path.join(cfg.root, "shaders", "se_border_shader.glsl"))
            self.kernel_filter = Shader(os.path.join(cfg.root, "shaders", "se_compute_kernel_filter.glsl"))
            self.mix_filtered = Shader(os.path.join(cfg.root, "shaders", "se_compute_mix.glsl"))
            self.line_shader = Shader(os.path.join(cfg.root, "shaders", "ce_line_shader.glsl"))
            self.icon_shader = Shader(os.path.join(cfg.root, "shaders", "se_icon_shader.glsl"))
            self.surface_model_shader = Shader(os.path.join(cfg.root, "shaders", "se_surface_model_shader.glsl"))
            self.line_3d_shader = Shader(os.path.join(cfg.root, "shaders", "se_line_3d_shader.glsl"))
            self.quad_3d_shader = Shader(os.path.join(cfg.root, "shaders", "se_quad_3d_shader.glsl"))
            self.depth_mask_shader = Shader(os.path.join(cfg.root, "shaders", "se_depth_mask_shader.glsl"))
            self.ray_trace_shader = Shader(os.path.join(cfg.root, "shaders", "se_overlay_ray_trace_shader.glsl"))
            self.overlay_blend_shader = Shader(os.path.join(cfg.root, "shaders", "se_overlay_blend_shader.glsl"))
            self.particle_shader = Shader(os.path.join(cfg.root, "shaders", "se_particle_shader.glsl"))
            self.edge_shader = Shader(os.path.join(cfg.root, "shaders", "se_depth_edge_detect.glsl"))
        finally:
            pass

    def render_filtered_frame(self, se_frame, camera, window, filters, camera3d=None, emphasize_roi=False):
        se_frame.update_model_matrix()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)

        self.fbo1.clear((0.0, 0.0, 0.0, 1.0))
        self.fbo2.clear((0.0, 0.0, 0.0, 1.0))
        self.fbo3.clear((0.0, 0.0, 0.0, 2.0))

        # if any filters will be applied below, reset the frame's data to the original raw pixel data
        pxd = None
        override_contrast_roi = False
        if SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE:
            # set the frame's .rendered_data to the raw image data - then render as usual.
            cfg.se_active_frame.rendered_data = cfg.se_active_frame.data
            cfg.se_active_frame.update_image_texture()
            pxd = cfg.se_active_frame.rendered_data
            override_contrast_roi = True

        # render the image to a framebuffer
        fake_camera_matrix = np.matrix([[2 / self.fbo1.width, 0, 0, 0], [0, 2 / self.fbo1.height, 0, 0], [0, 0, -2 / 100, 0], [0, 0, 0, 1]])
        self.fbo1.bind()
        self.quad_shader.bind()
        se_frame.quad_va.bind()
        se_frame.texture.bind(0)
        self.quad_shader.uniformmat4("cameraMatrix", fake_camera_matrix)
        self.quad_shader.uniformmat4("modelMatrix", np.identity(4))
        self.quad_shader.uniform1f("alpha", se_frame.alpha)
        self.quad_shader.uniform1f("contrastMin", 0.0)
        self.quad_shader.uniform1f("contrastMax", 1.0)
        glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.quad_shader.unbind()
        se_frame.quad_va.unbind()
        glActiveTexture(GL_TEXTURE0)
        self.fbo1.unbind()
        window.set_full_viewport()

        # filter framebuffer - but only if the SegmentationEditor says that the image needs updating!
        if SegmentationEditor.FRAME_TEXTURE_REQUIRES_UPDATE:
            self.kernel_filter.bind()
            compute_size = (int(np.ceil(se_frame.width / 16)), int(np.ceil(se_frame.height / 16)), 1)
            for fltr in filters:
                if not fltr.enabled:
                    continue
                self.kernel_filter.bind()

                # horizontal shader pass
                fltr.bind(horizontal=True)
                glBindImageTexture(0, self.fbo1.texture.renderer_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
                glBindImageTexture(1, self.fbo2.texture.renderer_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)
                self.kernel_filter.uniform1i("direction", 0)
                glDispatchCompute(*compute_size)
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

                # vertical shader pass
                fltr.bind(horizontal=False)
                glBindImageTexture(0, self.fbo2.texture.renderer_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
                glBindImageTexture(1, self.fbo3.texture.renderer_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)
                self.kernel_filter.uniform1i("direction", 1)
                glDispatchCompute(*compute_size)
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
                fltr.unbind()
                self.kernel_filter.unbind()

                # mix the filtered and the original image
                self.mix_filtered.bind()
                glBindImageTexture(0, self.fbo3.texture.renderer_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
                glBindImageTexture(1, self.fbo1.texture.renderer_id, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)
                self.mix_filtered.uniform1f("strength", fltr.strength)
                glDispatchCompute(*compute_size)
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
                self.mix_filtered.unbind()

            # update histogram
            self.fbo1.bind()
            pxd = glReadPixels(0, 0, self.fbo1.width, self.fbo1.height, GL_RED, GL_FLOAT)
            pxd = np.frombuffer(pxd, np.float32).reshape(self.fbo1.height, self.fbo1.width)
            self.fbo1.unbind()
            window.set_full_viewport()

        # render the framebuffer to the screen
        shader = self.quad_shader if not camera3d else self.quad_3d_shader
        vpmat = camera.view_projection_matrix if not camera3d else camera3d.matrix
        shader.bind()
        se_frame.quad_va.bind()
        self.fbo1.texture.bind(0)
        shader.uniformmat4("cameraMatrix", vpmat)
        shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        crop_x_lims = [se_frame.crop_roi[0] / se_frame.width, 1.0 - (se_frame.width - se_frame.crop_roi[2]) / se_frame.width]
        crop_y_lims = [(se_frame.height - se_frame.crop_roi[3]) / se_frame.height, 1.0 - se_frame.crop_roi[1] / se_frame.height]
        if override_contrast_roi:
            crop_x_lims = [0, 1]
            crop_y_lims = [0, 1]
        shader.uniform2f("xLims", crop_x_lims)
        shader.uniform2f("yLims", crop_y_lims)
        shader.uniform1f("alpha", se_frame.alpha if camera3d is None else SegmentationEditor.PICKING_FRAME_ALPHA)
        if camera3d is not None:
            glEnable(GL_DEPTH_TEST)
            shader.uniform1f("z_pos", (se_frame.current_slice - se_frame.n_slices / 2) * se_frame.pixel_size)
            shader.uniform1f("pixel_size", se_frame.pixel_size)  # This factor should be in the VA, but fixing that messed up the 2D view - cheap fix for now
        if se_frame.invert:
            shader.uniform1f("contrastMin", se_frame.contrast_lims[1])
            shader.uniform1f("contrastMax", se_frame.contrast_lims[0])
        else:
            shader.uniform1f("contrastMin", se_frame.contrast_lims[0])
            shader.uniform1f("contrastMax", se_frame.contrast_lims[1])
        if SegmentationEditor.PICKING_FRAME_ALPHA > 0.0 or camera3d is None:
            glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        shader.unbind()
        se_frame.quad_va.unbind()
        glActiveTexture(GL_TEXTURE0)
        if camera3d is not None:
            glDisable(GL_DEPTH_TEST)
        return pxd

    def render_models(self, se_frame, camera):
        # render overlays (from models)
        se_frame.quad_va.bind()
        self.f_segmentation_shader.bind()
        self.f_segmentation_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.f_segmentation_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        render_list = copy(cfg.se_models)
        if cfg.se_active_model in render_list:
            render_list.remove(cfg.se_active_model)
            render_list.append(cfg.se_active_model)
        for model in render_list:
            if not model.active or not model.show:
                continue
            if model.data is None:
                continue
            if model.blend:
                glBlendFunc(GL_DST_COLOR, GL_DST_ALPHA)
                glBlendEquation(GL_FUNC_ADD)
            else:
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glBlendEquation(GL_FUNC_ADD)
            self.f_segmentation_shader.uniform1f("alpha", model.alpha)
            clr = model.colour
            alpha = model.alpha
            self.f_segmentation_shader.uniform3f("colour", (clr[0] * alpha, clr[1] * alpha, clr[2] * alpha))
            self.f_segmentation_shader.uniform1f("threshold", model.threshold)
            model.texture.bind(0)
            glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        se_frame.quad_va.unbind()
        self.f_segmentation_shader.unbind()
        glActiveTexture(GL_TEXTURE0)

        # render border
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)
        self.border_shader.bind()
        se_frame.border_va.bind()
        self.border_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.border_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        self.border_shader.uniform1f("z_pos", 0)
        self.border_shader.uniform1f("alpha", 1.0)
        self.border_shader.uniform3f("borderColour", (0.0, 0.0, 0.0))
        glDrawElements(GL_LINES, se_frame.border_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.border_shader.unbind()
        se_frame.border_va.unbind()

    def render_segmentations(self, se_frame, camera):
        """
        se_frame: the SEFrame object to render.
        camera: a Camera object to render with.
        window: the Window object (to which the viewport size will be reset)
        filters: a list of Filter objectm, to apply to the se_frame pixeldata.
        """

        # render segmentation overlays
        se_frame.quad_va.bind()
        self.b_segmentation_shader.bind()
        self.b_segmentation_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.b_segmentation_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        render_list = copy(se_frame.features)
        if se_frame.active_feature in render_list:
            render_list.remove(se_frame.active_feature)
            render_list.append(se_frame.active_feature)
        for segmentation in render_list:
            if segmentation.hide:
                continue
            if segmentation.data is None:
                continue
            if not segmentation.contour:
                glBlendFunc(GL_DST_COLOR, GL_DST_ALPHA)
                glBlendEquation(GL_FUNC_ADD)
            else:
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glBlendEquation(GL_FUNC_ADD)
            self.b_segmentation_shader.uniform1f("alpha", segmentation.alpha)
            clr = segmentation.colour
            alpha = segmentation.alpha
            self.b_segmentation_shader.uniform3f("colour", (clr[0] * alpha, clr[1] * alpha, clr[2] * alpha))
            self.b_segmentation_shader.uniform1i("contour", int(segmentation.contour))

            segmentation.texture.bind(0)
            glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        se_frame.quad_va.unbind()
        self.b_segmentation_shader.unbind()
        glActiveTexture(GL_TEXTURE0)

        # render border
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)
        self.border_shader.bind()
        se_frame.border_va.bind()
        self.border_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.border_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        self.border_shader.uniform1f("z_pos", 0)
        self.border_shader.uniform1f("alpha", 1.0)
        self.border_shader.uniform3f("borderColour", (0.0, 0.0, 0.0))
        glDrawElements(GL_LINES, se_frame.border_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.border_shader.unbind()
        se_frame.border_va.unbind()

    def render_frame_border(self, se_frame, camera):
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)
        self.border_shader.bind()
        se_frame.border_va.bind()
        self.border_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.border_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        self.border_shader.uniform1f("z_pos", (se_frame.current_slice - se_frame.n_slices / 2))
        self.border_shader.uniform1f("alpha", SegmentationEditor.PICKING_FRAME_ALPHA)
        self.border_shader.uniform3f("borderColour", (0.0, 0.0, 0.0))
        glDrawElements(GL_LINES, se_frame.border_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.border_shader.unbind()
        se_frame.border_va.unbind()

    def render_crop_handles(self, se_frame, camera, handles):
        self.icon_shader.bind()
        h_va = handles[0].get_va()
        h_va.bind()
        handles[0].get_texture().bind(0)
        self.icon_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        for h in handles:
            h.compute_matrix(se_frame, camera)
            self.icon_shader.uniformmat4("modelMatrix", h.transform.matrix)
            self.icon_shader.uniform1i("invert", int(se_frame.invert))
            glDrawElements(GL_TRIANGLES, h_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.icon_shader.unbind()
        h_va.unbind()

    def render_surface_models(self, surface_models, camera, ambient_strength, spot_light, window_size):
        # IF required, prepare new image in the FBO
        if self.img_fbo_size != window_size:
            self.img_fbo_size = window_size
            self.img_fbo = FrameBuffer(window_size[0], window_size[1], "rgba32f")

        self.surface_model_shader.bind()
        self.surface_model_shader.uniformmat4("vpMat", camera.matrix)
        self.surface_model_shader.uniform3f("viewDir", camera.get_view_direction())
        self.surface_model_shader.uniform3f("lightDir", spot_light.vec)
        self.surface_model_shader.uniform1f("ambientStrength", ambient_strength)
        self.surface_model_shader.uniform1f("lightStrength", spot_light.strength)
        self.surface_model_shader.uniform3f("lightColour", spot_light.colour)
        self.surface_model_shader.uniform1i("style", SegmentationEditor.SELECTED_RENDER_STYLE)
        glEnable(GL_DEPTH_TEST)
        alpha_sorted_surface_models = sorted(surface_models, key=lambda x: x.alpha, reverse=True)
        for s in alpha_sorted_surface_models:
            if s.hide:
                continue
            self.surface_model_shader.uniform4f("color", [*s.colour, s.alpha])
            for blob in s.blobs.values():
                if blob.complete and not blob.hide:
                    if blob.painted:
                        self.surface_model_shader.uniform4f("color", [*blob.colour, s.alpha])
                    else:
                        self.surface_model_shader.uniform4f("color", [*s.colour, s.alpha])
                    blob.va.bind()
                    glDrawElements(GL_TRIANGLES, blob.va.indexBuffer.getCount(), GL_UNSIGNED_INT, None)
                    blob.va.unbind()
        self.surface_model_shader.unbind()
        glDisable(GL_DEPTH_TEST)
        # copy default FBO depth to img_fbo

        if SegmentationEditor.RENDER_SILHOUETTES and len(alpha_sorted_surface_models) > 0:
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.img_fbo.framebufferObject)
            glBlitFramebuffer(0, 0, window_size[0], window_size[1], 0, 0, window_size[0], window_size[1], GL_DEPTH_BUFFER_BIT, GL_NEAREST)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            self.edge_shader.bind()
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_fbo.depth_texture_renderer_id)
            self.edge_shader.uniform1f("threshold", SegmentationEditor.RENDER_SILHOUETTES_THRESHOLD)
            self.edge_shader.uniform1f("edge_alpha", SegmentationEditor.RENDER_SILHOUETTES_ALPHA)
            self.edge_shader.uniform1f("zmin", camera.clip_near)
            self.edge_shader.uniform1f("zmax", camera.clip_far)
            self.ndc_screen_va.bind()
            glDrawElements(GL_TRIANGLES, self.ndc_screen_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            self.ndc_screen_va.unbind()
            self.edge_shader.unbind()

    def render_surface_model_particles(self, surface_models, camera):
        if SegmentationEditor.RENDER_PARTICLES_XRAY:
            glDisable(GL_DEPTH_TEST)
        else:
            glEnable(GL_DEPTH_TEST)
        alpha_sorted_surface_models = sorted(surface_models, key=lambda x: x.alpha, reverse=True)
        self.particle_va.bind()
        self.particle_shader.bind()
        self.particle_shader.uniformmat4("vpMat", camera.matrix)
        for s in alpha_sorted_surface_models:
            if s.hide:
                continue
            if s.particle_size > 0.0:
                self.particle_shader.uniform3f("particleColour", s.particle_colour)
                self.particle_shader.uniform1f("particleSize", s.particle_size / 2.0)
                self.particle_shader.uniform1f("pixelSize", s.pixel_size)
                self.particle_shader.uniform3f("origin", s.center_xyz / 2.0)
                for p in s.particles:
                    self.particle_shader.uniform3f("particlePosition", p)
                    glDrawElements(GL_TRIANGLES, self.particle_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.particle_va.unbind()
        self.particle_shader.unbind()
        glDisable(GL_DEPTH_TEST)

    def render_line_va(self, va, camera):
        glEnable(GL_DEPTH_TEST)
        self.line_3d_shader.bind()
        self.line_3d_shader.uniformmat4("cameraMatrix", camera.matrix)
        va.bind()
        glDrawElements(GL_LINES, va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        va.unbind()
        self.line_3d_shader.unbind()
        glDisable(GL_DEPTH_TEST)

    def ray_trace_overlay(self, window_size, se_frame, camera, box_va):
        if se_frame.overlay is None:
            return

        # 1 - Rendering a depth mask to find where to START sampling rays.
        if self.ray_trace_fbo_size != window_size:
            self.ray_trace_fbo_size = window_size
            self.ray_trace_fbo_a.set_size(window_size[0], window_size[1])
            self.ray_trace_fbo_b.set_size(window_size[0], window_size[1])

        self.ray_trace_fbo_a.bind()
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
        glClear(GL_DEPTH_BUFFER_BIT)

        self.depth_mask_shader.bind()
        self.depth_mask_shader.uniformmat4("vpMat", camera.matrix)
        self.depth_mask_shader.uniform1i("override_z", 0)
        self.depth_mask_shader.uniform1f("override_z_val", 0.0)
        box_va.bind()  # box depth mask
        glDrawElements(GL_TRIANGLES, box_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        box_va.unbind()
        self.depth_mask_shader.unbind()
        self.ray_trace_fbo_a.unbind((0, 0, window_size[0], window_size[1]))

        # - Rendering a depth mask to find where to STOP sampling rays
        self.ray_trace_fbo_b.bind()
        glClearDepth(0.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_GREATER)
        glDepthMask(GL_TRUE)
        glClear(GL_DEPTH_BUFFER_BIT)

        self.depth_mask_shader.bind()
        self.depth_mask_shader.uniformmat4("vpMat", camera.matrix)
        self.depth_mask_shader.uniform1i("override_z", 0)
        self.depth_mask_shader.uniform1f("override_z_val", 0.0)
        box_va.bind()
        glDrawElements(GL_TRIANGLES, box_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        box_va.unbind()
        if SegmentationEditor.PICKING_FRAME_ALPHA != 0.0:  # frame used in the depth mask as well
            glDepthFunc(GL_LESS)
            self.depth_mask_shader.uniform1i("override_z", 1)
            self.depth_mask_shader.uniform1f("override_z_val", (se_frame.current_slice - se_frame.n_slices / 2) * se_frame.pixel_size)
            self.depth_mask_shader.uniform1f("pixel_size", se_frame.pixel_size)
            se_frame.quad_va.bind()  # frame depth mask as well.
            glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            se_frame.quad_va.unbind()
        self.depth_mask_shader.unbind()
        self.ray_trace_fbo_b.unbind((0, 0, window_size[0], window_size[1]))

        glDepthFunc(GL_LESS)
        glClearDepth(1.0)
        # 2 - ray tracing
        self.ray_trace_shader.bind()
        self.ray_trace_shader.uniformmat4("ipMat", camera.ipmat)
        self.ray_trace_shader.uniformmat4("ivMat", camera.ivmat)
        self.ray_trace_shader.uniformmat4("pMat", camera.pmat)
        self.ray_trace_shader.uniform1f("near", camera.clip_near)
        self.ray_trace_shader.uniform1f("far", camera.clip_far)
        self.ray_trace_shader.uniform2f("viewportSize", window_size)
        self.ray_trace_shader.uniform1f("zLim", se_frame.n_slices * se_frame.pixel_size / 2.0)
        self.ray_trace_shader.uniform1f("pixelSize", se_frame.pixel_size)
        self.ray_trace_shader.uniform2f("imgSize", [se_frame.overlay.size[1], se_frame.overlay.size[0]])
        self.ray_trace_shader.uniform1i("style", SegmentationEditor.BLEND_MODES_3D[SegmentationEditor.BLEND_MODES_LIST_3D[SegmentationEditor.OVERLAY_BLEND_MODE_3D]][3])
        self.ray_trace_shader.uniform1f("intensity", SegmentationEditor.OVERLAY_INTENSITY)
        se_frame.overlay.texture.bind(0)  # overlay image, to read from
        glActiveTexture(GL_TEXTURE0 + 1)  # depth buffer, to read from
        glBindTexture(GL_TEXTURE_2D, self.ray_trace_fbo_a.depth_texture_renderer_id)
        glActiveTexture(GL_TEXTURE0 + 2)
        glBindTexture(GL_TEXTURE_2D, self.ray_trace_fbo_b.depth_texture_renderer_id)
        self.ray_trace_fbo_a.texture.bind_image_slot(3, 1)  # fbo texture, to write to
        glDispatchCompute(int(window_size[0] // 16) + 1, int(window_size[1] // 16) + 1, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        SegmentationEditor.VIEW_REQUIRES_UPDATE = False

    def render_overlay_3d(self, alpha, intensity):
        # add overlay image to whatever has been rendered before.
        blend_mode = SegmentationEditor.BLEND_MODES_3D[SegmentationEditor.BLEND_MODES_LIST_3D[SegmentationEditor.OVERLAY_BLEND_MODE_3D]]
        glBlendFunc(blend_mode[0], blend_mode[1])
        glBlendEquation(blend_mode[2])
        glDisable(GL_DEPTH_TEST)
        self.overlay_blend_shader.bind()
        self.overlay_blend_shader.uniform1f("alpha", alpha)
        self.overlay_blend_shader.uniform1f("intensity", intensity)
        self.ray_trace_fbo_a.texture.bind(0)
        self.ndc_screen_va.bind()
        glDrawElements(GL_TRIANGLES, self.ndc_screen_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.ndc_screen_va.unbind()
        self.overlay_blend_shader.unbind()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)

    def add_line(self, start_xy, stop_xy, colour, subtract=False):
        if subtract:
            self.line_list_s.append((start_xy, stop_xy, colour))
        else:
            self.line_list.append((start_xy, stop_xy, colour))

    def add_circle(self, center_xy, radius, colour, segments=32, subtract=False):
        theta = 0
        start_xy = (center_xy[0] + radius * np.cos(theta), center_xy[1] + radius * np.sin(theta))
        for i in range(segments):
            theta = (i * 2 * np.pi / (segments - 1))
            stop_xy = (center_xy[0] + radius * np.cos(theta), center_xy[1] + radius * np.sin(theta))
            if subtract:
                self.line_list_s.append((start_xy, stop_xy, colour))
            else:
                self.line_list.append((start_xy, stop_xy, colour))
            start_xy = (center_xy[0] + radius * np.cos(theta), center_xy[1] + radius * np.sin(theta))

    def add_square(self, center_xy, size, colour, subtract=False):
        bottom = center_xy[1] - size / 2
        top = center_xy[1] + size / 2
        left = center_xy[0] - size / 2
        right = center_xy[0] + size / 2
        if subtract:
            self.line_list_s.append(((left, bottom), (right, bottom), colour))
            self.line_list_s.append(((left, top), (right, top), colour))
            self.line_list_s.append(((left, bottom), (left, top), colour))
            self.line_list_s.append(((right, bottom), (right, top), colour))
        else:
            self.line_list.append(((left, bottom), (right, bottom), colour))
            self.line_list.append(((left, top), (right, top), colour))
            self.line_list.append(((left, bottom), (left, top), colour))
            self.line_list.append(((right, bottom), (right, top), colour))

    def render_lines(self, camera):
        def render_lines_in_list(line_list):
            # make VA
            vertices = list()
            indices = list()
            i = 0
            for line in line_list:
                vertices += [*line[0], *line[2][:3]]
                vertices += [*line[1], *line[2][:3]]
                indices += [2*i, 2*i+1]
                i += 1
            # Reuse the VA's GL buffers each frame rather than allocating new
            # VertexBuffer/IndexBuffer objects (which leaked their GL buffers).
            if self.line_va.vertexBuffer is None:
                self.line_va.update(VertexBuffer(vertices), IndexBuffer(indices))
            else:
                self.line_va.vertexBuffer.update(vertices)
                self.line_va.indexBuffer.update(indices)

            # launch
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBlendEquation(GL_FUNC_ADD)
            self.line_shader.bind()
            self.line_va.bind()
            self.line_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
            glDrawElements(GL_LINES, self.line_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            self.line_shader.unbind()
            self.line_va.unbind()

        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR)
        glBlendEquation(GL_FUNC_ADD)
        render_lines_in_list(self.line_list_s)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)
        render_lines_in_list(self.line_list)

        self.line_list_s = list()
        self.line_list = list()

    def render_draw_list(self, camera):
        self.render_lines(camera)

    def render_background(self, blobs, base_col, intensity, res, shape=0):
        # Fullscreen papery base with soft feature-colour blobs / cards, drawn
        # right after the clear so the tomogram and annotations render on top.
        # blobs = [(x, y, radius, rgb, angle, alpha)]
        if self.background_blob_shader is None or not blobs:
            return
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glDisable(GL_BLEND)
        self.background_blob_shader.bind()
        self.background_blob_shader.uniform2f("uRes", res)
        self.background_blob_shader.uniform3f("uBase", base_col)
        self.background_blob_shader.uniform1f("uIntensity", intensity)
        self.background_blob_shader.uniform1i("uShape", shape)
        n = min(480, len(blobs))
        self.background_blob_shader.uniform1i("uN", n)
        for i in range(n):
            bx, by, br, bc, bang, balp = blobs[i]
            self.background_blob_shader.uniform4f(f"uA[{i}]", (bx, by, br, balp))
            self.background_blob_shader.uniform4f(f"uB[{i}]", (bc[0], bc[1], bc[2], bang))
        self.ndc_screen_va.bind()
        glDrawElements(GL_TRIANGLES, self.ndc_screen_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.ndc_screen_va.unbind()
        self.background_blob_shader.unbind()
        glEnable(GL_BLEND)
        glDepthMask(GL_TRUE)
        # leave depth test DISABLED: the 2D frame/brush/box pass draws in
        # painter's order and must not depth-cull against the tomogram.
        glDisable(GL_DEPTH_TEST)

    def render_overlay(self, se_frame, camera, blend_mode, alpha):
        if se_frame.overlay is None:
            return
        if alpha == 0.0:
            return
        glBlendFunc(blend_mode[0], blend_mode[1])
        glBlendEquation(blend_mode[2])
        glDisable(GL_DEPTH_TEST)
        shader_blend_code = blend_mode[3]
        se_frame.quad_va.bind()
        se_frame.overlay.texture.bind()
        self.overlay_shader.bind()
        if isinstance(camera, Camera3D):
            self.overlay_shader.uniform1i("render_3d", 1)
            self.overlay_shader.uniform1f("intensity", SegmentationEditor.OVERLAY_INTENSITY)
            self.overlay_shader.uniform1f("z_pos", (se_frame.current_slice - se_frame.n_slices / 2) * se_frame.pixel_size)
            self.overlay_shader.uniform1f("pixel_size", se_frame.pixel_size)  # This factor should be in the VA, but fixing that messed up the 2D view - cheap fix for now
        else:
            self.overlay_shader.uniform1i("render_3d", 0)
            self.overlay_shader.uniform1f("intensity", 1.0)
        self.overlay_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.overlay_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        self.overlay_shader.uniform1f("alpha", alpha)
        self.overlay_shader.uniform1i("shader_blend_code", shader_blend_code)
        glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.overlay_shader.unbind()
        se_frame.quad_va.unbind()
        glActiveTexture(GL_TEXTURE0)


class Camera:
    def __init__(self):
        self.view_matrix = np.identity(4)
        self.projection_matrix = np.identity(4)
        self.view_projection_matrix = np.identity(4)
        self.position = np.zeros(3)
        self.zoom = 1.0
        self.projection_width = 1
        self.projection_height = 1
        self.set_projection_matrix(cfg.window_width, cfg.window_height)

    def cursor_to_world_position(self, cursor_pos):
        """
        cursor_pus: list, [x, y]
        returns: [world_pos_x, world_pos_y]
        Converts an input cursor position to corresponding world position. Assuming orthographic projection matrix.
        """
        inverse_matrix = np.linalg.inv(self.view_projection_matrix)
        window_coordinates = (2 * cursor_pos[0] / cfg.window_width - 1, 1 - 2 * cursor_pos[1] / cfg.window_height)
        window_vec = np.matrix([*window_coordinates, 1.0, 1.0]).T
        world_vec = (inverse_matrix * window_vec)
        return [float(world_vec[0]), float(world_vec[1])]

    def world_to_screen_position(self, world_position):
        vec = np.matrix([world_position[0], world_position[1], 0.0, 1.0]).T
        vec_out = self.view_projection_matrix * vec
        screen_x = int((1 + float(vec_out[0])) * self.projection_width / 2.0)
        screen_y = int((1 - float(vec_out[1])) * self.projection_height / 2.0)
        return [screen_x, screen_y]

    def set_projection_matrix(self, window_width, window_height):
        self.projection_matrix = np.matrix([
            [2 / window_width, 0, 0, 0],
            [0, 2 / window_height, 0, 0],
            [0, 0, -2 / 100, 0],
            [0, 0, 0, 1],
        ])
        self.projection_width = window_width
        self.projection_height = window_height

    def on_update(self):
        self.view_matrix = np.matrix([
            [self.zoom, 0.0, 0.0, self.position[0] * self.zoom],
            [0.0, self.zoom, 0.0, self.position[1] * self.zoom],
            [0.0, 0.0, 1.0, self.position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.view_projection_matrix = np.matmul(self.projection_matrix, self.view_matrix)


class Light3D:
    def __init__(self):
        self.colour = (1.0, 1.0, 1.0)
        self.vec = (0.0, 1.0, 0.0)
        self.yaw = 20.0
        self.pitch = 0.0
        self.strength = 0.5

    def compute_vec(self, dyaw=0, dpitch=0):
        # Calculate the camera forward vector based on pitch and yaw
        cos_pitch = np.cos(np.radians(self.pitch + dpitch))
        sin_pitch = np.sin(np.radians(self.pitch + dpitch))
        cos_yaw = np.cos(np.radians(self.yaw + dyaw))
        sin_yaw = np.sin(np.radians(self.yaw + dyaw))

        forward = np.array([-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw])
        self.vec = forward


class Camera3D:
    def __init__(self):
        self.view_matrix = np.eye(4)
        self.projection_matrix = np.eye(4)
        self.view_projection_matrix = np.eye(4)
        self.focus = np.zeros(3)
        self.rotation = np.array([1.0, 0.0, 0.0, 0.0])  # orientation quaternion (w, x, y, z); identity == the old yaw=180 / pitch=0 view
        self.distance = 1120.0
        self.clip_near = 1e-1
        self.clip_far = 1e4
        self.projection_width = 1
        self.projection_height = 1
        self.set_projection_matrix(cfg.window_width, cfg.window_height)

    def set_projection_matrix(self, window_width, window_height):
        self.projection_width = window_width
        self.projection_height = window_height
        self.update_projection_matrix()

    def cursor_delta_to_world_delta(self, cursor_delta):
        return cursor_delta[0] * self.get_right() + cursor_delta[1] * self.get_up()

    # --- quaternion helpers (w, x, y, z) ---
    @staticmethod
    def _quat_mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ])

    @staticmethod
    def _quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _quat_normalize(q):
        n = np.linalg.norm(q)
        return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def _quat_from_axis_angle(axis, angle):
        axis = np.asarray(axis, dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-9 or angle == 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = axis / n
        s = np.sin(angle * 0.5)
        return np.array([np.cos(angle * 0.5), axis[0] * s, axis[1] * s, axis[2] * s])

    @staticmethod
    def _quat_to_matrix(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ])

    @staticmethod
    def _matrix_to_quat(m):
        t = m[0, 0] + m[1, 1] + m[2, 2]
        if t > 0:
            s = 0.5 / np.sqrt(t + 1.0)
            w, x, y, z = 0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
        return Camera3D._quat_normalize(np.array([w, x, y, z]))

    # --- orientation accessors (columns of the camera->world rotation matrix) ---
    def rotation_matrix(self):
        return Camera3D._quat_to_matrix(self.rotation)

    def get_forward(self):
        return self.rotation_matrix() @ np.array([0.0, 0.0, -1.0])  # camera looks along its own -Z

    def get_right(self):
        return self.rotation_matrix() @ np.array([1.0, 0.0, 0.0])

    def get_up(self):
        return self.rotation_matrix() @ np.array([0.0, 1.0, 0.0])

    @property
    def yaw(self):
        # azimuth in the old convention (180 at the start view), kept for the light coupling
        f = self.get_forward()
        return np.degrees(np.arctan2(-f[0], f[2]))

    @property
    def pitch(self):
        f = self.get_forward()
        return np.degrees(np.arcsin(np.clip(-f[1], -1.0, 1.0)))

    def set_euler(self, yaw, pitch):
        # Set an absolute yaw/pitch view (roll reset to 0); used by the Settings inputs.
        yaw_r, pitch_r = np.radians(yaw), np.radians(pitch)
        forward = Camera3D.normalize(np.array([-np.cos(pitch_r) * np.sin(yaw_r), -np.sin(pitch_r), np.cos(pitch_r) * np.cos(yaw_r)]))
        right = Camera3D.normalize(np.cross(forward, np.array([0.0, 1.0, 0.0])))
        up = np.cross(right, forward)
        self.rotation = Camera3D._matrix_to_quat(np.column_stack([right, up, -forward]))

    def rotate_world(self, axis, angle_deg):
        # Rotate the camera about a world-space axis (left-multiply applies it in world frame).
        q = Camera3D._quat_from_axis_angle(axis, np.radians(angle_deg))
        self.rotation = Camera3D._quat_normalize(Camera3D._quat_mul(q, self.rotation))

    def arcball_rotate(self, prev_cursor, cur_cursor, width, height, speed_deg):
        # ChimeraX-style tumble driven by the per-frame cursor motion (delta-based, so it never runs
        # out of travel the way absolute-position arcball does). Near the centre the motion turns the
        # scene about in-plane axes; near the edges it rolls about the view axis. Axes are in camera
        # space (right-multiply); conj makes the scene follow the cursor (drop it to invert).
        mx = cur_cursor[0] - prev_cursor[0]
        my = -(cur_cursor[1] - prev_cursor[1])  # screen y grows downward; flip to y-up
        if mx == 0 and my == 0:
            return
        speed = np.radians(speed_deg)
        r = 0.5 * min(width, height)
        px = (cur_cursor[0] - width * 0.5) / r
        py = (height * 0.5 - cur_cursor[1]) / r
        edge = min(1.0, px * px + py * py)  # 0 at the centre -> 1 at the edges
        q_inplane = Camera3D._quat_from_axis_angle([-my, mx, 0.0], np.hypot(mx, my) * speed * (1.0 - edge))
        q_roll = Camera3D._quat_from_axis_angle([0.0, 0.0, 1.0], (mx * -py + my * px) * speed * edge)
        q_delta = Camera3D._quat_mul(q_roll, q_inplane)
        self.rotation = Camera3D._quat_normalize(Camera3D._quat_mul(self.rotation, Camera3D._quat_conj(q_delta)))

    @property
    def matrix(self):
        return self.view_projection_matrix

    @property
    def vpmat(self):
        return self.view_projection_matrix

    @property
    def ivpmat(self):
        return np.linalg.inv(self.view_projection_matrix)

    @property
    def pmat(self):
        return self.projection_matrix

    @property
    def vmat(self):
        return self.view_matrix

    @property
    def ipmat(self):
        return np.linalg.inv(self.projection_matrix)

    @property
    def ivmat(self):
        return np.linalg.inv(self.view_matrix)

    def on_update(self):
        self.update_projection_matrix()
        self.update_view_projection_matrix()

    def update_projection_matrix(self):
        aspect_ratio = self.projection_width / self.projection_height
        self.projection_matrix = Camera3D.create_perspective_matrix(60.0, aspect_ratio, self.clip_near, self.clip_far)
        self.update_view_projection_matrix()

    @staticmethod
    def create_perspective_matrix(fov, aspect_ratio, near, far):
        S = 1 / (np.tan(0.5 * fov / 180.0 * np.pi))
        f = far
        n = near

        projection_matrix = np.zeros((4, 4))
        projection_matrix[0, 0] = S / aspect_ratio
        projection_matrix[1, 1] = S
        projection_matrix[2, 2] = -f / (f - n)
        projection_matrix[3, 2] = -1
        projection_matrix[2, 3] = -f * n / (f - n)

        return projection_matrix

    def update_view_projection_matrix(self):
        rot = self.rotation_matrix()
        right, up, back = rot[:, 0], rot[:, 1], rot[:, 2]  # camera +X, +Y, +Z in world (back == -forward)
        eye = self.focus + back * self.distance
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = right, up, back
        view[0, 3], view[1, 3], view[2, 3] = -np.dot(right, eye), -np.dot(up, eye), -np.dot(back, eye)
        self.view_matrix = view
        self.view_projection_matrix = self.projection_matrix @ self.view_matrix

    def get_view_direction(self):
        return -self.get_forward()  # from the focus toward the eye

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm


class QueuedExport:
    def __init__(self, directory, dataset, models, batch_size, export_overlay):
        self.title = dataset.title
        self.tag = dataset.path
        self.directory = directory
        self.dataset = dataset
        self.models = models
        self.export_overlay = export_overlay
        self.process = BackgroundProcess(self.do_export, (), name=f"{self.dataset.title} export")
        if self.models != []:
            self.colour = self.models[0].colour
        else:
            self.colour = (0.0, 0.0, 0.0)
        self.batch_size = batch_size

    def do_export(self, process):
        try:
            process.set_progress(0.0001)
            start_time = time.time()
            print(f"QueuedExport - loading dataset {self.dataset.path}")
            rx, ry = self.dataset.get_roi_indices()
            mrcd = np.array(mrcfile.open(self.dataset.path, mode='r', permissive=True).data[:, :, :])
            target_type_dict = {np.float32: np.dtype('float32'), float: np.dtype('float32'), np.dtype('int8'): np.dtype('uint8'), np.dtype('int16'): np.dtype('float32')}
            if mrcd.dtype not in target_type_dict:
                mrcd = mrcd.astype(float, copy=False)
            else:
                mrcd = np.array(mrcd.astype(target_type_dict[mrcd.dtype], copy=False), dtype=np.dtype('float32'))
            self.check_stop_request()
            n_slices = mrcd.shape[0]
            n_slices_total = (self.dataset.export_top - self.dataset.export_bottom) * max(1, len(self.models))
            n_slices_complete = 0
            segmentations = np.zeros((len(self.models), *mrcd.shape), dtype=np.uint8)
            m_idx = 0
            for m in self.models:
                print(f"QueuedExport - applying model {m.title} ({m.info})")
                self.colour = m.colour
                for j in range(self.dataset.export_bottom, self.dataset.export_top):
                    self.check_stop_request()
                    j_indices = np.clip(np.arange(j - m.model_depth // 2, j + m.model_depth // 2 + 1), 0, n_slices - 1)
                    segmented_slice = m.apply_to_slice(mrcd[j_indices, rx[0]:rx[1], ry[0]:ry[1]], self.dataset.pixel_size) * 255
                    segmentations[m_idx, j, rx[0]:rx[1], ry[0]:ry[1]] = segmented_slice
                    n_slices_complete += 1
                    self.process.set_progress(min([0.999, n_slices_complete / n_slices_total]))
                m_idx += 1

            # apply competition
            print(f"QueuedExport - model competition")
            emission_indices = list()
            absorption_indices = list()
            i = 0
            for m in self.models:
                if m.emit:
                    emission_indices.append(i)
                if m.absorb:
                    absorption_indices.append(i)
                i += 1
            if len(emission_indices) >= 1 and len(absorption_indices) >= 1:
                max_map = np.max(segmentations[emission_indices, :, :, :], axis=0)
                for i in absorption_indices:
                    self.check_stop_request()
                    segmentations[i, :, :, :][segmentations[i, :, :, :] < max_map] = 0

            # apply interactions
            print(f"QueuedExport - model interactions")
            for interaction in ModelInteraction.all:
                target_model = interaction.parent
                source_model = interaction.partner
                if target_model in self.models and source_model in self.models:
                    target_idx = self.models.index(target_model)
                    source_idx = self.models.index(source_model)
                    for i in range(self.dataset.export_bottom, self.dataset.export_top):
                        self.check_stop_request()
                        segmentations[target_idx, i, :, :] = interaction.apply_to_images(self.dataset.pixel_size, segmentations[source_idx, i, :, :], segmentations[target_idx, i, :, :])

            # save the mrc files
            i = 0
            self.colour = cfg.COLOUR_POSITIVE[0:3]
            for m in self.models:
                self.check_stop_request()
                print(f"QueuedExport - saving output of model {m.info}")
                out_path = os.path.join(self.directory, self.dataset.title+"__"+m.title+".mrc")
                with mrcfile.new(out_path, overwrite=True) as mrc:
                    s = segmentations[i, :, :, :].squeeze()
                    mrc.set_data(s)
                    mrc.voxel_size = self.dataset.pixel_size * 10.0
                n_slices_complete += n_slices
                i += 1

            # export overlay
            self.check_stop_request()
            if self.export_overlay and self.dataset.overlay is not None:
                print(f"QueuedExport - exporting overlay")
                overlay_grayscale = np.sum(self.dataset.overlay.pxd ** 2, axis=2)
                overlay_grayscale /= np.amax(overlay_grayscale)
                overlay_grayscale *= 255
                overlay_grayscale -= 128
                overlay_grayscale = overlay_grayscale.astype(np.int8)
                overlay_volume = np.tile(overlay_grayscale[np.newaxis, :, :], (n_slices, 1, 1))
                out_path = os.path.join(self.directory, self.dataset.title + "_overlay.mrc")
                with mrcfile.new(out_path, overwrite=True) as mrc:
                    mrc.set_data(overlay_volume)
                    mrc.voxel_size = self.dataset.pixel_size * 10.0

            self.process.set_progress(1.0)
            print(f"QueuedExport - done! ({time.time() - start_time:.2f} s.)\n")

        except Exception as e:
            if not "terminated by user" in str(e):
                cfg.set_error(e, "")
            print("An issue was encountered during export\n"
                  f"\tDataset: {self.dataset.path}\n"
                  f"\tError: ", e)
            self.process.set_progress(1.0)

    def check_stop_request(self):
        if self.process.stop_request.is_set():
            raise Exception("QueuedExport - process terminated by user. ")

    def start(self):
        self.process.set_progress(0.0001)
        self.process.start()

    def stop(self):
        if self.process.thread is not None:
            self.process.stop_request.set()


class QueuedMeshExtract:
    def __init__(self, mrcpath, threshold, min_size, save_dir, binning=1, pixel_size=1.0):
        self.title = mrcpath
        self.colour = (0.2, 0.9, 0.2)
        self.min_size = min_size
        self.pixel_size = pixel_size
        self.tag = mrcpath
        self.path = mrcpath
        self.threshold = threshold
        self.dir = save_dir
        self.binning = binning
        self.process = BackgroundProcess(self.do_export, (), name=f"{self.path} export mesh.")

    def do_export(self, process):
        try:
            surface_model = SurfaceModel(self.path, self.pixel_size, no_gpu=True)
            surface_model.level = self.threshold
            surface_model.dust = self.min_size
            surface_model.bin = self.binning
            surface_model._generate_model(process)
            surface_model.initialized = True
            self.check_stop_request()
            path = os.path.join(self.dir, os.path.splitext(os.path.basename(self.path))[0])+".obj"
            surface_model.save_as_obj(path=path)
            print(f"saved: {path}")
        except Exception as e:
            cfg.set_error(e, "Error in QueuedMeshExtract (exporting a segmentation as a 3D mesh) - see details below.")
        self.process.set_progress(1.0)

    def check_stop_request(self):
        if self.process.stop_request.is_set():
            raise Exception("QueuedMeshExtract - process terminated by user.")

    def start(self):
        self.process.set_progress(0.0001)
        self.process.start()

    def stop(self):
        if self.process.thread is not None:
            self.process.stop_request.set()


class QueuedExtract:
    def __init__(self, mrcpath, threshold, min_size, min_spacing, save_dir, binning=1, star_format=True):
        self.title = mrcpath
        self.tag = mrcpath
        self.path = mrcpath
        self.threshold = threshold
        self.min_size = min_size
        self.min_spacing = min_spacing
        self.binning = binning
        self.dir = save_dir
        self.colour = (0.2, 0.9, 0.2)
        self.star_format = star_format
        self.process = BackgroundProcess(self.do_export, (), name=f"{self.path} find coords.")

    def do_export(self, process):
        try:
            out_path = os.path.join(self.dir, os.path.splitext(os.path.basename(self.path))[0]+"_coords.star")
            pick_particles(mrcpath=self.path, threshold=self.threshold, min_spacing=self.min_spacing, min_size=self.min_size, out_path=out_path, process=self.process, binning=self.binning)
            for s in cfg.se_surface_models:
                if s.path == self.path:
                    s.find_coordinates()
        except Exception as e:
            cfg.set_error(e, "Error in QueuedExtract - see details below.")
        process.set_progress(1.0)

    def check_stop_request(self):
        if self.process.stop_request.is_set():
            raise Exception("QueuedExport - process terminated by user. ")

    def start(self):
        self.process.set_progress(0.0001)
        self.process.start()

    def stop(self):
        if self.process.thread is not None:
            self.process.stop_request.set()


class WorldSpaceIcon:
    # this class is only used to render and interact with the SegmentationEditor crop handles. fairly specific implementation
    crop_icon_va = None
    crop_icon_texture = None
    MIN_SIZE = 128

    def __init__(self, corner_idx=0):
        if WorldSpaceIcon.crop_icon_va is None:
            WorldSpaceIcon.init_opengl_objs()

        self.transform = Transform()
        self.corner_idx = corner_idx
        self.corner_positions_local = [[0, 0], [1, 0], [0, -1], [1, -1]]
        self.corner_positions = [[0, 0], [1, 0], [0, -1], [1, -1]]
        self.parent = None
        self.active = False
        self.moving_entire_roi = False

    def get_va(self):
        return WorldSpaceIcon.crop_icon_va

    def get_texture(self):
        return WorldSpaceIcon.crop_icon_texture

    @staticmethod
    def init_opengl_objs():
        # crop icon
        icon_vertices = [0, 0, 0, 0,
                         1, 0, 1, 0,
                         0, -1, 0, 1,
                         1, -1, 1, 1]
        icon_indices = [0, 1, 2, 2, 1, 3]
        WorldSpaceIcon.crop_icon_va = VertexArray(VertexBuffer(icon_vertices), IndexBuffer(icon_indices),
                                                  attribute_format="xyuv")
        WorldSpaceIcon.crop_icon_texture = Texture(format="rgba32f")
        pxd_icon_crop = np.asarray(Image.open(os.path.join(cfg.root, "icons", "icon_crop_256.png"))).astype(
            np.float32) / 255.0
        WorldSpaceIcon.crop_icon_texture.update(pxd_icon_crop)

    def affect_crop(self, pixel_coordinate):
        d = WorldSpaceIcon.MIN_SIZE
        ## this method is called when the crop handle is moved, onto pixel_coordinate
        if self.corner_idx == 0:
            self.parent.crop_roi[0] = pixel_coordinate[0]
            self.parent.crop_roi[0] = clamp(self.parent.crop_roi[0], 0, self.parent.crop_roi[2] - d)
            self.parent.crop_roi[1] = (self.parent.height - pixel_coordinate[1])
            self.parent.crop_roi[1] = clamp(self.parent.crop_roi[1], 0, self.parent.crop_roi[3] - d)
        elif self.corner_idx == 1:
            self.parent.crop_roi[2] = self.parent.width - (self.parent.width - pixel_coordinate[0])
            self.parent.crop_roi[2] = clamp(self.parent.crop_roi[2], self.parent.crop_roi[0] + d, self.parent.width)
            self.parent.crop_roi[1] = (self.parent.height - pixel_coordinate[1])
            self.parent.crop_roi[1] = clamp(self.parent.crop_roi[1], 0, self.parent.crop_roi[3] - d)
        elif self.corner_idx == 2:
            self.parent.crop_roi[2] = self.parent.width - (self.parent.width - pixel_coordinate[0])
            self.parent.crop_roi[2] = clamp(self.parent.crop_roi[2], self.parent.crop_roi[0] + d, self.parent.width)
            self.parent.crop_roi[3] = (self.parent.height - pixel_coordinate[1])
            self.parent.crop_roi[3] = clamp(self.parent.crop_roi[3], self.parent.crop_roi[1] + d, self.parent.height)
        elif self.corner_idx == 3:
            self.parent.crop_roi[0] = pixel_coordinate[0]
            self.parent.crop_roi[0] = clamp(self.parent.crop_roi[0], 0, self.parent.crop_roi[2] - d)
            self.parent.crop_roi[3] = (self.parent.height - pixel_coordinate[1])
            self.parent.crop_roi[3] = clamp(self.parent.crop_roi[3], self.parent.crop_roi[1] + d, self.parent.height)

    def move_crop_roi(self, dx, dy):
        test_roi = copy(self.parent.crop_roi)
        test_roi[0] += dx
        test_roi[2] += dx
        test_roi[1] += dy
        test_roi[3] += dy

        if test_roi[0] > 0 and test_roi[2] < self.parent.width:
            self.parent.crop_roi[0] = test_roi[0]
            self.parent.crop_roi[2] = test_roi[2]
            self.parent.crop_roi[0] = clamp(self.parent.crop_roi[0], 0, self.parent.width)
            self.parent.crop_roi[2] = clamp(self.parent.crop_roi[2], 0, self.parent.width)
        if test_roi[1] > 0 and test_roi[3] < self.parent.height:
            self.parent.crop_roi[1] = test_roi[1]
            self.parent.crop_roi[3] = test_roi[3]
            self.parent.crop_roi[1] = clamp(self.parent.crop_roi[1], 0, self.parent.height)
            self.parent.crop_roi[3] = clamp(self.parent.crop_roi[3], 0, self.parent.height)

    def convert_crop_roi_to_integers(self):
        for i in range(4):
            self.parent.crop_roi[i] = int(self.parent.crop_roi[i])

    def compute_matrix(self, se_frame, camera):
        self.parent = se_frame
        fpos = se_frame.transform.translation
        fx = [-1, 1, 1, -1]
        fy = [1, 1, -1, -1]
        # find position of the corner of the frame that this handle belongs to
        dx = 0
        dy = 0
        if self.corner_idx == 0:
            dx = se_frame.crop_roi[0]
            dy = -se_frame.crop_roi[1]
        elif self.corner_idx == 1:
            dx = -(se_frame.width - se_frame.crop_roi[2])
            dy = -se_frame.crop_roi[1]
        elif self.corner_idx == 2:
            dx = -(se_frame.width - se_frame.crop_roi[2])
            dy = se_frame.height - se_frame.crop_roi[3]
        elif self.corner_idx == 3:
            dx = se_frame.crop_roi[0]
            dy = se_frame.height - se_frame.crop_roi[3]
        self.transform.translation[0] = fpos[0] + (0.5 * fx[self.corner_idx] * se_frame.width + dx) * se_frame.pixel_size
        self.transform.translation[1] = fpos[1] + (0.5 * fy[self.corner_idx] * se_frame.height + dy) * se_frame.pixel_size
        self.transform.rotation = -self.corner_idx * 90.0
        self.transform.scale = 15.0 / camera.zoom
        self.transform.compute_matrix()

        # set icon's corner positions
        for i in range(4):
            local_corner_pos = tuple(self.corner_positions_local[i])
            vec = np.matrix([*local_corner_pos, 0.0, 1.0]).T
            world_corner_pos = self.transform.matrix * vec
            self.corner_positions[i] = [float(world_corner_pos[0]), float(world_corner_pos[1])]

    def is_hovered(self, camera, cursor_position):
        P = camera.cursor_to_world_position(cursor_position)
        A = self.corner_positions[0]
        B = self.corner_positions[1]
        D = self.corner_positions[3]
        ap = [P[0] - A[0], P[1] - A[1]]
        ab = [B[0] - A[0], B[1] - A[1]]
        ad = [D[0] - A[0], D[1] - A[1]]
        return (0 < ap[0] * ab[0] + ap[1] * ab[1] < ab[0] ** 2 + ab[1] ** 2) and (0 < ap[0] * ad[0] + ap[1] * ad[1] < ad[0] ** 2 + ad[1] ** 2)

