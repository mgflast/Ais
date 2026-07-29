import traceback
import dill as pickle
import os
from datetime import datetime
import sys
import platform
import shutil
import json

frozen = False
glfw_initialized = False


root = os.path.dirname(os.path.dirname(__file__))
app_name = "Ais"
version = "1.2.4"
license = "GNU GPL v3"
log_path = os.path.join(os.path.expanduser("~"), ".Ais", "Ais.log")
settings_path = os.path.join(os.path.expanduser("~"), ".Ais", "settings.txt")
feature_lib_path = os.path.join(os.path.expanduser("~"), ".Ais", "feature_library.txt")
os.makedirs(os.path.dirname(log_path), exist_ok=True)


filetype_segmentation = ".scns"
filetype_traindata = ".scnt"
filetype_semodel = ".scnm"
filetype_semodel_group = ".scnmgroup"


window_width = 1100
window_height = 700
cursor_pos = [0, 0]

error_msg = None
error_new = True
error_obj = None
error_logged = False
error_window_active = False

segmentation_editor = None
se_enabled = True

se_frames = list()
se_active_frame = None
se_models = list()
se_active_model = None
se_path = "..."
se_surface_models = list()

se_model_handle_overlap_mode = 0

controls_info_text = \
    "left mouse:        draw\n" \
    "    +shift:        place box\n" \
    "    +ctrl+shift:   extract template\n" \
    "right mouse:       erase\n" \
    "    +shift:        erase box\n" \
    "scroll:            change slice\n" \
    "    +shift:        zoom\n" \
    "    + ctrl:        change brush size\n" \
    "spacebar:          reset view\n" \
    "key A:             toggle autocontrast\n" \
    "key I:             toggle inversion\n" \
    "    +shift:        toggle interpolation\n" \
    "key C:             toggle cropping\n" \
    "key F:             toggle flood drawing mode\n" \
    "key Q:             hide 3d models\n" \
    "key O:             toggle overlay visibility\n" \
    "key W:             select previous feature\n" \
    "key S:             select next feature\n" \
    "key left:          previous slice\n" \
    "key right:         next slice\n" \
    "key up:            previous dataset\n" \
    "key down:          next dataset\n" \
    "key 1:             select 'Annotation' tab\n" \
    "key 2:             select 'Models' tab\n" \
    "key 3:             select 'Export' tab\n" \
    "key 4:             select 'Render' tab\n" \



def set_error(error_object, error_message):
    global error_msg, error_obj, error_new, error_logged
    error_msg = error_message + "\n\n"
    error_msg += "".join(traceback.TracebackException.from_exception(error_object).format())
    print(error_msg)
    error_obj = error_object
    error_new = True
    error_logged = False


def write_to_log(text):
    with open(log_path, "a") as f:
        f.write("\n\n ____________________ \n\n")
        f.write(text)


def start_log():
    with open(log_path, "w") as f:
        f.write(app_name+" version "+version+" "+license+"\n"+datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+"\n")
        f.write(f"OS: {platform.platform()}\n")
        f.write(f"Python version: {sys.version}")


def parse_settings():
    # If settings file not found, copy the one from core to the right location.
    if not os.path.exists(settings_path):
        shutil.copy(os.path.join(root, "core", "settings.txt"), settings_path)
    if not os.path.exists(os.path.join(os.path.dirname(settings_path), "models")):
        os.mkdir(os.path.join(os.path.dirname(settings_path), "models"))

    try:
        with open(settings_path, 'r') as f:
            sdict = json.load(f)
    except Exception as e:
        shutil.copy(os.path.join(root, "core", "settings.txt"), settings_path)
        parse_settings()
        return

    # Read settings - if any parameters are missing, insert them.
    with open(os.path.join(root, "core", "settings.txt"), 'r') as f:
        default_settings = json.load(f)

    for key in default_settings:
        if key not in sdict:
            sdict[key] = default_settings[key]

    with open(settings_path, 'w') as f:
        json.dump(sdict, f, indent=2)

    return sdict


settings = parse_settings()


def edit_setting(key, value):
    global settings
    settings[key] = value
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)
    print(key, value)


def push_recent(key, path, cap=12):
    # Move `path` to the front of the recent-files list stored under `key`.
    path = os.path.abspath(path)
    current = settings.get(key) or []
    updated = [p for p in current if os.path.normpath(p) != os.path.normpath(path)]
    updated.insert(0, path)
    edit_setting(key, updated[:cap])


def _norm_path(path):
    # Case- and separator-insensitive absolute path for identity comparison.
    # normcase folds case (and slashes) on Windows, which normpath alone does
    # not - without it a .mrc and its .scns twin whose stored paths differ only
    # in casing (e.g. drive Z: vs z:, or Pom-substituted subdirs) look distinct.
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _dataset_stem(path):
    # Dataset identity independent of its extension: a .mrc and its .scns twin
    # (same directory + basename) share one identity.
    stem, _ = os.path.splitext(_norm_path(path))
    return stem


def dedup_recent_datasets(paths):
    # Drop any .mrc whose same-named .scns is also present: a .scns supersedes
    # its .mrc twin. Order is preserved.
    scns_stems = {_dataset_stem(p) for p in paths
                  if os.path.splitext(p)[1].lower() == filetype_segmentation}
    return [p for p in paths
            if not (os.path.splitext(p)[1].lower() == ".mrc" and _dataset_stem(p) in scns_stems)]


def push_recent_dataset(path, cap=12):
    # Move a dataset (.mrc or .scns) to the front of RECENT_DATASETS, keeping at
    # most one entry per dataset. A .scns supersedes its same-named .mrc: saving
    # or opening a .scns evicts the .mrc twin, and opening a .mrc whose .scns is
    # already remembered promotes the .scns instead.
    key = "RECENT_DATASETS"
    path = os.path.abspath(path)
    stem, ext = os.path.splitext(path)
    current = settings.get(key) or []
    if ext.lower() == ".mrc":
        twin = stem + filetype_segmentation
        if any(_norm_path(p) == _norm_path(twin) for p in current):
            path = os.path.abspath(twin)
    updated = [p for p in current if _dataset_stem(p) != _dataset_stem(path)]
    updated.insert(0, path)
    edit_setting(key, dedup_recent_datasets(updated)[:cap])


if settings["POM_COMMAND_DIR"] == "":
    edit_setting("POM_COMMAND_DIR", os.path.join(os.path.expanduser("~"), ".Ais"))

class FeatureLibraryFeature:
    DEFAULT_COLOURS = [(66 / 255, 214 / 255, 164 / 255),
                       (255 / 255, 243 / 255, 0 / 255),
                       (255 / 255, 104 / 255, 0 / 255),
                       (255 / 255, 13 / 255, 0 / 255),
                       (174 / 255, 0 / 255, 255 / 255),
                       (21 / 255, 0 / 255, 255 / 255),
                       (0 / 255, 136 / 255, 266 / 255),
                       (0 / 255, 247 / 255, 255 / 255),
                       (0 / 255, 255 / 255, 0 / 255)]
    CLR_COUNTER = 0
    SORT_TITLE = "\n"

    def __init__(self):
        self.title = "New feature"
        self.colour = FeatureLibraryFeature.DEFAULT_COLOURS[FeatureLibraryFeature.CLR_COUNTER % len(FeatureLibraryFeature.DEFAULT_COLOURS)]
        self.box_size = 128
        self.brush_size = 10.0 # nm
        self.alpha = 1.0
        self.use = True
        self.dust = 1.0
        self.level = 128
        self.render_alpha = 1.0
        self.hide = False
        FeatureLibraryFeature.CLR_COUNTER += 1

    def to_dict(self):
        return vars(self)

    @staticmethod
    def from_dict(data):
        ret = FeatureLibraryFeature()
        ret.title = data['title']
        ret.colour = data['colour']
        ret.box_size = data['box_size']
        ret.brush_size = data['brush_size']
        ret.alpha = data['alpha']
        ret.use = data['use']
        ret.dust = data['dust']
        ret.level = data['level']
        ret.render_alpha = data['render_alpha']
        ret.hide = data['hide']
        return ret


DEFAULT_FEATURE_LIBRARY_NAME = "Default library"


def parse_feature_library():
    # returns (libraries, active_name): a dict mapping library name -> list of FeatureLibraryFeature, and the name of the active library.
    if not os.path.exists(feature_lib_path):
        shutil.copy(os.path.join(root, "core", "feature_library.txt"), feature_lib_path)

    with open(feature_lib_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):  # pre-1.2 format: a single, unnamed, feature collection.
        data = {"active": DEFAULT_FEATURE_LIBRARY_NAME, "libraries": {DEFAULT_FEATURE_LIBRARY_NAME: data}}

    libraries = dict()
    for name, flist in data["libraries"].items():
        libraries[name] = [FeatureLibraryFeature.from_dict(f) for f in flist]
    if len(libraries) == 0:
        libraries[DEFAULT_FEATURE_LIBRARY_NAME] = list()
    active = data.get("active")
    if active not in libraries:
        active = list(libraries.keys())[0]
    return libraries, active


feature_libraries, active_feature_library = parse_feature_library()
feature_library = feature_libraries[active_feature_library]  # the active library; always one of the lists in feature_libraries.
feature_library_session = dict()


def save_feature_library():
    data = {"active": active_feature_library,
            "libraries": {name: [f.to_dict() for f in flist] for name, flist in feature_libraries.items()}}
    with open(feature_lib_path, 'w') as f:
        f.write(json.dumps(data, indent=2))


def reload_feature_libraries():
    global feature_libraries, active_feature_library, feature_library
    feature_libraries, active_feature_library = parse_feature_library()
    feature_library = feature_libraries[active_feature_library]


def set_active_feature_library(name):
    global active_feature_library, feature_library
    if name in feature_libraries:
        active_feature_library = name
        feature_library = feature_libraries[name]


def add_feature_library():
    base = "New library"
    name = base
    i = 2
    while name in feature_libraries:
        name = f"{base} {i}"
        i += 1
    feature_libraries[name] = list()
    set_active_feature_library(name)
    return name


def rename_feature_library(old_name, new_name):
    global feature_libraries, active_feature_library
    new_name = new_name.strip()
    if old_name not in feature_libraries or new_name == "" or new_name == old_name:
        return False
    if new_name in feature_libraries:
        return False
    feature_libraries = {new_name if k == old_name else k: v for k, v in feature_libraries.items()}
    if active_feature_library == old_name:
        active_feature_library = new_name
    return True


def delete_feature_library(name):
    if name not in feature_libraries or len(feature_libraries) <= 1:
        return
    feature_libraries.pop(name)
    if active_feature_library == name:
        set_active_feature_library(list(feature_libraries.keys())[0])



def apply_feature_library():
    flib_dict = dict()
    for f in feature_library_session.values():
        flib_dict[f.title] = f
    for f in feature_library:  # preference given to the static feature library
        flib_dict[f.title] = f

    for s in se_frames:
        for f in s.features:
            if f.title in flib_dict:
                library_feature = flib_dict[f.title]
                f.set_box_size(library_feature.box_size)
                f.brush_size = library_feature.brush_size
                f.alpha = library_feature.alpha
                f.colour = library_feature.colour

    for f in se_surface_models:
        if s.title in flib_dict:
            library_feature = flib_dict[s.title]
            f.colour = library_feature.colour
            f.alpha = library_feature.render_alpha
            f.dust = library_feature.dust
            f.level = library_feature.level
            f.hide = library_feature.hide



def sort_frames_by_feature(title):
    global se_frames
    frame_has_feature = list()
    for frame in se_frames:
        has_feature = False
        for feature in frame.features:
            if feature.title == title:
                has_feature = True
                frame.features.remove(feature)
                frame.features.insert(0, feature)
        frame_has_feature.append(has_feature)

    sorted_se_frames = sorted(zip(se_frames, frame_has_feature), key=lambda x: x[1], reverse=True)
    se_frames = [frame for frame, _ in sorted_se_frames]


COLOUR_TEST_A = (1.0, 0.0, 1.0, 1.0)
COLOUR_TEST_B = (0.0, 1.0, 1.0, 1.0)
COLOUR_TEST_C = (1.0, 1.0, 0.0, 1.0)
COLOUR_TEST_D = (1.0, 1.0, 1.0, 1.0)

COLOUR_WINDOW_BACKGROUND = (0.94, 0.94, 0.94, 0.94)
COLOUR_PANEL_BACKGROUND = (0.94, 0.94, 0.94, 0.94)
COLOUR_TITLE_BACKGROUND = (0.87, 0.87, 0.83, 0.96)
COLOUR_TITLE_BACKGROUND_LIGHT = (0.96, 0.96, 0.93, 0.93)
COLOUR_FRAME_BACKGROUND = (0.87, 0.87, 0.83, 0.96)
COLOUR_FRAME_BACKGROUND_LIGHT = (0.92, 0.92, 0.90, 0.96)
COLOUR_FRAME_ACTIVE = (0.91, 0.91, 0.86, 0.94)
COLOUR_FRAME_DARK = (0.83, 0.83, 0.76, 0.94)
COLOUR_FRAME_EXTRA_DARK = (0.76, 0.76, 0.71, 0.94)
COLOUR_MAIN_MENU_BAR = (0.882, 0.882, 0.882, 0.94)
COLOUR_MAIN_MENU_BAR_TEXT = (0.0, 0.0, 0.0, 0.94)
COLOUR_MAIN_MENU_BAR_HILIGHT = (0.96, 0.95, 0.92, 0.94)
COLOUR_MENU_WINDOW_BACKGROUND = (0.96, 0.96, 0.96, 0.94)
COLOUR_DROP_TARGET = COLOUR_FRAME_DARK
COLOUR_HEADER = COLOUR_FRAME_DARK
COLOUR_HEADER_ACTIVE = COLOUR_FRAME_ACTIVE
COLOUR_HEADER_HOVERED = COLOUR_FRAME_EXTRA_DARK
COLOUR_TEXT = (0.0, 0.0, 0.0, 1.0)
COLOUR_CHECK_MARK = (0.0, 0.0, 0.0, 1.0)
COLOUR_TEXT_ACTIVE = (0.0, 0.0, 0.2, 1.0)
COLOUR_TEXT_DISABLED = (0.7, 0.7, 0.7, 1.0)
COLOUR_FRAME_DISABLED = (0.9, 0.9, 0.9, 1.0)
COLOUR_TEXT_FADE = COLOUR_FRAME_EXTRA_DARK
WINDOW_ROUNDING = 5.0
CONTEXT_MENU_SIZE = (200, 98)
ERROR_WINDOW_HEIGHT = 80
COLOUR_ERROR_WINDOW_BACKGROUND = (0.94, 0.94, 0.94, 0.94)
COLOUR_ERROR_WINDOW_HEADER = (0.87, 0.87, 0.83, 0.96)
COLOUR_ERROR_WINDOW_HEADER_NEW = (0.87, 0.87, 0.83, 0.96)
COLOUR_ERROR_WINDOW_TEXT = (0.0, 0.0, 0.0, 1.0)
COLOUR_CM_WINDOW_TEXT = (0.0, 0.0, 0.0, 1.0)
COLOUR_CM_OPTION_HOVERED = (1.0, 1.0, 1.0, 1.0)
COLOUR_TRANSPARENT = (1.0, 1.0, 1.0, 0.0)
COLOUR_FRAME_BACKGROUND_BLUE = (0.76, 0.76, 0.83, 1.0)
COLOUR_POSITIVE = (0.1, 0.8, 0.1, 1.0)
COLOUR_NEGATIVE = (0.8, 0.1, 0.1, 1.0)
COLOUR_NEUTRAL = (0.6, 0.6, 0.6, 1.0)
COLOUR_NEUTRAL_LIGHT = (0.8, 0.8, 0.8, 1.0)
COLOUR_HIGHLIGHT = (1.0, 1.0, 0.1, 1.0)
COLOUR_FLIB_HIGHLIGHT = (1.0, 0.9, 0.3, 0.3)
COLOUR_SESSION_FEATURE = (0.94, 0.94, 0.94, 0.94)
COLOUR_SESSION_FEATURE_BORDER = (0.0, 0.0, 0.0, 0.3)

# --- Theme (light / dark). set_theme() swaps every theme-dependent COLOUR_*.
_THEME_LIGHT = {
    "COLOUR_WINDOW_BACKGROUND": (0.94, 0.94, 0.94, 0.94),
    "COLOUR_PANEL_BACKGROUND": (0.94, 0.94, 0.94, 0.94),
    "COLOUR_TITLE_BACKGROUND": (0.87, 0.87, 0.83, 0.96),
    "COLOUR_TITLE_BACKGROUND_LIGHT": (0.96, 0.96, 0.93, 0.93),
    "COLOUR_FRAME_BACKGROUND": (0.87, 0.87, 0.83, 0.96),
    "COLOUR_FRAME_BACKGROUND_LIGHT": (0.92, 0.92, 0.90, 0.96),
    "COLOUR_FRAME_ACTIVE": (0.91, 0.91, 0.86, 0.94),
    "COLOUR_FRAME_DARK": (0.83, 0.83, 0.76, 0.94),
    "COLOUR_FRAME_EXTRA_DARK": (0.76, 0.76, 0.71, 0.94),
    "COLOUR_MAIN_MENU_BAR": (0.882, 0.882, 0.882, 0.94),
    "COLOUR_MAIN_MENU_BAR_TEXT": (0.0, 0.0, 0.0, 0.94),
    "COLOUR_MAIN_MENU_BAR_HILIGHT": (0.96, 0.95, 0.92, 0.94),
    "COLOUR_MENU_WINDOW_BACKGROUND": (0.96, 0.96, 0.96, 0.94),
    "COLOUR_DROP_TARGET": (0.83, 0.83, 0.76, 0.94),
    "COLOUR_HEADER": (0.83, 0.83, 0.76, 0.94),
    "COLOUR_HEADER_ACTIVE": (0.91, 0.91, 0.86, 0.94),
    "COLOUR_HEADER_HOVERED": (0.76, 0.76, 0.71, 0.94),
    "COLOUR_TEXT": (0.0, 0.0, 0.0, 1.0),
    "COLOUR_TEXT_ACTIVE": (0.0, 0.0, 0.2, 1.0),
    "COLOUR_TEXT_DISABLED": (0.7, 0.7, 0.7, 1.0),
    "COLOUR_FRAME_DISABLED": (0.9, 0.9, 0.9, 1.0),
    "COLOUR_TEXT_FADE": (0.76, 0.76, 0.71, 0.94),
    "COLOUR_ERROR_WINDOW_BACKGROUND": (0.94, 0.94, 0.94, 0.94),
    "COLOUR_ERROR_WINDOW_HEADER": (0.87, 0.87, 0.83, 0.96),
    "COLOUR_ERROR_WINDOW_HEADER_NEW": (0.87, 0.87, 0.83, 0.96),
    "COLOUR_ERROR_WINDOW_TEXT": (0.0, 0.0, 0.0, 1.0),
    "COLOUR_CM_WINDOW_TEXT": (0.0, 0.0, 0.0, 1.0),
    "COLOUR_CM_OPTION_HOVERED": (1.0, 1.0, 1.0, 1.0),
    "COLOUR_FRAME_BACKGROUND_BLUE": (0.76, 0.76, 0.83, 1.0),
    "COLOUR_NEUTRAL": (0.6, 0.6, 0.6, 1.0),
    "COLOUR_NEUTRAL_LIGHT": (0.8, 0.8, 0.8, 1.0),
    "COLOUR_SESSION_FEATURE": (0.94, 0.94, 0.94, 0.94),
    "COLOUR_SESSION_FEATURE_BORDER": (0.0, 0.0, 0.0, 0.3),
    "COLOUR_CHECK_MARK": (0.0, 0.0, 0.0, 1.0),
}

_THEME_DARK = {
    "COLOUR_WINDOW_BACKGROUND": (0.12, 0.12, 0.14, 0.94),
    "COLOUR_PANEL_BACKGROUND": (0.12, 0.12, 0.14, 0.94),
    "COLOUR_TITLE_BACKGROUND": (0.19, 0.19, 0.22, 0.96),
    "COLOUR_TITLE_BACKGROUND_LIGHT": (0.24, 0.24, 0.27, 0.93),
    "COLOUR_FRAME_BACKGROUND": (0.20, 0.20, 0.23, 0.96),
    "COLOUR_FRAME_BACKGROUND_LIGHT": (0.26, 0.26, 0.29, 0.96),
    "COLOUR_FRAME_ACTIVE": (0.31, 0.31, 0.35, 0.94),
    "COLOUR_FRAME_DARK": (0.16, 0.16, 0.19, 0.94),
    "COLOUR_FRAME_EXTRA_DARK": (0.10, 0.10, 0.12, 0.94),
    "COLOUR_MAIN_MENU_BAR": (0.15, 0.15, 0.17, 0.94),
    "COLOUR_MAIN_MENU_BAR_TEXT": (0.90, 0.90, 0.92, 0.94),
    "COLOUR_MAIN_MENU_BAR_HILIGHT": (0.29, 0.29, 0.33, 0.94),
    "COLOUR_MENU_WINDOW_BACKGROUND": (0.17, 0.17, 0.19, 0.96),
    "COLOUR_DROP_TARGET": (0.16, 0.16, 0.19, 0.94),
    "COLOUR_HEADER": (0.16, 0.16, 0.19, 0.94),
    "COLOUR_HEADER_ACTIVE": (0.31, 0.31, 0.35, 0.94),
    "COLOUR_HEADER_HOVERED": (0.10, 0.10, 0.12, 0.94),
    "COLOUR_TEXT": (0.90, 0.90, 0.92, 1.0),
    "COLOUR_TEXT_ACTIVE": (0.72, 0.80, 1.0, 1.0),
    "COLOUR_TEXT_DISABLED": (0.45, 0.45, 0.48, 1.0),
    "COLOUR_FRAME_DISABLED": (0.20, 0.20, 0.22, 1.0),
    "COLOUR_TEXT_FADE": (0.10, 0.10, 0.12, 0.94),
    "COLOUR_ERROR_WINDOW_BACKGROUND": (0.12, 0.12, 0.14, 0.94),
    "COLOUR_ERROR_WINDOW_HEADER": (0.19, 0.19, 0.22, 0.96),
    "COLOUR_ERROR_WINDOW_HEADER_NEW": (0.19, 0.19, 0.22, 0.96),
    "COLOUR_ERROR_WINDOW_TEXT": (0.90, 0.90, 0.92, 1.0),
    "COLOUR_CM_WINDOW_TEXT": (0.90, 0.90, 0.92, 1.0),
    "COLOUR_CM_OPTION_HOVERED": (0.30, 0.30, 0.34, 1.0),
    "COLOUR_FRAME_BACKGROUND_BLUE": (0.24, 0.24, 0.33, 1.0),
    "COLOUR_NEUTRAL": (0.55, 0.55, 0.58, 1.0),
    "COLOUR_NEUTRAL_LIGHT": (0.30, 0.30, 0.33, 1.0),
    "COLOUR_SESSION_FEATURE": (0.12, 0.12, 0.14, 0.94),
    "COLOUR_SESSION_FEATURE_BORDER": (0.85, 0.85, 0.88, 0.3),
    "COLOUR_CHECK_MARK": (0.90, 0.90, 0.92, 1.0),
}


def set_theme(dark):
    globals().update(_THEME_DARK if dark else _THEME_LIGHT)


set_theme(settings.get("DARK_MODE", False))

TOOLTIP_APPEAR_DELAY = 1.0
TOOLTIP_HOVERED_TIMER = 0.0
TOOLTIP_HOVERED_START_TIME = 0.0

CE_WIDGET_ROUNDING = 50.0