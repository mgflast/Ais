import os
from PIL import Image
import numpy as np
import tifffile
import glob
from Ais.core import config as cfg
import time
import mrcfile
from scipy.ndimage import label, distance_transform_edt
from skimage.feature import peak_local_max
import starfile
import pandas as pd
from hashlib import blake2b
timer = 0.0

PARTICLE_ID_LENGTH = 10   # 5.95 bits per character, so ~59 bits of id (see add_particle_ids)
_ID_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'   # letters first


def _particle_id(key):
    # 62 symbols (a-z, A-Z, 0-9) = 5.954 bits per character. Adding '-_' for base64's 64 symbols,
    # or '.+' on top of that, would add 0.6 / 1.2 bits across the whole id - never a shorter id, so
    # it stays plain alphanumeric. The first character comes from the top digest byte, which the
    # other 13 don't reach, and is always a letter: an id can't be read back as a number.
    d = blake2b(key.encode('utf-8'), digest_size=16).digest()
    n = int.from_bytes(d, 'big')
    out = [_ID_ALPHABET[d[0] % 52]]
    for _ in range(PARTICLE_ID_LENGTH - 1):
        n, r = divmod(n, 62)
        out.append(_ID_ALPHABET[r])
    return ''.join(out)


def add_particle_ids(df, volume_path, column='aisParticleID'):
    """Give every picked particle a unique id: a base64url blake2b hash of the segmented volume's
    filename - which carries both the tomogram and the picked feature - plus the particle's
    coordinates and, where present, its Euler angles.

    Content-derived rather than random, so re-picking a tomogram reproduces the same ids instead of
    renumbering everything, and two particles can only share an id if they are the same feature at
    the same position in the same tomogram. Including the volume filename is what keeps two features
    picked from one tomogram apart when they land on the same voxel.

    10 characters is ~59 bits: an expected 7e-7 colliding pairs in a 1M particle set, 7e-5 at 10M,
    2e-3 at 50M. Deliberately not longer - a shared id is about as harmful as the duplicate
    particles a set that size already holds. 8 characters would be too short (a 1-in-370 collision
    at 1M, and 1-in-4 at 10M)."""
    tag = os.path.basename(volume_path)
    fields = ('rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
              'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi')
    vals = [df[f].to_numpy(dtype=float) if f in df.columns else None for f in fields]
    ids = []
    for i in range(len(df)):
        key = tag + '|' + '|'.join('' if v is None else f'{v[i]:.4f}' for v in vals)
        ids.append(_particle_id(key))
    df[column] = pd.Series(ids, index=df.index, dtype=object)
    return df

def generate_thumbnail(data, overlay, colour):
    data = np.squeeze(data)
    if data.ndim == 3:                    # slab (Z, H, W) from a 2.5D/3D model -> centre slice (matches the 2D overlay)
        data = data[data.shape[0] // 2]
    overlay = np.squeeze(overlay)
    s = min(data.shape)
    sqdata = data[:s, :s].astype(np.float32)
    overlay = overlay[:s, :s]
    if cfg.se_active_frame:
        sqdata -= cfg.se_active_frame.contrast_lims[0]
        sqdata = sqdata / (cfg.se_active_frame.contrast_lims[1] - cfg.se_active_frame.contrast_lims[0])
        sqdata *= 255
    else:
        sqdata -= np.amin(sqdata)
        sqdata = sqdata / np.amax(sqdata)
        sqdata = sqdata * 255
    sqdata = np.clip(sqdata, 0, 255)
    mask = overlay > 0.5
    thumbnail = np.zeros((*sqdata.shape, 3), dtype=np.uint8)
    thumbnail[:, :, 0] = sqdata
    thumbnail[:, :, 1] = sqdata
    thumbnail[:, :, 2] = sqdata
    thumbnail[mask, 0] = colour[0] * 255
    thumbnail[mask, 1] = colour[1] * 255
    thumbnail[mask, 2] = colour[2] * 255
    thumbnail = Image.fromarray(thumbnail)
    thumbnail = thumbnail.resize((512, 512))
    return thumbnail


def coords_from_tsv(coords_path):
    coords = []
    with open(coords_path, 'r') as file:
        for line in file:
            x, y, z = map(int, line.strip().split('\t'))
            coords.append((x, y, z))
    return coords


def coords_from_star(coords_path):
    try:
        coords = list()
        df = starfile.read(coords_path)
        if 'rlnCoordinateX' in df.columns:
            xk, yk, zk = 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'
            scale = 1.0
        else:
            xk, yk, zk = 'wrpCoordinateX1', 'wrpCoordinateY1', 'wrpCoordinateZ1'
            scale = 0.1
        for _, row in df.iterrows():
            coords.append((int(row[xk] * scale), int(row[yk] * scale), int(row[zk] * scale)))
        return coords
    except Exception as e:
        cfg.set_error(e, f"Could not parse coordinates from star file {coords_path}")
        return []


def coords_from_tsv_to_star(tsv_path, delete_tsv=False):
    tomo_name = os.path.basename(tsv_path[:tsv_path.rfind("__")])
    star_path = os.path.splitext(tsv_path)[0]+'.star'
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'])
    df['rlnMicrographName'] = tomo_name
    starfile.write({'particles': df}, star_path, overwrite=True)
    if delete_tsv:
        os.remove(tsv_path)


def extract_particles(vol_path, coords_path, boxsize, unbin=1, two_dimensional=False, normalize=True):
    coords = coords_from_tsv(coords_path)
    coords = np.array(coords) * unbin
    data = mrcfile.mmap(vol_path, mode='r').data
    imgs = list()

    d = boxsize // 2
    for p in coords:
        x, y, z = p
        if two_dimensional:
            imgs.append(data[z, y - d:y + d, x - d:x + d])
        else:
            imgs.append(data[z - d:z + d, y - d:y + d, x - d:x + d])

    if normalize:
        for i in range(len(imgs)):
            img = np.array(imgs[i]).astype(np.float32)
            img -= np.mean(img)
            img /= np.std(img)
            imgs[i] = img
    return imgs


def _blob_orientation(coords_zyx, mode, sign, vol_shape):
    """Euler angles (rot, tilt, psi) from a blob's principal axes.

    mode 'normal'    -> disk normal (smallest principal axis)
    mode 'long-axis' -> rod axis (largest principal axis)
    sign 'z'         -> chosen vector forced to +z component;
         'center'/'out' -> pointed toward/away from the volume centre.
    Near-isotropic blobs have no defined orientation and return (0, 0, 0).
    rot is left at 0 — a single axis fixes only tilt and psi.
    """
    from Ais.core.filaments import tangent_to_euler_zyz
    c = coords_zyx - coords_zyx.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(c.T @ c / len(c))  # ascending
    if mode == 'normal':
        vec, ratio = eigvecs[:, 0], eigvals[0] / max(eigvals[1], 1e-9)
    else:
        vec, ratio = eigvecs[:, 2], eigvals[1] / max(eigvals[2], 1e-9)
    if ratio > 0.7:  # too isotropic for this mode -> orientation undefined
        return 0.0, 0.0, 0.0
    v_xyz = vec[::-1]  # (z, y, x) -> (x, y, z)
    if sign == 'z':
        if v_xyz[2] < 0:
            v_xyz = -v_xyz
    else:
        to_centre_xyz = (np.array(vol_shape) / 2.0 - coords_zyx.mean(axis=0))[::-1]
        pointing_in = np.dot(v_xyz, to_centre_xyz) > 0
        if (sign == 'center') != pointing_in:
            v_xyz = -v_xyz
    rot, tilt, psi = tangent_to_euler_zyz(v_xyz)[0]
    return float(rot), float(tilt), float(psi)


def pick_particles(mrcpath="", threshold=128, margin=16, min_spacing=10.0, min_size=None, save_txt=True, out_path=None, process=None, array=None, array_pixel_size=None, return_coords=False, binning=1, pixel_size=None, verbose=True, centroid=False, min_particles=0, orient=None, orient_sign='z'):
    ## TODO: clean up
    """
    Ais pick particles blob mode.
    min_spacing: in nanometer
    min_size: in cubic nanometer
    pixel_size: in nanometer
    """
    if array is None:
        if verbose:
            print(f'Reading {mrcpath}')
        data = mrcfile.read(mrcpath)
        if process:
            process.set_progress(0.2)
        if pixel_size is None:
            pixel_size = mrcfile.open(mrcpath, header_only=True).voxel_size.x / 10.0
    else:
        data = array
        pixel_size = array_pixel_size
    if data.dtype == np.float32:
        threshold /= 255
    if data.dtype == np.int8:
        threshold /= 2
    data = data.astype(np.float32)
    if binning > 1:
        z, y, x = data.shape
        b = int(binning)
        data = data[:z // b * b, :y // b * b, :x // b * b]
        _type = data.dtype
        data = data.reshape((z // b, b, y // b, b, x // b, b)).mean(5, dtype=_type).mean(3, dtype=_type).mean(1, dtype=_type)
        pixel_size *= b
        margin = margin // b
    min_distance = max(3, int(min_spacing / pixel_size))
    min_size = min_size / pixel_size ** 3

    binary_vol = data > threshold
    if process:
        process.set_progress(0.4)
    # set regions with insufficient volume to 0
    if verbose:
        print(f'Labelling maxima in {mrcpath}')
    labeled, n = label(binary_vol)
    voxels_per_group = np.bincount(labeled.ravel())
    small_blobs = np.where(voxels_per_group < min_size)[0]
    for sb in small_blobs:
        binary_vol[labeled == sb] = 0


    class Particle:
        def __init__(self, coordinate, score, orientation=None):
            self.coordinate = coordinate
            self.score = score
            self.orientation = orientation

    particles = list()

    if not centroid:    # pick by distance transform peaks, particle score is distance to edge of blob.
        distance = distance_transform_edt(binary_vol, return_distances=True)
        coordinates = peak_local_max(distance, min_distance=min_distance)
        for c in coordinates:
            particles.append(Particle(c, distance[c[0], c[1], c[2]]))
    else:               # pick by centroid of labeled regions, particle score is size of blob.
        coordinates = []
        scores = []
        orientations = []
        for label_id in range(1, n + 1):
            if label_id in small_blobs:
                continue
            mask = labeled == label_id
            coords = np.argwhere(mask)
            if len(coords) > 0:
                coordinates.append(coords.mean(axis=0))
                scores.append(np.sum(mask) * (10.0 * pixel_size)**3)
                if orient is not None:
                    orientations.append(_blob_orientation(coords, orient, orient_sign, binary_vol.shape))
                else:
                    orientations.append(None)
        coordinates = np.array(coordinates) if coordinates else np.empty((0, 3))
        for c, s, o in zip(coordinates, scores, orientations):
            particles.append(Particle(c, s, o))

    # margin is in original px; binning already scaled it (line above), so it is applied here in the
    # binned frame the coordinates live in - e.g. -m 16 with -b 4 excludes 4 binned px = 16 original px.
    if margin > 0:
        mz, my, mx = binary_vol.shape
        particles = [p for p in particles
                     if margin <= p.coordinate[0] < mz - margin
                     and margin <= p.coordinate[1] < my - margin
                     and margin <= p.coordinate[2] < mx - margin]

    particles.sort(key=lambda x: x.score, reverse=True)
    coordinates = [p.coordinate for p in particles]
    scores = [p.score for p in particles]
    orientations = [p.orientation for p in particles]

    if len(coordinates) < min_particles:
        return len(coordinates), 0

    if not return_coords:
        if not save_txt:
            return len(coordinates)

        if out_path is None:
            out_path = os.path.splitext(mrcpath)[0] + "_coords.star"

        write_angles = orient is not None and centroid
        cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ', 'rlnMicrographName', 'aisParticleSize']
        if write_angles:
            cols += ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
        df = pd.DataFrame(columns=cols)
        micrograph_name = os.path.basename(mrcpath).split("__")[0]
        for i in range(len(coordinates)):
            row = [coordinates[i][2] * binning, coordinates[i][1] * binning, coordinates[i][0] * binning, micrograph_name, scores[i]]
            if write_angles:
                rot, tilt, psi = orientations[i]
                row += [rot, tilt, psi]
            df.loc[len(df)] = row
        add_particle_ids(df, mrcpath)
        starfile.write({'particles': df}, out_path, overwrite=True)
        if process:
            process.set_progress(0.99)
        return len(coordinates), 0
    else:
        if process:
            process.set_progress(0.99)
        return coordinates


def bin_2d_array(a, b):
    y, x = a.shape
    a = a[:y//b*b, :x//b*b]
    a = a.reshape((y//b, b, x//b, b)).mean(3).mean(1)
    return a


def bin_mrc(path, bin_factor):
    print(f"Loading '{path}'")
    data = mrcfile.read(path)
    pxs = mrcfile.open(path, header_only=True, permissive=True).voxel_size.x
    z, y, x = data.shape
    b = int(bin_factor)
    print(f"Binning dataset by factor {b} (dtype = {data.dtype})")
    data = data[:z // b * b, :y // b * b, :x // b * b]
    _type = data.dtype
    data = data.reshape((z // b, b, y // b, b, x // b, b)).mean(5).mean(3).mean(1)
    data = data.astype(_type)
    out_path = path[:path.rfind('.mrc')]+f"_bin{b}.mrc"
    print(f"Saving dataset as: '{out_path}'")
    with mrcfile.new(out_path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = pxs * b
    return out_path


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    # from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters?noredirect=1&lq=1
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)


def save_tiff(array, path, pixel_size_nm=100, axes="ZXY"):
    if not (path[-5:] == ".tiff" or path[-4:] == ".tif"):
        path += ".tif"
    if "/" in path:
        root = path.rfind("/")
        root = path[:root]
        if not os.path.exists(root):
            os.makedirs(root)
    tifffile.imwrite(path, array.astype(np.float32), resolution=(
        1. / (1e-7 * pixel_size_nm), 1. / (1e-7 * pixel_size_nm),
        'CENTIMETER'))  # Changed to astype npfloat32 on 230105 to fix importing tifffile tiff with PIL Image open. Default for tifffile export unspecified-float np array appears to be 64bit which PIL doesnt support.


def save_png(array, path, alpha=True):
    try:
        if not path[-4:] == '.png':
            path += '.png'
        if "/" in path:
            root = path.rfind("/")
            root = path[:root]
            if not os.path.exists(root):
                os.makedirs(root)
        if array.dtype != np.dtype(np.uint8):
            array = array.astype(np.uint8)
        if alpha:
            Image.fromarray(array, mode="RGBA").save(path)
        else:
            Image.fromarray(array, mode="RGB").save(path)
    except Exception as e:
        cfg.set_error(e, "Error exporting image as .png. Is the path valid?")

#
# def plot_histogram(data, bins='auto', title=None):
#     plt.hist(data, bins=bins)
#     if title:
#         plt.title(title)
#     plt.show()


def get_filetype(path):
    return path[path.rfind("."):]


def apply_lut_to_float_image(image, lut, contrast_lims=None):
    if len(image.shape) != 2:
        print("Image input in apply_lut_to_float_image is not 2D.")
        return False
    if isinstance(lut, list):
        _lut = np.asarray(lut)
    L, n = np.shape(_lut)
    if contrast_lims is None:
        contrast_lims = (np.amin(image), np.amax(image))
    image = (L - 1) * (image - contrast_lims[0]) / (contrast_lims[1] - contrast_lims[0])
    image = image.astype(int)
    w, h = image.shape
    out_img = np.zeros((w, h, n))
    for x in range(w):
        out_img[x, :, :] = _lut[image[x, :], :]
    return out_img


def is_path_tiff(path):
    if path[-4:] == ".tif" or path[-5:] == ".tiff":
        return True
    return False


def load(path, tag=None):
    if is_path_tiff(path):
        return loadtiff(path)
    else:
        return loadfolder(path, tag)


def loadfolder(path, tag=None):
    if path[-1] != "/":
        path += "/"
    pre_paths = glob.glob(path + "*.tiff") + glob.glob(path + "*.tif")
    paths = list()
    if tag is not None:
        for path in pre_paths:
            if tag in path:
                paths.append(path)
    else:
        paths = pre_paths
    _data = Image.open(paths[0])
    _frame = np.asarray(_data)

    width = _frame.shape[0]
    height = _frame.shape[1]
    depth = len(paths)
    data = np.zeros((depth, width, height), dtype=np.float32)
    _n = len(paths)
    for f in range(0, _n):
        printProgressBar(f, _n, prefix="Loading tiff file\t", length=30, printEnd="\r")
        _data = Image.open(paths[f])
        data[f, :, :] = np.asarray(_data)
    return data


def loadtiff(path, dtype=np.int16):
    "Loads stack as 3d array with dimensions (frames, width, height)"
    _data = Image.open(path)
    _frame = np.asarray(_data)

    width = _frame.shape[0]
    height = _frame.shape[1]
    depth = _data.n_frames
    data = np.zeros((depth, width, height), dtype=dtype)
    for f in range(0, depth):
        printProgressBar(f, depth - 1, prefix="Loading tiff file\t", length=30, printEnd="\r")
        _data.seek(f)
        data[f, :, :] = np.asarray(_data)
    data = np.squeeze(data)
    return data


def tiff_to_mrc(path_in, path_out, apix=1.0):
    data = loadtiff(path_in, dtype=np.float32)
    if path_out[-4:] != ".mrc":
        path_out += ".mrc"
    with mrcfile.new(path_out, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = apix


def tic():
    global timer
    timer = time.time_ns()


def toc(msg):
    print(msg + f" {(time.time_ns() - timer) * 1e-9:.3} seconds")


def clamp(a, _min, _max):
    return min(max(a, _min), _max)

def icosphere_va():
    v = np.array([[0., 0.52573111, 0.85065081],
        [0., -0.52573111, 0.85065081],
        [0.52573111, 0.85065081, 0.],
        [-0.52573111, 0.85065081, 0.],
        [0.85065081, 0., 0.52573111],
        [-0.85065081, 0., 0.52573111],
        [-0., -0.52573111, -0.85065081],
        [-0., 0.52573111, -0.85065081],
        [-0.52573111, -0.85065081, -0.],
        [0.52573111, -0.85065081, -0.],
        [-0.85065081, -0., -0.52573111],
        [0.85065081, -0., -0.52573111],
        [0., 0.20177411, 0.97943209],
        [0., -0.20177411, 0.97943209],
        [0.20177411, 0.73002557, 0.65295472],
        [0.40354821, 0.85472883, 0.32647736],
        [-0.20177411, 0.73002557, 0.65295472],
        [-0.40354821, 0.85472883, 0.32647736],
        [0.32647736, 0.40354821, 0.85472883],
        [0.65295472, 0.20177411, 0.73002557],
        [-0.32647736, 0.40354821, 0.85472883],
        [-0.65295472, 0.20177411, 0.73002557],
        [0.32647736, -0.40354821, 0.85472883],
        [0.65295472, -0.20177411, 0.73002557],
        [-0.32647736, -0.40354821, 0.85472883],
        [-0.65295472, -0.20177411, 0.73002557],
        [-0.20177411, -0.73002557, 0.65295472],
        [-0.40354821, -0.85472883, 0.32647736],
        [0.20177411, -0.73002557, 0.65295472],
        [0.40354821, -0.85472883, 0.32647736],
        [0.20177411, 0.97943209, 0.],
        [-0.20177411, 0.97943209, 0.],
        [0.73002557, 0.65295472, 0.20177411],
        [0.85472883, 0.32647736, 0.40354821],
        [0.40354821, 0.85472883, -0.32647736],
        [0.20177411, 0.73002557, -0.65295472],
        [0.73002557, 0.65295472, -0.20177411],
        [0.85472883, 0.32647736, -0.40354821],
        [-0.73002557, 0.65295472, 0.20177411],
        [-0.85472883, 0.32647736, 0.40354821],
        [-0.40354821, 0.85472883, -0.32647736],
        [-0.20177411, 0.73002557, -0.65295472],
        [-0.73002557, 0.65295472, -0.20177411],
        [-0.85472883, 0.32647736, -0.40354821],
        [0.85472883, -0.32647736, 0.40354821],
        [0.73002557, -0.65295472, 0.20177411],
        [0.97943209, 0., 0.20177411],
        [0.97943209, 0., -0.20177411],
        [-0.85472883, -0.32647736, 0.40354821],
        [-0.73002557, -0.65295472, 0.20177411],
        [-0.97943209, 0., 0.20177411],
        [-0.97943209, 0., -0.20177411],
        [-0., -0.20177411, -0.97943209],
        [-0., 0.20177411, -0.97943209],
        [-0.20177411, -0.73002557, -0.65295472],
        [-0.40354821, -0.85472883, -0.32647736],
        [0.20177411, -0.73002557, -0.65295472],
        [0.40354821, -0.85472883, -0.32647736],
        [-0.32647736, -0.40354821, -0.85472883],
        [-0.65295472, -0.20177411, -0.73002557],
        [0.32647736, -0.40354821, -0.85472883],
        [0.65295472, -0.20177411, -0.73002557],
        [-0.32647736, 0.40354821, -0.85472883],
        [-0.65295472, 0.20177411, -0.73002557],
        [0.32647736, 0.40354821, -0.85472883],
        [0.65295472, 0.20177411, -0.73002557],
        [-0.20177411, -0.97943209, -0.],
        [0.20177411, -0.97943209, -0.],
        [-0.73002557, -0.65295472, -0.20177411],
        [-0.85472883, -0.32647736, -0.40354821],
        [0.73002557, -0.65295472, -0.20177411],
        [0.85472883, -0.32647736, -0.40354821],
        [-0.35682209, 0., 0.93417236],
        [-0.57735027, 0.57735027, 0.57735027],
        [0., 0.93417236, 0.35682209],
        [0.57735027, 0.57735027, 0.57735027],
        [0.35682209, 0., 0.93417236],
        [-0.57735027, -0.57735027, 0.57735027],
        [-0.93417236, 0.35682209, 0.],
        [0., 0.93417236, -0.35682209],
        [0.93417236, 0.35682209, 0.],
        [0.57735027, -0.57735027, 0.57735027],
        [0.35682209, 0., -0.93417236],
        [0.57735027, -0.57735027, -0.57735027],
        [0., -0.93417236, -0.35682209],
        [-0.57735027, -0.57735027, -0.57735027],
        [-0.35682209, 0., -0.93417236],
        [0.57735027, 0.57735027, -0.57735027],
        [0.93417236, -0.35682209, 0.],
        [0., -0.93417236, 0.35682209],
        [-0.93417236, -0.35682209, 0.],
        [-0.57735027, 0.57735027, -0.57735027]])
    f = np.array([[ 0, 20, 12],
        [20, 21, 72],
        [20, 72, 12],
        [12, 72, 13],
        [21,  5, 25],
        [21, 25, 72],
        [72, 25, 24],
        [72, 24, 13],
        [13, 24,  1],
        [ 0, 16, 20],
        [16, 17, 73],
        [16, 73, 20],
        [20, 73, 21],
        [17,  3, 38],
        [17, 38, 73],
        [73, 38, 39],
        [73, 39, 21],
        [21, 39,  5],
        [ 0, 14, 16],
        [14, 15, 74],
        [14, 74, 16],
        [16, 74, 17],
        [15,  2, 30],
        [15, 30, 74],
        [74, 30, 31],
        [74, 31, 17],
        [17, 31,  3],
        [ 0, 18, 14],
        [18, 19, 75],
        [18, 75, 14],
        [14, 75, 15],
        [19,  4, 33],
        [19, 33, 75],
        [75, 33, 32],
        [75, 32, 15],
        [15, 32,  2],
        [ 0, 12, 18],
        [12, 13, 76],
        [12, 76, 18],
        [18, 76, 19],
        [13,  1, 22],
        [13, 22, 76],
        [76, 22, 23],
        [76, 23, 19],
        [19, 23,  4],
        [ 1, 24, 26],
        [24, 25, 77],
        [24, 77, 26],
        [26, 77, 27],
        [25,  5, 48],
        [25, 48, 77],
        [77, 48, 49],
        [77, 49, 27],
        [27, 49,  8],
        [ 5, 39, 50],
        [39, 38, 78],
        [39, 78, 50],
        [50, 78, 51],
        [38,  3, 42],
        [38, 42, 78],
        [78, 42, 43],
        [78, 43, 51],
        [51, 43, 10],
        [ 3, 31, 40],
        [31, 30, 79],
        [31, 79, 40],
        [40, 79, 41],
        [30,  2, 34],
        [30, 34, 79],
        [79, 34, 35],
        [79, 35, 41],
        [41, 35,  7],
        [ 2, 32, 36],
        [32, 33, 80],
        [32, 80, 36],
        [36, 80, 37],
        [33,  4, 46],
        [33, 46, 80],
        [80, 46, 47],
        [80, 47, 37],
        [37, 47, 11],
        [ 4, 23, 44],
        [23, 22, 81],
        [23, 81, 44],
        [44, 81, 45],
        [22,  1, 28],
        [22, 28, 81],
        [81, 28, 29],
        [81, 29, 45],
        [45, 29,  9],
        [ 7, 64, 53],
        [64, 65, 82],
        [64, 82, 53],
        [53, 82, 52],
        [65, 11, 61],
        [65, 61, 82],
        [82, 61, 60],
        [82, 60, 52],
        [52, 60,  6],
        [11, 71, 61],
        [71, 70, 83],
        [71, 83, 61],
        [61, 83, 60],
        [70,  9, 57],
        [70, 57, 83],
        [83, 57, 56],
        [83, 56, 60],
        [60, 56,  6],
        [ 9, 67, 57],
        [67, 66, 84],
        [67, 84, 57],
        [57, 84, 56],
        [66,  8, 55],
        [66, 55, 84],
        [84, 55, 54],
        [84, 54, 56],
        [56, 54,  6],
        [ 8, 68, 55],
        [68, 69, 85],
        [68, 85, 55],
        [55, 85, 54],
        [69, 10, 59],
        [69, 59, 85],
        [85, 59, 58],
        [85, 58, 54],
        [54, 58,  6],
        [10, 63, 59],
        [63, 62, 86],
        [63, 86, 59],
        [59, 86, 58],
        [62,  7, 53],
        [62, 53, 86],
        [86, 53, 52],
        [86, 52, 58],
        [58, 52,  6],
        [ 2, 36, 34],
        [36, 37, 87],
        [36, 87, 34],
        [34, 87, 35],
        [37, 11, 65],
        [37, 65, 87],
        [87, 65, 64],
        [87, 64, 35],
        [35, 64,  7],
        [ 4, 44, 46],
        [44, 45, 88],
        [44, 88, 46],
        [46, 88, 47],
        [45,  9, 70],
        [45, 70, 88],
        [88, 70, 71],
        [88, 71, 47],
        [47, 71, 11],
        [ 1, 26, 28],
        [26, 27, 89],
        [26, 89, 28],
        [28, 89, 29],
        [27,  8, 66],
        [27, 66, 89],
        [89, 66, 67],
        [89, 67, 29],
        [29, 67,  9],
        [ 5, 50, 48],
        [50, 51, 90],
        [50, 90, 48],
        [48, 90, 49],
        [51, 10, 69],
        [51, 69, 90],
        [90, 69, 68],
        [90, 68, 49],
        [49, 68,  8],
        [ 3, 40, 42],
        [40, 41, 91],
        [40, 91, 42],
        [42, 91, 43],
        [41,  7, 62],
        [41, 62, 91],
        [91, 62, 63],
        [91, 63, 43],
        [43, 63, 10]])
    return v.flatten(), f.flatten()
