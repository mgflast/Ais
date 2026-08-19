import mrcfile
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import label
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from collections import deque
import pandas as pd
import json
import starfile
from Ais.core.util import add_particle_ids


def prune_skeleton(skel, min_branch_length=5):
    coords = np.column_stack(np.where(skel))
    if coords.size == 0:
        return []

    voxel_to_index = {tuple(c): i for i, c in enumerate(coords)}
    neighbor_shifts = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    adjacency = [[] for _ in range(len(coords))]
    voxel_set = set(voxel_to_index.keys())

    for i, (z, y, x) in enumerate(coords):
        for dz, dy, dx in neighbor_shifts:
            nbr = (z + dz, y + dy, x + dx)
            if nbr in voxel_set:
                adjacency[i].append(voxel_to_index[nbr])

    # Iteratively remove short terminal branches
    keep = set(range(len(coords)))
    changed = True
    while changed:
        changed = False
        endpoints = [i for i in keep if i < len(adjacency) and len([n for n in adjacency[i] if n in keep]) == 1]

        for ep in endpoints:
            path = [ep]
            current = ep
            while len(path) < min_branch_length:
                neighbors = [n for n in adjacency[current] if n in keep and n not in path]
                if not neighbors:
                    break
                current = neighbors[0]
                path.append(current)
                active_neighbors = [n for n in adjacency[current] if n in keep]
                if len(active_neighbors) != 2:
                    break

            if len(path) < min_branch_length:
                keep -= set(path)
                changed = True

    # Now split at remaining branch points
    branch_points = {i for i in keep if len([n for n in adjacency[i] if n in keep]) > 2}
    visited = set(branch_points)
    components = []

    for start in keep:
        if start in visited:
            continue
        component = []
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for nbr in adjacency[node]:
                if nbr in keep and nbr not in visited:
                    queue.append(nbr)
        if len(component) >= 4:
            components.append(component)

    results = []
    for component in components:
        comp_set = set(component)
        comp_adj = {i: [n for n in adjacency[i] if n in comp_set] for i in component}
        endpoints = [i for i in component if len(comp_adj[i]) == 1]
        if len(endpoints) < 2:
            continue

        start, end = endpoints[0], endpoints[1]
        visited_comp = {start}
        parent = {start: None}
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node == end:
                break
            for nbr in comp_adj[node]:
                if nbr not in visited_comp:
                    visited_comp.add(nbr)
                    parent[nbr] = node
                    queue.append(nbr)

        path_indices = []
        cur = end
        while cur is not None:
            path_indices.append(cur)
            cur = parent.get(cur)
        path_indices.reverse()

        out_vol = np.zeros_like(skel, dtype=bool)
        for idx in path_indices:
            z, y, x = coords[idx]
            out_vol[z, y, x] = True

        # Classify endpoints: check if they were branch points in the full skeleton
        # 0 = end (true terminal), 1 = junction (was a branch point neighbor)
        def endpoint_type(idx):
            # Check if any neighbor in the full skeleton (not just this component) is a branch point
            for nbr in adjacency[idx]:
                if nbr in branch_points:
                    return 2  # junction
            return 1  # true end

        start_type = endpoint_type(path_indices[0])
        end_type = endpoint_type(path_indices[-1])

        results.append((out_vol, start_type, end_type))

    return results if results else [(np.zeros_like(skel, dtype=bool), 1, 1)]


def parameterize_instances(labels, min_branch_length_px=5):
    filaments = []
    for j in range(1, int(labels.max()) + 1):
        mask = (labels == j)
        if not np.any(mask):
            continue

        skel = skeletonize(mask)
        pruned_results = prune_skeleton(skel, min_branch_length_px)

        for pruned_skel, start_type, end_type in pruned_results:
            coords = np.column_stack(np.where(pruned_skel))
            if len(coords) > 0:
                filaments.append(Filament(coords, start_type=start_type, end_type=end_type))

    return filaments


def enforce_tangent_consistency(tangent_xyz):
    t = tangent_xyz.copy()
    for i in range(1, t.shape[0]):
        if np.dot(t[i], t[i-1]) < 0:   # angle > 90° → flip
            t[i] *= -1
    return t


def tangent_to_euler_zyz(tangent_xyz):
    tangent_xyz = np.atleast_2d(tangent_xyz)
    tangent_xyz = tangent_xyz / np.linalg.norm(tangent_xyz, axis=1, keepdims=True)

    tilt = np.degrees(np.arccos(tangent_xyz[:, 2]))
    psi = np.degrees(np.arctan2(tangent_xyz[:, 1], -tangent_xyz[:, 0]))
    rot = np.zeros_like(tilt)

    return np.column_stack((rot, tilt, psi))



class Filament:
    SHIFTS = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]
    SMOOTHING = 500.0

    def __init__(self, skeleton_coordinates, start_type=1, end_type=1):
        self.coords = skeleton_coordinates
        self.valid = False
        self.tck = None
        self.length = None
        self.start_type = start_type  # 1 = true end, 2 = junction
        self.end_type = end_type
        self.gen_spline()
        self.linearize_spline()

    def gen_spline(self):
        if len(self.coords) < 4:
            return

        skel_voxels = set(map(tuple, self.coords))
        endpoints = []
        for c in self.coords:
            c_tup = tuple(c)
            nbr_count = sum(
                ((c_tup[0] + dz, c_tup[1] + dy, c_tup[2] + dx) in skel_voxels) for dz, dy, dx in Filament.SHIFTS)
            if nbr_count == 1:
                endpoints.append(c_tup)
        if not endpoints:
            return

        start = endpoints[0]

        visited = set()
        path = []
        queue = deque([start])
        while queue:
            v = queue.pop()
            if v not in visited:
                visited.add(v)
                path.append(v)
                for dz, dy, dx in Filament.SHIFTS:
                    nbr = (v[0] + dz, v[1] + dy, v[2] + dx)
                    if nbr in skel_voxels and nbr not in visited:
                        queue.append(nbr)

        path = np.array(path)
        if path.shape[0] < 4:
            return
        self.coords = path
        self.tck, u = splprep(path.T, s=Filament.SMOOTHING)
        self.valid = True

    def linearize_spline(self, n_samples=100):
        if not self.valid:
            return
        u_test = np.linspace(0, 1, n_samples)
        coords = np.array(splev(u_test, self.tck))
        diffs = np.diff(coords, axis=1)
        dists = np.sqrt(np.sum(diffs**2, axis=0))
        cumdist = np.concatenate(([0], np.cumsum(dists)))
        total_length = cumdist[-1]
        self.length = total_length
        arc_positions = np.linspace(0, total_length, n_samples)
        u_new = np.interp(arc_positions, cumdist, u_test)
        coords_uniform = np.array(splev(u_new, self.tck))
        tck_new, _ = splprep(coords_uniform, u=np.linspace(0, 1, n_samples), s=0, k=self.tck[2])
        self.tck = tck_new

    def unbin(self, b=2):
        self.coords *= b
        self.gen_spline()
        self.linearize_spline()
        return self

    def to_dict(self):
        return {
            "skeleton_coordinates": self.coords.tolist(),
            "start_type": self.start_type,
            "end_type": self.end_type,
        }

    @staticmethod
    def read(p):
        with open(p, 'r') as f:
            return Filament.from_dict(json.load(f))

    @staticmethod
    def from_dict(d):
        skeleton_coordinates = d['skeleton_coordinates']
        f = Filament(skeleton_coordinates)
        f.start_type = d.get('start_type', 1)
        f.end_type = d.get('end_type', 1)
        return f

    @classmethod
    def from_path(cls, path):
        with open(path, 'r') as f:
            filaments = json.load(f)
        out_filaments = list()
        for f in filaments:
            out_filaments.append(cls.from_dict(f))
        return out_filaments

    def __str__(self):
        return f"Filament with length {self.length:.1f}, {len(self.coords)}-voxel skeleton."

    def plot(self):
        filament_coords = np.array(splev(np.linspace(0, 1, 100), self.tck))
        plt.plot(filament_coords[2], filament_coords[1], linewidth=10, alpha=0.2)

    def sample_coordinates(self, spacing, offset=0, twist_per_sample=0.0):
        if self.length is None or self.length == 0.0:
            return np.array([])

        x = np.arange(offset, self.length, spacing) / self.length
        if not x.any():
            return np.array([])

        coords = np.array(splev(x, self.tck)).T  # [z, y, x]

        tangents = np.array(splev(x, self.tck, der=1)).T
        tangents_xyz = tangents[:, [2, 1, 0]]  # normal vector in [x, y, z]
        tangents_xyz - tangents_xyz / (np.linalg.norm(tangents_xyz, axis=1, keepdims=True) + 1e-12)
        tangents_xyz = enforce_tangent_consistency(tangents_xyz)
        euler_angles = tangent_to_euler_zyz(tangents_xyz)
        twist_angles = (np.arange(len(coords)) * twist_per_sample) % 360.0
        euler_angles[:, 0] = (euler_angles[:, 0] + twist_angles) % 360.0

        drdx = np.array(splev(x, self.tck, der=1)).T
        ddrddx = np.array(splev(x, self.tck, der=2)).T
        radius_of_curvature = (np.linalg.norm(drdx, axis=1)**3 / np.linalg.norm(np.cross(drdx, ddrddx), axis=1))
        radius_of_curvature = radius_of_curvature[:, None]

        # Connectivity: 0 = middle, start_type/end_type for first/last
        connectivity = np.zeros(len(coords), dtype=int)
        connectivity[0] = self.start_type
        connectivity[-1] = self.end_type

        return np.hstack([coords, euler_angles, radius_of_curvature, connectivity[:, None]])


def bin_volume(vol, b):
    j, k, l = vol.shape
    vol = vol[:j // b * b, :k // b * b, :l // b * b]
    vol = vol.reshape((j // b, b, k // b, b, l // b, b)).mean(axis=(1, 3, 5))
    return vol


def pick_filament(mrcpath, out_path, threshold, spacing_nm, size_nm, binning, margin, pixel_size=1.0, min_length=50.0, twist_per_sample=0.0):
    """
    volume_path: path to a semantic segmentation volume.
    threshold: threshold to binarize the volume (0-255)
    spacing: spacing between picked points along the filament (in nm)
    size: minimum blob size (in nm3)
    binning: binning factor to apply to the volume before processing
    margin: margin (in nm) to exclude points too close to the border of the volume
    pixel_size: original pixel size of the volume (in nm)
    min_length: minimum length of filaments to keep (in nm)
    """
    volume = mrcfile.read(mrcpath)

    if volume.dtype == np.float32:
        threshold /= 255
    elif volume.dtype == np.int8:
        threshold /= 2

    volume = volume.astype(np.float32)
    volume = bin_volume(volume, binning)
    pixel_size *= binning
    margin = int(margin) // int(binning)   # -m is original px (as in blob mode); coords here are binned

    labels, _ = label(volume > threshold)
    group_sizes = np.bincount(labels.ravel())
    for n in np.where(group_sizes < (size_nm / (pixel_size**3)))[0].tolist():
        labels[labels == n] = 0

    filaments = parameterize_instances(labels, int(min_length // pixel_size))

    df = pd.DataFrame(columns=['rlnCoordinateZ', 'rlnCoordinateY', 'rlnCoordinateX', 'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi', 'aisRadiusOfCurvature', 'aisLog10RadiusOfCurvature', 'aisFilamentID', 'aisFilamentConnectivity'])
    for j, f in enumerate(filaments):
        c = f.sample_coordinates(spacing=spacing_nm / pixel_size, offset=0.0, twist_per_sample=twist_per_sample)
        for z, y, x, rot, tilt, psi, roc, conn in c:
            _roc = roc * pixel_size * 10.0
            df.loc[len(df)] = [z, y, x, rot, tilt, psi, _roc, np.log10(_roc), int(j), int(conn)]

    j, k, l = volume.shape
    df = df[(df['rlnCoordinateZ'] > margin) & (df['rlnCoordinateZ'] < j - margin) &
            (df['rlnCoordinateY'] > margin) & (df['rlnCoordinateY'] < k - margin) &
            (df['rlnCoordinateX'] > margin) & (df['rlnCoordinateX'] < l - margin)]

    df['rlnCoordinateX'] *= binning
    df['rlnCoordinateY'] *= binning
    df['rlnCoordinateZ'] *= binning
    df['rlnMicrographName'] = os.path.basename(mrcpath).split("__")[0]+".mrc"
    add_particle_ids(df, mrcpath)
    starfile.write({'particles': df}, out_path, overwrite=True)

    return len(df), len(filaments)


if __name__ == "__main__":
    pick_filament('Z:/compu_projects/easymode/segmented/20230908_lmla003_ts_006_10.00Apx__microtubule.mrc', 'C:/Users/Mart Last/Desktop/temp.star', 128, 10, 100, 2, 16, pixel_size=1.0, min_length=10.0)