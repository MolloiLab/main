# -*- coding: utf-8 -*-
"""
Gammex Phantom Generation for CatSim
Creates a 3D digital phantom with Calcium and Iodine inserts
Exports per-material RAW masks and Catsim-style JSON
"""

import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------
# Geometry: Cylinder utilities
#-------------------------------
def create_cylinder_3d(nx, ny, nz, grid_x, grid_y, grid_z, center_xy, radius, z_min=-2.5, z_max=2.5):
    """
    Create a 3D boolean mask for a cylinder.
    
    Parameters:
    - nx, ny, nz: grid dimensions
    - grid_x, grid_y, grid_z: coordinate arrays (cell centers)
    - center_xy: (cx, cy) center of cylinder in cm
    - radius: cylinder radius in cm
    - z_min, z_max: Z extent of cylinder in cm
    
    Returns: Boolean mask (nx, ny, nz)
    """
    cx, cy = center_xy
    r2 = radius ** 2
    
    # Create meshgrid for vectorized operations
    X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    
    # Distance from cylinder axis
    dist_xy_sq = (X - cx)**2 + (Y - cy)**2
    
    # Cylinder mask: inside radius AND within Z bounds
    mask = (dist_xy_sq <= r2) & (Z >= z_min) & (Z <= z_max)
    
    return mask

def create_voxel_grid(nx, ny, nz, fov_xy=40.0, fov_z=10.0):
    """
    Create coordinate arrays for voxel grid.
    
    Returns: grid_x, grid_y, grid_z, dx, dz
    """
    dx = fov_xy / nx
    dz = fov_z / nz
    
    # Cell centers
    start_x = -fov_xy / 2 + dx / 2
    start_z = -fov_z / 2 + dz / 2
    
    grid_x = np.linspace(start_x, start_x + (nx - 1) * dx, nx)
    grid_y = np.linspace(start_x, start_x + (ny - 1) * dx, ny)  # Square FOV
    grid_z = np.linspace(start_z, start_z + (nz - 1) * dz, nz)
    
    return grid_x, grid_y, grid_z, dx, dz

#-------------------------------
# Phantom Assembly
#-------------------------------

def assemble_gammex_phantom(nx=512, ny=512, nz=64, fov_xy=40.0, fov_z=10.0):
    """
    Build the Gammex phantom with labeled regions.
    Matches DukeSim layout:
    - Inner ring (5 cm): 7 iodine + 1 solid_water
    - Outer ring (10.5 cm): 5 calcium + 3 water
    - Center: water
    
    Returns:
    - int_phantom: 3D integer array with material labels
    - metadata: dict with grid parameters
    """
    # Create grid
    grid_x, grid_y, grid_z, dx, dz = create_voxel_grid(nx, ny, nz, fov_xy, fov_z)
    
    # Initialize phantom (label 0 = air)
    int_phantom = np.zeros((nx, ny, nz), dtype=np.int8)
    
    print(f"Creating Gammex phantom {nx}×{ny}×{nz}")
    print(f"FOV: {fov_xy}×{fov_xy}×{fov_z} cm, Voxel: {dx:.3f}×{dx:.3f}×{dz:.3f} cm")
    
    # --- 1. Phantom Body (Solid Water, 33cm diameter) ---
    print("  Adding phantom body (solid water)...")
    body_mask = create_cylinder_3d(nx, ny, nz, grid_x, grid_y, grid_z, 
                                     (0.0, 0.0), radius=16.5)
    int_phantom[body_mask] = 1  # Label 1: ncat_water (body)
    
    # --- 2. Inner Ring (7 Iodine + 1 Solid Water at 5.0 cm radius) ---
    # Order: I_20_0, I_15_0, I_10_0, I_7_5, I_5_0, I_2_5, I_2_0, solid_water
    inner_ring_configs = [
        ('I_20_0',  2),
        ('I_15_0',  3),
        ('I_10_0',  4),
        ('I_7.5',   5),
        ('I_5_0',   6),
        ('I_2.5',   7),
        ('I_2_0',   8),
        ('ncat_water', 9)  # Extra solid water insert
    ]
    
    n_inner = len(inner_ring_configs)
    inner_radius = 5.0
    
    print(f"  Adding {n_inner} inserts (inner ring at {inner_radius} cm)...")
    
    for idx, (name, label) in enumerate(inner_ring_configs):
        # Match DukeSim: start at π (180°), go counterclockwise
        angle = math.pi - idx * (2 * math.pi / n_inner)
        cx = inner_radius * math.cos(angle)
        cy = inner_radius * math.sin(angle)
        
        mask = create_cylinder_3d(nx, ny, nz, grid_x, grid_y, grid_z, 
                                   (cx, cy), radius=1.4)
        int_phantom[mask] = label
        print(f"    {name}: {np.sum(mask)} voxels at ({cx:.2f}, {cy:.2f}), angle={math.degrees(angle):.1f}°")
    
    # --- 3. Outer Ring (5 Calcium + 3 Water at 10.5 cm radius) ---
    # Order: Ca_400, Ca_300, Ca_200, Ca_100, Ca_50, water, water, water
    outer_ring_configs = [
        ('Ca_400', 10),
        ('Ca_300', 11),
        ('Ca_200', 12),
        ('Ca_100', 13),
        ('Ca_50',  14),
        ('water',  15),
        ('water', 16),  # Different label for tracking
        ('water', 17)   # Different label for tracking
    ]
    
    n_outer = len(outer_ring_configs)
    outer_radius = 10.5
    
    print(f"  Adding {n_outer} inserts (outer ring at {outer_radius} cm)...")
    
    for idx, (name, label) in enumerate(outer_ring_configs):
        # Match DukeSim: start at 120°, go counterclockwise
        angle = math.radians(120) - idx * (2 * math.pi / n_outer)
        cx = outer_radius * math.cos(angle)
        cy = outer_radius * math.sin(angle)
        
        mask = create_cylinder_3d(nx, ny, nz, grid_x, grid_y, grid_z, 
                                   (cx, cy), radius=1.4)
        int_phantom[mask] = label
        print(f"    {name}: {np.sum(mask)} voxels at ({cx:.2f}, {cy:.2f}), angle={math.degrees(angle):.1f}°")
    
    # --- 4. Central Water Reference ---
    print("  Adding central water reference...")
    water_mask = create_cylinder_3d(nx, ny, nz, grid_x, grid_y, grid_z, 
                                      (0.0, 0.0), radius=1.4)
    int_phantom[water_mask] = 18  # Label 18: central water
    
    metadata = {
        'nx': nx, 'ny': ny, 'nz': nz,
        'fov_xy': fov_xy, 'fov_z': fov_z,
        'dx': dx, 'dz': dz,
        'grid_x': grid_x, 'grid_y': grid_y, 'grid_z': grid_z
    }
    
    print(f"Phantom assembly complete. Labels: {np.unique(int_phantom)}")
    return int_phantom, metadata

#-------------------------------
# Export per-material RAW masks
#-------------------------------
def export_gammex_material_maps_raw(
    int_phantom,
    outdir=".",
    value_true=1,
    value_false=0
):
    """
    Export one headerless .raw per MATERIAL (not per label).
    Multiple labels can map to the same material.
    """
    # Map each label to its CatSim material name
    label_to_material = {
        0: "air",
        1: "ncat_water",
        2: "i20",
        3: "i15",
        4: "i10",
        5: "i7.5",
        6: "i5",
        7: "i2.5",
        8: "i2",
        9: "ncat_water",  # ✅ Same as label 1 and 18
        10: "ca400",
        11: "ca300",
        12: "ca200",
        13: "ca100",
        14: "ca50",
        15: "water",
        16: "water",      # ✅ Same as labels 15 and 17
        17: "water",      # ✅ Same as labels 15 and 16
        18: "ncat_water"  # ✅ Same as labels 1 and 9
    }
    
    os.makedirs(outdir, exist_ok=True)
    
    nx, ny, nz = int_phantom.shape
    
    # Get unique material names
    unique_materials = sorted(set(label_to_material.values()))
    
    print(f"  Exporting {len(unique_materials)} unique materials from {len(label_to_material)} labels...")
    
    for material in unique_materials:
        # Find ALL labels that use this material
        labels_for_material = [label for label, mat in label_to_material.items() if mat == material]
        
        # Combine all voxels with any of these labels
        combined_mask = np.zeros_like(int_phantom, dtype=bool)
        for label in labels_for_material:
            combined_mask |= (int_phantom == label)
        
        # Create binary data
        data = np.where(combined_mask, value_true, value_false).astype(np.int8)
        
        # Transpose to (nz, ny, nx)
        data = data.transpose(2, 1, 0)
        
        # Filename based on material name
        fname = f"gammex_{nx}_{ny}_{nz}_{material}.raw"
        path = os.path.join(outdir, fname)
        
        # Write file
        data_contig = np.ascontiguousarray(data)
        with open(path, "wb") as f:
            f.write(data_contig.tobytes(order="C"))
        
        voxel_count = np.sum(combined_mask)
        label_str = ", ".join(str(l) for l in labels_for_material)
        print(f"    {material}: {voxel_count} voxels (from labels {label_str})")
    
    return None
#-------------------------------
# Emit Catsim/XCIST-style JSON
#-------------------------------
def write_gammex_catsim_json(
    outdir,
    nx, ny, nz,
    x_size=0.078125,
    y_size=0.078125,
    z_size=0.15625,
    x_offset=None,
    y_offset=None,
    z_offset=None,
    datatype="int8"
):
    """
    Generate CatSim JSON config.
    Now exports only UNIQUE materials, not duplicate labels.
    """
    # List of unique materials (in order)
    unique_materials = [
        "air",     # 0
        "ncat_water",   # 1 (body + inserts)
        "i20",          # 2
        "i15",          # 3
        "i10",          # 4
        "i7.5",         # 5
        "i5",           # 6
        "i2.5",         # 7
        "i2",           # 8
        "ca400",        # 9
        "ca300",        # 10
        "ca200",        # 11
        "ca100",        # 12
        "ca50",         # 13
        "water"         # 14 (outer ring + center)
    ]
    
    n_materials = len(unique_materials)
    
    # Build filename list
    volumefractionmap_filename = [
        f"gammex_{nx}_{ny}_{nz}_{mat}.raw" 
        for mat in unique_materials
    ]
    
    # Material names for CatSim (same as filenames without .raw)
    mat_name = unique_materials
    
    # Replicate parameters for each material
    volumefractionmap_datatype = [datatype] * n_materials
    cols_arr = [nx] * n_materials
    rows_arr = [ny] * n_materials
    slices_arr = [nz] * n_materials
    
    # Convert cm to mm
    x_size_arr = [x_size * 10.0] * n_materials
    y_size_arr = [y_size * 10.0] * n_materials
    z_size_arr = [z_size * 10.0] * n_materials
    
    # Offsets
    x_off = nx // 2 if x_offset is None else x_offset
    y_off = ny // 2 if y_offset is None else y_offset
    z_off = nz // 2 if z_offset is None else z_offset
    
    x_offset_arr = [x_off] * n_materials
    y_offset_arr = [y_off] * n_materials
    z_offset_arr = [z_off] * n_materials
    
    # Build JSON structure
    data = {
        "n_materials": n_materials,
        "mat_name": mat_name,
        "volumefractionmap_filename": volumefractionmap_filename,
        "volumefractionmap_datatype": volumefractionmap_datatype,
        "cols": cols_arr,
        "rows": rows_arr,
        "slices": slices_arr,
        "x_size": x_size_arr,
        "y_size": y_size_arr,
        "z_size": z_size_arr,
        "x_offset": x_offset_arr,
        "y_offset": y_offset_arr,
        "z_offset": z_offset_arr,
    }
    
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, "gammex_config.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nWrote {json_path}")
    print(f"  Unique materials: {n_materials}")
    print(f"  (Merged duplicate labels into single material masks)")
    
    return json_path

#-------------------------------
# Visualization
#-------------------------------
def visualize_gammex_phantom(int_phantom, slice_idx=None):
    """
    Visualize a central slice of the Gammex phantom.
    """
    nx, ny, nz = int_phantom.shape
    
    if slice_idx is None:
        slice_idx = nz // 2
    
    # Extract slice (transpose for correct orientation)
    slice_data = int_phantom[:, :, slice_idx].T
    
    plt.figure(figsize=(10, 10))
    plt.imshow(slice_data, cmap='tab20', origin='lower', interpolation='nearest')
    plt.colorbar(label='Material Label', shrink=0.8)
    plt.title(f"Gammex Phantom - Slice {slice_idx}/{nz}")
    plt.xlabel("X (voxels)")
    plt.ylabel("Y (voxels)")
    plt.tight_layout()
    plt.show()

#-------------------------------
# Main execution
#-------------------------------
if __name__ == "__main__":
    # Parameters
    NX, NY, NZ = 512, 512, 64
    FOV_XY = 40.0  # cm
    FOV_Z = 10.0   # cm
    
    # Output directory
    outdir = r"D:\CatSim\gecatsim\phantom\Gammex_Phantom"
    
    # 1. Create integer phantom
    print("="*60)
    print("STEP 1: Creating Gammex Phantom")
    print("="*60)
    int_phantom, metadata = assemble_gammex_phantom(
        nx=NX, ny=NY, nz=NZ,
        fov_xy=FOV_XY, fov_z=FOV_Z
    )
    
    # 2. Export per-material RAW masks
    print("\n" + "="*60)
    print("STEP 2: Exporting Material RAW Masks")
    print("="*60)
    # In main execution (line ~315):
    export_gammex_material_maps_raw(
        int_phantom,
        outdir=outdir,
        value_true=1,      # ✅ ONLY CHANGE THIS (was 255)
        value_false=0
    )
    
    # 3. Generate Catsim JSON config
    print("\n" + "="*60)
    print("STEP 3: Generating Catsim JSON Config")
    print("="*60)
    write_gammex_catsim_json(
        outdir=outdir,
        nx=NX, ny=NY, nz=NZ,
        x_size=FOV_XY / NX,  # cm per voxel
        y_size=FOV_XY / NY,
        z_size=FOV_Z / NZ,
        datatype="int8"
    )
    
    # 4. Visualize
    print("\n" + "="*60)
    print("STEP 4: Visualization")
    print("="*60)
    visualize_gammex_phantom(int_phantom)
    
    print("\n✅ Gammex phantom generation complete!")
    print(f"📁 Output directory: {outdir}")