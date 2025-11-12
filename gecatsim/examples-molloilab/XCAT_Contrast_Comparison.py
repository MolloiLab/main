#!/usr/bin/env python
"""
XCAT Contrast Comparison - Multi-Phantom CT Simulation

This script simulates the same anatomical phantom with three different
levels of iodine contrast enhancement in the blood, demonstrating contrast-
enhanced CT imaging.

Phantoms:
1. Low Contrast:    1.0% iodine in blood
2. Medium Contrast: 2.0% iodine in blood
3. High Contrast:   3.0% iodine in blood

All phantoms share the same anatomy (density maps) but differ only in
blood material composition.

Author: CatSim Example
Date: 2024
"""

# ============================================================================
# IMPORTS
# ============================================================================
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

print("="*80)
print("XCAT CONTRAST COMPARISON - MULTI-PHANTOM SIMULATION")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define the three phantom variants
phantoms = [
    {
        'name': 'Low Contrast (1.0% Iodine)',
        'filename': 'Adult_Female_50percentile_Chest_Phantom_slab_400_1650x1050x1/adult_female_contrast_low.json',
        'results_name': 'xcat_contrast_low',
        'iodine_pct': 1.0
    },
    {
        'name': 'Medium Contrast (2.0% Iodine)',
        'filename': 'Adult_Female_50percentile_Chest_Phantom_slab_400_1650x1050x1/adult_female_contrast_medium.json',
        'results_name': 'xcat_contrast_medium',
        'iodine_pct': 2.0
    },
    {
        'name': 'High Contrast (3.0% Iodine)',
        'filename': 'Adult_Female_50percentile_Chest_Phantom_slab_400_1650x1050x1/adult_female_contrast_high.json',
        'results_name': 'xcat_contrast_high',
        'iodine_pct': 3.0
    }
]

# Common scan parameters (will be same for all phantoms)
scan_params = {
    'kVp': 120,  # Will be determined by spectrum file
    'mA': 800,
    'views_per_rotation': 500,
    'fov_mm': 450.0,
    'image_size': 512,
    'slice_count': 4,
    'slice_thickness_mm': 0.568,
}

# Storage for reconstructed images
reconstructed_images = []

# ============================================================================
# SIMULATE EACH PHANTOM
# ============================================================================

for i, phantom_config in enumerate(phantoms):
    print("="*80)
    print(f"PHANTOM {i+1}/3: {phantom_config['name']}")
    print("="*80)
    print()

    # Initialize CatSim
    ct = xc.CatSim()

    # ========================================================================
    # PHANTOM CONFIGURATION
    # ========================================================================
    print("Phantom Configuration:")
    ct.phantom.callback = "Phantom_Voxelized"
    ct.phantom.projectorCallback = "C_Projector_Voxelized"
    ct.phantom.filename = phantom_config['filename']
    ct.phantom.centerOffset = [0.0, 0.0, 0.0]
    ct.phantom.scale = 1.0
    print(f"  Type:             Voxelized XCAT")
    print(f"  Iodine Level:     {phantom_config['iodine_pct']}% in blood")
    print(f"  File:             {phantom_config['filename']}")
    print()

    # ========================================================================
    # SCANNER HARDWARE
    # ========================================================================
    print("Scanner Configuration:")
    ct.scanner.detectorCallback = "Detector_ThirdgenCurved"
    ct.scanner.sid = 540.0
    ct.scanner.sdd = 950.0
    ct.scanner.detectorColCount = 900
    ct.scanner.detectorRowCount = 16
    ct.scanner.detectorColSize = 1.0
    ct.scanner.detectorRowSize = 1.0
    ct.scanner.detectorMaterial = "Lumex"
    ct.scanner.detectorDepth = 3.0
    print(f"  Geometry:         SID={ct.scanner.sid}mm, SDD={ct.scanner.sdd}mm")
    print(f"  Detector:         {ct.scanner.detectorColCount}×{ct.scanner.detectorRowCount}, {ct.scanner.detectorColSize}mm pitch")
    print()

    # ========================================================================
    # PROTOCOL
    # ========================================================================
    print("Scan Protocol:")
    ct.protocol.scanTrajectory = "Gantry_Helical"
    ct.protocol.mA = scan_params['mA']
    ct.protocol.spectrumFilename = "tungsten_tar7.0_120_filt.dat"  # 120 kVp
    ct.protocol.viewsPerRotation = scan_params['views_per_rotation']
    ct.protocol.viewCount = scan_params['views_per_rotation']
    ct.protocol.stopViewId = ct.protocol.viewCount - 1
    ct.protocol.rotationTime = 1.0
    ct.protocol.tableSpeed = 0  # Axial scan
    print(f"  Technique:        120 kVp, {ct.protocol.mA} mA")
    print(f"  Views:            {ct.protocol.viewCount} ({ct.protocol.viewsPerRotation}/rotation)")
    print(f"  Rotation Time:    {ct.protocol.rotationTime} sec")
    print()

    # ========================================================================
    # PHYSICS
    # ========================================================================
    print("Physics Simulation:")
    ct.physics.energyCount = 20
    ct.physics.colSampleCount = 2
    ct.physics.rowSampleCount = 1
    ct.physics.srcXSampleCount = 2
    ct.physics.srcYSampleCount = 1
    ct.physics.viewSampleCount = 1
    ct.physics.enableQuantumNoise = 1
    ct.physics.enableElectronicNoise = 1
    print(f"  Energy Bins:      {ct.physics.energyCount}")
    print(f"  Sampling:         Det {ct.physics.colSampleCount}×{ct.physics.rowSampleCount}, " +
          f"Focal {ct.physics.srcXSampleCount}×{ct.physics.srcYSampleCount}")
    print(f"  Noise:            Quantum + Electronic")
    print()

    # ========================================================================
    # RECONSTRUCTION
    # ========================================================================
    print("Reconstruction:")
    ct.recon.fov = scan_params['fov_mm']
    ct.recon.imageSize = scan_params['image_size']
    ct.recon.sliceCount = scan_params['slice_count']
    ct.recon.sliceThickness = scan_params['slice_thickness_mm']
    ct.recon.reconType = 'fdk_equiAngle'
    ct.recon.kernelType = 'standard'
    pixel_size = ct.recon.fov / ct.recon.imageSize
    print(f"  Algorithm:        FDK (equiAngle)")
    print(f"  Matrix:           {ct.recon.imageSize}×{ct.recon.imageSize}×{ct.recon.sliceCount}")
    print(f"  FOV:              {ct.recon.fov} mm")
    print(f"  Pixel Size:       {pixel_size:.3f} mm")
    print()

    # ========================================================================
    # OUTPUT
    # ========================================================================
    ct.resultsName = phantom_config['results_name']
    ct.do_Recon = 1
    print(f"Output: {ct.resultsName}")
    print()

    # ========================================================================
    # RUN SIMULATION
    # ========================================================================
    print(f"Running simulation... (Started: {datetime.now().strftime('%H:%M:%S')})")
    ct.run_all()

    # ========================================================================
    # RECONSTRUCTION
    # ========================================================================
    print(f"Reconstructing... (Started: {datetime.now().strftime('%H:%M:%S')})")
    recon.recon(ct)

    # ========================================================================
    # LOAD RECONSTRUCTED IMAGE
    # ========================================================================
    imgFname = f"{ct.resultsName}_{ct.recon.imageSize}x{ct.recon.imageSize}x{ct.recon.sliceCount}.raw"
    img = xc.rawread(imgFname, [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize], 'float')

    # Store the middle slice and parameters
    middle_slice = np.int32(ct.recon.sliceCount / 2)
    reconstructed_images.append({
        'image': img[middle_slice, :, :],
        'name': phantom_config['name'],
        'iodine_pct': phantom_config['iodine_pct'],
        'params': {
            'kVp': 120,
            'mA': ct.protocol.mA,
            'matrix': f"{ct.recon.imageSize}×{ct.recon.imageSize}",
            'fov': ct.recon.fov,
            'pixel_size': pixel_size,
            'slice_thickness': ct.recon.sliceThickness,
        }
    })

    print(f"✓ Phantom {i+1}/3 complete")
    print()

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
print("="*80)
print("CREATING COMPARISON FIGURE")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('XCAT Phantom - Contrast Enhancement Comparison', fontsize=16, fontweight='bold')

for i, img_data in enumerate(reconstructed_images):
    ax = axes[i]

    # Display image
    im = ax.imshow(img_data['image'], cmap='gray', vmin=-300, vmax=500)

    # Title with phantom name
    ax.set_title(img_data['name'], fontsize=14, fontweight='bold', pad=10)

    # Add scan parameters as text overlay
    params = img_data['params']
    param_text = (
        f"kVp: {params['kVp']}\n"
        f"mA: {params['mA']}\n"
        f"Matrix: {params['matrix']}\n"
        f"FOV: {params['fov']:.0f} mm\n"
        f"Pixel: {params['pixel_size']:.3f} mm\n"
        f"Slice: {params['slice_thickness']:.3f} mm"
    )

    ax.text(0.02, 0.98, param_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            color='white',
            family='monospace')

    ax.axis('off')

# Add colorbar
cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.02, fraction=0.046)
cbar.set_label('HU (Hounsfield Units)', fontsize=12)

plt.tight_layout()

# Save figure
output_filename = 'xcat_contrast_comparison.png'
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"✓ Saved comparison figure: {output_filename}")

# Display
plt.show()

print()
print("="*80)
print("SIMULATION COMPLETE")
print("="*80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("Generated files:")
for phantom_config in phantoms:
    print(f"  - {phantom_config['results_name']}_512x512x4.raw")
print(f"  - {output_filename}")
print()
