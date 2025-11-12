#!/usr/bin/env python
"""
XCAT Technique Comparison - mA and kVp Variation with Scatter

This script performs systematic CT technique comparisons using a non-contrast
XCAT phantom. Two sets of simulations are performed:

SET 1: mA Variation (Constant kVp = 120)
  - Low Dose:    200 mA
  - Medium Dose: 400 mA
  - High Dose:   800 mA

SET 2: kVp Variation (Constant mA = 400)
  - Low Energy:    80 kVp
  - Medium Energy: 100 kVp
  - High Energy:   120 kVp

All simulations include Compton scatter modeling using the convolution-based
scatter model. Results are displayed side-by-side with complete scan parameters.

Phantom: Adult Female 50th percentile chest (non-contrast)
Scatter Model: Scatter_ConvolutionModel (Compton scatter)

Author: Molloi Lab
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

print("="*80)
print("XCAT TECHNIQUE COMPARISON - mA AND kVp VARIATION WITH SCATTER")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# CONFIGURATION - SET 1: mA VARIATION (CONSTANT kVp = 120)
# ============================================================================
print("="*80)
print("CONFIGURATION - SET 1: mA VARIATION")
print("="*80)
print()

ma_variations = [
    {
        'name': 'Low Dose (200 mA)',
        'ma': 200,
        'kvp': 120,
        'spectrum_file': 'tungsten_tar7.0_120_filt.dat',
        'results_name': 'xcat_120kVp_200mA_scatter'
    },
    {
        'name': 'Medium Dose (400 mA)',
        'ma': 400,
        'kvp': 120,
        'spectrum_file': 'tungsten_tar7.0_120_filt.dat',
        'results_name': 'xcat_120kVp_400mA_scatter'
    },
    {
        'name': 'High Dose (800 mA)',
        'ma': 800,
        'kvp': 120,
        'spectrum_file': 'tungsten_tar7.0_120_filt.dat',
        'results_name': 'xcat_120kVp_800mA_scatter'
    }
]

# ============================================================================
# CONFIGURATION - SET 2: kVp VARIATION (CONSTANT mA = 400)
# ============================================================================
print("="*80)
print("CONFIGURATION - SET 2: kVp VARIATION")
print("="*80)
print()

kvp_variations = [
    {
        'name': 'Low Energy (80 kVp)',
        'ma': 400,
        'kvp': 80,
        'spectrum_file': 'tungsten_tar7.0_80_filt.dat',
        'results_name': 'xcat_80kVp_400mA_scatter'
    },
    {
        'name': 'Medium Energy (100 kVp)',
        'ma': 400,
        'kvp': 100,
        'spectrum_file': 'tungsten_tar7.0_100_filt.dat',
        'results_name': 'xcat_100kVp_400mA_scatter'
    },
    {
        'name': 'High Energy (120 kVp)',
        'ma': 400,
        'kvp': 120,
        'spectrum_file': 'tungsten_tar7.0_120_filt.dat',
        'results_name': 'xcat_120kVp_400mA_scatter'
    }
]

# ============================================================================
# COMMON PARAMETERS
# ============================================================================
phantom_file = 'Adult_Female_50percentile_Chest_Phantom_slab_400_1650x1050x1/adult_female_non_contrast.json'

common_params = {
    'fov_mm': 450.0,
    'image_size': 512,
    'slice_count': 4,
    'slice_thickness_mm': 0.568,
    'views_per_rotation': 500,
}

# Storage for reconstructed images
ma_images = []
kvp_images = []

# ============================================================================
# FUNCTION TO RUN SIMULATION
# ============================================================================
def run_simulation(config, phantom_file, common_params, simulation_type):
    """
    Run a complete CT simulation with specified parameters.

    Args:
        config: Dictionary with simulation parameters (ma, kvp, spectrum, etc.)
        phantom_file: Path to phantom JSON file
        common_params: Common scan parameters
        simulation_type: 'mA' or 'kVp' for labeling

    Returns:
        Dictionary with reconstructed image and metadata
    """
    print("="*80)
    print(f"SIMULATION: {config['name']}")
    print(f"Type: {simulation_type} variation")
    print("="*80)
    print()

    # Initialize CatSim
    ct = xc.CatSim()

    # ========================================================================
    # PHANTOM CONFIGURATION
    # ========================================================================
    print("1. PHANTOM CONFIGURATION:")
    ct.phantom.callback = "Phantom_Voxelized"
    ct.phantom.projectorCallback = "C_Projector_Voxelized"
    ct.phantom.filename = phantom_file
    ct.phantom.centerOffset = [0.0, 0.0, 0.0]
    ct.phantom.scale = 1.0
    print(f"   Type:              Voxelized XCAT (Non-Contrast)")
    print(f"   File:              {phantom_file}")
    print(f"   Materials:         9 tissues (ncat_blood - no iodine)")
    print()

    # ========================================================================
    # SCANNER HARDWARE
    # ========================================================================
    print("2. SCANNER HARDWARE:")
    ct.scanner.detectorCallback = "Detector_ThirdgenCurved"
    ct.scanner.sid = 540.0                    # Source-to-isocenter distance (mm)
    ct.scanner.sdd = 950.0                    # Source-to-detector distance (mm)
    ct.scanner.detectorColCount = 900         # Number of detector columns
    ct.scanner.detectorRowCount = 16          # Number of detector rows
    ct.scanner.detectorColSize = 1.0          # Column pitch (mm)
    ct.scanner.detectorRowSize = 1.0          # Row pitch (mm)
    ct.scanner.detectorColOffset = 0.25       # Column offset
    ct.scanner.detectorRowOffset = 0.0        # Row offset
    ct.scanner.detectorMaterial = "Lumex"     # Scintillator material
    ct.scanner.detectorDepth = 3.0            # Detector depth (mm)
    ct.scanner.detectorPrefilter = ['graphite', 1.0]  # Prefilter
    ct.scanner.detectionCallback = "Detection_EI"     # Energy-integrating
    ct.scanner.detectionGain = 15.0           # Conversion gain (e-/keV)
    ct.scanner.detectorColFillFraction = 0.9  # Active area fraction
    ct.scanner.detectorRowFillFraction = 0.9  # Active area fraction
    ct.scanner.eNoise = 5000.0                # Electronic noise (e-)
    print(f"   Geometry:          SID={ct.scanner.sid}mm, SDD={ct.scanner.sdd}mm")
    print(f"   Detector Array:    {ct.scanner.detectorColCount}×{ct.scanner.detectorRowCount}")
    print(f"   Pixel Size:        {ct.scanner.detectorColSize}×{ct.scanner.detectorRowSize} mm")
    print(f"   Material:          {ct.scanner.detectorMaterial} ({ct.scanner.detectorDepth} mm)")
    print(f"   Detection:         {ct.scanner.detectionCallback}, Gain={ct.scanner.detectionGain} e-/keV")
    print()

    # ========================================================================
    # SCAN PROTOCOL
    # ========================================================================
    print("3. SCAN PROTOCOL:")
    ct.protocol.scanTrajectory = "Gantry_Helical"
    ct.protocol.mA = config['ma']             # Tube current (THIS VARIES)
    ct.protocol.dutyRatio = 1.0               # Continuous exposure
    ct.protocol.spectrumFilename = config['spectrum_file']  # kVp spectrum (THIS VARIES)
    ct.protocol.viewsPerRotation = common_params['views_per_rotation']
    ct.protocol.viewCount = common_params['views_per_rotation']
    ct.protocol.startViewId = 0
    ct.protocol.stopViewId = ct.protocol.viewCount - 1
    ct.protocol.rotationTime = 1.0            # Rotation period (sec)
    ct.protocol.rotationDirection = 1         # Clockwise
    ct.protocol.startAngle = 0                # Starting angle (degrees)
    ct.protocol.tiltAngle = 0                 # Gantry tilt (degrees)
    ct.protocol.tableSpeed = 0                # Table speed (mm/sec) - 0 for axial
    ct.protocol.startZ = 0                    # Starting z-position (mm)
    ct.protocol.wobbleDistance = 0.0          # Flying focal spot distance (mm)
    ct.protocol.focalspotOffset = [0, 0, 0]   # Focal spot offset (mm)
    ct.protocol.maxPrep = 9                   # Upper limit of prep
    ct.protocol.airViewCount = 1              # Air scan views
    ct.protocol.offsetViewCount = 1           # Offset scan views
    print(f"   >>> Technique:     {config['kvp']} kVp, {config['ma']} mA <<<")
    print(f"   Spectrum File:     {config['spectrum_file']}")
    print(f"   Trajectory:        {ct.protocol.scanTrajectory}")
    print(f"   Views:             {ct.protocol.viewCount} ({ct.protocol.viewsPerRotation}/rotation)")
    print(f"   Rotation Time:     {ct.protocol.rotationTime} sec")
    print(f"   Table Speed:       {ct.protocol.tableSpeed} mm/sec (axial)")
    print()

    # ========================================================================
    # PHYSICS SIMULATION (WITH SCATTER)
    # ========================================================================
    print("4. PHYSICS SIMULATION:")

    # Energy sampling
    ct.physics.energyCount = 20               # Number of energy bins
    ct.physics.monochromatic = -1             # -1 = polychromatic

    # Spatial sampling (optimized for stability with voxelized phantom)
    ct.physics.colSampleCount = 2             # Detector column samples
    ct.physics.rowSampleCount = 1             # Detector row samples
    ct.physics.srcXSampleCount = 2            # Focal spot lateral samples
    ct.physics.srcYSampleCount = 1            # Focal spot longitudinal samples
    ct.physics.viewSampleCount = 1            # View angle samples

    # *** SCATTER MODELING - COMPTON SCATTER ENABLED ***
    ct.physics.scatterCallback = "Scatter_ConvolutionModel"  # Convolution-based scatter
    ct.physics.scatterKernelCallback = ""                    # Default kernel
    ct.physics.scatterScaleFactor = 1                        # Scale factor

    # Noise modeling
    ct.physics.enableQuantumNoise = 1         # Enable quantum (Poisson) noise
    ct.physics.enableElectronicNoise = 1      # Enable electronic noise

    # Recalculation flags
    ct.physics.recalcDet = 0                  # Detector geometry
    ct.physics.recalcSrc = 0                  # Source geometry
    ct.physics.recalcGantry = 2               # Gantry model (per subview)
    ct.physics.recalcRayAngle = 0             # Ray angles
    ct.physics.recalcSpec = 0                 # Spectrum
    ct.physics.recalcFilt = 0                 # Filters
    ct.physics.recalcFlux = 0                 # Flux
    ct.physics.recalcPht = 0                  # Phantom

    # Callbacks
    ct.physics.rayAngleCallback = "Detector_RayAngles_2D"
    ct.physics.fluxCallback = "Detection_Flux"
    ct.physics.prefilterCallback = "Detection_prefilter"
    ct.physics.DASCallback = "Detection_DAS"
    ct.physics.outputCallback = "WriteRawView"
    ct.physics.dump_period = 100              # Save every 100 views

    print(f"   Energy Bins:       {ct.physics.energyCount}")
    print(f"   Sampling:          Det {ct.physics.colSampleCount}×{ct.physics.rowSampleCount}, " +
          f"Focal {ct.physics.srcXSampleCount}×{ct.physics.srcYSampleCount}, View {ct.physics.viewSampleCount}")
    print(f"   >>> SCATTER:       {ct.physics.scatterCallback} (Compton) <<<")
    print(f"   Scatter Kernel:    Default")
    print(f"   Scatter Scale:     {ct.physics.scatterScaleFactor}")
    print(f"   Noise:             Quantum + Electronic")
    print()

    # ========================================================================
    # RECONSTRUCTION
    # ========================================================================
    print("5. RECONSTRUCTION:")
    ct.recon.fov = common_params['fov_mm']                      # Field of view (mm)
    ct.recon.imageSize = common_params['image_size']            # Matrix size
    ct.recon.sliceCount = common_params['slice_count']          # Number of slices
    ct.recon.sliceThickness = common_params['slice_thickness_mm']  # Slice spacing (mm)
    ct.recon.centerOffset = [0.0, 0.0, 0.0]                     # Center offset (mm)
    ct.recon.reconType = 'fdk_equiAngle'                        # FDK algorithm
    ct.recon.kernelType = 'standard'                            # Reconstruction kernel
    ct.recon.startAngle = 0                                     # Starting angle
    ct.recon.unit = 'HU'                                        # Output units
    ct.recon.mu = 0.02                                          # Water attenuation (/mm)
    ct.recon.huOffset = -1000                                   # HU offset

    pixel_size = ct.recon.fov / ct.recon.imageSize

    print(f"   Algorithm:         {ct.recon.reconType}")
    print(f"   Kernel:            {ct.recon.kernelType}")
    print(f"   Matrix:            {ct.recon.imageSize}×{ct.recon.imageSize}×{ct.recon.sliceCount}")
    print(f"   FOV:               {ct.recon.fov} mm")
    print(f"   Pixel Size:        {pixel_size:.3f} mm")
    print(f"   Slice Thickness:   {ct.recon.sliceThickness} mm")
    print(f"   Units:             {ct.recon.unit}")
    print()

    # ========================================================================
    # OUTPUT
    # ========================================================================
    ct.resultsName = config['results_name']
    ct.do_Recon = 1
    print(f"6. OUTPUT:")
    print(f"   Results Name:      {ct.resultsName}")
    print()

    # ========================================================================
    # RUN SIMULATION
    # ========================================================================
    print("-"*80)
    print(f"RUNNING SIMULATION: {config['name']}")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print("-"*80)
    ct.run_all()
    print(f"✓ Simulation complete: {datetime.now().strftime('%H:%M:%S')}")
    print()

    # ========================================================================
    # RECONSTRUCTION
    # ========================================================================
    print("-"*80)
    print(f"RECONSTRUCTING: {config['name']}")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print("-"*80)
    recon.recon(ct)
    print(f"✓ Reconstruction complete: {datetime.now().strftime('%H:%M:%S')}")
    print()

    # ========================================================================
    # LOAD RECONSTRUCTED IMAGE
    # ========================================================================
    imgFname = f"{ct.resultsName}_{ct.recon.imageSize}x{ct.recon.imageSize}x{ct.recon.sliceCount}.raw"
    img = xc.rawread(imgFname, [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize], 'float')

    middle_slice = np.int32(ct.recon.sliceCount / 2)

    return {
        'image': img[middle_slice, :, :],
        'name': config['name'],
        'ma': config['ma'],
        'kvp': config['kvp'],
        'params': {
            'matrix': f"{ct.recon.imageSize}×{ct.recon.imageSize}",
            'fov': ct.recon.fov,
            'pixel_size': pixel_size,
            'slice_thickness': ct.recon.sliceThickness,
            'scatter': 'ON (Compton)',
        }
    }

# ============================================================================
# RUN ALL SIMULATIONS
# ============================================================================

print("\n" + "="*80)
print("RUNNING SET 1: mA VARIATION (120 kVp CONSTANT)")
print("="*80 + "\n")

for config in ma_variations:
    result = run_simulation(config, phantom_file, common_params, 'mA')
    ma_images.append(result)

print("\n" + "="*80)
print("RUNNING SET 2: kVp VARIATION (400 mA CONSTANT)")
print("="*80 + "\n")

for config in kvp_variations:
    result = run_simulation(config, phantom_file, common_params, 'kVp')
    kvp_images.append(result)

# ============================================================================
# DISPLAY RESULTS - SET 1: mA VARIATION
# ============================================================================
print("\n" + "="*80)
print("CREATING COMPARISON FIGURES")
print("="*80 + "\n")

# Figure 1: mA Variation
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
fig1.suptitle('XCAT Phantom - mA Variation (120 kVp, Scatter ON)',
              fontsize=16, fontweight='bold')

for i, img_data in enumerate(ma_images):
    ax = axes1[i]
    im = ax.imshow(img_data['image'], cmap='gray', vmin=-1000, vmax=200)

    ax.set_title(img_data['name'], fontsize=14, fontweight='bold', pad=10)

    params = img_data['params']
    param_text = (
        f"kVp: {img_data['kvp']}\n"
        f"mA: {img_data['ma']}\n"
        f"Matrix: {params['matrix']}\n"
        f"FOV: {params['fov']:.0f} mm\n"
        f"Pixel: {params['pixel_size']:.3f} mm\n"
        f"Slice: {params['slice_thickness']:.3f} mm\n"
        f"Scatter: {params['scatter']}"
    )

    ax.text(0.02, 0.98, param_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            color='white',
            family='monospace')

    ax.axis('off')

cbar1 = fig1.colorbar(im, ax=axes1, orientation='horizontal', pad=0.02, fraction=0.046)
cbar1.set_label('HU (Hounsfield Units)', fontsize=12)
plt.tight_layout()

output_file1 = 'xcat_mA_comparison_scatter.png'
plt.savefig(output_file1, dpi=150, bbox_inches='tight')
print(f"✓ Saved mA comparison: {output_file1}")

# ============================================================================
# DISPLAY RESULTS - SET 2: kVp VARIATION
# ============================================================================

# Figure 2: kVp Variation
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('XCAT Phantom - kVp Variation (400 mA, Scatter ON)',
              fontsize=16, fontweight='bold')

for i, img_data in enumerate(kvp_images):
    ax = axes2[i]
    im = ax.imshow(img_data['image'], cmap='gray', vmin=-1000, vmax=200)

    ax.set_title(img_data['name'], fontsize=14, fontweight='bold', pad=10)

    params = img_data['params']
    param_text = (
        f"kVp: {img_data['kvp']}\n"
        f"mA: {img_data['ma']}\n"
        f"Matrix: {params['matrix']}\n"
        f"FOV: {params['fov']:.0f} mm\n"
        f"Pixel: {params['pixel_size']:.3f} mm\n"
        f"Slice: {params['slice_thickness']:.3f} mm\n"
        f"Scatter: {params['scatter']}"
    )

    ax.text(0.02, 0.98, param_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            color='white',
            family='monospace')

    ax.axis('off')

cbar2 = fig2.colorbar(im, ax=axes2, orientation='horizontal', pad=0.02, fraction=0.046)
cbar2.set_label('HU (Hounsfield Units)', fontsize=12)
plt.tight_layout()

output_file2 = 'xcat_kVp_comparison_scatter.png'
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"✓ Saved kVp comparison: {output_file2}")

plt.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ALL SIMULATIONS COMPLETE")
print("="*80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("Generated Files:")
print("\nmA Variation (120 kVp):")
for config in ma_variations:
    print(f"  - {config['results_name']}_512x512x4.raw")
print("\nkVp Variation (400 mA):")
for config in kvp_variations:
    print(f"  - {config['results_name']}_512x512x4.raw")
print("\nComparison Figures:")
print(f"  - {output_file1}")
print(f"  - {output_file2}")
print()
