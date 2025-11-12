#!/usr/bin/env python
"""
XCAT Complete CT Simulation - Comprehensive Example

This script provides a complete, self-documenting CT simulation workflow using
a voxelized XCAT phantom. Every parameter is explicitly defined and explained.

Components covered:
1. Phantom (Voxelized XCAT adult female chest - 9 materials)
2. Scanner Hardware (geometry, detector, focal spot)
3. X-ray Spectrum and Beam Filtration
4. Scan Protocol (trajectory, views, timing)
5. Physics Simulation (sampling, noise, scatter)
6. Reconstruction (FDK algorithm)
7. Visualization

Phantom: Adult Female 50th percentile chest, 1650×1050×1 voxels (0.25mm res)
Materials: water, muscle, lung, dry_spine, dry_rib, adipose, blood, heart, cartilage

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

print("="*80)
print("XCAT CT SIMULATION - COMPREHENSIVE WORKFLOW")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# INITIALIZE CATSIM WITH DEFAULTS
# ============================================================================
print("Initializing CatSim with default configuration...")
ct = xc.CatSim()
print("✓ Default configuration loaded")
print()

# ============================================================================
# SECTION 1: PHANTOM CONFIGURATION
# ============================================================================
print("-" * 80)
print("SECTION 1: PHANTOM CONFIGURATION (VOXELIZED XCAT)")
print("-" * 80)

# Phantom Type and Model
ct.phantom.callback = "Phantom_Voxelized"              # Function to read voxelized phantom
ct.phantom.projectorCallback = "C_Projector_Voxelized" # Projection algorithm for voxelized data

# Phantom File Selection
# Using voxelized XCAT adult female chest phantom
# The JSON file references 9 material density files (ncat_water, muscle, lung, etc.)
# Phantom dimensions: 1650 × 1050 × 1 voxels, 0.25mm × 0.25mm × 50mm voxels
ct.phantom.filename = 'Adult_Female_50percentile_Chest_Phantom_slab_400_1650x1050x1/adult_female_non_contrast.json'

# Phantom Positioning and Scaling
ct.phantom.centerOffset = [0.0, 0.0, 0.0]  # Offset from isocenter [x, y, z] in mm
ct.phantom.scale = 1.0                      # Scaling factor (1.0 = actual size)

print(f"  Phantom Type:     {ct.phantom.callback}")
print(f"  Projector:        {ct.phantom.projectorCallback}")
print(f"  Phantom File:     {ct.phantom.filename}")
print(f"  Center Offset:    {ct.phantom.centerOffset} mm")
print(f"  Scale Factor:     {ct.phantom.scale}x")
print()

# ============================================================================
# SECTION 2: SCANNER HARDWARE CONFIGURATION
# ============================================================================
print("-" * 80)
print("SECTION 2: SCANNER HARDWARE")
print("-" * 80)

# --- Scanner Geometry ---
print("2.1 Scanner Geometry:")
ct.scanner.detectorCallback = "Detector_ThirdgenCurved"  # Curved detector array
ct.scanner.sid = 540.0  # Source-to-isocenter distance (mm)
ct.scanner.sdd = 950.0  # Source-to-detector distance (mm)
print(f"  Detector Type:    {ct.scanner.detectorCallback}")
print(f"  SID:              {ct.scanner.sid} mm")
print(f"  SDD:              {ct.scanner.sdd} mm")
print()

# --- Detector Array ---
print("2.2 Detector Array:")
ct.scanner.detectorColsPerMod = 1    # Columns per module
ct.scanner.detectorRowsPerMod = 16   # Rows per module
ct.scanner.detectorColCount = 900    # Total columns
ct.scanner.detectorRowCount = ct.scanner.detectorRowsPerMod  # Total rows
ct.scanner.detectorColSize = 1.0     # Column pitch (mm)
ct.scanner.detectorRowSize = 1.0     # Row pitch (mm)
ct.scanner.detectorColOffset = 0.25  # Column offset (in detector columns)
ct.scanner.detectorRowOffset = 0.0   # Row offset (in detector rows)
print(f"  Array Size:       {ct.scanner.detectorColCount} cols × {ct.scanner.detectorRowCount} rows")
print(f"  Pixel Size:       {ct.scanner.detectorColSize} × {ct.scanner.detectorRowSize} mm")
print(f"  Modules:          {ct.scanner.detectorColsPerMod} × {ct.scanner.detectorRowsPerMod}")
print(f"  Offset:           col={ct.scanner.detectorColOffset}, row={ct.scanner.detectorRowOffset}")
print()

# --- Detector Material and Response ---
print("2.3 Detector Material & Response:")
ct.scanner.detectorMaterial = "Lumex"        # Scintillator material
ct.scanner.detectorDepth = 3.0               # Sensor depth (mm)
ct.scanner.detectorPrefilter = ['graphite', 1.0]  # Prefilter [material, thickness_mm]
ct.scanner.detectionCallback = "Detection_EI"     # Energy-integrating detector
ct.scanner.detectionGain = 15.0              # Conversion factor (electrons/keV)
ct.scanner.detectorColFillFraction = 0.9     # Active area fraction (column direction)
ct.scanner.detectorRowFillFraction = 0.9     # Active area fraction (row direction)
ct.scanner.eNoise = 5000.0                   # Electronic noise std dev (electrons)
print(f"  Material:         {ct.scanner.detectorMaterial}")
print(f"  Depth:            {ct.scanner.detectorDepth} mm")
print(f"  Prefilter:        {ct.scanner.detectorPrefilter[0]} {ct.scanner.detectorPrefilter[1]} mm")
print(f"  Detection Type:   {ct.scanner.detectionCallback}")
print(f"  Gain:             {ct.scanner.detectionGain} e-/keV")
print(f"  Fill Fraction:    {ct.scanner.detectorColFillFraction} × {ct.scanner.detectorRowFillFraction}")
print(f"  Electronic Noise: {ct.scanner.eNoise} e- (σ)")
print()

# --- Focal Spot ---
print("2.4 Focal Spot:")
ct.scanner.focalspotCallback = "SetFocalspot"  # Focal spot model
ct.scanner.focalspotShape = "Uniform"          # Shape: "Uniform" or "Gaussian"
ct.scanner.targetAngle = 7.0                   # Target angle (degrees)
ct.scanner.focalspotWidth = 1.0                # Width (mm)
ct.scanner.focalspotLength = 1.0               # Length (mm)
print(f"  Callback:         {ct.scanner.focalspotCallback}")
print(f"  Shape:            {ct.scanner.focalspotShape}")
print(f"  Size:             {ct.scanner.focalspotWidth} × {ct.scanner.focalspotLength} mm")
print(f"  Target Angle:     {ct.scanner.targetAngle}°")
print()

# ============================================================================
# SECTION 3: X-RAY SPECTRUM AND BEAM CONFIGURATION
# ============================================================================
print("-" * 80)
print("SECTION 3: X-RAY SPECTRUM AND BEAM")
print("-" * 80)

# --- X-ray Technique ---
print("3.1 X-ray Technique:")
ct.protocol.mA = 800                  # Tube current (mA)
ct.protocol.dutyRatio = 1.0           # Tube ON time fraction (1.0 = continuous)
print(f"  Tube Current:     {ct.protocol.mA} mA")
print(f"  Duty Ratio:       {ct.protocol.dutyRatio}")
print()

# --- Spectrum ---
print("3.2 Spectrum:")
ct.protocol.spectrumCallback = "Spectrum"
# Available spectra:
#   - tungsten_tar7.0_80_filt.dat   : 80 kVp
#   - tungsten_tar7.0_120_filt.dat  : 120 kVp
#   - tungsten_tar7.0_140_filt.dat  : 140 kVp
ct.protocol.spectrumFilename = "tungsten_tar7.0_120_filt.dat"
ct.protocol.spectrumScaling = 1       # Scaling factor
ct.protocol.spectrumUnit_mm = 0       # Spectrum in photons/sec/mm²/<current>?
ct.protocol.spectrumUnit_mA = 1       # Spectrum in photons/sec/<area>/mA?
print(f"  Callback:         {ct.protocol.spectrumCallback}")
print(f"  Spectrum File:    {ct.protocol.spectrumFilename}")
print(f"  Scaling:          {ct.protocol.spectrumScaling}")
print(f"  Unit (mm):        {bool(ct.protocol.spectrumUnit_mm)}")
print(f"  Unit (mA):        {bool(ct.protocol.spectrumUnit_mA)}")
print()

# --- Beam Filtration ---
print("3.3 Beam Filtration:")
ct.protocol.filterCallback = "Xray_Filter"
# Bowtie filter options: "large.txt", "medium.txt", "small.txt", or [] for none
ct.protocol.bowtie = "large.txt"
# Additional flat filtration: [material, thickness_mm]
ct.protocol.flatFilter = ['Al', 3.0]
print(f"  Filter Callback:  {ct.protocol.filterCallback}")
print(f"  Bowtie Filter:    {ct.protocol.bowtie}")
print(f"  Flat Filter:      {ct.protocol.flatFilter[0]} {ct.protocol.flatFilter[1]} mm")
print()

# ============================================================================
# SECTION 4: SCAN PROTOCOL
# ============================================================================
print("-" * 80)
print("SECTION 4: SCAN PROTOCOL")
print("-" * 80)

# --- Scan Types ---
print("4.1 Scan Types:")
# [air_scan, offset_scan, phantom_scan, prep_view]
ct.protocol.scanTypes = [1, 1, 1, 1]
print(f"  Scan Types:       {ct.protocol.scanTypes}")
print(f"                    [air={ct.protocol.scanTypes[0]}, " +
      f"offset={ct.protocol.scanTypes[1]}, " +
      f"phantom={ct.protocol.scanTypes[2]}, " +
      f"prep={ct.protocol.scanTypes[3]}]")
print()

# --- Scan Trajectory ---
print("4.2 Scan Trajectory:")
ct.protocol.scanTrajectory = "Gantry_Helical"  # "Gantry_Helical" or "Gantry_Axial"
ct.protocol.viewsPerRotation = 500    # Views per 360° rotation (500 for faster sim)
ct.protocol.viewCount = 500           # Total views in scan
ct.protocol.startViewId = 0           # Index of first view
ct.protocol.stopViewId = ct.protocol.viewCount - 1  # Index of last view
print(f"  Trajectory:       {ct.protocol.scanTrajectory}")
print(f"  Views/Rotation:   {ct.protocol.viewsPerRotation}")
print(f"  Total Views:      {ct.protocol.viewCount}")
print(f"  View Range:       {ct.protocol.startViewId} to {ct.protocol.stopViewId}")
print()

# --- Reference Scans ---
print("4.3 Reference Scans:")
ct.protocol.airViewCount = 1      # Views averaged for air scan
ct.protocol.offsetViewCount = 1   # Views averaged for offset scan
print(f"  Air Views:        {ct.protocol.airViewCount}")
print(f"  Offset Views:     {ct.protocol.offsetViewCount}")
print()

# --- Gantry Motion ---
print("4.4 Gantry Motion:")
ct.protocol.rotationTime = 1.0         # Rotation period (seconds)
ct.protocol.rotationDirection = 1      # 1=clockwise, -1=counterclockwise
ct.protocol.startAngle = 0             # Starting angle (degrees)
ct.protocol.tiltAngle = 0              # Gantry tilt (degrees)
print(f"  Rotation Time:    {ct.protocol.rotationTime} sec")
print(f"  Direction:        {'CW' if ct.protocol.rotationDirection == 1 else 'CCW'}")
print(f"  Start Angle:      {ct.protocol.startAngle}°")
print(f"  Tilt Angle:       {ct.protocol.tiltAngle}°")
print()

# --- Table Motion ---
print("4.5 Table Motion:")
ct.protocol.tableSpeed = 0             # Table speed (mm/sec) - 0 for axial
ct.protocol.startZ = 0                 # Starting z-position (mm)
print(f"  Table Speed:      {ct.protocol.tableSpeed} mm/sec")
print(f"  Start Z:          {ct.protocol.startZ} mm")
print()

# --- Focal Spot Control ---
print("4.6 Focal Spot Control:")
ct.protocol.wobbleDistance = 0.0       # Flying focal spot distance (mm)
ct.protocol.focalspotOffset = [0, 0, 0]  # Position offset [x, y, z] (mm)
print(f"  Wobble Distance:  {ct.protocol.wobbleDistance} mm")
print(f"  Focal Offset:     {ct.protocol.focalspotOffset} mm")
print()

# --- Preprocessing ---
print("4.7 Preprocessing:")
ct.protocol.maxPrep = 9  # Upper limit of prep (low signal correction)
print(f"  Max Prep:         {ct.protocol.maxPrep}")
print()

# Calculate and display scan parameters
scan_time = (ct.protocol.viewCount / ct.protocol.viewsPerRotation) * ct.protocol.rotationTime
print(f"  >>> Calculated Scan Time: {scan_time:.2f} seconds")
print()

# ============================================================================
# SECTION 5: PHYSICS SIMULATION
# ============================================================================
print("-" * 80)
print("SECTION 5: PHYSICS SIMULATION")
print("-" * 80)

# --- Energy Sampling ---
print("5.1 Energy Sampling:")
ct.physics.energyCount = 20       # Number of energy bins
ct.physics.monochromatic = -1     # -1=polychromatic, or specify energy in keV
print(f"  Energy Bins:      {ct.physics.energyCount}")
print(f"  Monochromatic:    {ct.physics.monochromatic} (-1 = polychromatic)")
print()

# --- Spatial Sampling ---
# NOTE: These sampling parameters are critical for NCAT phantom stability
# Using lower values (from working Sim_Sample_XCAT.py) prevents bus errors
print("5.2 Spatial Sampling:")
ct.physics.colSampleCount = 2      # Detector column samples
ct.physics.rowSampleCount = 1      # Detector row samples (reduced for NCAT stability)
ct.physics.srcXSampleCount = 2     # Focal spot lateral samples
ct.physics.srcYSampleCount = 1     # Focal spot longitudinal samples (reduced for NCAT)
ct.physics.viewSampleCount = 1     # View angle samples (reduced for NCAT)
print(f"  Detector:         {ct.physics.colSampleCount} × {ct.physics.rowSampleCount}")
print(f"  Focal Spot:       {ct.physics.srcXSampleCount} × {ct.physics.srcYSampleCount}")
print(f"  View:             {ct.physics.viewSampleCount}")
print()

# --- Recalculation Flags ---
print("5.3 Recalculation Flags (0=never, 1=per view, 2=per subview):")
ct.physics.recalcDet = 0           # Detector geometry
ct.physics.recalcSrc = 0           # Source geometry
ct.physics.recalcGantry = 2        # Gantry model
ct.physics.recalcRayAngle = 0      # Ray angles
ct.physics.recalcSpec = 0          # Spectrum
ct.physics.recalcFilt = 0          # Filters
ct.physics.recalcFlux = 0          # Flux
ct.physics.recalcPht = 0           # Phantom
print(f"  Detector:         {ct.physics.recalcDet}")
print(f"  Source:           {ct.physics.recalcSrc}")
print(f"  Gantry:           {ct.physics.recalcGantry}")
print(f"  Ray Angles:       {ct.physics.recalcRayAngle}")
print(f"  Spectrum:         {ct.physics.recalcSpec}")
print(f"  Filters:          {ct.physics.recalcFilt}")
print(f"  Flux:             {ct.physics.recalcFlux}")
print(f"  Phantom:          {ct.physics.recalcPht}")
print()

# --- Noise Settings ---
print("5.4 Noise:")
ct.physics.enableQuantumNoise = 1      # Photon counting noise (Poisson)
ct.physics.enableElectronicNoise = 1   # Electronic noise (Gaussian)
print(f"  Quantum Noise:    {'ON' if ct.physics.enableQuantumNoise else 'OFF'}")
print(f"  Electronic Noise: {'ON' if ct.physics.enableElectronicNoise else 'OFF'}")
print()

# --- Physics Models ---
print("5.5 Physics Model Callbacks:")
ct.physics.rayAngleCallback = "Detector_RayAngles_2D"
ct.physics.fluxCallback = "Detection_Flux"
ct.physics.scatterCallback = ""                # "" = no scatter
ct.physics.scatterKernelCallback = ""          # Custom scatter kernel
ct.physics.scatterScaleFactor = 1              # Scatter scaling
ct.physics.prefilterCallback = "Detection_prefilter"
ct.physics.crosstalkCallback = ""              # X-ray crosstalk
ct.physics.lagCallback = ""                    # Detector lag
ct.physics.opticalCrosstalkCallback = ""       # Optical crosstalk
ct.physics.DASCallback = "Detection_DAS"       # Data acquisition
print(f"  Ray Angles:       {ct.physics.rayAngleCallback}")
print(f"  Flux:             {ct.physics.fluxCallback}")
print(f"  Scatter:          {ct.physics.scatterCallback if ct.physics.scatterCallback else 'OFF'}")
print(f"  Prefilter:        {ct.physics.prefilterCallback}")
print(f"  DAS:              {ct.physics.DASCallback}")
print()

# --- I/O Preferences ---
print("5.6 I/O:")
ct.physics.outputCallback = "WriteRawView"
ct.physics.dump_period = 100  # Save data every N views
print(f"  Output Callback:  {ct.physics.outputCallback}")
print(f"  Dump Period:      {ct.physics.dump_period} views")
print()

# ============================================================================
# SECTION 6: RECONSTRUCTION CONFIGURATION
# ============================================================================
print("-" * 80)
print("SECTION 6: RECONSTRUCTION")
print("-" * 80)

# --- Image Geometry ---
print("6.1 Image Geometry:")
# FOV set to 450mm to fully capture the voxelized phantom (1650×0.25mm = 412.5mm width)
ct.recon.fov = 450.0                   # Field of view diameter (mm)
ct.recon.imageSize = 512               # Matrix size (pixels × pixels)
ct.recon.sliceCount = 4                # Number of slices to reconstruct
ct.recon.sliceThickness = 0.568        # Slice spacing (mm)
ct.recon.centerOffset = [0.0, 0.0, 0.0]  # Offset from rotation center [x, y, z] (mm)
pixel_size = ct.recon.fov / ct.recon.imageSize
print(f"  FOV:              {ct.recon.fov} mm")
print(f"  Matrix Size:      {ct.recon.imageSize} × {ct.recon.imageSize}")
print(f"  Pixel Size:       {pixel_size:.3f} mm")
print(f"  Slices:           {ct.recon.sliceCount}")
print(f"  Slice Thickness:  {ct.recon.sliceThickness} mm")
print(f"  Center Offset:    {ct.recon.centerOffset} mm")
print()

# --- Reconstruction Algorithm ---
print("6.2 Algorithm:")
# Options: 'fdk_equiAngle' (circular), 'helical_equiAngle' (helical)
ct.recon.reconType = 'fdk_equiAngle'
# Kernel options: 'R-L', 'S-L', 'standard', 'soft', 'bone'
ct.recon.kernelType = 'standard'
ct.recon.startAngle = 0  # Starting angle (degrees, 0=source at top)
print(f"  Recon Type:       {ct.recon.reconType}")
print(f"  Kernel:           {ct.recon.kernelType}")
print(f"  Start Angle:      {ct.recon.startAngle}°")
print()

# --- Output Units ---
print("6.3 Output Units:")
ct.recon.unit = 'HU'                   # Options: 'HU', '/mm', '/cm'
ct.recon.mu = 0.02                     # Water attenuation (/mm) at spectrum energy
ct.recon.huOffset = -1000              # HU offset (typically -1000)
print(f"  Units:            {ct.recon.unit}")
print(f"  μ (water):        {ct.recon.mu} /mm")
print(f"  HU Offset:        {ct.recon.huOffset}")
print()

# --- Output Options ---
print("6.4 Output Options:")
ct.recon.printReconParameters = False
ct.recon.saveImageVolume = True
ct.recon.saveSingleImages = False
ct.recon.displayImagePictures = False
ct.recon.saveImagePictureFiles = False
print(f"  Print Params:     {ct.recon.printReconParameters}")
print(f"  Save Volume:      {ct.recon.saveImageVolume}")
print(f"  Save Singles:     {ct.recon.saveSingleImages}")
print(f"  Display Images:   {ct.recon.displayImagePictures}")
print()

# ============================================================================
# SECTION 7: OUTPUT CONFIGURATION
# ============================================================================
print("-" * 80)
print("SECTION 7: OUTPUT")
print("-" * 80)
ct.resultsName = "xcat_complete_sim"
ct.do_Recon = 1  # Enable reconstruction
print(f"  Results Name:     {ct.resultsName}")
print(f"  Do Recon:         {'Yes' if ct.do_Recon else 'No'}")
print()

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================
print("="*80)
print("CONFIGURATION SUMMARY")
print("="*80)
print(f"Phantom:      {ct.phantom.filename}")
print(f"Scanner:      {ct.scanner.detectorColCount}×{ct.scanner.detectorRowCount}, " +
      f"SID={ct.scanner.sid}mm, SDD={ct.scanner.sdd}mm")
print(f"Protocol:     {ct.protocol.viewCount} views, {ct.protocol.mA} mA, " +
      f"{ct.protocol.spectrumFilename}")
print(f"Physics:      {ct.physics.energyCount} energy bins, " +
      f"noise={'ON' if ct.physics.enableQuantumNoise else 'OFF'}")
print(f"Recon:        {ct.recon.imageSize}×{ct.recon.imageSize}×{ct.recon.sliceCount}, " +
      f"FOV={ct.recon.fov}mm, {ct.recon.reconType}")
print(f"Output:       {ct.resultsName}.*")
print("="*80)
print()

# ============================================================================
# RUN SIMULATION
# ============================================================================
print("="*80)
print("RUNNING CT SIMULATION")
print("="*80)
print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
print()

try:
    ct.run_all()
    print()
    print("="*80)
    print("✓ SIMULATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    print("Output files generated:")
    print(f"  {ct.resultsName}.air    - Air scan data")
    print(f"  {ct.resultsName}.offset - Offset scan data")
    print(f"  {ct.resultsName}.scan   - Phantom scan data (raw projections)")
    print(f"  {ct.resultsName}.prep   - Preprocessed projections")
    print()
except Exception as e:
    print(f"\n✗ ERROR during simulation: {e}")
    raise

# ============================================================================
# RECONSTRUCTION
# ============================================================================
print("="*80)
print("RUNNING FDK RECONSTRUCTION")
print("="*80)
print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
print()

try:
    recon.recon(ct)
    print()
    print("="*80)
    print("✓ RECONSTRUCTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    img_filename = f"{ct.resultsName}_{ct.recon.imageSize}x{ct.recon.imageSize}x{ct.recon.sliceCount}.raw"
    print(f"Reconstructed volume saved as: {img_filename}")
    print()
except Exception as e:
    print(f"\n✗ ERROR during reconstruction: {e}")
    raise

# ============================================================================
# VISUALIZATION
# ============================================================================
print("="*80)
print("VISUALIZATION")
print("="*80)

# Read reconstructed volume
img_filename = f"{ct.resultsName}_{ct.recon.imageSize}x{ct.recon.imageSize}x{ct.recon.sliceCount}.raw"
img = xc.rawread(img_filename, [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize], 'float')

print(f"Image shape: {img.shape}")
print(f"HU range: [{img.min():.1f}, {img.max():.1f}]")
print(f"Mean HU: {img.mean():.1f}")
print(f"Std HU: {img.std():.1f}")
print()

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f'XCAT CT Simulation Results - {ct.phantom.filename}',
             fontsize=16, fontweight='bold')

# Plot all slices
n_slices = img.shape[0]
for i in range(n_slices):
    ax = plt.subplot(2, 4, i+1)
    im = ax.imshow(img[i, :, :], cmap='gray', vmin=-200, vmax=200)
    ax.set_title(f'Slice {i}\nSoft Tissue Window', fontsize=10)
    ax.axis('off')

# Plot middle slice with different windows
if n_slices > 0:
    middle_idx = n_slices // 2
    middle_slice = img[middle_idx, :, :]

    # Lung window
    ax = plt.subplot(2, 4, 5)
    ax.imshow(middle_slice, cmap='gray', vmin=-1000, vmax=0)
    ax.set_title(f'Slice {middle_idx}\nLung Window', fontsize=10)
    ax.axis('off')

    # Bone window
    ax = plt.subplot(2, 4, 6)
    ax.imshow(middle_slice, cmap='gray', vmin=-200, vmax=1000)
    ax.set_title(f'Slice {middle_idx}\nBone Window', fontsize=10)
    ax.axis('off')

    # Profile plot
    ax = plt.subplot(2, 4, 7)
    center_row = ct.recon.imageSize // 2
    profile = middle_slice[center_row, :]
    ax.plot(profile, 'b-', linewidth=2)
    ax.set_title('Horizontal Profile', fontsize=10)
    ax.set_xlabel('Column')
    ax.set_ylabel('HU')
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = plt.subplot(2, 4, 8)
    ax.hist(middle_slice.flatten(), bins=100, color='steelblue', alpha=0.7)
    ax.set_title('HU Histogram', fontsize=10)
    ax.set_xlabel('HU')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{ct.resultsName}_results.png', dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved as: {ct.resultsName}_results.png")
plt.show()

# ============================================================================
# COMPLETION
# ============================================================================
print()
print("="*80)
print("SIMULATION COMPLETE")
print("="*80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("All output files:")
print(f"  {ct.resultsName}.air")
print(f"  {ct.resultsName}.offset")
print(f"  {ct.resultsName}.scan")
print(f"  {ct.resultsName}.prep")
print(f"  {img_filename}")
print(f"  {ct.resultsName}_results.png")
print("="*80)
