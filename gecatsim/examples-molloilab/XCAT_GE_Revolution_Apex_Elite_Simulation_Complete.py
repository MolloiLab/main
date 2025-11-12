#!/usr/bin/env python
"""
XCAT GE Revolution Apex Elite CT Simulation - Hyper-Realistic Example

This script provides a complete, self-documenting CT simulation workflow
using a voxelized XCAT phantom. Every parameter is explicitly defined and
explained.

This script has been modified to use the Ground Truth (GT) hardware
parameters for the GE Revolution Apex Elite (K213715), GE's flagship
energy-integrating detector (EID) scanner.

============================================================================
GE REVOLUTION APEX ELITE - GROUND TRUTH PARAMETERS
============================================================================

This script has been modified with hyper-realistic parameters based on
publicly available technical data and FDA 510(k) filings for the
GE Revolution Apex Elite (K213715).

| Parameter | Ground Truth Value | Source URL (or derivation) |
| :--- | :--- | :--- |
| Scanner Model | Revolution Apex Elite | https://pmc.ncbi.nlm.nih.gov/articles/PMC10332658/ |
| FDA 510(k) | K213715 | https://www.accessdata.fda.gov/cdrh_docs/pdf21/K213715.pdf |
| Gantry Aperture | 80 cm | https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133705.pdf |
| Scan FOV (SFOV) | 50 cm | https://www.gehealthcare.com/-/jssmedia/gehc/us/images/products/goldseal/goldseal-ct-redesign/sell-sheet-goldseal-revolution-ct-ex160-us-jb02387xx_v2.pdf |
| SID | 62.6 cm (626.0 mm) | https://www.gehealthcare.com/-/jssmedia/gehc/us/images/products/goldseal/goldseal-ct-redesign/sell-sheet-goldseal-revolution-ct-ex160-us-jb02387xx_v2.pdf |
| SDD | 109.7 cm (1097.0 mm) | https://www.gehealthcare.com/-/jssmedia/gehc/us/images/products/goldseal/goldseal-ct-redesign/sell-sheet-goldseal-revolution-ct-ex160-us-jb02387xx_v2.pdf |
| Detector Model | Gemstone Clarity | https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133705.pdf |
| Detector Rows | 256 | https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133705.pdf |
| Detector Row Size | 0.625 mm | https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133705.pdf |
| Z-Coverage | 160 mm | https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133705.pdf |
| Total Detector Cells | 212,992 | https://www.gehealthcare.com/-/jssmedia/gehc/us/images/products/goldseal/goldseal-ct-redesign/sell-sheet-goldseal-revolution-ct-ex160-us-jb02387xx_v2.pdf |
| Detector Columns | 832 | (Derived: 212,992 cells / 256 rows) |
| Tube Model | Quantix 160 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10332658/ |
| Generator Power | 108 kW | https://pmc.ncbi.nlm.nih.gov/articles/PMC10332658/ |
| Target Anode Angle | 10 degrees | https://pmc.ncbi.nlm.nih.gov/articles/PMC10332658/ |
| kVp Options | 70, 80, 100, 120, 140 kV | https://www.gehealthcare.com/-/jssmedia/gehc/us/images/products/goldseal/goldseal-ct-redesign/sell-sheet-goldseal-revolution-ct-ex160-us-jb02387xx_v2.pdf |
| Max mA | 1300 mA (at 70/80 kV) | https://www.gehealthcare.com/-/jssmedia/global/products/images/revolution-apex-platform/quantix_whitepaper_jb78157xx.pdf |
| Focal Spot Sizes | 1.0x0.7, 1.6x1.2, 2.0x1.2 mm | https://www.gehealthcare.com/-/jssmedia/gehc/us/images/products/goldseal/goldseal-ct-redesign/sell-sheet-goldseal-revolution-ct-ex160-us-jb02387xx_v2.pdf |
| Focal Spot Control | Magnetic Wobble | https://www.gehealthcare.com/-/jssmedia/global/products/images/revolution-apex-platform/quantix_whitepaper_jb78157xx.pdf |
| Rotation Times | 0.23, 0.28, 0.35... 1.0 s | https://www.accessdata.fda.gov/cdrh_docs/pdf21/K213715.pdf |
| Helical Pitch Values | 0.5, 0.992, 1.375, 1.531 | https://ajronline.org/doi/pdf/10.2214/AJR.18.19851 |
| Views per Rotation | Up to 2,496 | https://info.ncdhhs.gov/dhsr/coneed/reviews/2020/dec/3455-Cabarrus-Carolinas-HealthCare-System-Imaging-Kannapolis-061206-Exemption.pdf |

============================================================================
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
print("XCAT CT SIMULATION - GE REVOLUTION APEX ELITE (HYPER-REALISTIC)")
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
ct.phantom.callback = "Phantom_Voxelized"             # Function to read voxelized phantom
ct.phantom.projectorCallback = "C_Projector_Voxelized" # Projection algorithm for voxelized data

# Phantom File Selection
# Using voxelized XCAT adult female chest phantom
# The JSON file references 9 material density files (ncat_water, muscle, lung, etc.)
# Phantom dimensions: 1650 × 1050 × 1 voxels, 0.25mm × 0.25mm × 50mm voxels
ct.phantom.filename = 'Adult_Female_50percentile_Chest_Phantom_slab_400_1650x1050x1/adult_female_non_contrast.json' # estimate

# Phantom Positioning and Scaling
ct.phantom.centerOffset = [0.0, 0.0, 0.0]  # Offset from isocenter [x, y, z] in mm
ct.phantom.scale = 1.0                     # Scaling factor (1.0 = actual size)

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
print("SECTION 2: SCANNER HARDWARE (GE REVOLUTION APEX ELITE)")
print("-" * 80)

# --- Scanner Geometry ---
print("2.1 Scanner Geometry:")
ct.scanner.detectorCallback = "Detector_ThirdgenCurved"  # Curved detector array # GT
ct.scanner.sid = 626.0  # Source-to-isocenter distance (mm) # GT
ct.scanner.sdd = 1097.0 # Source-to-detector distance (mm) # GT
print(f"  Detector Type:    {ct.scanner.detectorCallback}")
print(f"  SID:              {ct.scanner.sid} mm")
print(f"  SDD:              {ct.scanner.sdd} mm")
print()

# --- Detector Array ---
print("2.2 Detector Array (Gemstone Clarity):")
ct.scanner.detectorColsPerMod = 1     # Columns per module # estimate
ct.scanner.detectorRowsPerMod = 256   # Rows per module # GT
ct.scanner.detectorColCount = 832     # Total columns # GT (Derived)
ct.scanner.detectorRowCount = ct.scanner.detectorRowsPerMod  # Total rows
ct.scanner.detectorColSize = 1.053    # Column pitch (mm) # estimate (Derived to cover 500mm SFOV)
ct.scanner.detectorRowSize = 0.625    # Row pitch (mm) # GT
ct.scanner.detectorColOffset = 0.25   # Column offset (in detector columns) # estimate (Standard quarter-detector offset)
ct.scanner.detectorRowOffset = 0.0    # Row offset (in detector rows) # estimate
print(f"  Array Size:         {ct.scanner.detectorColCount} cols × {ct.scanner.detectorRowCount} rows")
print(f"  Pixel Size:         {ct.scanner.detectorColSize:.3f} × {ct.scanner.detectorRowSize} mm")
print(f"  Modules:            {ct.scanner.detectorColsPerMod} × {ct.scanner.detectorRowsPerMod}")
print(f"  Offset:             col={ct.scanner.detectorColOffset}, row={ct.scanner.detectorRowOffset}")
print()

# --- Detector Material and Response ---
print("2.3 Detector Material & Response:")
ct.scanner.detectorMaterial = "Lumex"       # Scintillator material (Gemstone Clarity proxy) # estimate
ct.scanner.detectorDepth = 3.0              # Sensor depth (mm) # estimate
ct.scanner.detectorPrefilter = ['graphite', 1.0]  # Prefilter [material, thickness_mm] # estimate
ct.scanner.detectionCallback = "Detection_EI"     # Energy-integrating detector # GT
ct.scanner.detectionGain = 15.0             # Conversion factor (electrons/keV) # estimate
ct.scanner.detectorColFillFraction = 0.9    # Active area fraction (column direction) # estimate
ct.scanner.detectorRowFillFraction = 0.9    # Active area fraction (row direction) # estimate
ct.scanner.eNoise = 5000.0                  # Electronic noise std dev (electrons) # estimate
print(f"  Material:           {ct.scanner.detectorMaterial}")
print(f"  Depth:              {ct.scanner.detectorDepth} mm")
print(f"  Prefilter:          {ct.scanner.detectorPrefilter[0]} {ct.scanner.detectorPrefilter[1]} mm")
print(f"  Detection Type:     {ct.scanner.detectionCallback}")
print(f"  Gain:               {ct.scanner.detectionGain} e-/keV")
print(f"  Fill Fraction:      {ct.scanner.detectorColFillFraction} × {ct.scanner.detectorRowFillFraction}")
print(f"  Electronic Noise:   {ct.scanner.eNoise} e- (σ)")
print()

# --- Focal Spot ---
print("2.4 Focal Spot (Quantix 160 Tube):")
ct.scanner.focalspotCallback = "SetFocalspot"  # Focal spot model
ct.scanner.focalspotShape = "Uniform"          # Shape: "Uniform" or "Gaussian" # estimate
ct.scanner.targetAngle = 10.0                  # Target angle (degrees) # GT
ct.scanner.focalspotWidth = 1.0                # Width (mm) # GT (from 1.0x0.7)
ct.scanner.focalspotLength = 0.7               # Length (mm) # GT (from 1.0x0.7)
print(f"  Callback:           {ct.scanner.focalspotCallback}")
print(f"  Shape:              {ct.scanner.focalspotShape}")
print(f"  Size:               {ct.scanner.focalspotWidth} × {ct.scanner.focalspotLength} mm")
print(f"  Target Angle:       {ct.scanner.targetAngle}°")
print()

# ============================================================================
# SECTION 3: X-RAY SPECTRUM AND BEAM CONFIGURATION
# ============================================================================
print("-" * 80)
print("SECTION 3: X-RAY SPECTRUM AND BEAM")
print("-" * 80)

# --- X-ray Technique ---
print("3.1 X-ray Technique:")
# Max mA at 120kV is (108,000 W / 120 V) = 900 mA. 800mA is a realistic high-dose scan.
ct.protocol.mA = 800                   # Tube current (mA) # GT (Realistic value for 120kV)
ct.protocol.dutyRatio = 1.0            # Tube ON time fraction (1.0 = continuous) # estimate
print(f"  Tube Current:       {ct.protocol.mA} mA")
print(f"  Duty Ratio:         {ct.protocol.dutyRatio}")
print()

# --- Spectrum ---
print("3.2 Spectrum:")
ct.protocol.spectrumCallback = "Spectrum"
# Available spectra:
#   - tungsten_tar10.0_120_filt.dat : 120 kVp (GT)
#   - tungsten_tar10.0_80_filt.dat  : 80 kVp (GT)
#   - etc.
ct.protocol.spectrumFilename = "tungsten_tar10.0_120_filt.dat" # GT (10.0 degree angle, 120 kVp)
ct.protocol.spectrumScaling = 1        # Scaling factor
ct.protocol.spectrumUnit_mm = 0        # Spectrum in photons/sec/mm²/<current>?
ct.protocol.spectrumUnit_mA = 1        # Spectrum in photons/sec/<area>/mA?
print(f"  Callback:           {ct.protocol.spectrumCallback}")
print(f"  Spectrum File:      {ct.protocol.spectrumFilename}")
print(f"  Scaling:            {ct.protocol.spectrumScaling}")
print(f"  Unit (mm):          {bool(ct.protocol.spectrumUnit_mm)}")
print(f"  Unit (mA):          {bool(ct.protocol.spectrumUnit_mA)}")
print()

# --- Beam Filtration ---
print("3.3 Beam Filtration:")
ct.protocol.filterCallback = "Xray_Filter"
# Bowtie filter options: "large.txt", "medium.txt", "small.txt", or for none
ct.protocol.bowtie = "large.txt" # estimate (Appropriate for chest/body)
# Additional flat filtration: [material, thickness_mm]
ct.protocol.flatFilter = ['Al', 3.0] # estimate
print(f"  Filter Callback:    {ct.protocol.filterCallback}")
print(f"  Bowtie Filter:      {ct.protocol.bowtie}")
print(f"  Flat Filter:        {ct.protocol.flatFilter[0]} {ct.protocol.flatFilter[1]} mm")
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
print(f"  Scan Types:         {ct.protocol.scanTypes}")
print(f"                      [air={ct.protocol.scanTypes[0]}, " +
      f"offset={ct.protocol.scanTypes[1]}, " +
      f"phantom={ct.protocol.scanTypes[2]}, " +
      f"prep={ct.protocol.scanTypes[3]}]")
print()

# --- Scan Trajectory ---
print("4.2 Scan Trajectory:")
ct.protocol.scanTrajectory = "Gantry_Helical"  # "Gantry_Helical" or "Gantry_Axial" # GT
ct.protocol.viewsPerRotation = 2496  # Views per 360° rotation # GT
ct.protocol.viewCount = 7488         # Total views in scan # estimate (3 rotations * 2496 VPR)
ct.protocol.startViewId = 0          # Index of first view
ct.protocol.stopViewId = ct.protocol.viewCount - 1  # Index of last view
print(f"  Trajectory:         {ct.protocol.scanTrajectory}")
print(f"  Views/Rotation:     {ct.protocol.viewsPerRotation}")
print(f"  Total Views:        {ct.protocol.viewCount}")
print(f"  View Range:         {ct.protocol.startViewId} to {ct.protocol.stopViewId}")
print()

# --- Reference Scans ---
print("4.3 Reference Scans:")
ct.protocol.airViewCount = 1    # Views averaged for air scan # estimate
ct.protocol.offsetViewCount = 1 # Views averaged for offset scan # estimate
print(f"  Air Views:          {ct.protocol.airViewCount}")
print(f"  Offset Views:       {ct.protocol.offsetViewCount}")
print()

# --- Gantry Motion ---
print("4.4 Gantry Motion:")
ct.protocol.rotationTime = 0.5         # Rotation period (seconds) # GT (0.5s is a valid Apex speed)
ct.protocol.rotationDirection = 1      # 1=clockwise, -1=counterclockwise # estimate
ct.protocol.startAngle = 0             # Starting angle (degrees) # estimate
ct.protocol.tiltAngle = 0              # Gantry tilt (degrees) # estimate
print(f"  Rotation Time:      {ct.protocol.rotationTime} sec")
print(f"  Direction:          {'CW' if ct.protocol.rotationDirection == 1 else 'CCW'}")
print(f"  Start Angle:        {ct.protocol.startAngle}°")
print(f"  Tilt Angle:         {ct.protocol.tiltAngle}°")
print()

# --- Table Motion ---
print("4.5 Table Motion:")
ct.protocol.pitch = 0.992 # Helical pitch # GT (0.992 is a documented value)
# Table Speed = (Z_Coverage * Pitch) / RotationTime
# Table Speed = (160mm * 0.992) / 0.5s = 317.44 mm/s
ct.protocol.tableSpeed = 317.44        # Table speed (mm/sec) # GT (Derived)
ct.protocol.startZ = 0                 # Starting z-position (mm) # estimate
print(f"  Helical Pitch:      {ct.protocol.pitch}")
print(f"  Table Speed:        {ct.protocol.tableSpeed} mm/sec")
print(f"  Start Z:            {ct.protocol.startZ} mm")
print()

# --- Focal Spot Control ---
print("4.6 Focal Spot Control:")
ct.protocol.wobbleDistance = 0.5       # Flying focal spot distance (mm) # estimate (GT scanner has wobble, distance is estimate)
ct.protocol.focalspotOffset = [0, 0, 0]  # Position offset [x, y, z] (mm) # estimate
print(f"  Wobble Distance:    {ct.protocol.wobbleDistance} mm")
print(f"  Focal Offset:       {ct.protocol.focalspotOffset} mm")
print()

# --- Preprocessing ---
print("4.7 Preprocessing:")
ct.protocol.maxPrep = 9  # Upper limit of prep (low signal correction) # estimate
print(f"  Max Prep:           {ct.protocol.maxPrep}")
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
ct.physics.energyCount = 20     # Number of energy bins # estimate
ct.physics.monochromatic = -1   # -1=polychromatic, or specify energy in keV # estimate
print(f"  Energy Bins:        {ct.physics.energyCount}")
print(f"  Monochromatic:      {ct.physics.monochromatic} (-1 = polychromatic)")
print()

# --- Spatial Sampling ---
# NOTE: These sampling parameters are critical for NCAT phantom stability
# Using lower values (from working Sim_Sample_XCAT.py) prevents bus errors
print("5.2 Spatial Sampling:")
ct.physics.colSampleCount = 2     # Detector column samples # estimate
ct.physics.rowSampleCount = 1     # Detector row samples (reduced for NCAT stability) # estimate
ct.physics.srcXSampleCount = 2    # Focal spot lateral samples # estimate
ct.physics.srcYSampleCount = 1    # Focal spot longitudinal samples (reduced for NCAT) # estimate
ct.physics.viewSampleCount = 1    # View angle samples (reduced for NCAT) # estimate
print(f"  Detector:           {ct.physics.colSampleCount} × {ct.physics.rowSampleCount}")
print(f"  Focal Spot:         {ct.physics.srcXSampleCount} × {ct.physics.srcYSampleCount}")
print(f"  View:               {ct.physics.viewSampleCount}")
print()

# --- Recalculation Flags ---
print("5.3 Recalculation Flags (0=never, 1=per view, 2=per subview):")
ct.physics.recalcDet = 0        # Detector geometry # estimate
ct.physics.recalcSrc = 0        # Source geometry # estimate
ct.physics.recalcGantry = 2     # Gantry model # estimate
ct.physics.recalcRayAngle = 0   # Ray angles # estimate
ct.physics.recalcSpec = 0       # Spectrum # estimate
ct.physics.recalcFilt = 0       # Filters # estimate
ct.physics.recalcFlux = 0       # Flux # estimate
ct.physics.recalcPht = 0        # Phantom # estimate
print(f"  Detector:           {ct.physics.recalcDet}")
print(f"  Source:             {ct.physics.recalcSrc}")
print(f"  Gantry:             {ct.physics.recalcGantry}")
print(f"  Ray Angles:         {ct.physics.recalcRayAngle}")
print(f"  Spectrum:           {ct.physics.recalcSpec}")
print(f"  Filters:            {ct.physics.recalcFilt}")
print(f"  Flux:               {ct.physics.recalcFlux}")
print(f"  Phantom:            {ct.physics.recalcPht}")
print()

# --- Noise Settings ---
print("5.4 Noise:")
ct.physics.enableQuantumNoise = 1     # Photon counting noise (Poisson) # estimate
ct.physics.enableElectronicNoise = 1  # Electronic noise (Gaussian) # estimate
print(f"  Quantum Noise:      {'ON' if ct.physics.enableQuantumNoise else 'OFF'}")
print(f"  Electronic Noise:   {'ON' if ct.physics.enableElectronicNoise else 'OFF'}")
print()

# --- Physics Models ---
print("5.5 Physics Model Callbacks:")
ct.physics.rayAngleCallback = "Detector_RayAngles_2D"
ct.physics.fluxCallback = "Detection_Flux"
ct.physics.scatterCallback = "Scatter_ConvolutionModel" # estimate (Enabling realistic scatter)
ct.physics.scatterKernelCallback = ""      # Custom scatter kernel # estimate
ct.physics.scatterScaleFactor = 1          # Scatter scaling # estimate
ct.physics.prefilterCallback = "Detection_prefilter"
ct.physics.crosstalkCallback = ""          # X-ray crosstalk # estimate
ct.physics.lagCallback = ""                # Detector lag # estimate
ct.physics.opticalCrosstalkCallback = ""   # Optical crosstalk # estimate
ct.physics.DASCallback = "Detection_DAS"   # Data acquisition
print(f"  Ray Angles:         {ct.physics.rayAngleCallback}")
print(f"  Flux:               {ct.physics.fluxCallback}")
print(f"  Scatter:            {ct.physics.scatterCallback if ct.physics.scatterCallback else 'OFF'}")
print(f"  Prefilter:          {ct.physics.prefilterCallback}")
print(f"  DAS:                {ct.physics.DASCallback}")
print()

# --- I/O Preferences ---
print("5.6 I/O:")
ct.physics.outputCallback = "WriteRawView"
ct.physics.dump_period = 100  # Save data every N views # estimate
print(f"  Output Callback:    {ct.physics.outputCallback}")
print(f"  Dump Period:        {ct.physics.dump_period} views")
print()

# ============================================================================
# SECTION 6: RECONSTRUCTION CONFIGURATION
# ============================================================================
print("-" * 80)
print("SECTION 6: RECONSTRUCTION")
print("-" * 80)

# --- Image Geometry ---
print("6.1 Image Geometry:")
# FOV set to 500mm (GT)
ct.recon.fov = 500.0                   # Field of view diameter (mm) # GT (Max SFOV)
ct.recon.imageSize = 512               # Matrix size (pixels × pixels) # estimate (Standard clinical)
ct.recon.sliceCount = 100              # Number of slices to reconstruct # estimate
ct.recon.sliceThickness = 0.625        # Slice spacing (mm) # GT (Native detector row size)
ct.recon.centerOffset = [0.0, 0.0, 0.0]  # Offset from rotation center [x, y, z] (mm) # estimate
pixel_size = ct.recon.fov / ct.recon.imageSize
print(f"  FOV:                {ct.recon.fov} mm")
print(f"  Matrix Size:        {ct.recon.imageSize} × {ct.recon.imageSize}")
print(f"  Pixel Size:         {pixel_size:.3f} mm")
print(f"  Slices:             {ct.recon.sliceCount}")
print(f"  Slice Thickness:    {ct.recon.sliceThickness} mm")
print(f"  Center Offset:      {ct.recon.centerOffset} mm")
print()

# --- Reconstruction Algorithm ---
print("6.2 Algorithm:")
# Options: 'fdk_equiAngle' (circular), 'helical_equiAngle' (helical)
ct.recon.reconType = 'helical_equiAngle' # GT (Matches helical trajectory)
# Kernel options: 'R-L', 'S-L', 'standard', 'soft', 'bone'
ct.recon.kernelType = 'standard' # estimate
ct.recon.startAngle = 0  # Starting angle (degrees, 0=source at top) # estimate
print(f"  Recon Type:         {ct.recon.reconType}")
print(f"  Kernel:             {ct.recon.kernelType}")
print(f"  Start Angle:        {ct.recon.startAngle}°")
print()

# --- Output Units ---
print("6.3 Output Units:")
ct.recon.unit = 'HU'      # Options: 'HU', '/mm', '/cm'
ct.recon.mu = 0.02        # Water attenuation (/mm) at spectrum energy # estimate
ct.recon.huOffset = -1000 # HU offset (typically -1000)
print(f"  Units:              {ct.recon.unit}")
print(f"  μ (water):          {ct.recon.mu} /mm")
print(f"  HU Offset:          {ct.recon.huOffset}")
print()

# --- Output Options ---
print("6.4 Output Options:")
ct.recon.printReconParameters = False
ct.recon.saveImageVolume = True
ct.recon.saveSingleImages = False
ct.recon.displayImagePictures = False
ct.recon.saveImagePictureFiles = False
print(f"  Print Params:       {ct.recon.printReconParameters}")
print(f"  Save Volume:        {ct.recon.saveImageVolume}")
print(f"  Save Singles:       {ct.recon.saveSingleImages}")
print(f"  Display Images:     {ct.recon.displayImagePictures}")
print()

# ============================================================================
# SECTION 7: OUTPUT CONFIGURATION
# ============================================================================
print("-" * 80)
print("SECTION 7: OUTPUT")
print("-" * 80)
ct.resultsName = "xcat_GE_Apex_Elite_sim" # GT
ct.do_Recon = 1  # Enable reconstruction
print(f"  Results Name:       {ct.resultsName}")
print(f"  Do Recon:           {'Yes' if ct.do_Recon else 'No'}")
print()

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================
print("="*80)
print("CONFIGURATION SUMMARY (GE REVOLUTION APEX ELITE)")
print("="*80)
print(f"Phantom:      {ct.phantom.filename}")
print(f"Scanner:      {ct.scanner.detectorColCount}×{ct.scanner.detectorRowCount}, " +
      f"SID={ct.scanner.sid}mm, SDD={ct.scanner.sdd}mm")
print(f"Protocol:     {ct.protocol.viewCount} views ({ct.protocol.viewsPerRotation} VPR), {ct.protocol.rotationTime}s rot, {ct.protocol.pitch} pitch, " +
      f"{ct.protocol.mA} mA, {ct.protocol.spectrumFilename}")
print(f"Physics:      {ct.physics.energyCount} energy bins, " +
      f"noise={'ON' if ct.physics.enableQuantumNoise else 'OFF'}, scatter={'ON' if ct.physics.scatterCallback else 'OFF'}")
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
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    ct.run_all()
    print()
    print("="*80)
    print("✓ SIMULATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Output files generated:")
    print(f"  {ct.resultsName}.air     - Air scan data")
    print(f"  {ct.resultsName}.offset - Offset scan data")
    print(f"  {ct.resultsName}.scan    - Phantom scan data (raw projections)")
    print(f"  {ct.resultsName}.prep    - Preprocessed projections")
    print()
except Exception as e:
    print(f"\n✗ ERROR during simulation: {e}")
    raise

# ============================================================================
# RECONSTRUCTION
# ============================================================================
print("="*80)
print("RUNNING FDK RECONSTRUCTION (HELICAL)")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    recon.recon(ct)
    print()
    print("="*80)
    print("✓ RECONSTRUCTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

# Create comprehensive visualization with ground truth table
fig = plt.figure(figsize=(20, 12))
fig.suptitle('XCAT CT Simulation Results - GE Revolution Apex Elite (K213715)',
             fontsize=18, fontweight='bold', y=0.98)

# Add Ground Truth Parameters Table
gt_params_text = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       GROUND TRUTH SCANNER PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Scanner:     GE Revolution Apex Elite (K213715)
 Detector:    Gemstone Clarity* (256×832 array)
              *Simulated with Lumex scintillator
 Geometry:    SID = 626 mm, SDD = 1097 mm
 Z-Coverage:  160 mm (0.625 mm/row)
 Tube:        Quantix 160 (108 kW, 10° anode)
 Technique:   120 kVp, 800 mA
 Rotation:    0.5 sec, 2496 views/rotation
 Pitch:       0.992 (helical), 317.4 mm/sec
 Focal Spot:  1.0 × 0.7 mm (wobble enabled)
 Recon FOV:   500 mm, 512×512 matrix, 100 slices
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
ax_table = plt.subplot(3, 5, 1)
ax_table.text(0.05, 0.5, gt_params_text,
             transform=ax_table.transAxes,
             fontsize=9,
             verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='navy', linewidth=2))
ax_table.axis('off')

# Plot all slices (show a subset if too many)
n_slices = img.shape[0]
plot_slices = min(n_slices, 3) # Show up to 3 slices to make room for table
slice_indices = np.linspace(0, n_slices - 1, plot_slices, dtype=int)

for i, slice_idx in enumerate(slice_indices):
    ax = plt.subplot(3, 5, i+2)
    im = ax.imshow(img[slice_idx, :, :], cmap='gray', vmin=-200, vmax=200)
    ax.set_title(f'Slice {slice_idx}\nSoft Tissue Window', fontsize=10)
    ax.axis('off')

# Plot middle slice with different windows
if n_slices > 0:
    middle_idx = n_slices // 2
    middle_slice = img[middle_idx, :, :]

    # Lung window
    ax = plt.subplot(3, 5, 6)
    ax.imshow(middle_slice, cmap='gray', vmin=-1000, vmax=0)
    ax.set_title(f'Slice {middle_idx}\nLung Window', fontsize=10)
    ax.axis('off')

    # Bone window
    ax = plt.subplot(3, 5, 7)
    ax.imshow(middle_slice, cmap='gray', vmin=-200, vmax=1000)
    ax.set_title(f'Slice {middle_idx}\nBone Window', fontsize=10)
    ax.axis('off')

    # Profile plot
    ax = plt.subplot(3, 5, 8)
    center_row = ct.recon.imageSize // 2
    profile = middle_slice[center_row, :]
    ax.plot(profile, 'b-', linewidth=2)
    ax.set_title('Horizontal Profile (Middle Slice)', fontsize=10)
    ax.set_xlabel('Column')
    ax.set_ylabel('HU')
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = plt.subplot(3, 5, 9)
    ax.hist(middle_slice.flatten(), bins=100, color='steelblue', alpha=0.7, range=(-1500, 1500))
    ax.set_title('HU Histogram (Middle Slice)', fontsize=10)
    ax.set_xlabel('HU')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics text box
    ax = plt.subplot(3, 5, 10)
    stats_text = f"""
    Image Statistics:
    ━━━━━━━━━━━━━━━━━━━
    Shape: {img.shape[0]}×{img.shape[1]}×{img.shape[2]}

    HU Range:
      Min: {img.min():.1f}
      Max: {img.max():.1f}

    Mean: {img.mean():.1f} HU
    Std:  {img.std():.1f} HU

    Median: {np.median(img):.1f} HU
    """
    ax.text(0.1, 0.5, stats_text,
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axis('off')

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