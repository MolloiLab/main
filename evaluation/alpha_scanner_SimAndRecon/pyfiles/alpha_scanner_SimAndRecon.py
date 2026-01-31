import os
import numpy as np
import matplotlib.pyplot as plt

###------------ import XCIST-CatSim
from my_commonTools import *
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
import gecatsim.pyfiles.CommonTools as CommonTools


##--------- Initialize
# Get the user path (go up to experiment_15 directory)
userPath = getUserPath()
# userPath = userPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(userPath)
CommonTools.my_path.add_search_path(userPath)

# Add the cfg directory to the search path so my_path.find() can locate the files
cfgPath = os.path.join(userPath, "cfg")
CommonTools.my_path.add_search_path(cfgPath)

# Use my_path.find() to locate all config files
phantomCfgPathname = CommonTools.my_path.find("cfg", "Phantom_Water.cfg", "")  # Change to Gammex Phantom later
physicsCfgPathname = CommonTools.my_path.find("cfg", "Physics_Default.cfg", "")
scannerCfgPathname = CommonTools.my_path.find("cfg", "Scanner_PCCT.cfg", "")

ct = xc.CatSim(phantomCfgPathname, scannerCfgPathname, physicsCfgPathname)

ct.experimentDirectory = os.path.join(userPath, "examples", "evaluation", "alpha_scanner_SimAndRecon")
ct.experimentName = "alpha_scanner_Water_Phantom"
ct.resultsName = os.path.join(ct.experimentDirectory, ct.experimentName, ct.experimentName)

##--------- Make changes to parameters (optional)
# ct.scanner.eNoise = 20000.0
ct.protocol.viewsPerRotation = 500
ct.protocol.viewCount = ct.protocol.viewsPerRotation
ct.protocol.stopViewId = ct.protocol.viewCount-1
ct.protocol.mA = 400
ct.protocol.spectrumFilename = "tungsten_tar7.0_120_filt.dat"
ct.physics.energyCount = 12
ct.protocol.flatFilter = ['Sn', 0.5] # Tin filter of 0.5 mm

# ct.protocol.fluxScale = 300.0 # to fix the empty detectors

ct.scanner.detectorColSize = 0.2     #0.2    (1)        # detector column pitch or size (in mm)
ct.scanner.detectorRowSize = 0.2     #0.2     (1)       # detector row pitch or size (in mm)
ct.scanner.detectorColCount = 1984  #1984    (900)        # total number of detector columns
ct.scanner.detectorRowsPerMod = 288 #288       (4)
ct.scanner.detectorRowCount = ct.scanner.detectorRowsPerMod
ct.scanner.detectorMaterial = "CZT"    # CdTe
ct.scanner.detectorResponseFilename = 'PC_spectral_response_CZT0.25x0.25x1.6'  # also CdTe, just change the name 
ct.scanner.detectorDepth = 3.0  # mm
ct.scanner.detectorBinThreshold = [0, 20, 120] # in keV
ct.scanner.focalspotWidth = 0.4    #0.4
ct.scanner.focalspotLength = 0.5   #0.5

# ct.physics.monochromatic = 100
# print(xc.pyfiles.GetMu.GetMu('water', ct.physics.monochromatic))
# ct.recon.mu = xc.pyfiles.GetMu.GetMu('water', ct.physics.monochromatic)[0] / 10
# ct.recon.mu = np.mean(xc.pyfiles.GetMu.GetMu('water', 77)) / 10 # NOTE: divided by 10 to convert cm to mm
ct.recon.mu = 0.017061810195446014 # TEMP
ct.recon.fov = 400                   # in mm.
ct.imageSize = 512
ct.recon.sliceThickness = 0.4
# ct.recon.sliceCount = 4        # number of slices to reconstruct

# ct.recon.displayWindowMin = -100       # In HU.
# ct.recon.displayWindowMax = 600        # In HU.
# ct.recon.displayWindow = ct.recon.displayWindowMax - ct.recon.displayWindowMin
# ct.recon.displayLevel = (ct.recon.displayWindowMax + ct.recon.displayWindowMin)/2
ct.recon.saveImagePictureFiles = True


    ##--------- Run simulation
mA_settings = [400, 200, 40]
for mA in mA_settings:
    ct.experimentName = f"alpha_scanner_Water_Phantom_{mA}_mA"
    ct.protocol.mA = mA
    ct = initializeExperimentDirectory(ct)

    if not ct.scanner.detectorSumBins:
        ct.do_prep = 0
    ct.run_all()  # run the scans defined by protocol.scanTypes

    ##--------- Prep and recon for each bin
    nBin = len(ct.scanner.detectorBinThreshold)-1
    bins = ct.scanner.detectorBinThreshold

    nCol = ct.scanner.detectorColCount
    nRow = ct.scanner.detectorRowCount
    nView = ct.protocol.viewCount

    airscan = xc.rawread("%s.air" % ct.resultsName, [nRow, nCol, nBin], 'float')
    offsetscan = xc.rawread("%s.offset" % ct.resultsName, [nRow, nCol, nBin], 'float')
    airscan -= offsetscan
    scan_fname = f"{ct.resultsName}.scan"

    scan_mmap = np.memmap(
        scan_fname,
        dtype=np.float32,
        mode='r',
        shape=(nView, nRow, nCol, nBin)
    )

    for binId in range(nBin):

        print(f"Processing bin {binId}: {bins[binId]}–{bins[binId+1]} keV")

        newFname = os.path.join(
            ct.experimentDirectory,
            ct.experimentName,
            f"{ct.experimentName}_bin_{bins[binId]}_{bins[binId+1]}keV"
        )

        # --- create PREP file on disk (not in RAM)
        prep_mmap = np.memmap(
            newFname + '.prep',
            dtype=np.float32,
            mode='w+',
            shape=(nView, nRow, nCol)
        )

        air = airscan[:, :, binId]
        offset = offsetscan[:, :, binId]

        for v in range(nView):
            # Read ONE view
            view = scan_mmap[v, :, :, binId]

            # Apply offset
            view = view - offset

            # In-place log prep
            with np.errstate(divide='ignore', invalid='ignore'):
                prep_view = -np.log(view / air)

            prep_view[~np.isfinite(prep_view)] = 0
            prep_view = np.clip(prep_view, 0, 20)

            # Write ONE view to disk
            prep_mmap[v, :, :] = prep_view

            # Free memory explicitly
            del view, prep_view

        del prep_mmap  # flush to disk

        # --- Reconstruction
        ct.resultsName = newFname
        ct.do_Recon = 1
        recon.recon(ct)

        import gc
        gc.collect()



# phantomscan = xc.rawread("%s.scan" % ct.resultsName, [nView, nRow, nCol, nBin], 'float')
# airscan -= offsetscan
# phantomscan -= offsetscan


 # Create the results folder if it doesn't exist.
# if not os.path.exists(resultsPath):
#     os.makedirs(resultsPath)
    
# ct.resultsName = os.path.join(resultsPath, ct.experimentName)


# for binId in range(nBin):
#     prep = np.where(phantomscan[:,:,:,binId]==0,0,-np.log(phantomscan[:,:,:,binId]/airscan[:,:,binId]))
#     prep = np.where(prep<0,0,prep)
#     prep = np.where(prep>20,20,prep)

#     # Save to experimentDirectory to match non-copy version
#     # newFname = os.path.join(ct.experimentDirectory, "test_bin%s" % binId)
#     newFname = os.path.join(ct.experimentDirectory, ct.experimentName, 
#                              f"{ct.experimentName}_bin_{bins[binId]}_{bins[binId+1]}keV")
#     xc.rawwrite(newFname+'.prep', prep)
#     ct.resultsName = newFname

#     ##--------- Reconstruction
#     ct.do_Recon = 1
#     recon.recon(ct)


##--------- Show results
# imgFname = "%s_%dx%dx%d.raw" %(ct.resultsName, ct.recon.imageSize, ct.recon.imageSize, ct.recon.sliceCount)
# img = xc.rawread(imgFname, [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize], 'float')
# plt.imshow(img[2,:,:], cmap='gray', vmin=-200, vmax=200)
# plt.show()