import scipy.io as sio
import numpy as np
from gecatsim.pyfiles.GetMu import GetMu

# --------------------------------------------------
# User settings
# --------------------------------------------------
czt_file = './gecatsim/response_matrix/PC_spectral_response_CZT0.25x0.25x1.6.mat'
cdte_file = './gecatsim/response_matrix/PC_spectral_response_CdTe0.25x0.25x2.0.mat'

# Detector properties
rho_cdte = 5.85        # g/cm^3
thickness_mm = 2.0
thickness_cm = thickness_mm / 10.0

# --------------------------------------------------
# Load CZT response matrix
# --------------------------------------------------
mat = sio.loadmat(czt_file)

E = mat['Evec0'].flatten()      # true photon energy bins (keV)
D = mat['Dvec0'].flatten()      # deposited energy bins (keV)
res_mat = mat['res_mat']        # shape: [n_Dbin, n_Ebin]

# --------------------------------------------------
# Compute original CZT QE(E)
# --------------------------------------------------
qe_czt = np.sum(res_mat, axis=0)
qe_czt = np.clip(qe_czt, 1e-12, None)  # prevent divide-by-zero

# --------------------------------------------------
# Compute CdTe QE(E) from physics
# --------------------------------------------------
mu_cdte = GetMu("CdTe", E)  # mass attenuation [cm^2/g]

qe_cdte = 1.0 - np.exp(-mu_cdte * rho_cdte * thickness_cm)
qe_cdte = np.clip(qe_cdte, 0.0, 1.0)

# --------------------------------------------------
# Scale response matrix column-by-column
# --------------------------------------------------
scale = qe_cdte / qe_czt

res_mat_cdte = res_mat * scale[np.newaxis, :]

# --------------------------------------------------
# Assemble CdTe response file (CatSim format)
# --------------------------------------------------
cdte_mat = {
    'Evec0': mat['Evec0'],
    'Dvec0': mat['Dvec0'],
    'n_Ebin0': mat['n_Ebin0'],
    'n_Dbin0': mat['n_Dbin0'],
    'res_mat': res_mat_cdte
}

# --------------------------------------------------
# Save CdTe response
# --------------------------------------------------
sio.savemat(cdte_file, cdte_mat)

print("CdTe response matrix successfully generated.")

