import numpy as np
import tables

def get_features_labels(file_name, remove_mass_pt_window=True):
    features = ['fj_jetNTracks', 'fj_nSV', 'fj_tau0_trackEtaRel_0', 'fj_tau0_trackEtaRel_1',
                'fj_tau0_trackEtaRel_2', 'fj_tau1_trackEtaRel_0', 'fj_tau1_trackEtaRel_1',
                'fj_tau1_trackEtaRel_2', 'fj_tau_flightDistance2dSig_0', 'fj_tau_flightDistance2dSig_1',
                'fj_tau_vertexDeltaR_0', 'fj_tau_vertexEnergyRatio_0', 'fj_tau_vertexEnergyRatio_1',
                'fj_tau_vertexMass_0', 'fj_tau_vertexMass_1', 'fj_trackSip2dSigAboveBottom_0',
                'fj_trackSip2dSigAboveBottom_1', 'fj_trackSip2dSigAboveCharm_0', 'fj_trackSipdSig_0',
                'fj_trackSipdSig_0_0', 'fj_trackSipdSig_0_1', 'fj_trackSipdSig_1', 'fj_trackSipdSig_1_0',
                'fj_trackSipdSig_1_1', 'fj_trackSipdSig_2', 'fj_trackSipdSig_3', 'fj_z_ratio']
    spectators = ['fj_sdmass', 'fj_pt']
    labels = ['fj_isQCD*sample_isQCD', 'fj_isH*fj_isBB']
    nfeatures = len(features)
    nspectators = len(spectators)
    nlabels = len(labels)
    h5file = tables.open_file(file_name, 'r')
    njets = getattr(h5file.root, features[0]).shape[0]
    feature_array = np.zeros((njets, nfeatures))
    spec_array = np.zeros((njets, nspectators))
    label_array = np.zeros((njets, nlabels))
    for i, feat in enumerate(features):
        feature_array[:, i] = getattr(h5file.root, feat)[:]
    for i, spec in enumerate(spectators):
        spec_array[:, i] = getattr(h5file.root, spec)[:]
    for i, label in enumerate(labels):
        prod0, prod1 = label.split('*')
        fact0 = getattr(h5file.root, prod0)[:]
        fact1 = getattr(h5file.root, prod1)[:]
        label_array[:, i] = np.multiply(fact0, fact1)
    if remove_mass_pt_window:
        mask = (spec_array[:, 0] > 40) & (spec_array[:, 0] < 200) & \
               (spec_array[:, 1] > 300) & (spec_array[:, 1] < 2000)
        feature_array = feature_array[mask]
        label_array = label_array[mask]
    one_hot_mask = np.sum(label_array, axis=1) == 1
    feature_array = feature_array[one_hot_mask]
    label_array = label_array[one_hot_mask]
    h5file.close()
    return feature_array, label_array
