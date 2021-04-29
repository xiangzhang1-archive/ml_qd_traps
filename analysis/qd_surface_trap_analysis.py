import os
from QD_trap_py import load_data
from QD_trap_py import build_feature

### This is the root folder and list of sub-folders for data. 
### It should refer to the top-level "data" folder
### For now we are using a set of test data
data_root = '../data-test/'
data_dirlist = [item for item in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, item))]

# Load dataset.
load_qd_data = load_data.LoadData(data_root, data_dirlist)
load_qd_data.read_sys_info()
load_qd_data.read_old_struct()
load_qd_data.read_new_struct()
load_qd_data.read_en_eigval()
load_qd_data.read_pdos_ion()
load_qd_data.combine_data()
nrecord = load_qd_data.nrecord
input_vars = load_qd_data.input_vars
input_var_labels = load_qd_data.input_var_labels
print('\n')

# Define feature space for both classification and regression.
# Please find options of data and target features in the appendix.
# The flags 'rebuild_nn_list' determine whether to recalculate or read data from files.
rebuild_nn_list = False
feat_data_list = ['stoi', 'at_mask_conf_s001_pbi']
feat_target_list = ['ntrap_deep', 'at_trap_shaw_vb_pdos_ratio', 'at_trap_shaw_cb_pdos_ratio']

feat_param_list = {'trap_shaw_en_tol': 0.2,     # Energy criterion for identifying shallow traps (in eV).
                   'band_rel_crit': 0.3,        # The relevant energy range of bands near the edge (in eV).
                   'loc_crit_def1': 1.5,        # Localization criterion with definition 1 (sum over weighted std) for trap state,
                                                # say the value exceeds that of the reference QD times 'loc_crit_def1'.
                   'trap_atom_crit': 4.0,       # Atoms with pdos exceeds that of the element averaging value times 'trap_atom_crit'
                                                # are viewed as responsible to the trap state.
                   'temper': 300.0,             # Temperature for estimate Boltzmann distribution.
                   'conf_pb_aggr_bl_crit': 3.5,                 # distance criterion for Pb aggregation.
                   'conf_i_aggr_bl_crit': 3.9,                  # distance criterion for Pb aggregation.
                   'bond_len_est': [[2.9, 3.2], [0.0]],         # Estimated bond length list for elem [[1-2, 1-3], [2-3]].
                   'bond_len_tol': [[0.5, 0.7], [0.0]],         # Bond length tolerance.
                   }

if (feat_data_list.count('at_mask_conf_s001_pbi') != 0 and feat_data_list.count('nconf_s001_pbi') == 0):
    feat_data_list.append('nconf_s001_pbi')
if (feat_data_list.count('at_mask_conf_pb_aggr') != 0 and feat_data_list.count('nconf_pb_aggr') == 0):
    feat_data_list.append('nconf_pb_aggr')
if (feat_data_list.count('at_mask_conf_i_aggr') != 0 and feat_data_list.count('nconf_i_aggr') == 0):
    feat_data_list.append('nconf_i_aggr')
if (feat_data_list.count('at_mask_conf_brok_pbs') != 0 and feat_data_list.count('nconf_brok_pbs') == 0):
    feat_data_list.append('nconf_brok_pbs')    

build_feat = build_feature.BuildFeature(nrecord, input_vars, input_var_labels, feat_data_list, \
             feat_target_list, feat_param_list)
build_feat.analyze_struct(rebuild_nn_list)
build_feat.analyze_eigval()
build_feat.analyze_trap()
build_feat.write_data_file()
build_feat.build_feat_data() 
build_feat.build_feat_target()




#*******************************************************************************
#                                  Appendix
#*******************************************************************************
#
# Data feature options:

# Features of each dot:
# 'stoi': stoichiometry of the entire dot.
# 'core_shape': only Wulff construction is compatible with this version.
# 'core_diameter': core diameter in unit of angstrom.
# 'core_surf_site_prop': proportion of various types of surface sites in QD core.
# 'lig_site_prop': proportion of various types of ligand sites.
# 'lig_codnum_prop': proportion of ligands with various coordination numbers.
# 'nconf_s001_pbi': Pb-I bonds at the {001} facet.
# 'nconf_pb_aggr': Pb atom aggregation.
# 'nconf_i_aggr': I atom aggregation.
# 'nconf_brok_pbs': broken Pb-S bond during relaxation.

# Note: the location and bonding environment (coordination number) of each atom or ligand are defined as follows,
# options of core/ligand sites: 0:inside, 1:{001}, 2:{111}, 3:{011}, 4:{001}-{001} edge, 5:{111}-{111} edge, 6:{011}-{011} edge, 
#                             7:{001}-{111} edge, 8:{001}-{011} edge, 9:{111-011} edge, 10:corner)
# for 2-fold ligand, the edge site is defined by that both of the attached Pb atoms are at edge, while the corner site is defined\
# by that at least one of the Pb is at the corner while the other is at edge.
# for 3-fold ligand, the edge site is defined by that at least two of the attached Pb atoms are at edge, while the corner site \
# is defined by that at least one of the Pb is at the corner while the others are at edge.
#
# Features of each atom:
# 'at_core_lig': whether the atom is inside the core on at the ligand (0: core, 1: ligand).
# 'at_core_atom_site': type of atom site for QD core.
# 'at_lig_atom_site': type of atom site for ligands.
# 'at_mask_conf_s001_pbi': mask for the atoms within the configuration of Pb-I bonds at the {001} facet (0: no, 1: yes).
# 'at_mask_conf_pb_aggr': mask for the atoms within the configuration of Pb atom aggregation (0: no, 1: yes).
# 'at_mask_conf_i_aggr': mask for the atoms within the configuration of I atom aggregation (0: no, 1: yes).
# 'at_mask_conf_brok_pbs': mask for the atoms within the configuration of broken Pb-S bond (0: no, 1: yes).

#--------------------------------------
# Target feature options:

# Features of each dot:
# 'vb_edge': edge of valence band estimated by comparison to reference stoichiometric QDs.
# 'cb_edge': edge of conduction band estimated by comparison to reference stoichiometric QDs.
# 'opt_homo': eigenvalues of states below the vb_edge with delocalized wavefunction.
# 'opt_lumo': eigenvalues of states above the cb_edge with delocalized wavefunction.
# 'opt_gap': opt_lumo - opt_homo in unit of eV.
# 'ntrap_deep': number of deep trap states.
# 'ntrap_shaw_vb': number of shallow trap states around the vb edge.
# 'ntrap_shaw_cb': number of shallow trap states around the cb edge.
# 'ave_loc_vb': average localization of states near the vb edge.
# 'ave_loc_cb': average localization of states near the cb edge.
# 'ave_loc_mg': average localization of mid-gap states.
# 'trap_deep_en_dist_vb': energy of the lowest deep trap state away from the vb edge.
# 'trap_deep_en_dist_cb': energy of the highest deep trap state away from the cb edge.
# 'trap_shaw_vb_en_dist': energy of the highest shallow trap state around vb from the vb edge.
# 'trap_shaw_cb_en_dist': energy of the highest shallow trap state around cb from the cb edge.
# 'mask_trap_deep': mask for QDs with deep trap states (0: no, 1: yes).
# 'mask_trap_shaw_vb': mask for QDs with shallow trap states near vb (0: no, 1: yes).
# 'mask_trap_shaw_cb': mask for QDs with shallow trap states near cb (0: no, 1: yes).
#
# Features of each atom:
# 'at_mask_trap_deep': mask for atoms responsible to deep traps (0: no, 1: yes).
# 'at_mask_trap_shaw_vb': mask for atoms responsible to shallow traps near vb (0: no, 1: yes).
# 'at_mask_trap_shaw_cb': mask for atoms responsible to shallow traps near cb (0: no, 1: yes).
# 'at_trap_deep_pdos_ratio': ratio of pdos on each atom to the average value for deep trap states.
# 'at_trap_shaw_vb_pdos_ratio': ratio of pdos on each atom to the average value for shallow trap states near vb.
# 'at_trap_shaw_cb_pdos_ratio': ratio of pdos on each atom to the average value for shallow trap states near cb.

