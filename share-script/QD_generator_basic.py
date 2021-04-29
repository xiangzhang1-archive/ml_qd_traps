# run with
# python -u QD_generator_basic.py >> log

import os
import math
import operator
import numpy as np
import mathutils
import random

core_elem_list = ['Pb', 'S']          # core element list
core_elem_stoi = [2.0, -2.0]          # core element stoichiometry
core_elem_nve = [14, 6]                # core element number of valence electrons
core_elem_nve_nod = [4, 6]                # core element number of valence electrons
#lig_elem_list = ['C', 'S', 'H']       # ligand element list
lig_elem_list = ['I']                 # ligand element list
lig_stoi = -1.0                       # ligand element stoichiometry
#lig_elem_nve = [4, 6, 1]                  # ligand element number of valence electrons
lig_elem_nve = [7]                  # ligand element number of valence electrons
core_anchor_elem_list = ['Pb']        # element list for anchor atoms in the core
#lig_name = 'EDT'                      # ligand name used for reading input file
lig_name = 'I'                        # ligand name used for reading input file
#lig_del_atom_id_list = [4]            # atom id for delete during passivation
lig_del_atom_id_list = []            # atom id for delete during passivation
#lig_ori_orig_ref_id = [8, 0]          # original orientation of ligand is determined by 2 reference atoms (start from 0)
lig_ori_orig_ref_id = [0, 0]          # original orientation of ligand is determined by 2 reference atoms (start from 0)
stoi_target = 6.0                # target of total stoichiometry
nconfig = 1                      # number of generating configurations 
a0 = 5.9878 / 2.0                # lattice constant
temper = 300.0                   # temperature in unit of K
imax = 8                         # crystal range, shoud be larger than the dot
center_shift = 0.0               # shift of center in unit of a0
gamma_001 = 1.0                  # surface enegy of {001} surface in unit of eV
gamma_111 = 0.8                  # surface enegy of {111} surface in unit of eV
gamma_011 = 0.8                  # surface enegy of {011} surface in unit of eV
h_001 = 3.0 * a0                 # distance from the {001} facet to the origin
core_bond_len = a0               # bond length inside the core
bond_len_tol = 0.1               # tolerance of bond length in unit of angstrom
lig_bind_en_001 = 0.985            # ligand binding energy on {001} facet in unit of eV
lig_bind_en_111 = 1.025            # ligand binding energy on {111} facet in unit of eV
lig_bind_en_011 = 1.0            # ligand binding energy on {011} facet in unit of eV
lig_dist_001 = 3.0               # edge-to-edge distance between ligand and surface {001}
lig_dist_111 = 2.0               # edge-to-edge distance between ligand and surface {111}
lig_dist_011 = 2.0               # edge-to-edge distance between ligand and surface {011}
overlap_dist = 3.0               # exclusion distance between atoms
anchor_ntry_max = 100            # for each candidate anchor point, quit after too many trails
dist_to_surface_tol = 0.5        # distance from the surface to define surface atoms in core in angstrom
box_vac = 13.00000               # vacuum of simulation box in unit of angstrom
kb = 1.38064852e-23              # boltzmann constant in SI
ev = 1.60217656e-19              # eV in SI
tiny = 0.0001

mydir=os.getcwd()
print('my dir = ', mydir) 
print('core element list = ', core_elem_list)
print('lig element list = ', lig_elem_list)
print('core element stoichiometry = ', core_elem_stoi)
print('ligand stoichiometry = ', lig_stoi)
print('lattice constant = ', a0, ' angstrom \n')

#-------------------------------------------------------------------------------
# Generate the core of nanocrystal.
class Gen_Core:

    def __init__(self, core_elem_list):
        self.core_elem_list = core_elem_list
        self.core_nelem = len(self.core_elem_list)
        
    def bulk_gen(self):
        jmax = imax
        kmax = imax
        self.bulk_coord_list = [[] for i in range(self.core_nelem)]
        for i in range(-imax, imax+1):     
            for j in range(-jmax, jmax+1):
                for k in range(-kmax, kmax+1):
                    temp_list = [a0 * (p + center_shift) for p in [i, j, k]]
                    if ((i + j + k) % 2 == 0):
                        self.bulk_coord_list[0].append(temp_list)
                    else:
                        self.bulk_coord_list[1].append(temp_list)
        self.bulk_elem_natom = [len(self.bulk_coord_list[i]) for i in range(2)]
    
    def wulff_cut(self):
        self.core_shape = 'wulff'
        h_111 = h_001 * gamma_111 / gamma_001
        h_011 = h_001 * gamma_011 / gamma_001
        core_atom_mask = [[0 for j in range(self.bulk_elem_natom[i])] for i in range(self.core_nelem)]
        for i in range(self.core_nelem):
            for j in range(self.bulk_elem_natom[i]):
                core_atom_mask[i][j] = 1
                coord = self.bulk_coord_list[i][j]
                if (abs(coord[0]) > h_001+tiny or abs(coord[1]) > h_001+tiny or abs(coord[2]) > h_001+tiny):
                    core_atom_mask[i][j] = 0
                if (abs(coord[0]) + abs(coord[1]) + abs(coord[2]) > math.sqrt(3.0) * h_111 + tiny):
                    core_atom_mask[i][j] = 0
                tempdb = math.sqrt(2.0) * h_011 + tiny
                if (abs(coord[0]) + abs(coord[1]) > tempdb or abs(coord[0]) + abs(coord[2]) > tempdb \
                        or abs(coord[1]) + abs(coord[2]) > tempdb):
                    core_atom_mask[i][j] = 0
        self.core_orig_elem_natom = [0 for i in range(self.core_nelem)]
        self.core_orig_coord_list = [[] for i in range(2)]
        for i in range(self.core_nelem):
            for j in range(self.bulk_elem_natom[i]):
                if (core_atom_mask[i][j] == 1):
                    self.core_orig_coord_list[i].append(self.bulk_coord_list[i][j])
                    self.core_orig_elem_natom[i] = self.core_orig_elem_natom[i] + 1

    def sphere_cut(self):
        self.core_shape = 'sphere'
        radius = h_001
        core_atom_mask = [[0 for j in range(self.bulk_elem_natom[i])] for i in range(self.core_nelem)]
        for i in range(self.core_nelem):
            for j in range(self.bulk_elem_natom[i]):
                core_atom_mask[i][j] = 1
                coord = self.bulk_coord_list[i][j]
                dist = np.linalg.norm(coord)
                if (dist > radius + tiny):
                    core_atom_mask[i][j] = 0
        self.core_orig_elem_natom = [0 for i in range(self.core_nelem)]
        self.core_orig_coord_list = [[] for i in range(2)]
        for i in range(self.core_nelem):
            for j in range(self.bulk_elem_natom[i]):
                if (core_atom_mask[i][j] == 1):
                    self.core_orig_coord_list[i].append(self.bulk_coord_list[i][j])
                    self.core_orig_elem_natom[i] = self.core_orig_elem_natom[i] + 1
        
        
                    
    def modify_surface(self, core_bond_len, bond_len_tol):
        core_orig_elem_natom_max = max(self.core_orig_elem_natom)
        num_nn_list = [[0 for j in range(core_orig_elem_natom_max)] for i in range(self.core_nelem)]
        self.core_elem_natom = [0 for i in range(self.core_nelem)]
        self.core_coord_list = [[] for i in range(self.core_nelem)]
        for i in range(self.core_nelem):
            for j in range(self.core_orig_elem_natom[i]):
                coord1 = self.core_orig_coord_list[i][j]
                for ii in range(self.core_nelem):
                    for jj in range(self.core_orig_elem_natom[ii]):
                       coord2 = self.core_orig_coord_list[ii][jj]
                       if (i != ii or j !=jj):
                           vec = list(map(operator.sub, coord2, coord1))
                           dist = np.linalg.norm(vec)
                           if (abs(dist - core_bond_len) < bond_len_tol):
                               num_nn_list[i][j] = num_nn_list[i][j] + 1
                if (num_nn_list[i][j] > 1):
                    self.core_coord_list[i].append(self.core_orig_coord_list[i][j])
                    self.core_elem_natom[i] = self.core_elem_natom[i] + 1
        coord_flat = []
        for i in range(self.core_nelem):
            for j in range(len(self.core_coord_list[i])):
                coord_flat.append(self.core_coord_list[i][j])
        natom_tot = len(coord_flat)
        self.dot_diameter = 0.0
        for i in range(natom_tot-1):
            for j in range(i, natom_tot):
                vec = np.subtract(coord_flat[i], coord_flat[j])
                dist = np.linalg.norm(vec)
                self.dot_diameter = max(self.dot_diameter, dist)
        
        
#-------------------------------------------------------------------------------
# Find anchor points on the dot surface.
class Find_Anchor:

    def __init__(self, core_elem_list, core_coord_list, core_anchor_elem_list):
        self.core_elem_list = core_elem_list
        self.core_coord_list = core_coord_list
        self.core_anchor_elem_list = core_anchor_elem_list 
        self.core_anchor_nelem = len(self.core_anchor_elem_list)

    def search_core_anchor(self, dist_to_surface_tol):
        self.core_anchor_id_list_001 = [[] for i in range(self.core_anchor_nelem)]
        self.core_anchor_id_list_111 = [[] for i in range(self.core_anchor_nelem)]
        self.core_anchor_id_list_011 = [[] for i in range(self.core_anchor_nelem)]
        for i in range(self.core_anchor_nelem):
            eid = core_elem_list.index(self.core_anchor_elem_list[i])
            surf_dist_far_001 = 0.00
            surf_dist_far_111 = 0.00
            surf_dist_far_011 = 0.00
            for aid in range(len(self.core_coord_list[i])):
                coord = self.core_coord_list[eid][aid]
                dist_001 = abs(coord[0])
                surf_dist_far_001 = max(surf_dist_far_001, dist_001)
                dist_111 = (abs(coord[0]) + abs(coord[1]) + abs(coord[2])) / math.sqrt(3.0)
                surf_dist_far_111 = max(surf_dist_far_111, dist_111)
                dist_011 = (abs(coord[0]) + abs(coord[1])) / math.sqrt(2.0)
                surf_dist_far_011 = max(surf_dist_far_011, dist_011)
            for aid in range(len(self.core_coord_list[i])):
                coord = self.core_coord_list[eid][aid]
                dist_001 = max(abs(coord[0]), abs(coord[1]), abs(coord[2]))
                if (surf_dist_far_001 - dist_001 < dist_to_surface_tol):
                    self.core_anchor_id_list_001[i].append([eid, aid])
                dist_111 = (abs(coord[0]) + abs(coord[1]) + abs(coord[2])) / math.sqrt(3.0)
                if (surf_dist_far_111 - dist_111 < dist_to_surface_tol):
                    self.core_anchor_id_list_111[i].append([eid, aid])
                dist_011 = max(abs(coord[0]) + abs(coord[1]), abs(coord[0]) + abs(coord[2]), \
                               abs(coord[1]) + abs(coord[2])) / math.sqrt(2.0)
                if (surf_dist_far_011 - dist_011 < dist_to_surface_tol):
                    self.core_anchor_id_list_011[i].append([eid, aid])

    def search_lig_anchor(self, core_bond_len, bond_len_tol, lig_dist_001, lig_dist_111, lig_dist_011, tiny):
        self.lig_anchor_coord_list_001 = [[] for i in range(self.core_anchor_nelem)]
        self.lig_anchor_coord_list_111 = [[] for i in range(self.core_anchor_nelem)]
        self.lig_anchor_coord_list_011 = [[] for i in range(self.core_anchor_nelem)]
        self.lig_dir_vec_001 = [[] for i in range(self.core_anchor_nelem)]
        self.lig_dir_vec_111 = [[] for i in range(self.core_anchor_nelem)]
        self.lig_dir_vec_011 = [[] for i in range(self.core_anchor_nelem)]
        for i in range(self.core_anchor_nelem):
            print('search for anchor element ', self.core_anchor_elem_list[i])
            for j in range(len(self.core_anchor_id_list_001[i])):
                eid = self.core_anchor_id_list_001[i][j][0]
                aid = self.core_anchor_id_list_001[i][j][1]
                coord_c = self.core_coord_list[eid][aid]
                dist_abs = [abs(coord_c[i]) for i in range(3)]
                max_id = dist_abs.index(max(dist_abs))
                if (max_id == 0):
                    vec_perp = [1.00, 0.00, 0.00]
                elif (max_id == 1):
                    vec_perp = [0.00, 1.00, 0.00]
                elif (max_id == 2):
                    vec_perp = [0.00, 0.00, 1.00]
                vec_sign = np.sign(coord_c[max_id])
                coord_lig = coord_c[:]
                coord_lig[max_id] = coord_lig[max_id] + lig_dist_001 * vec_sign
                self.lig_anchor_coord_list_001[i].append(coord_lig)
                dir_vec = [(vec_perp[kk] * vec_sign) for kk in range(3)]
                self.lig_dir_vec_001[i].append(dir_vec)
            print('number of candidate anchors on {001} surface = ', len(self.lig_anchor_coord_list_001[i]))
            for j in range(len(self.core_anchor_id_list_111[i])):
                dist_cc_target = core_bond_len * math.sqrt(2.0)
                eid1 = self.core_anchor_id_list_111[i][j][0]
                aid1 = self.core_anchor_id_list_111[i][j][1]
                coord_c1 = self.core_coord_list[eid1][aid1]
                for k in range(j+1, len(self.core_anchor_id_list_111[i])):
                    eid2 = self.core_anchor_id_list_111[i][k][0]
                    aid2 = self.core_anchor_id_list_111[i][k][1]
                    coord_c2 = self.core_coord_list[eid2][aid2]
                    for p in range(k+1, len(self.core_anchor_id_list_111[i])):
                        eid3 = self.core_anchor_id_list_111[i][p][0]
                        aid3 = self.core_anchor_id_list_111[i][p][1]
                        coord_c3 = self.core_coord_list[eid3][aid3]
                        vec12 = np.subtract(coord_c2, coord_c1)
                        vec13 = np.subtract(coord_c3, coord_c1)
                        vec23 = np.subtract(coord_c3, coord_c2)
                        dist_cc12 = np.linalg.norm(vec12)
                        dist_cc13 = np.linalg.norm(vec13)
                        dist_cc23 = np.linalg.norm(vec23)
                        if (abs(dist_cc12 - dist_cc_target) < bond_len_tol and \
                            abs(dist_cc13 - dist_cc_target) < bond_len_tol and \
                            abs(dist_cc23 - dist_cc_target) < bond_len_tol):
                            cc_center = np.average([coord_c1, coord_c2, coord_c3], axis=0)
                            vec = np.cross(vec12, vec13)
                            vec_perp = vec / np.linalg.norm(vec)
                            vec_sign = [np.sign(vec_perp[ii] * cc_center[ii]) for ii in range(3)]
                            if (vec_sign[0] != vec_sign[1] or vec_sign[0] != vec_sign[2] or vec_sign[1] != vec_sign[2]):
                                'Error: fail to determine normal for {111} surface!'
                            else:
                                coord_lig = [(cc_center[ii] + vec_perp[ii] * vec_sign[ii] * lig_dist_111) for ii in range(3)]
                                self.lig_anchor_coord_list_111[i].append(coord_lig)
                                dir_vec = [(vec_perp[kk] * vec_sign[kk]) for kk in range(3)]
                                self.lig_dir_vec_111[i].append(dir_vec)
            print('number of candidate anchors on {111} surface = ', len(self.lig_anchor_coord_list_111[i]))
            for j in range(len(self.core_anchor_id_list_011[i])):
                dist_cc_target12 = core_bond_len * math.sqrt(2.0)
                dist_cc_target23 = core_bond_len * math.sqrt(2.0)
                dist_cc_target13 = core_bond_len * math.sqrt(2.0) * 2.0
                dist_cc_target24 = core_bond_len * 2.0
                eid1 = self.core_anchor_id_list_011[i][j][0]
                aid1 = self.core_anchor_id_list_011[i][j][1]
                coord_c1 = self.core_coord_list[eid1][aid1]
                for k in range(1, len(self.core_anchor_id_list_011[i])):
                    eid2 = self.core_anchor_id_list_011[i][k][0]
                    aid2 = self.core_anchor_id_list_011[i][k][1]
                    coord_c2 = self.core_coord_list[eid2][aid2]
                    for p in range(len(self.core_anchor_id_list_011[i])):
                        eid3 = self.core_anchor_id_list_011[i][p][0]
                        aid3 = self.core_anchor_id_list_011[i][p][1]
                        coord_c3 = self.core_coord_list[eid3][aid3]
                        vec12 = np.subtract(coord_c2, coord_c1)
                        dist_cc12 = np.linalg.norm(vec12)
                        vec13 = np.subtract(coord_c3, coord_c1)
                        dist_cc13 = np.linalg.norm(vec13)
                        vec23 = np.subtract(coord_c3, coord_c2)
                        dist_cc23 = np.linalg.norm(vec23)
                        if (abs(dist_cc12 - dist_cc_target12) < bond_len_tol and \
                               abs(dist_cc13 - dist_cc_target13) < bond_len_tol and \
                               abs(dist_cc23 - dist_cc_target23) < bond_len_tol):
                            cc_center1 = np.average([coord_c2, coord_c1], axis=0)
                            cc_center2 = np.average([coord_c2, coord_c3], axis=0)
                            for q in range(p+1, len(self.core_anchor_id_list_011[i])):
                                eid4 = self.core_anchor_id_list_011[i][q][0]
                                aid4 = self.core_anchor_id_list_011[i][q][1]
                                coord_c4 = self.core_coord_list[eid4][aid4]
                                vec24 = np.subtract(coord_c4, coord_c2)
                                dist_cc24 = np.linalg.norm(vec24)
                                if (abs(dist_cc24 - dist_cc_target24) < bond_len_tol):
                                    vec = np.cross(vec24, vec12)
                                    vec_perp = vec / np.linalg.norm(vec)
                                    vec_sign = [np.sign(vec_perp[ii] * coord_c2[ii]) for ii in range(3)]
                                    if ((vec_perp[2] < tiny and vec_sign[0] == vec_sign[1]) or \
                                            (vec_perp[1] < tiny and vec_sign[0] == vec_sign[2]) or \
                                            (vec_perp[0] < tiny and vec_sign[1] == vec_sign[2])):
                                        coord_lig1 = [(cc_center1[ii] + vec_perp[ii] * vec_sign[ii] * lig_dist_011) for ii in range(3)]
                                        coord_lig2 = [(cc_center2[ii] + vec_perp[ii] * vec_sign[ii] * lig_dist_011) for ii in range(3)]
                                        self.lig_anchor_coord_list_011[i].append(coord_lig1)
                                        self.lig_anchor_coord_list_011[i].append(coord_lig2)
                                        dir_vec1 = [(vec_perp[kk] * vec_sign[kk]) for kk in range(3)]
                                        dir_vec2 = dir_vec1[:]
                                        self.lig_dir_vec_011[i].append(dir_vec1)
                                        self.lig_dir_vec_011[i].append(dir_vec2)
                                    else:
                                        print('Error: fail to determine normal for {011} surface!')
                                    break
            print('number of candidate anchors on {011} surface = ', len(self.lig_anchor_coord_list_011[i]))
        print('\n')
    
#-------------------------------------------------------------------------------
# Attach ligands to the dot surface.
class Attach_Ligand:

    def __init__(self, lig_name, lig_del_atom_id_list, lig_ori_orig_ref_id, overlap_dist, core_elem_list, core_elem_stoi, \
                 lig_elem_list, lig_stoi, stoi_target, core_coord_list, lig_anchor_coord_list_001, lig_anchor_coord_list_111, \
                 lig_anchor_coord_list_011, lig_dir_vec_001, lig_dir_vec_111, lig_dir_vec_011, anchor_ntry_max, tiny):
        self.lig_name = lig_name
        self.lig_del_atom_id_list = lig_del_atom_id_list
        self.lig_ori_orig_ref_id = lig_ori_orig_ref_id
        self.overlap_dist = overlap_dist
        self.core_elem_list = core_elem_list
        self.core_elem_stoi = core_elem_stoi
        self.lig_elem_list = lig_elem_list
        self.lig_stoi = lig_stoi
        self.stoi_target = stoi_target
        self.core_coord_list = core_coord_list
        self.lig_anchor_coord_list_001 = lig_anchor_coord_list_001[0]
        self.lig_anchor_coord_list_111 = lig_anchor_coord_list_111[0]
        self.lig_anchor_coord_list_011 = lig_anchor_coord_list_011[0]
        self.lig_dir_vec_001 = lig_dir_vec_001[0]
        self.lig_dir_vec_111 = lig_dir_vec_111[0]
        self.lig_dir_vec_011 = lig_dir_vec_011[0]
        self.tiny = tiny
        self.lig_nelem = len(self.lig_elem_list)
        self.lig_coord_list = [[] for i in range(self.lig_nelem)]
        self.lig_ori_ref = [0, 0, 1]
        self.lig_anchor_ref = [0.00 for i in range(3)]
        self.anchor_ntry_max = anchor_ntry_max
        
    def rot_matrix_axis_angle(self, axis, theta):
        axis = np.asarray(axis)
        axis = axis / np.linalg.norm(axis)
        a = math.cos(theta / 2.0)
        b, c, d = axis * math.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2.0 * (bc-ad), 2.0 * (bd+ac)],
                     [2.0 * (bc+ad), aa+cc-bb-dd, 2.0 * (cd-ab)],
                     [2.0 * (bd-ac), 2.0 * (cd+ab), aa+dd-bb-cc]])

    def sel_id_rand_mask_list(self, list_old, list_mask):
        list_new = []
        list_old_id = []
        for i in range(len(list_old)):
            if (list_mask[i] == 0):
                list_new.append(list_old[i])
                list_old_id.append(i)
        new_id = random.randint(0, len(list_new) - 1)
        old_id = list_old_id[new_id]
        return old_id

    def add_new_lig(self, lig_anchor_coord, lig_dir_vec):
        self.new_lig_atom_coord = []
        self.new_lig_atom_elem = self.lig_atom_elem[:]
        ref_coord = self.lig_atom_coord[:]
        natom = len(self.lig_atom_coord)
        coord_rot = []
        axis = self.lig_ori_ref[:]
        angle = random.uniform(0.0, 2.0 * math.pi)
        for i in range(natom):
            vec1 = np.asarray(self.lig_atom_coord[i])
            vec2 = np.dot(self.rot_matrix_axis_angle(axis, angle), vec1)
            coord_rot.append(vec2)
        lig_dir_vec = np.asarray(lig_dir_vec)
        lig_dir_vec = lig_dir_vec / np.linalg.norm(lig_dir_vec)
        lig_ori_orig = self.lig_ori_ref[:]
        lig_ori_orig = np.asarray(lig_ori_orig)
        lig_ori_orig = lig_ori_orig / np.linalg.norm(lig_ori_orig)
        vec = np.cross(lig_ori_orig, lig_dir_vec)
        if (np.linalg.norm(vec) < self.tiny):
            temp_vec = np.asarray([1.0, 1.0, 1.0])
            temp_vec = temp_vec / np.linalg.norm(temp_vec)
            vec = np.cross(lig_ori_orig, temp_vec)
        axis = vec / np.linalg.norm(vec)
        angle = np.arccos(np.dot(lig_ori_orig, lig_dir_vec) / (np.linalg.norm(lig_ori_orig) * np.linalg.norm(lig_dir_vec)))
        for i in range(natom):
            vec1 = np.asarray(coord_rot[i])
            vec2 = np.dot(self.rot_matrix_axis_angle(axis, angle), vec1)
            vec3 = np.add(vec2, lig_anchor_coord) 
            self.new_lig_atom_coord.append(vec3)

    def check_new_lig_overlap(self):
        num_overlap = 0
        new_coord = self.new_lig_atom_coord
        old_coord = []
        for i in range(len(self.core_coord_list)):
            for j in range(len(self.core_coord_list[i])):
                old_coord.append(self.core_coord_list[i][j])
        for i in range(len(new_coord)):
            if (i != self.lig_anchor_atom_id):
                for j in range(len(old_coord)):
                    vec = np.subtract(new_coord[i], old_coord[j])
                    dist = np.linalg.norm(vec)
                    if (dist < self.overlap_dist):
                        num_overlap = num_overlap + 1
        old_coord = []
        for i in range(len(self.lig_atom_coord_list)):
            for j in range(len(self.lig_atom_coord_list[i])):
                old_coord.append(self.lig_atom_coord_list[i][j])
        for i in range(len(new_coord)):
            for j in range(len(old_coord)):
                vec = np.subtract(new_coord[i], old_coord[j])
                dist = np.linalg.norm(vec)
                if (dist < self.overlap_dist):
                    num_overlap = num_overlap + 1
        if (num_overlap < 1):
            qoverlap = False
        else:
            qoverlap = True
        return qoverlap        

    def init_lig(self):
        filename = './ligand-struct/' + lig_name + '.xyz'
        with open(filename, 'r') as f:
            mylist = f.read().splitlines()
        lig_natom_orig = int(mylist[0])
        lig_atom_elem_orig = []
        lig_coord_orig = []
        for i in range(2, lig_natom_orig+2):
            line = mylist[i].split()
            lig_atom_elem_orig.append(line[0])
            coord = [float(line[1]), float(line[2]), float(line[3])]
            lig_coord_orig.append(coord)
        self.lig_natom = lig_natom_orig - len(self.lig_del_atom_id_list)
        lig_atom_elem = []
        lig_coord = []
        ref_coord = lig_coord_orig[self.lig_ori_orig_ref_id[0]]
        if (self.lig_ori_orig_ref_id[0] == self.lig_ori_orig_ref_id[1]):
            self.lig_anchor_atom_id = self.lig_ori_orig_ref_id[0] 
            for i in range(lig_natom_orig):
                lig_atom_elem.append(lig_atom_elem_orig[i])
                vec1 = np.subtract(lig_coord_orig[i], ref_coord)
                lig_coord.append(vec1)
        else:
            vec = np.subtract(lig_coord_orig[self.lig_ori_orig_ref_id[1]], lig_coord_orig[self.lig_ori_orig_ref_id[0]]) 
            lig_ori_orig = vec / np.linalg.norm(vec)
            ref_vec = self.lig_ori_ref
            vec = np.cross(lig_ori_orig, ref_vec)
            axis = vec / np.linalg.norm(vec)
            angle = np.arccos(np.dot(lig_ori_orig, ref_vec) / (np.linalg.norm(lig_ori_orig) * np.linalg.norm(ref_vec)))
            atom_id = -1
            for i in range(lig_natom_orig):
                if (self.lig_del_atom_id_list.count(i) == 0):
                    lig_atom_elem.append(lig_atom_elem_orig[i])
                    vec1 = np.subtract(lig_coord_orig[i], ref_coord)
                    vec2 = np.dot(self.rot_matrix_axis_angle(axis, angle), vec1)
                    lig_coord.append(vec2)
                    atom_id = atom_id + 1
                    if (i == self.lig_ori_orig_ref_id[0]):
                        self.lig_anchor_atom_id = atom_id
            axis = ref_vec[:]
            angle = random.uniform(0.0, 2.0 * math.pi)
            for i in range(self.lig_natom):
                vec1 = np.asarray(lig_coord[i])
                vec2 = np.dot(self.rot_matrix_axis_angle(axis, angle), vec1)
                lig_coord[i] = vec2
        filename = './ligand-struct/' + lig_name + '.rot.xyz'
        with open(filename, 'w') as f:
            f.write(str(self.lig_natom) + '\n')
            f.write('\n')
            for i in range(self.lig_natom):
                f.write('{0:8} {1:.5f}  {2:.5f}  {3:.5f} \n'.format(lig_atom_elem[i], lig_coord[i][0], \
                            lig_coord[i][1], lig_coord[i][2]))
        self.lig_nelem = len(self.lig_elem_list)
        self.lig_atom_elem = lig_atom_elem[:]
        self.lig_atom_coord = lig_coord[:]
                
    def add_lig(self, temper, kb, ev, lig_bind_en_001, lig_bind_en_111, lig_bind_en_011):
        kbt = kb * temper / ev
        prob_tot = math.exp(lig_bind_en_001 / kbt) + math.exp(lig_bind_en_111 / kbt) + math.exp(lig_bind_en_011 / kbt)
        prob_001 = math.exp(lig_bind_en_001 / kbt) / prob_tot
        prob_111 = math.exp(lig_bind_en_111 / kbt) / prob_tot
        prob_011 = math.exp(lig_bind_en_011 / kbt) / prob_tot
        prob_list = [prob_001, prob_111, prob_011]
        core_elem_natom = [len(core_coord_list[i]) for i in range(len(core_elem_list))]
        core_stoi = np.dot(core_elem_natom, self.core_elem_stoi)
        nlig_target = int(round((stoi_target - core_stoi) / lig_stoi))
        if (nlig_target <= 0):
            self.nlig = nlig_target
            return
        lig_anchor_coord_list = [self.lig_anchor_coord_list_001, self.lig_anchor_coord_list_111, self.lig_anchor_coord_list_011]
        lig_dir_vec_list = [self.lig_dir_vec_001, self.lig_dir_vec_111, self.lig_dir_vec_011]
        nanchor_max_list = [len(lig_anchor_coord_list[i]) for i in range(3)]
        lig_anchor_coord_list_mask = [[0 for j in range(nanchor_max_list[i])] for i in range(3)]
        self.lig_anchor_coord_sel_list = [[] for i in range(3)]
        self.lig_atom_coord_list = []
        self.lig_atom_elem_list = []
        print('target number of ligands = ', nlig_target)
#        nlig_target = 10
        surf_try_mask = [0 for i in range(3)]
        for i in range(3):
            if (nanchor_max_list[i] == 0):
                surf_try_mask[i] = 1
        for lig_id in range(nlig_target):
            if (sum(surf_try_mask) == 3):
                break
            istat = 0
            ntry = 0
            ntry1 = 0
            while (istat == 0 and ntry < 1e4):
                if (sum(surf_try_mask) == 3):
                    break
                rand_num = random.random()
                if (sum(surf_try_mask) == 0):
                    if (rand_num < prob_list[0]):
                        sid = 0
                    elif (rand_num < prob_list[0] + prob_list[1]):
                        sid = 1
                    else:
                        sid = 2
                elif (sum(surf_try_mask) == 1):
                    if (surf_try_mask[0] == 1):
                        if (rand_num < (prob_list[1] / (prob_list[1] + prob_list[2]))):
                            sid = 1
                        else:
                            sid = 2
                    elif (surf_try_mask[1] == 1):
                        if (rand_num < (prob_list[0] / (prob_list[0] + prob_list[2]))):
                            sid = 0
                        else:
                            sid = 2
                    else:
                        if (rand_num < (prob_list[0] / (prob_list[0] + prob_list[1]))):
                            sid = 0
                        else:
                            sid = 1
                elif (sum(surf_try_mask) == 2):
                    if (surf_try_mask[0] == 0):
                        sid = 0
                    elif (surf_try_mask[1] == 0):
                        sid = 1
                    else:
                        sid = 2
                if (surf_try_mask[sid] == 0):
                    while (istat == 0 and surf_try_mask[sid] == 0):
                        aid = self.sel_id_rand_mask_list(lig_anchor_coord_list[sid], lig_anchor_coord_list_mask[sid])
                        lig_anchor_coord_list_mask[sid][aid] = 1
                        lig_anchor_coord_sel = lig_anchor_coord_list[sid][aid]
                        lig_dir_vec_sel = lig_dir_vec_list[sid][aid]
                        ntry1 = ntry1 + 1 
                        ntry2 = 0
                        while (istat == 0 and ntry2 < self.anchor_ntry_max):
                            ntry2 = ntry2 + 1
                            self.add_new_lig(lig_anchor_coord_sel, lig_dir_vec_sel)
                            qoverlap = self.check_new_lig_overlap()
                            if (qoverlap == False):
                                self.lig_atom_coord_list.append(self.new_lig_atom_coord)
                                self.lig_atom_elem_list.append(self.new_lig_atom_elem)
                                self.lig_anchor_coord_sel_list[sid].append(lig_anchor_coord_sel)
                                istat = 1
                        if (sum(lig_anchor_coord_list_mask[sid]) == nanchor_max_list[sid]):
                            surf_try_mask[sid] = 1
                            if (sid == 0):
                                print('warning: candidate anchors on {001} surface used up.')
                            elif (sid == 1):
                                print('warning: candidate anchors on {111} surface used up.')
                            else:
                                print('warning: candidate anchors on {011} surface used up.')
                ntry = ntry + 1
            print('add ligand ', lig_id, 'with ', ntry1, 'trails on anchor.')


        nlig_real = len(self.lig_atom_coord_list)
        self.nlig = nlig_real
        self.lig_elem_natom = [0 for i in range(self.lig_nelem)]
        for i in range(nlig_real):
            for j in range(len(self.lig_atom_coord_list[i])):
                eid = self.lig_elem_list.index(self.lig_atom_elem_list[i][j])
                self.lig_elem_natom[eid] = self.lig_elem_natom[eid] + 1
                self.lig_coord_list[eid].append(self.lig_atom_coord_list[i][j])
        self.stoi_real = core_stoi + lig_stoi * float(nlig_real)
        self.prob_001_real = float(len(self.lig_anchor_coord_sel_list[0])) / float(nlig_real)
        self.prob_111_real = float(len(self.lig_anchor_coord_sel_list[1])) / float(nlig_real)
        self.prob_011_real = float(len(self.lig_anchor_coord_sel_list[2])) / float(nlig_real)
        print('target total stoichiometry = ', stoi_target)
        print('target number of ligands = ', nlig_target)
        print('target ligand probablity on {001} = ', prob_001)
        print('target ligand probablity on {111} = ', prob_111)
        print('target ligand probablity on {011} = ', prob_011)
        print('real total stoichiometry = ', self.stoi_real)
        print('real number of ligands = ', nlig_real)
        print('real ligand probablity on {001} = ', self.prob_001_real)
        print('real ligand probablity on {111} = ', self.prob_111_real)
        print('real ligand probablity on {011} = ', self.prob_011_real)
                
#-------------------------------------------------------------------------------
# Write structure to file.
class Output_Struct:

    def __init__(self, core_elem_list, lig_elem_list, core_coord_list, lig_coord_list, conf_id, lig_name, nlig):
        self.core_elem_list = core_elem_list
        self.lig_elem_list = lig_elem_list
        self.core_coord_list = core_coord_list
        self.lig_coord_list = lig_coord_list
        self.core_nelem = len(self.core_elem_list)
        self.lig_nelem = len(self.lig_elem_list)
        self.conf_id = conf_id
        self.lig_name = lig_name
        self.nlig = nlig

    def combine_core_lig(self):
        self.elem_list = self.core_elem_list + self.lig_elem_list
        self.coord_list = self.core_coord_list + self.lig_coord_list
        self.nelem = len(self.elem_list)
        self.elem_natom = []
        for i in range(self.nelem):
            if (self.coord_list[i] == [[]]):
                self.elem_natom.append(0)
            else:
                self.elem_natom.append(len(self.coord_list[i]))
        self.natom = sum(self.elem_natom)
        self.sysname = ''
        for i in range(self.core_nelem):
            if (len(self.core_coord_list[i]) != 0):
                self.sysname = self.sysname + self.core_elem_list[i] + str(len(self.core_coord_list[i]))
        if (self.nlig != 0):
            self.sysname = self.sysname + self.lig_name + str(self.nlig)
        for i in range(10000):
            filename1 = './struct-xyz/' + self.sysname + '.cf-' + str(i+1) + '.xyz'
            filename2 = './struct-poscar/' + self.sysname + '.cf-' + str(i+1) + '.vasp'
            if ((os.path.isfile(filename1) == False or os.path.isfile(filename2) == False) and i == 0):
                self.sysid = 1
                break
            elif (os.path.isfile(filename1) == False or os.path.isfile(filename2) == False):
                self.sysid = i+1
                break
        self.sysname = self.sysname + '.cf-' + str(self.sysid)
        print('system name = ', self.sysname)
        self.elem_red_list = []
        for i in range(self.nelem):
            if (self.elem_red_list.count(self.elem_list[i]) == 0):
                self.elem_red_list.append(self.elem_list[i])
        self.nelem_red = len(self.elem_red_list)
        self.elem_red_natom = [0 for i in range(self.nelem_red)]
        self.coord_red_list = [[] for i in range(self.nelem_red)]
        for i in range(self.nelem):
            for j in range(self.nelem_red):
                if (self.elem_list[i] == self.elem_red_list[j]):
                    self.elem_red_natom[j] = self.elem_red_natom[j] + self.elem_natom[i]
                    for k in range(self.elem_natom[i]):
                        self.coord_red_list[j].append(self.coord_list[i][k])
                    break
        coord_dir = [[] for i in range(3)]
        for i in range(self.nelem):
            for j in range(self.elem_natom[i]):
                for k in range(3):
                    coord_dir[k].append(self.coord_list[i][j][k])
        coord_reg = [(max(coord_dir[i]) - min(coord_dir[i])) for i in range(3)]
        self.box = [(coord_reg[i] + box_vac) for i in range(3)]
        self.frac_coord_red_list = [[] for i in range(self.nelem_red)]
        for i in range(self.nelem_red):
            for j in range(self.elem_red_natom[i]):
                temp_coord = []
                for k in range(3):
                    temp_coord.append(self.coord_red_list[i][j][k] / self.box[k])
                self.frac_coord_red_list[i].append(temp_coord)

    def write_xyz(self):
        path = 'struct-xyz'
        if not os.path.exists(path):
            os.makedirs(path)
        filename = path + '/' + self.sysname + '.xyz'
        with open(filename, 'w') as f:
            f.write(str(self.natom) + '\n')
            f.write('\n')
            for i in range(self.nelem):
                for j in range(self.elem_natom[i]):
                    f.write('{0:8} {1:.5f}  {2:.5f}  {3:.5f} \n'.format(self.elem_list[i], self.coord_list[i][j][0], \
                        self.coord_list[i][j][1], self.coord_list[i][j][2]))

    def write_poscar(self):
        path = 'struct-poscar'
        if not os.path.exists(path):
            os.makedirs(path)
        filename = path + '/' + self.sysname + '.vasp'
        with open(filename, 'w') as f:
            f.write(self.sysname + '\n')
            f.write('1.0000000  \n')
            f.write('    {0:10.5f}  {1:10.5f}  {2:10.5f} \n'.format(self.box[0], 0.0, 0.0))
            f.write('    {0:10.5f}  {1:10.5f}  {2:10.5f} \n'.format(0.0, self.box[1], 0.0))
            f.write('    {0:10.5f}  {1:10.5f}  {2:10.5f} \n'.format(0.0, 0.0, self.box[2]))
            f.write(''.join(str(component)+'  ' for component in self.elem_red_list))
            f.write('\n')
            f.write(''.join(str(component)+'  ' for component in self.elem_red_natom))
            f.write('\n' + 'Direct' + '\n')
            for i in range(self.nelem_red):
                for j in range(self.elem_red_natom[i]):
                    temp_coord = self.frac_coord_red_list[i][j]
                    f.write('{0:.5f}  {1:.5f}  {2:.5f} \n'.format(temp_coord[0],temp_coord[1], temp_coord[2]))

    def write_info(self, core_shape, stoi_real, stoi_target, prob_001_real, prob_111_real, prob_011_real, dot_diameter, \
                   core_elem_nve, lig_elem_nve, center_shift, gamma_001, gamma_111, gamma_011, h_001, lig_bind_en_001, \
                   lig_bind_en_111, lig_bind_en_011, lig_dist_001, lig_dist_111, lig_dist_011, overlap_dist, box_vac):
        self.core_elem_natom = [len(self.core_coord_list[i]) for i in range(self.core_nelem)]
        self.lig_elem_natom = [len(self.lig_coord_list[i]) for i in range(self.lig_nelem)]
        natom_tot = sum(self.core_elem_natom) + sum(self.lig_elem_natom)
        nve_core = np.dot(self.core_elem_natom, core_elem_nve)
        nve_core_nod = np.dot(self.core_elem_natom, core_elem_nve_nod)
        nve_lig = np.dot(self.lig_elem_natom, lig_elem_nve)
        nve_tot = nve_core + nve_lig
        nvb_tot = math.ceil(float(nve_tot) / 2.0)
        nve_tot_nod = nve_core_nod + nve_lig
        nvb_tot_nod = math.ceil(float(nve_tot_nod) / 2.0)
        core_elem_natom_str = ''
        core_elem_list_str = ''
        for i in range(self.core_nelem):
            core_elem_list_str = core_elem_list_str + self.core_elem_list[i] + '  '
            core_elem_natom_str = core_elem_natom_str + str(self.core_elem_natom[i]) + '  '
        lig_elem_natom_str = ''
        lig_elem_list_str = ''
        for i in range(self.lig_nelem):
            lig_elem_list_str = lig_elem_list_str + self.lig_elem_list[i] + '  '
            lig_elem_natom_str = lig_elem_natom_str + str(self.lig_elem_natom[i]) + '  '
        path = 'struct-info'
        if not os.path.exists(path):
            os.makedirs(path)
        filename = path + '/' + self.sysname + '.info.dat'
        with open(filename, 'w') as f:
            f.write(self.sysname + '\n')
            f.write('core shape: \n')
            f.write(core_shape + '\n')
            f.write('core diameter: \n')
            f.write(' {0:.3f} \n'.format(dot_diameter))
            f.write('total number of atoms: \n')
            f.write(' {0:5d} \n'.format(natom_tot))
            f.write('number of ligs: \n')
            f.write(' {0:5d} \n'.format(self.nlig))
            f.write('total stoichiometry: \n')
            f.write(' {0:.3f} \n'.format(stoi_real))
            f.write('number of valence electrons: \n')
            f.write(' {0:5d} \n'.format(nve_tot))
            f.write('number of valence bands: \n')
            f.write(' {0:5d} \n'.format(nvb_tot))
            f.write('number of valence electrons without d orbitals: \n')
            f.write(' {0:5d} \n'.format(nve_tot_nod))
            f.write('number of valence bands without d orbitals: \n')
            f.write(' {0:5d} \n'.format(nvb_tot_nod))
            f.write(' \n')
            f.write('proportion of ligand attached on {001} surface: \n')
            f.write(' {0:.3f} \n'.format(prob_001_real))
            f.write('proportion of ligand attached on {111} surface: \n')
            f.write(' {0:.3f} \n'.format(prob_111_real))
            f.write('proportion of ligand attached on {011} surface: \n')
            f.write(' {0:.3f} \n'.format(prob_011_real))
            f.write('number of core elements: \n')
            f.write(' {0:5d} \n'.format(self.core_nelem))
            f.write('core element list: \n')
            f.write(core_elem_list_str + '\n')
            f.write('number of atom per core element: \n')
            f.write(core_elem_natom_str + '\n')
            f.write('number of lig elements: \n')
            f.write(' {0:5d} \n'.format(self.lig_nelem))
            f.write('lig element list: \n')
            f.write(lig_elem_list_str + '\n')
            f.write('number of atom per lig element: \n')
            f.write(lig_elem_natom_str + '\n')
            f.write(' \n')
            f.write('input parameter to build the structure: \n')
            f.write('stoi_target = {0:.3f} \n'.format(stoi_target))
            f.write('gamma_001 = {0:.3f} \n'.format(gamma_001))
            f.write('gamma_111= {0:.3f} \n'.format(gamma_111))
            f.write('gamma_011 = {0:.3f} \n'.format(gamma_011))
            f.write('h_001 = {0:.3f} \n'.format(h_001))
            f.write('lig_bind_en_001 = {0:.3f} \n'.format(lig_bind_en_001))
            f.write('lig_bind_en_111 = {0:.3f} \n'.format(lig_bind_en_111))
            f.write('lig_bind_en_011 = {0:.3f} \n'.format(lig_bind_en_011))
            f.write('lig_dist_001 = {0:.3f} \n'.format(lig_dist_001))
            f.write('lig_dist_111= {0:.3f} \n'.format(lig_dist_111))
            f.write('lig_dist_011 = {0:.3f} \n'.format(lig_dist_011))
            f.write('overlap_dist = {0:.3f} \n'.format(overlap_dist))
            f.write('box_vac = {0:.3f} \n'.format(box_vac))
                
                
#-------------------------------------------------------------------------------
#------------------------------------------------------------------------------- 
# Main program.

random.seed()
gen_core = Gen_Core(core_elem_list)
gen_core.bulk_gen()
gen_core.wulff_cut()
#gen_core.sphere_cut()
gen_core.modify_surface(core_bond_len, bond_len_tol)
core_shape = gen_core.core_shape
core_coord_list = gen_core.core_coord_list
dot_diameter = gen_core.dot_diameter
find_anchor = Find_Anchor(core_elem_list, core_coord_list, core_anchor_elem_list)
find_anchor.search_core_anchor(dist_to_surface_tol)
find_anchor.search_lig_anchor(core_bond_len, bond_len_tol, lig_dist_001, lig_dist_111, lig_dist_011, tiny)
lig_anchor_coord_list_001 = find_anchor.lig_anchor_coord_list_001
lig_anchor_coord_list_111 = find_anchor.lig_anchor_coord_list_111
lig_anchor_coord_list_011 = find_anchor.lig_anchor_coord_list_011
lig_dir_vec_001 = find_anchor.lig_dir_vec_001
lig_dir_vec_111 = find_anchor.lig_dir_vec_111
lig_dir_vec_011 = find_anchor.lig_dir_vec_011
print('\n')
print('generate ', nconfig, ' passivant configurations')
print('\n')
for conf_id in range(nconfig):
    print ('generating configuration ', conf_id+1)
    attach_ligand = Attach_Ligand(lig_name, lig_del_atom_id_list, lig_ori_orig_ref_id, overlap_dist, core_elem_list, core_elem_stoi, \
                     lig_elem_list, lig_stoi, stoi_target, core_coord_list, lig_anchor_coord_list_001, lig_anchor_coord_list_111, \
                     lig_anchor_coord_list_011, lig_dir_vec_001, lig_dir_vec_111, lig_dir_vec_011, anchor_ntry_max, tiny)
    attach_ligand.init_lig()
    attach_ligand.add_lig(temper, kb, ev, lig_bind_en_001, lig_bind_en_111, lig_bind_en_011)
    nlig = attach_ligand.nlig
    if (nlig < 0):
        print('target number of ligands = ',nlig)
        print('Stop!')
        break
    elif (nlig == 0):
        stoi_real = 0
        prob_001_real = 0.0
        prob_111_real = 0.0
        prob_011_real = 0.0
        lig_coord_list = [[] for i in range(len(lig_elem_list))] 
    else:
        stoi_real = attach_ligand.stoi_real
        prob_001_real = attach_ligand.prob_001_real
        prob_111_real = attach_ligand.prob_111_real
        prob_011_real = attach_ligand.prob_011_real
        lig_coord_list = attach_ligand.lig_coord_list
    output_struct = Output_Struct(core_elem_list, lig_elem_list, core_coord_list, lig_coord_list, conf_id, lig_name, nlig)
    output_struct.combine_core_lig()
    output_struct.write_xyz()
    output_struct.write_poscar()
    output_struct.write_info(core_shape, stoi_real, stoi_target, prob_001_real, prob_111_real, prob_011_real, dot_diameter, \
                             core_elem_nve, lig_elem_nve, center_shift, gamma_001, gamma_111, gamma_011, h_001, lig_bind_en_001, \
                             lig_bind_en_111, lig_bind_en_011, lig_dist_001, lig_dist_111, lig_dist_011, overlap_dist, box_vac)
    if (nlig == 0):
        break
    print('\n')
if (nlig >= 0):
    print('element list = ', output_struct.elem_list)
    print('number of atoms per element = ', output_struct.elem_natom)
    print('reduced element list = ', output_struct.elem_red_list)
    print('number of atoms per reduced element = ', output_struct.elem_red_natom)










