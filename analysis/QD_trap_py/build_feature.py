"""
Define feature space for both classification and regression.
"""

import numpy as np
#import matplotlib.pyplot as plt
import random
import math
import os

__all__ = ['BuildFeature']

class BuildFeature:

    def __init__(self, nrecord, input_vars, input_var_labels, feat_data_list, feat_target_list, \
                 feat_param_list):
        self.nrecord = nrecord
        self.input_vars = input_vars
        self.input_var_labels = input_var_labels
        self.feat_data_list = feat_data_list
        self.feat_target_list = feat_target_list
        self.nfeat_data = len(feat_data_list)
        self.nfeat_target = len(feat_target_list)
        self.feat_param_list = feat_param_list
        self.tiny = 1e-4
        np.set_printoptions(precision=3)
        self.result_dir = 'analysis-result'
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

##-------------------------------------------------------------------------------------------------
    def analyze_struct(self, rebuild_nn_list):
        self.natom = []
        if ('dot_type_id' in self.input_var_labels):
            self.dot_type_id = np.array(self.input_vars[self.input_var_labels.index('dot_type_id')])
        if ('dot_type_name' in self.input_var_labels):
            self.dot_type_name = np.array(self.input_vars[self.input_var_labels.index('dot_type_name')])
        if ('cf_id' in self.input_var_labels):
            self.cf_id = np.array(self.input_vars[self.input_var_labels.index('cf_id')])
        if ('core_shape' in self.input_var_labels):
            self.core_shape = np.array(self.input_vars[self.input_var_labels.index('core_shape')])
        if ('core_diameter' in self.input_var_labels):
            self.core_diameter = np.array(self.input_vars[self.input_var_labels.index('core_diameter')])
        if ('core_elem_natom' in self.input_var_labels):
            self.core_elem_natom = np.array(self.input_vars[self.input_var_labels.index('core_elem_natom')])
        if ('nlig' in self.input_var_labels):
            self.nlig = np.array(self.input_vars[self.input_var_labels.index('nlig')])
        if ('stoi' in self.input_var_labels):
            self.stoi = np.array(self.input_vars[self.input_var_labels.index('stoi')])
        if ('pass_prop_001_old' in self.input_var_labels):
            self.pass_prop_001_old = np.array(self.input_vars[self.input_var_labels.index('pass_prop_001_old')])
        if ('pass_prop_111_old' in self.input_var_labels):
            self.pass_prop_111_old = np.array(self.input_vars[self.input_var_labels.index('pass_prop_111_old')])
        if ('pass_prop_011_old' in self.input_var_labels):
            self.pass_prop_011_old = np.array(self.input_vars[self.input_var_labels.index('pass_prop_011_old')])
        if ('elem_name' in self.input_var_labels):
            self.elem_name = np.array(self.input_vars[self.input_var_labels.index('elem_name')])
        if ('elem_natom' in self.input_var_labels):
            self.elem_natom = np.array(self.input_vars[self.input_var_labels.index('elem_natom')])
        if ('old_lc_const' in self.input_var_labels):
            self.old_lc_const = np.array(self.input_vars[self.input_var_labels.index('old_lc_const')])
        if ('old_lc_vec' in self.input_var_labels):
            self.old_lc_vec = np.array(self.input_vars[self.input_var_labels.index('old_lc_vec')])
        if ('old_frac_coord' in self.input_var_labels):
            self.old_frac_coord = np.array(self.input_vars[self.input_var_labels.index('old_frac_coord')])
        if ('new_lc_const' in self.input_var_labels):
            self.new_lc_const = np.array(self.input_vars[self.input_var_labels.index('new_lc_const')])
        if ('new_lc_vec' in self.input_var_labels):
            self.new_lc_vec = np.array(self.input_vars[self.input_var_labels.index('new_lc_vec')])
        if ('new_frac_coord' in self.input_var_labels):
            self.new_frac_coord = np.array(self.input_vars[self.input_var_labels.index('new_frac_coord')])
        for reid in range(self.nrecord):
            self.natom.append(sum(self.elem_natom[reid]))
        self.type_red_name = np.unique(self.dot_type_name)
        self.ntype_red = len(self.type_red_name)
        mid = [list(self.dot_type_name).index(self.type_red_name[i]) for i in range(self.ntype_red)]
        self.type_red_core_shape = [self.core_shape[mid[i]] for i in range(len(mid))]
        self.type_red_stoi = [self.stoi[mid[i]] for i in range(len(mid))]
        self.type_red_core_diameter = [self.core_diameter[mid[i]] for i in range(len(mid))]
        self.type_red_core_elem_natom = [self.core_elem_natom[mid[i]] for i in range(len(mid))]
        print("analyzing structures... \n")

        # Build nearest neighboring lists.
        if (rebuild_nn_list):
            print("building nearest neighboring lists... \n")
            self.nn_list_old = []
            self.nn_list_new = []
            self.num_nn_old = []
            self.num_nn_new = []
            bond_len_est = self.feat_param_list['bond_len_est']
            bond_len_tol = self.feat_param_list['bond_len_tol']
            self.nelem = len(bond_len_est) + 1
            bond_len_est_mat = [[0.0 for j in range(self.nelem)] for i in range(self.nelem)]
            bond_len_tol_mat = [[0.0 for j in range(self.nelem)] for i in range(self.nelem)]
            for i in range(0, self.nelem-1):
                for j in range(i+1, self.nelem):
                    bond_len_est_mat[i][j] = float(bond_len_est[i][j-i-1])
                    bond_len_tol_mat[i][j] = float(bond_len_tol[i][j-i-1])
                    bond_len_est_mat[j][i] = bond_len_est_mat[i][j]
                    bond_len_tol_mat[j][i] = bond_len_tol_mat[i][j]
            for reid in range(self.nrecord):
                print('{0}, '.format(reid), end='')
                elem_natom = self.elem_natom[reid]
                atom_elem_id = []
                for i in range(len(elem_natom)):
                    for j in range(elem_natom[i]):
                        atom_elem_id.append(i)
                lc_const = self.old_lc_const[reid]
                lc_vec = self.old_lc_vec[reid]
                frac_coords = self.old_frac_coord[reid]
                natom = len(frac_coords)
                coords = [[frac_coords[i][j] * lc_const * lc_vec[j][j] for j in range(3)] for i in range(natom)]
                nn_list_old = [[] for i in range(natom)]
                for atid1 in range(natom-1):
                    for atid2 in range(atid1+1, natom):
                        dist = np.linalg.norm(np.subtract(coords[atid2], coords[atid1]))
                        bl_est = bond_len_est_mat[atom_elem_id[atid1]][atom_elem_id[atid2]]
                        bl_tol = bond_len_tol_mat[atom_elem_id[atid1]][atom_elem_id[atid2]]
                        if (abs(dist - bl_est) < bl_tol):
                            nn_list_old[atid1].append(atid2)
                            nn_list_old[atid2].append(atid1)
                num_nn_old = [len(nn_list_old[i]) for i in range(natom)]
                self.nn_list_old.append(nn_list_old)
                self.num_nn_old.append(num_nn_old)
                lc_const = self.new_lc_const[reid]
                lc_vec = self.new_lc_vec[reid]
                frac_coords = self.new_frac_coord[reid]
                natom = len(frac_coords)
                coords = [[frac_coords[i][j] * lc_const * lc_vec[j][j] for j in range(3)] for i in range(natom)]
                nn_list_new = [[] for i in range(natom)]
                for atid1 in range(natom-1):
                    for atid2 in range(atid1+1, natom):
                        dist = np.linalg.norm(np.subtract(coords[atid2], coords[atid1]))
                        bl_est = bond_len_est_mat[atom_elem_id[atid1]][atom_elem_id[atid2]]
                        bl_tol = bond_len_tol_mat[atom_elem_id[atid1]][atom_elem_id[atid2]]
                        if (abs(dist - bl_est) < bl_tol):
                            nn_list_new[atid1].append(atid2)
                            nn_list_new[atid2].append(atid1)
                num_nn_new = [len(nn_list_new[i]) for i in range(natom)]
                self.nn_list_new.append(nn_list_new)
                self.num_nn_new.append(num_nn_new)
            fileout = './analysis-result/nn_list.dat'
            with open(fileout, 'w') as f11:
                f11.write("# self.nrecord (number of samples) \n")
                f11.write("# record id, number of atoms \n")
                f11.write("# nearest neighboring list before relaxation \n")
                f11.write("# nearest neighboring list after relaxation \n")
                f11.write("\n")
                f11.write("{0} \n".format(self.nrecord))
                f11.write("\n")
                for reid in range(self.nrecord):
                    f11.write("{0}   {1} \n".format(reid, self.natom[reid]))
                    for atid in range(self.natom[reid]):
                        temp_list = self.nn_list_old[reid][atid]
                        f11.write(" ".join("{0}".format(str(x)) for x in temp_list) + "\n")
                        temp_list = self.nn_list_new[reid][atid]
                        f11.write(" ".join("{0}".format(str(x)) for x in temp_list) + "\n")
                    f11.write("\n")                
            print("\n")
        else:
            print("read nn list from file nn_list.dat ...\n")
            self.nn_list_old = [[] for i in range(self.nrecord)]
            self.nn_list_new = [[] for i in range(self.nrecord)]
            filein = './analysis-result/nn_list.dat'
            with open(filein, 'r') as f12:
                for i in range(7):
                    f12.readline()
                for reid in range(self.nrecord):
                    f12.readline()
                    for atid in range(self.natom[reid]):
                        line = f12.readline().rstrip('\n')
                        str1 = line.split()
                        nn_list = [int(str1[j]) for j in range(len(str1))]
                        self.nn_list_old[reid].append(nn_list)
                        line = f12.readline().rstrip('\n')
                        str1 = line.split()
                        nn_list = [int(str1[j]) for j in range(len(str1))]
                        self.nn_list_new[reid].append(nn_list)
                    f12.readline()
            self.num_nn_old = [[len(self.nn_list_old[i][j]) for j in range(self.natom[i])] for i in range(self.nrecord)]
            self.num_nn_new = [[len(self.nn_list_new[i][j]) for j in range(self.natom[i])] for i in range(self.nrecord)]               

        # Find the location and bonding environment (coordination number) of each atom or ligand.
        # Options of core/ligand sites: 0:inside, 1:{001}, 2:{111}, 3:{011}, 4:{001}-{001} edge, 5:{111}-{111} edge, 6:{011}-{011} edge, 
        #                             7:{001}-{111} edge, 8:{001}-{011} edge, 9:{111-011} edge, 10:corner)
        # For 2-fold ligand, the edge site is defined by that both of the attached Pb atoms are at edge, while the corner site is defined\
        # by that at least one of the Pb is at the corner while the other is at edge.
        # For 3-fold ligand, the edge site is defined by that at least two of the attached Pb atoms are at edge, while the corner site \
        # is defined by that at least one of the Pb is at the corner while the others are at edge.
        print("assigning atom types...\n")
        self.core_atom_site = [[] for i in range(self.nrecord)]
        self.core_surf_site_prop = []
        self.lig_site = [[] for i in range(self.nrecord)]
        self.lig_codnum = [[] for i in range(self.nrecord)]
        self.lig_site_prop = []
        self.lig_codnum_prop = []
        ntype_site = 11
        for reid in range(self.nrecord):
            print('{0}, '.format(reid), end='')
            lc_const = self.old_lc_const[reid]
            lc_vec = self.old_lc_vec[reid]
            frac_coords = self.old_frac_coord[reid]
            natom = len(frac_coords)
            elem_natom = self.elem_natom[reid]
            coords = [[frac_coords[i][j] * lc_const * lc_vec[j][j] for j in range(3)] for i in range(natom)]
            s001_far = max([max([abs(coords[i][j]) for j in range(3)]) for i in range(elem_natom[0])])
            s111_far = max([(abs(coords[i][0]) + abs(coords[i][1]) + abs(coords[i][2])) for i in range(elem_natom[0])])
            s011_far = max([max([abs(coords[i][0])+abs(coords[i][1]), abs(coords[i][0])+abs(coords[i][2]), \
                                 abs(coords[i][1])+abs(coords[i][2])]) for i in range(elem_natom[0])])
            s001_mask = [0 for i in range(natom)]
            s111_mask = [0 for i in range(natom)]
            s011_mask = [0 for i in range(natom)]
            ds_tol = 1e-2
            for atid in range(elem_natom[0]+elem_natom[1]):
                tlist = [[1, 0, 0], [0, 1, 0], [0, 0, 1],[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
                s001_ds = [coords[atid][0]*tlist[i][0] + coords[atid][1]*tlist[i][1] + coords[atid][2]*tlist[i][2] \
                           for i in range(len(tlist))]
                tlist = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]
                s111_ds = [coords[atid][0]*tlist[i][0] + coords[atid][1]*tlist[i][1] + coords[atid][2]*tlist[i][2] \
                           for i in range(len(tlist))]
                tlist = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [-1, 1, 0], [1, 0, -1], [0, 1, -1]]
                s011_ds = [coords[atid][0]*tlist[i][0] + coords[atid][1]*tlist[i][1] + coords[atid][2]*tlist[i][2] \
                           for i in range(len(tlist))]
                for i in range(len(s001_ds)):
                    if (abs(abs(s001_ds[i]) - s001_far) < ds_tol):
                        s001_mask[atid] = s001_mask[atid] + 1
                for i in range(len(s111_ds)):
                    if (abs(abs(s111_ds[i]) - s111_far) < ds_tol):
                        s111_mask[atid] = s111_mask[atid] + 1
                for i in range(len(s011_ds)):
                    if (abs(abs(s011_ds[i]) - s011_far) < ds_tol):
                        s011_mask[atid] = s011_mask[atid] + 1
                if (s001_mask[atid] == 0 and s111_mask[atid] == 0 and s011_mask[atid] == 0):
                    self.core_atom_site[reid].append(0)
                elif (s001_mask[atid] == 1 and s111_mask[atid] == 0 and s011_mask[atid] == 0):
                    self.core_atom_site[reid].append(1)
                elif (s001_mask[atid] == 0 and s111_mask[atid] == 1 and s011_mask[atid] == 0):
                    self.core_atom_site[reid].append(2)
                elif (s001_mask[atid] == 0 and s111_mask[atid] == 0 and s011_mask[atid] == 1):
                    self.core_atom_site[reid].append(3)
                elif (s001_mask[atid] == 2 and s111_mask[atid] == 0 and s011_mask[atid] == 0):
                    self.core_atom_site[reid].append(4)
                elif (s001_mask[atid] == 0 and s111_mask[atid] == 2 and s011_mask[atid] == 0):
                    self.core_atom_site[reid].append(5)
                elif (s001_mask[atid] == 0 and s111_mask[atid] == 0 and s011_mask[atid] == 2):
                    self.core_atom_site[reid].append(6)
                elif (s001_mask[atid] == 1 and s111_mask[atid] == 1 and s011_mask[atid] == 0):
                    self.core_atom_site[reid].append(7)
                elif (s001_mask[atid] == 1 and s111_mask[atid] == 0 and s011_mask[atid] == 1):
                    self.core_atom_site[reid].append(8)
                elif (s001_mask[atid] == 0 and s111_mask[atid] == 1 and s011_mask[atid] == 1):
                    self.core_atom_site[reid].append(9)
                else:
                    self.core_atom_site[reid].append(10)
                    
            core_site_natom = [self.core_atom_site[reid].count(i) for i in range(ntype_site)]
            sum_surf = sum(core_site_natom[1:])
            core_surf_site_prop = [float(core_site_natom[i]) / float(sum_surf) for i in range(ntype_site)]
            core_surf_site_prop[0] = 0.0
            self.core_surf_site_prop.append(core_surf_site_prop)
            for atid in range(elem_natom[0]+elem_natom[1], natom):
                istat = 0
                linker_list = []
                for i in range(self.num_nn_new[reid][atid]):
                    if (self.nn_list_new[reid][atid][i] < elem_natom[0]):
                        linker_list.append(self.nn_list_new[reid][atid][i])
                nlinker = len(linker_list)
                linker_site_list = [self.core_atom_site[reid][linker_list[i]] for i in range(len(linker_list))]
                self.lig_codnum[reid].append(nlinker)
                if (nlinker == 1):
                    for pid in [1, 4, 7, 8, 10]:
                        if (linker_site_list[0] == pid):
                            self.lig_site[reid].append(pid)
                            istat = 1
                elif (nlinker == 2):
                    if (linker_site_list.count(10) >= 1):
                        self.lig_site[reid].append(10)
                        istat = 1
                    else:
                        for pid in range(4,10):
                            if (linker_site_list == [pid, pid]):
                                self.lig_site[reid].append(pid)
                                istat = 1
                        if (istat == 0):
                            if (linker_site_list[0] <= 3):
                                self.lig_site[reid].append(linker_site_list[0])
                            else:
                                self.lig_site[reid].append(linker_site_list[1])
                elif (nlinker == 3):
                    pid_edge = [5, 7, 9]
                    pid_edge_corner = pid_edge + [10]
                    if (linker_site_list.count(10) >= 1):
                        if (pid_edge_corner.count(linker_site_list[0]) == 1 and pid_edge_corner.count(linker_site_list[1]) == 1 \
                            and pid_edge_corner.count(linker_site_list[2]) == 1):
                            self.lig_site[reid].append(10)
                            istat = 1
                    if (istat == 0 and sum([pid_edge_corner.count(linker_site_list[j]) for j in range(nlinker)]) >= 2):
                        coord_linker = [coords[linker_list[k]] for k in range(nlinker)]
                        coord_extend = [np.subtract(np.add(coord_linker[(k+1)%3], coord_linker[(k+2)%3]), coord_linker[k]) for k in range(3)]
                        nextend = 0
                        for j in range(3):
                            for k in range(elem_natom[0]):
                                if (np.linalg.norm(np.subtract(coords[k], coord_extend[j])) < ds_tol):
                                    nextend = nextend + 1
                                    break
                        if (nextend < 3):
                            for k in range(3):
                                if (linker_site_list[k] != 10):
                                    self.lig_site[reid].append(linker_site_list[k])
                                    istat = 1
                                    break
                    if (istat == 0):
                        self.lig_site[reid].append(2)
                        istat = 1
                if (istat == 0):
                    print("error: fail to identify ligand site for ligand ", atid+1-elem_natom[0]-elem_natom[1])
                    print(nlinker)
                    print(linker_site_list)
            lig_site_natom = [self.lig_site[reid].count(i) for i in range(ntype_site)]
            sum_site = sum(lig_site_natom)
            lig_site_prop = [float(lig_site_natom[i]) / float(sum_site) for i in range(ntype_site)]
            self.lig_site_prop.append(lig_site_prop)
            lig_codnum_natom = [self.lig_codnum[reid].count(i) for i in range(4)]
            sum_codnum = sum(lig_codnum_natom)
            lig_codnum_prop = [float(lig_codnum_natom[i]) / float(sum_codnum) for i in range(4)]
            self.lig_codnum_prop.append(lig_codnum_prop)
                             
        # Check existence of important configurations:
        # 'conf_s001_pbi': Pb-I bonds in the middle of a {001} surface.
        # 'conf_pb_aggr': Pb aggregates with very short Pb-Pb bond length.
        # 'brok_pbs': broken Pb-S bonds during relaxation.
        print('\n')
        print('checking important configurations...')
        if ('nconf_s001_pbi' in self.feat_data_list):
            self.nconf_s001_pbi = [0 for i in range(self.nrecord)]
            self.conf_s001_pbi_atom_id = [[] for i in range(self.nrecord)]
        if ('nconf_pb_aggr' in self.feat_data_list):
            conf_pb_aggr_bl_crit = self.feat_param_list['conf_pb_aggr_bl_crit']
            self.nconf_pb_aggr = [0 for i in range(self.nrecord)]
            self.conf_pb_aggr_atom_id = [[] for i in range(self.nrecord)]
        if ('nconf_i_aggr' in self.feat_data_list):
            conf_i_aggr_bl_crit = self.feat_param_list['conf_i_aggr_bl_crit']
            self.nconf_i_aggr = [0 for i in range(self.nrecord)]
            self.conf_i_aggr_atom_id = [[] for i in range(self.nrecord)]
        if ('nconf_brok_pbs' in self.feat_data_list):
            self.nconf_brok_pbs = [0 for i in range(self.nrecord)]
            self.conf_brok_pbs_atom_id = [[] for i in range(self.nrecord)]
            self.nconf_s001_pbi = [0 for i in range(self.nrecord)]
            self.conf_s001_pbi_atom_id = [[] for i in range(self.nrecord)]
        
        for reid in range(self.nrecord):
            print('{0}, '.format(reid), end='')
            lc_const = self.old_lc_const[reid]
            lc_vec = self.old_lc_vec[reid]
            frac_coords = self.old_frac_coord[reid]
            natom = len(frac_coords)
            coords_old = [[frac_coords[i][j] * lc_const * lc_vec[j][j] for j in range(3)] for i in range(natom)]
            lc_const = self.new_lc_const[reid]
            lc_vec = self.new_lc_vec[reid]
            frac_coords = self.new_frac_coord[reid]
            coords_new = [[frac_coords[i][j] * lc_const * lc_vec[j][j] for j in range(3)] for i in range(natom)]
            elem_natom = self.elem_natom[reid]
            if ('s001_pbi' in self.feat_data_list):
                for atid in range(elem_natom[0]+elem_natom[1], natom):
                    lig_id = atid - elem_natom[0] - elem_natom[1] 
                    if (self.lig_site[reid][lig_id] == 1):
                        self.nconf_s001_pbi[reid] = self.nconf_s001_pbi[reid] + 1
                        self.conf_s001_pbi_atom_id[reid].append(atid)
                        atid_list2 = self.nn_list_new[reid][atid]
                        self.conf_s001_pbi_atom_id[reid] = self.conf_s001_pbi_atom_id[reid] + atid_list2
                        for j in range(len(atid_list2)):
                            atid_list3 = self.nn_list_new[reid][atid_list2[j]]
                            self.conf_s001_pbi_atom_id[reid] = self.conf_s001_pbi_atom_id[reid] + atid_list3
                self.conf_s001_pbi_atom_id[reid] = list(np.unique(self.conf_s001_pbi_atom_id[reid]))
                
            if ('pb_aggr' in self.feat_data_list):
                atom_mask = [0 for i in range(natom)]
                for atid1 in range(0, elem_natom[0]-1):
                    if (self.core_atom_site[reid][atid1] != 0 and atom_mask[atid1] == 0):
                        for atid2 in range(0, elem_natom[0]-1):
                            if (self.core_atom_site[reid][atid2] != 0 and atid2 != atid1):
                                dist12 = np.linalg.norm(np.subtract(coords_new[atid2], coords_new[atid1]))
                                for atid3 in range(atid2+1, elem_natom[0]):
                                    if (self.core_atom_site[reid][atid3] != 0 and atid3 != atid1):
                                        dist13 = np.linalg.norm(np.subtract(coords_new[atid3], coords_new[atid1]))
                                        if (dist12 < conf_pb_aggr_bl_crit and dist13 < conf_pb_aggr_bl_crit):
                                            self.nconf_pb_aggr[reid] = self.nconf_pb_aggr[reid] + 1
                                            atom_mask[atid1] = 1
                                            atom_mask[atid2] = 1
                                            atom_mask[atid3] = 1
                                            self.conf_pb_aggr_atom_id[reid].append(atid1)
                                            self.conf_pb_aggr_atom_id[reid].append(atid2)
                                            self.conf_pb_aggr_atom_id[reid].append(atid3)
                                            atid_list2 = [atid1, atid2, atid3]
                                            for j in range(len(atid_list2)):
                                                atid_list3 = self.nn_list_new[reid][atid_list2[j]]
                                                self.conf_pb_aggr_atom_id[reid] = self.conf_pb_aggr_atom_id[reid] + atid_list3
                for atid1 in range(0, elem_natom[0]-1):
                    if (self.core_atom_site[reid][atid1] != 0):
                        for atid2 in range(atid1+1, elem_natom[0]):
                            if (self.core_atom_site[reid][atid2] != 0):
                                if (atom_mask[atid1] == 0 or atom_mask[atid2] == 0):
                                    dist12 = np.linalg.norm(np.subtract(coords_new[atid2], coords_new[atid1]))
                                    if (dist12 < conf_pb_aggr_bl_crit):
                                        self.nconf_pb_aggr[reid] = self.nconf_pb_aggr[reid] + 1
                                        atom_mask[atid1] = 1
                                        atom_mask[atid2] = 1
                                        self.conf_pb_aggr_atom_id[reid].append(atid1)
                                        self.conf_pb_aggr_atom_id[reid].append(atid2)
                                        atid_list2 = [atid1, atid2]
                                        for j in range(len(atid_list2)):
                                            atid_list3 = self.nn_list_new[reid][atid_list2[j]]
                                            self.conf_pb_aggr_atom_id[reid] = self.conf_pb_aggr_atom_id[reid] + atid_list3
                self.conf_pb_aggr_atom_id[reid] = list(np.unique(self.conf_pb_aggr_atom_id[reid]))
                        
            if ('i_aggr' in self.feat_data_list):
                atom_mask = [0 for i in range(natom)]
                for atid1 in range(elem_natom[0]+elem_natom[1], natom):
                    if (atom_mask[atid1] == 0):
                        for atid2 in range(elem_natom[0]+elem_natom[1], natom-1):
                            if (atid2 != atid1):
                                dist12 = np.linalg.norm(np.subtract(coords_new[atid2], coords_new[atid1]))
                                for atid3 in range(atid2+1, natom):
                                    if (atid3 != atid1):
                                        dist13 = np.linalg.norm(np.subtract(coords_new[atid3], coords_new[atid1]))
                                        if (dist12 < conf_i_aggr_bl_crit and dist13 < conf_i_aggr_bl_crit):
                                            self.nconf_i_aggr[reid] = self.nconf_i_aggr[reid] + 1
                                            atom_mask[atid1] = 1
                                            atom_mask[atid2] = 1
                                            atom_mask[atid3] = 1
                                            self.conf_i_aggr_atom_id[reid].append(atid1)
                                            self.conf_i_aggr_atom_id[reid].append(atid2)
                                            self.conf_i_aggr_atom_id[reid].append(atid3)
                                            atid_list2 = [atid1, atid2, atid3]
                                            for j in range(len(atid_list2)):
                                                atid_list3 = self.nn_list_new[reid][atid_list2[j]]
                                                self.conf_i_aggr_atom_id[reid] = self.conf_i_aggr_atom_id[reid] + atid_list3
                for atid1 in range(elem_natom[0]+elem_natom[1], natom-1):
                    for atid2 in range(atid1+1, natom):
                        if (atom_mask[atid1] == 0 or atom_mask[atid2] == 0):
                            dist12 = np.linalg.norm(np.subtract(coords_new[atid2], coords_new[atid1]))
                            if (dist12 < conf_i_aggr_bl_crit):
                                self.nconf_i_aggr[reid] = self.nconf_i_aggr[reid] + 1
                                atom_mask[atid1] = 1
                                atom_mask[atid2] = 1
                                self.conf_i_aggr_atom_id[reid].append(atid1)
                                self.conf_i_aggr_atom_id[reid].append(atid2)
                                atid_list2 = [atid1, atid2]
                                for j in range(len(atid_list2)):
                                    atid_list3 = self.nn_list_new[reid][atid_list2[j]]
                                    self.conf_i_aggr_atom_id[reid] = self.conf_i_aggr_atom_id[reid] + atid_list3
                self.conf_i_aggr_atom_id[reid] = list(np.unique(self.conf_i_aggr_atom_id[reid]))

            if ('brok_pbs' in self.feat_data_list):
                for atid1 in range(0, elem_natom[0]):
                    if (self.core_atom_site[reid][atid1] != 0):
                        nns_old = 0
                        nns_id_old = []
                        for j in range(self.num_nn_old[reid][atid1]):
                            if (self.nn_list_old[reid][atid1][j] >= elem_natom[0] \
                                and self.nn_list_old[reid][atid1][j] < elem_natom[0]+elem_natom[1]):
                                nns_old = nns_old+1
                                nns_id_old.append(self.nn_list_old[reid][atid1][j]) 
                        nns_id_new = []
                        for j in range(self.num_nn_new[reid][atid1]):
                            if (self.nn_list_new[reid][atid1][j] >= elem_natom[0] \
                                and self.nn_list_new[reid][atid1][j] < elem_natom[0]+elem_natom[1]):
                                nns_id_new.append(self.nn_list_new[reid][atid1][j])
                        if (len(nns_id_new) < len(nns_id_old)):
                            self.nconf_brok_pbs[reid] = self.nconf_brok_pbs[reid]+1
                            self.conf_brok_pbs_atom_id[reid].append(atid1)
                            atid_list2 = [atid1]
                            for j in range(len(nns_id_old)):
                                if (nns_id_new.count(nns_id_old[j]) == 0):
                                    self.conf_brok_pbs_atom_id[reid].append(nns_id_old[j])
                                    atid_list2.append(nns_id_old[j])
                                for k in range(len(atid_list2)):
                                    atid_list3 = self.nn_list_new[reid][atid_list2[k]]
                self.conf_brok_pbs_atom_id[reid] = list(np.unique(self.conf_brok_pbs_atom_id[reid]))
                
        print('\n')
        if ('s001_pbi' in self.feat_data_list):
            print('number of Pb-I bonds in the middle of a {001} surface')
            print(self.nconf_s001_pbi)
        if ('pb_aggr' in self.feat_data_list):           
            print('number of Pb aggregation')
            print(self.nconf_pb_aggr)
        if ('i_aggr' in self.feat_data_list):
            print('number of I aggregation')
            print(self.nconf_i_aggr)
        if ('brok_pbs' in self.feat_data_list):
            print('number of broken Pb-S bonds')
            print(self.nconf_brok_pbs)       

##-------------------------------------------------------------------------------------------------
    def analyze_eigval(self):
        self.band_id = []
        self.eigval_orig = []
        self.eigval = []
        self.occ = []
        self.vac = []
        self.nband = []
        self.homo_id = []
        self.homo_bid = []
        self.homo = []
        self.lumo = []
        self.band_gap = []
        self.vb_edge = []
        self.cb_edge = []
        self.edge_gap = []
        if ('band_id' in self.input_var_labels):
            self.band_id = np.array(self.input_vars[self.input_var_labels.index('band_id')])
        if ('eigval_orig' in self.input_var_labels):
            self.eigval_orig = np.array(self.input_vars[self.input_var_labels.index('eigval_orig')])
        if ('occ' in self.input_var_labels):
            self.occ = np.array(self.input_vars[self.input_var_labels.index('occ')])
        if ('vac' in self.input_var_labels):
            self.vac = np.array(self.input_vars[self.input_var_labels.index('vac')])
        if ('ef' in self.input_var_labels):
            self.ef = np.array(self.input_vars[self.input_var_labels.index('ef')])
        if ('energy' in self.input_var_labels):
            self.energy = np.array(self.input_vars[self.input_var_labels.index('energy')])
        # Find HOMO, LUMO and band gap.
        for reid in range(self.nrecord):
            nband = len(self.band_id[reid])
            self.nband.append(nband)
            eigval = []
            for bid in range(nband):
                eigval.append(self.eigval_orig[reid][bid] - self.vac[reid])
            self.eigval.append(eigval)
            for bid in range(nband-1):
                if (self.occ[reid][bid] > 0.5 and self.occ[reid][bid+1] < 0.5):
                    homo_id = self.band_id[reid][bid]
                    self.homo_id.append(homo_id)
                    self.homo_bid.append(bid)
                    self.homo.append(self.eigval[reid][bid])
                    self.lumo.append(self.eigval[reid][bid+1])
                    self.band_gap.append(self.lumo[reid] - self.homo[reid])
                    break
                
        # Estimate the pristine gap for types of QD by referring to the corresponding stoichiometric QD type.  
        self.type_red_vb_edge = []
        self.type_red_cb_edge = []
        self.type_red_edge_gap = []
        self.type_red_ref_type_id = []
        self.type_red_edge_gap_ref = []
        self.band_gap = np.array(self.band_gap)
        type_red_band_gap_ave = []
        type_red_band_gap_min = []
        type_red_band_gap_max = []
        for tyid in range(self.ntype_red):
            type_red_band_gap_ave.append(np.average(self.band_gap[self.dot_type_id == tyid]))
            type_red_band_gap_max.append(np.amax(self.band_gap[self.dot_type_id == tyid]))
            type_red_band_gap_min.append(np.amin(self.band_gap[self.dot_type_id == tyid]))
        for tyid in range(self.ntype_red):
            istat = 0 
            for tyid2 in range(self.ntype_red):
                if ((self.type_red_stoi[tyid2] < self.tiny) and \
                    (self.type_red_core_shape[tyid] == self.type_red_core_shape[tyid2])\
                    and (abs(self.type_red_core_diameter[tyid]-self.type_red_core_diameter[tyid2]) < self.tiny) \
                    and list(self.type_red_core_elem_natom[tyid]) == list(self.type_red_core_elem_natom[tyid2])):
                    istat = 1
                    self.type_red_ref_type_id.append(tyid2)
                    self.type_red_edge_gap_ref.append([type_red_band_gap_ave[tyid2], type_red_band_gap_min[tyid2], \
                                                       type_red_band_gap_max[tyid2]])
                    break
            if (istat == 0):
                sysname = self.type_red_name[tyid] + str(tyid)
                print("Error: cannot find reference stoichiometric QD for QD type {0}".format(sysname))
                
        # Determine the band edges of each QD based on degeneracy and gap.
        trap_shaw_en_tol = self.feat_param_list['trap_shaw_en_tol']
        gap_min_buffer = 0.1
        gap_max_buffer = 1.0
        edge_ndeg_min = 3
        edge_deg_tol = 0.30
        ef_nband_buffer = 4
        self.vb_edge_id = []
        self.cb_edge_id = []
        self.vb_edge_bid = []
        self.cb_edge_bid = []
        print("analyzing eigenvalues... \n")
        for reid in range(self.nrecord):
            homo_bid = list(self.band_id[reid]).index(self.homo_id[reid])
            tyid = self.dot_type_id[reid]
            istat = 0
            if (abs(self.stoi[reid]) < self.tiny or self.stoi[reid] < 0.0):
                for bid in range(max(edge_ndeg_min+1, homo_bid-ef_nband_buffer), self.nband[reid]-1, 1):
                    if (self.eigval[reid][bid+1] - self.eigval[reid][bid] > trap_shaw_en_tol
                        and self.eigval[reid][bid] - self.eigval[reid][bid-1] < trap_shaw_en_tol \
                        and self.eigval[reid][bid] - self.eigval[reid][bid-edge_ndeg_min+1] < edge_deg_tol):
                        self.vb_edge_id.append(self.band_id[reid][bid])
                        self.vb_edge_bid.append(bid)
                        self.vb_edge.append(self.eigval[reid][bid])
                        istat = 1
                        break
                if (istat == 1):
                    istat = 0
                    for bid in range(self.vb_edge_bid[reid]+1, self.nband[reid]-edge_ndeg_min, 1):
                        if (self.eigval[reid][bid] - self.vb_edge[reid] > self.type_red_edge_gap_ref[tyid][1] - gap_min_buffer \
                            and self.eigval[reid][bid] - self.vb_edge[reid]  < self.type_red_edge_gap_ref[tyid][1] + gap_max_buffer \
                            and self.eigval[reid][bid] - self.eigval[reid][bid-1] > trap_shaw_en_tol \
                            and self.eigval[reid][bid+1] - self.eigval[reid][bid] < trap_shaw_en_tol \
                            and self.eigval[reid][bid+edge_ndeg_min-1] - self.eigval[reid][bid] < edge_deg_tol):
                            istat = 1
                            self.cb_edge_id.append(self.band_id[reid][bid])
                            self.cb_edge_bid.append(bid)
                            self.cb_edge.append(self.eigval[reid][bid])
                            self.edge_gap.append(self.cb_edge[reid] - self.vb_edge[reid])
                            break
            elif (self.stoi[reid] > 0.0):
                for bid in range(min(self.nband[reid]-edge_ndeg_min, homo_bid+ef_nband_buffer), 0, -1):
                    if (self.eigval[reid][bid] - self.eigval[reid][bid-1] > trap_shaw_en_tol \
                        and self.eigval[reid][bid+1] - self.eigval[reid][bid] < trap_shaw_en_tol \
                        and self.eigval[reid][bid+edge_ndeg_min-1] - self.eigval[reid][bid] < edge_deg_tol):
                        self.cb_edge_id.append(self.band_id[reid][bid])
                        self.cb_edge_bid.append(bid)
                        self.cb_edge.append(self.eigval[reid][bid])
                        istat = 1
                        break
                if (istat == 1):
                    istat = 0
                    for bid in range(self.cb_edge_bid[reid]-1, edge_ndeg_min, -1):
                        if (self.cb_edge[reid] - self.eigval[reid][bid] > self.type_red_edge_gap_ref[tyid][1] - gap_min_buffer \
                            and self.cb_edge[reid] - self.eigval[reid][bid] < self.type_red_edge_gap_ref[tyid][1] + gap_max_buffer \
                            and self.eigval[reid][bid+1] - self.eigval[reid][bid] > trap_shaw_en_tol \
                            and self.eigval[reid][bid] - self.eigval[reid][bid-1] < trap_shaw_en_tol \
                            and self.eigval[reid][bid] - self.eigval[reid][bid-edge_ndeg_min+1] < edge_deg_tol):
                            istat = 1
                            self.vb_edge_id.append(self.band_id[reid][bid])
                            self.vb_edge_bid.append(bid)
                            self.vb_edge.append(self.eigval[reid][bid])
                            self.edge_gap.append(self.cb_edge[reid] - self.vb_edge[reid])
                            break
            if (istat == 0):
                sysname = self.dot_type_name[reid] + '.cf-' + str(self.cf_id[reid])
                print("Error: fail to determine band edge for {0}".format(sysname))

##-------------------------------------------------------------------------------------------------
    def analyze_trap(self):       
        self.opt_homo = []
        self.opt_lumo = []
        self.opt_gap = []
        self.opt_homo_id = []
        self.opt_lumo_id = []
        self.ave_loc_vb = []
        self.ave_loc_cb = []
        self.ave_loc_mg = []
        self.ntrap_deep = []
        self.ntrap_shaw_vb = []
        self.ntrap_shaw_cb = []
        self.trap_deep_en_range = []
        self.trap_deep_en_wid = []
        self.trap_deep_en_dist_vb = []
        self.trap_deep_en_dist_cb = []
        self.trap_shaw_vb_en_range = []
        self.trap_shaw_cb_en_range = []
        self.trap_shaw_vb_en_wid = []
        self.trap_shaw_cb_en_wid = []
        self.trap_shaw_vb_en_dist = []
        self.trap_shaw_cb_en_dist = []
   
        #Initialization
        kb = 1.380648e-23
        ev = 1.602176e-19
        print('analyzing trap states... \n')
        if ('pdos_ion' in self.input_var_labels):
            self.pdos_ion = np.array(self.input_vars[self.input_var_labels.index('pdos_ion')])
        loc_crit_def1 = float(self.feat_param_list['loc_crit_def1'])
        band_rel_crit = float(self.feat_param_list['band_rel_crit'])
        temper = float(self.feat_param_list['temper'])
        self.rel_band_mask = [[0 for j in range(self.nband[i])] for i in range(self.nrecord)]
        self.vb_boltz_fac = [[0.0 for j in range(self.nband[i])] for i in range(self.nrecord)]
        self.cb_boltz_fac = [[0.0 for j in range(self.nband[i])] for i in range(self.nrecord)]
        self.mg_boltz_fac = [[0.0 for j in range(self.nband[i])] for i in range(self.nrecord)]
        for reid in range(self.nrecord):
            for bid in range(self.nband[reid]):
                if (self.eigval[reid][bid] > self.vb_edge[reid] - band_rel_crit \
                                and self.eigval[reid][bid] < self.cb_edge[reid] + band_rel_crit):
                    self.rel_band_mask[reid][bid] = 1
                    if (self.band_id[reid][bid] <= self.vb_edge_id[reid]):
                        self.vb_boltz_fac[reid][bid] = math.exp(self.eigval[reid][bid] / (kb * temper / ev))
                    elif (self.band_id[reid][bid] >= self.cb_edge_id[reid]):
                        self.cb_boltz_fac[reid][bid] = math.exp(-self.eigval[reid][bid] / (kb * temper / ev))
                    else:
                        self.mg_boltz_fac[reid][bid] = 1.0
        for reid in range(self.nrecord):
            boltz_vb_sum = sum(self.vb_boltz_fac[reid])
            boltz_cb_sum = sum(self.cb_boltz_fac[reid])
            boltz_mg_sum = sum(self.mg_boltz_fac[reid])
            if (boltz_cb_sum == 0):
                print(reid, self.vb_edge_id[reid], self.cb_edge_id[reid])
            for bid in range(self.nband[reid]):
                self.vb_boltz_fac[reid][bid] = self.vb_boltz_fac[reid][bid] / boltz_vb_sum
                self.cb_boltz_fac[reid][bid] = self.cb_boltz_fac[reid][bid] / boltz_cb_sum
                if (boltz_mg_sum > 1e-20):
                    self.mg_boltz_fac[reid][bid] = self.mg_boltz_fac[reid][bid] / boltz_mg_sum
        self.en_trap_mask = [[0 for j in range(self.nband[i])] for i in range(self.nrecord)]
        self.spa_trap_mask_def1 = [[0 for j in range(self.nband[i])] for i in range(self.nrecord)]
        self.wfn_loc_def1 = [[0.0 for j in range(self.nband[i])] for i in range(self.nrecord)]
        self.pdos_ion_norm = [[] for i in range(self.nrecord)]
        self.pdos_ion_split = [[] for i in range(self.nrecord)]
        for reid in range(self.nrecord):
            elem_natom = self.elem_natom[reid]
            nelem = len(elem_natom)
            elemid_split = np.cumsum(elem_natom)[0:-1]
            for bid in range(self.nband[reid]):
                pdos_list = self.pdos_ion[reid][bid]
                pdos_list_sum = sum(pdos_list)
                pdos_ion_norm = [pdos_list[i]/pdos_list_sum for i in range(len(pdos_list))]
                pdos_ion_split = np.array_split(pdos_ion_norm, elemid_split)
                self.pdos_ion_norm[reid].append(pdos_ion_norm)
                self.pdos_ion_split[reid].append(pdos_ion_split)
        
        # Find out reference values of localization calibrated by stoichiometric QDs.
        self.type_red_vb_loc_def1_ref = [0.0 for i in range(self.ntype_red)]
        self.type_red_cb_loc_def1_ref = [0.0 for i in range(self.ntype_red)]
        for tyid in range(self.ntype_red):
            if (abs(self.type_red_stoi[tyid]) < self.tiny):
                loc_vb_def1 = []
                loc_cb_def1 = []
                for reid in range(self.nrecord):
                    elem_natom = self.elem_natom[reid]
                    nelem = len(elem_natom)
                    if (self.dot_type_id[reid] == tyid):
                        for bid in range(self.nband[reid]):
                            if (self.rel_band_mask[reid][bid] == 1):
                                pdos_ion_split = self.pdos_ion_split[reid][bid]
                                pdos_elem_std_weig = [np.std(pdos_ion_split[i])*math.sqrt(elem_natom[i]) for i in range(nelem)]
                                loc_def1 = sum(pdos_elem_std_weig)
                                if (self.band_id[reid][bid] <= self.vb_edge_id[reid]):
                                    loc_vb_def1.append(loc_def1)
                                else:
                                    loc_cb_def1.append(loc_def1)
                self.type_red_vb_loc_def1_ref[tyid] = np.average(loc_vb_def1)
                self.type_red_cb_loc_def1_ref[tyid] = np.average(loc_cb_def1)
        for tyid in range(self.ntype_red):
            self.type_red_vb_loc_def1_ref[tyid] = self.type_red_vb_loc_def1_ref[self.type_red_ref_type_id[tyid]]
            self.type_red_cb_loc_def1_ref[tyid] = self.type_red_cb_loc_def1_ref[self.type_red_ref_type_id[tyid]]
        print('Reference localization (definition 1):')
        print(np.array(self.type_red_vb_loc_def1_ref))
        print(np.array(self.type_red_cb_loc_def1_ref))
        print("\n")

        # Identify spatial and energy traps.
        for reid in range(self.nrecord):
            elem_natom = self.elem_natom[reid]
            nelem = len(elem_natom)
            tyid = self.dot_type_id[reid]
            ave_loc_vb = 0.0
            ave_loc_cb = 0.0
            ave_loc_mg = 0.0
            for bid in range(self.nband[reid]):
                if (self.rel_band_mask[reid][bid] == 1):
                    pdos_ion_split = self.pdos_ion_split[reid][bid]
                    pdos_elem_std_weig = [np.std(pdos_ion_split[i])*math.sqrt(elem_natom[i]) for i in range(nelem)]
                    loc_def1 = sum(pdos_elem_std_weig)
                    self.wfn_loc_def1[reid][bid] = loc_def1
                    if (self.band_id[reid][bid] <= self.vb_edge_id[reid]):
                        loc_ref = self.type_red_vb_loc_def1_ref[tyid] 
                    elif (self.band_id[reid][bid] >= self.cb_edge_id[reid]):
                        loc_ref = self.type_red_cb_loc_def1_ref[tyid]
                    else:
                        loc_ref = min(self.type_red_vb_loc_def1_ref[tyid], self.type_red_cb_loc_def1_ref[tyid])
                    if (loc_def1 > loc_crit_def1*loc_ref):
                        self.spa_trap_mask_def1[reid][bid] = 1
                    if (bid <= self.vb_edge_bid[reid]):
                        ave_loc_vb = ave_loc_vb + loc_def1*self.vb_boltz_fac[reid][bid]
                    elif (bid >= self.cb_edge_bid[reid]):
                        ave_loc_cb = ave_loc_cb + loc_def1*self.cb_boltz_fac[reid][bid]
                    else:
                        ave_loc_mg = ave_loc_mg + loc_def1*self.mg_boltz_fac[reid][bid]
            self.ave_loc_vb.append(ave_loc_vb)
            self.ave_loc_cb.append(ave_loc_cb)
            self.ave_loc_mg.append(ave_loc_mg)
            for bid in range(self.homo_bid[reid], 0, -1):
                if (bid <= self.vb_edge_bid[reid] and self.spa_trap_mask_def1[reid][bid] == 0):
                    self.opt_homo.append(self.eigval[reid][bid])
                    self.opt_homo_id.append(self.band_id[reid][bid])
                    break
            for bid in range(self.homo_bid[reid]+1, self.nband[reid]):
                if (bid >= self.cb_edge_bid[reid] and self.spa_trap_mask_def1[reid][bid] == 0):
                    self.opt_lumo.append(self.eigval[reid][bid])
                    self.opt_lumo_id.append(self.band_id[reid][bid])
                    break
            self.opt_gap.append(self.opt_lumo[reid] - self.opt_homo[reid])
            self.ntrap_deep.append(self.cb_edge_id[reid] - self.vb_edge_id[reid] - 1)
            self.ntrap_shaw_vb.append(sum(self.spa_trap_mask_def1[reid][0:self.vb_edge_bid[reid]+1]))
            self.ntrap_shaw_cb.append(sum(self.spa_trap_mask_def1[reid][self.cb_edge_bid[reid]:-1]))
            opt_homo = self.opt_homo[reid]
            opt_lumo = self.opt_lumo[reid]
            trap_deep_low = 0.0
            trap_deep_high = 0.0
            if (self.ntrap_deep[reid] > 0):
                trap_deep_low = self.eigval[reid][self.vb_edge_bid[reid]+1]
                trap_deep_high = self.eigval[reid][self.cb_edge_bid[reid]-1]
                self.trap_deep_en_dist_vb.append(trap_deep_low - opt_homo)
                self.trap_deep_en_dist_cb.append(opt_lumo - trap_deep_high)
            else:
                self.trap_deep_en_dist_vb.append(0.0)
                self.trap_deep_en_dist_cb.append(0.0)
                
            self.trap_deep_en_range.append([trap_deep_low, trap_deep_high])
            self.trap_deep_en_wid.append(trap_deep_high - trap_deep_low)
            trap_shaw_vb_low = 0.0
            trap_shaw_vb_high = 0.0
            trap_shaw_vb_en_dist = 0.0
            trap_shaw_cb_low = 0.0
            trap_shaw_cb_high = 0.0
            trap_shaw_cb_en_dist = 0.0
            if (self.ntrap_shaw_vb[reid] > 0):
                istat = 0
                for bid in range(self.vb_edge_bid[reid], 0, -1):
                    if (self.spa_trap_mask_def1[reid][bid] == 1):
                        if (istat == 0):
                            trap_shaw_vb_high = self.eigval[reid][bid]
                            istat = 1
                            if (self.ntrap_shaw_vb[reid] == 1):
                                trap_shaw_vb_low = trap_shaw_vb_high
                        else:
                            trap_shaw_vb_low = self.eigval[reid][bid]
                trap_shaw_vb_en_dist = trap_shaw_vb_high - opt_homo
            if (self.ntrap_shaw_cb[reid] > 0):
                istat = 0
                for bid in range(self.cb_edge_bid[reid], self.nband[reid], 1):
                    if (self.spa_trap_mask_def1[reid][bid] == 1):
                        if (istat == 0):
                            trap_shaw_cb_low = self.eigval[reid][bid]
                            istat = 1
                            if (self.ntrap_shaw_cb[reid] == 1):
                                trap_shaw_cb_high = trap_shaw_cb_low
                        else:
                            trap_shaw_cb_high = self.eigval[reid][bid]
                trap_shaw_cb_en_dist = opt_lumo - trap_shaw_cb_low
            self.trap_shaw_vb_en_range.append([trap_shaw_vb_low, trap_shaw_vb_high])
            self.trap_shaw_vb_en_wid.append(trap_shaw_vb_high - trap_shaw_vb_low)
            self.trap_shaw_vb_en_dist.append(trap_shaw_vb_en_dist)
            self.trap_shaw_cb_en_range.append([trap_shaw_cb_low, trap_shaw_cb_high])
            self.trap_shaw_cb_en_wid.append(trap_shaw_cb_high - trap_shaw_cb_low)
            self.trap_shaw_cb_en_dist.append(trap_shaw_cb_en_dist)

        #Identify atoms responsible for the spatial traps.
        self.trap_deep_dist_vb = [[] for i in range(self.nrecord)]
        self.trap_deep_dist_cb = [[] for i in range(self.nrecord)]
        self.trap_shaw_vb_dist = [[] for i in range(self.nrecord)]
        self.trap_shaw_cb_dist = [[] for i in range(self.nrecord)]
        self.trap_deep_atom_id = [[] for i in range(self.nrecord)]
        self.trap_shaw_vb_atom_id = [[] for i in range(self.nrecord)]
        self.trap_shaw_cb_atom_id = [[] for i in range(self.nrecord)]
        self.trap_deep_atom_elem = [[] for i in range(self.nrecord)]
        self.trap_shaw_vb_atom_elem = [[] for i in range(self.nrecord)]
        self.trap_shaw_cb_atom_elem = [[] for i in range(self.nrecord)]
        self.trap_deep_atom_pdos_ratio = [[] for i in range(self.nrecord)]
        self.trap_shaw_vb_atom_pdos_ratio = [[] for i in range(self.nrecord)]
        self.trap_shaw_cb_atom_pdos_ratio = [[] for i in range(self.nrecord)]
        trap_atom_crit = float(self.feat_param_list['trap_atom_crit'])
        for reid in range(self.nrecord):
            elem_natom = self.elem_natom[reid]
            nelem = len(elem_natom)
            for bid in range(self.nband[reid]):
                if (self.spa_trap_mask_def1[reid][bid] == 1):
                    pdos_ion_split = self.pdos_ion_split[reid][bid]
                    atom_id = -1
                    atom_id_list = []
                    atom_elem_list = []
                    atom_pdos_ratio_list = []
                    for emid in range(nelem):
                        pdos_ave = np.average(pdos_ion_split[emid])
                        for aid in range(elem_natom[emid]):
                            atom_id = atom_id + 1
                            if (pdos_ion_split[emid][aid] > pdos_ave * trap_atom_crit):
                                atom_id_list.append(atom_id)
                                atom_elem_list.append(self.elem_name[reid][emid])
                                atom_pdos_ratio_list.append(pdos_ion_split[emid][aid] / pdos_ave)
                    if (bid <= self.vb_edge_bid[reid]):
                        self.trap_shaw_vb_dist[reid].append(self.eigval[reid][bid] - self.opt_homo[reid])
                        self.trap_shaw_vb_atom_id[reid].append(atom_id_list)
                        self.trap_shaw_vb_atom_elem[reid].append(atom_elem_list)
                        self.trap_shaw_vb_atom_pdos_ratio[reid].append(atom_pdos_ratio_list)
                    elif (bid >= self.cb_edge_bid[reid]):
                        self.trap_shaw_cb_dist[reid].append(self.eigval[reid][bid] - self.opt_lumo[reid])
                        self.trap_shaw_cb_atom_id[reid].append(atom_id_list)
                        self.trap_shaw_cb_atom_elem[reid].append(atom_elem_list)
                        self.trap_shaw_cb_atom_pdos_ratio[reid].append(atom_pdos_ratio_list)
                    else:
                        self.trap_deep_dist_vb[reid].append(self.eigval[reid][bid] - self.opt_homo[reid])
                        self.trap_deep_dist_cb[reid].append(self.eigval[reid][bid] - self.opt_lumo[reid])
                        self.trap_deep_atom_id[reid].append(atom_id_list)
                        self.trap_deep_atom_elem[reid].append(atom_elem_list)
                        self.trap_deep_atom_pdos_ratio[reid].append(atom_pdos_ratio_list)        

##-------------------------------------------------------------------------------------------------
    def write_data_file(self):
        filename1 = self.result_dir + '/wfn_loc_def1.dat'
        with open(filename1, 'w') as f1:
            f1.write("# system name \n")
            f1.write("# number of deep traps, shallow traps near vb, shallow traps near cb \n")
            f1.write("# deep trap energy range, energy width, energy away from optical HOMO, LUMO \n")
            f1.write("# shallow traps near vb energy range, energy width, energy away from optical HOMO \n")
            f1.write("# shallow traps near cb energy range, energy width, energy away from optical LUMO \n")
            f1.write("# average localization with definition 1 for vb, cb, and deep trap states \n")
            f1.write("# wavefunction localization with definition 1 for occupied states \n")
            f1.write("# wavefunction localization with definition 1 for unoccupied states \n")
            f1.write("# spacial trap mask based on wfn loc def1 for occupied states \n")
            f1.write("# spacial trap mask based on wfn loc def1 for unoccupied states \n")
            f1.write("\n")
            for reid in range(self.nrecord):
                sysname = self.dot_type_name[reid] + '.cf-' + str(self.cf_id[reid])
                f1.write(sysname + '\n')
                f1.write("{0}  {1}  {2} \n".format(self.ntrap_deep[reid], self.ntrap_shaw_vb[reid], self.ntrap_shaw_cb[reid]))
                f1.write("{0:.3f}  {1:.3f}  {2:.3f}  {3:.3f}  {4:.3f} \n".format(self.trap_deep_en_range[reid][0], \
                        self.trap_deep_en_range[reid][1], self.trap_deep_en_wid[reid], self.trap_deep_en_dist_vb[reid], \
                        self.trap_deep_en_dist_cb[reid]))
                f1.write("{0:.3f}  {1:.3f}  {2:.3f}  {3:.3f} \n".format(self.trap_shaw_vb_en_range[reid][0], \
                        self.trap_shaw_vb_en_range[reid][1], self.trap_shaw_vb_en_wid[reid], self.trap_shaw_vb_en_dist[reid]))
                f1.write("{0:.3f}  {1:.3f}  {2:.3f}  {3:.3f} \n".format(self.trap_shaw_cb_en_range[reid][0], \
                        self.trap_shaw_cb_en_range[reid][1], self.trap_shaw_cb_en_wid[reid], self.trap_shaw_cb_en_dist[reid]))
                f1.write("{0:.3f}  {1:.3f}  {2:.3f} \n".format(self.ave_loc_vb[reid], self.ave_loc_cb[reid], self.ave_loc_mg[reid]))
                temp_list = self.wfn_loc_def1[reid][0:self.homo_bid[reid]+1]
                f1.write(" ".join("{:.3f}".format(x) for x in temp_list) + '\n')
                temp_list = self.wfn_loc_def1[reid][self.homo_bid[reid]+1:-1]
                f1.write(" ".join("{:.3f}".format(x) for x in temp_list) + '\n')
                temp_list = self.spa_trap_mask_def1[reid][0:self.homo_bid[reid]+1]
                f1.write(" ".join(str(x) for x in temp_list) + '\n')
                temp_list = self.spa_trap_mask_def1[reid][self.homo_bid[reid]+1:-1]
                f1.write(" ".join(str(x) for x in temp_list) + '\n')
                f1.write("\n")
                
        filename2 = self.result_dir + '/homo_lumo_gap.dat'
        with open(filename2, 'w') as f2:
            f2.write("# system name \n")
            f2.write("# homo_id, lumo_id, homo, lumo, band gap \n")
            f2.write("# vb_edge_id, cb_edge_id, vb_edge, cb_edge, edge_gap \n")
            f2.write("# opt_homo_id, opt_lumo_id, opt_homo, opt_lumo, opt_band gap \n")
            f2.write("\n")
            for reid in range(self.nrecord):
                sysname = self.dot_type_name[reid] + '.cf-' + str(self.cf_id[reid])
                f2.write(sysname + '\n')
                f2.write("{0}  {1}  {2:.3f}  {3:.3f}  {4:.3f} \n"\
                         .format(self.homo_id[reid], self.homo_id[reid]+1, self.homo[reid], self.lumo[reid], self.band_gap[reid]))
                f2.write("{0}  {1}  {2:.3f}  {3:.3f}  {4:.3f} \n"\
                         .format(self.vb_edge_id[reid], self.cb_edge_id[reid], self.vb_edge[reid], self.cb_edge[reid], self.edge_gap[reid]))
                f2.write("{0}  {1}  {2:.3f}  {3:.3f}  {4:.3f} \n"\
                         .format(self.opt_homo_id[reid], self.opt_lumo_id[reid], self.opt_homo[reid], self.opt_lumo[reid], self.opt_gap[reid]))
                f2.write("\n")
                
        filename3 = self.result_dir + '/atom_project.dat'
        with open(filename3, 'w') as f3:
            f3.write("# system name \n")
            f3.write("# energy level of deep traps referred to delocalized (optical) HOMO \n")
            f3.write("# energy level of deep traps referred to delocalized (optical) LUMO \n")
            f3.write("# atom id / element associated with deep traps \n")
            f3.write("# energy level of shallow traps near vb referred to delocalized (optical) HOMO \n")
            f3.write("# atom id / element associated with shallow traps near vb \n")
            f3.write("# energy level of shallow traps near cb referred to delocalized (optical) LUMO \n")
            f3.write("# atom id / element associated with shallow traps near cb \n")
            f3.write("\n")
            for reid in range(self.nrecord):
                sysname = self.dot_type_name[reid] + '.cf-' + str(self.cf_id[reid])
                f3.write(sysname + '\n')
                temp_list = self.trap_deep_dist_vb[reid]
                if (temp_list == []):
                    f3.write(" nan \n")
                    f3.write(" nan \n")
                    f3.write(" nan \n")
                    f3.write(" nan \n")
                    f3.write(" nan \n")
                else:
                    f3.write(" ".join("{:.3f}".format(x) for x in temp_list) + '\n')
                    temp_list = self.trap_deep_dist_cb[reid]
                    f3.write(" ".join("{:.3f}".format(x) for x in temp_list) + '\n')
                    temp_list = self.trap_deep_atom_id[reid]
                    f3.write(" ".join("{0}".format(x) for x in temp_list) + '\n')
                    temp_list = self.trap_deep_atom_elem[reid]
                    f3.write(" ".join("{0}".format(x) for x in temp_list) + '\n')
                    temp_list = self.trap_deep_atom_pdos_ratio[reid]
                    f3.write(" ".join("{0}".format(x) for x in temp_list) + '\n')
                temp_list = self.trap_shaw_vb_dist[reid]
                if (temp_list == []):
                    f3.write(" nan \n")
                    f3.write(" nan \n")
                    f3.write(" nan \n")
                    f3.write(" nan \n")
                else:
                    f3.write(" ".join("{:.3f}".format(x) for x in temp_list) + '\n')
                    temp_list = self.trap_shaw_vb_atom_id[reid]
                    f3.write(" ".join("{0}".format(x) for x in temp_list) + '\n')
                    temp_list = self.trap_shaw_vb_atom_elem[reid]
                    f3.write(" ".join("{0}".format(x) for x in temp_list) + '\n')
                    temp_list = self.trap_shaw_vb_atom_pdos_ratio[reid]
                    f3.write(" ".join("{0}".format(x) for x in temp_list) + '\n')
                temp_list = self.trap_shaw_cb_dist[reid]
                if (temp_list == []):
                    f3.write(" nan \n")
                    f3.write(" nan \n")
                    f3.write(" nan \n")
                    f3.write(" nan \n")
                else:
                    f3.write(" ".join("{:.3f}".format(x) for x in temp_list) + '\n')
                    temp_list = self.trap_shaw_cb_atom_id[reid]
                    f3.write(" ".join("{0}".format(x) for x in temp_list) + '\n')
                    temp_list = self.trap_shaw_cb_atom_elem[reid]
                    f3.write(" ".join("{0}".format(x) for x in temp_list) + '\n')
                    temp_list = self.trap_shaw_cb_atom_pdos_ratio[reid]
                    f3.write(" ".join("{0}".format(x) for x in temp_list) + '\n')
                f3.write("\n")
                
        filename4 = self.result_dir + '/struct_feature.dat'
        with open(filename4, 'w') as f4:
            f4.write("# structural features \n")
            f4.write("# number of sample containing feature \n")
            f4.write("# sample name, number of features, number of atoms associated with the feature \n")
            f4.write("# relevant atom id \n")
            if ('s001_pbi' in self.feat_data_list):
                f4.write("\n")
                f4.write("Pb-I bonds in the middle of a {001} surface \n")
                f4.write(" {0} \n".format(np.count_nonzero(self.nconf_s001_pbi)))
                for reid in range(self.nrecord):
                    if (self.nconf_s001_pbi[reid] > 0):
                        sysname = self.dot_type_name[reid] + '.cf-' + str(self.cf_id[reid])
                        f4.write("{0}  {1}  {2} \n".format(sysname, self.nconf_s001_pbi[reid], len(self.conf_s001_pbi_atom_id[reid])))
                        temp_list = self.conf_s001_pbi_atom_id[reid]
                        f4.write(" ".join("{0}".format(x) for x in temp_list) + '\n')
            if ('pb_aggr' in self.feat_data_list):
                f4.write("\n")
                f4.write("Pb aggregation \n")
                f4.write(" {0} \n".format(np.count_nonzero(self.nconf_pb_aggr)))
                for reid in range(self.nrecord):
                    if (self.nconf_pb_aggr[reid] > 0):
                        sysname = self.dot_type_name[reid] + '.cf-' + str(self.cf_id[reid])
                        f4.write("{0}  {1}  {2} \n".format(sysname, self.nconf_pb_aggr[reid], len(self.conf_pb_aggr_atom_id[reid])))
                        temp_list = self.conf_pb_aggr_atom_id[reid]
                        f4.write(" ".join("{0}".format(x) for x in temp_list) + '\n')
            if ('i_aggr' in self.feat_data_list):
                f4.write("\n")
                f4.write("I aggregation \n")
                f4.write(" {0} \n".format(np.count_nonzero(self.nconf_i_aggr)))
                for reid in range(self.nrecord):
                    if (self.nconf_i_aggr[reid] > 0):
                        sysname = self.dot_type_name[reid] + '.cf-' + str(self.cf_id[reid])
                        f4.write("{0}  {1}  {2} \n".format(sysname, self.nconf_i_aggr[reid], len(self.conf_i_aggr_atom_id[reid])))
                        temp_list = self.conf_i_aggr_atom_id[reid]
                        f4.write(" ".join("{0}".format(x) for x in temp_list) + '\n')
            if ('brok_pbs' in self.feat_data_list):
                f4.write("\n")
                f4.write("Broken Pb-S bond \n")
                f4.write(" {0} \n".format(np.count_nonzero(self.nconf_brok_pbs)))
                for reid in range(self.nrecord):
                    if (self.nconf_brok_pbs[reid] > 0):
                        sysname = self.dot_type_name[reid] + '.cf-' + str(self.cf_id[reid])
                        f4.write("{0}  {1}  {2} \n".format(sysname, self.nconf_brok_pbs[reid], len(self.conf_brok_pbs_atom_id[reid])))
                        temp_list = self.conf_brok_pbs_atom_id[reid]
                        f4.write(" ".join("{0}".format(x) for x in temp_list) + '\n')


##-------------------------------------------------------------------------------------------------
    def build_feat_data(self):

        print("building feature for data...\n")
        print(self.feat_data_list)
        self.data_feats = []
        self.data_feat_labels = []

        # Data features of each QD.
        if ('stoi' in self.feat_data_list):
            self.data_feats.append(self.stoi)
            self.data_feat_labels.append('stoi')
        if ('core_shape' in self.feat_data_list):
            self.data_feats.append(self.core_shape)
            self.data_feat_labels.append('core_shape')
        if ('core_diameter' in self.feat_data_list):
            self.data_feats.append(self.core_diameter)
            self.data_feat_labels.append('core_diameter')
        if ('core_surf_site_prop' in self.feat_data_list):
            self.data_feats.append(self.core_surf_site_prop)
            self.data_feat_labels.append('core_surf_site_prop')
        if ('lig_site_prop' in self.feat_data_list):
            self.data_feats.append(self.lig_site_prop)
        if ('lig_codnum_prop' in self.feat_data_list):
            self.data_feats.append(self.lig_codnum_prop)
        if ('nconf_s001_pbi' in self.feat_data_list):
            self.data_feats.append(self.nconf_s001_pbi)
            self.data_feat_labels.append('nconf_s001_pbi')
        if ('nconf_pb_aggr' in self.feat_data_list):
            self.data_feats.append(self.nconf_pb_aggr)
            self.data_feat_labels.append('nconf_pb_aggr')
        if ('nconf_i_aggr' in self.feat_data_list):
            self.data_feats.append(self.nconf_i_aggr)
            self.data_feat_labels.append('nconf_i_aggr')
        if ('nconf_brok_pbs' in self.feat_data_list):
            self.data_feats.append(self.nconf_brok_pbs)
            self.data_feat_labels.append('nconf_brok_pbs')

        # Data features of each atom.
        if ('at_core_lig' in self.feat_data_list):
            self.at_core_lig = []
            for reid in range(self.nrecord):
                for atid in range(self.elem_natom[reid][0]+self.elem_natom[reid][1]):
                    self.at_core_lig.append(0)
                for atid in range(self.elem_natom[reid][0]+self.elem_natom[reid][1], self.natom[reid]):
                    self.at_core_lig.append(1)
            self.data_feats.append(self.at_core_lig)
            self.data_feat_labels.append('at_core_lig')
        if ('at_core_atom_site' in self.feat_data_list):
            self.at_core_atom_site = []
            for reid in range(self.nrecord):
                for atid in range(self.elem_natom[reid][0]+self.elem_natom[reid][1]):
                    self.at_core_atom_site.append(self.core_atom_site[reid][atid])
                for atid in range(self.elem_natom[reid][0]+self.elem_natom[reid][1], self.natom[reid]):
                    self.at_core_atom_site.append(-1)
            self.data_feats.append(self.at_core_atom_site)
            self.data_feat_labels.append('at_core_atom_site')
        if ('at_lig_atom_site' in self.feat_data_list):
            self.at_lig_atom_site = []
            for reid in range(self.nrecord):
                for atid in range(self.elem_natom[reid][0]+self.elem_natom[reid][1]):
                    self.at_lig_atom_site.append(-1)
                for atid in range(self.elem_natom[reid][0]+self.elem_natom[reid][1], self.natom[reid]):
                    self.at_lig_atom_site.append(self.lig_site[reid][atid-self.elem_natom[reid][0]-self.elem_natom[reid][1]])
            self.data_feats.append(self.at_lig_atom_site)
            self.data_feat_labels.append('at_lig_atom_site')
        if ('at_mask_conf_s001_pbi' in self.feat_data_list):
            self.at_mask_conf_s001_pbi = [[0 for j in range(self.natom[i])] for i in range(self.nrecord)]
            for reid in range(self.nrecord):
                for i in range(self.nconf_s001_pbi[reid]):
                    atid = self.conf_s001_pbi_atom_id[reid][i]
                    self.at_mask_conf_s001_pbi[reid][atid] = 1
            flat_array = np.array(self.at_mask_conf_s001_pbi).flatten()
            self.data_feats.append(flat_array)
            self.data_feat_labels.append('at_mask_conf_s001_pbi')
        if ('at_mask_conf_pb_aggr' in self.feat_data_list):
            self.at_mask_conf_pb_aggr = [[0 for j in range(self.natom[i])] for i in range(self.nrecord)]
            for reid in range(self.nrecord):
                for i in range(self.nconf_pb_aggr[reid]):
                    atid = self.conf_pb_aggr_atom_id[reid][i]
                    self.at_mask_conf_pb_aggr[reid][atid] = 1
            flat_array = np.array(self.at_mask_conf_pb_aggr).flatten()
            self.data_feats.append(flat_array)
            self.data_feat_labels.append('at_mask_conf_pb_aggr')
        if ('at_mask_conf_i_aggr' in self.feat_data_list):
            self.at_mask_conf_i_aggr = [[0 for j in range(self.natom[i])] for i in range(self.nrecord)]
            for reid in range(self.nrecord):
                for i in range(self.nconf_i_aggr[reid]):
                    atid = self.conf_i_aggr_atom_id[reid][i]
                    self.at_mask_conf_i_aggr[reid][atid] = 1
            flat_array = np.array(self.at_mask_conf_i_aggr).flatten()
            self.data_feats.append(flat_array)
            self.data_feat_labels.append('at_mask_conf_i_aggr')
        if ('at_mask_conf_brok_pbs' in self.feat_data_list):
            self.at_mask_conf_brok_pbs = [[0 for j in range(self.natom[i])] for i in range(self.nrecord)]
            for reid in range(self.nrecord):
                for i in range(self.nconf_brok_pbs[reid]):
                    atid = self.conf_brok_pbs_atom_id[reid][i]
                    self.at_mask_conf_brok_pbs[reid][atid] = 1
            flat_array = np.array(self.at_mask_conf_brok_pbs).flatten()
            self.data_feats.append(flat_array)
            self.data_feat_labels.append('at_mask_conf_brok_pbs')
        
##-------------------------------------------------------------------------------------------------
    def build_feat_target(self):
        
        print("building feature for target...\n")
        print(self.feat_target_list)
        self.target_feats = []
        self.target_feat_labels = []

        # Data features of each QD.
        if ('vb_edge' in self.feat_target_list):
            self.target_feats.append(self.vb_edge)
            self.target_feat_labels.append('vb_edge')
        if ('cb_edge' in self.feat_target_list):
            self.target_feats.append(self.cb_edge)
            self.target_feat_labels.append('cb_edge')
        if ('opt_homo' in self.feat_target_list):
            self.target_feats.append(self.opt_homo)
            self.target_feat_labels.append('opt_homo')
        if ('opt_lumo' in self.feat_target_list):
            self.target_feats.append(self.opt_lumo)
            self.target_feat_labels.append('opt_lumo')
        if ('opt_gap' in self.feat_target_list):
            self.target_feats.append(self.opt_gap)
            self.target_feat_labels.append('opt_gap')
        if ('ntrap_deep' in self.feat_target_list):
            self.target_feats.append(self.ntrap_deep)
            self.target_feat_labels.append('ntrap_deep')
        if ('ntrap_shaw_vb' in self.feat_target_list):
            self.target_feats.append(self.ntrap_shaw_vb)
            self.target_feat_labels.append('ntrap_shaw_vb')
        if ('ntrap_shaw_cb' in self.feat_target_list):
            self.target_feats.append(self.ntrap_shaw_cb)
            self.target_feat_labels.append('ntrap_shaw_cb')
        if ('ave_loc_vb' in self.feat_target_list):
            self.target_feats.append(self.ave_loc_vb)
            self.target_feat_labels.append('ave_loc_vb')
        if ('ave_loc_cb' in self.feat_target_list):
            self.target_feats.append(self.ave_loc_cb)
            self.target_feat_labels.append('ave_loc_cb')
        if ('ave_loc_mg' in self.feat_target_list):
            self.target_feats.append(self.ave_loc_mg)
            self.target_feat_labels.append('ave_loc_mg')
        if ('trap_deep_en_dist_vb' in self.feat_target_list):
            self.target_feats.append(self.trap_deep_en_dist_vb)
            self.target_feat_labels.append('trap_deep_en_dist_vb')
        if ('trap_deep_en_dist_cb' in self.feat_target_list):
            self.target_feats.append(self.trap_deep_en_dist_cb)
            self.target_feat_labels.append('trap_deep_en_dist_cb')
        if ('trap_shaw_vb_en_dist' in self.feat_target_list):
            self.target_feats.append(self.trap_shaw_vb_en_dist)
            self.target_feat_labels.append('trap_shaw_vb_en_dist')
        if ('trap_shaw_cb_en_dist' in self.feat_target_list):
            self.target_feats.append(self.trap_shaw_cb_en_dist)
            self.target_feat_labels.append('trap_shaw_cb_en_dist')
        if ('mask_trap_deep' in self.feat_target_list):
            self.mask_trap_deep = [0 for i in range(self.nrecord)]
            for reid in range(self.nrecord):
                if (self.ntrap_deep[reid] > 0):
                    self.mask_trap_deep[reid] = 1
            self.target_feats.append(self.mask_trap_deep)
            self.target_feat_labels.append('mask_trap_deep')
        if ('mask_trap_shaw_vb' in self.feat_target_list):
            self.mask_trap_shaw_vb = [0 for i in range(self.nrecord)]
            for reid in range(self.nrecord):
                if (self.ntrap_shaw_vb[reid] > 0):
                    self.mask_trap_shaw_vb[reid] = 1
            self.target_feats.append(self.mask_trap_shaw_vb)
            self.target_feat_labels.append('mask_trap_shaw_vb')
        if ('mask_trap_shaw_cb' in self.feat_target_list):
            self.mask_trap_shaw_cb = [0 for i in range(self.nrecord)]
            for reid in range(self.nrecord):
                if (self.ntrap_shaw_cb[reid] > 0):
                    self.mask_trap_shaw_cb[reid] = 1
            self.target_feats.append(self.mask_trap_shaw_cb)
            self.target_feat_labels.append('mask_trap_shaw_cb')

        # Data features of each atom.
        if ('at_mask_trap_deep' in self.feat_target_list):
            self.at_mask_trap_deep = [[0 for j in range(self.natom[i])] for i in range(self.nrecord)]
            for reid in range(self.nrecord):
                for i in range(len(self.trap_deep_atom_id[reid])):
                    for j in range(len(self.trap_deep_atom_id[reid])):
                        atid = self.trap_deep_atom_id[reid][i][j]
                        self.at_mask_trap_deep[reid][atid] = 1
            flat_array = np.array(self.at_mask_trap_deep).flatten()
            self.target_feats.append(flat_array)
            self.target_feat_labels.append('at_mask_trap_deep')
        if ('at_mask_trap_shaw_vb' in self.feat_target_list):
            self.at_mask_trap_shaw_vb = [[0 for j in range(self.natom[i])] for i in range(self.nrecord)]
            for reid in range(self.nrecord):
                for i in range(len(self.trap_shaw_vb_atom_id[reid])):
                    for j in range(len(self.trap_shaw_vb_atom_id[reid])):
                        atid = self.trap_shaw_vb_atom_id[reid][i][j]
                        self.at_mask_trap_shaw_vb[reid][atid] = 1
            flat_array = np.array(self.at_mask_trap_shaw_vb).flatten()
            self.target_feats.append(flat_array)
            self.target_feat_labels.append('at_mask_trap_shaw_vb')
        if ('at_mask_trap_shaw_cb' in self.feat_target_list):
            self.at_mask_trap_shaw_cb = [[0 for j in range(self.natom[i])] for i in range(self.nrecord)]
            for reid in range(self.nrecord):
                for i in range(len(self.trap_shaw_cb_atom_id[reid])):
                    for j in range(len(self.trap_shaw_cb_atom_id[reid])):
                        atid = self.trap_shaw_cb_atom_id[reid][i][j]
                        self.at_mask_trap_shaw_cb[reid][atid] = 1
            flat_array = np.array(self.at_mask_trap_shaw_cb).flatten()
            self.target_feats.append(flat_array)
            self.target_feat_labels.append('at_mask_trap_shaw_cb')
        if ('at_trap_deep_pdos_ratio' in self.feat_target_list):
            self.at_trap_deep_pdos_ratio = []
            for reid in range(self.nrecord):
                pdos_list = [[] for i in range(self.natom[reid])]
                for bid in range(len(self.trap_deep_atom_pdos_ratio[reid])):
                    for cid in range(len(self.trap_deep_atom_pdos_ratio[reid][bid])):
                        atid = self.trap_deep_atom_id[reid][bid][cid]
                        pdos_list[atid].append(self.trap_deep_atom_pdos_ratio[reid][bid][cid])
                pdos_ave_list = [0.0 for i in range(self.natom[reid])]
                for atid2 in range(self.natom[reid]):
                    if (len(pdos_list[atid2]) > 0):
                        pdos_ave_list[atid2] = np.average(pdos_list[atid2])
                self.at_trap_deep_pdos_ratio.append(pdos_ave_list)
            flat_array = np.array(self.at_trap_deep_pdos_ratio).flatten()
            self.target_feats.append(flat_array)
            self.target_feat_labels.append('at_trap_deep_pdos_ratio')
        if ('at_trap_shaw_vb_pdos_ratio' in self.feat_target_list):
            self.at_trap_shaw_vb_pdos_ratio = []
            for reid in range(self.nrecord):
                pdos_list = [[] for i in range(self.natom[reid])]
                for bid in range(len(self.trap_shaw_vb_atom_pdos_ratio[reid])):
                    for cid in range(len(self.trap_shaw_vb_atom_pdos_ratio[reid][bid])):
                        atid = self.trap_shaw_vb_atom_id[reid][bid][cid]
                        pdos_list[atid].append(self.trap_shaw_vb_atom_pdos_ratio[reid][bid][cid])
                pdos_ave_list = [0.0 for i in range(self.natom[reid])]
                for atid2 in range(self.natom[reid]):
                    if (len(pdos_list[atid2]) > 0):
                        pdos_ave_list[atid2] = np.average(pdos_list[atid2])
                self.at_trap_shaw_vb_pdos_ratio.append(pdos_ave_list)
            flat_array = np.array(self.at_trap_shaw_vb_pdos_ratio).flatten()
            self.target_feats.append(flat_array)
            self.target_feat_labels.append('at_trap_shaw_vb_pdos_ratio')
        if ('at_trap_shaw_cb_pdos_ratio' in self.feat_target_list):
            self.at_trap_shaw_cb_pdos_ratio = []
            for reid in range(self.nrecord):
                pdos_list = [[] for i in range(self.natom[reid])]
                for bid in range(len(self.trap_shaw_cb_atom_pdos_ratio[reid])):
                    for cid in range(len(self.trap_shaw_cb_atom_pdos_ratio[reid][bid])):
                        atid = self.trap_shaw_cb_atom_id[reid][bid][cid]
                        pdos_list[atid].append(self.trap_shaw_cb_atom_pdos_ratio[reid][bid][cid])
                pdos_ave_list = [0.0 for i in range(self.natom[reid])]
                for atid2 in range(self.natom[reid]):
                    if (len(pdos_list[atid2]) > 0):
                        pdos_ave_list[atid2] = np.average(pdos_list[atid2])
                self.at_trap_shaw_cb_pdos_ratio.append(pdos_ave_list)
            flat_array = np.array(self.at_trap_shaw_cb_pdos_ratio).flatten()
            self.target_feats.append(flat_array)
            self.target_feat_labels.append('at_trap_shaw_cb_pdos_ratio')
            
            
        
