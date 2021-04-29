"""
Load data of general information, structure, energy and projected density of states.
"""

import os

__all__ = ['LoadData']

class LoadData:

    def __init__(self, data_root, data_dirlist):
        self.data_root = data_root
        self.data_dirlist = data_dirlist

    def read_sys_info(self):
        """
        Read file *.system-info.dat generated during initial construction of QDs.
	### These infos seems to be generated from build_features. 
	In addition to constructing the POSCAR, also generated these informations	
        """
        print(" ")
        self.sys_info_sid = []
        self.core_shape = []
        self.core_diameter = []
        self.core_elem_natom = []
        self.nlig = []
        self.stoi = []
        self.pass_prop_001_old = []
        self.pass_prop_111_old = []
        self.pass_prop_011_old = []
        self.dot_type_id = []
        self.dot_type_name = []
        for rtid in range(len(self.data_dirlist)):
            sysname_full = self.data_dirlist[rtid]
            strlist = sysname_full.split('.')
            sysname = strlist[0]
            filein = self.data_root + '/' + sysname_full + '/' + sysname_full + '.system-info.dat'
            print("read file: {0} ".format(sysname_full + '.system-info.dat'))
            with open(filein, 'r') as f:
                line = f.readline().rstrip('\n')
                nsample = int(line.split()[0])
                f.readline()
                f.readline() 
                for i in range(nsample):
                    line = f.readline().rstrip('\n')
                    self.sys_info_sid.append([rtid, int(line.split()[1])])
                    f.readline()
                    f.readline()
                    f.readline()
                    self.core_shape.append(f.readline().rstrip('\n'))
                    f.readline()
                    self.core_diameter.append(float(f.readline().rstrip('\n')))
                    f.readline()
                    f.readline()
                    f.readline()
                    self.nlig.append(int(f.readline().rstrip('\n')))
                    f.readline()
                    self.stoi.append(float(f.readline().rstrip('\n')))
                    for j in range(10):
                        f.readline()
                    self.pass_prop_001_old.append(float(f.readline().rstrip('\n')))
                    f.readline()
                    self.pass_prop_111_old.append(float(f.readline().rstrip('\n')))
                    f.readline()
                    self.pass_prop_011_old.append(float(f.readline().rstrip('\n')))
                    for j in range(5):
                        f.readline()
                    line = f.readline().rstrip('\n')
                    str1 = line.split()
                    self.core_elem_natom.append([int(str1[0]), int(str1[1])])
                    for j in range(23):
                        f.readline()
                    self.dot_type_id.append(rtid)
                    self.dot_type_name.append(sysname_full)
        print("numble of samples = {0}".format(len(self.sys_info_sid)))
        for i in range(len(self.sys_info_sid)-1):
            if (self.sys_info_sid[i+1][0] == self.sys_info_sid[i][0] and \
                self.sys_info_sid[i+1][1] - self.sys_info_sid[i][1] !=1):
                for j in range(self.sys_info_sid[i][1]+1, self.sys_info_sid[i+1][1]):
                    print("warning: missing data from " + self.data_dirlist[self.sys_info_sid[i][0]] \
                          + '.cf-' + str(j))

    def read_old_struct(self):
        """
        Read file *.struct-origin-poscar.dat generated during initial construction of QDs.
        """
        self.old_struct_sid = []
        self.old_struct_lc_const = []
        self.old_struct_lc_vec = []
        self.old_struct_elem_name = []
        self.old_struct_elem_natom = []
        self.old_struct_frac_coord = []
        for rtid in range(len(self.data_dirlist)):
            sysname_full = self.data_dirlist[rtid]
            strlist = sysname_full.split('.')
            sysname = strlist[0]
            filein = self.data_root + '/' + sysname_full + '/' + sysname_full \
                     + '.struct-origin-poscar.dat'
            print("read file: {0} ".format(sysname_full + '.struct-origin-poscar.dat'))
            with open(filein, 'r') as f:
                line = f.readline().rstrip('\n')
                nsample = int(line.split()[0])
                for i in range(nsample):
                    f.readline()
                    f.readline()
                    line = f.readline().rstrip('\n')
                    self.old_struct_sid.append([rtid, int(line.split()[1])])
                    f.readline()
                    f.readline()
                    self.old_struct_lc_const.append(float(f.readline().rstrip('\n')))
                    lc_vec = []
                    for j in range(3):
                        line = f.readline().rstrip('\n')
                        str1 = line.split()
                        lc_vec_sub = [float(str1[k]) for k in range(len(str1))]
                        lc_vec.append(lc_vec_sub)
                    self.old_struct_lc_vec.append(lc_vec)
                    line = f.readline().rstrip('\n')
                    self.old_struct_elem_name.append(line.split())
                    line = f.readline().rstrip('\n')
                    str1 = line.split()
                    elem_natom = [int(str1[j]) for j in range(len(str1))]
                    self.old_struct_elem_natom.append(elem_natom)
                    f.readline()
                    natom = sum(elem_natom)
                    coords = []
                    for j in range(natom):
                        line = f.readline().rstrip('\n')
                        str1 = line.split()
                        coord = [float(str1[k]) for k in range(len(str1))]
                        coords.append(coord)
                    self.old_struct_frac_coord.append(coords)
        print("numble of samples = {0}".format(len(self.old_struct_sid)))
        for i in range(len(self.old_struct_sid)-1):
            if (self.old_struct_sid[i+1][0] == self.old_struct_sid[i][0] and \
                self.old_struct_sid[i+1][1] - self.old_struct_sid[i][1] !=1):
                for j in range(self.old_struct_sid[i][1]+1, self.old_struct_sid[i+1][1]):
                    print("warning: missing data from " + self.data_dirlist[self.old_struct_sid[i][0]] \
                          + '.cf-' + str(j))

    def read_new_struct(self):
        """
        Read file *.struct-relax-poscar.dat relaxed by DFT using VASP.
        """
        self.new_struct_sid = []
        self.new_struct_lc_const = []
        self.new_struct_lc_vec = []
        self.new_struct_elem_name = []
        self.new_struct_elem_natom = []
        self.new_struct_frac_coord = []
        self.natom = []
        for rtid in range(len(self.data_dirlist)):
            sysname_full = self.data_dirlist[rtid]
            strlist = sysname_full.split('.')
            sysname = strlist[0]
            filein = self.data_root + '/' + sysname_full + '/' + sysname_full \
                     + '.struct-relax-poscar.dat'
            print("read file: {0} ".format(sysname_full + '.struct-relax-poscar.dat'))
            with open(filein, 'r') as f:
                line = f.readline().rstrip('\n')
                nsample = int(line.split()[0])
                for i in range(nsample):
                    f.readline()
                    f.readline()
                    line = f.readline().rstrip('\n')
                    self.new_struct_sid.append([rtid, int(line.split()[1])])
                    f.readline()
                    f.readline()
                    self.new_struct_lc_const.append(float(f.readline().rstrip('\n')))
                    lc_vec = []
                    for j in range(3):
                        line = f.readline().rstrip('\n')
                        str1 = line.split()
                        lc_vec_sub = [float(str1[k]) for k in range(len(str1))]
                        lc_vec.append(lc_vec_sub)
                    self.new_struct_lc_vec.append(lc_vec)
                    line = f.readline().rstrip('\n')
                    self.new_struct_elem_name.append(line.split())
                    line = f.readline().rstrip('\n')
                    str1 = line.split()
                    elem_natom = [int(str1[j]) for j in range(len(str1))]
                    self.new_struct_elem_natom.append(elem_natom)
                    f.readline()
                    natom = sum(elem_natom)
                    coords = []
                    for j in range(natom):
                        line = f.readline().rstrip('\n')
                        str1 = line.split()
                        coord = [float(str1[k]) for k in range(len(str1))]
                        for k in range(3):
                            if (coord[k] > 0.50):
                                coord[k] = coord[k] - 1.0
                        coords.append(coord)
                    self.new_struct_frac_coord.append(coords)
            self.natom.append(natom)
        print("numble of samples = {0}".format(len(self.new_struct_sid)))
        for i in range(len(self.new_struct_sid)-1):
            if (self.new_struct_sid[i+1][0] == self.new_struct_sid[i][0] and \
                self.new_struct_sid[i+1][1] - self.new_struct_sid[i][1] !=1):
                for j in range(self.new_struct_sid[i][1]+1, self.new_struct_sid[i+1][1]):
                    print("warning: missing data from " + self.data_dirlist[self.new_struct_sid[i][0]] \
                          + '.cf-' + str(j))

    def read_en_eigval(self):
        """
        Read file *.en-eigval-vasp.dat calculated by DFT using VASP.
        """
        self.en_eigval_sid = []
        self.band_id = []
        self.energy = []
        self.eigval_orig = []
        self.occ = []
        self.vac = []
        self.nband = []
        self.ef = []
        for rtid in range(len(self.data_dirlist)):
            sysname_full = self.data_dirlist[rtid]
            strlist = sysname_full.split('.')
            sysname = strlist[0]
            filein = self.data_root + '/' + sysname_full + '/' + sysname_full \
                     + '.en-eigval-vasp.dat'
            print("read file: {0} ".format(sysname_full + '.en-eigval-vasp.dat'))
            with open(filein, 'r') as f:
                line = f.readline().rstrip('\n')
                nsample = int(line.split()[0])
                lid = 2
                for i in range(2000):
                    line = f.readline().rstrip('\n')
                    str1 = line.split()
                    lid = lid + 1
                    if (len(str1) > 0 and str1[0] == 'vacuum'):
                        break
                nband = lid - 9
                self.nband.append(nband)
            with open(filein, 'r') as f:
                f.readline()
                f.readline()
                for i in range(nsample):
                    f.readline()
                    line = f.readline().rstrip('\n')
                    self.en_eigval_sid.append([rtid, int(line.split()[1])])
                    f.readline()
                    f.readline()
                    band_id = []
                    eigval_orig = []
                    occ = []
                    for j in range(nband):
                        line = f.readline().rstrip('\n')
                        str1 = line.split()
                        band_id.append(int(str1[0]))
                        eigval_orig.append(float(str1[1]))
                        occ.append(float(str1[2]))
                    self.band_id.append(band_id)
                    self.eigval_orig.append(eigval_orig)
                    self.occ.append(occ)
                    f.readline()
                    f.readline()
                    line = f.readline().rstrip('\n')
                    self.vac.append(float(line.split()[1]))
                    f.readline()
                    f.readline()
                    line = f.readline().rstrip('\n')
                    self.energy.append(float(line.split()[3]))
                    f.readline()
                    f.readline()
                    line = f.readline().rstrip('\n')
                    self.ef.append(float(line.split()[2]))
                    f.readline()
        print("numble of samples = {0}".format(len(self.en_eigval_sid)))
        for i in range(len(self.en_eigval_sid)-1):
            if (self.en_eigval_sid[i+1][0] == self.en_eigval_sid[i][0] and \
                self.en_eigval_sid[i+1][1] - self.en_eigval_sid[i][1] !=1):
                for j in range(self.en_eigval_sid[i][1]+1, self.en_eigval_sid[i+1][1]):
                    print("warning: missing data from " + self.data_dirlist[self.en_eigval_sid[i][0]] \
                          + '.cf-' + str(j))

    def read_pdos_ion(self):
        """
        Read file *.procar-vasp.dat calculated by DFT using VASP.
        """
        self.pdos_ion_sid = []
        self.pdos_ion = []
        for rtid in range(len(self.data_dirlist)):
            sysname_full = self.data_dirlist[rtid]
            strlist = sysname_full.split('.')
            sysname = strlist[0]
            filein = self.data_root + '/' + sysname_full + '/' + sysname_full \
                     + '.procar-vasp.dat'
            print("read file: {0} ".format(sysname_full + '.procar-vasp.dat'))
            with open(filein, 'r') as f:
                line = f.readline().rstrip('\n')
                nsample = int(line.split()[0])
                f.readline()
                for i in range(nsample):
                    f.readline()
                    line = f.readline().rstrip('\n')
                    self.pdos_ion_sid.append([rtid, int(line.split()[1])])
                    f.readline()
                    pdos_ion = []
                    for j in range(self.nband[rtid]):
                        pdos_ion_sub = []
                        for k in range(3):
                            f.readline()
                        for k in range(self.natom[rtid]):
                           line = f.readline().rstrip('\n')
                           str1 = line.split()
                           pdos_ion_sub.append(float(str1[4]))
                        f.readline()
                        f.readline()
                        pdos_ion.append(pdos_ion_sub)
                    self.pdos_ion.append(pdos_ion)
                    f.readline()
        print("numble of samples = {0}".format(len(self.pdos_ion_sid)))
        for i in range(len(self.pdos_ion_sid)-1):
            if (self.pdos_ion_sid[i+1][0] == self.pdos_ion_sid[i][0] and \
                self.pdos_ion_sid[i+1][1] - self.pdos_ion_sid[i][1] !=1):
                for j in range(self.pdos_ion_sid[i][1]+1, self.pdos_ion_sid[i+1][1]):
                    print("warning: missing data from " + self.data_dirlist[self.pdos_ion_sid[i][0]] \
                          + '.cf-' + str(j))

    def combine_data(self):
        """
        Select data with all information available and pass input variables to feature generator.
        """
        print(" ")
        print("combine data...")
        self.record_sid = []
        for i in range(len(self.sys_info_sid)):
            sid = self.sys_info_sid[i]
            if (sid in self.sys_info_sid and sid in self.old_struct_sid and sid in self.new_struct_sid \
                and sid in self.en_eigval_sid and sid in self.pdos_ion_sid):
                self.record_sid.append(sid)
            else:
                print("warning: delete data sid {0}".format(sid))
        self.nrecord = len(self.record_sid)
        print("total number of records = ", self.nrecord)
        sys_info_labels = ['dot_type_id', 'dot_type_name', 'cf_id', 'core_shape', 'core_diameter', 'core_elem_natom', \
                           'nlig', 'stoi', 'pass_prop_001_old', 'pass_prop_111_old', 'pass_prop_011_old']
        old_struct_labels = ['elem_name', 'elem_natom', 'old_lc_const', 'old_lc_vec', 'old_frac_coord']
        new_struct_labels = ['new_lc_const', 'new_lc_vec', 'new_frac_coord']
        en_eigval_labels = ['band_id', 'energy', 'eigval_orig', 'occ', 'vac', 'ef']
        pdos_ion_labels = ['pdos_ion']
        self.input_var_labels = sys_info_labels + old_struct_labels + new_struct_labels \
                                + en_eigval_labels + pdos_ion_labels
        print('{0} input variables:'.format(len(self.input_var_labels)))
        for i in range(len(self.input_var_labels)):
            print('{0} : {1}'.format(i, self.input_var_labels[i]))
        self.input_vars = [[] for i in range(len(self.input_var_labels))]
        vid = 0
        for i in range(len(self.sys_info_sid)):
           if (self.sys_info_sid[i] in self.record_sid):
               self.input_vars[vid].append(self.dot_type_id[i])
               self.input_vars[vid+1].append(self.dot_type_name[i])
               self.input_vars[vid+2].append(self.sys_info_sid[i][1])
               self.input_vars[vid+3].append(self.core_shape[i])
               self.input_vars[vid+4].append(self.core_diameter[i])
               self.input_vars[vid+5].append(self.core_elem_natom[i])
               self.input_vars[vid+6].append(self.nlig[i])
               self.input_vars[vid+7].append(self.stoi[i])
               self.input_vars[vid+8].append(self.pass_prop_001_old[i])
               self.input_vars[vid+9].append(self.pass_prop_111_old[i])
               self.input_vars[vid+10].append(self.pass_prop_011_old[i])
        vid = vid + len(sys_info_labels)
        for i in range(len(self.old_struct_sid)):
           if (self.old_struct_sid[i] in self.record_sid):
               self.input_vars[vid].append(self.old_struct_elem_name[i])
               self.input_vars[vid+1].append(self.old_struct_elem_natom[i])
               self.input_vars[vid+2].append(self.old_struct_lc_const[i])
               self.input_vars[vid+3].append(self.old_struct_lc_vec[i])
               self.input_vars[vid+4].append(self.old_struct_frac_coord[i])
        vid = vid + len(old_struct_labels)
        for i in range(len(self.new_struct_sid)):
           if (self.new_struct_sid[i] in self.record_sid):
               self.input_vars[vid].append(self.new_struct_lc_const[i])
               self.input_vars[vid+1].append(self.new_struct_lc_vec[i])
               self.input_vars[vid+2].append(self.new_struct_frac_coord[i])
        vid = vid + len(new_struct_labels)
        for i in range(len(self.en_eigval_sid)):
           if (self.en_eigval_sid[i] in self.record_sid):
               self.input_vars[vid].append(self.band_id[i])
               self.input_vars[vid+1].append(self.energy[i])
               self.input_vars[vid+2].append(self.eigval_orig[i])
               self.input_vars[vid+3].append(self.occ[i])
               self.input_vars[vid+4].append(self.vac[i])
               self.input_vars[vid+5].append(self.ef[i])
        vid = vid + len(en_eigval_labels)
        for i in range(len(self.pdos_ion_sid)):
           if (self.pdos_ion_sid[i] in self.record_sid):
               self.input_vars[vid].append(self.pdos_ion[i])
        
        
        
               
            
        
                      



                    
                
                
            
                
                
                    
                
    
                
                
            
