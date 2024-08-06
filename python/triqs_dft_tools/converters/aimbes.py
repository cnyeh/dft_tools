##########################################################################
#
# TRIQS: a Toolbox for Research in Interacting Quantum Systems
#
# Copyright (C) 2011 by M. Aichhorn, L. Pourovskii, V. Vildosola
#
# TRIQS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# TRIQS. If not, see <http://www.gnu.org/licenses/>.
#
##########################################################################

"""
AIMBES to HDF5 converter for TRIQS/dft_tools
"""

import os
import shutil
import numpy as np
from scipy.constants import physical_constants
HARTREE_EV = physical_constants['Hartree energy in eV'][0]

from h5 import HDFArchive
from triqs.utility import mpi
import itertools

from triqs.gf import *

from .aimb_utils.iaft import IAFT


class AimbesConverter(object):
    """
    Converter from aimbes output to a hdf5 file that can used as input for the SumkDFT class. 
    
    Attributes:
         
    """
    
    def __init__(self, aimbes_h5, wan_h5, out_h5=None):
        """
        Parameters
        ----------
        :param aimbes_h5: string 
            Name of hdf5 archive of the aimbes output 
        :param wan_h5: string 
            Name of hdf5 archive of the Wannier90 converter from dft_tools
        :param out_h5: string
            Name of output hdf5 archive
        """

        self._name = "AIMBESConverter"
        
        self.aimbes_h5 = aimbes_h5
        self.wan_h5 = wan_h5
        self.seedname = wan_h5[:-3] 
        self.out_h5 = self.seedname+".aimbes.h5" if out_h5 is None else out_h5
        
    def __str__(self):
        mpi.report(f"AIMBES Converter for the input of SumkDFT\n"
                   f"-----------------------------------------\n"
                   f"  AIMBES h5 archive: {self.aimbes_h5}\n"
                   f"  Wannier90 h5: {self.wan_h5}\n"
                   f"  Output h5: {self.out_h5}\n")
        
    def convert_mbpt_input(self, aimb_it_1e=None, aimb_it_2e=None, unit='eV'):
        """
        
        :param aimb_it_1e: 
        :param aimb_it_2e: 
        :param unit: 
        :return: 
        """    
        aimb_data = read_aimbes_data(self.aimbes_h5, aimb_it_1e, aimb_it_2e, unit)
        mbpt_data = aimbes_to_dft_tools(aimb_data)
        
        mpi.report(f"Writing mbpt data from AIMBES into \'mbpt_input\' h5 group in {self.out_h5}.") 
        # check if W90 h5 exist 
        if not os.path.exists(self.wan_h5):
            raise FileNotFoundError(f"File {self.wan_h5} does not exist")
        # copy W90 h5 file
        shutil.copyfile(self.wan_h5, self.out_h5)

        # dump aimb data into ouput h5
        with HDFArchive(self.out_h5, 'a') as ar:
            if 'mbpt_input' not in ar:
                ar.create_group('mbpt_input')
            mbpt = ar['mbpt_input']

            for key, value in mbpt_data.items():
                mbpt[key] = value

        mpi.report(f'Finished writing results in {self.out_h5}/mbpt_input.\n')


def read_aimbes_data(aimbes_h5, aimb_it_1e=None, aimb_it_2e=None, unit='eV'):
    """
    Extract mbpt data from an aimbes h5 archive 
    :param aimbes_h5: string 
        Name of hdf5 archive of the aimbes results
    :param aimb_it_1e: int 
        iteration for downfold_1e group 
    :param aimb_it_2e: int 
        iteration for downfold_2e group
    :param unit: string 
        unit for mbpt data: "hartree" or "ev"    
        
    :return: dict 
        mbpt data in a dictionary 
    """
    # check input h5 exist 
    if not os.path.exists(aimbes_h5):
        raise FileNotFoundError(f"File {aimbes_h5} does not exist")
    
    unit = unit.lower()
    if unit.lower() not in ['hartree', 'ev']:
        raise ValueError(f"read_aimbes_data: Unit must be \"hartree\" or \"ev\".")    
    conv_fac = 1.0 if unit.lower() == "hartree" else HARTREE_EV
    mpi.report(f'aimbes h5 reader for downfolded Hamiltonian')
    mpi.report('-------------------------------------------')
    mpi.report(f'- unit = {unit}')
    mpi.report('- currently only support a single impurity\n')
    aimb_data = {'n_inequiv_shells': 1}
    with HDFArchive(aimbes_h5, 'r') as ar:
        # system metadata
        aimb_data['nspin'] = ar['system/number_of_spins']
        aimb_data['nkpts'] = ar['system/number_of_kpoints']
        aimb_data['nkpts_ibz'] = ar['system/number_of_kpoints_ibz']
        aimb_data['kpts'] = ar['system/kpoints_crys']
        aimb_data['kpt_weights'] = ar['system/k_weight'][:aimb_data['nkpts_ibz']]
        if nspin == 1: # follow QE notation in which k_weights has a factor of 2 in spin-unpolarized cases
            aimb_data['kpt_weights'] *= 2.0

        # Mesh info on the imaginary axis
        aimb_data['beta'] = ar['imaginary_fourier_transform']['beta'] / conv_fac
        aimb_data['iaft_type'] = ar['imaginary_fourier_transform']['source']
        aimb_data['iaft_lambda'] = ar['imaginary_fourier_transform']['lambda'] 
        aimb_data['iaft_wmax'] = aimb_data['iaft_lambda'] / aimb_data['beta'] * conv_fac
        prec = ar['imaginary_fourier_transform']['prec']
        if prec == 'high':
            aimb_data['iaft_prec'] = 1e-15
        elif prec == 'mid':
            aimb_data['iaft_prec'] = 1e-10
        elif prec == 'low':
            aimb_data['iaft_prec'] = 1e-6

        # Screened interactions from downfold_2e group 
        aimb_data['it_2e'] = ar['downfold_2e/final_iter'] if aimb_it_2e is None else aimb_it_2e
        b_grp = ar['downfold_2e/iter{}'.format(aimb_data['it_2e'])]
        mpi.report("Reading results from downfold_2e iter {}.".format(aimb_data['it_2e']))
        
        Vloc_list, Uloc_w_list = [], []
        for ish in range(aimb_data['n_inequiv_shells']):
            # interactions are stored in the chemist notation
            Vloc_chem = b_grp['Vloc_abcd']
            Uloc_w_chem = b_grp['Uloc_wabcd']
            # switch inner two indices to match triqs notation
            Vloc = np.zeros(Vloc_chem.shape, dtype=complex)
            Uloc_w = np.zeros(Uloc_w_chem.shape, dtype=complex)
            n_orb = Vloc.shape[0]
            for or1, or2, or3, or4 in itertools.product(range(n_orb), repeat=4):
                Vloc[or1, or2, or3, or4] = Vloc_chem[or1, or3, or2, or4]
                for w in range(Uloc_w_chem.shape[0]):
                    Uloc_w[w, or1, or2, or3, or4] = Uloc_w_chem[w, or1, or3, or2, or4]
            Vloc_list.append(Vloc * conv_fac)
            Uloc_w_list.append(Uloc_w * conv_fac)
        aimb_data['Vloc'] = Vloc_list
        aimb_data['Uloc_w'] = Uloc_w_list
        
        # Downfolded one-body components
        if 'downfold_1e' not in ar:
            mpi.report(f"Downfold_1e group not found in {aimbes_h5}. "
                       f"The resulting h5 will only contain screened interaction data.\n")
            return aimb_data

        aimb_data['it_1e'] = ar['downfold_1e/final_iter'] if aimb_it_1e is None else aimb_it_1e
        f_grp = ar['downfold_1e/iter{}'.format(aimb_data['it_1e'])]
        mpi.report("Reading results from downfold_1e iter {}.".format(aimb_data['it_1e']))

        aimb_data['mu'] = f_grp['mu'] * conv_fac
        Gloc_list, G0_list, delta_list, H0_list, Vhf_list, Vhf_dc_list = [], [], [], [], [], [] 
        hopping_list = []
        for ish in range(aimb_data['n_inequiv_shells']):
            Gloc_list.append(f_grp['Gloc_wsIab'][:,:,ish] / conv_fac)
            G0_list.append(f_grp['g_weiss_wsIab'][:,:,ish] / conv_fac)
            delta_list.append(f_grp['delta_wsIab'][:,:,ish] * conv_fac)
            H0_list.append(f_grp['H0_sIab'][:,ish] * conv_fac)
            Vhf_list.append(f_grp['Vhf_gw_sIab'][:,ish] * conv_fac)
            Vhf_dc_list.append(f_grp['Vhf_dc_sIab'][:,ish] * conv_fac)
            hopping = f_grp['H0_skIab'][:,:,ish] + f_grp['Vhf_skIab'][:,:,ish]
            hopping_list.append(hopping * conv_fac)

        aimb_data['number_of_spins'] = Gloc_list[0].shape[1]
        aimb_data['Gloc_w'] = Gloc_list
        aimb_data['g_weiss_w'] = G0_list
        aimb_data['delta_w'] = delta_list
        aimb_data['H0'] = H0_list
        aimb_data['Vhf_mbpt'] = Vhf_list
        aimb_data['Vhf_dc'] = Vhf_dc_list
        if 'Vcorr_gw_sIab' in f_grp:
            mpi.report('Found Vcorr_sIab in the aimbes h5, '
                       'i.e. Embedding on top of an effective QP Hamiltonian.')
            aimb_data['qp_emb'] = True
            # FIXME k-dependent hopping is missing 
            Vcorr_list, Vcorr_dc_list, eal_list = [], [], []
            for ish in range(aimb_data['n_inequiv_shells']):
                Vcorr_list.append(f_grp['Vcorr_gw_sIab'][:,0] * conv_fac)
                Vcorr_dc_list.append(f_grp['Vcorr_dc_sIab'][:,0] * conv_fac)
                eal = H0_list[ish] + Vhf_list[ish] - Vhf_dc_list[ish] + Vcorr_list[ish] - Vcorr_dc_list[ish]
                eal_list.append(eal)
                
                hopping_list[ish] += f_grp['Vcorr_skIab'][:,:,ish] * conv_fac
                
            aimb_data['hopping'] = hopping_list
            aimb_data['Vcorr_mbpt'] = Vcorr_list
            aimb_data['Vcorr_dc'] = Vcorr_dc_list
            aimb_data['effective_atomic_level'] = eal_list
        else:
            aimb_data['qp_emb'] = False
            eal_list, Simp_list, Simp_dc_list = [], [], []
            hopping_w_list = []
            for ish in range(aimb_data['n_inequiv_shells']):
                hopping_w_list.append(f_grp['Sigma_wskIab'][:,:,:,ish] * conv_fac)
                Simp_list.append(f_grp['Sigma_gw_wsIab'][:,:,ish] * conv_fac)
                Simp_dc_list.append(f_grp['Sigma_dc_wsIab'][:,:,ish] * conv_fac)
                eal = H0_list[ish] + Vhf_list[ish] - Vhf_dc_list[ish]
                eal_list.append(eal)
            # FIXME frequency dependent hopping
            aimb_data['hopping'] = hopping_list
            aimb_data['hopping_w'] = hopping_w_list
            aimb_data['effective_atomic_level'] = eal_list
            aimb_data['Sigma_mbpt_w'] = Simp_list
            aimb_data['Sigma_dc_w'] = Simp_dc_list
    mpi.report("")
        
    return aimb_data

        
def aimbes_to_dft_tools(aimb_data, dlr_wmax=None, dlr_eps=None):
    """
    Convert aimbes data into the format that dft_tools like, e.g. block structure and dlr mesh.
    :param aimb_data:
    :param dlr_wmax: 
    :param dlr_eps: 
    :return: 
    """
    if aimb_data['iaft_type'].lower() == 'dlr':
        raise ValueError('iaft_type = dlr is not supported yet')

    mpi.report("AIMBES outputs are found in the IR grids -> Performing transformations from the IR to "
               "DLR meshes.")
    if aimb_data['number_of_spins'] != 1:
        raise ValueError('number of spins = {} is currently not supported '
                         'in AIMBES converter!'.format(aimb_data['number_of_spins']))

    data_dict = {'nspin': aimb_data['nspin'], 'nkpts': aimb_data['nkpts'], 'nkpts_ibz': aimb_data['nkpts_ibz'],
                 'kpts': aimb_data['kpts'], 'kpt_weights': aimb_data['kpt_weights'], 'beta': aimb_data['beta']}

    if dlr_wmax is None:
        data_dict['dlr_wmax'] = aimb_data['iaft_wmax']
    if dlr_eps is None:
        if aimb_data['iaft_prec'] == 1e-15:
            data_dict['dlr_eps'] = 1e-13
        elif aimb_data['iaft_prec'] == 1e-10:
            data_dict['dlr_eps'] = 1e-10
        elif aimb_data['iaft_prec'] == 1e-6:
            data_dict['dlr_eps'] = 1e-6
        else:
            raise ValueError("Incorrect \'iaft_prec\' = {} from aimbes data".format(aimb_data['iaft_prec']))
    
    data_dict['dlr_symmetrize'] = True

    ir_kernel = IAFT(beta=aimb_data['beta'], lmbda=aimb_data['iaft_lambda'], prec=aimb_data['iaft_prec'], verbal=True)
    iw_mesh_dlr_b = MeshDLRImFreq(beta=data_dict['beta'], statistic='Boson',
                                  w_max=data_dict['dlr_wmax'], eps=data_dict['dlr_eps'],
                                  symmetrize=True)
    iw_mesh_dlr_f = MeshDLRImFreq(beta=data_dict['beta'], statistic='Fermion',
                                  w_max=data_dict['dlr_wmax'], eps=data_dict['dlr_eps'],
                                  symmetrize=True)

    (
        V_list,
        U_dlr_list,
        hopping_list, 
        eal_list,
        H0_list,
        Vhf_mbpt_list,
        Vhf_dc_list,
        G0_dlr_list,
        delta_dlr_list,
        Gloc_dlr_list,
    ) = [], [], [], [], [], [], [], [], [], []
    for ish in range(aimb_data['n_inequiv_shells']):
        # fit IR Uloc on DLR iw mesh
        Uloc_w = aimb_data['Uloc_w'][ish]
        temp = _get_dlr_from_IR(Uloc_w, ir_kernel, iw_mesh_dlr_b, dim=4)
        Uloc_dlr = BlockGf(name_list=['up', 'down'], block_list=[temp, temp], make_copies=True)
        U_dlr_list.append(Uloc_dlr)

        Vloc = aimb_data['Vloc'][ish]
        V_list.append({'up': Vloc.copy(), 'down': Vloc})
        
        hopping = aimb_data['hopping'][ish][0]
        hopping_list.append({'up': hopping.copy(), 'down': hopping})

        eal, H0, Vhf, Vhf_dc = (aimb_data['effective_atomic_level'][ish][0], aimb_data['H0'][ish][0],
                                    aimb_data['Vhf_mbpt'][ish][0], aimb_data['Vhf_dc'][ish][0])
        eal_list.append({'up': eal.copy(), 'down': eal})
        H0_list.append({'up': H0.copy(), 'down': H0})
        Vhf_mbpt_list.append({'up': Vhf.copy(), 'down': Vhf})
        Vhf_dc_list.append({'up': Vhf_dc.copy(), 'down': Vhf_dc})

        G0 = aimb_data['g_weiss_w'][ish]
        temp = _get_dlr_from_IR(G0[:, 0, :, :], ir_kernel, iw_mesh_dlr_f, dim=2)
        G0_dlr = BlockGf(name_list=['up', 'down'], block_list=[temp, temp], make_copies=True)
        G0_dlr_list.append(G0_dlr)

        delta = aimb_data['delta_w'][ish]
        temp = _get_dlr_from_IR(delta[:, 0, :, :], ir_kernel, iw_mesh_dlr_f, dim=2)
        delta_dlr = BlockGf(name_list=['up', 'down'], block_list=[temp, temp], make_copies=True)
        delta_dlr_list.append(delta_dlr)
        
        Gloc = aimb_data['Gloc_w'][ish]
        temp = _get_dlr_from_IR(Gloc[:, 0, :, :], ir_kernel, iw_mesh_dlr_f, dim=2)
        Gloc_dlr = BlockGf(name_list=['up', 'down'], block_list=[temp, temp], make_copies=True)
        Gloc_dlr_list.append(Gloc_dlr)

    data_dict['Vloc'] = V_list
    data_dict['Uloc_w'] = U_dlr_list
    data_dict['hopping'] = hopping_list
    data_dict['effective_atomic_level'] = eal_list
    data_dict['H0'] = H0_list
    data_dict['Vhf_mbpt'] = Vhf_mbpt_list
    data_dict['Vhf_dc'] = Vhf_dc_list
    data_dict['G0_dlr'] = G0_dlr_list
    data_dict['delta_dlr'] = delta_dlr_list
    data_dict['Gloc_dlr'] = Gloc_dlr_list

    if not aimb_data['qp_emb']:
        dc_imp_list, Sigma_dlr_list, Sigma_dc_dlr_list = [], [], []
        hopping_w_list = []
        for ish in range(aimb_data['n_inequiv_shells']):
            dc_imp_list.append({'up': Vhf_dc_list[ish]['up'].copy(), 'down': Vhd_dc_list[ish]['down'].copy()})

            Simp_mbpt = aimb_dat['Sigma_mbpt_w'][ish]
            temp = _get_dlr_from_IR(Simp_mbpt[:, 0, :, :], ir_kernel, iw_mesh_dlr_f, dim=2)
            Sigma_dlr = BlockGf(name_list=['up', 'down'], block_list=[temp, temp], make_copies=True)
            Sigma_dlr_list.append(Sigma_dlr)
            
            Simp_dc = aimb_data['Sigma_dc_w'][ish]
            temp = _get_dlr_from_IR(Simp_dc[:, 0, :, :], ir_kernel, iw_mesh_dlr_f, dim=2)
            Sigma_dc_dlr = BlockGf(name_list=['up', 'down'], block_list=[temp, temp], make_copies=True)
            Sigma_dc_dlr_list.append(Sigma_dc_dlr)
            
            hopping_w = aimb_data['hopping_w'][ish] # wskab
            # FIXME what would be the best way to store lattice Green's function 

        data_dict['dc_imp'] = dc_imp_list
        data_dict['Sigma_mbpt_dlr'] = Sigma_dlr_list
        data_dict['Sigma_dc_dlr'] = Sigma_dc_list
    else:
        dc_imp_list, Vcorr_list, Vcorr_dc_list = [], [], []
        for ish in range(aimb_data['n_inequiv_shells']):
            Vcorr, Vcorr_dc = aimb_data['Vcorr_mbpt'][ish][0], aimb_data['Vcorr_dc'][ish][0]
            Vcorr_list.append({'up': Vcorr.copy(), 'down': Vcorr})
            Vcorr_dc_list.append({'up': Vcorr_dc.copy(), 'down': Vcorr_dc})
            dc_imp_list.append({'up': Vhf_dc_list[ish]['up']+Vcorr_dc, 'down': Vhf_dc_list[ish]['down']+Vcorr_dc})
        data_dict['Vcorr_mbpt'] = Vcorr_list
        data_dict['Vcorr_dc'] = Vcorr_dc_list
        data_dict['dc_imp'] = dc_imp_list

    return data_dict


def _get_dlr_from_IR(Gf_ir, ir_kernel, mesh_dlr_iw, dim=2):
    r"""
    Interpolate a given Gf from IR mesh to DLR mesh

    Parameters
    ----------
    Gf_ir : np.ndarray
        Green's function on IR mesh
    ir_kernel : sparse_ir
        IR kernel object
    mesh_dlr_iw : MeshDLRImFreq
        DLR mesh
    dim : int, optional
        dimension of the Green's function, defaults to 2

    Returns
    -------
    Gf_dlr : BlockGf or Gf
        Green's function on DLR mesh
    """

    n_orb = Gf_ir.shape[-1]
    stats = 'f' if mesh_dlr_iw.statistic == 'Fermion' else 'b'

    if stats == 'b':
        Gf_ir_pos = Gf_ir.copy()
        Gf_ir = np.zeros([Gf_ir_pos.shape[0] * 2 - 1] + [n_orb] * dim, dtype=complex)
        Gf_ir[: Gf_ir_pos.shape[0]] = Gf_ir_pos[::-1]
        Gf_ir[Gf_ir_pos.shape[0] :] = Gf_ir_pos[1:]

    Gf_dlr_iw = Gf(mesh=mesh_dlr_iw, target_shape=[n_orb] * dim)

    # prepare idx array for spare ir
    if stats == 'f':
        mesh_dlr_iw_idx = np.array([iwn.index for iwn in mesh_dlr_iw])
    else:
        mesh_dlr_iw_idx = np.array([iwn.index for iwn in mesh_dlr_iw])

    Gf_dlr_iw.data[:] = ir_kernel.w_interpolate(Gf_ir, mesh_dlr_iw_idx, stats=stats, ir_notation=False)

    return Gf_dlr_iw