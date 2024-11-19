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
Utility functions for reading and writing QE output files.
"""

from triqs.utility import mpi
import numpy as np
import xml.etree.ElementTree as ET

from scipy.constants import physical_constants
HARTREE_EV = physical_constants['Hartree energy in eV'][0]


def read_qe_misc_data(qe_xml_file, file_nnkp=None, file_isym=None):
    """
    Read miscellaneous QE outputs from a QE xml file.

    Parameters
    ----------
    qe_xml_file : str
        XML file from QE pwscf
    
    file_nnkp : str, optional
        nnkpt file from the Wannier90 pre-processing step

    file_isym: str, optional
        Text file containing symmetry information, pre-generated using symWannier package
    
    Returns
    -------
    nbnd : int 
        Number of KS bands 
    
    fermi : float
        Fermi energy in eV
    
    recv : numpy.ndarray(3, 3)
        Reciprocal lattice in the unit of 2*pi/lattice_constant
    
    occs : numpy.ndarray(nkpts_full, nbnd)
        K-dependent occupations from DFT
    """
    # Load and parse the XML file
    mpi.report("--------------------------------------------")
    mpi.report(f"*** Reading QE misc data from {qe_xml_file}")
    mpi.report("--------------------------------------------\n")
    tree = ET.parse(qe_xml_file)
    root = tree.getroot()
    bs_node = root.find("output/band_structure")

    # Metadata
    lsda = parse_bool(bs_node.find('lsda').text)
    nspin = 2 if lsda else 1
    nbnd = int(bs_node.find('nbnd').text)
    mp_grid = np.zeros(3, dtype=int)
    mp_grid_att = bs_node.find('starting_k_points/monkhorst_pack').attrib
    for i in range(3):
        mp_grid[i] = mp_grid_att[f'nk{i + 1}']
    nkpts = mp_grid[0] * mp_grid[1] * mp_grid[2]
    nkpts_ibz = int(bs_node.find('nks').text)
    fermi = float(bs_node.find('fermi_energy').text) * HARTREE_EV

    mpi.report(f"nspin, nbnd, nkpts, nkpts_ibz : {nspin}, {nbnd}, {nkpts}, {nkpts_ibz}\n")
    mpi.report(f"Fermi energy: {fermi:.3f} eV\n")

    # reciprocal vectors
    recv = np.zeros((3,3), dtype=float)
    #alat = float(root.find("output/atomic_structure").attrib['alat'])
    #tpiba = 2 * np.pi / alat
    recv_node = root.find("output/basis_set/reciprocal_lattice")
    for i in range(3):
        bi = recv_node.find(f'b{i+1}').text.strip().split()
        for j in range(3):
            recv[i, j] = float(bi[j]) #* tpiba

    # Occupations and KS energies
    occs = np.zeros((nkpts_ibz, nbnd), dtype=int)
    eigvals = np.zeros((nkpts_ibz, nbnd), dtype=float)
    ik = 0
    for ks_k in bs_node.findall("ks_energies"):
        occ_k = ks_k.find("occupations")
        eps_k = ks_k.find("eigenvalues")

        nbnd_k = int(occ_k.attrib['size'])
        assert nbnd_k == nbnd

        for i, x in enumerate(occ_k.text.strip().split()):
            occs[ik, i] = int(float(x))
        for i, x in enumerate(eps_k.text.strip().split()):
            eigvals[ik, i] = float(x) * HARTREE_EV
        ik += 1
        
    if nkpts != nkpts_ibz:
        mpi.report(f"Symmetries detected in {qe_xml_file}: nkpts ({nkpts}) != nkpts_ibz ({nkpts_ibz}). \n"
                   f"Use symWannier package to unfold occupations and KS energies to the full BZ.\n")
        assert file_nnkp is not None, "Unfolding to the full BZ requires providing {seedname}.nnkp through `file_nnkp`."
        assert file_isym is not None, "Unfolding to the full BZ requires providing {seedname}.isym through `file_isym`."
        try:
            from py_w90_driver.symwannier.nnkp import Nnkp
            from symwannier.sym import Sym
        except ImportError:
            raise ImportError(
                "read_qe_misc_data: Unfolding KS energies and occupations requires the "
                "symWannier package (https://github.com/wannier-utils-dev/symWannier/tree/main). \n"
                "Ensure to append \"/path/to/symWannier/src\" to your \"PYTHONPATH\". \n"
                "Otherwise, you can disable symmetires in QE.")

        nnkp = Nnkp(file_nnkp=file_nnkp)
        sym = Sym(file_sym=file_isym, nnkp=nnkp)
        assert nkpts == sym.nkf, (f"nkpoints mismatch ({nkpts}, {sym.nkf}) while unfolding the IBZ. \n"
                                  f"Double check the kmesh from `nscf` and `pw2wannier90` calculations.")

        occs_fbz = np.zeros((nkpts, nbnd), dtype=int)
        eigvals_fbz = np.zeros((nkpts, nbnd), dtype=float)

        for ik, k in enumerate(sym.full_kpoints):
            iks = sym.equiv[ik]
            occs_fbz[ik, :] = occs[iks, :]
            eigvals_fbz[ik, :] = eigvals[iks, :]

        mpi.report("--------------------------------------------\n")
        return nbnd, fermi, recv, occs_fbz#, eigvals_fbz
    else:
        mpi.report("--------------------------------------------\n")
        return nbnd, fermi, recv, occs#, eigvals


def parse_bool(value):
    if value.lower() in ("true", "1"):
        return True
    elif value.lower() in ("false", "0"):
        return False
    else:
        raise ValueError(f"Cannot convert '{value}' to a boolean.")
    