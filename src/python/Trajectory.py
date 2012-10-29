# This file is part of MSMBuilder.
#
# Copyright 2011 Stanford University
#
# MSMBuilder is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""Trajectory stores a sequence of conformations.
"""

import copy
import os
import tables
import numpy as np

from msmbuilder import PDB
from msmbuilder import io
from msmbuilder.Conformation import ConformationBaseClass, Conformation
from msmbuilder import xtc
from msmbuilder import dcd

import logging
logger = logging.getLogger(__name__)

MAXINT16 = np.iinfo(np.int16).max
MAXINT32 = np.iinfo(np.int32).max
DEFAULT_PRECISION = 1000


def _convert_to_lossy_integers(data, precision=DEFAULT_PRECISION):
    """Implementation of the lossy compression used in Gromacs XTC using the 
    pytables library.  Convert 32 bit floats into 16 bit integers.  These 
    conversion functions have been optimized for memory use.  Further memory 
    reduction would require an in-place astype() operation, which one could 
    create using ctypes."""
    if (np.max(data) * float(precision) < MAXINT16) and \
            (np.min(data) * float(precision) > -MAXINT16):
        data *= float(precision) 
        rounded_data = data.astype("int16")
        data /= float(precision) 
    else:
        data *= float(precision)
        rounded = data.astype("int32")
        data /= float(precision)
        logger.error("Data range too large for int16: try removing center of \
        mass motion, check for 'blowing up, or just use .h5 or .xtc format.'")
    return rounded


def _convert_from_lossy_integers(data, precision=DEFAULT_PRECISION):
    """Implementation of the lossy compression used in Gromacs XTC using the 
    pytables library.  Convert 16 bit integers into 32 bit floats."""
    data_copy = data.astype("float32")
    data_copy /= float(precision)
    return data_copy


class Trajectory(ConformationBaseClass):
    """This is the representation of a sequence of  conformations.

    Notes:
    Use classmethod load_from_pdb to create an instance of this class from a 
    PDB filename.
    The Trajectory is a dictionary-like object.
    The dictionary key 'XYZList' contains a numpy array of XYZ coordinates, 
    stored such that:
    X[i,j,k] gives Frame i, Atom j, Coordinate k.
    """
    def __init__(self, conf):
        """Create a Trajectory from a single conformation.

        Leaves the XYZList key as an empty list for easy appending.

        Parameters
        ----------
        S: Conformation
            The single conformation to build a trajectory from

        Notes
        -----
        You probably want to use something else to load trajectories

        See Also
        --------
        load_from_pdb
        load_from_hdf
        load_from_lhdf
        """

        ConformationBaseClass.__init__(self, conf)
        self["XYZList"] = []
        if "XYZList" in conf:
            self["XYZList"] = conf["XYZList"].copy()

    def subsample(self, stride):
        """Keep only the frames at some interval

        Shorten trajectory by taking every `stride`th frame. Note that this is
        an inplace operation!

        Parameters
        ----------
        stride: int
            interval

        Notes
        -----
        Trajectory supports fancy indexing, so you can do this by yourself

        Example
        -------
        >> mytrajectory.subsample(10)

        >> mytrajectory[::10]

        These do the same thing except that one is inplace and one is not

        """
        if stride == 1:
            return

        self["XYZList"] = self["XYZList"][::stride].copy()

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice) or \
                isinstance(key, np.ndarray):

            if isinstance(key, int):
                key = [key]

            newtraj = copy.copy(self)
            newtraj['XYZList'] = self['XYZList'][key]

            return newtraj

        return super(Trajectory, self).__getitem__(key)

    def __len__(self):
        return len(self['XYZList'])

    def __add__(self, other):
        """Addition means concatenation of two trajectory instances"""
        # Check type of other
        if not isinstance(other, Trajectory):
            raise TypeError('You can only add two Trajectory instances')
        concatenated_trajs = copy.deepcopy(self)
        # Simply copy the XYZList in here if the Trajectory is Empty.
        if 'XYZList' not in self and 'XYZList' not in other:
            pass
        elif 'XYZList' not in self:
            concatenated_trajs['XYZList'] = copy.deepcopy(other['XYZList'])
        elif 'XYZList' not in other:
            concatenated_trajs['XYZList'] = copy.deepcopy(self['XYZList'])
        else:
            if self['XYZList'].shape[1] != other['XYZList'].shape[1]:
                raise TypeError('The two trajectories don\'t have the same \
                                number of atoms')
            concatenated_trajs['XYZList'] = np.vstack((self['XYZList'], \
                                                        other['XYZList']))

        return concatenated_trajs

    def __iadd__(self, other):
        """Addition means concatenation of two trajectory instances"""
        # NOTE: CRS thinks this should just use __add__ like so:
        return self + other

        ## Check type of other
        #if not isinstance(other, Trajectory):
        #    raise TypeError('You can only add two Trajectory instances')
        ## Simply copy the XYZList in here if the Trajectory is Empty.
        #if 'XYZList' not in self:
        #    self['XYZList'] = copy.deepcopy(other['XYZList'])
        #else:
        #    # Check number of atoms.
        #    if not self['XYZList'].shape[1] == other['XYZList'].shape[1]:
        #        raise TypeError('The two trajectories don\'t have the same number of atoms')
        #    self['XYZList'] = np.vstack((self['XYZList'], other['XYZList']))
        #return self

    def restrict_atom_indices(self, atom_indices):
        ConformationBaseClass.restrict_atom_indices(self, atom_indices)

        self['XYZList'] = copy.copy(self['XYZList'][:, atom_indices])

    def save_to_lhdf(self, filename, precision=DEFAULT_PRECISION):
        """Save a Trajectory instance to a Lossy HDF File.

        First, remove the XYZList key because it should be written using the
        special CArray operation.  This file format is roughly equivalent to
        an XTC and should comparable file sizes but with better IO performance.

        Parameters
        ----------
        Filename: str
            location to save to
        Precision : float, optional
            Precision to save xyzlist
        """
        # This doesn't get saved it's just regenerated every time the trajectory is loaded
        index_list = self.pop('IndexList') 
        
        xyzlist = self.pop('XYZList')
        rounded = _convert_to_lossy_integers(xyzlist, precision)
        self['XYZList'] = rounded

        io.saveh(filename, **self)

        # Return the instance to what it was before the save:
        self['XYZList'] = xyzlist
        self['IndexList'] = index_list 
        
    def save_to_xtc(self, filename, precision=DEFAULT_PRECISION):
        """Dump the coordinates to XTC

        Parameters
        ----------
        Filename: str
            location to save to
        precision: float, optional
            I'm not really sure what this does (RTM 6/27).
        """
 
        if os.path.exists(filename):
            raise IOError("%s already exists" % filename)

        XTCFile = xtc.XTCWriter(filename)

        for i in range(len(self["XYZList"])):
            XTCFile.write(self["XYZList"][i], 1, i, np.eye(3, 3, dtype='float32'), precision)

    def save_to_pdb(self, filename):
        """Dump the coordinates to PDB

        Parameters
        ----------
        Filename: str
            location to save to

        Notes
        -----
        Don't use this for a very big trajectory. PDB is plaintext and takes a lot
        of memory
        """

        for i in range(len(self["XYZList"])):
            PDB.WritePDBConformation(filename, self["AtomID"], self["AtomNames"], \
                                        self["ResidueNames"], self["ResidueID"], \
                                        self["XYZList"][i], self["ChainID"])

    def save_to_xyz(self, filename):
        """Dump the coordinates to XYZ format

        Parameters
        ----------
        Filename: str
            location to save to

        Notes
        -----
        TODO: What exactly is the XYZ format? (RTM 6/27) 
            CRS: I think tinker uses it?
        """

        output_lines = []
        title = 'From MSMBuilder save_to_xyz function'
        for i in range(len(self["XYZList"])):
            xyz = self["XYZList"][i]
            num_atoms = xyz.shape[0]
            output_lines.append('%i' % num_atoms)
            output_lines.append(title)
            for j in range(xyz.shape[0]):
                output_lines.append("%-5s% 12.6f% 12.6f% 12.6f\n" % (self["AtomNames"][j], \
                                    xyz[j, 0], xyz[j, 1], xyz[j, 2]))
                # The above line seems to me to have an extra % for no reason "%-5s%<-here" 

        with open(filename, 'w') as f:
            f.writelines(output_lines)

    def save(self, filename, precision=DEFAULT_PRECISION):
        """Dump the coordinates to disk in format auto-detected by filename

        Parameters
        ----------
        Filename: str
            location to save to

        Notes
        -----
        Formats supported are h5, xtc, pdb, lh5 and xyz
        """

        extension = os.path.splitext(filename)[1]

        if extension == '.h5':
            self.save_to_hdf(Filename)
        elif extension == '.xtc':
            self.save_to_xtc(Filename)
        elif extension == '.pdb':
            self.save_to_pdb(Filename)
        elif extension == '.lh5':
            self.save_to_lhdf(Filename, precision=precision)
        elif extension == '.xyz':
            self.save_to_xyz(Filename)
        else:
            raise IOError("File: %s. I don't understand the extension '%s'" % \ 
                            (Filename, extension) )

    def append_pdb(self, filename):
        """Add on to a pdb file

        Parameters
        ----------
        Filename: str
            location to save to

        NOTE: I don't think this will actually work (CRS 10/29/12) because:
            - conf_from_pdb could be a trajectory or a single conformation
            - This function never saved anything. Does this even belong here? 
            - The docstring indicates you're adding to a PDB, whereas this is adding a pdb
                to the current trajectory
            - I need help deciding what the intended purpose is.
        """
        try:
            self["XYZList"] = self["XYZList"].tolist()
        except:
            pass
        conf_from_pdb = Conformation.load_from_pdb(filename)
        temp_xyz = conf_from_pdb["XYZ"]
        if len(temp_xyz) != len(self["XYZList"][1]): # CRS switched to axis=1 since that 
            # should be the number of atoms.
            raise NameError("Tried to add wrong number of coordinates.")
        else:
            self["XYZList"].append(temp)

    @classmethod
    def load_from_pdb(cls, filename):
        """Create a Trajectory from a PDB Filename

        Parameters
        ----------
        filename: str
            location to load from
        """
        return Trajectory(PDB.LoadPDB(filename, AllFrames=True)) # Still need to pep8ify PDB.py

    @classmethod
    def load_from_xtc(cls, xtc_filename_list, pdb_filename=None, conf=None, pre_allocate=True,\
                    just_inspect=False, discard_overlapping_frames=False):
        """Create a Trajectory from a collection of XTC files

        Parameters
        ----------
        xtc_filename_list: list
            list of files to load from
        pdb_filename: str, optional
            XTC format doesnt have the connectivity information, which needs to be
            supplied. You can either supply it by giving a path to the PDB file (here)
            or by suppling a Conf or Traj object containing the right connectivity
            (next arg)
        conf: Conformation, optional
            A conformation (actually passing another trajectory will work) that has
            the right atom labeling
        pre_allocate: bool, optional
            This doesnt do anything
        just_inspect: bool, optional
            Dont actually load, just return dimensions
        discard_overlapping_frames: bool, optional
            Check for redundant frames and discard them. (RTM 6/27 should this be default True?)

        Returns
        -------
        trajectory: Trajectory
            Trajectory loaded from disk. OR, if you supplied `just_inspect`=True,
            then just the shape
        """
        
        # temp_traj is the to be trajectory object, but each key is added in steps, so it
        # is not complete until the very end
        if pdb_filename != None:
            temp_traj = Trajectory.load_from_pdb(pdb_filename)
        elif conf != None:
            temp_traj = Trajectory(conf)
        else:
            raise Exception("ERROR: Need a conformation to construct a trajectory.")

        if not just_inspect:

            temp_traj["XYZList"] = []
            num_redundant = 0

            for i, c in enumerate(xtc.XTCReader(XTCFilenameList)):
                # check to see if we have redundant frames as we load them up
                if discard_overlapping_frames:
                    if i > 0:
                        if np.sum(np.abs(c.coords - temp_traj["XYZList"][-1])) < 10. ** -8:
                            num_redundant += 1
                    temp_traj["XYZList"].append(np.array(c.coords).copy())

                else:
                    temp_traj["XYZList"].append(np.array(c.coords).copy())

            temp_traj["XYZList"] = np.array(temp_traj["XYZList"])
            if num_redundant != 0:
                logger.warning("Found and discarded %d redundant snapshots in loaded traj", \
                                 num_redundant)
        else: # just_inspect is True, so return the shape of the data
            i = 0
            for c in xtc.XTCReader(XTCFilenameList):
                if i == 0:
                    conf_shape = np.shape(c.coords)
                i += 1
                # This loop assumes that all of the XTC's are the SAME. This is true for F@h
                # datasets, so I suppose it's alright. But to be general we should check 
                # every XTC I think
            shape = np.array((i, conf_shape[0], conf_shape[1]))
            return shape

        return temp_traj

    @classmethod
    def load_from_dcd(cls, filename_list, pdb_filename=None, conf=None, pre_allocate=True, \
                        just_inspect=False):
        """Create a Trajectory from a Filename."""

        # temp_traj is the to be trajectory object, but each key is added in steps, so it
        # is not complete until the very end

        if pdb_filename != None:
            temp_traj = Trajectory.load_from_pdb(pdb_filename)
        elif conf != None:
            temp_traj = Trajectory(conf)
        else:
            raise Exception("ERROR: Need a conformation to construct a trajectory.")

        if not just_inspect:
            temp_traj["XYZList"] = []
            for c in dcd.DCDReader(filename_list):
                temp_traj["XYZList"].append(c.copy())
            temp_traj["XYZList"] = np.array(temp_traj["XYZList"])
        else:  # This is wasteful to read everything in just to get the length
            xyz = []
            for c in dcd.DCDReader(filename_list):
                xyz.append(c.copy())
            xyz = np.array(xyz)
            return xyz.shape

        return temp_traj

    @classmethod
    def load_from_trr(cls, trr_filename_list, pdb_filename=None, conf=None, pre_allocate=True,\
                        just_inspect=False):
        """Load a trajectory from a single or many .trr files."""
        # temp_traj is the to be trajectory object, but each key is added in steps, so it
        # is not complete until the very end
        if pdb_filename != None:
            temp_traj = Trajectory.load_from_pdb(pdb_filename)
        elif conf != None:
            temp_traj = Trajectory(conf)
        else:
            raise Exception("ERROR: Need a conformation to construct a trajectory.")
        if not just_inspect:
            temp_traj["XYZList"] = []
            temp_traj["Velocities"] = [] # Do we care about these? (and Forces)
            temp_traj["Forces"] = []
            for c in xtc.TRRReader(trr_filename_list):
                A["XYZList"].append(np.array(c.coords).copy())
                A["Velocities"].append(np.array(c.velocities).copy())
                A["Forces"].append(np.array(c.forces).copy())
            A["XYZList"] = np.array(A["XYZList"])
            A["Velocities"] = np.array(A["Velocities"])
            A["Forces"] = np.array(A["Forces"])
        else:
            i = 0
            for c in xtc.TRRReader(TRRFilenameList):
                if i == 0:
                    ConfShape = np.shape(c.coords)
                i += 1
            Shape = np.array((i, ConfShape[0], ConfShape[1]))
            return(Shape)
        return(A)

    @classmethod
    def load_from_pdbList(cls, Filenames):
        """Create a Trajectory with title Title from a Filename."""
        A = Trajectory.load_from_pdb(Filenames[0])
        for f in Filenames[1:]:
            A.AppendPDB(f)
        A["XYZList"] = np.array(A["XYZList"])
        return(A)

    @classmethod
    def enum_chunks_from_hdf(cls, TrajFilename, Stride=None, AtomIndices=None, ChunkSize=50000):
        """
        Function to read trajectory files which have been saved as HDF.

        This function is an iterable, so should be used like:

        from msmbuilder import Trajectory
        for trajectory_chunk in Trajectory.enum_chunks_from_hdf(
            ... # Do something with each chunk. The chunk looks like a regular Trajectory instance

        Inputs:
        - TrajFilename: Filename to find the trajectory
        - Stride [None]: Integer number of frames to subsample the trajectory
        - AtomIndices [None]: np.ndarray of atom indices to read in (0-indexed)
        - ChunkSize [100000]: Integer number of frames to read in a chunk
            NOTE: ChunkSize will change in order to be a multiple of the input Stride
                This is necessary in order to make sure the Stride and chunks line up

        Outputs:
        - Nothing. This is an iterable function, so it yields Trajectory instances
        """
        RestrictAtoms = False
        if AtomIndices != None:
            RestrictAtoms = True
        if Stride != None:
            while ChunkSize % Stride != 0:  # Need to do this in order to make sure we stride correctly.
                                            # since we read in chunks, and then we need the strides
                                            # to line up
                ChunkSize -= 1

        A={}
        F=tables.File(TrajFilename,'r')
        # load all the data other than XYZList

        if RestrictAtoms:
            A['AtomID'] = np.array(F.root.AtomID[AtomIndices], dtype=np.int32)
            A['AtomNames'] = np.array(F.root.AtomNames[AtomIndices])
            A['ChainID'] = np.array(F.root.ChainID[AtomIndices])
            A['ResidueID'] = np.array(F.root.ResidueID[AtomIndices], dtype=np.int32)
            A['ResidueNames'] = np.array(F.root.ResidueNames[AtomIndices])

            # IndexList is a VLArray, so we need to read the whole list with node.read() (same as node[:]) and then loop through each
                # row (residue) and remove the atom indices that are not wanted
            #A['IndexList'] = [ [ i for i in row if (i in AtomIndices) ] for row in F.root.IndexList[:] ]
            
        else:
            A['AtomID'] = np.array( F.root.AtomID[:], dtype=np.int32 )
            A['AtomNames'] = np.array( F.root.AtomNames[:] )
            A['ChainID'] = np.array( F.root.ChainID[:])
            A['ResidueID'] = np.array( F.root.ResidueID[:], dtype=np.int32 )
            A['ResidueNames'] = np.array( F.root.ResidueNames[:] )
            
            #A['IndexList'] = F.root.IndexList[:]

        #A['SerializerFilename'] = os.path.abspath(TrajFilename)

        # Loaded everything except XYZList

        Shape = F.root.XYZList.shape
        begin_range_list = np.arange(0, Shape[0], ChunkSize)
        end_range_list = np.concatenate((begin_range_list[1:], [Shape[0]]))

        for r0, r1 in zip(begin_range_list, end_range_list):

            if RestrictAtoms:
                A['XYZList'] = np.array(F.root.XYZList[r0: r1: Stride, AtomIndices])
            else:
                A['XYZList'] = np.array(F.root.XYZList[r0: r1: Stride])

            yield cls(A)

        F.close()

        return

    @classmethod
    def enum_chunks_from_lhdf(cls, TrajFilename, precision=DEFAULT_PRECISION, Stride=None, AtomIndices=None, ChunkSize=50000):
        """
        Method to read trajectory files which have been saved as LHDF.
        Note that this method simply calls the enum_chunks_from_hdf method.

        This function is an iterable, so should be used like:

        from msmbuilder import Trajectory
        for trajectory_chunk in Trajectory.enum_chunks_from_lhdf(
            ... # Do something with each chunk. The chunk looks like a regular Trajectory instance

        Inputs:
        - TrajFilename: Filename to find the trajectory
        - precision [1000]: precision used when saving as lossy integers
        - Stride [None]: Integer number of frames to subsample the trajectory
        - AtomIndices [None]: np.ndarray of atom indices to read in (0-indexed)
        - ChunkSize [100000]: Integer number of frames to read in a chunk
            NOTE: ChunkSize will change in order to be a multiple of the input Stride
                This is necessary in order to make sure the Stride and chunks line up

        Outputs:
        - Nothing. This is an iterable function, so it yields Trajectory instances
        """
        for A in cls.enum_chunks_from_hdf(TrajFilename, Stride, AtomIndices, ChunkSize):
            A['XYZList'] = _convert_from_lossy_integers(A['XYZList'], precision)
            yield A

        return

    @classmethod
    def load_from_hdf(cls, TrajFilename, JustInspect=False, Stride=None, AtomIndices=None, ChunkSize=50000):
        """
        Method to load a trajectory which was saved as HDF

        Inputs:
        - TrajFilename: Filename to find the trajectory
        - JustInspect [False]: If True, then the method returns the shape of the
            XYZList stored on disk
        - Stride [None]: Integer number of frames to subsample the trajectory
        - AtomIndices [None]: np.ndarray of atom indices to read in (0-indexed)
        - ChunkSize [100000]: Integer number of frames to read in a chunk
            NOTE: ChunkSize will change in order to be a multiple of the input Stride
                This is necessary in order to make sure the Stride and chunks line up

        Outputs:
        - A: Trajectory instance read from disk
        """
        if not JustInspect:
            chunk_list = list(cls.enum_chunks_from_hdf(TrajFilename, Stride=Stride, AtomIndices=AtomIndices, ChunkSize=ChunkSize))
            A = chunk_list[0]
            A['XYZList'] = np.concatenate([t['XYZList'] for t in chunk_list])
            return A

        else:
            F1 = tables.File(TrajFilename)
            Shape = F1.root.XYZList.shape
            F1.close()
            return(Shape)

    @classmethod
    def load_from_lhdf(cls, TrajFilename, JustInspect=False, precision=DEFAULT_PRECISION, Stride=None, AtomIndices=None, ChunkSize=50000):
        """
        Method to load a trajectory which was saved as LHDF

        Inputs:
        - TrajFilename: Filename to find the trajectory
        - JustInspect [False]: If True, then the method returns the shape of the
            XYZList stored on disk
        - Stride [None]: Integer number of frames to subsample the trajectory
        - precision [1000]: precision used when saving as lossy integers
        - AtomIndices [None]: np.ndarray of atom indices to read in (0-indexed)
        - ChunkSize [100000]: Integer number of frames to read in a chunk
            NOTE: ChunkSize will change in order to be a multiple of the input Stride
                This is necessary in order to make sure the Stride and chunks line up

        Outputs:
        - A: Trajectory instance read from disk
        """
        if not JustInspect:
            A = cls.load_from_hdf(TrajFilename, Stride=Stride, AtomIndices=AtomIndices)
            A['XYZList'] = _convert_from_lossy_integers(A['XYZList'], precision)
            return A

        else:
            F1 = tables.File(TrajFilename)
            Shape = F1.root.XYZList.shape
            F1.close()
            return(Shape)

    @classmethod
    def read_xtc_frame(cls, TrajFilename, WhichFrame):
        """Read a single frame from XTC trajectory file without loading file into memory."""
        i = 0
        for c in xtc.XTCReader(TrajFilename):
            if i == WhichFrame:
                return(np.array(c.coords))
            i += 1
        raise Exception("Frame %d not found in file %s; last frame found was %d" % (WhichFrame, TrajFilename, i))

    @classmethod
    def read_dcd_frame(cls, TrajFilename, WhichFrame):
        """Read a single frame from DCD trajectory without loading file into memory."""
        reader = dcd.DCDReader(TrajFilename, firstframe=WhichFrame, lastframe=WhichFrame)
        xyz = None
        for c in reader:
            xyz = c.copy()
        if xyz == None:
            raise Exception("Frame %s not found in file %s." % (WhichFrame, TrajFilename))

        return xyz

    @classmethod
    def read_hdf_frame(cls, TrajFilename, WhichFrame):
        """Read a single frame from HDF5 trajectory file without loading file into memory."""
        F1 = tables.File(TrajFilename)
        XYZ = F1.root.XYZList[WhichFrame]
        F1.close()
        return(XYZ)

    @classmethod
    def read_lhdf_frame(cls, TrajFilename, WhichFrame, precision=DEFAULT_PRECISION):
        """Read a single frame from Lossy LHDF5 trajectory file without loading file into memory."""
        F1 = tables.File(TrajFilename)
        XYZ = F1.root.XYZList[WhichFrame]
        F1.close()
        XYZ = _convert_from_lossy_integers(XYZ, precision)
        return(XYZ)

    @classmethod
    def read_frame(cls, TrajFilename, WhichFrame, Conf=None):
        extension = os.path.splitext(TrajFilename)[1]

        if extension == '.xtc':
            return(Trajectory.read_xtc_frame(TrajFilename, WhichFrame))
        elif extension == '.h5':
            return(Trajectory.read_hdf_frame(TrajFilename, WhichFrame))
        elif extension == '.lh5':
            return(Trajectory.read_lhdf_frame(TrajFilename, WhichFrame))
        elif extension == '.dcd':
            return(Trajectory.read_dcd_frame(TrajFilename, WhichFrame))
        else:
            raise IOError("Incorrect file type--cannot get conformation %s" % TrajFilename)

    @classmethod
    def load_trajectory_file(cls, Filename, JustInspect=False, Conf=None, 
                             Stride=1, AtomIndices=None):
        """Loads a trajectory into memory, automatically deciding which methods to call based on filetype.  For XTC files, this method uses a pre-registered Conformation filename as a pdb."""
        
        extension = os.path.splitext(Filename)[1]
        
        # check to see if we're supposed to load only a subset of the atoms
        if AtomIndices != None:
            if (extension == '.lh5') or (extension == '.h5'):
                pass # we deal with this below
            else:
                raise NotImplementedError('AtomIndices kwarg option only'
                                          'available for .lh5 & .h5 format')
            

        # if we're not going to load a subset of the atoms, then proceed 
        if extension == '.h5':
            return Trajectory.load_from_hdf(Filename, JustInspect=JustInspect, Stride=Stride, AtomIndices=AtomIndices)

        elif extension == '.xtc':
            if Conf == None:
                raise Exception("Need to register a Conformation to use XTC Reader.")
            return Trajectory.load_from_xtc(Filename, Conf=Conf, JustInspect=JustInspect)[::Stride]

        elif extension == '.dcd':
            if Conf == None:
                raise Exception("Need to register a Conformation to use DCD Reader.")
            return Trajectory.load_from_dcd(Filename, Conf=Conf, JustInspect=JustInspect)[::Stride]

        elif extension == '.lh5':
            return Trajectory.load_from_lhdf(Filename, JustInspect=JustInspect, Stride=Stride, AtomIndices=AtomIndices)

        elif extension == '.pdb':
            return Trajectory.load_from_pdb(Filename)[::Stride]

        else:
            raise IOError("File: %s. I don't understand the extension '%s'" % (Filename, extension))

    @classmethod
    def append_frames_to_file(cls, filename, XYZList, precision=DEFAULT_PRECISION,
                           discard_overlapping_frames=False):
        """Append an array of XYZ data to an existing .h5 or .lh5 file.
        """

        extension = os.path.splitext(filename)[1]

        if extension in [".h5", ".lh5"]:
            File = tables.File(filename, "a")
        else:
            raise Exception("File must be .h5 or .lh5")

        if not File.root.XYZList.shape[1:] == XYZList.shape[1:]:
            raise Exception("Error: data cannot be appended to trajectory due to incorrect shape.")

        if extension == ".h5":
            z = XYZList
        elif extension == ".lh5":
            z = _convert_to_lossy_integers(XYZList, precision)

        # right now this only checks for the last written frame
        if discard_overlapping_frames:
            while (File.root.XYZList[-1, :, :] == z[0, :, :]).all():
                z = z[1:, :, :]

        File.root.XYZList.append(z)

        File.flush()
        File.close()

    @classmethod
    def _reduce_redundant_snapshots(cls, trajectory):
        """ Takes a trajectory object, and removes data from the 'XYZList' entry
            that are duplicated snapshots. Does this by checking if two contiguous
            snapshots are binary equivalent.
        """

        not_done = True
        i = 0  # index for the snapshot we're working on
        n = 0  # counts the number of corrections

        while not_done:

            # check to see if we are done
            if i == trajectory["XYZList"].shape[0] - 1:
                break

            if (trajectory["XYZList"][i, :, :] == trajectory["XYZList"][i + 1, :, :]).all():
                trajectory = trajectory[:i] + trajectory[i + 1:]
                n += 1
            else:
                i += 1

        if n != 0:
            logger.warning("Found and eliminated %d redundant snapshots in trajectory", n)

        return trajectory
