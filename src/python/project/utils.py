import os
import logging
from glob import glob
from msmbuilder.utils import keynat
from msmbuilder import Trajectory
from msmbuilder.utils import keynat

from project import Project
from validators import ValidationError
logger = logging.getLogger(__name__)

def get_project_object( traj_directory, conf_filename, out_filename=None ):
    """
    This function constructs a msmbuilder.Project object 
    given a directory of trajectories saved as .lh5's. 

    Note that this is only really necessary when a script
    like ConvertDataToLHDF.py converts the data but fails
    to write out the ProjectInfo.yaml file.

    This function can also be used to combine two projects
    by copying and renaming the trajectories in a new 
    folder. Though, it's probably more efficient to just
    do some bash stuff to cat the ProjectInfo.yaml's 
    together and rename the trajectories.
    
    Inputs:
    -------
    1) traj_directory : directory to find the trajectories
    2) conf_filename : file to find the conformation
    3) out_filename [ None ] : if None, then this function 
        does not save the project file, but if given, the
        function will save the project file and also
        return the object

    Outputs:
    -------
    project : msmbuilder.Project object corresponding to 
        your project.
    """

    traj_paths = sorted( os.listdir( traj_directory ), key=keynat ) # relative to the traj_directory
    traj_paths = [ os.path.join( traj_directory, filename ) for filename in traj_paths ] # relative to current directory

    traj_lengths = []

    for traj_filename in traj_paths: # Get the length of each trajectory
        logger.info( traj_filename )
        traj_lengths.append( Trajectory.load_from_lhdf( traj_filename, JustInspect=True )[0] ) 
        # With JustInspect=True this just returns the shape of the XYZList

    project = Project({'conf_filename': conf_filename,
                       'traj_lengths': traj_lengths,
                       'traj_paths': traj_paths,
                       'traj_errors': [None] * len(traj_paths),
                       'traj_converted_from': [ [None] ] * len(traj_paths) })

    if out_filename is None:
        return project
    else:
        project.save( out_filename )
        logger.info('Saved project file to %s', out_filename)
        return project
