"""MSMBuilder: a python library for Markov state models of conformational dynamics

MSMBuilder (https://simtk.org/home/msmbuilder)
is a library that provides tools for analyzing molecular dynamics
simulations, particularly through the construction
of Markov state models for conformational dynamics.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

DOCLINES = __doc__.split("\n")

import os
import sys
import tempfile
import shutil
import subprocess
from glob import glob
from distutils.version import StrictVersion
from distutils.command.build_scripts import build_scripts
from setuptools import setup
PY3 = sys.version_info >= (3,0)

#########################################
VERSION = "2.8.3"
ISRELEASED = False
__author__ = "MSMBuilder Team"
__version__ = VERSION
########################################


def warn_on_version(module_name, minimum=None, package_name=None, recommend_conda=True):
    if package_name is None:
        package_name = module_name

    class VersionError(Exception):
        pass

    msg = None
    try:
        package = __import__(module_name)
        if minimum is not None:
            try:
                v = package.version.short_version
            except AttributeError:
                v = package.__version__
            if StrictVersion(v) < StrictVersion(minimum):
                raise VersionError
    except ImportError:
        if minimum is None:
            msg = 'MSMBuilder requires the python package "%s", which is not installed.' % package_name
        else:
            msg = 'MSMBuilder requires the python package "%s", version %s or later.' % (package_name, minimum)
    except VersionError:    
        msg = ('MSMBuilder requires the python package "%s", version %s or '
               ' later. You have version %s installed. You will need to upgrade.') % (package_name, minimum, v)

    if recommend_conda:
        install = ('\nTo install %s, we recommend the conda package manger. See http://conda.pydata.org for info on conda.\n'
                   'Using conda, you can install it with::\n\n    $ conda install %s') % (package_name, package_name)
        install += '\n\nAlternatively, with pip you can install the package with:\n\n    $ pip install %s' % package_name
    else:
        install = '\nWith pip you can install the package with:\n\n    $ pip install %s' % package_name
    
    if msg:
        banner = ('==' * 40)
        print('\n'.join([banner, banner, "", msg, install, "", banner, banner]))


class mybuildscript(build_scripts):
    TEMPLATE = '''
from __future__ import print_function
import sys

print('The MSMBuilder script {basename}.py has been\\nrenamed {basename}.', file=sys.stderr, end=' ')
print('You can access it with:\\n\\n  $ {basename}\\n\\nor\\n\\n  $ msmb {basename}\\n', file=sys.stderr)
'''
    def copy_scripts(self):
        exclude = ['msmb']
        try:
            tdir = tempfile.mkdtemp()
            self.scripts = []
            for fn in find_console_scripts():
                basename = fn.split(':')[0].split('.')[-1]
                if basename not in exclude:
                    path = '{}/{}.py'.format(tdir, basename)
                    with open(path, 'w') as f:
                        f.write(self.TEMPLATE.format(basename=basename))
                    self.scripts.append(path)

            if PY3:
                super().copy_scripts()
            else:
                build_scripts.copy_scripts(self)
        finally:
            shutil.rmtree(tdir)


def find_console_scripts():
    console_scripts = []
    exclude = ['__init__.py']
    for fn in glob('scripts/*.py'):
        dirname, filename = os.path.split(fn)
        if filename not in exclude:
            basename, _ = os.path.splitext(filename)
            console_scripts.append(
                '{basename} = msmbuilder.scripts.{basename}:entry_point'.format(basename=basename))

    return console_scripts


# metadata for setup()
metadata = {
    'name': 'msmbuilder',
    'version': VERSION,
    'author': __author__,
    'author_email': 'msmbuilder-user@stanford.edu',
    'license': 'GPL v3.0',
    'url': 'https://simtk.org/home/msmbuilder',
    'download_url': 'https://simtk.org/home/msmbuilder',
    'platforms': ["Linux", "Mac OS X"],
    'description': DOCLINES[0],
    'long_description':"\n".join(DOCLINES[2:]),
    'packages': ['msmbuilder', 'msmbuilder.scripts', 'msmbuilder.project',
                 'msmbuilder.lumping', 'msmbuilder.metrics', 'msmbuilder.reduce',
                 'msmbuilder.reference'],
    'package_dir': {'msmbuilder': 'MSMBuilder', 'msmbuilder.scripts': 'scripts',
                    'msmbuilder.reference': 'reference'},
    'package_data': {'msmbuilder.reference': [os.path.relpath(os.path.join(a[0], b), 'reference') for a in os.walk('reference') for b in a[2]]},
    'zip_safe': False,
    'entry_points': {'console_scripts': find_console_scripts()},

    # CUSTOM HACK TO DYNAMICALLY CREATE STUB SCRIPTS TO DIRECT PEOPLE TO THE NEW ENTRY POINTS
    # (`Cluster.py` vs. `Cluster`)
    'cmdclass': {'build_scripts': mybuildscript},
    'scripts': [''],
}



# Return the git revision as a string
# copied from numpy setup.py
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def write_version_py(filename='MSMBuilder/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM MSMBUILDER SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version' : FULLVERSION,
                       'git_revision' : GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()

write_version_py()
setup(**metadata)

# running these after setup() ensures that they show
# at the bottom of the output, since setup() prints
# a lot to stdout. helps them not get lost
warn_on_version('numpy', '1.6.0')
warn_on_version('scipy', '0.11.0')
warn_on_version('tables', '2.4.0', package_name='pytables')
warn_on_version('fastcluster', '1.1.13')
warn_on_version('yaml', package_name='pyyaml')
warn_on_version('mdtraj', '0.8.0')
