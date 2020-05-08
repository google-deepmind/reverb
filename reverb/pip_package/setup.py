# python3
# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Installs dm-reverb."""
import fnmatch
import os
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.install import install as InstallCommandBase
import sys


if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)
else:
  raise ValueError('Please pass a project name with the --project_name flag.')


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True


def find_files(pattern, root):
  """Return all the files matching pattern below root dir."""
  for dirpath, _, files in os.walk(root):
    for filename in fnmatch.filter(files, pattern):
      yield os.path.join(dirpath, filename)


class InstallCommand(InstallCommandBase):
  """Override the dir where the headers go."""

  def finalize_options(self):
    ret = super().finalize_options()
    # We need to set this manually because we are not using setuptools to
    # compile the shared libraries we are distributing.
    self.install_lib = self.install_platlib
    return ret


setup(
    name=project_name,
    # TODO(b/155888926): Improve how we determine version(s).
    version='0.0.1',
    description=('Reverb is an efficient and easy-to-use data storage and '
                 'transport system designed for machine learning research.'),
    long_description='',
    long_description_content_type='text/markdown',
    author='DeepMind',
    author_email='DeepMind <no-reply@google.com>',
    url='',
    license='Apache 2.0',
    packages=find_packages(),
    headers=list(find_files('*.proto', 'reverb')),
    include_package_data=True,
    install_requires=['dm-tree', 'portpicker'],
    distclass=BinaryDistribution,
    cmdclass={
        'install': InstallCommand,
    },
    python_requires='>=3',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='tensorflow deepmind reinforcement learning machine replay jax',
)
