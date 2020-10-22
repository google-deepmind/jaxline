# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Setup for pip package."""

import os
import setuptools


here = os.path.dirname(os.path.abspath(__file__))


def _get_version():
  """Returns the JAXline version."""
  with open(os.path.join(here, 'jaxline', '__init__.py')) as f:
    try:
      version_line = next(
          line for line in f if line.startswith('__version__'))
    except StopIteration:
      raise ValueError('__version__ not defined in jaxline/__init__.py')
    else:
      ns = {}
      exec(version_line, ns)  # pylint: disable=exec-used
      return ns['__version__']


def _parse_requirements(path):
  with open(os.path.join(here, path)) as f:
    return [
        line.rstrip() for line in f
        if not (line.isspace() or line.startswith('#'))
    ]


EXTRA_PACKAGES = {
    'jax': ['jax>=0.1.71'],
    'jaxlib': ['jaxlib>=0.1.49'],
    'tensorflow': ['tensorflow>=2'],
    'tensorflow with gpu': ['tensorflow-gpu>=2'],
}


setuptools.setup(
    name='jaxline',
    version=_get_version(),
    url='https://github.com/deepmind/jaxline',
    description='JAXline is a distributed JAX training framework.',
    license='Apache 2.0',
    author='DeepMind',
    author_email='jaxline-copybara@google.com',
    long_description=open(os.path.join(here, 'README.md')).read(),
    long_description_content_type='text/markdown',
    # Contained modules and scripts.
    packages=setuptools.find_namespace_packages(exclude=['*_test.py']),
    install_requires=_parse_requirements('requirements.txt'),
    extras_require=EXTRA_PACKAGES,
    requires_python='>=3.6',
    include_package_data=True,
    zip_safe=False,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
)
