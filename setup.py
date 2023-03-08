from os import path
from setuptools import setup

# get current directory
here = path.abspath(path.dirname(__file__))

# get the long description from the README file
#with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    #long_description = f.read()

# # read the API version from disk
# with open(path.join(here, 'vantage6', 'tools', 'VERSION')) as fp:
#     __version__ = fp.read()

# setup the package
setup(
    name='v6_LinReg_py',
    version="1.0.0",
    description='federated Linear Regression',
    #long_description=long_description,
    #long_description_content_type='text/markdown',
    #url='https://github.com/IKNL/v6_boilerplate-py',
    packages=['v6_LinReg_py'],
    python_requires='>=3.6',
    install_requires=[
        'scikit-learn',
        'numpy',
    ]

)
