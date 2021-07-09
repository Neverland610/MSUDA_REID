from setuptools import setup, find_packages


setup(name='MMT',
      version='1.1.0',
      install_requires=[
          'numpy', 'torch==1.3.1', 'torchvision==0.2.2',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Domain Adaptation',
          'Person Re-identification',
          'Deep Learning',
      ])
