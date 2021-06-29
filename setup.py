try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

long_description = ''
with open("README.md", "r") as f:
    long_description = f.read()

install_requires = []
with open("requirements.txt", "r") as f:
    install_requires = [i for i in f.read().replace(' ', '').splitlines() if i[0] != '#']

setup(name='lign',
      version='0.1.0',
      description='A deep learning framework for implementing lifelong learning on vector and graph data',
      long_description=long_description,
      license="Mozilla Public License Version 2.0",
      author='Josue N Rivera',
      author_email='josue.n.rivera@outlook.com',
      url='https://github.com/JosueCom/LIGN',
      packages=['lign'],
      install_requires=install_requires,
     )