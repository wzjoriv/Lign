from setuptools import setup, find_packages

long_description = ''
with open("README.md", "r") as f:
    long_description = f.read()

install_requires = []
with open("requirements.txt", "r") as f:
    install_requires = [i for i in f.read().replace(' ', '').splitlines() if i[0] != '#']

setup(name='lign',
		version='0.1.0',
		description='A framework for working with graphs alongside PyTorch',
		long_description=long_description,
		long_description_content_type="text/markdown",
		license="Mozilla Public License Version 2.0",
		author='Josue N Rivera',
		author_email='josue.n.rivera@outlook.com',
		url='https://github.com/JosueCom/Lign',
		packages=find_packages(),
		package_data = {
			"lign.utils": ["defaults/*.lign"]
		},
		install_requires=install_requires,
		python_requires='>=3.7.0'
     )