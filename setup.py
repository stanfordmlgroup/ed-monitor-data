import setuptools
setuptools.setup(
    name='ED Monitor',
    version='0.1',
    description='Common files for the ED Monitor project',
    url='https://github.com/stanfordmlgroup/ed-monitor-data',
    author='AIHC',
    install_requires=['torch', 'pytorch-lightning', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'tqdm'],
    author_email='',
    packages=setuptools.find_packages(),
    zip_safe=False
)
