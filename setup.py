from setuptools import find_packages, setup

setup(
    name='higgs2bbclassifier',
    version='0.1.0',
    author='Umit Akdeniz',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'xgboost',
        'hydra-core',
        'mlflow'
    ],
)
