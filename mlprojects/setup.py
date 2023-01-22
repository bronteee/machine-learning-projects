from setuptools import setup

setup(
    name='mlprojects',
    version='0.1',
    description='A collection of machine learning projects',
    author='Bronte Sihan Li',
    author_email='li.siha@northeastern.edu',
    packages=['project1', 'utility'],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
    ],
)
