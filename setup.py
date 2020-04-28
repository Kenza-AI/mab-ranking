from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='mab-ranking',
    version='0.0.1',
    setup_cfg=True,
    python_requires='~=3.5',
    packages=find_packages(where='.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['setuptools>=39.1.0'],
    url='https://github.com/Kenza-AI/mab-ranking',
    install_requires=[
        'boto3',
        'paramiko>=2.4.2, <2.4.99',
        'pathlib2>=2.3.0, <2.3.99',
        'requests>=2.20.0, <2.20.99',
        'sagemaker>=1.50.0, <1.50.99',
        'six>=1.10, <1.11.99',
        'future>=0.16.0, <0.17.99'
    ],
    test_suite='tests',
    zip_safe=True
)
