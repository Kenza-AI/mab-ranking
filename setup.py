from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().split()

setup(
    name='mab-ranking',
    version='0.0.1',
    setup_cfg=True,
    python_requires='>=3.5',
    packages=find_packages(where='.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['setuptools>=39.1.0'],
    url='https://github.com/Kenza-AI/mab-ranking',
    install_requires=requirements,
    test_suite='tests',
    zip_safe=True
)
