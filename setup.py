from setuptools import find_packages, setup

setup(
    name='MultiMSM',
    packages=find_packages(include=["MultiMSM","MultiMSM.MSM","MultiMSM.util"]),
    version='0.1.0',
    description='Create multiple MSMs over a discretized parameter',
    author='onehalfatsquared',
    license='MIT',
    test_suite='testing',
    install_requires=['numpy','scipy','scikit-learn']
)
