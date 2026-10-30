from setuptools import find_packages, setup

from typing import List


def get_requirements(file_path: str) -> List[str]:
    with open(file_path, encoding='utf-8') as file:
        requirements = file.read().splitlines()

    requirements = [req.strip() for req in requirements
                    if req.strip() and not req.strip().startswith('#')]
    return requirements


setup(
    name='nasa_cmapss_rul',
    version='0.1.0',
    description='NASA C-MAPSS remaining useful life prediction project',
    author='Mahim',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
