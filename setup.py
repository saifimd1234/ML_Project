"""
setup.py is a script for configuring the installation of a Python project. In the context of a machine learning (ML) project, setup.py serves several important purposes:

1. **Dependency Management**: Lists all the dependencies required for the project, ensuring that anyone who installs the project gets all the necessary libraries and packages.
2. **Package Distribution**: Facilitates the distribution of the project as a package that can be easily installed using package managers like pip.
3. **Metadata**: Provides metadata about the project such as the name, version, author, and description, which is useful for documentation and distribution.
4. **Entry Points**: Defines entry points for the project, such as command-line scripts, making it easier to run the project or its components.
5. **Versioning**: Helps in maintaining and managing different versions of the project, which is crucial for reproducibility in ML experiments.
6. **Custom Commands**: Allows the definition of custom commands that can be run during the installation process, which can be useful for setting up the environment or compiling extensions.

Overall, setup.py is essential for the smooth installation, distribution, and management of dependencies in an ML project.
"""

from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='ml_project',
    version='0.1',
    author='Mohammad Saifi',
    author_email='saifimd1234@gmail.com',       
    description='A machine learning project template',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'), # Read dependencies from requirements.txt
)