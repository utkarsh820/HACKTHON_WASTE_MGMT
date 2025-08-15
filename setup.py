from setuptools import setup, find_packages
from typing import List

def get_requirements(file:str) -> List[str]:
    '''
    Get the list of requirements from a requirements file.
    
    '''
    with open(file, 'r') as file:
        requirements = [line.strip() for line in file if line.strip() and not line.startswith('#')]
        return requirements
    
if __name__ == "__main__":
    setup (
        name='Hackathon_Waste_Management',
        version='0.1',
        author='Utkarsh Pandey',
        author_email='Xixama@proton.me',
        description='A mini-hackthon project : Waste Management',
        packages=find_packages(),
        install_requires=get_requirements('requirements.txt'),
        python_requires='>=3.10,<3.13',
    )

