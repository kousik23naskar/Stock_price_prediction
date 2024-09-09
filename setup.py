from setuptools import find_packages, setup
from typing import List

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]

    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n',"") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

__version__ = "0.0.0"
GIT_REPO_NAME = "Stock_price_prediction"
GIT_REPO_USER_NAME = "kousik23naskar"
AUTHOR_USER_NAME = "Kousik Naskar"
SRC_REPO = "StockPricePrediction"
AUTHOR_EMAIL = "kousik23naskar@gmail.com"


setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A python package for stock price prediction app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{GIT_REPO_USER_NAME}/{GIT_REPO_NAME}",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
