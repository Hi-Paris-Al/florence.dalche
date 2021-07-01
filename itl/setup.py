from itl import __version__, __authors__
from setuptools import setup


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not
            line.startswith("#")]


install_reqs = parse_requirements('./requirements.txt')
reqs = [str(ir) for ir in install_reqs]

setup(name='itl',
      version=__version__,
      description='Infinite Tasks Learning',
      author=__authors__,
      author_email='mail@romainbrault.com',
      license='MIT',
      packages=['itl', 'itl.nqn', 'itl.cost', 'itl.penalty', 'itl.datasets'],
      include_package_data=True,
      install_requires=reqs,
      zip_safe=False)
