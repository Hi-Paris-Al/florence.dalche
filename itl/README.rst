.. -*- mode: rst -*-

.. |Codecov| image:: https://codecov.io/bb/romainbrault/itl/branch/master/graph/badge.svg?token=1bSX6qqV6U
.. _Codecov: https://codecov.io/bb/romainbrault/itl

.. |CircleCI| image:: https://circleci.com/bb/RomainBrault/itl.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/bb/RomainBrault/itl


.. |Python36| image:: https://img.shields.io/badge/Python-3.x-blue.svg
.. _Python36: https://www.python.org/downloads/release/python-364/


Infinite-Task Learning
======================

|Codecov|_ |CircleCI|_ |Python36|_

With applicatons to Infinite Quantile Regression, Infinite Level Set
Estimation, and Infinite Joint Cost Sensitive Learning.

Links
-----

- `Bitbucket git <https://bitbucket.org/RomainBrault/itl>`_
- `ARXIV paper <https://arxiv.org/pdf/1805.08809.pdf>`_

Dependencies
------------

- TexLive (full) 2016 or later
- Git and Sconstruct
- python and pip 3
- all the python packages listed in the file ``requirements.txt``

Usage
-----

To copy the repository type in a terminal ::

    git clone git@bitbucket.org:RomainBrault/itl.git

The project is composed of a folder 'experiments' which contains the numerical
experiments of the paper and the results. The folder 'doc' contains different
version of the paper and various drafts. To install the library run ::

    pip install .


Contribute
----------

Code
~~~~

To generate the requirements.txt file automatically use pigar ::

    pigar -P itl -p ./requirements.txt

Paper
~~~~~

To compile the paper with XeLaTeX and biber, go into the paper folder ::

    cd doc/NIPS_2018

and run the Sconstruct file ::

    scons

The resulting pdf should be in the subfolder build, and the temporary files in
the subfolder build.

Experiments
~~~~~~~~~~~

The experiments of the paper can be reproduced using the library in the folder
``demos/NIPS_2018/``. Simply run ::

    pip install .

to install the python library and run one of the files in the subfolder
``experiments/itl/demos/``. E.g. ::

    python demos/NIPS_2018/icsl_vs.py --show --save_graph=./tflog

When compiling the paper for the first time, the figure are generated
automatically by running python on the files present in the folder
``demos/NIPS_2018/``. The results are stored in the folder
``doc/NIPS_2018/build/src/fig/``. If the pdf or eps files are already presents,
the figures are not generated, and latex will use the one provided in the
aformentioned folder. For the sake of fast installation we provide a cache
version of our build folder.

the computation graph can be visualized using ::

    tensorboard --localdir ./tflog

and opening a browser (usually) at ``http://localhost:6006``


Contact and authors
-------------------

- Florence d'Alche-Buc, Telecom ParisTech, LTCI
- Romain Brault: mail@romainbrault.com (corresponding author),
  Centrale-Supelec, L2S
- Alex Lambert: alex.lambert@telecom-paristech.fr (corresponding author),
  Telecom ParisTech, LTCI
- Maxime Sangnier, Universite Pierre et Marie Curie
- Zoltan Szabo, Ecole Polytechnique, CMAP
