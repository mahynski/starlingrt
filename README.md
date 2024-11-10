![Workflow](https://github.com/mahynski/my_package/actions/workflows/python-app.yml/badge.svg?branch=main)
[![Documentation Status](https://readthedocs.org/projects/my_package/badge/?version=latest)](https://my_package.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/mahynski/my_package/branch/main/graph/badge.svg?token=YSLBQ33C7F)](https://codecov.io/gh/mahynski/my_package)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![DOI](https://zenodo.org/badge/331207062.svg)](https://zenodo.org/badge/latestdoi/331207062)
<!--[![DOI](https://zenodo.org/badge/{github_id}.svg)](https://zenodo.org/badge/latestdoi/{github_id})-->

<!--
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
-->

STARLINGrt : [I]nteractive [R]etention [T]ime vi[S]ualization for gas chromatography
===

<img src="docs/_static/logo_small.png" height="100" align="left" />

STARLINGrt is a tool for analyzing retention times from gas chromatogaphy.  It can be used to compare times for the same substance to determine a consensus value by visualizing results from a library.  At the moment it is configured to work with the outputs from [MassHunter](https://www.agilent.com/en/product/software-informatics/mass-spectrometry-software).  The code produces an interactive HTML file using [Bokeh](https://bokeh.org/) which can be modified interactively, saved, exported and shared easily between different users.  The name "starling" was selected as a reverse acronym of the tool's purpose. 

<br/>
<br/>

Installation
===

We recommend creating a [virtual environment](https://docs.python.org/3/library/venv.html) or, e.g., a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) then installing startlingrt with [pip](https://pip.pypa.io/en/stable/):

~~~bash
$ pip install startlingrt
~~~

You can also install from this GitHub repo source:

~~~bash
$ git clone git@github.com:mahynski/startlingrt.git
$ cd startlingrt
$ pip install .
$ python -m pytest # Optional unittests
~~~

Documentation
===

Documentation is hosted at [https://startlingrt.readthedocs.io/](https://startlingrt.readthedocs.io/) via [readthedocs](https://about.readthedocs.com/).

