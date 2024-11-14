![Workflow](https://github.com/mahynski/starlingrt/actions/workflows/python-app.yml/badge.svg?branch=main)
[![Documentation Status](https://readthedocs.org/projects/starlingrt/badge/?version=latest)](https://starlingrt.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/mahynski/starlingrt/branch/main/graph/badge.svg?token=YSLBQ33C7F)](https://codecov.io/gh/mahynski/starlingrt)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![DOI](https://zenodo.org/badge/331207062.svg)](https://zenodo.org/badge/latestdoi/331207062)
<!--[![DOI](https://zenodo.org/badge/{github_id}.svg)](https://zenodo.org/badge/latestdoi/{github_id})-->

<!--
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
-->

STARLINGrt : [I]nteractive [R]etention [T]ime vi[S]ualization for gas chromatography
===

<img src="docs/_static/logo.png" height="100" align="left" />

STARLINGrt is a tool for analyzing retention times from gas chromatogaphy.  It can be used to compare times for the same substance to determine a consensus value by visualizing results from a library.  At the moment it is configured to work with the outputs from [MassHunter(TM)](https://www.agilent.com/en/product/software-informatics/mass-spectrometry-software) but is extensible by subclassing "data._SampleBase" (see samples.py for an example).  The code produces an interactive HTML file using [Bokeh](https://bokeh.org/) which can be modified interactively, saved, exported and shared easily between different users.  The name "starling" was selected as a reverse acronym of the tool's purpose.

Installation
===

We recommend creating a [virtual environment](https://docs.python.org/3/library/venv.html) or, e.g., a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) then installing startlingrt with [pip](https://pip.pypa.io/en/stable/):

~~~bash
$ conda create -n starlingrt-env python=3.10
$ conda activate starlingrt-env
$ pip install startlingrt
~~~

You can also install from this GitHub repo source:

~~~bash
$ git clone git@github.com:mahynski/startlingrt.git
$ cd startlingrt
$ conda create -n starlingrt-env python=3.10
$ conda activate starlingrt-env
$ pip install .
$ python -m pytest # Optional unittests
~~~

To install this into a Jupyter kernel:

~~~bash
$ conda activate starlingrt-env
$ python -m ipykernel install --user --name starlingrt-kernel --display-name "starlingrt-kernel"
~~~

Example
===

Here is a simple example (see `docs/_static/example.py`):

~~~python
import os
import starlingrt

from starlingrt import sample, data, functions, visualize

def load_mass_hunter(input_directory):
    """
    Parameters   
    ---------
    input_directory : str
        Directory to seach for raw folders are in.

    Returns
    -------
    samples : list(sample.MassHunterSample)
        List of Samples collected from all directories in `input_directory`.
    """
    ...
    return samples

top_entries = starlingrt.data.Utilities.select_top_entries(
    starlingrt.data.Utilities.create_entries(
        load_mass_hunter(
            "path/to/data/"
        )
    )
)

starlingrt.visualize.make(
    top_entries=top_entries, 
    width=1200,
    threshold=starlingrt.functions.estimate_threshold(starlingrt.functions.get_dataframe(top_entries)[0]),
    output_filename='summary.html',
)
~~~

Documentation
===

Documentation is hosted at [https://startlingrt.readthedocs.io/](https://startlingrt.readthedocs.io/) via [readthedocs](https://about.readthedocs.com/).

The logo was generated using Google Gemini with the prompt "Design a logo involving a starling and gas chromatography" on Nov. 9, 2024.

Contributors
===

This code was developed during a collaboration with:

* [Prof. Nives Ogrinc](https://orcid.org/0000-0002-0773-0095)
* [Dr. Lidija Strojnik](https://orcid.org/0000-0003-1898-9147)
