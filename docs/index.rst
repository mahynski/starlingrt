.. starlingrt documentation master file, created by
   sphinx-quickstart on Thu Nov 14 09:34:19 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

starlingrt documentation
========================

.. image:: https://github.com/mahynski/starlingrt/actions/workflows/python-app.yml/badge.svg?branch=main
   :target: https://github.com/mahynski/starlingrt/actions
.. image:: https://readthedocs.org/projects/starlingrt/badge/?version=latest
   :target: https://starlingrt.readthedocs.io/en/latest/?badge=latest
.. image:: https://codecov.io/gh/mahynski/starlingrt/graph/badge.svg?token=7EILPHJ40F 
   :target: https://codecov.io/gh/mahynski/starlingrt
.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
.. image:: https://zenodo.org/badge/331207062.svg
   :target: https://zenodo.org/badge/latestdoi/331207062

----

.. image:: ./_static/logo.png
   :height: 100
   :align: left

STARLINGrt is a tool for analyzing retention times from gas chromatogaphy mass spectrometry (GCMS).  It can be used to determine a consensus value for compounds by visualizing a collection of results.  Compound identification(s) made at a given retention time are assumed to be provided by a separate code which analyzes the mass spectrometry data collected at that time.  Currently, STARLINGrt is configured to work with the outputs from `MassHunter(TM) <https://www.agilent.com/en/product/software-informatics/mass-spectrometry-software>`_ but is extensible by subclassing "data._SampleBase" (see samples.py for an example).  The code produces an interactive HTML file using `Bokeh <https://bokeh.org/>`_ which can be modified interactively, saved, exported and shared easily between different users.  The name "starling" was selected as a reverse acronym of the tool's purpose.

Use Cases
###################

Imagine you have multiple GCMS output files which have been used to identify chemicals at different retention times, e.g., using some sort of library. 
In principle, these could correspond to analyses of a range of different mixtures; regardless, an individual component should elute at the same time regardless of what it is combined with. However, natural variations in:

* the retention times can cause confusion when other compounds coelute or elute at very similar times,
* the mass spectrometry peak location(s) at a given retention time can cause the identification routine to identify the same compound differently.

Given these uncertainties we would like to learn things like:

1. What is a consensus value, or at least a natural range, of retention times for each compound identified?
2. What compounds elute at similar points and are commonly confused with each other?
3. Are there any analyses that identify a compound at a retention time far away from its consensus value (data cleaning)?
4. What is a natural "gap" in retention times that can be used to "ideally" divide all compounds from their "neighbors"?

This visualization tool helps users answer these questions by exploring their data with interactive graphs. The output of this tool is an HTML file that acts as a self-contained summary of your data, how you cleaned / modified it, and an be easily shared between users.

Example
###################

Here is a simple example (see `docs/_static/example.py`):

.. code-block:: python

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


Contributors
###################
This code was developed during a collaboration with:

* `Prof. Nives Ogrinc <https://orcid.org/0000-0002-0773-0095`_
* `Dr. Lidija Strojnik <https://orcid.org/0000-0003-1898-9147>`_

License Information
###################
* See `LICENSE.md <https://github.com/mahynski/starlingrt/blob/main/LICENSE.md>`_ for more information.

* Any mention of commercial products is for information only; it does not imply recommendation or endorsement by `NIST <https://www.nist.gov/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   install
   examples

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`