Install
========

We recommend creating a `virtual environment <https://docs.python.org/3/library/venv.html>`_ or, e.g., a `conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ then installing startlingrt with `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block:: bash

    conda create -n starlingrt-env python=3.10
    conda activate starlingrt-env
    pip install startlingrt


You can also install from this GitHub repo source:

.. code-block:: bash

    git clone git@github.com:mahynski/startlingrt.git
    cd startlingrt
    conda create -n starlingrt-env python=3.10
    conda activate starlingrt-env
    pip install .
    python -m pytest # Optional unittests


To install this into a Jupyter kernel:

.. code-block:: bash

    conda activate starlingrt-env
    python -m ipykernel install --user --name starlingrt-kernel --display-name "starlingrt-kernel"

