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

PyPi Package Template
===

1. Choose a name that does not exist in [pypi](https://pypi.org/). You can check by going to https://pypi.org/simple/{my_awesome_new_package}; if you recieve a 404 error the name has not been taken.
2. Replace all "my_package" names, strings, etc. in this repo with your desired package name (e.g., "my_awesome_new_package").  The command below makes this simple; run this after cloning the repo locally.

~~~bash
$ for file in $(find . -type f -not -path "./.git/*"); do sed -i "s/my_package/my_awesome_new_package/g" $file; done
~~~

2. Get coding!

Documentation
===

Documentation is stored in the `docs/` folder and is currently set up to use [sphinx](https://www.sphinx-doc.org/en/master/).

First build the `requirements.txt` needed to build the documentation.

~~~bash
$ cd docs
$ pip install Sphinx
$ pip install pip-tools
$ pip-compile requirements.in
~~~

Adjust the `docs/conf.py` as desired. Then run `docs/make_docs.sh` to setup the documentation initially.  You can manually add and adjust later.

~~~bash
$ bash make_docs.sh
~~~

Go to [https://about.readthedocs.com/](https://about.readthedocs.com/) to link your repo to build and host the documentation automatically!  The `.readthedocs.yml` file contains the configuration for this which you can adjust as needed.

Unittests
===

Build [unittests](https://docs.python.org/3/library/unittest.html) in the `tests/` directory.  The `pyproject.toml` automatically configures pytest to look in `tests/`.  The following will run all unittests in this directory.

~~~bash
$ python -m pytest
~~~

The GitHub workflow in `.github/workflows/python-app.yml` will also run these tests and perform coverage checks using this command.  This workflow is triggered automatically on the main branch, but you can adjust this file so this is automatically triggered on others as well.

Linting
===

Automatic code linting is provided via [pre-commit](https://pre-commit.com/); refer to the `.pre-commit-config.yaml` file for the specific configuration which you can adjust as needed.

Run pre-commit to lint new code, then commit the changes.

~~~bash
$ pre-commit run --all-files
~~~

Citation
===

Update the CITATION.cff, CODEOWNERS, and pyproject.toml files to include all code authors and maintainers appropriately.

Best Practices
===

Other best practices include [typing](https://docs.python.org/3/library/typing.html) - see [mypy](https://mypy-lang.org/).

~~~bash
$ mypy --ignore-missing-imports my_new_file.py
~~~

You can generate a logo or other art using [gemini](https://gemini.google.com/app) or other AI tools. [Note](https://lib.guides.umd.edu/c.php?g=1340355&p=9896961#:~:text=The%20Chicago%20Manual%20of%20Style's,prompt%20that%20generated%20the%20image.) that "the Chicago Manual of Style's website recommends you cite AI-generated images like any other image, while including both the name of the AI tool that generated the image, the company that created the AI, and the prompt that generated the image."
