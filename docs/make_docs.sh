#!/bin/bash
sphinx-apidoc -o ./ ../starlingrt/;
make clean html;
make html;

# Run these commands the first time to set up
# pip install pip-tools
# pip-compile requirements.in
