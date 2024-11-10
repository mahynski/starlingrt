"""
Unittests template example.
author: Nathan A. Mahynski

"""
import unittest

from my_package import * # Bad form, but just a placeholder example

class DummyTest(unittest.TestCase):
    """Perform dummy tests."""
  
    @classmethod  
    def setUpClass(self):
        """Set up things for all members of this test class."""
        return

    def test_dummy(self):
        """Perform a dummy test."""
        return
