"""
Structures to store samples from different mass spectrometers.
Author: Nathan A. Mahynski

"""

import xlrd

import numpy as np

from starlingrt import data

from typing import Any

"""
Subclass data.SampleBase with as needed for other mass spectrometers.
"""

class MassHunterSample(data._SampleBase):
	"""Class to store the MSRep.xls output from MassHunter(TM)."""
	def read(self, filename: str) -> None:
		"""
		Read data from MSRep.xls file.

		This assumes a specific formatted output from MassHunter(TM) which is checked below.

		Parameters
		----------
		filename : str
			Pathname of MSRep.xls file.
		"""
		self._filename = filename # type: ignore [misc]

		wb = xlrd.open_workbook(self._filename)
		intres = wb.sheet_by_name('IntRes')
		libres = wb.sheet_by_name('LibRes')

		# Record metadata at the top of the IntRes tab
		self._metadata = [intres.cell(i, 0).value for i in range(4)]

		# Check that the columns are as expected
		column_names = ["Compound number (#)",
			"RT (min)",
			"Scan number (#)",
			"Area (Ab*s)",
			"Baseline Heigth (Ab)",
			"Absolute Heigth (Ab)",
			"Peak Width 50% (min)",
			"Start Time (min)",
			"End Time (min)",
			"Start Height (Ab)",
			"End Height (Ab)",
			"Peak Type"
		]
		assert(intres.row_values(5) == column_names), f'Column names in the IntRes tab of {self._filename} are not as expected.'

		# Read all compounds
		self._compounds = [] # type: ignore [misc]
		for row_idx in range(6, intres.nrows):		
			self._compounds.append(
				data.Compound(		
					number = intres.cell(row_idx, column_names.index('Compound number (#)')).value,		
					rt = intres.cell(row_idx, column_names.index('RT (min)')).value,		
					scan_number = intres.cell(row_idx, column_names.index('Scan number (#)')).value,
					area = intres.cell(row_idx, column_names.index('Area (Ab*s)')).value,
					baseline_height = intres.cell(row_idx, column_names.index('Baseline Heigth (Ab)')).value,
					absolute_height = intres.cell(row_idx, column_names.index('Absolute Heigth (Ab)')).value,
					peak_width = intres.cell(row_idx, column_names.index('Peak Width 50% (min)')).value,
				)
			)  

		# Read all the hits for each compound
	
		# Check metadata at the top of the LibRes tab is the same
		check_meta = [libres.cell(i, 0).value for i in range(4)]
		assert(check_meta == self._metadata), f'Metadata from the LibRes tab disagrees with IntRes in {self._filename}'

		# Check that the columns are as expected
		column_names = ["Compound number (#)",
			"RT (min)",
			"Scan number (#)",
			"Area (Ab*s)",
			"Baseline Heigth (Ab)",
			"Absolute Heigth (Ab)",
			"Peak Width 50% (min)",
			"Hit Number",
			"Hit Name",
			"Quality",
			"Mol Weight (amu)",
			"CAS Number",
			"Library",
			"Entry Number Library"
		]
		assert(libres.row_values(8) == column_names), f'Column names in the LibRes tab of {self._filename} are not as expected.'

		# Hits for each compounds
		self._hits = {} # type: ignore [misc]
		for row_idx in range(9, libres.nrows):
			cpd = libres.cell(row_idx, column_names.index('Compound number (#)')).value
			if cpd:
				cpd_no = int(cpd)
				self._hits[cpd_no] = []
			
			self._hits[cpd_no].append(
				data.Hit(		
					number = libres.cell(row_idx, column_names.index('Hit Number')).value,		
					name = libres.cell(row_idx, column_names.index('Hit Name')).value,		
					quality = libres.cell(row_idx, column_names.index('Quality')).value,
					mol_weight = libres.cell(row_idx, column_names.index('Mol Weight (amu)')).value,
					cas_number = libres.cell(row_idx, column_names.index('CAS Number')).value,
					library = libres.cell(row_idx, column_names.index('Library')).value,
					entry_number_library = libres.cell(row_idx, column_names.index('Entry Number Library')).value,
				)
			)

		# Check that all compounds from IntRes are in LibRes
		assert(np.asarray(sorted(self._hits.keys()) == np.arange(1, len(self._compounds)+1)).all()), f'Hits are either not ordered correctly or are missing for certain compounds in {self._filename}' 
