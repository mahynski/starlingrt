"""
Tools for reading MSRep.xls files from disk produced by MassHunter(TM).

Author: Nathan A. Mahynski
"""

import xlrd
import copy
import os
import hashlib
import numpy as np

def load(input_directory):
	"""
	Load data from the input/ directory.

	The data should be laid out in a folder such that:

	input_directory/
		ORG123_YYYY_MM_DD.D/
			MSRep.xls
		ORG456_YYYY_MM_DD.D/
			MSRep.xls
		ORG789_YYYY_MM_DD.D/
			MSRep.xls
		...

	Parameters
	----------
	input_directory : str
		Directory to seach for raw folders are in.

	Returns
	-------
	samples : list(Sample)
		List of Samples collected from all directories in `input_directory`.
	"""
	samples = []
	folders = sorted([f for f in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, f))])
	for f in folders:
		fname = os.path.join(input_directory, os.path.join(f, "MSRep.xls"))
		if os.path.isfile(fname):
			try:
				samples.append(Sample(fname))
			except Exception as e:
				raise Exception(f'Unable to read {fname} : {e}')
		else:
			raise Exception(f'Could not locate MSRep.xls file in {fname}')

	return samples

def create_entries(samples):
	"""
	Extract all entries from samples.
	
	Parameters
	----------
	samples : list(Sample)
		List of Samples collected from all directories in `input_directory`; see load().

	Returns
	-------
	total_entries : dict(hash:Entry)
		Dictionary of all Entry in `samples` whose keys are sha1 hashes and values are Entry objects.
	"""
	total_entries = {}
	checksum = 0
	for sample in samples:
		for entry in sample.entries():
			checksum += 1
			descr_ = '_'.join(['_'.join([a, str(b)]) for a,b in sorted(list(entry.get_params().items()))])
			hash_ = hashlib.sha1(descr_.encode('utf-8'))
			total_entries[hash_.hexdigest()] = entry

	assert(len(total_entries) == checksum), 'Error : hash conflicts found'
	return total_entries

def select_top_entries(total_entries):
    """
	Trim down the entries to just have the top (quality) hits (i.e., `hit_number` == 1).
	
	Parameters
	----------
	total_entries : dict(hash:Entry)
		Dictionary of all Entry in `samples` whose keys are sha1 hashes.

	Returns
	-------
	top_entries : dict(hash:Entry)
		Dictionary of all Entry with `hit_number` == 1 whose keys are sha1 hashes and values are Entry objects.
	"""
    top_entries = {}
    for k,v in total_entries.items():
        if v.hit_number == 1:
            top_entries[k] = v
    
    return top_entries

def group_entries_by_name(entries):
	"""
	Group entries with the same hit name.
	
	Parameters
	----------
	entries : dict(hash:Entry)
		Dictionary of Entry whose keys are sha1 hashes and values are Entry objects.

	Returns
	-------
	groups : dict(name:(Entry, hash))
		Dictionary of Entry whose keys are hit names and values are tuples of (Entry objects, hash).
	"""
	groups = {}
	for hash,entry in entries.items():
		if entry.hit_name in groups:
			groups[entry.hit_name].append((entry, hash))
		else:
			groups[entry.hit_name] = [(entry, hash)]

	return groups

def group_entries_by_rt(entries):
	"""
	Group entries with the same retention time.
	
	Parameters
	----------
	entries : dict(hash:Entry)
		Dictionary of Entry whose keys are sha1 hashes and values are Entry objects.

	Returns
	-------
	groups : dict(name:Entry)
		Dictionary of Entry whose keys are retention times and values are Entry objects.
	"""
	groups = {}
	for entry in entries.values():
		if entry.rt in groups:
			groups[entry.rt].append(entry)
		else:
			groups[entry.rt] = [entry]

	return groups

class Entry:
	def __init__(self, sample_filename, compound_number, rt, scan_number, area, baseline_height, absolute_height, peak_width, hit_number, hit_name, quality, mol_weight, cas_number, library, entry_number_library):
		"""
		Create an Entry.

		This is essentially a combination of Hit and Sample intended to "unroll" their information into
		a flat data structure more amenable for searching.
		"""
		self.set_params(
			**{
				"sample_filename": sample_filename, 
				"compound_number": compound_number,
				"rt": rt,
				"scan_number": scan_number,
				"area": area,
				"baseline_height": baseline_height,
				"absolute_height": absolute_height,
				"peak_width": peak_width,
				"hit_number": hit_number, 
				"hit_name": hit_name, 
				"quality": quality, 
				"mol_weight": mol_weight, 
				"cas_number": cas_number,
				"library": library,
				"entry_number_library": entry_number_library
			}
		)

	def set_params(self, **parameters):
		"""
		Set parameters.
		"""		
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	def get_params(self, deep=True):		
		"""
		Get parameters.
		"""		
		return {		
			"sample_filename": self.sample_filename, 
			"compound_number": self.compound_number,
			"rt": self.rt,
			"scan_number": self.scan_number,
			"area": self.area,
			"baseline_height": self.baseline_height,
			"absolute_height": self.absolute_height,
			"peak_width": self.peak_width,
			"hit_number": self.hit_number, 
			"hit_name": self.hit_name, 
			"quality": self.quality, 
			"mol_weight": self.mol_weight, 
			"cas_number": self.cas_number,
			"library": self.library,
			"entry_number_library": self.entry_number_library
		}

	def __repr__(self):
		return "<Entry at 0x{:x}>".format(id(self))

class Hit:
	"""
	A possible assignment to a peak from the library in use.
	"""
	def __init__(self, number: int, name: str, quality: int, mol_weight: float, cas_number: str, library: str, entry_number_library: int):
		"""
		Initialize the Hit.

		Names are consistent with MSRep.xls file (cf. LibRes tab).
		"""
		self.set_params(
			**{
				"number": int(number), 
				"name": str(name),
				"quality": int(quality),
				"mol_weight": float(mol_weight),
				"cas_number": str(cas_number),
				"library": str(library),
				"entry_number_library": int(entry_number_library)
			}
		)
	
	def set_params(self, **parameters):
		"""
		Set parameters.
		"""		
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	def get_params(self, deep=True):		
		"""
		Get parameters.
		"""		
		return {		
			"number": self.number, 
			"name": self.name,
			"quality": self.quality,
			"mol_weight": self.mol_weight,
			"cas_number": self.cas_number,
			"library": self.library,
			"entry_number_library": self.entry_number_library
		}

class Compound:
	"""
	A compound is a peak in the GCMS output that has been detected and must be assigned to one or more library Hits.
	"""
	def __init__(self, number: int, rt: float, scan_number: int, area: int, baseline_height: int, absolute_height: int, peak_width: float):
		"""
		Initialize the Compound.

		Names are consistent with MSRep.xls file (cf. LibRes and IntRes tabs).
		"""
		self.set_params(
			**{
				"number": int(number), 
				"rt": float(rt),
				"scan_number": int(scan_number),
				"area": int(area),
				"baseline_height": int(baseline_height),
				"absolute_height": int(absolute_height),
				"peak_width": float(peak_width)
			}
		)

	def set_params(self, **parameters):
		"""
		Set parameters.
		"""
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	def get_params(self, deep=True):		
		"""
		Get parameters.
		"""		
		return {		
			"number": self.number, 
			"rt": self.rt,
			"scan_number": self.scan_number,
			"area": self.area,
			"baseline_height": self.baseline_height,
			"absolute_height": self.absolute_height,
			"peak_width": self.peak_width
		}

class Sample:
	"""
	Structure to store the MSRep.xls output from MassHunter(TM).
	"""
	def __init__(self, filename):
		"""
		Parameters
		----------
		filename : str
			Path to MSRep.xls file to read.
		"""
		try:
			self.read(filename)
		except Exception as e:
			raise IOError(f"Unable to read from {filename} : {e}")

	@property
	def filename(self):
		return copy.copy(self._filename)

	@property
	def compounds(self):
		return copy.copy(self._compounds)

	@property
	def hits(self):
		return copy.copy(self._hits)

	def entries(self):
		"""
		Extract all Entry from Samples.
		
		Returns
		----------
		all_entries : list(Entry)
			List of all Entry created from known Samples and their Hits.
		"""
		all_entries = []
		for compound in self._compounds:
			for hit in self.sorted_hits(compound.number):
				all_entries.append(
					Entry(
						sample_filename="/".join(self._filename.split('/')[-2:]), # Only use 1 level in directory
						compound_number=compound.number,
						rt=compound.rt,
						scan_number=compound.scan_number,
						area=compound.area,
						baseline_height=compound.baseline_height,
						absolute_height=compound.absolute_height,
						peak_width=compound.peak_width,
						hit_number=hit.number, 
						hit_name=hit.name,
						quality=hit.quality,
						mol_weight=hit.mol_weight,
						cas_number=hit.cas_number,
						library=hit.library,
						entry_number_library=hit.entry_number_library
					)
				)

		return all_entries

	def sorted_hits(self, compound_number):
		"""
		Hits should be sorted by quality, but this makes sure. 
		A secondary sort is done by hit number to be consistent with MassHunter(TM)'s ordering.

		Parameters
		----------
		compound_number : int
			Compound number (starting from 1) in the IntRes tab from the MSRep.xls file.

		Returns
		-------
		hits : dict(int:Hit)
			Hits sorted first by quality and then by the number MassHunter(TM) assigned when it
			performed this sort.  This *should* not change the default ordering in the MSRep.xls
			file.

		Example
		-------
		>>> s = Sample(...)
		>>> sorted_hits = s.sorted_hits(compound_number=42)
		"""
		return sorted(self._hits[compound_number], key=lambda x: (x.get_params().get('quality'), -x.get_params().get('number')), reverse=True)

	def read(self, filename):
		"""
		Read data from MSRep.xls file.

		This assumes a specific formatted output from MassHunter(TM) which is checked below.
		In principal, this routine can be version controlled to adapt to future API or I/O changes.

		Parameters
		----------
		filename : str
			Pathname of MSRep.xls file.
		"""
		self._filename = filename

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
		self._compounds = [] # List of all compounds identified
		for row_idx in range(6, intres.nrows):		
			self._compounds.append(
				Compound(		
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
		self._hits = {} 
		for row_idx in range(9, libres.nrows):
			cpd = libres.cell(row_idx, column_names.index('Compound number (#)')).value
			if cpd:
				cpd_no = int(cpd)
				self._hits[cpd_no] = []
			
			self._hits[cpd_no].append(
				Hit(		
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
		assert((sorted(self._hits.keys()) == np.arange(1, len(self._compounds)+1)).all()), f'Hits are either not ordered correctly or are missing for certain compounds in {self._filename}'
