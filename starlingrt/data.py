"""
Structures to organize raw data into.
Author: Nathan A. Mahynski

"""

import copy
import hashlib
import numpy as np

from typing import Any, ClassVar

class Entry:
	"""
	Create an Entry.

	This is essentially a combination of Hit and Compound intended to "unroll" their information into a flat data structure more amenable for searching.
	"""
	def __init__(self, sample_filename: str, compound_number: int, rt: int, scan_number: int, area: int, baseline_height: int, absolute_height: int, peak_width: float, hit_number: int, hit_name: str, quality: int, mol_weight: float, cas_number: str, library: str, entry_number_library: int):
		"""
		Initialize the Entry.
		
		Parameters
		----------
		sample_filename : str
			Mass spectrometer output file data was read from.

		compound_number : int
			Compound / peak integer index.

		rt : float
			Retention time of the peak.

		scan_number : int
			Scan number.

		area : int
			Peak area.

		baseline_height : int
			Baseline peak height.

		absolute_height: int
			Absolute peak height.

		peak_width : float
			Peak width.

		hit_number : int
			Number assigned to the hit.

		hit_name : str
			Hit name assigned by the library used.

		quality : int
			The quality of the identification, as reported by the library.

		mol_weight : float
			Molecular weight of the assignment.

		cas_number : str
			CAS Number of the assigned hit.

		library : str
			Library used for identification.

		entry_number_library : int
			Number the assigned compount is in the library used.
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

	def set_params(self, **parameters: Any) -> "Entry":
		"""Set parameters."""		
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	def get_params(self) -> dict[str, Any]:		
		"""Get parameters."""		
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

	Each peak (Compound) in the MSRep.xls file > LibRes tab is assigned various Hits.
	"""
	def __init__(self, number: int, name: str, quality: int, mol_weight: float, cas_number: str, library: str, entry_number_library: int) -> None:
		"""
		Initialize the Hit.

		Parameters
		----------
		number : int
			Number assigned to the hit.

		name : str
			Hit name assigned by the library used.

		quality : int
			The quality of the identification, as reported by the library.

		mol_weight : float
			Molecular weight of the assignment.

		cas_number : str
			CAS Number of the assigned hit.

		library : str
			Library used for identification.

		entry_number : int
			Number the assigned compount is in the library used.
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
	
	def set_params(self, **parameters: Any) -> "Hit":
		"""Set parameters."""		
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	def get_params(self) -> dict[str, Any]:		
		"""Get parameters."""		
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

	Each peak (Compound) in the MSRep.xls file > LibRes tab is assigned various Hits.
	"""
	def __init__(self, number: int, rt: float, scan_number: int, area: int, baseline_height: int, absolute_height: int, peak_width: float) -> None:
		"""
		Initialize the Compound.

		Parameters
		----------
		number : int
			Compound / peak integer index.

		rt : float
			Retention time of the peak.

		scan_number : int
			Scan number.

		area : int
			Peak area.

		baseline_height : int
			Baseline peak height.

		absolute_height: int
			Absolute peak height.

		peak_width : float
			Peak width.
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

	def set_params(self, **parameters: Any) -> "Compound":
		"""Set parameters."""
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	def get_params(self) -> dict[str, Any]:		
		"""Get parameters."""		
		return {		
			"number": self.number, 
			"rt": self.rt,
			"scan_number": self.scan_number,
			"area": self.area,
			"baseline_height": self.baseline_height,
			"absolute_height": self.absolute_height,
			"peak_width": self.peak_width
		}

class _SampleBase:
	"""Base class to store the output from a mass spectrometer."""

	_filename: ClassVar[str]
	_compounds: ClassVar[list]
	_hits : ClassVar[dict]

	def __init__(self, filename: str) -> None:
		"""
		Instantiate the Sample.

		Parameters
		----------
		filename : str
			Path to mass spectrometer output file to read.
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

	@property
	def entries(self) -> list[Entry]:
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

	def sorted_hits(self, compound_number: int) -> dict[int, Hit]:
		"""
		Hits should be sorted by quality, but this makes sure. 

		A secondary sort is done by hit number to be consistent with mass spectrometer's ordering.

		Parameters
		----------
		compound_number : int
			Compound number (starting from 1) in the mass spectrometer's output file.

		Returns
		-------
		hits : dict(int, Hit)
			Hits sorted first by quality and then by the number the mass spectrometer assigned when it performed this sort.

		Example
		-------
		>>> s = Sample(...)
		>>> sorted_hits = s.sorted_hits(compound_number=42)
		"""
		return sorted(self._hits[compound_number], key=lambda x: (x.get_params().get('quality'), -x.get_params().get('number')), reverse=True)

	def read(self, *args, **kwargs) -> None:
		"""
		Function to read in the data from mass spectrometer output files.
		"""
		self._filename = ""
		self._compounds = []
		self._hits = {}
		raise NotImplementedError

class Utilities:
    """Utility functions for manipulating data structures."""
    
    @staticmethod
    def create_entries(samples: list) -> dict[str, Entry]:
        """
        Extract all Entry from samples.
        
        Parameters
        ----------
        samples : list(_SampleBase)
            List of Samples collected from all directories in `input_directory`.

        Returns
        -------
        total_entries : dict(str, Entry)
            Dictionary of all Entry in `samples` whose keys are sha1 hashes and values are Entry objects.
        """
        total_entries = {}
        checksum = 0
        for sample in samples:
            for entry in sample.entries:
                checksum += 1
                descr_ = '_'.join(['_'.join([a, str(b)]) for a,b in sorted(list(entry.get_params().items()))])
                hash_ = hashlib.sha1(descr_.encode('utf-8'))
                total_entries[hash_.hexdigest()] = entry

        assert(len(total_entries) == checksum), 'Error : hash conflicts found'
        return total_entries

    @staticmethod
    def select_top_entries(total_entries: dict[str, Entry]) -> dict[str, Entry]:
        """
        Trim down the entries to just have the top (quality) hits (i.e., `hit_number` == 1).
        
        Parameters
        ----------
        total_entries : dict(str, Entry)
            Dictionary of all Entry in `samples` whose keys are sha1 hashes.

        Returns
        -------
        top_entries : dict(str, Entry)
            Dictionary of all Entry with `hit_number` == 1 whose keys are sha1 hashes and values are Entry objects.
        """
        top_entries = {}
        for k,v in total_entries.items():
            if v.hit_number == 1:
                top_entries[k] = v
        
        return top_entries

    @staticmethod
    def group_entries_by_name(entries: dict[str, Entry]) -> dict[str, tuple[Entry, str]]:
        """
        Group entries with the same hit name.
        
        Parameters
        ----------
        entries : dict(str, Entry)
            Dictionary of Entry whose keys are sha1 hashes and values are Entry objects.

        Returns
        -------
        groups : dict(str, (Entry, str))
            Dictionary of Entry whose keys are hit names and values are tuples of (Entry objects, hash).
        """
        groups = {}
        for hash,entry in entries.items():
            if entry.hit_name in groups:
                groups[entry.hit_name].append((entry, hash))
            else:
                groups[entry.hit_name] = [(entry, hash)]

        return groups

    @staticmethod
    def group_entries_by_rt(entries: dict[str, Entry]) -> dict[str, Entry]:
        """
        Group entries with the same retention time.
        
        Parameters
        ----------
        entries : dict(str, Entry)
            Dictionary of Entry whose keys are sha1 hashes and values are Entry objects.

        Returns
        -------
        groups : dict(str, Entry)
            Dictionary of Entry whose keys are retention times and values are Entry objects.
        """
        groups = {}
        for entry in entries.values():
            if entry.rt in groups:
                groups[entry.rt].append(entry)
            else:
                groups[entry.rt] = [entry]

        return groups
