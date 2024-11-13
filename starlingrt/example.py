"""
A simple example using MassHunter(TM) data.
Author: Nathan A. Mahynski

"""
import os

from starlingrt import sample, data

def load_mass_hunter(input_directory):
	"""
	Load MassHunter(TM) data from the input/ directory.

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
	samples : list(sample.MassHunterSample)
		List of Samples collected from all directories in `input_directory`.
	"""
	samples = []
	folders = sorted([f for f in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, f))])
	for f in folders:
		fname = os.path.join(input_directory, os.path.join(f, "MSRep.xls"))
		if os.path.isfile(fname):
			try:
				samples.append(sample.MassHunterSample(fname))
			except Exception as e:
				raise Exception(f'Unable to read {fname} : {e}')
		else:
			raise Exception(f'Could not locate MSRep.xls file in {fname}')

	return samples

if __name__ == "__main__":
    top_entries = data.Utilities.select_top_entries(
        data.Utilities.create_entries(
            load_mass_hunter(
                '../../data/raw/sample_data/Med_2022_vhodni/'
            )
        )
    )

    """
    # 1. Estimate the threshold first
    df, _, _ = get_dataframe(top_entries)
    threshold = estimate_threshold(df, display=True)
    """

    """
    # 2. Use this threshold to create groups and create interactive table.
    threshold = estimate_threshold(get_dataframe(top_entries)[0], display=False)
    print(f'Using a threshold of {threshold}')

    make(
        top_entries=top_entries, 
        width=1200, # Width of the HTML output
        threshold=threshold,
        output_filename='summary.html',
        )
    """