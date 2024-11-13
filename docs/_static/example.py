"""
A simple example using MassHunter(TM) data.

Author: Nathan A. Mahynski
"""
import os
import starlingrt

from starlingrt import sample, data, functions, visualize


def load_mass_hunter(input_directory):
    """
    Load MassHunter(TM) data from the input/ directory.

    The data is assumed to be laid out in a tree structure like this:

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
    folders = sorted(
        [
            f
            for f in os.listdir(input_directory)
            if os.path.isdir(os.path.join(input_directory, f))
        ]
    )
    for f in folders:
        fname = os.path.join(input_directory, os.path.join(f, "MSRep.xls"))
        if os.path.isfile(fname):
            try:
                samples.append(starlingrt.sample.MassHunterSample(fname))
            except Exception as e:
                raise Exception(f"Unable to read {fname} : {e}")
        else:
            raise Exception(f"Could not locate MSRep.xls file in {fname}")

    return samples


if __name__ == "__main__":
    top_entries = starlingrt.data.Utilities.select_top_entries(
        starlingrt.data.Utilities.create_entries(
            load_mass_hunter("/path/to/data/")
        )
    )

    """
    # 1. Estimate the threshold first
    df, _, _ = starlingrt.functions.get_dataframe(top_entries)
    threshold = starlingrt.functions.estimate_threshold(df, display=True)
    """

    """
    # 2. Use this threshold to create groups and create interactive table.
    threshold = starlingrt.functions.estimate_threshold(
        starlingrt.functions.get_dataframe(top_entries)[0],
        display=False
    )
    print(f'Using a threshold of {threshold}')

    starlingrt.visualize.make(
        top_entries=top_entries,
        width=1200,
        threshold=threshold,
        output_filename='summary.html',
    )
    """
