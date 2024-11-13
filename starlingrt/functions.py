"""
Functions to manipulate and compute properties of GCMS data.
Author: Nathan A. Mahynski

"""

import load
import bokeh
import scipy

import numpy as np
import pandas as pd

from numpy.typing import NDArray, ArrayLike

def flag_entry_rt(entries: list[Entry], min_entries: int = 10, k: float = 3.0, cv: int = 5, style: str = "classical") -> NDArray[np.bool_]:
    """
    Flag entries with anomolous retention times based on the group's consensus.
    
    If a point is considered an outlier in any single fold, it is flagged.
    
    Parameters
    ----------
    entries : list(Entry)
        List of entries (e.g., grouped by name) to examine.

    min_entries : int, optional(default=10)
        Minimum length of `entries` to do KFold cross validation (CV), otherwise ignore `cv` and do LOOCV is instead.

    k : float, optional(default=3.0)
        Number of standard deviations from center allowed.  `k` = 1.5 is more appropriate if using the `robust` approach based on the IQR as a measure of "spread"; `k` = 3 is more appropriate for `classical`, but can be reasonable for the `robust` approach as well.
    
    cv : int, optional(default=5)
        Number of folds to use in cross-validation. If len(`entries`) > `cv`, LOOCV is used instead.

    style: str, optional(default="classical")
        When `classical` use mean and std vs. `robust` which uses median and iqr for center and spread, respectively.  Inliers are considered those for which: `center - k*spread < x < center + k*spread`. The rest are flagged.
        
    Returns
    -------
    mask : ndarray(bool)
        Mask of outliers corresponding to the ordering in `entries`.
    """
    from sklearn.model_selection import KFold
    
    if style == 'classical':
        center_f = np.mean
        spread_f = np.std
    elif style == 'robust':
        center_f = np.median
        spread_f = scipy.stats.iqr # rng = (25,75) by default
    else:
        raise ValueError(f'Unrecognized style : {style}')
    
    entries = np.asarray(entries)
    outlier_mask = np.zeros(len(entries), dtype=bool)
    
    if (len(entries) < min_entries) or (len(entries) < cv):
        # Do LOOCV instead of KFold CV
        cv = len(entries)
    
    if len(entries) < 2:
        # Not enough examples to estimate distribution, assuming inliers
        return outlier_mask
    else:
        kf = KFold(n_splits=cv, shuffle=False)
        retention_times = np.array([e[0].rt for e in entries])
        for train_index, test_index in kf.split(entries):
            # Estimate center and spread from training fraction
            rt_train = retention_times[train_index]
            center = center_f(rt_train)
            spread = spread_f(rt_train)

            # Test all (including training set)
            inliers = (center - k*spread < retention_times) & (retention_times < center + k*spread)
            for i in np.where(~inliers)[0]:
                outlier_mask[i] = True # If any fold finds a point to be an outlier, it is flagged that way

    return outlier_mask

def make_histograms(by_name: dict[str, tuple], k_values: ArrayLike, bins: int = 10, cv: int = 3, style: str = "robust", min_entries: int = 5) -> tuple[dict, dict, dict]:
    """
    Make histograms of retention times for each compound by name.

    Parameters
    ----------
    by_name : dict(str, (Entry, str))
        Dictionary of Entry whose keys are hit names and values are tuples of (Entry, hash).
    
    k_values : array-like
        List of `k` values to use to flag "outlying" retention times.

    bins : int
        Number of histogram bins to use for retention times.

    cv : int, optional(default=3)
        Number of folds to use in cross-validation.

    style: str, optional(default="classical")
        When `classical` use mean and std vs. `robust` which uses median and iqr for center and spread, respectively.  Inliers are considered those for which: `center - k*spread < x < center + k*spread`. The rest are flagged.
        
    min_entries : int, optional(default=5)
        Minimum length of `entries` to do KFold cross validation (CV), otherwise ignore `cv` and do LOOCV is instead.

    Returns
    -------
    histograms : dict(str, list)
        Values of the histogram for each compound.

    bin_edges : dict(str, list)
        Bin edges for the histogram of each compound.

    points : dict(str, dict(str, dict(str, list)))
        Nested dictionary of points which are of concern or not of concern for each `k` value; e.g., points['methane']['3.0']['concern'] = {'x': rention_times, 'y': staggered_bin_counts}.
    """
    histograms = {}
    bin_edges = {}
    for name in by_name:
        rt = [entry[0].rt for entry in by_name[name]]
        hist, edges = np.histogram(rt, bins=bins)
        histograms[name] = hist.tolist()
        bin_edges[name] = edges.tolist()
    
    
    points = {}
    for name in by_name:
        points[name] = {}
        for k in k_values:
            points[name][str(k)] = {}
            potential_issues = flag_entry_rt(
                by_name[name], 
                min_entries=min_entries, 
                k=k, 
                cv=cv, 
                style=style
            )
            
            concerns = [x[0].rt for x in np.array(by_name[name])[potential_issues]]
            not_concern = [x[0].rt for x in np.array(by_name[name])[~potential_issues]]
            
            def get_bin(x, edges):
                bin_ = 0;
                while x > edges[bin_+1]:
                    bin_ += 1
                return bin_
            
            def stagger_y(points, bin_edges, name):
                bin_counter = {}
                y = []
                for x in points:
                    bin_ = get_bin(x, bin_edges[name])
                    if bin_ in bin_counter:
                        bin_counter[bin_] += 1
                    else:
                        bin_counter[bin_] = 1
                    y.append(bin_counter[bin_])
                return y

            if len(concerns) > 0:
                points[name][str(k)]['concern'] = {
                    'x': concerns,
                    'y': stagger_y(concerns, bin_edges, name)
                }
            else:
                points[name][str(k)]['concern'] = {
                    'x': [],
                    'y': []
                }
            
            if len(not_concern) > 0:
                points[name][str(k)]['not_concern'] = {
                    'x': not_concern,
                    'y': stagger_y(not_concern, bin_edges, name)
                }
            else:
                points[name][str(k)]['not_concern'] = {
                    'x': [],
                    'y': []
                }

    return histograms, bin_edges, points

def make_dataframe(entries: dict) -> pd.DataFrame:
    """
    Create a dataframe out of the entries.
    
    Parameters
    ----------
    entries : dict(str, Entry)
		Dictionary of Entry whose keys are sha1 hashes and values are Entry objects.
 
    Returns
    -------
    dataframe : pd.DataFrame
        DataFrame of Entries sorted by Hit name.
    """
    by_name = load.group_entries_by_name(entries) 
    columns = sorted(by_name[list(by_name.keys())[0]][0][0].get_params().keys())
    data = []
    for name in by_name.keys():
        for entry,hash in by_name[name]:
            raw = entry.get_params()
            data.append([raw[k] for k in columns]+[str(hash)])
    df = pd.DataFrame(data=data, columns=columns+['hash'])
    
    return df

def get_dataframe(entries: dict, target: Union[str, None] = None, pm: int = 0) -> tuple[pd.DataFrame, pd.api.typing.DataFrameGroupBy, list]:
    """
    Get dataframe centered on a target.
    
    Parameters
    ----------
    entries : dict(str, Entry)
		Dictionary of Entry whose keys are sha1 hashes and values are Entry objects.
 
    target : str, optional(default=None)
        Name of retention time group to target.

    pm : int, optional(default=0)
        Number of neighbors around `target` to select.

    Returns
    -------
    results : pd.DataFrame
        DataFrame of retention times for selected target and any selected neighbors.

    name_groups : pd.api.typing.DataFrameGroupBy
        Pandas GroupBy object containing the `entries` grouped by Hit name.

    order_cats_used : list
        Ordered list of names of the groups selected.
    """
    df = make_dataframe(entries)

    # Add the number of observations to the x-axis labels
    name_groups = df.groupby('hit_name')
    ordered_cats = name_groups['rt'].median().sort_values().index.tolist()
    
    df['origin'] = df.sample_filename.apply(lambda x:x.strip().split('/')[-2:-1][0])
    
    def select(target, pm=0):
        """Select the pm number of neighbors."""
        if target is None:
            # Look at all compounds
            return ordered_cats
        else:
            if target not in ordered_cats:
                raise ValueError(f'{target} not found')
            else:
                center = ordered_cats.index(target)
                return ordered_cats[np.max([0, center-pm]):np.min([len(ordered_cats), center+pm+1])]
    
    order_cats_used = select(target, pm)
    mask = df['hit_name'].apply(lambda x: x in order_cats_used)
    
    return df.loc[mask], name_groups, order_cats_used

def get_quantiles_df(name_groups: pd.api.typing.DataFrameGroupBy) -> pd.DataFrame:
    """
    Get the 0.25, 0.50, and 0.75 percentiles of the groups.

    Parameters
    ----------
    name_groups : pd.api.typing.DataFrameGroupBy
        Pandas GroupBy object containing the `entries` grouped by Hit name.

    Returns
    -------
    dataframe : pd.DataFrame
        DataFrame summarizing the IQR and whisker bounds for the `name_groups`.
    """
    q1 = name_groups['rt'].quantile(q=0.25)
    q2 = name_groups['rt'].quantile(q=0.5)
    q3 = name_groups['rt'].quantile(q=0.75)

    iqr = q3 - q1
    upper = q3 + 1.5*iqr # Do not bound by max/min, just report ranges
    lower = q1 - 1.5*iqr # Do not bound by max/min, just report ranges
    
    df = pd.concat([
        pd.DataFrame(q1).rename(columns={'rt':'q1'}),
        pd.DataFrame(q2).rename(columns={'rt':'q2'}),
        pd.DataFrame(q3).rename(columns={'rt':'q3'}),
        pd.DataFrame(upper).rename(columns={'rt':'upper'}),
        pd.DataFrame(lower).rename(columns={'rt':'lower'}),
    ], axis=1)
    
    df['new_name'] = df.index.copy()
    
    return df

def closest_rt(rt : float, df_iqr : pd.DataFrame) -> str:
    """
    Suggest the (visible) species with the closest median retention time.

    Parameters
    ----------
    rt : float
        Retention time targeted.

    df_iqr : pd.DataFrame
        DataFrame summarizing the IQR and whisker bounds for the `name_groups`.

    Returns
    -------
    guess : str
        Name of the Hit with the closes retention time.
    """
    return df_iqr.sort_values(by='q2', axis=0, key=lambda x: (x-rt)**2).iloc[0].new_name

def group_by_rt_step(
    df: pd.DataFrame,
    threshold: float = 0.04
) -> list[tuple[int, str, float, float]]:
    """
    Create groups based on similar retention times.
    
    This algorithm simply sorts based on retention time, then creates a new group when a gap between consecutive (sorted) points exceeds the threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of retention times (see `get_dataframe`).

    threshold : float, optional(default=0.04)
        Minimum retention time gap between consecutive compounds to be resolved as different.
    
    Returns
    -------
    rt_groups : list(tuples)
        List of tuples of (index, hit_name, quality, rt) by group.
    """
    rt_sorted_df = df.sort_values(by='rt', ascending=True)
    
    rt_groups = []
    for i, (index, row) in enumerate(rt_sorted_df.iterrows()):
        if i == 0: # First time
            level = row['rt']
            rt_groups.append([(index, row['hit_name'], row['quality'], row['rt'])])
        else:
            if (row['rt'] - level) > threshold: 
                rt_groups.append([(index, row['hit_name'], row['quality'], row['rt'])])
            else:
                rt_groups[-1].append((index, row['hit_name'], row['quality'], row['rt']))
            level = row['rt']
            
    return rt_groups

def suggest_names(
    rt_groups : list[tuple[int, str, float, float]]
) -> tuple[list, dict, list]:
    """
    Suggest the best name for group compounds with similar retention times.
    
    Computes a "probability" using the quality of each observation in a group to determine the most likely name.
    
    Parameters
    ----------
    rt_groups : list(tuple)
        List of tuples of (index, hit_name, quality, rt) by group (see `group_by_rt_step`).
    
    Returns
    -------
    suggested_name : list
        Suggested name of each group.

    ties : dict
        Dictionary of ties.

    entropy : list
        Entropy of each group.
    """
    ties = {}
    suggested_name = []
    entropy = []
    for i,group in enumerate(rt_groups):
        probs = {}
        for entry in group:
            hit_name, quality = entry[1], entry[2]
            if hit_name in probs:
                probs[hit_name] += quality
            else:
                probs[hit_name] = float(quality)
        sum_ = np.sum(list(probs.values()))
        
        entropy.append(0)
        for hit_name in probs:
            probs[hit_name] /= sum_
            entropy[-1] -= probs[hit_name]*np.log(probs[hit_name])

        sorted_probs = sorted(probs.items(), key=lambda x:x[1], reverse=True)
        best_guess, best_prob = sorted_probs[0]

        if len(sorted_probs) > 1:
            # Check if there is a tie
            if best_prob == sorted_probs[1][1]:
                ties[i] = sorted_probs

        suggested_name.append(best_guess)
    
    return suggested_name, ties, entropy

def assign_suggestions(
    df,
    rt_groups,
    suggested_name,
    ties
):
    """
    Unroll the suggestions and assign them to the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of retention times (see `get_dataframe`).

    rt_groups : list(tuple)
        List of tuples of (index, hit_name, quality, rt) by group (see `group_by_rt_step`).

    suggested_name : list
        Suggested name of each group (see `suggest_names`).

    ties : dict
        Dictionary of ties (see `suggest_names`).
        
    Returns
    -------
    dataframe : pd.DataFrame
        DataFrame with "suggested_name" and "flag" columns added.
    """
    new_df = df.copy()
    
    suggested_name = np.asarray(suggested_name)
    unrolled_names = [""]*len(df)
    second_suggestion = [""]*len(df)
    flags = [0]*len(df)
    for i,group in enumerate(rt_groups):
        flag = ''
        suggestion = suggested_name[i]
        if i in ties: # Flag if this group's assignment was tied b.w multiple compounds
            flag = 'Tied for most likely compound'
        if np.sum(suggested_name == suggestion) > 1: # Flag if multiple groups being assigned this name
            flag = 'Suggested compound has multiple RT groups'

        for entry in group:
            index = entry[0]

            assert(unrolled_names[index] == "")
            unrolled_names[index] = suggestion

            if i in ties:
                assert(second_suggestion[index] == "")
                second_suggestion[index] = ties[i][1][0]

            flags[index] = flag
            
    new_df['suggested_name'] = unrolled_names
    new_df['second_suggestion'] = second_suggestion
    new_df['flag'] = flags
    
    return new_df

def estimate_threshold(
    df: pd.DataFrame,
    thresholds: Sequence = np.logspace(-4, 1, 100),
    display: bool = False,
) -> float:
    """
    Estimate the minimum gap (threshold) to separate groups from each other.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of retention times for selected target and any selected neighbors (see `get_dataframe`).

    thresholds : Sequence, optional(default=np.logspace(-4, 1, 100))
        Sequence of thresholds to try.

    display : bool, optional(default=False)
        Whether or not to display the results visually.

    Returns
    -------
    threshold : float
        Choice of threshold.
    """
    rt_sorted_df = df.sort_values(by='rt', ascending=True)
    
    duplicates = []
    confusion = []
    size = []
    disorder = []
    for res in thresholds:
        rt_groups = group_by_rt_step(rt_sorted_df, res)
        suggested_name, ties, entropy = suggest_names(rt_groups)
        names, counts = np.unique(suggested_name, return_counts=True)
        
        confusion.append(len(ties))
        duplicates.append(np.sum(counts > 1))
        size.append(len(rt_groups))
        disorder.append(np.median(entropy))
        
    product = np.array(duplicates)*np.array(disorder)

    if display:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
            
        ax = axes[0]
        ax.plot(thresholds, duplicates, label='Duplicates')
        ax.plot(thresholds, confusion, label='Group Assignments with Ties')
        ax.plot(thresholds, size, label='Retention Time Groups')
        ax.plot(thresholds, disorder, label='Median Entropy of Groups')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Value')
        ax.legend(loc='best')
        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.axhline(1, color='green', ls='--') # Everything lumped into one group
        ax.axhline(
            len(rt_sorted_df['rt'].unique()), 
            color='green', ls='--') # Everything on its own up to some numerical precision

        ax = axes[1]
        ax.plot(thresholds, product)
        ax.set_xscale('log')
        ax.text(thresholds[np.argmax(product)], np.max(product)*1.01, 
                '(%.3f, %.3f)'%(thresholds[np.argmax(product)], np.max(product))
                )
        ax.set_ylabel(r'Duplicates $\times$ Median Entropy')
        ax.set_xlabel('Threshold')
    
    return thresholds[np.argmax(product)]