"""
Utilities to compute pairwise agreement indices among experts.

This module provides:
- a weighted quantile function that correctly ignores invalid values
  (e.g. NaNs on the diagonal of the agreement matrix),
- a function to compute agreement index statistics for each target
  question.

The agreement index is defined as:

    I(i, j) = [min(M_i, M_j) - max(m_i, m_j)] /
              [max(M_i, M_j) - min(m_i, m_j)]

where:
- m_i, M_i are the lower and upper bounds of expert i uncertainty interval,
- m_j, M_j are the lower and upper bounds of expert j uncertainty interval.

The index is computed pairwise for all distinct couples of experts.
"""

from __future__ import annotations

import numpy as np


def weighted_quantile(
    values,
    quantiles,
    sample_weight=None,
    values_sorted=False,
    old_style=False,
):
    """
    Compute weighted quantiles while correctly ignoring invalid values.

    This function is similar to `numpy.percentile`, but it supports
    observation weights. Compared with the original implementation,
    this version explicitly removes NaN values and also removes the
    corresponding weights. This is essential when the input contains
    placeholders such as the diagonal of a pairwise matrix.

    Parameters
    ----------
    values : array-like
        Input data. It can have any shape and is flattened internally.
    quantiles : array-like
        Quantiles to compute, expressed in the [0, 1] interval.
        For example, [0.05, 0.5, 0.95].
    sample_weight : array-like, optional
        Non-negative weights associated with `values`. Must have the
        same size as `values` after flattening. If None, equal weights
        are used.
    values_sorted : bool, optional
        If True, `values` are assumed to be already sorted in ascending
        order, and no sorting is performed.
    old_style : bool, optional
        If True, applies the normalization convention used in older
        weighted percentile implementations for closer compatibility
        with `numpy.percentile`.

    Returns
    -------
    numpy.ndarray
        Array of weighted quantiles.

    Raises
    ------
    ValueError
        If quantiles are outside [0, 1], if input sizes are inconsistent,
        or if no valid data with positive total weight are available.

    Notes
    -----
    Original idea adapted from:
    https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """
    values = np.ravel(np.asarray(values, dtype=float))
    quantiles = np.ravel(np.asarray(quantiles, dtype=float))

    if sample_weight is None:
        sample_weight = np.ones_like(values, dtype=float)
    else:
        sample_weight = np.ravel(np.asarray(sample_weight, dtype=float))

    if values.size != sample_weight.size:
        raise ValueError(
            "`values` and `sample_weight` must have the same size.")

    if not np.all((0.0 <= quantiles) & (quantiles <= 1.0)):
        raise ValueError("`quantiles` should be in the [0, 1] interval.")

    # Remove invalid entries from both values and weights.
    valid_mask = np.isfinite(values) & np.isfinite(sample_weight)
    values = values[valid_mask]
    sample_weight = sample_weight[valid_mask]

    # Remove zero or negative weights.
    positive_mask = sample_weight > 0.0
    values = values[positive_mask]
    sample_weight = sample_weight[positive_mask]

    if values.size == 0:
        raise ValueError("No valid data points with positive weight.")

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight

    if old_style:
        if weighted_quantiles[-1] == weighted_quantiles[0]:
            return np.full_like(quantiles, values[0], dtype=float)
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        total_weight = np.sum(sample_weight)
        if total_weight <= 0.0:
            raise ValueError("Sum of weights must be positive.")
        weighted_quantiles /= total_weight

    return np.interp(quantiles, weighted_quantiles, values)


def calculate_index(TQ_array, weight, background_measure):
    """
    Compute agreement index statistics for each target question.

    For each target question, the function computes the pairwise
    agreement index between all distinct couples of experts, then
    summarizes the resulting matrix with:
    - weighted mean,
    - weighted standard deviation,
    - weighted 5th, 50th, and 95th percentiles.

    Parameters
    ----------
    TQ_array : numpy.ndarray
        Array with shape (E, 3, Nq), where:
        - E is the number of experts,
        - 3 corresponds to the elicited percentiles
          (typically lower, median, upper),
        - Nq is the number of target questions.

        Only the first and third percentile are used here to define the
        uncertainty interval of each expert.
    weight : array-like
        Expert weights, of length E.
    background_measure : array-like
        Sequence of length Nq defining the scale of each question.
        Supported values are:
        - "uni": use values in the original scale,
        - "log": compute the agreement in logarithmic space.

    Returns
    -------
    indexMean : numpy.ndarray
        Array of shape (Nq,) with the weighted mean agreement index for
        each target question.
    indexStd : numpy.ndarray
        Array of shape (Nq,) with the weighted standard deviation of the
        agreement index for each target question.
    indexQuantiles : numpy.ndarray
        Array of shape (Nq, 3) containing the weighted quantiles
        [0.05, 0.5, 0.95] for each target question.

    Notes
    -----
    The diagonal of the pairwise agreement matrix is set to NaN because
    self-comparisons are not defined. These NaNs are excluded from both
    the weighted mean/std computation and the weighted quantile
    computation.
    """
    TQ_array = np.asarray(TQ_array, dtype=float)
    weight = np.asarray(weight, dtype=float)

    n_questions = TQ_array.shape[2]
    n_experts = TQ_array.shape[0]

    index_tot = np.full((n_questions, n_experts, n_experts),
                        np.nan,
                        dtype=float)
    index_mean = np.zeros(n_questions, dtype=float)
    index_std = np.zeros(n_questions, dtype=float)
    index_quantiles = np.zeros((n_questions, 3), dtype=float)

    weight_table = np.outer(weight, weight)

    for i in range(n_questions):
        for j in range(n_experts):
            if background_measure[i] == "uni":
                m_j = TQ_array[j, 0, i]
                M_j = TQ_array[j, 2, i]
            elif background_measure[i] == "log":
                m_j = np.log(TQ_array[j, 0, i])
                M_j = np.log(TQ_array[j, 2, i])
            else:
                raise ValueError(
                    f"Unsupported background_measure '{background_measure[i]}'"
                    f" for question index {i}.")

            for h in range(n_experts):
                if h == j:
                    continue

                if background_measure[i] == "uni":
                    m_h = TQ_array[h, 0, i]
                    M_h = TQ_array[h, 2, i]
                elif background_measure[i] == "log":
                    m_h = np.log(TQ_array[h, 0, i])
                    M_h = np.log(TQ_array[h, 2, i])

                m_un = np.minimum(m_j, m_h)
                M_un = np.maximum(M_j, M_h)

                m_in = np.maximum(m_j, m_h)
                M_in = np.minimum(M_j, M_h)

                denominator = M_un - m_un
                numerator = M_in - m_in

                # Handle degenerate cases robustly.
                # If both intervals collapse to the same point, agreement is 1.
                # Otherwise, if the denominator is zero for any unexpected
                # reason, return NaN.
                if np.isclose(denominator, 0.0):
                    if np.isclose(numerator, 0.0):
                        index_tot[i, j, h] = 1.0
                    else:
                        index_tot[i, j, h] = np.nan
                else:
                    index_tot[i, j, h] = numerator / denominator

        # Mask self-comparisons and any invalid entries.
        valid_mask = np.isfinite(index_tot[i, :, :])
        valid_values = index_tot[i, :, :][valid_mask]
        valid_weights = weight_table[valid_mask]

        if valid_values.size == 0:
            index_mean[i] = np.nan
            index_std[i] = np.nan
            index_quantiles[i, :] = np.nan
            continue

        # Weighted quantiles computed only on valid pairwise comparisons.
        index_quantiles[i, :] = weighted_quantile(
            valid_values,
            [0.05, 0.5, 0.95],
            sample_weight=valid_weights,
        )

        # Weighted mean and standard deviation on valid entries only.
        index_mean[i] = np.average(valid_values, weights=valid_weights)
        variance = np.average(
            (valid_values - index_mean[i])**2,
            weights=valid_weights,
        )
        index_std[i] = np.sqrt(variance)

    return index_mean, index_std, index_quantiles
