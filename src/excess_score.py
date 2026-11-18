"""Architecture-only NSC with a random-permutation reference."""
import math
from numbers import Real

from nsc import layer_capacities, original_nsc

# Selected on the development grid; fixed during evaluation.
SELECTED_LAMBDA = 0.5


def _coefficient(value):
    """Validate and return a finite nonnegative penalty coefficient."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('lambda must be a finite nonnegative real number')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('lambda must be a finite nonnegative real number')
    return value


def capacity_statistics(values):
    """Normalize by the maximum to avoid overflow and an underflowed mean."""
    values = list(values)
    if not values:
        raise ValueError('At least one layer is required')
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in values):
        raise ValueError('Capacities must be finite nonnegative real numbers')
    values = [float(v) for v in values]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Capacities must be finite nonnegative real numbers')
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError('Total capacity is not representable as a finite float') from exc
    if not math.isfinite(total):
        raise ValueError('Total capacity must be finite')
    length = len(values)
    maximum = max(values)
    if maximum == 0:
        return dict(total=0.0, mean=0.0, adjacent=0.0, global_cv2=0.0)
    scaled = [v / maximum for v in values]
    mean = math.fsum(scaled) / length
    adjacent = (math.fsum(((b - a) / mean) ** 2
                         for a, b in zip(scaled, scaled[1:])) / (length - 1)
                if length > 1 else 0.0)
    global_cv2 = math.fsum(((v - mean) / mean) ** 2 for v in scaled) / length
    return dict(total=total, mean=total / length,
                adjacent=adjacent, global_cv2=global_cv2)


def penalized_capacity(values, coefficient, mechanism='adjacent'):
    """Apply the adjacent or global penalty to layer capacities."""
    coefficient = _coefficient(coefficient)
    if mechanism not in ('adjacent', 'global_cv2'):
        raise ValueError('Unknown imbalance mechanism')
    stats = capacity_statistics(values)
    total, mismatch = stats['total'], stats[mechanism]
    if coefficient == 0 or mismatch == 0 or total == 0:
        return total
    denominator = 1 + coefficient * mismatch
    if math.isfinite(denominator):
        return total / denominator
    # Avoid overflow in lambda*D.
    return (total / coefficient) / (mismatch + 1 / coefficient)


def adjacent_score(config, coefficient):
    """Compute architecture-only NSC with the adjacent mismatch penalty."""
    coefficient = _coefficient(coefficient)
    if coefficient == 0:
        return original_nsc(config)
    return penalized_capacity(layer_capacities(config), coefficient, 'adjacent')


def global_score(config, coefficient):
    """Compute architecture-only NSC with the global CV-squared penalty."""
    coefficient = _coefficient(coefficient)
    if coefficient == 0:
        return original_nsc(config)
    return penalized_capacity(layer_capacities(config), coefficient, 'global_cv2')



def excess_statistics(values):
    """E[D_adj] = 2L/(L-1)*CV^2; excess is exactly zero for L <= 2."""
    values = list(values)
    stats = capacity_statistics(values)
    length = len(values)
    reference = 2 * length / (length - 1) * stats['global_cv2'] if length > 1 else 0.0
    excess = max(0.0, stats['adjacent'] - reference) if length > 2 else 0.0
    return dict(**stats, random_order_reference=reference, excess=excess)


def excess_capacity(values, coefficient):
    """Penalize adjacent mismatch above its random-order expectation."""
    coefficient = _coefficient(coefficient)
    stats = excess_statistics(values)
    total, mismatch = stats['total'], stats['excess']
    if coefficient == 0 or mismatch == 0 or total == 0:
        return total
    denominator = 1 + coefficient * mismatch
    if math.isfinite(denominator):
        return total / denominator
    return (total / coefficient) / (mismatch + 1 / coefficient)


def excess_score(config, coefficient):
    """Compute the excess-adjacency score directly from a configuration."""
    coefficient = _coefficient(coefficient)
    if coefficient == 0:
        return original_nsc(config)
    return excess_capacity(layer_capacities(config), coefficient)


def score(config):
    """Return the official student score at the fixed development coefficient."""
    return excess_score(config, SELECTED_LAMBDA)


def ablation(config):
    """Remove the permutation reference at the same lambda."""
    return adjacent_score(config, SELECTED_LAMBDA)
