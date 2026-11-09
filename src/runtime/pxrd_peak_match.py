"""PXRD peak matching and the per-space-group stopping rule.

Ported from the authors' ``pxrd_csp_stop`` 1.0.0 engine. The numerical
settings reproduce the manuscript results; changing them changes B.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifParser
from scipy.optimize import differential_evolution


GRID_START = 2.00
GRID_STOP = 69.98
GRID_STEP = 0.02
GRID = np.round(
    np.arange(GRID_START, GRID_STOP + GRID_STEP / 2.0, GRID_STEP), 2
)
FWHM = 0.15
PSEUDO_VOIGT_ETA = 0.50
WAVELENGTH = float(XRDCalculator(wavelength="CuKa").wavelength)

LEVELS = {
    "major": {"height": 0.050, "prominence": 0.015},
    "medium": {"height": 0.015, "prominence": 0.006},
    "fine": {"height": 0.005, "prominence": 0.003},
}
ENERGY_WINDOW_KJ_MOL = 3.0
SCORE_THRESHOLD = 0.80
WEIGHT_M, WEIGHT_D, WEIGHT_H = 1.00, 0.30, 0.30
SOURCE_FLOOR = 0.70
COUNTERPART_FLOOR = 0.05
RESTARTS, MAXITER, POPSIZE = 3, 75, 10
REFINE_LEVELS = (
    ("major", 0.38, 0.60, 0.45),
    ("medium", 0.46, 0.70, 0.35),
    ("fine", 0.58, 0.85, 0.20),
)


def load_structure(cif_path: Path) -> Structure:
    """Load the first non-primitive structure from a CIF file."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return CifParser(
            str(cif_path), occupancy_tolerance=10.0
        ).parse_structures(primitive=False)[0]


def broaden_sticks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    zero_shift: float = 0.0,
) -> np.ndarray:
    """Apply the deterministic 0.15 degree, 50:50 pseudo-Voigt profile."""

    two_theta = np.asarray(two_theta, dtype=float) + float(zero_shift)
    intensity = np.asarray(intensity, dtype=float)
    keep = (two_theta >= GRID_START) & (two_theta <= GRID_STOP)
    two_theta = two_theta[keep]
    intensity = intensity[keep]
    broadened = np.zeros(GRID.shape, dtype=np.float64)
    for start in range(0, len(two_theta), 128):
        x = two_theta[start : start + 128, None]
        h = intensity[start : start + 128, None]
        scaled = (GRID[None, :] - x) / FWHM
        lorentz = 1.0 / (1.0 + 4.0 * scaled * scaled)
        gaussian = np.exp(-4.0 * math.log(2.0) * scaled * scaled)
        profile = PSEUDO_VOIGT_ETA * lorentz + (1.0 - PSEUDO_VOIGT_ETA) * gaussian
        broadened += np.sum(h * profile, axis=0)
    np.maximum(broadened, 0.0, out=broadened)
    maximum = float(np.max(broadened, initial=0.0))
    if maximum > 0.0:
        broadened *= 100.0 / maximum
    return broadened


def simulate_structure(
    structure: Structure,
    *,
    extended_range: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[list[dict[str, object]]], np.ndarray]:
    """Calculate Cu K-alpha sticks, hkl metadata, and a broadened profile."""

    margin = 0.45 if extended_range else 0.0
    pattern = XRDCalculator(wavelength="CuKa", symprec=0).get_pattern(
        structure,
        scaled=True,
        two_theta_range=(GRID_START - margin, GRID_STOP + margin),
    )
    stick_x = np.asarray(pattern.x, dtype=float)
    stick_i = np.asarray(pattern.y, dtype=float)
    profile = broaden_sticks(stick_x, stick_i)
    return stick_x, stick_i, list(pattern.hkls), profile


def gaussian_smooth(y: np.ndarray, xstep: float, sigma_deg: float) -> np.ndarray:
    sigma = max(sigma_deg / xstep, 0.25)
    radius = max(2, int(math.ceil(4.0 * sigma)))
    xx = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * np.square(xx / sigma))
    kernel /= float(np.sum(kernel))
    return np.convolve(np.asarray(y, dtype=float), kernel, mode="same")


def snip_baseline(y: np.ndarray, xstep: float, *, width_deg: float = 1.2) -> np.ndarray:
    """SNIP clipping on the log(1 + sqrt(I)) scale used for the reported results."""

    baseline = np.log1p(np.sqrt(np.clip(np.asarray(y, dtype=float), 0.0, None)))
    max_k = max(3, int(round(width_deg / xstep)))
    for k in range(1, max_k + 1):
        middle = baseline[k:-k]
        clipped = 0.5 * (baseline[:-2 * k] + baseline[2 * k :])
        baseline[k:-k] = np.minimum(middle, clipped)
    return np.square(np.expm1(baseline))


def corrected_pattern(
    pattern: np.ndarray, *, sigma_deg: float = 0.075
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(pattern[:, 0], dtype=float)
    y = np.asarray(pattern[:, 1], dtype=float)
    keep = (x >= GRID_START) & (x <= GRID_STOP)
    x, y = x[keep], y[keep]
    step = float(np.median(np.diff(x)))
    smoothed = gaussian_smooth(np.clip(y, 0.0, None), step, sigma_deg)
    baseline = snip_baseline(smoothed, step)
    corrected = gaussian_smooth(
        np.clip(smoothed - baseline, 0.0, None), step, sigma_deg * 0.45
    )
    maximum = float(np.max(corrected, initial=0.0))
    if maximum > 0.0:
        corrected /= maximum
    return x, corrected


def extract_peaks(
    x: np.ndarray,
    y: np.ndarray,
    *,
    height_cut: float,
    prominence_cut: float,
    min_distance_deg: float = 0.10,
    prominence_window_deg: float = 0.60,
) -> tuple[np.ndarray, np.ndarray]:
    """Return peak positions and normalized heights."""

    idx = np.flatnonzero((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:])) + 1
    if len(idx) == 0:
        return np.empty(0), np.empty(0)
    step = float(np.median(np.diff(x)))
    window = max(2, int(round(prominence_window_deg / step)))
    prominence = np.zeros(len(idx), dtype=float)
    for n, point in enumerate(idx):
        left = float(np.min(y[max(0, point - window) : point + 1]))
        right = float(np.min(y[point : min(len(y), point + window + 1)]))
        prominence[n] = y[point] - max(left, right)
    valid = (y[idx] >= height_cut) & (prominence >= prominence_cut)
    idx = idx[valid]
    if len(idx) == 0:
        return np.empty(0), np.empty(0)
    chosen: list[int] = []
    for offset in np.argsort(y[idx])[::-1]:
        point = int(idx[offset])
        if all(abs(float(x[point] - x[other])) >= min_distance_deg for other in chosen):
            chosen.append(point)
    chosen.sort()
    selected = np.asarray(chosen, dtype=int)
    return x[selected], y[selected]


def peak_levels(x: np.ndarray, y: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: extract_peaks(
            x,
            y,
            height_cut=float(config["height"]),
            prominence_cut=float(config["prominence"]),
        )
        for name, config in LEVELS.items()
    }


def soft_peak_f1(
    exp_x: np.ndarray,
    exp_i: np.ndarray,
    calc_x: np.ndarray,
    calc_i: np.ndarray,
    tolerance: float,
) -> float:
    if len(exp_x) == 0 or len(calc_x) == 0:
        return 0.0
    distance = np.abs(exp_x[:, None] - calc_x[None, :])
    exp_soft = np.exp(-0.5 * np.square(np.min(distance, axis=1) / tolerance))
    calc_soft = np.exp(-0.5 * np.square(np.min(distance, axis=0) / tolerance))
    exp_weight = np.sqrt(np.maximum(exp_i, 1e-12))
    calc_weight = np.sqrt(np.maximum(calc_i, 1e-12))
    exp_weight /= max(float(np.sum(exp_weight)), 1e-12)
    calc_weight /= max(float(np.sum(calc_weight)), 1e-12)
    recall = float(np.dot(exp_weight, exp_soft))
    precision = float(np.dot(calc_weight, calc_soft))
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def maximum_ordered_match_count(
    exp_x: np.ndarray, calc_x: np.ndarray, tolerance: float
) -> int:
    i = j = matched = 0
    while i < len(exp_x) and j < len(calc_x):
        delta = float(exp_x[i] - calc_x[j])
        if abs(delta) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif delta < 0.0:
            i += 1
        else:
            j += 1
    return matched


def representative_hkls(
    peak_x: np.ndarray,
    stick_x: np.ndarray,
    stick_hkls: list[list[dict[str, object]]],
) -> tuple[np.ndarray, float, int]:
    """Map each extracted peak to the nearest actual theoretical reflection."""

    if len(peak_x) == 0:
        return np.empty((0, 3), dtype=float), 0.0, 0
    nearest = np.argmin(np.abs(peak_x[:, None] - stick_x[None, :]), axis=1)
    residual = np.abs(peak_x - stick_x[nearest])
    hkls: list[tuple[int, int, int]] = []
    for index in nearest:
        entries = stick_hkls[int(index)]
        if not entries:
            raise RuntimeError(f"No hkl metadata for stick {index}")
        hkls.append(tuple(int(v) for v in entries[0]["hkl"]))
    return (
        np.asarray(hkls, dtype=float),
        float(np.max(residual, initial=0.0)),
        int(len(nearest) - len(np.unique(nearest))),
    )


def reciprocal_metric(
    parameters: tuple[float, float, float, float, float, float]
) -> np.ndarray:
    a, b, c, alpha, beta, gamma = parameters
    ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
    volume_factor_squared = (
        1.0 + 2.0 * ca * cb * cg - ca * ca - cb * cb - cg * cg
    )
    if not np.isfinite(volume_factor_squared) or volume_factor_squared <= 1e-8:
        raise np.linalg.LinAlgError(
            "Cell angles do not define a physically invertible lattice"
        )
    direct = np.asarray(
        [
            [a * a, a * b * cg, a * c * cb],
            [a * b * cg, b * b, b * c * ca],
            [a * c * cb, b * c * ca, c * c],
        ],
        dtype=float,
    )
    return np.linalg.inv(direct)


def cell_from_params(
    base: np.ndarray, params: np.ndarray
) -> tuple[float, float, float, float, float, float, float]:
    lengths = base[:3] * np.exp(params[:3])
    angles = base[3:] + params[3:6]
    return (
        float(lengths[0]),
        float(lengths[1]),
        float(lengths[2]),
        float(angles[0]),
        float(angles[1]),
        float(angles[2]),
        float(params[-1]),
    )


def angles_for_hkl(
    hkl: np.ndarray, reciprocal: np.ndarray, zero_shift: float
) -> np.ndarray:
    inv_d2 = np.einsum("ni,ij,nj->n", hkl, reciprocal, hkl)
    argument = 0.5 * WAVELENGTH * np.sqrt(np.maximum(inv_d2, 0.0))
    result = np.full(len(hkl), 180.0, dtype=float)
    valid = argument < 0.999999
    result[valid] = np.rad2deg(2.0 * np.arcsin(argument[valid])) + zero_shift
    return result


def refine_cell(
    structure: Structure,
    exp_peaks: dict[str, tuple[np.ndarray, np.ndarray]],
    calc_peaks: dict[str, tuple[np.ndarray, np.ndarray]],
    stick_x: np.ndarray,
    stick_hkls: list[list[dict[str, object]]],
    *,
    seed: int,
    maxiter: int,
    popsize: int,
) -> dict[str, object]:
    lattice = structure.lattice
    base = np.asarray(
        [lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma],
        dtype=float,
    )
    bounds = [(-0.10, 0.10)] * 3 + [(-4.0, 4.0)] * 3 + [(-0.40, 0.40)]
    assignments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    residuals: list[float] = []
    duplicate_assignments = 0
    for name, (positions, intensity) in calc_peaks.items():
        hkl, residual, duplicates = representative_hkls(positions, stick_x, stick_hkls)
        assignments[name] = (hkl, intensity)
        residuals.append(residual)
        duplicate_assignments += duplicates

    def trial(name: str, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b, c, alpha, beta, gamma, zero = cell_from_params(base, params)
        positions = angles_for_hkl(
            assignments[name][0],
            reciprocal_metric((a, b, c, alpha, beta, gamma)),
            zero,
        )
        order = np.argsort(positions)
        return positions[order], assignments[name][1][order]

    def objective(params: np.ndarray) -> float:
        values: list[float] = []
        try:
            for name, soft_tolerance, count_tolerance, level_weight in REFINE_LEVELS:
                exp_x, exp_i = exp_peaks[name]
                calc_x, calc_i = trial(name, params)
                soft = soft_peak_f1(
                    exp_x, exp_i, calc_x, calc_i, tolerance=soft_tolerance
                )
                count = maximum_ordered_match_count(exp_x, calc_x, count_tolerance)
                count_f1 = 2.0 * count / max(len(exp_x) + len(calc_x), 1)
                values.append(
                    level_weight
                    * math.sqrt(max(soft, 0.0) * max(count_f1, 0.0))
                )
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return 1e6
        fit = float(np.sum(values))
        scales = np.asarray([0.10] * 3 + [4.0] * 3)
        correction = float(np.mean(np.square(params[:6] / scales)))
        regularizer = 0.004 * correction + 0.001 * abs(float(params[-1])) / 0.40
        return -(fit - regularizer)

    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        popsize=popsize,
        maxiter=maxiter,
        polish=True,
        tol=0.003,
        updating="immediate",
        workers=1,
    )
    params = np.asarray(result.x, dtype=float)
    a, b, c, alpha, beta, gamma, zero = cell_from_params(base, params)
    return {
        "params": params,
        "cell": (a, b, c, alpha, beta, gamma),
        "zero_shift": zero,
        "objective": float(-result.fun),
        "nfev": int(result.nfev),
        "assignment_max_residual": float(max(residuals, default=0.0)),
        "duplicate_peak_to_stick_assignments": int(duplicate_assignments),
    }


def structure_with_lattice(
    structure: Structure,
    cell: tuple[float, float, float, float, float, float],
) -> Structure:
    lattice = Lattice.from_parameters(*cell)
    return Structure(
        lattice,
        structure.species_and_occu,
        structure.frac_coords,
        coords_are_cartesian=False,
        site_properties=structure.site_properties,
    )


def tolerance_for_pair(x: float, y: float) -> float:
    theta = 0.5 * (float(x) + float(y))
    return 1.00 + 0.007 * max(theta - 10.0, 0.0)


def maximum_variable_tolerance_pairs(
    source_x: np.ndarray, target_x: np.ndarray
) -> list[tuple[int, int]]:
    """Maximum-cardinality order-preserving assignment for the hard gate."""

    ns, nt = len(source_x), len(target_x)
    dp = np.zeros((ns + 1, nt + 1), dtype=np.int16)
    trace = np.zeros((ns + 1, nt + 1), dtype=np.uint8)
    for i in range(1, ns + 1):
        for j in range(1, nt + 1):
            best = int(dp[i - 1, j])
            action = 1
            if int(dp[i, j - 1]) > best:
                best = int(dp[i, j - 1])
                action = 2
            delta = abs(float(source_x[i - 1] - target_x[j - 1]))
            if delta <= tolerance_for_pair(source_x[i - 1], target_x[j - 1]):
                diagonal = int(dp[i - 1, j - 1]) + 1
                if diagonal >= best:
                    best = diagonal
                    action = 3
            dp[i, j] = best
            trace[i, j] = action
    pairs: list[tuple[int, int]] = []
    i, j = ns, nt
    while i > 0 and j > 0:
        action = int(trace[i, j])
        if action == 3:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def strong_guard(
    exp_major: tuple[np.ndarray, np.ndarray],
    calc_major: tuple[np.ndarray, np.ndarray],
    *,
    source_floor: float = SOURCE_FLOOR,
    counterpart_floor: float = COUNTERPART_FLOOR,
) -> dict[str, object]:
    exp_x, exp_i = exp_major
    calc_x, calc_i = calc_major
    exp_strong = exp_i >= source_floor
    calc_strong = calc_i >= source_floor
    exp_targets = exp_i >= counterpart_floor
    calc_targets = calc_i >= counterpart_floor
    forward = maximum_variable_tolerance_pairs(
        exp_x[exp_strong], calc_x[calc_targets]
    )
    reverse = maximum_variable_tolerance_pairs(
        calc_x[calc_strong], exp_x[exp_targets]
    )
    return {
        "exp_strong_n": int(np.sum(exp_strong)),
        "calc_strong_n": int(np.sum(calc_strong)),
        "forward_matched": int(len(forward)),
        "reverse_matched": int(len(reverse)),
        "guard": bool(
            len(forward) == int(np.sum(exp_strong))
            and len(reverse) == int(np.sum(calc_strong))
        ),
    }


def ordered_score_pairs(
    exp_x: np.ndarray,
    exp_i: np.ndarray,
    calc_x: np.ndarray,
    calc_i: np.ndarray,
) -> list[tuple[int, int]]:
    ne, nc = len(exp_x), len(calc_x)
    dp = np.full((ne + 1, nc + 1), np.inf, dtype=float)
    trace = np.zeros((ne + 1, nc + 1), dtype=np.uint8)
    dp[0, 0] = 0.0
    for i in range(1, ne + 1):
        dp[i, 0] = dp[i - 1, 0] + exp_i[i - 1] ** 2
        trace[i, 0] = 1
    for j in range(1, nc + 1):
        dp[0, j] = dp[0, j - 1] + calc_i[j - 1] ** 2
        trace[0, j] = 2
    for i in range(1, ne + 1):
        for j in range(1, nc + 1):
            skip_exp = dp[i - 1, j] + exp_i[i - 1] ** 2
            skip_calc = dp[i, j - 1] + calc_i[j - 1] ** 2
            delta = abs(float(exp_x[i - 1] - calc_x[j - 1]))
            allowed = tolerance_for_pair(exp_x[i - 1], calc_x[j - 1])
            match = np.inf
            if delta <= allowed:
                strength = max(float(exp_i[i - 1]), float(calc_i[j - 1]))
                match = dp[i - 1, j - 1]
                match += 0.05 * strength**2 * delta / allowed
                match += 0.02 * abs(float(exp_i[i - 1] - calc_i[j - 1]))
            action = int(np.argmin([skip_exp, skip_calc, match]))
            dp[i, j] = (skip_exp, skip_calc, match)[action]
            trace[i, j] = action + 1
    pairs: list[tuple[int, int]] = []
    i, j = ne, nc
    while i > 0 or j > 0:
        action = int(trace[i, j])
        if action == 3:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == 1:
            i -= 1
        elif action == 2:
            j -= 1
        else:
            raise RuntimeError(f"Broken score traceback at ({i}, {j})")
    pairs.reverse()
    return pairs


def score_components(
    exp_medium: tuple[np.ndarray, np.ndarray],
    calc_medium: tuple[np.ndarray, np.ndarray],
) -> dict[str, object]:
    exp_x, exp_i = exp_medium
    calc_x, calc_i = calc_medium
    pairs = ordered_score_pairs(exp_x, exp_i, calc_x, calc_i)
    exp_matched = {i for i, _ in pairs}
    calc_matched = {j for _, j in pairs}
    missing_exp = np.asarray(
        [i for i in range(len(exp_x)) if i not in exp_matched], dtype=int
    )
    extra_calc = np.asarray(
        [j for j in range(len(calc_x)) if j not in calc_matched], dtype=int
    )
    denominator = float(np.sum(exp_i**2) + np.sum(calc_i**2))
    exp_weight = 3.0 * np.power(8.0, (10.0 - exp_x[missing_exp]) / 30.0)
    calc_weight = 3.0 * np.power(8.0, (10.0 - calc_x[extra_calc]) / 30.0)
    missing_loss = float(
        np.sum(exp_weight * exp_i[missing_exp] ** 2)
        + np.sum(calc_weight * calc_i[extra_calc] ** 2)
    ) / max(denominator, 1e-12)
    if pairs:
        ei = np.asarray([i for i, _ in pairs], dtype=int)
        cj = np.asarray([j for _, j in pairs], dtype=int)
        strength = np.maximum(exp_i[ei], calc_i[cj])
        allowed = np.asarray(
            [tolerance_for_pair(exp_x[i], calc_x[j]) for i, j in pairs], dtype=float
        )
        position_loss = float(
            np.sum(strength * np.abs(exp_x[ei] - calc_x[cj]) / allowed)
            / max(float(np.sum(strength)), 1e-12)
        )
        height_loss = float(
            np.sum(np.abs(exp_i[ei] - calc_i[cj]))
            / max(float(np.sum(strength)), 1e-12)
        )
    else:
        position_loss = 1.0
        height_loss = 1.0
    count_loss = (len(missing_exp) + len(extra_calc)) / max(
        len(exp_x) + len(calc_x), 1
    )
    return {
        "M": missing_loss,
        "D": position_loss,
        "H": height_loss,
        "N": float(count_loss),
        "B": math.exp(-(WEIGHT_M * missing_loss + WEIGHT_D * position_loss + WEIGHT_H * height_loss)),
        "n_exp_medium": int(len(exp_x)),
        "n_calc_medium": int(len(calc_x)),
        "n_match_medium": int(len(pairs)),
        "n_missing_exp_medium": int(len(missing_exp)),
        "n_extra_calc_medium": int(len(extra_calc)),
    }


def load_experimental_pxrd(path: str | Path) -> np.ndarray:
    """Read two columns (degrees 2theta, intensity) and normalize as in the reported runs.

    Negative intensities are clipped, the pattern is linearly interpolated onto the
    0.02 degree grid within its measured range, the minimum is subtracted and the
    maximum scaled to 100. Background correction follows in ``corrected_pattern``.
    """

    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 2 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("experimental PXRD must be finite two-column data")
    data = data[np.argsort(data[:, 0], kind="mergesort")]
    x = GRID[(GRID >= data[0, 0]) & (GRID <= data[-1, 0])]
    if len(x) < 5:
        raise ValueError(f"experimental PXRD must cover at least five grid points within {GRID_START}-{GRID_STOP} degrees")
    y = np.interp(x, data[:, 0], np.clip(data[:, 1], 0.0, None))
    y -= y.min()
    if y.max() <= 0:
        raise ValueError("experimental PXRD has no varying signal")
    return np.column_stack((x, y * (100.0 / y.max())))


def experimental_peaks(pattern: np.ndarray) -> tuple[dict, float, float]:
    x, y = corrected_pattern(pattern)
    if float(np.max(y, initial=0.0)) <= 0.0:
        raise ValueError("experimental PXRD has no signal after background correction")
    return peak_levels(x, y), float(x.min()), float(x.max())


def _in_range(profile: np.ndarray, xlo: float, xhi: float) -> np.ndarray:
    keep = (GRID >= xlo - 1e-12) & (GRID <= xhi + 1e-12)
    return np.column_stack((GRID[keep], np.asarray(profile, dtype=float)[keep]))


def refinement_seed(cif_path: Path, restart: int) -> int:
    """Seed from the candidate name used by the reported runs (CIF stem without _opt)."""

    name = Path(cif_path).stem.removesuffix("_opt")
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=4).digest()
    return (int.from_bytes(digest, "little") + restart) % (2**32)


def score_restart(task: tuple[str, str, int, dict, float, float]) -> dict[str, object]:
    """Refine one candidate cell from one seed, then score it against experiment."""

    candidate_id, cif_path, restart, exp_levels, xlo, xhi = task
    structure = load_structure(Path(cif_path))
    stick_x, _, stick_hkls, raw_profile = simulate_structure(structure)
    raw_levels = peak_levels(*corrected_pattern(_in_range(raw_profile, xlo, xhi)))
    seed = refinement_seed(Path(cif_path), restart)
    refined = refine_cell(structure, exp_levels, raw_levels, stick_x, stick_hkls,
                          seed=seed, maxiter=MAXITER, popsize=POPSIZE)
    refined_x, refined_i, _, _ = simulate_structure(
        structure_with_lattice(structure, refined["cell"]), extended_range=True)
    profile = broaden_sticks(refined_x, refined_i, zero_shift=float(refined["zero_shift"]))
    calc_levels = peak_levels(*corrected_pattern(_in_range(profile, xlo, xhi)))
    score = score_components(exp_levels["medium"], calc_levels["medium"])
    guard = strong_guard(exp_levels["major"], calc_levels["major"])
    return {
        "candidate_id": candidate_id,
        "restart": restart,
        "seed": seed,
        "refine_objective": refined["objective"],
        "zero_shift": refined["zero_shift"],
        **dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), refined["cell"])),
        **score,
        **{f"guard_{key}": value for key, value in guard.items()},
        "match": bool(guard["guard"] and score["B"] >= SCORE_THRESHOLD),
    }


def evaluate_pool(exp_levels: dict, xlo: float, xhi: float,
                  candidates: list[tuple[str, str, float]], *, workers: int = 1) -> tuple[list[dict], dict | None]:
    """Score one space-group pool of (candidate_id, cif_path, relative energy).

    Each candidate keeps the restart with the largest refinement objective. Only
    the highest-B candidate decides; no lower-ranked candidate is substituted.
    Returns all candidate rows ordered by decision rank and the matching row or None.
    """

    tasks = [(cid, path, restart, exp_levels, xlo, xhi)
             for cid, path, _ in candidates for restart in range(RESTARTS)]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            results = list(executor.map(score_restart, tasks))
    else:
        results = [score_restart(task) for task in tasks]
    energies = {cid: energy for cid, _, energy in candidates}
    rows = []
    for cid, _, _ in candidates:
        restarts = [r for r in results if r["candidate_id"] == cid]
        best = min(restarts, key=lambda r: (-r["refine_objective"], -r["B"], r["restart"]))
        rows.append({**best, "relative_energy_kj_mol": energies[cid],
                     "restart_B_min": min(r["B"] for r in restarts),
                     "restart_B_max": max(r["B"] for r in restarts)})
    rows.sort(key=lambda r: (-r["B"], -r["refine_objective"], r["relative_energy_kj_mol"], r["candidate_id"]))
    return rows, (rows[0] if rows and rows[0]["match"] else None)
