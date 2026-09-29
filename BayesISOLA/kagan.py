import numpy as np
from numpy.typing import ArrayLike

# Four physically equivalent orientations of a double-couple frame.
_DC_SYMMETRIES = (
    np.diag([1.0, 1.0, 1.0]),
    np.diag([1.0, -1.0, -1.0]),
    np.diag([-1.0, 1.0, -1.0]),
    np.diag([-1.0, -1.0, 1.0]),
)


def dc_matrix_from_sdr(
    strike: float,
    dip: float,
    rake: float,
) -> np.ndarray:
    """
    Construct a normalized double-couple moment tensor in NED coordinates.

    Parameters
    ----------
    strike, dip, rake
        Fault-plane angles in degrees. Strike is clockwise from north,
        dip is measured downward from horizontal, and rake follows the
        standard seismological convention.

    Returns
    -------
    numpy.ndarray
        Symmetric 3 x 3 moment tensor in north-east-down coordinates.
    """
    angles = np.asarray([strike, dip, rake], dtype=float)

    if not np.isfinite(angles).all():
        raise ValueError("strike, dip and rake must be finite.")

    if not 0.0 <= dip <= 90.0:
        raise ValueError("dip must lie within [0, 90] degrees.")

    strike, dip, rake = np.deg2rad(angles)

    sd, cd = np.sin(dip), np.cos(dip)
    s2d, c2d = np.sin(2.0 * dip), np.cos(2.0 * dip)

    ss, cs = np.sin(strike), np.cos(strike)
    s2s, c2s = np.sin(2.0 * strike), np.cos(2.0 * strike)

    sr, cr = np.sin(rake), np.cos(rake)

    mnn = -(sd * cr * s2s + s2d * sr * ss**2)
    mee = sd * cr * s2s - s2d * sr * cs**2
    mdd = s2d * sr

    mne = sd * cr * c2s + 0.5 * s2d * sr * s2s
    mnd = -(cd * cr * cs + c2d * sr * ss)
    med = -(cd * cr * ss - c2d * sr * cs)

    return np.array([
        [mnn, mne, mnd],
        [mne, mee, med],
        [mnd, med, mdd],
    ])


def _dc_orientation_frame(moment_tensor: ArrayLike) -> np.ndarray:
    """
    Return the right-handed P-B-T eigenvector frame of a moment tensor.
    """
    mt = np.asarray(moment_tensor, dtype=float)

    if mt.shape != (3, 3):
        raise ValueError("moment_tensor must have shape (3, 3).")

    if not np.isfinite(mt).all():
        raise ValueError("moment_tensor must contain only finite values.")

    # Enforce symmetry and remove any isotropic component.
    mt = 0.5 * (mt + mt.T)
    mt = mt - np.trace(mt) / 3.0 * np.eye(3)

    scale = np.linalg.norm(mt)

    if scale <= np.finfo(float).eps:
        raise ValueError(
            "The tensor has no stable deviatoric orientation."
        )

    # eigh returns eigenvectors in ascending eigenvalue order:
    # pressure, intermediate/null, tension.
    eigenvalues, frame = np.linalg.eigh(mt)

    # Repeated eigenvalues imply an undefined rotation around one axis.
    minimum_gap = np.min(np.abs(np.diff(eigenvalues)))

    if minimum_gap <= 1e-10 * scale:
        raise ValueError(
            "The tensor has repeated principal values; "
            "the DC orientation is not unique."
        )

    # Eigenvectors have arbitrary signs. Enforce a proper rotation frame.
    if np.linalg.det(frame) < 0.0:
        frame[:, 0] *= -1.0

    return frame


def kagan_angle_mt(
    moment_tensor_a: ArrayLike,
    moment_tensor_b: ArrayLike,
) -> float:
    """
    Compute the minimum DC-symmetry rotation between two
    moment-tensor principal-axis orientations.

    The isotropic component of each tensor is removed before the
    eigensystem is calculated. For general non-double-couple tensors,
    the result describes the rotation between their deviatoric
    principal-axis frames rather than a strict classical Kagan angle.

    Returns
    -------
    float
        Minimum double-couple rotation angle in degrees, from 0 to 120.
    """
    frame_a = _dc_orientation_frame(moment_tensor_a)
    frame_b = _dc_orientation_frame(moment_tensor_b)

    candidate_angles = []

    for symmetry in _DC_SYMMETRIES:
        rotation = frame_a.T @ frame_b @ symmetry

        cosine = (np.trace(rotation) - 1.0) / 2.0
        cosine = np.clip(cosine, -1.0, 1.0)

        candidate_angles.append(
            np.degrees(np.arccos(cosine))
        )

    return float(min(candidate_angles))


def kagan_angle_sdr(
    sdr_a: ArrayLike,
    sdr_b: ArrayLike,
) -> float:
    """
    Compute the classical Kagan angle between two double-couple
    focal mechanisms defined by strike, dip and rake.

    Parameters
    ----------
    sdr_a, sdr_b
        Sequences containing (strike, dip, rake), in degrees.
        Either nodal plane may be supplied.

    Returns
    -------
    float
        Kagan angle in degrees.
    """
    sdr_a = np.asarray(sdr_a, dtype=float)
    sdr_b = np.asarray(sdr_b, dtype=float)

    if sdr_a.shape != (3,) or sdr_b.shape != (3,):
        raise ValueError(
            "Each SDR input must contain exactly "
            "(strike, dip, rake)."
        )

    mt_a = dc_matrix_from_sdr(*sdr_a)
    mt_b = dc_matrix_from_sdr(*sdr_b)

    return kagan_angle_mt(mt_a, mt_b)