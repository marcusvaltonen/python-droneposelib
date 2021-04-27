import numpy as np
import droneposelib as dpl


def pflat(x):
    """Divide by the last coordinate to get an inhomogenous representation."""
    # Enfoce 2D
    if x.ndim == 1:
        x = np.expand_dims(x, 1)
    return x / x[-1, :]


def skew(a):
    a = np.squeeze(a)
    """Create skew-symmetric matrix corresponding to cross product."""
    return np.array([
            [0, -a[2], a[1]],
            [a[2], 0, -a[0]],
            [-a[1], a[0], 0]])


def radialdistort(x, kappa):
    """Applies radial distortion to the (homogeneous or inhomogeneous) coordinates x
    using the parameter kappa. Assumes the division model."""

    # Check if homogeneous
    ishom = x.shape[0] == 3

    if ishom:
        x = pflat(x)
        x = x[:2, :]

    # Compute distorted radius
    ru2 = np.sum(x**2, axis=0)
    ru = np.sqrt(ru2)

    # Compute distorted radius
    if kappa == 0:
        rd = ru
    else:
        rd = 0.5 / kappa / ru - np.sign(kappa) * np.sqrt(0.25 / kappa**2 / ru2 - 1.0 / kappa)

    # Compute distorted coordinates
    y = np.tile(rd / ru, (x.shape[0], 1)) * x

    if ishom:
        y = np.vstack((y, np.ones((1, y.shape[1]))))

    return y


def generate_points_realistic(N=100, distortion_param=0, rng=None):
    """Generates two poses and the corresponding scene points and image points."""
    # Check if a seed is used (for unittests)
    if not rng:
        rng = np.random.default_rng()

    # Relative translation
    t = 2 * rng.random((3, 1)) - 1

    # Make sure the baseline is okay
    t = t / np.linalg.norm(t)

    # Calibration matrix
    f = rng.random() * 200 + 200
    K = np.diag([f, f, 1.0])
    Kinv = np.diag([1.0 / f, 1.0 / f, 1.0])

    R1, _ = np.linalg.qr(rng.random((3, 3)))
    R2, _ = np.linalg.qr(rng.random((3, 3)))

    R = R2 @ R1.T

    P1 = K @ np.hstack((R1, np.zeros((3, 1))))
    P2 = K @ np.hstack((R2, t))

    # Fundamental matrix
    F = Kinv.T @ skew(t) @ R @ Kinv

    # Generate points with y-coordinate in front of scene
    X = np.vstack((
            6 * rng.random((1, N)) - 3,
            5 * rng.random((1, N)) + 3,
            6 * rng.random((1, N)) - 3,
            np.ones((1, N))))

    # Generate point correspondences (pinhole)
    x1 = pflat(P1 @ X)
    x2 = pflat(P2 @ X)

    # Add radial distortion (if desired)
    x1u = x1
    x2u = x2

    if distortion_param < 0:
        x1 = radialdistort(x1, distortion_param)
        x2 = radialdistort(x2, distortion_param)

    return R1, R2, f, F, x1, x2, R, t, x1u, x2u


def compare_to_gt(sols, f, F, r):
    """Compute relative errors compared to ground truth."""
    F /= F[2, 2]
    normF = np.linalg.norm(F)
    f_err = min([abs(f - sol['f']) / f for sol in sols])
    F_err = min([np.linalg.norm(F - sol['F'] / sol['F'][2, 2]) / normF for sol in sols])
    r_err = min([abs(r - sol['r']) / abs(r) for sol in sols])

    return f_err, F_err, r_err


if __name__ == '__main__':
    # Test a minimal sample
    print('frEfr:')
    N = 4
    distortion_param = -1e-07
    R1, R2, f, F, x1, x2, R, t, x1u, x2u = generate_points_realistic(N, distortion_param)

    print(f'F =\n{F / F[2, 2]}')
    print(f'f = {f}')
    print(f'r = {distortion_param}')
    print(f'x1 =\n{x1[:2, :]}')
    print(f'x2 =\n{x2[:2, :]}')

    use_fast_solver = False
    out = dpl.get_valtonenornhag_arxiv_2021_frEfr(np.asfortranarray(x1[:2, :]), np.asfortranarray(x2[:2, :]),
                                                  np.asfortranarray(R1), np.asfortranarray(R2), use_fast_solver)
    f_err, F_err, r_err = compare_to_gt(out, f, F, distortion_param)

    print(f'Focal length error: {f_err}')
    print(f'Fundamental matrix error: {F_err}')
    print(f'Radial distortion parameter error: {r_err}')
