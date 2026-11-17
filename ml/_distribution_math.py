# Numerical implementation extracted unchanged from the embedded
# analytic_moments_fast.py helper in R08_compact_distribution_critic.ipynb.
# Keep its fixed beta/copula/clamping/quadrature policy in sync when reviewing changes.
"""Fixed moment-matched source-cell reference estimators.

Both candidates consume exactly the same decoded five-byte source-cell moment
payload and query intersections. They never receive raw points, labels, raw2
counts, fitted models or the candidate ARM answer. The reference is a structural
distribution approximation, not a confidence interval or an exact answer.
Beta moment matching and Gaussian copulas are classical components; no novelty
or effectiveness claim follows from this implementation alone.
"""
import numpy as np
from scipy.special import betainc, ndtri
VARIANTS = ('beta_independent', 'beta_gaussian_copula')
MOMENT_COLUMNS = ('mean_x', 'mean_y', 'variance_x', 'variance_y', 'covariance_xy')
EPSILON = 1e-06
RHO_LIMIT = 0.995
QUADRATURE_ORDER = 16
CDF_BLOCK_SIZE = 2048
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(QUADRATURE_ORDER)

def _validated_inputs(rectangles, moments):
    rectangles = np.asarray(rectangles, float)
    moments = np.asarray(moments, float)
    if rectangles.ndim != 2 or rectangles.shape[1] != 4 or moments.shape != (len(rectangles), 5):
        raise ValueError('Expected Nx4 local rectangles and Nx5 decoded moments')
    if not np.isfinite(rectangles).all() or not np.isfinite(moments).all():
        raise ValueError('Nonfinite local geometry or moments')
    if ((rectangles < 0) | (rectangles > 1)).any() or (rectangles[:, 0] > rectangles[:, 1]).any() or (rectangles[:, 2] > rectangles[:, 3]).any():
        raise ValueError('Query intersections must be closed subrectangles of[0,1]^2')
    if ((moments[:, :2] < 0) | (moments[:, :2] > 1)).any() or (moments[:, 2:4] < 0).any():
        raise ValueError('Decoded means must lie in[0,1]; variances must be nonnegative')
    return (rectangles, moments)

def _marginal_limits(lo, hi, mean, variance):
    """Return F(lo-) and F(hi), including exact Dirac-axis handling.

    Quantization can yield an inconsistent mean/variance pair. The fixed policy
    clamps nondegenerate means to[eps,1-eps] and variances to the largest valid
    beta moment, multiplied by1-eps. It never uses labels to choose a clamp.
    A decoded zero variance is represented by a point mass at the decoded mean.
    """
    point = variance == 0
    lower = np.empty(len(lo), float)
    upper = np.empty(len(lo), float)
    lower[point] = (lo[point] > mean[point]).astype(float)
    upper[point] = (hi[point] >= mean[point]).astype(float)
    continuous = ~point
    if continuous.any():
        m = np.clip(mean[continuous], EPSILON, 1 - EPSILON)
        v = np.minimum(variance[continuous], m * (1 - m) * (1 - EPSILON))
        k = m * (1 - m) / v - 1.0
        alpha = m * k
        beta = (1 - m) * k
        lower[continuous] = betainc(alpha, beta, lo[continuous])
        upper[continuous] = betainc(alpha, beta, hi[continuous])
    if not np.isfinite(np.r_[lower, upper]).all():
        raise ValueError('Moment-matched beta produced a nonfinite marginal CDF')
    return (np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0))

def gaussian_copula_cdf(u, v, rho):
    """Deterministic Gaussian-copula CDF using a fixed16-point integral.

    C(u,v;r)=uv+integral[0,r] phi2(ndtri(u),ndtri(v);t)dt.
    The substitution t=sin(theta) removes1/sqrt(1-t²), so the fixed16-point
    rule integrates over[0,asin(r)] and stays accurate near the rho clamp.
    Endpoint marginals are handled exactly before inverse normal evaluation.
    No scipy.stats multivariate CDF or randomized numerical integration is used.
    Interior CDF rows are processed in blocks of at most2048 by16 quadrature
    nodes. This changes array execution only, not the fixed numerical rule.
    """
    u, v, rho = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(rho, float))
    original_shape = u.shape
    u = u.reshape(-1)
    v = v.reshape(-1)
    rho = rho.reshape(-1)
    if not np.isfinite(np.r_[u, v, rho]).all() or ((u < 0) | (u > 1) | (v < 0) | (v > 1)).any():
        raise ValueError('Copula probabilities must be finite in[0,1]')
    rho = np.clip(rho, -RHO_LIMIT, RHO_LIMIT)
    result = u * v
    interior = (u > 0) & (u < 1) & (v > 0) & (v < 1) & (rho != 0)
    if interior.any():
        h = ndtri(u[interior])
        k = ndtri(v[interior])
        angle = np.arcsin(rho[interior])
        integral = np.empty(len(h), float)
        for start in range(0, len(h), CDF_BLOCK_SIZE):
            stop = min(start + CDF_BLOCK_SIZE, len(h))
            hs = h[start:stop, None]
            ks = k[start:stop, None]
            t = np.sin(0.5 * angle[start:stop, None] * (_GL_NODES[None, :] + 1.0))
            one_minus = 1.0 - t * t
            exponent = -0.5 * ((hs - ks * t) ** 2 / one_minus + ks * ks)
            integral[start:stop] = np.sum(np.exp(exponent) * _GL_WEIGHTS[None, :], axis=1)
        result[interior] += 0.5 * angle * integral / (2.0 * np.pi)
    upper = np.minimum(u, v)
    lower = np.minimum(np.maximum(u + v - 1.0, 0.0), upper)
    result = np.maximum(np.minimum(result, upper), lower)
    return result.reshape(original_shape)

def boundary_probability(rectangles, moments, variant='beta_independent'):
    """Approximate local rectangle probability from decoded five moments."""
    if variant not in VARIANTS:
        raise ValueError('Unknown fixed analytic moment variant')
    rectangles, moments = _validated_inputs(rectangles, moments)
    if not len(rectangles):
        return np.empty(0, float)
    lx, ux = _marginal_limits(rectangles[:, 0], rectangles[:, 1], moments[:, 0], moments[:, 2])
    ly, uy = _marginal_limits(rectangles[:, 2], rectangles[:, 3], moments[:, 1], moments[:, 3])
    if variant == 'beta_independent':
        probability = (ux - lx) * (uy - ly)
    else:
        denominator = np.sqrt(moments[:, 2] * moments[:, 3])
        rho = np.divide(moments[:, 4], denominator, out=np.zeros(len(moments)), where=denominator > 0)
        rho = np.clip(rho, -RHO_LIMIT, RHO_LIMIT)
        corners = gaussian_copula_cdf(np.concatenate((ux, lx, ux, lx)), np.concatenate((uy, uy, ly, ly)), np.tile(rho, 4)).reshape(4, len(rho))
        probability = corners[0] - corners[1] - corners[2] + corners[3]
    if not np.isfinite(probability).all():
        raise ValueError('Nonfinite analytic moment probability')
    return np.clip(probability, 0.0, 1.0)

def estimate_reference(rectangles, moments, mass, query_index, L_full, variant='beta_independent'):
    """Aggregate exact fully contained mass plus estimated boundary mass.

    Only source/query metadata is an argument. query_index selects the output
    query for each partial-cell row; it is not an identity feature or a label.
    """
    rectangles, moments = _validated_inputs(rectangles, moments)
    mass = np.asarray(mass, float)
    query_index = np.asarray(query_index)
    full = np.asarray(L_full, float)
    if mass.shape != (len(rectangles),) or query_index.shape != (len(rectangles),) or full.ndim != 1 or (not np.isfinite(mass).all()) or (not np.isfinite(full).all()) or (mass < 0).any() or (full < 0).any():
        raise ValueError('Invalid partial masses, query indices or fully-contained masses')
    if not np.issubdtype(query_index.dtype, np.integer) or (query_index < 0).any() or (query_index >= len(full)).any():
        raise ValueError('Partial row query index outside the reference-count output')
    p = boundary_probability(rectangles, moments, variant)
    return full + np.bincount(query_index, weights=mass * p, minlength=len(full))

def protocol():
    return dict(variants=list(VARIANTS), moment_columns=list(MOMENT_COLUMNS), mean_clamp_epsilon=EPSILON, variance_policy='zero=>decoded point mass; positive=>min(v,m*(1-m)*(1-eps)) after mean clamp', covariance_policy='rho=cov/sqrt(vx*vy), clipped to[-.995,.995]; zero variance=>rho0', gaussian_copula_integration='fixed16-point Gauss-Legendre Plackett integral with t=sin(theta) substitution; exact probability endpoints', quadrature_order=QUADRATURE_ORDER, probability_clipping='CDF Fréchet bounds, then final rectangle[0,1]', execution='blocked2048 CDF rows x16 quadrature nodes; all4 rectangle corners in one CDF call', source_model='moment-matched beta marginals; independent or fixed Gaussian copula', quantization='caller supplies round-trip decoded mu/255, variance/1020, covariance/508 five-byte moments', training=False, labels_at_inference=False, raw2_at_inference=False, H_at_reference_inference=False, confidence_or_accuracy_guarantee=False, novelty_claim=False)
