// Execution-only scalar implementation of the frozen R08 numerical policy.
// Special functions are supplied by this process's SciPy Cython C API.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {
using Beta = double (*)(double, double, double, int);
using Ndtri = double (*)(double, int);
Beta beta_cdf = nullptr;
Ndtri inverse_normal = nullptr;
double nodes[16], weights[16];
constexpr double epsilon = 1e-6;
constexpr double rho_limit = .995;
constexpr double pi = 3.141592653589793238462643383279502884;

inline double clip(double x, double low=0., double high=1.) {
    return std::max(low, std::min(x, high));
}
inline double f32(double x) { return static_cast<double>(static_cast<float>(clip(x))); }

void marginal(double low, double high, double mean, double variance, double &lower, double &upper) {
    if (variance == 0.) {
        lower = low > mean ? 1. : 0.;
        upper = high >= mean ? 1. : 0.;
    } else {
        const double m = clip(mean, epsilon, 1.-epsilon);
        const double v = std::min(variance, m*(1.-m)*(1.-epsilon));
        const double k = m*(1.-m)/v-1.;
        lower = clip(beta_cdf(m*k, (1.-m)*k, low, 1));
        upper = clip(beta_cdf(m*k, (1.-m)*k, high, 1));
    }
}

double copula(double u, double v, double rho) {
    double result = u*v;
    if (u > 0. && u < 1. && v > 0. && v < 1. && rho != 0.) {
        const double h = inverse_normal(u, 1), k = inverse_normal(v, 1);
        const double angle = std::asin(rho);
        double terms[16];
        for (int i=0; i<16; ++i) {
            const double t = std::sin(.5*angle*(nodes[i]+1.));
            const double one_minus = 1.-t*t;
            const double d = h-k*t;
            const double exponent = -.5*(d*d/one_minus+k*k);
            terms[i] = std::exp(exponent)*weights[i];
        }
        // NumPy's pairwise reduction for a contiguous sixteen-element row.
        double partial[8];
        for (int i=0; i<8; ++i) partial[i] = terms[i]+terms[i+8];
        const double integral = ((partial[0]+partial[1])+(partial[2]+partial[3]))
                              + ((partial[4]+partial[5])+(partial[6]+partial[7]));
        result += .5*angle*integral/(2.*pi);
    }
    const double upper = std::min(u,v);
    const double lower = std::min(std::max(u+v-1.,0.),upper);
    return std::max(std::min(result,upper),lower);
}

double probability(const double *r, const double *m) {
    double lx, ux, ly, uy;
    marginal(r[0],r[1],m[0],m[2],lx,ux);
    marginal(r[2],r[3],m[1],m[3],ly,uy);
    const double denominator = std::sqrt(m[2]*m[3]);
    const double rho = clip(denominator > 0. ? m[4]/denominator : 0.,-rho_limit,rho_limit);
    return clip(copula(ux,uy,rho)-copula(lx,uy,rho)-copula(ux,ly,rho)+copula(lx,ly,rho));
}

double children(const double *r, const double *counts, const double *moments, bool uniform) {
    double result = 0.;
    for (int j=0; j<4; ++j) {
        const double x0 = j >= 2 ? .5 : 0., x1 = x0+.5;
        const double y0 = j%2 ? .5 : 0., y1 = y0+.5;
        if (counts[j] == 0.) continue;
        const bool full = r[0]<=x0 && r[1]>=x1 && r[2]<=y0 && r[3]>=y1;
        double p = full ? 1. : 0.;
        const double lo_x=std::max(r[0],x0), hi_x=std::min(r[1],x1);
        const double lo_y=std::max(r[2],y0), hi_y=std::min(r[3],y1);
        if (!full && lo_x<hi_x && lo_y<hi_y) {
            const double child[4] = {f32((lo_x-x0)*2.),f32((hi_x-x0)*2.),
                                     f32((lo_y-y0)*2.),f32((hi_y-y0)*2.)};
            p = uniform ? (child[1]-child[0])*(child[3]-child[2]) : probability(child,moments+5*j);
        }
        result += counts[j]*p;
    }
    return result;
}
} // namespace

extern "C" int shape_kernel_init(void *beta, void *ndtri, const double *gl_nodes, const double *gl_weights) {
    if (!beta || !ndtri || !gl_nodes || !gl_weights) return 1;
    beta_cdf = reinterpret_cast<Beta>(beta);
    inverse_normal = reinterpret_cast<Ndtri>(ndtri);
    for (int i=0; i<16; ++i) { nodes[i]=gl_nodes[i]; weights[i]=gl_weights[i]; }
    return 0;
}

extern "C" int shape_kernel_batch(std::size_t n, const double *bounds, const std::uint64_t *mass,
    const double *moments, const std::int32_t *selected, const double *child_counts,
    const double *child_moments, std::size_t qn, const double *queries, int uniform, double *output) {
    if (!beta_cdf || !inverse_normal) return 1;
    for (std::size_t qi=0; qi<qn; ++qi) {
        const double *q=queries+4*qi;
        std::uint64_t full_mass=0;
        double boundary=0.;
        for (std::size_t i=0; i<n; ++i) {
            const double *b=bounds+4*i;
            if (q[0]>b[1] || q[1]<b[0] || q[2]>b[3] || q[3]<b[2]) continue;
            if (q[0]<=b[0] && q[1]>=b[1] && q[2]<=b[2] && q[3]>=b[3]) {
                full_mass += mass[i]; continue;
            }
            const double r[4] = {f32((q[0]-b[0])/(b[1]-b[0])),f32((q[1]-b[0])/(b[1]-b[0])),
                                 f32((q[2]-b[2])/(b[3]-b[2])),f32((q[3]-b[2])/(b[3]-b[2]))};
            const auto s=selected[i];
            boundary += s<0 ? static_cast<double>(mass[i])*probability(r,moments+5*i)
                            : children(r,child_counts+4*s,child_moments+20*s,uniform!=0);
        }
        output[qi]=static_cast<double>(full_mass)+boundary;
        if (!std::isfinite(output[qi])) return 2;
    }
    return 0;
}
