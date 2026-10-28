package benchmark;

import histogram.SimpleSpatialHistogramOpt;
import org.apache.commons.math3.distribution.BetaDistribution;
import utils.UtilsFunctionHistogram;

/**
 * Small JVM boundary for the latency benchmark. Count arrays use [x][y].
 * Query arguments use xmin,ymin,xmax,ymax; the repository API uses a different order.
 * No histogram construction, merging, or estimation algorithm is replaced here.
 */
public final class HistogramBridge {
    private final SimpleSpatialHistogramOpt histogram;
    private final double xmin;
    private final double ymin;
    private final double xmax;
    private final double ymax;
    private BetaDistribution beta;

    public HistogramBridge(double[] xEdges, double[] yEdges, double[][] counts) {
        validateAxis(xEdges, "xEdges");
        validateAxis(yEdges, "yEdges");
        int nx = xEdges.length - 1;
        int ny = yEdges.length - 1;
        if (counts == null || counts.length != nx) {
            throw new IllegalArgumentException("counts must have xEdges.length-1 rows");
        }
        double[][] copied = new double[nx][ny];
        for (int i = 0; i < nx; i++) {
            if (counts[i] == null || counts[i].length != ny) {
                throw new IllegalArgumentException("counts rows must have yEdges.length-1 entries");
            }
            for (int j = 0; j < ny; j++) {
                if (!Double.isFinite(counts[i][j]) || counts[i][j] < 0) {
                    throw new IllegalArgumentException("counts must be finite and nonnegative");
                }
                copied[i][j] = counts[i][j];
            }
        }
        xmin = xEdges[0];
        ymin = yEdges[0];
        xmax = xEdges[nx];
        ymax = yEdges[ny];
        histogram = new SimpleSpatialHistogramOpt(xmin, ymin, xmax, ymax, nx, ny,
                copied, xEdges.clone(), yEdges.clone());
    }

    public static HistogramBridge create(double[] xEdges, double[] yEdges, double[][] counts) {
        return new HistogramBridge(xEdges, yEdges, counts);
    }

    private static void validateAxis(double[] edges, String name) {
        if (edges == null || edges.length < 2) {
            throw new IllegalArgumentException(name + " must contain at least two edges");
        }
        for (int i = 0; i < edges.length; i++) {
            if (!Double.isFinite(edges[i]) || (i > 0 && edges[i] <= edges[i - 1])) {
                throw new IllegalArgumentException(name + " must be finite and strictly increasing");
            }
        }
    }

    private static void validateQuery(double x0, double y0, double x1, double y1) {
        if (!Double.isFinite(x0) || !Double.isFinite(y0) || !Double.isFinite(x1)
                || !Double.isFinite(y1) || x1 <= x0 || y1 <= y0) {
            throw new IllegalArgumentException("query must be finite with xmin<xmax and ymin<ymax");
        }
    }

    private boolean outside(double x0, double y0, double x1, double y1) {
        return x0 >= xmax || x1 <= xmin || y0 >= ymax || y1 <= ymin;
    }

    public double estimate(double x0, double y0, double x1, double y1) {
        validateQuery(x0, y0, x1, y1);
        // A histogram assigns no mass to zero-area edge contact. This also avoids
        // the existing estimator's max-domain-edge index overrun.
        if (outside(x0, y0, x1, y1)) {
            return 0;
        }
        return histogram.RangeQuery(Math.max(xmin, x0), Math.min(xmax, x1),
                Math.max(ymin, y0), Math.min(ymax, y1));
    }

    /** Histogram-only initialization, recorded separately from per-query scoring. */
    public void initSkewness(double scale, double epsilon) {
        if (!Double.isFinite(scale) || scale <= 0 || !Double.isFinite(epsilon) || epsilon <= 0) {
            throw new IllegalArgumentException("Skewness scale and epsilon must be finite and positive");
        }
        // ComputeBetaDist in this checkout uses nx rather than ny as its density
        // vector stride. Preserve core code and explicitly reject affected shapes.
        if (histogram.numLonBucket != histogram.numLatBucket) {
            throw new IllegalStateException("Skewness scoring requires a square grid in this checkout: "
                    + "UtilsFunctionHistogram.ComputeBetaDist has a rectangular-grid density indexing defect");
        }
        beta = UtilsFunctionHistogram.ComputeBetaDist(histogram.getData(),
                histogram.getLonBoundary(), histogram.getLatBoundary(), scale, epsilon);
    }

    public double score(double x0, double y0, double x1, double y1) {
        validateQuery(x0, y0, x1, y1);
        if (beta == null) {
            throw new IllegalStateException("Call initSkewness before scoring");
        }
        if (outside(x0, y0, x1, y1)) {
            return 0;
        }
        return histogram.RangeQueryWithRatio(beta, Math.max(xmin, x0), Math.min(xmax, x1),
                Math.max(ymin, y0), Math.min(ymax, y1))[2];
    }

    /** Existing Java histogram for team feature adapters. Treat the object as read-only. */
    public SimpleSpatialHistogramOpt getHistogram() {
        return histogram;
    }
}
