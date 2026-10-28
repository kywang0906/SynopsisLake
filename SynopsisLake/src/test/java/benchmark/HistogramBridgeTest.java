package benchmark;

import org.junit.Test;
import static org.junit.Assert.*;

public class HistogramBridgeTest {
    private static HistogramBridge square() {
        return HistogramBridge.create(new double[]{0, 1, 2}, new double[]{0, 1, 2},
                new double[][]{{4, 8}, {12, 16}});
    }

    @Test
    public void estimatesWholeAndPartialUniformCells() {
        HistogramBridge bridge = square();
        assertEquals(40, bridge.estimate(0, 0, 2, 2), 1e-12);
        assertEquals(1, bridge.estimate(0, 0, 0.5, 0.5), 1e-12);
        assertEquals(10, bridge.estimate(0.5, 0.5, 1.5, 1.5), 1e-12);
    }

    @Test
    public void handlesNonuniformRectangularGrid() {
        HistogramBridge bridge = HistogramBridge.create(new double[]{0, 1, 3},
                new double[]{0, 2, 3, 5}, new double[][]{{2, 4, 6}, {8, 10, 12}});
        assertEquals(42, bridge.estimate(-1, -1, 4, 6), 1e-12);
        assertEquals(4, bridge.estimate(1, 0, 2, 2), 1e-12);
        assertEquals(6, bridge.estimate(0.5, 1, 2, 2.5), 1e-12);
    }

    @Test
    public void handlesOutsideAndAllDomainEdgeContacts() {
        HistogramBridge bridge = square();
        assertEquals(0, bridge.estimate(3, 3, 4, 4), 0);
        assertEquals(0, bridge.estimate(2, 0, 3, 1), 0);
        assertEquals(0, bridge.estimate(-1, 0, 0, 1), 0);
        assertEquals(0, bridge.estimate(0, 2, 1, 3), 0);
        assertEquals(0, bridge.estimate(0, -1, 1, 0), 0);
        assertEquals(1, bridge.estimate(-1, -1, 0.5, 0.5), 1e-12);
    }

    @Test
    public void scoreUsesExistingSquareGridSkewness() {
        HistogramBridge bridge = square();
        bridge.initSkewness(1, 1e-6);
        double score = bridge.score(0.1, 0.2, 1.6, 1.8);
        assertTrue(Double.isFinite(score));
        assertTrue(score >= 0);
        assertEquals(0, bridge.score(2, 0, 3, 1), 0);
        assertEquals(0, bridge.score(0, 2, 1, 3), 0);
    }

    @Test(expected = IllegalStateException.class)
    public void refusesRectangularSkewnessRatherThanSilentlyChangingCore() {
        HistogramBridge.create(new double[]{0, 1, 2}, new double[]{0, 1, 2, 3},
                new double[][]{{1, 2, 3}, {4, 5, 6}}).initSkewness(1, 1e-6);
    }

    @Test(expected = IllegalStateException.class)
    public void scoringRequiresExplicitInitialization() {
        square().score(0, 0, 1, 1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsUnorderedEdges() {
        HistogramBridge.create(new double[]{0, 2, 1}, new double[]{0, 1},
                new double[][]{{1}, {2}});
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsShapeMismatch() {
        HistogramBridge.create(new double[]{0, 1}, new double[]{0, 1, 2}, new double[][]{{1}});
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNegativeCounts() {
        HistogramBridge.create(new double[]{0, 1}, new double[]{0, 1}, new double[][]{{-1}});
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNonfiniteCounts() {
        HistogramBridge.create(new double[]{0, 1}, new double[]{0, 1}, new double[][]{{Double.NaN}});
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsEmptyQuery() {
        square().estimate(0, 0, 0, 1);
    }
}
