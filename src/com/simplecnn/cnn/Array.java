package com.simplecnn.cnn;

import com.simplecnn.functional.IncompatibleDimensionsException;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

/**
 * Methods for array/vector operations
 *
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class Array {
    /**
     * Add two vectors
     *
     * @param x first vector
     * @param y second vector
     * @return x + y
     * @throws IncompatibleDimensionsException if x.length != y.length
     */
    public static double[] add(double[] x, double[] y) throws IncompatibleDimensionsException {
        if (x.length != y.length) {
            throw new IncompatibleDimensionsException();
        }

        return IntStream.range(0, x.length).mapToDouble(i -> x[i] + y[i]).toArray();
    }

    /**
     * Add two matrices
     *
     * @param x first matrix
     * @param y second matrix
     * @return x + y
     * @throws IncompatibleDimensionsException if width of x != width of y || x.length != y.length
     */
    public static double[][] add(double[][] x, double[][] y) throws IncompatibleDimensionsException {
        if (x.length == 0 || x.length != y.length || x[0].length != y[0].length) {
            throw new IncompatibleDimensionsException();
        }

        return IntStream
                .range(0, x.length)
                .mapToObj(i -> IntStream
                        .range(0, x[i].length)
                        .mapToDouble(j -> x[i][j] + y[i][j])
                        .toArray())
                .toArray(double[][]::new);
    }

    /**
     * Subtract two vectors
     *
     * @param x first vector
     * @param y second vector
     * @return x - y
     * @throws IncompatibleDimensionsException if x.length != y.length
     */
    public static double[] sub(double[] x, double[] y) throws IncompatibleDimensionsException {
        if (x.length != y.length) {
            throw new IncompatibleDimensionsException();
        }

        return IntStream.range(0, x.length).mapToDouble(i -> x[i] - y[i]).toArray();
    }

    /**
     * Subtract two matrices
     *
     * @param x first matrix
     * @param y second matrix
     * @return x - y
     * @throws IncompatibleDimensionsException if width of x != width of y || x.length != y.length
     */
    public static double[][] sub(double[][] x, double[][] y) throws IncompatibleDimensionsException {
        if (x.length == 0 || x.length != y.length || x[0].length != y[0].length) {
            throw new IncompatibleDimensionsException();
        }

        return IntStream
                .range(0, x.length)
                .mapToObj(i -> IntStream
                        .range(0, x[i].length)
                        .mapToDouble(j -> x[i][j] - y[i][j])
                        .toArray())
                .toArray(double[][]::new);
    }

    /**
     * Calculate the hadamard product of two vectors
     *
     * @param x first vector
     * @param y second vector
     * @return hadamard product of x and y
     * @throws IncompatibleDimensionsException if x.length != y.length
     */
    public static double[] had(double[] x, double[] y) throws IncompatibleDimensionsException {
        if (x.length != y.length) {
            throw new IncompatibleDimensionsException();
        }

        return IntStream.range(0, x.length).mapToDouble(i -> x[i] * y[i]).toArray();
    }

    /**
     * Calculate the hadamard product of two matrices
     *
     * @param x first matrix
     * @param y second matrix
     * @return hadamard product of x and y
     * @throws IncompatibleDimensionsException if width of x != width of y || x.length != y.length
     */
    public static double[][] had(double[][] x, double[][] y) throws IncompatibleDimensionsException {
        if (x.length == 0 || x.length != y.length || x[0].length != y[0].length) {
            throw new IncompatibleDimensionsException();
        }

        return IntStream
                .range(0, x.length)
                .mapToObj(i -> IntStream
                        .range(0, x[i].length)
                        .mapToDouble(j -> x[i][j] * y[i][j])
                        .toArray())
                .toArray(double[][]::new);
    }

    /**
     * Performs a component-wise division of one vector by another
     *
     * @param x dividend
     * @param y divisor
     * @return x / y (component-wise as a vector)
     * @throws IncompatibleDimensionsException if x.length != y.length
     */
    public static double[] div(double[] x, double[] y) throws IncompatibleDimensionsException {
        if (x.length != y.length) {
            throw new IncompatibleDimensionsException();
        }

        return IntStream.range(0, x.length).mapToDouble(i -> x[i] / y[i]).toArray();
    }

    /**
     * Calculate the product of a vector with the transpose of another vector.
     *
     * @param a first vector
     * @param b second vector, which will be transposed
     * @return result matrix of a * bT (where bT is the transpose of b)
     */
    public static double[][] axbT(double[] a, double[] b) {
        return Arrays
                .stream(a)
                .mapToObj(row -> Arrays.stream(b)
                        .map(e -> row * e)
                        .toArray())
                .toArray(double[][]::new);
    }

    /**
     * Multiply a matrix with a vector
     *
     * @param x matrix
     * @param y vector
     * @return x * y
     * @throws IncompatibleDimensionsException x.length == 0 || width of x != y.length
     */
    public static double[] mul(double[][] x, double[] y) throws IncompatibleDimensionsException {
        if (x.length == 0 || x[0].length != y.length) {
            throw new IncompatibleDimensionsException();
        }

        return Arrays
                .stream(x)
                .mapToDouble(row -> IntStream
                        .range(0, y.length)
                        .mapToDouble(j -> row[j] * y[j])
                        .sum())
                .toArray();
    }

    /**
     * Scale a vector by a given factor
     *
     * @param s scale factor
     * @param x vector to scale
     * @return s * x
     */
    public static double[] scale(double s, double[] x) {
        return map(x, e -> s * e);
    }

    /**
     * Multiply a matrix by a scale factor
     *
     * @param s scale factor
     * @param x matrix
     * @return s * x
     */
    public static double[][] scale(double s, double[][] x) {
        return map(x, e -> s * e);
    }

    /**
     * Calculate the absolut value of a vector
     *
     * @param x vector
     * @return absolut value of x
     */
    public static double abs(double[] x) {
        return Math.sqrt(sum(map(x, e -> Math.pow(e, 2.))));
    }

    /**
     * Calculate the transpose of a matrix
     *
     * @param x matrix
     * @return transpose of x
     */
    public static double[][] trans(double[][] x) {
        return x.length == 0
                ? new double[0][0]
                : IntStream
                .range(0, x[0].length)
                .mapToObj(i -> Arrays
                        .stream(x)
                        .mapToDouble(col -> col[i])
                        .toArray())
                .toArray(double[][]::new);
    }

    /**
     * Sum over all entries of a vector
     *
     * @param x matrix
     * @return sum
     */
    public static double sum(double[] x) {
        return Arrays.stream(x).sum();
    }

    /**
     * Sum over all entries of a matrix
     *
     * @param x matrix
     * @return sum
     */
    public static double sum(double[][] x) {
        return Arrays.stream(x).mapToDouble(Array::sum).sum();
    }

    /**
     * Map every entry of a vector with a function
     *
     * @param x vector
     * @param f mapping function
     * @return f(x) (vectorized)
     */
    public static double[] map(double[] x, DoubleUnaryOperator f) {
        return Arrays.stream(x).map(f).toArray();
    }

    /**
     * Map every entry of a matrix with a function
     *
     * @param x matrix
     * @param f mapping function
     * @return f(x) (component-wise as a matrix)
     */
    public static double[][] map(double[][] x, DoubleUnaryOperator f) {
        return Arrays.stream(x).map(row -> map(row, f)).toArray(double[][]::new);
    }

    /**
     * Vectorized natural logarithm
     *
     * @param x input vector
     * @return log(x) (component-wise as a vector)
     */
    public static double[] log(double[] x) {
        return map(x, Math::log);
    }

    /**
     * Component-wise natural logarithm of a matrix
     *
     * @param x matrix
     * @return log(x) (component-wise as a matrix)
     */
    public static double[][] log(double[][] x) {
        return map(x, Math::log);
    }

    /**
     * Vectorized exp function
     *
     * @param x input vector
     * @return exp(x) (vectorized)
     */
    public static double[] exp(double[] x) {
        return map(x, Math::exp);
    }

    /**
     * Component-wise exp of a matrix
     *
     * @param x input matrix
     * @return exp(x) (component-wise as a matrix)
     */
    public static double[][] exp(double[][] x) {
        return map(x, Math::exp);
    }

    /**
     * Vectorized square function
     *
     * @param x input vector
     * @return x² (component-wise as a vector)
     */
    public static double[] sqr(double[] x) {
        return map(x, e -> Math.pow(e, 2.));
    }

    /**
     * Component-wise square of a matrix
     *
     * @param x input matrix
     * @return x² (component-wise as a matrix)
     */
    public static double[][] sqr(double[][] x) {
        return map(x, e -> Math.pow(e, 2.));
    }

    /**
     * Create a deep copy of a vector
     *
     * @param in vector
     * @return deep copy of in
     */
    public static double[] copy(double[] in) {
        return Arrays.stream(in).toArray();
    }

    /**
     * Create a deep copy of an array
     *
     * @param in array
     * @return deep copy of in
     */
    public static double[][] copy(double[][] in) {
        return Arrays.stream(in).map(Array::copy).toArray(double[][]::new);
    }
}
