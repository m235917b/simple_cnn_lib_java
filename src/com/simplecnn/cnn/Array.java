package com.simplecnn.cnn;

/**
 * Methods for array/vector operations
 *
 * @author Marvin Bergmann
 */
public class Array {
    /**
     * Multiply a matrix with a vector
     *
     * @param x matrix
     * @param y vector
     * @return x * y
     * @throws IncompatibleDimensionsException if x.length == 0 || width of x != y.length
     */
    public static float[] mul(float[][] x, float[] y) throws IncompatibleDimensionsException {
        if (x.length == 0 || x[0].length != y.length) {
            throw new IncompatibleDimensionsException();
        }

        final float[] out = new float[x.length];

        for (int i = 0; i < x.length; ++i) {
            float sum = 0.f;

            for (int j = 0; j < y.length; ++j) {
                sum += x[i][j] * y[j];
            }

            out[i] = sum;
        }

        return out;
    }

    /**
     * Sum two vectors
     *
     * @param x first vector
     * @param y second vector
     * @return x + y
     * @throws IncompatibleDimensionsException if x.length != y.length
     */
    public static float[] sum(float[] x, float[] y) throws IncompatibleDimensionsException {
        if (x.length != y.length) {
            throw new IncompatibleDimensionsException();
        }

        final float[] out = new float[x.length];

        for (int i = 0; i < x.length; ++i) {
            out[i] = x[i] + y[i];
        }

        return out;
    }

    /**
     * Calculate the sum of two matrices
     *
     * @param x first matrix
     * @param y second matrix
     * @return x + y
     * @throws IncompatibleDimensionsException if width of x != width of y || x.length != y.length
     */
    public static float[][] sum(float[][] x, float[][] y) throws IncompatibleDimensionsException {
        if (x.length == 0 || x.length != y.length || x[0].length != y[0].length) {
            throw new IncompatibleDimensionsException();
        }

        final float[][] out = new float[x.length][x[0].length];

        for (int i = 0; i < x.length; ++i) {
            for (int j = 0; j < x[0].length; ++j) {
                out[i][j] = x[i][j] + y[i][j];
            }
        }

        return out;
    }

    /**
     * Scale a vector by a given factor
     *
     * @param s scale factor
     * @param x vector to scale
     * @return s * x
     */
    public static float[] scale(float s, float[] x) {
        final float[] out = new float[x.length];

        for (int i = 0; i < x.length; ++i) {
            out[i] = s * x[i];
        }

        return out;
    }

    /**
     * Multiply a matrix by a scale factor
     *
     * @param s scale factor
     * @param x matrix
     * @return s * x
     * @throws IncompatibleDimensionsException if x.length == 0
     */
    public static float[][] scale(float s, float[][] x) throws IncompatibleDimensionsException {
        if (x.length == 0) {
            throw new IncompatibleDimensionsException();
        }

        final float[][] out = new float[x.length][x[0].length];

        for (int i = 0; i < x.length; ++i) {
            for (int j = 0; j < x[0].length; ++j) {
                out[i][j] = x[i][j] * s;
            }
        }

        return out;
    }

    /**
     * Calculate the absolut value of a vector
     *
     * @param x vector
     * @return absolut value of x
     */
    public static float abs(float[] x) {
        float out = 0.f;

        for (float v : x) {
            out += Math.pow(v, 2.f);
        }

        return (float) Math.sqrt(out);
    }

    /**
     * Calculate the transpose of a matrix
     *
     * @param x matrix
     * @return transpose of x
     * @throws IncompatibleDimensionsException if x.length == 0
     */
    public static float[][] trans(float[][] x) throws IncompatibleDimensionsException {
        if (x.length == 0) {
            throw new IncompatibleDimensionsException();
        }

        final float[][] out = new float[x[0].length][x.length];

        for (int i = 0; i < x.length; ++i) {
            for (int j = 0; j < x[0].length; ++j) {
                out[j][i] = x[i][j];
            }
        }

        return out;
    }

    /**
     * Calculate the product of a vector with the transpose of another vector.
     *
     * @param a first vector
     * @param b second vector, which will be transposed
     * @return result matrix of a * bT (where bT is the transpose of b)
     */
    public static float[][] axbT(float[] a, float[] b) {
        final float[][] out = new float[a.length][b.length];

        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < b.length; ++j) {
                out[i][j] = a[i] * b[j];
            }
        }

        return out;
    }

    /**
     * Calculate the hadamard product of two vectors
     *
     * @param x first vector
     * @param y second vector
     * @return hadamard product of x and y
     * @throws IncompatibleDimensionsException if x.length != y.length
     */
    public static float[] had(float[] x, float[] y) throws IncompatibleDimensionsException {
        if (x.length != y.length) {
            throw new IncompatibleDimensionsException();
        }

        final float[] out = new float[x.length];

        for (int i = 0; i < x.length; ++i) {
            out[i] = x[i] * y[i];
        }

        return out;
    }

    /**
     * Create a deep copy of an array
     *
     * @param in array
     * @return deep copy of in
     * @throws IncompatibleDimensionsException if in.length == 0
     */
    public static float[][] copy(float[][] in) throws IncompatibleDimensionsException {
        if (in.length == 0) {
            throw new IncompatibleDimensionsException();
        }

        final float[][] out = new float[in.length][in[0].length];

        for (int i = 0; i < in.length; ++i) {
            System.arraycopy(in[i], 0, out[i], 0, in[0].length);
        }

        return out;
    }

    /**
     * Create a deep copy of a vector
     *
     * @param in vector
     * @return deep copy of in
     */
    public static float[] copy(float[] in) {
        final float[] out = new float[in.length];

        System.arraycopy(in, 0, out, 0, in.length);

        return out;
    }
}
