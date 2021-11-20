package com.simplecnn.functional;

import com.simplecnn.cnn.Array;

/**
 * The sigmoid activation function and its derivative
 *
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class Sigmoid implements Activation {
    @Override
    public double[] apply(double[] x) {
        return Array.map(x, e -> 1. / (1. + Math.exp(-e)));
    }

    @Override
    public double[] applyD(double[] x) {
        return Array.map(x, e -> Math.exp(-e) / Math.pow(1. + Math.exp(-e), 2.));
    }
}
