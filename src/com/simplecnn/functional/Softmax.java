package com.simplecnn.functional;

import com.simplecnn.cnn.Array;

/**
 * The softmax activation function and its derivative
 *
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class Softmax implements Activation {
    @Override
    public double[] apply(double[] x) {
        return Array.map(x, e -> Math.exp(e) / Array.sum(Array.exp(x)));
    }

    @Override
    public double[] applyD(double[] x) {
        // (exp(x) % (denominator - exp(x)) / denominator² and "%" is the hadamard product
        return Array.map(x, e ->
                (Array.sum(Array.exp(x)) - Math.exp(e)) / Math.pow(Array.sum(Array.exp(x)), 2.));
    }
}
