package com.simplecnn.functional;

import com.simplecnn.cnn.Array;

/**
 * Squared error cost function and its derivative
 *
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class Squared implements Cost {
    @Override
    public double apply(double[][] desired, double[][] input)
            throws IncompatibleDimensionsException {
        return desired.length == 0
                ? 0.
                : Array.sum(Array.sqr(Array.sub(desired, input))) / (desired.length * desired[0].length);
    }

    @Override
    public double[] applyD(double[] desired, double[] output) {
        try {
            return Array.sub(output, desired);
        } catch (IncompatibleDimensionsException e) {
            throw new RuntimeException(e);
        }
    }
}
