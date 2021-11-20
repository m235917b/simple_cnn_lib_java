package com.simplecnn.cnn;

/**
 * Squared error cost function and its derivative
 *
 * @author Marvin Bergmann
 */
public class Squared implements Cost {
    @Override
    public double apply(double[][] desired, double[][] input) throws IncompatibleDimensionsException {
        return desired.length == 0
                ? 0.
                : Array.sum(Array.sqr(Array.sub(desired, input))) / (desired.length * desired[0].length);
    }

    @Override
    public double[] applyD(double[] desired, double[] output) throws IncompatibleDimensionsException {
        return Array.sub(output, desired);
    }
}
