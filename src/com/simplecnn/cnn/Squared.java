package com.simplecnn.cnn;

/**
 * Squared error function and its derivative
 *
 * @author Marvin Bergmann
 */
public class Squared implements ErrorFunction {
    @Override
    public float apply(float[][] desired, float[][] input) throws IncompatibleDimensionsException {
        if (desired.length == 0) {
            return 0.f;
        }

        return Array.sum(Array.sqr(Array.sub(desired, input))) / (desired.length * desired[0].length);
    }

    @Override
    public float[] applyD(float[] desired, float[] output) throws IncompatibleDimensionsException {
        return Array.sub(output, desired);
    }
}
