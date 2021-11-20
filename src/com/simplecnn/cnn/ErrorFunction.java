package com.simplecnn.cnn;

/**
 * Error function interface for the output layer.
 * <p>
 * Since we can not simply compute the derivative of a function, it is helpful for
 * generalizing the use of a function and its derivative interchangeably, to create
 * wrapper classes for each function-derivative pair. Then we can access both via
 * one reference.
 *
 * @author Marvin Bergmann
 */
public interface ErrorFunction {
    /**
     * Get the error of the network for the given input batch
     *
     * @param desired desired outputs for each vector in batch
     * @param output  output batch
     * @return error
     * @throws IncompatibleDimensionsException if length of desired vectors != length of output vectors
     */
    float apply(float[][] desired, float[][] output) throws IncompatibleDimensionsException;

    /**
     * Apply the vectorized derivative of the error function
     *
     * @param desired desired output
     * @param output  computed output of the net
     * @return value of the derivative
     * @throws IncompatibleDimensionsException if desired.length != output.length
     */
    float[] applyD(float[] desired, float[] output) throws IncompatibleDimensionsException;
}
