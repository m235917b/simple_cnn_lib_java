package com.simplecnn.cnn;

/**
 * Cost/loss function interface for the output layer.
 * <p>
 * Since we can not simply compute the derivative of a function, it is helpful for
 * generalizing the use of a function and its derivative interchangeably, to create
 * wrapper classes for each function-derivative pair. Then we can access both via
 * one reference.
 *
 * @author Marvin Bergmann
 */
public interface Cost {
    /**
     * Get the cost of the network for the given input batch
     *
     * @param desired desired outputs for each vector in batch
     * @param output  output batch
     * @return cost
     * @throws IncompatibleDimensionsException if length of desired vectors != length of output vectors
     */
    double apply(double[][] desired, double[][] output) throws IncompatibleDimensionsException;

    /**
     * Apply the vectorized derivative of the loss function
     *
     * @param desired desired output
     * @param output  computed output of the net
     * @return value of the derivative
     * @throws IncompatibleDimensionsException if desired.length != output.length
     */
    double[] applyD(double[] desired, double[] output) throws IncompatibleDimensionsException;
}
