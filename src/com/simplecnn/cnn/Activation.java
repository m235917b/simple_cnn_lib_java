package com.simplecnn.cnn;

/**
 * Activation function interface for neural net layers.
 * <p>
 * Since we can not simply compute the derivative of a function, it is helpful for
 * generalizing the use of a function and its derivative interchangeably, to create
 * wrapper classes for each function-derivative pair. Then we can access both via
 * one reference.
 *
 * @author Marvin Bergmann
 */
public interface Activation {
    /**
     * Apply the activation function
     *
     * @param x argument
     * @return function value
     */
    float[] apply(float[] x);

    /**
     * Apply the derivative of the activation function
     *
     * @param x argument
     * @return value of the derivative
     */
    float[] applyD(float[] x);
}
