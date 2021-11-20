package com.simplecnn.functional;

import com.simplecnn.cnn.Array;

/**
 * Cross-entropy cost function and its derivative
 *
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class CrossEnt implements Cost {
    @Override
    public double apply(double[][] desired, double[][] output)
            throws IncompatibleDimensionsException {
        // Calculate mean of entries from  (desired - 1) % log(1 - output) - desired % log(output)
        // ("%" is the hadamard product)
        return Array.sum(
                Array.sub(
                        Array.had(
                                Array.map(desired, e -> e - 1.),
                                Array.map(output, e -> Math.log(1. - e))
                        ),
                        Array.had(
                                desired,
                                Array.log(output)
                        )
                )
        ) / (desired.length * desired[0].length);
    }

    @Override
    public double[] applyD(double[] desired, double[] output) {
        try {
            return Array.div(Array.map(desired, e -> 1. - 2. * e), output);
        } catch (IncompatibleDimensionsException e) {
            throw new RuntimeException(e);
        }
    }
}
