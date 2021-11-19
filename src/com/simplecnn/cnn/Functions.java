package com.simplecnn.cnn;

import java.util.function.BiFunction;

/**
 * Some useful mathematical functions
 *
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class Functions {
    // Real-valued squared error function
    public static final BiFunction<float[], float[], Float> squared = (desired, actual) -> {
        try {
            return (float) Math.pow(Array.abs(Array.sum(desired, Array.scale(-1.f, actual))), 2.f);
        } catch (IncompatibleDimensionsException e) {
            e.printStackTrace();
        }

        return 0.f;
    };

    // Derivative of vectorized squared error
    public static final BiFunction<float[], float[], float[]> squaredD = (desired, actual) -> {
        try {
            return Array.sum(actual, Array.scale(-1.f, desired));
        } catch (IncompatibleDimensionsException e) {
            e.printStackTrace();
        }

        return new float[0];
    };

    // Cross-entropy error function
    public static final BiFunction<float[], float[], float[]> crossEnt = (desired, actual) -> {
        final float[] out = new float[desired.length];

        for (int i = 0; i < desired.length; ++i) {
            out[i] = (float) (-desired[i] * Math.log(actual[i])
                    - (1.f - desired[i]) * Math.log(1.f - actual[i]));
        }

        return out;
    };
}
