package com.simplecnn.cnn;

import java.util.function.Function;

/**
 * The sigmoid activation function and its derivative
 *
 * @author Marvin Bergmann
 */
public class Sigmoid implements Activation {
    private final Function<float[], float[]> sigmoid = x -> {
        final float[] out = new float[x.length];

        for (int i = 0; i < x.length; ++i) {
            out[i] = (float) (1.f / (1.f + Math.exp(-x[i])));
        }

        return out;
    };

    private final Function<float[], float[]> sigmoidD = x -> {
        final float[] out = new float[x.length];

        for (int i = 0; i < x.length; ++i) {
            out[i] = (float) (Math.exp(-x[i]) / Math.pow(1.f + Math.exp(-x[i]), 2.f));
        }

        return out;
    };

    @Override
    public float[] apply(float[] x) {
        return sigmoid.apply(x);
    }

    @Override
    public float[] applyD(float[] x) {
        return sigmoidD.apply(x);
    }
}
