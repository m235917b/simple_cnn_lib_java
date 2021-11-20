package com.simplecnn.cnn;

/**
 * The softmax activation function and its derivative
 *
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class Softmax implements Activation {
    @Override
    public float[] apply(float[] x) {
        final float[] out = new float[x.length];
        float denominator = 0.f;

        // Calculate common denominator
        for (float v : x) {
            denominator += Math.exp(v);
        }

        // Calculate the softmax values
        for (int i = 0; i < x.length; ++i) {
            out[i] = (float) Math.exp(x[i]) / denominator;
        }

        return out;
    }

    @Override
    public float[] applyD(float[] x) {
        final float[] out = new float[x.length];
        float denominator = 0.f;

        // Calculate common denominator
        for (float v : x) {
            denominator += Math.exp(v);
        }

        // Calculate derivatives component wise
        for (int i = 0; i < x.length; ++i) {
            out[i] = (float) (Math.exp(x[i]) * (denominator - Math.exp(x[i])) / Math.pow(denominator, 2.f));
        }

        return out;
    }
}
