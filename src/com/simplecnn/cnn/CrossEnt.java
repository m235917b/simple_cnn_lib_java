package com.simplecnn.cnn;

public class CrossEnt implements ErrorFunction {
    @Override
    public float apply(float[][] desired, float[][] output) throws IncompatibleDimensionsException {
        // Calculate mean of entries from  (desired - 1) * log(1 - output) - desired * log(output)
        return Array.sum(
                Array.sub(
                        Array.had(
                                Array.add(desired, Array.matrix(desired.length, desired[0].length, -1.f)),
                                Array.log(Array.sub(
                                        Array.matrix(desired.length, desired[0].length, 1.f),
                                        output
                                ))
                        ),
                        Array.had(
                                desired,
                                Array.log(output)
                        )
                )
        ) / (desired.length * desired[0].length);
    }

    @Override
    public float[] applyD(float[] desired, float[] output) throws IncompatibleDimensionsException {
        return Array.div(
                Array.add(
                        Array.vector(desired.length, 1.f),
                        Array.scale(-2.f, desired)),
                output);
    }
}
