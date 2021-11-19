package com.simplecnn.cnn;

import java.util.Random;

/**
 * A single layer of neurons with dendrites to previous layer and individual activation function.
 * This class also contains some helper methods for learning algorithms.
 *
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class Layer implements Cloneable {
    // Random number generator for weights
    private static final Random rnd = new Random();

    // Amount of neurons in this layer
    private final int neurons;
    // Amount of neurons from previous layer
    private final int neuronsPrev;
    // Each entry is an array of weights of dendrites to the neurons of the previous layer
    private float[][] weights;
    // Each entry is an array of biases for the neurons
    private float[] biases;
    // Activation function for this layer to make the network non-linear
    private final Activation act;
    // Cache for last input (needed for gradient descent)
    private float[] in;
    // Cache for last computed output before applying the activation function (needed for gradient descent)
    private float[] out;
    // Iterators for mutation cycling for genetic learning
    private int mutItI;
    private int mutItJ;

    /**
     * Constructor generates a new layer with biases and its own activation function
     *
     * @param weights matrix of weights for this layer
     *                (height equals to the number of neurons in this layer,
     *                width equals to the number of neurons in the previous layer)
     * @param biases  vector of biases for this layer (one for each neuron)
     * @param act     activation function for this layer
     * @throws InvalidInputFormatException     if weights has dimension 0 || weights.length != biases.length
     * @throws IncompatibleDimensionsException if weights or biases has dimension 0
     */
    public Layer(float[][] weights, float[] biases, Activation act)
            throws InvalidInputFormatException, IncompatibleDimensionsException {
        if (weights.length == 0 || weights.length != biases.length) {
            throw new InvalidInputFormatException();
        }

        this.neurons = weights.length;
        this.neuronsPrev = weights[0].length;

        // Deep copy the arrays

        this.weights = Array.copy(weights);
        this.biases = Array.copy(biases);

        this.act = act;

        this.mutItI = 0;
        this.mutItJ = 0;
    }

    /**
     * Getter for last computed output before applying the activation function.
     * Needed for gradient descent of the next layer.
     *
     * @return last computed output before applying the activation function
     */
    public float[] getOut() {
        return out;
    }

    /**
     * Getter for the activation function, since the next layer needs its derivative
     * for calculating the delta for this layer, if trained with backpropagation,
     * if different activation functions are used per layer.
     *
     * @return activation function and its derivative
     */
    public Activation getAct() {
        return act;
    }

    /**
     * Forward the input values through this layer and get corresponding outputs
     *
     * @param input input values from previous layer or input layer
     * @return array of calculated values after forwarding
     * @throws IncompatibleDimensionsException if input.length != neurons
     */
    public float[] forward(float[] input) throws IncompatibleDimensionsException {
        in = input;
        out = Array.sum(Array.mul(weights, input), biases);
        return act.apply(out);
    }

    /**
     * Change the next weight by a random amount for genetic learning
     *
     * @param learningRate determines how drastic the changes will be
     * @return true if the network has been cycled completely
     */
    public boolean mutateNext(float learningRate) {
        weights[mutItI][mutItJ] += (2.f * rnd.nextFloat() - 1.f) * learningRate;

        //Cycle the mutation iterators

        if (++mutItJ == neuronsPrev) {
            mutItJ = 0;

            if (++mutItI == neurons) {
                mutItI = 0;
                return true;
            }
        }

        return false;
    }

    /**
     * Calculate the delta for the previous layer for backpropagation / gradient descent.
     * Since the delta depends on the weights of this layer, we need to calculate delta
     * always in the next layer.
     *
     * @param delta   delta for this layer
     * @param outPrev output of the previous layer, before applying the activation function
     * @param actPrev activation function of the previous layer
     * @return delta for the previous layer
     * @throws IncompatibleDimensionsException if delta.length != neurons || outPrev.length !=
     *                                         number of neurons in previous layer
     */
    public float[] getDeltaPrev(
            float[] delta,
            float[] outPrev,
            Activation actPrev
    ) throws IncompatibleDimensionsException {
        /*
         * If we define deltaPrev as d(cost)/d(outPrev) (Where "cost" is the cost function) which can be seen as
         * the error of the previous layer, we can write
         * deltaPrev = d(cost)/d(out) * d(out)/d(in) * d(in)/d(outPrev) (where "out" is the output
         * of this layer without applying the activation function and "in" is the input for this layer, aka the
         * output of the previous layer after applying the activation function) according to the chain rule.
         *
         * Now we get in = act(outPrev) => d(in)/d(outPrev) = derAct(outPrev) and
         * out = weights * in + b => d(out)/d(in) = weightsT (where "weightsT" is the transpose of "weights").
         * This is true, because if we calculate the gradient of "out" with respect to "in", we get a gradient
         * vector for each entry in "out" (one entry for the partial derivative for each entry in "in") and the
         * gradient of "out" is defined as the matrix where each column is one of those gradient vectors (first
         * column for the first entry in "out", e.t.c.), this resulting matrix will be "weightsT".
         *
         * With this we get deltaPrev = weightsT * delta % derAct(outPrev) (where % is the hadamard product),
         * since "delta" is defined as d(cost)/d(out).
         */
        return Array.had(actPrev.applyD(outPrev), Array.mul(Array.trans(weights), delta));
    }

    /**
     * Update the weights and biases according to the delta for this layer to decrease the
     * error of this layer for one single input-output pair.
     *
     * @param delta        delta for this layer
     * @param learningRate learning rate to adjust the speed of gradient descent. If training
     *                     on a batch, this should also be divided by the size of the batch,
     *                     so the mean of the errors for every input-output pair can be
     *                     calculated and the network will improve for every input.
     * @throws IncompatibleDimensionsException if delta.length != neurons
     */
    public void gradientDescent(
            float[] delta,
            float learningRate
    ) throws IncompatibleDimensionsException {
        // Update the biases

        /*
         * Since d(cost)/d(biases) = d(cost)/d(out) * d(out)/d(biases) according to the chain rule,
         * if "out" is the output of this layer before applying the activation function and
         * d(cost)/d(out) = delta per definition, and out = weights * in + biases (where "in" is the
         * input for this layer, aka the output with activation function from the previous layer), it
         * follows, that d(out)/d(biases) = 1 and thus d(cost)/d(biases) = 1 * delta.
         *
         * Thus, the rate of change for bias j is delta(j).
         *
         * Now we multiply that with the learning rate to adjust the speed of gradient descent, and then
         * we subtract that from the old bias to lower its error.
         */
        biases = Array.sum(biases, Array.scale(-learningRate, delta));

        // Update the weights

        /*
         * Since out = weights * in + biases (where "out" is the output of this layer before applying
         * the activation function and "in" is the input for this layer, aka the output of the previous
         * layer after applying the activation function) => d(out)/d(weights) = in, we can calculate
         * d(cost)/d(weights) = d(cost)/d(out) * d(out)/d(weights) = delta * inT (where inT is the
         * transpose of "in") according to the chain rule. Now we have a gradient matrix G for the weights
         * and can update the weights with weights = weights - learningRate * G. We subtract, since we
         * want to decrease the error and multiply by the learning rate to adjust the speed of gradient
         * descent.
         */
        weights = Array.sum(weights, Array.scale(-learningRate, Array.axbT(delta, in)));
    }

    @Override
    public String toString() {
        final StringBuilder out = new StringBuilder();

        for (float[] weight : weights) {
            out.append("[");
            for (int j = 0; j < neuronsPrev; ++j) {
                out.append(weight[j]);

                if (j < neuronsPrev - 1) {
                    out.append(", ");
                }
            }

            out.append("]\n");
        }

        out.append("Biases: [");

        for (int i = 0; i < neurons; ++i) {
            out.append(biases[i]);

            if (i < neurons - 1) {
                out.append(", ");
            }
        }

        out.append("]");

        return out.toString();
    }

    @Override
    public Layer clone() {
        try {
            final Layer clone = (Layer) super.clone();

            // Create a deep copy of the arrays

            clone.weights = Array.copy(weights);
            clone.biases = Array.copy(biases);

            return clone;
        } catch (CloneNotSupportedException | IncompatibleDimensionsException e) {
            throw new AssertionError();
        }
    }

    // Factory methods

    /**
     * Create a layer with random weights between -1 and 1 and sigmoid activation function.
     *
     * @param neurons  number of neurons in this layer
     * @param preLayer number of neurons from previous layer
     * @return randomly generated layer
     * @throws InvalidInputFormatException     if neurons <= 0 || preLayer <= 0
     * @throws IncompatibleDimensionsException if neurons <= 0 || preLayer <= 0
     */
    public static Layer randomSigmoid(int neurons, int preLayer)
            throws InvalidInputFormatException, IncompatibleDimensionsException {
        final float[][] weights = new float[neurons][preLayer];
        final float[] biases = new float[neurons];

        // Fill layer with random weights between -1 and 1
        for (int i = 0; i < neurons; ++i) {
            for (int j = 0; j < preLayer; ++j) {
                weights[i][j] = rnd.nextFloat() * 2.f - 1.f;
            }
        }

        // Fill layer with random biases between -1 and 1
        for (int i = 0; i < biases.length; ++i) {
            biases[i] = rnd.nextFloat() * 2.f - 1.f;
        }

        return new Layer(weights, biases, new Sigmoid());
    }

    /**
     * Create a layer with random weights between -1 and 1 and given activation function.
     *
     * @param neurons  number of neurons in this layer
     * @param preLayer number of neurons from previous layer
     * @param act      activation function for this layer
     * @return randomly generated layer
     * @throws InvalidInputFormatException     if neurons <= 0 || preLayer <= 0
     * @throws IncompatibleDimensionsException if neurons <= 0 || preLayer <= 0
     */
    public static Layer random(int neurons, int preLayer, Activation act)
            throws InvalidInputFormatException, IncompatibleDimensionsException {
        final float[][] weights = new float[neurons][preLayer];
        final float[] biases = new float[neurons];

        // Fill layer with random weights between -1 and 1
        for (int i = 0; i < neurons; ++i) {
            for (int j = 0; j < preLayer; ++j) {
                weights[i][j] = rnd.nextFloat() * 2.f - 1.f;
            }
        }

        // Fill layer with random biases between -1 and 1
        for (int i = 0; i < biases.length; ++i) {
            biases[i] = rnd.nextFloat() * 2.f - 1.f;
        }

        return new Layer(weights, biases, act);
    }
}