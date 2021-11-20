package com.simplecnn.cnn;

import com.simplecnn.functional.Activation;
import com.simplecnn.functional.IncompatibleDimensionsException;
import com.simplecnn.functional.InvalidInputFormatException;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

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
    public final int neurons;
    // Amount of neurons from previous layer
    public final int neuronsPrev;
    // Each entry is an array of weights of dendrites to the neurons of the previous layer
    private double[][] weights;
    // Each entry is an array of biases for the neurons
    private double[] biases;
    // Activation function for this layer to make the network non-linear
    private final Activation act;

    // Fields for gradient descent learning algorithm

    // Cache for last input
    private double[] in;
    // Cache for last computed output before applying the activation function
    private double[] out;

    // Fields for genetic learning algorithm

    // Cache for last mutated index (between 0 and neurons * neuronsPrev + neurons,
    // so it maps the layer linearly through the matrix and then the biases
    private int index;
    // Cache for value of last mutated index before mutation, to reverse mutation changes
    private double lastValue;

    /**
     * Constructor generates a new layer with biases and its own activation function
     *
     * @param weights matrix of weights for this layer
     *                (height equals to the number of neurons in this layer,
     *                width equals to the number of neurons in the previous layer)
     * @param biases  vector of biases for this layer (one for each neuron)
     * @param act     activation function for this layer
     * @throws InvalidInputFormatException if weights has dimension 0 || weights.length != biases.length
     */
    public Layer(double[][] weights, double[] biases, Activation act)
            throws InvalidInputFormatException {
        if (weights.length == 0 || weights.length != biases.length) {
            throw new InvalidInputFormatException();
        }

        this.neurons = weights.length;
        this.neuronsPrev = weights[0].length;

        // Deep copy the arrays

        this.weights = Array.copy(weights);
        this.biases = Array.copy(biases);

        this.act = act;
    }

    /**
     * Forward the input values through this layer and get corresponding outputs
     *
     * @param input input values from previous layer or input layer
     * @return array of calculated values after forwarding
     */
    public double[] forward(double[] input) {
        try {
            in = input;
            out = Array.add(Array.mul(weights, input), biases);
            return act.apply(out);
        } catch (IncompatibleDimensionsException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Calculate the delta for the previous layer for backpropagation / gradient descent.
     * Since the delta depends on the weights of this layer, we need to calculate delta
     * always in the next layer.
     *
     * @param delta delta for this layer
     * @return delta for the previous layer
     */
    private double[] getDeltaPrev(double[] delta) {
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
         *
         * This method only returns the part requiring information from this layer, to keep the data
         * encapsulated.
         */
        try {
            return Array.mul(Array.trans(weights), delta);
        } catch (IncompatibleDimensionsException e) {
            throw new RuntimeException(e);
        }
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
     * @return delta for the previous layer
     */
    public double[] gradientDescent(double[] delta, double learningRate) {
        try {
            /*
             * Calculate missing part for the gradient (delta),
             * which couldn't be calculated in the next layer.
             * (The complete Formula would be delta(i) = w(i+1) * delta(i+1) * activationDi(out),
             * where "w(i+1)" and "delta(i+1)" are the weight matrix and delta of the next layer
             * and "delta(i)", "activationDi" are the activation function and derivative of the
             * activation function of this layer and "out" is the last output from this layer
             * before applying the activation function).
             *
             * But "delta" is only w(i+1) * delta(i+1), the part that needs information from the
             * next layer. The part needing information from this layer is calculated here, so
             * we keep the data encapsulated.
             */
            final double[] deltaFinal = Array.had(act.applyD(out), delta);

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
            biases = Array.add(biases, Array.scale(-learningRate, deltaFinal));

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
            weights = Array.add(weights, Array.scale(-learningRate, Array.axbT(deltaFinal, in)));

            return getDeltaPrev(delta);
        } catch (IncompatibleDimensionsException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Change a randomly chosen weight or bias by a random amount for genetic learning
     *
     * @param mutationRate determines how drastic the changes will be
     */
    public void mutate(double mutationRate) {
        // Generate random index
        index = rnd.nextInt(neurons * neuronsPrev + neurons);

        // Determine, whether "index" points to a weight, or bias
        if (index >= neurons * neuronsPrev) {
            final int i = index - neurons * neuronsPrev;

            // Cache bias before mutating it
            lastValue = biases[i];
            // Change bias by a random amount
            biases[i] = biases[i] + (2. * rnd.nextDouble() - 1.) * mutationRate;
        } else {
            final int i = index / neuronsPrev, j = index % neuronsPrev;

            // Cache weight before mutating it
            lastValue = weights[i][j];
            // Change weight by a random amount
            weights[i][j] = weights[i][j] + (2.f * rnd.nextDouble() - 1.f) * mutationRate;
        }
    }

    /**
     * Reverse the changes made by the last call of "mutate". Useful if the mutated
     * network performs worse than before.
     */
    public void reverseMutation() {
        // Determine whether index is pointing to a weight or bias
        if (index >= neurons * neuronsPrev) {
            // Reverse changes on bias
            biases[index - neurons * neuronsPrev] = lastValue;
        } else {
            // Reverse changes on weights
            weights[index / neuronsPrev][index % neuronsPrev] = lastValue;
        }
    }

    @Override
    public String toString() {
        return Arrays.deepToString(weights) + "\nBiases: " + Arrays.toString(biases);
    }

    @Override
    public Layer clone() {
        try {
            final Layer clone = (Layer) super.clone();

            // Create a deep copy of the arrays

            clone.weights = Array.copy(weights);
            clone.biases = Array.copy(biases);

            return clone;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();
        }
    }

    // Factory methods

    /**
     * Create a layer with random weights between -1 and 1 and given activation function.
     *
     * @param neurons    number of neurons in this layer
     * @param neuronsPre number of neurons from previous layer
     * @param act        activation function for this layer
     * @return randomly generated layer
     * @throws InvalidInputFormatException if neurons <= 0 || neuronsPre <= 0
     */
    public static Layer random(int neurons, int neuronsPre, Activation act)
            throws InvalidInputFormatException {
        return new Layer(IntStream
                .range(0, neurons)
                .mapToObj(i -> IntStream
                        .range(0, neuronsPre)
                        .mapToDouble(j -> rnd.nextDouble() * 2. - 1.)
                        .toArray())
                .toArray(double[][]::new),
                IntStream
                        .range(0, neurons)
                        .mapToDouble(i -> rnd.nextDouble() * 2. - 1.)
                        .toArray(),
                act);
    }
}
