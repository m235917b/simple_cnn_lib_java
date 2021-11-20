package com.simplecnn.cnn;

import java.util.Random;

/**
 * A simple Convolutional Neural Network (CNN) with some training algorithms like stochastic
 * gradient descent or genetic learning algorithms. The network can have different types of
 * layers, each with its own, individual activation function.
 *
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class Network implements Cloneable {
    // Random number generator
    private static final Random rnd = new Random();

    // Array to save the layers
    private Layer[] layers;
    // Error function with derivative for output layer
    private final ErrorFunction err;

    /**
     * Constructor that generates a CNN with the given layers.
     *
     * @param layers array of layers for this network
     * @param err    error function with derivative for output layer
     * @throws InvalidInputFormatException if layers has no layer
     */
    public Network(Layer[] layers, ErrorFunction err) throws InvalidInputFormatException {
        if (layers.length < 1) {
            throw new InvalidInputFormatException();
        }

        this.layers = layers;
        this.err = err;
    }

    /**
     * Feed the input vector to the network and calculate output values
     *
     * @param input vector of input values
     * @return vector of output values
     * @throws IncompatibleDimensionsException if input.length != layers[0].neuronsPrev
     *                                         (size of input layer)
     */
    public float[] forward(float[] input) throws IncompatibleDimensionsException {
        float[] out = input;

        for (Layer layer : layers) {
            out = layer.forward(out);
        }

        return out;
    }

    /**
     * Feed an entire batch through the network and calculate the output values
     *
     * @param input input batch
     * @return batch of output values
     * @throws IncompatibleDimensionsException if length of input vectors != layers[0].neurons
     *                                         (size of input layer)
     */
    public float[][] forward(float[][] input) throws IncompatibleDimensionsException {
        float[][] out = new float[input.length][layers[layers.length - 1].neurons];

        for (int i = 0; i < out.length; ++i) {
            out[i] = forward(input[i]);
        }

        return out;
    }

    /**
     * Train network via backpropagation and gradient descent on mini batch
     *
     * @param desired      desired output values for the batch
     * @param input        input batch
     * @param learningRate rate of change for the weights
     * @throws IncompatibleDimensionsException if input.length != layers[0].neuronsPrev (size of input layer)
     *                                         || desired.length !=
     *                                         layers[layers.length - 1].neurons (size of output layer)
     */
    public void backProp(
            float[][] desired,
            float[][] input,
            float learningRate
    ) throws IncompatibleDimensionsException {
        if (input.length != desired.length) {
            throw new IncompatibleDimensionsException();
        }

        float[] result;
        float[] delta;

        // Train on the batch
        for (int i = 0; i < input.length; ++i) {
            // Forward the input to generate cached values
            result = forward(input[i]);

            // Calculate delta for the last layer
            delta = err.applyD(desired[i], result);

            // Iterate backwards through the net
            for (int j = layers.length - 1; j >= 0; --j) {
                // Update layer
                layers[j].gradientDescent(delta, learningRate / input.length);
                if (j != 0) {
                    // Get next delta
                    delta = layers[j].getDeltaPrev(delta);
                }
            }
        }
    }

    /**
     * Changes a randomly chosen weight or bias by a random amount, tests whether the net
     * performs better than before and reverses the changes, if not. Resulting in either
     * a reduction of the error of this net, or no change.
     *
     * @param desired      desired values for each input vector
     *                     (order must match those of "input")
     * @param input        input batch
     * @param mutationRate how drastic the changes will be
     * @return new error of the network
     * @throws IncompatibleDimensionsException if length of input vectors != number of neurons
     *                                         of input layer
     */
    public float evolve(float[][] desired, float[][] input, float mutationRate)
            throws IncompatibleDimensionsException {
        // Get error before change

        float errOld = err.apply(desired, forward(input));

        // Randomly choose a layer to change a weight or bias of it
        final int index = rnd.nextInt(layers.length);
        layers[index].mutate(mutationRate);

        // Get error after change

        float errNew = err.apply(desired, forward(input));

        if (errNew > errOld) {
            layers[index].reverseMutation();
            return errOld;
        }

        return errNew;
    }

    @Override
    public String toString() {
        final StringBuilder out = new StringBuilder();

        for (int l = 0; l < layers.length; ++l) {
            out.append(layers[l]);

            if (l < layers.length - 1) {
                out.append("\n----------\n");
            }
        }

        return out.toString();
    }

    @Override
    public Network clone() {
        try {
            final Network clone = (Network) super.clone();

            // Create a deep copy of the layer array
            clone.layers = new Layer[layers.length];

            for (int i = 0; i < layers.length; ++i) {
                clone.layers[i] = layers[i].clone();
            }

            return clone;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();
        }
    }

    // Factory methods

    /**
     * Generate CNN with random weights between -1 and 1 and sigmoid activation function
     *
     * @param layout array of sizes of the layers where the first value is the input layer
     * @param err    error function and derivative for output layer
     * @return generated CNN
     * @throws InvalidInputFormatException     if layout has less than 2 layers (input, output)
     * @throws IncompatibleDimensionsException if layout has dimension 0
     */
    public static Network randomSigmoid(int[] layout, ErrorFunction err)
            throws InvalidInputFormatException, IncompatibleDimensionsException {
        final Layer[] layers = new Layer[layout.length - 1];

        for (int l = 1; l < layout.length; ++l) {
            layers[l - 1] = Layer.randomSigmoid(layout[l], layout[l - 1]);
        }

        return new Network(layers, err);
    }

    /**
     * Generate CNN with random weights between -1 and 1 and given activation functions
     *
     * @param layout array of sizes of the layers where the first value is the input layer
     * @param acts   array of activation functions for each layer
     * @param err    error function and derivative for output layer
     * @return generated CNN
     * @throws InvalidInputFormatException     if layout has less than 2 layers (input, output)
     * @throws IncompatibleDimensionsException if layout.length == 0 || layout.length != acts.length
     */
    public static Network random(int[] layout, Activation[] acts, ErrorFunction err)
            throws InvalidInputFormatException, IncompatibleDimensionsException {
        if (layout.length != acts.length) {
            throw new IncompatibleDimensionsException();
        }

        final Layer[] layers = new Layer[layout.length - 1];

        for (int l = 1; l < layout.length; ++l) {
            layers[l - 1] = Layer.random(layout[l], layout[l - 1], acts[l]);
        }

        return new Network(layers, err);
    }
}
