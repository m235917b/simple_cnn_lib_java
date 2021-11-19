package com.simplecnn.cnn;

/**
 * A simple Convolutional Neural Network (CNN) with some training algorithms like stochastic
 * gradient descent or genetic learning algorithms. The network can have different types of
 * layers, each with its own, individual activation function.
 *
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class Network implements Cloneable {
    // Array to save the layers
    private Layer[] layers;
    // Iterator to cycle through the layers
    private int mutIt;

    /**
     * Constructor that generates a CNN with the given layers.
     *
     * @param layers array of layers for this network
     * @throws InvalidInputFormatException if layers has no layer
     */
    public Network(Layer[] layers) throws InvalidInputFormatException {
        if (layers.length < 1) {
            throw new InvalidInputFormatException();
        }

        this.layers = layers;
        this.mutIt = 0;
    }

    /**
     * Feed the input vector to the network and calculate output values
     *
     * @param input vector of input values
     * @return vector of output values
     * @throws IncompatibleDimensionsException if input.length != layers[0].length (size of input layer)
     */
    public float[] forward(float[] input) throws IncompatibleDimensionsException {
        float[] out = input;

        for (Layer layer : layers) {
            out = layer.forward(out);
        }

        return out;
    }

    /**
     * Change the weights of this network for genetic learning
     *
     * @param learningRate determines how drastic the changes will be
     */
    public void mutateNext(float learningRate) {
        if (layers.length == 0 || layers[mutIt].mutateNext(learningRate)) {
            // Cycle the mutation iterator
            if (++mutIt == layers.length) {
                mutIt = 0;
            }
        }
    }

    /**
     * Train network via backpropagation and gradient descent on mini batch
     *
     * @param learningRate rate of change for the weights
     * @param input        input batch
     * @param output       output batch
     * @throws IncompatibleDimensionsException if input.length != layers[0].length (size of input layer)
     *                                         || output !=
     *                                         layers[layers.length - 1].length (size of output layer)
     */
    public void backProp(
            float learningRate,
            float[][] input,
            float[][] output
    ) throws IncompatibleDimensionsException {
        if (input.length != output.length) {
            throw new IncompatibleDimensionsException();
        }

        float[] result;
        float[] delta;

        // Train on the batch
        for (int i = 0; i < input.length; ++i) {
            // Forward the input to generate cached values
            result = forward(input[i]);

            // Calculate delta for the last layer
            delta = Functions.squaredD.apply(output[i], result);

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
     * @param dim array of sizes of the layers where the first value is the input layer
     * @return generated CNN
     * @throws InvalidInputFormatException     if dim has less than 2 layers (input, output)
     * @throws IncompatibleDimensionsException if dim has dimension 0
     */
    public static Network randomSigmoid(int[] dim)
            throws InvalidInputFormatException, IncompatibleDimensionsException {
        final Layer[] layers = new Layer[dim.length - 1];

        for (int l = 1; l < dim.length; ++l) {
            layers[l - 1] = Layer.randomSigmoid(dim[l], dim[l - 1]);
        }

        return new Network(layers);
    }

    /**
     * Generate CNN with random weights between -1 and 1 and given activation functions
     *
     * @param dim  array of sizes of the layers where the first value is the input layer
     * @param acts array of activation functions for each layer
     * @return generated CNN
     * @throws InvalidInputFormatException     if dim has less than 2 layers (input, output)
     * @throws IncompatibleDimensionsException if dim.length == 0 || dim.length != acts.length
     */
    public static Network random(int[] dim, Activation[] acts)
            throws InvalidInputFormatException, IncompatibleDimensionsException {
        if (dim.length != acts.length) {
            throw new IncompatibleDimensionsException();
        }

        final Layer[] layers = new Layer[dim.length - 1];

        for (int l = 1; l < dim.length; ++l) {
            layers[l - 1] = Layer.random(dim[l], dim[l - 1], acts[l]);
        }

        return new Network(layers);
    }
}
