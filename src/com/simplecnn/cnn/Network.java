package com.simplecnn.cnn;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
    private final Cost cost;

    /**
     * Constructor that generates a CNN with the given layers.
     *
     * @param layers array of layers for this network
     * @param cost   cost function with derivative for output layer
     * @throws InvalidInputFormatException if layers has no layer
     */
    public Network(Layer[] layers, Cost cost) throws InvalidInputFormatException {
        if (layers.length < 1) {
            throw new InvalidInputFormatException();
        }

        this.layers = layers;
        this.cost = cost;
    }

    /**
     * Feed the input vector to the network and calculate output values
     *
     * @param input vector of input values
     * @return vector of output values
     * @throws IncompatibleDimensionsException if input.length != layers[0].neuronsPrev
     *                                         (size of input layer)
     */
    public double[] forward(double[] input) throws IncompatibleDimensionsException {
        double[] out = input;

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
     */
    public double[][] forward(double[][] input) {
        return Arrays
                .stream(input)
                .map(row -> {
                    try {
                        return forward(row);
                    } catch (IncompatibleDimensionsException e) {
                        throw new RuntimeException(e);
                    }
                })
                .toArray(double[][]::new);
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
            double[][] desired,
            double[][] input,
            double learningRate
    ) throws IncompatibleDimensionsException {
        if (input.length != desired.length) {
            throw new IncompatibleDimensionsException();
        }

        double[] result;
        double[] delta;

        // Train on the batch
        for (int i = 0; i < input.length; ++i) {
            // Forward the input to generate cached values
            result = forward(input[i]);

            // Calculate delta for the last layer
            delta = cost.applyD(desired[i], result);

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
    public double evolve(double[][] desired, double[][] input, double mutationRate)
            throws IncompatibleDimensionsException {
        // Get error before change

        double errOld = cost.apply(desired, forward(input));

        // Randomly choose a layer to change a weight or bias of it
        final int index = rnd.nextInt(layers.length);
        layers[index].mutate(mutationRate);

        // Get error after change

        double errNew = cost.apply(desired, forward(input));

        if (errNew > errOld) {
            layers[index].reverseMutation();
            return errOld;
        }

        return errNew;
    }

    @Override
    public String toString() {
        return Arrays
                .stream(layers)
                .map(Layer::toString)
                .collect(Collectors.joining("\n----------\n"));
    }

    @Override
    public Network clone() {
        try {
            final Network clone = (Network) super.clone();

            // Create deep copy of layers

            clone.layers = Arrays.stream(layers).map(Layer::clone).toArray(Layer[]::new);

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
     * @param cost   cost function and derivative for output layer
     * @return generated CNN
     * @throws InvalidInputFormatException if layout has less than 2 layers (input, output)
     */
    public static Network randomSigmoid(int[] layout, Cost cost) throws InvalidInputFormatException {
        return new Network(IntStream
                .range(1, layout.length)
                .mapToObj(i -> {
                    try {
                        return Layer.random(layout[i], layout[i - 1], new Sigmoid());
                    } catch (InvalidInputFormatException e) {
                        throw new RuntimeException(e);
                    }
                })
                .toArray(Layer[]::new),
                cost);
    }

    /**
     * Generate CNN with random weights between -1 and 1 and given activation functions
     *
     * @param layout array of sizes of the layers where the first value is the input layer
     * @param acts   array of activation functions for each layer
     * @param cost   cost function and derivative for output layer
     * @return generated CNN
     * @throws InvalidInputFormatException if layout has less than 2 layers (input, output)
     */
    public static Network random(int[] layout, Activation[] acts, Cost cost) throws InvalidInputFormatException {
        return new Network(IntStream
                .range(1, layout.length)
                .mapToObj(i -> {
                    try {
                        return Layer.random(layout[i], layout[i - 1], acts[i]);
                    } catch (InvalidInputFormatException e) {
                        throw new RuntimeException(e);
                    }
                })
                .toArray(Layer[]::new),
                cost);
    }
}
