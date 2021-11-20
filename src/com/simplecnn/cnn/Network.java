package com.simplecnn.cnn;

import com.simplecnn.functional.*;

import java.util.*;
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
     */
    public double[] forward(double[] input) {
        return TailCall
                .recIt(
                        input,
                        Arrays.asList(layers).iterator(),
                        (acc, layer) -> layer.forward(acc)
                ).eval()
                .getOrElse(new double[0]);
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
                .map(this::forward)
                .toArray(double[][]::new);
    }

    /**
     * Train network via backpropagation and gradient descent on mini batch
     *
     * @param desired      desired output values for the batch
     * @param input        input batch
     * @param learningRate rate of change for the weights
     */
    public void backProp(
            double[][] desired,
            double[][] input,
            double learningRate
    ) {
        /*
         * Run through each input-output pair in the batch by popping them from the list and feeding them
         * to the derivative of the cost function to get the delta for the last layer (which is the first
         * to be processed).
         * Then iterate backwards through the layers and calculate the deltas for each.
         * If we reach the input layer, the iterator will start at the output layer again cyclically to
         * process the next batch, until the batch is empty.
         *
         * "RecursiveIterator.of" first defines how the iterator will behave/change after each iteration,
         * then it defines the task for the iterator as a Function of a "Result", taking the accumulator
         * and iterator as input.
         *
         * We return a result, to be able to define a termination condition. Its value is the return
         * value of the tail-recursive function defined by this expression. The accumulator is used to
         * pass arguments/return values from one iteration to the next function call. Lastly, the function
         * needs the iterator, to be able to change its behaviour according to the progress of the recursion.
         * This could be passed down via the accumulator as argument if we used a special data structure,
         * like a Pair for it, but this would make this already complex expression even more complex.
         *
         * Finally, the "eval" method starts the recursion by giving an initial value for the accumulator
         * (in this case the first delta, which is the delta of the cost function needed by the output
         * layer), and an initial value for the iterator.
         */
        RecursiveIterator.<double[], Pair<Integer, Integer>>of(it -> it.getSnd() <= 0
                                ? Pair.of(it.getFst() + 1, layers.length - 1)
                                : Pair.of(it.getFst(), it.getSnd() - 1),
                        (acc, it) -> it.getSnd() <= 0
                                ? it.getFst() >= desired.length
                                ? Result.empty()
                                : Result.of(cost.applyD(desired[it.getFst()], forward(input[it.getFst()])))
                                : Result.of(layers[it.getSnd()]
                                .gradientDescent(acc, learningRate / desired.length)))
                .eval(cost.applyD(desired[0], forward(input[0])),
                        Pair.of(1, layers.length - 1));
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
            // Recover old state, if net performs worse now
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
