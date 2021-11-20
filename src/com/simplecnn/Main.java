package com.simplecnn;

import com.simplecnn.cnn.*;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main {

    public static TailCall<LinkedList<double[]>> reverse(LinkedList<double[]> acc, LinkedList<double[]> l) {
        if (!l.isEmpty()) {
            acc.add(l.pop());
            return TailCall.call(() -> reverse(acc, l));
        } else {
            return TailCall.ret(Result.of(acc));
        }
    }

    public static void main(String[] args) {
        try {
            // Convolutional neural network with random weights between -1 and 1 and sigmoid activation function
            final Network net = Network.randomSigmoid(new int[]{5, 7, 7, 3}, new Squared());

            // Input data set (each entry is an input vector for the network)
            final double[][] input = {
                    {.3, .7, .2, -.4, -.9},
                    {.5, -.3, -.1, .6, .8},
                    {.1, .1, .1, -.9, .6}
            };

            // Desired output for the input data set
            // (first entry is the desired output vector for the first input vector in input)
            final double[][] desired = {
                    {1., 1., 0.},
                    {0., 0., 1.},
                    {1., 0., 0.}
            };

            Cost ef = new Squared();

            // Train network for 10000 epochs
            for (int i = 0; i < 10000; ++i) {
                // Use backpropagation as learning algorithm
                net.backProp(
                        Arrays.stream(desired).collect(Collectors.toCollection(LinkedList::new)),
                        Arrays.stream(input).collect(Collectors.toCollection(LinkedList::new)),
                        .1).eval();

                // Calculate error for batch
                System.out.println(ef.apply(desired, net.forward(input)));
            }

            System.out.println(Arrays.deepToString(net.forward(input)));

            LinkedList<Double> ll = Stream.of(1., 2., 3., .5).collect(Collectors.toCollection(LinkedList::new));

            System.out.println(TailCall.recIt(0., ll.iterator(), Double::sum).eval().getOrElse(-1.));

            /*for (int i = 0; i < 10000; ++i) {
                // Use genetic learning algorithm
                System.out.println(net.evolve(desired, input, .1));
            }*/
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
