package com.simplecnn;

import com.simplecnn.cnn.*;
import com.simplecnn.functional.Cost;
import com.simplecnn.functional.Squared;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        try {
            // Convolutional neural network with random weights between -1 and 1 and sigmoid activation function
            final Network net = Network.randomSigmoid(new int[]{5, 7, 7, 3}, new Squared());

            // Input data set (each entry is an input vector for the network)
            final double[][] input = new double[][]{
                    new double[]{.3, .7, .2, -.4, -.9},
                    new double[]{.5, -.3, -.1, .6, .8},
                    new double[]{.1, .1, .1, -.9, .6}
            };

            // Desired output for the input data set
            // (first entry is the desired output vector for the first input vector in input)
            final double[][] desired = new double[][]{
                    new double[]{1., 1., 0.},
                    new double[]{0., 0., 1.},
                    new double[]{1., 0., 0.}
            };

            final Cost ef = new Squared();

            // Train network for 10000 epochs
            for (int i = 0; i < 10000; ++i) {
                // Use backpropagation as learning algorithm
                net.backProp(desired, input, .1);

                // Calculate error for batch
                System.out.println(ef.apply(desired, net.forward(input)));
            }

            /* for (int i = 0; i < 10000; ++i) {
                // Use genetic learning algorithm
                System.out.println(net.evolve(desired, input, .1));
            } */

            System.out.println(Arrays.deepToString(net.forward(input)));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
