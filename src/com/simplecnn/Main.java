package com.simplecnn;

import com.simplecnn.cnn.ErrorFunction;
import com.simplecnn.cnn.Network;
import com.simplecnn.cnn.Squared;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
        try {
            // Convolutional neural network with random weights between -1 and 1 and sigmoid activation function
            final Network net = Network.randomSigmoid(new int[]{5, 7, 7, 3}, new Squared());

            // Input data set (each entry is an input vector for the network)
            final float[][] input = {
                    {.3f, .7f, .2f, -.4f, -.9f},
                    {.5f, -.3f, -.1f, .6f, .8f},
                    {.1f, .1f, .1f, -.9f, .6f}
            };

            // Desired output for the input data set
            // (first entry is the desired output vector for the first input vector in input)
            final float[][] desired = {
                    {1.f, 1.f, 0.f},
                    {0.f, 0.f, 1.f},
                    {1.f, 0.f, 0.f}
            };

            ErrorFunction ef = new Squared();

            // Train network for 1000 epochs
            for (int i = 0; i < 10000; ++i) {
                // Use backpropagation as learning algorithm
                net.backProp(desired, input, .1f);
                // Calculate error for batch
                System.out.println(ef.apply(desired, net.forward(input)));
            }

            /*for (int i = 0; i < 10000; ++i) {
                System.out.println(net.evolve(desired, input, 0.1f));
            }*/

            for (float[] in : input) {
                System.out.println(Arrays.toString(net.forward(in)));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
