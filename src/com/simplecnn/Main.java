package com.simplecnn;

import com.simplecnn.cnn.*;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
        try {
            // Convolutional neural network with random weights between -1 and 1 and sigmoid activation function
            final Network nett = Network.randomSigmoid(new int[]{5, 7, 7, 3}, new Squared());
            final Network net = nett.clone();

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

            System.out.println(nett);

            // Train network for 1000 epochs
            for (int i = 0; i < 10000; ++i) {
                // Use backpropagation as learning algorithm
                net.backProp(desired, input, .1);
                // Calculate error for batch
                System.out.println(ef.apply(desired, net.forward(input)));
            }

            for (int i = 0; i < 10000; ++i) {
                System.out.println(net.evolve(desired, input, 0.1));
            }

            for (double[] in : input) {
                System.out.println(Arrays.toString(net.forward(in)));
            }

            System.out.println(net);
            System.out.println(nett);

            Layer[] l1 = new Layer[]{
                    Layer.random(5, 7, new Sigmoid()),
                    Layer.random(3, 2, new Sigmoid())
            };
            Layer[] l2 = Arrays.stream(l1).map(Layer::clone).toArray(Layer[]::new);
            System.out.println("***" + Arrays.toString(l1) + "***");
            System.out.println(Arrays.toString(l2) + "+++");
            for (int i = 0; i < 100; ++i) {
                l2[0].mutate(.1);
                l2[1].mutate(.3);
            }
            System.out.println(Arrays.toString(l1) + "***");
            System.out.println(Arrays.toString(l2));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
