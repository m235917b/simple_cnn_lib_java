package com.simplecnn;

import com.simplecnn.cnn.Functions;
import com.simplecnn.cnn.Network;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
        try {
            final Network net = Network.randomSigmoid(new int[]{5, 7, 7, 3});

            final float[][] input = {
                    {.3f, .7f, .2f, -.4f, -.9f},
                    {.5f, -.3f, -.1f, .6f, .8f},
                    {.1f, .1f, .1f, -.9f, .6f}
            };

            final float[][] output = {
                    {1.f, 1.f, 0.f},
                    {0.f, 0.f, 1.f},
                    {1.f, 0.f, 0.f}
            };

            for (int i = 0; i < 10000; ++i) {
                net.backProp(.1f, input, output);

                float err = 0.f;

                for (int j = 0; j < input.length; ++j) {
                    err += Functions.squared.apply(output[j], net.forward(input[j]));
                }

                System.out.println(err / input.length);
            }

            for (float[] in : input) {
                System.out.println(Arrays.toString(net.forward(in)));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
