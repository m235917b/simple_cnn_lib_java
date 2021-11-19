package com.simplecnn.cnn;

/**
 * Exception for incompatible dimensions of vectors / matrices for operations on them
 * requiring fitting widths / lengths.
 *
 * @author Marvin Bergmann
 */
public class IncompatibleDimensionsException extends Exception {
    public IncompatibleDimensionsException() {
        super("The dimensions of the vectors / matrices are incompatible for this operation!");
    }
}
