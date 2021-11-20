package com.simplecnn.functional;

/**
 * Exception for invalid inputs. For example, a network needs at least an output and one
 * input layer, thus a vector giving only the size for one or zero layers would be invalid.
 *
 * @author Marvin Bergmann
 */
public class InvalidInputFormatException extends Exception {
    public InvalidInputFormatException() {
        super("The input format is invalid for this operation!");
    }
}
