package com.simplecnn.functional;

import java.util.function.Function;

/**
 * This is a simple wrapper class to work with optional values and errors.
 * It has the same goal as Java's Optional class. But this library uses
 * its own Type for it, because Optional has some restrictions regarding
 * its use as a type for fields and parameters.
 *
 * @param <V> type of the value to be held in the Result
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public abstract class Result<V> {
    /**
     * Test whether Result is empty, or does contain a value
     *
     * @return true if Result is of type Result.Empty
     */
    public abstract boolean isEmpty();

    /**
     * Returns the internal value if one is present, otherwise a default value
     *
     * @param defaultValue the default to be returned, if this Result is empty
     * @return value if Result is Success, else "defaultValue"
     */
    public abstract V getOrElse(V defaultValue);

    /**
     * Returns the internal value if one is present, otherwise throws an exception
     *
     * @param e Exception to be thrown, if Result was Empty
     * @return value if Result is Success
     * @throws Exception if Result is Empty
     */
    public abstract V getOrThrow(Exception e) throws Exception;

    /**
     * Maps the internal value to another value or type
     *
     * @param f   Mapping function for the value. It must accept an input parameter of the same type
     *            as the value and returns a value of arbitrary type.
     * @param <T> type of the image of "f"
     * @return a Result containing the image of "f" (the new value)
     */
    public abstract <T> Result<T> map(Function<V, T> f);

    /**
     * Subclass of Result to represent a valid value and containing it
     *
     * @param <V> type of the internal value
     */
    public static class Success<V> extends Result<V> {
        // The internal value represented by the Result
        private final V value;

        /**
         * Private constructor to only allow creation through factory methods
         *
         * @param value the value to be wrapped by this class
         */
        private Success(V value) {
            this.value = value;
        }

        @Override
        public boolean isEmpty() {
            return false;
        }

        @Override
        public V getOrElse(V defaultValue) {
            return value;
        }

        @Override
        public V getOrThrow(Exception e) {
            return value;
        }

        @Override
        public <T> Result<T> map(Function<V, T> f) {
            return new Success<>(f.apply(value));
        }
    }

    /**
     * Subclass of Result to represent the absence of a value or an exceptional case
     *
     * @param <V> Type of the value to be represented by this class. Even though
     *            this value is non-existent we need to have the type parameter for
     *            compatibility and interchangeability with super class and Success
     */
    public static class Empty<V> extends Result<V> {
        /**
         * Private constructor to only allow creation through factory methods
         */
        private Empty() {

        }

        @Override
        public boolean isEmpty() {
            return true;
        }

        @Override
        public V getOrElse(V defaultValue) {
            return defaultValue;
        }

        @Override
        public V getOrThrow(Exception e) {
            throw new IllegalStateException("Get on empty result!");
        }

        @Override
        public <T> Result<T> map(Function<V, T> f) {
            return new Empty<>();
        }
    }

    /**
     * Factory method for creating a Success with a value
     *
     * @param value the value to be contained inside the Result
     * @param <V>   type of the value to be contained in the Result
     * @return a new instance of Result.Success containing "value"
     */
    public static <V> Result<V> of(V value) {
        return new Success<>(value);
    }

    /**
     * Factory method for creating an Empty Result
     *
     * @param <V> type of the value that this Result should have held
     * @return a new instance of Result.Empty
     */
    public static <V> Result<V> empty() {
        return new Empty<>();
    }
}
