package com.simplecnn.functional;

import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.IntFunction;

/**
 * @param <T>
 * @param <U>
 * @param <R>
 * @param <E>
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public interface ThrowingFunction<T, U, R, E extends Exception> {
    R apply(T t, U u) throws E;

    static <T, U, R> BiFunction<T, U, R> biFunc(ThrowingFunction<T, U, R, Exception> f) {
        return (t, u) -> {
            try {
                return f.apply(t, u);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        };
    }

    static <T> BinaryOperator<T> biOp(ThrowingFunction<T, T, T, Exception> f) {
        return (t, u) -> {
            try {
                return f.apply(t, u);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        };
    }

    static <T> IntFunction<T> intFunc(ThrowingFunction<Integer, Integer, T, Exception> f) {
        return i -> {
            try {
                return f.apply(i, i);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        };
    }
}
