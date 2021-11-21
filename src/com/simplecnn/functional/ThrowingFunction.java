package com.simplecnn.functional;

import java.util.function.BiFunction;
import java.util.function.IntFunction;

/**
 * Wrapper for functions of some functional interfaces the library uses, to encapsulate the use
 * of try-catch blocks inside of lambda expressions, since this is somewhat ugly and enlarges
 * those expressions.
 * <p>
 * It implements the functions by creating an instance of the original functional interface,
 * defining the task over the new "R apply(T t, U u) throws E" method almost identical to how
 * the original interfaces do. The difference being, that the actual task is redefined as
 * executing the original task inside a try-catch block, in case it trows an exception.
 * <p>
 * Finally this new, modified task is then defined to be the task of the object from the
 * original interface. This makes it possible, to create and use functions of functional
 * interfaces normally as usual, but now they can define tasks/functions that could throw
 * an exception.
 * <p>
 * In other words, the class enables the user to inject a try-catch block into a function
 * from a functional interface, and expand its functionality. Of course the try-catch block
 * can simply be put inside the defining lambda expression, but in many cases they are pretty
 * long and complicated even without a try-catch block, so it helps to encapsulate this
 * for better code readability.
 * <p>
 * This class is build with the structure of a BiFunction, to cover as many functions as
 * possible. But this means, that for functions with fewer parameters, it will still be
 * defined as a function with 2 parameters. The second parameter then just has no
 * functionality and can be ignored.
 *
 * @param <T> type of input parameter
 * @param <U> type of input parameter
 * @param <R> type of return value
 * @param <E> type of the Exception
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public interface ThrowingFunction<T, U, R, E extends Exception> {
    /**
     * Execute the task/function/method defined in this object
     *
     * @param t first parameter
     * @param u second parameter
     * @return return value of the function call this instance represents
     * @throws E the Exception that could be thrown
     */
    R apply(T t, U u) throws E;

    /**
     * Factory method to create a BiFunction injected with a try-catch
     *
     * @param f   The function first has to be defined as an instance of ThrowingFunction and then
     *            be passed to this method, to create the BiFunction from it.
     * @param <T> type of the first parameter
     * @param <U> type of the second parameter
     * @param <R> type of the value, that the defined function will return
     * @return instance of BiFunction but with its task having executed inside a try-catch block
     */
    static <T, U, R> BiFunction<T, U, R> biFunc(ThrowingFunction<T, U, R, Exception> f) {
        return (t, u) -> {
            try {
                return f.apply(t, u);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        };
    }

    /**
     * Factory method to create a IntFunction with injected try-catch block
     *
     * @param f   The function first has to be defined as an instance of ThrowingFunction and then
     *            be passed to this method, to create the BiFunction from it.
     * @param <T> type of the arguments the IntFunction operates on
     * @return instance of IntFunction but with its task having executed inside a try-catch block
     */
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
