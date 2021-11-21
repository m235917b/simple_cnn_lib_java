package com.simplecnn.functional;

import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * This class implements a very generalized version of an iterator, that can be used to iterate
 * over virtually anything that could be iterated over. This is possible, because through the
 * use of a generic transition function, virtually anything can be used as an iterator/index,
 * and it can be given any desired behaviour.
 * <p>
 * That means, that the user must implement the detailed behaviour of it, but on the other hand,
 * he has complete control over how to use it. And by choosing the type of the iterator and mapping
 * it accordingly to the desired scenario, it can be used to iterate over virtually anything that
 * consists of more than one element/unit. Also, the transition function for the iterator can be
 * defined freely and the terminating condition is independent of the iterator (but can of course be
 * formulated, such that it does if needed), which enables almost any desired behaviour (like iterating
 * cyclically over a collection, until some terminating condition is met). This makes this class a very
 * generalized, but useful tool for everything having to do with iterations and recursion.
 * <p>
 * The main purpose however is recursion. This class allows to formulate a definition for a
 * recursive function functionally and then execute it stack-safe!
 *
 * @param <E> type of the return value for each iteration
 * @param <T> type of the iterator
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class RecursiveIterator<E, T> {
    // Transition function for the iterator
    private final Function<T, T> next;
    // This function represents the iteration/recursive function, thus defining the actual task
    private final BiFunction<E, T, Result<E>> f;

    /**
     * Constructor for creating a RecursiveIterator object.
     *
     * @param next transition function for the iterator
     * @param f    function defining the actual task/behaviour for each iteration
     */
    private RecursiveIterator(Function<T, T> next, BiFunction<E, T, Result<E>> f) {
        this.next = next;
        this.f = f;
    }

    /**
     * Execute one iteration/recursion step, by first executing the defined function/task "f", then defining
     * but suspending the recursive call for the next iteration (which makes it stack-save). The next
     * call on this method "iterate" will then execute the previously defined call. The return value can
     * be handed to the next iteration with the first parameter "acc", the accumulator. The transition
     * function "next" is applied after executing the function on the iterator, giving the next value/state,
     * according to the user definition.
     * <p>
     *
     * @param acc An accumulator, which is the return value from the last iteration, or if this one is the
     *            first, it is the initial/neutral value the user can define. The return value of the function
     *            f will be passed to the next iteration automatically, there is no need to take care of it.
     * @param it  The iterator, which is needed for implementing the actual iteration process over a set of
     *            elements (for example defining an integer iterator and using this parameter as index for
     *            an array, it is possible to iterate over the array and executing a task on the elements
     *            and even including the return value of the last iteration, if necessary). The iterator can
     *            also be used just to change the behaviour of the function over the span of the process
     *            or even as a second, general parameter, since anything is allowed as iterator, as long
     *            as the transition function is defined properly and the function checks for possible
     *            bound conditions, if necessary.
     * @return The suspended definition of the next recursive call, wrapped in a class responsible for
     * stack-safety, called TailCall. If the terminating condition is met, it will instead contain
     * the last return value from the task function "f". The task function "f" returns a Result of the
     * actual value, so that the termination condition can be defined by returning an empty Result
     * instead of a Result with a return value. The method will still return the last non-empty return
     * value, which it can access via the first parameter "acc" as explained.
     */
    public TailCall<E> iterate(E acc, T it) {
        return f.apply(acc, it)
                .map(v -> TailCall.call(() -> iterate(v, next.apply(it))))
                .getOrElse(TailCall.ret(Result.of(acc)));
    }

    /**
     * This method executes recursively and stack-save the iterations defined via the method
     * TailCall<E> iterate(E acc, T it) through "next" and "f".
     *
     * @param initial the initial value for the accumulator, which is the parameter for the
     *                first iteration
     * @param itStart start position/value of the iterator
     * @return The last returned non-empty value, or if there was none or an error occurred, it
     * returns the initial value (which can therefore be used to define a "null" state as initial
     * value and thus return optional values, if wished).
     */
    public E eval(E initial, T itStart) {
        return iterate(initial, itStart).eval().getOrElse(initial);
    }

    /**
     * Factory method for creating a general recursive iterator.
     *
     * @param next Transition function for the iterator, which will be applied to it after each iteration.
     * @param f    Function defining the actual behaviour/task for the iterations, or the defined recursive
     *             function. It accepts a value of type <E> for the accumulator/return value from the last
     *             iteration or the initial value, and a value of type <T> which is the iterator. It returns
     *             a value of type Result<E>. It must be <E>, because otherwise it would not be possible to
     *             define a recursive process, which can pass down information, and it's wrapped in a Result,
     *             so the function can tell this object when to terminate the execution (by returning a
     *             Result.<E>Empty).
     * @param <E>  type of the return values of the function.
     * @param <T>  type of the iterator.
     * @return an instance of a RecursiveIterator<E, T> defined by "next" and "f"
     */
    public static <E, T> RecursiveIterator<E, T> of(
            Function<T, T> next,
            BiFunction<E, T, Result<E>> f) {
        return new RecursiveIterator<>(next, f);
    }
}
