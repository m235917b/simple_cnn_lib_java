package com.simplecnn.functional;

import java.util.function.Supplier;

/**
 * This is a wrapper class for recursive function/method calls. This allows recursive calls
 * to be suspended thus enable lazy recursion, so it is stack-save. By suspending the calls,
 * enabling to execute single recursive calls, without invoking the next recursive call,
 * before the last function/method could be exited (either because it is not tail-recursive,
 * or because Java stays inside the calling function until the return statement as been
 * evaluated, even if it won't be used, which would be tail-call-optimization).
 * <p>
 * This can be accomplished, by instead of calling itself recursively, the method just
 * defines the parameters that would be handed to the next call, i.e. how the call
 * would have been executed and storing that information as a TailCall instance, without
 * actually calling it. The TailCall-wrapper object ist then returned and the function/
 * method can be exited, before the next call will be executed, thus clearing the stack.
 * <p>
 * When the function/method as been exited, the suspended call can be executed without
 * creating a nested hierarchy structure on the stack, so a stack overflow is prevented.
 * <p>
 * Although the iteration through the suspended calls and their successive execution is
 * done in a non-functional way, it is encapsulated inside this class and only needs to
 * be defined one time. But by allowing every other function/method to be defined
 * recursive and in a functional friendly way from then on, without ever again having
 * anything to do with this small non-functional part, the use of this class will still
 * make the programs much more functional without having to risk a stack overflow.
 *
 * @param <R> type of the return value of the recursive function/method
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public abstract class TailCall<R> {
    /**
     * Executes the suspended function/method calls iteratively, until the termination condition for the
     * recursion has been met and the last call has been executed.
     *
     * @return return value from the last/terminating call, wrapped in a Result, to also map void methods
     * (since a TailCall function/method always must return an instance of TailCall, but the
     * function/method has nothing to pass as value, wrapped in a result he can now pass a Result.Empty)
     */
    public abstract Result<R> eval();

    /**
     * Executes (only) the next suspended function/method call.
     *
     * @return TailCall wrapper object containing either the next suspended function/method call,
     * or the final return value from the last, terminating call.
     */
    public abstract TailCall<R> execute();

    /**
     * Tests whether this object represents a function/method call, or a return value
     *
     * @return true, if this object represents a function/method call
     */
    public abstract boolean isCall();

    /**
     * Subclass of TailCall to hold suspended (mostly recursive, but this is not necessary)
     * function/method calls by defining the call in a Supplier without actually calling
     * the function. So recursive call can be executed, without invoking the next recursive
     * call. In this manner the calls can be executed independent of each other (but of
     * course still in correct order), making it stack-safe.
     *
     * @param <R> Type of the return value that the function/method should return, which
     *            is mapped by this TailCall instance.
     */
    public static class Call<R> extends TailCall<R> {
        // Supplier containing the definition for the next function/method call
        private final Supplier<TailCall<R>> task;

        /**
         * Private constructor for only allowing creation through factory methods
         *
         * @param task the definition/description for the suspended function/method call wrapped inside
         *             a Supplier, so it can be invoked at any time
         */
        private Call(Supplier<TailCall<R>> task) {
            this.task = task;
        }

        @Override
        public Result<R> eval() {
            TailCall<R> tc = this;
            while (tc.isCall()) {
                tc = tc.execute();
            }

            return tc.eval();
        }

        @Override
        public TailCall<R> execute() {
            return task.get();
        }

        @Override
        public boolean isCall() {
            return true;
        }
    }

    /**
     * Subclass of TailCall to terminate the recursion, if the termination conditions
     * have been met and holding the last return value, so it can be passed to the
     * original, external as the result of the whole iteration process caller.
     * <p>
     * The value is again wrapped inside a Result, so even void methods can be
     * mapped by this class.
     *
     * @param <R> type of the return value to be held
     */
    public static class Return<R> extends TailCall<R> {
        // The return value held by this instance
        private final Result<R> value;

        /**
         * Private constructor, to only allow object creation through the use of
         * factory methods.
         *
         * @param value the return value that should be passed down
         */
        private Return(Result<R> value) {
            this.value = value;
        }

        @Override
        public Result<R> eval() {
            return value;
        }

        @Override
        public TailCall<R> execute() {
            throw new IllegalStateException("Return can not execute anything!");
        }

        @Override
        public boolean isCall() {
            return false;
        }
    }

    /**
     * Factory method for creating an instance of TailCall.Call, so it will represent a
     * suspended, recursive function/method call which can be executed.
     *
     * @param task Supplier for the next recursive function/method call
     * @param <R>  type of the return value, that the function/method returns, which is
     *             mapped by this instance
     * @return instance of TailCall.Call
     */
    public static <R> TailCall<R> call(Supplier<TailCall<R>> task) {
        return new TailCall.Call<>(task);
    }

    /**
     * Factory method to create an instance of TailCall.Return, so it will represent a
     * returned value without a recursive function/method call, usually the last
     * recursive step until the termination condition had been met.
     *
     * @param value the value that the function/method returned wrapped in a Result,
     *              or for a void function/method a Result.Empty
     * @param <R>   type of the returned value
     * @return instance of TailCall.Return
     */
    public static <R> TailCall<R> ret(Result<R> value) {
        return new Return<>(value);
    }
}
