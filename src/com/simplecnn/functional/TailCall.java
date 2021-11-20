package com.simplecnn.functional;

import java.util.Iterator;
import java.util.function.BiFunction;
import java.util.function.Supplier;

/**
 * @param <R>
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public abstract class TailCall<R> {

    public abstract Result<R> eval();

    public abstract TailCall<R> execute();

    public abstract boolean isCall();

    public static class Call<R> extends TailCall<R> {
        private final Supplier<TailCall<R>> task;

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

    public static class Return<R> extends TailCall<R> {
        private final Result<R> value;

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

    public static <R> TailCall<R> call(Supplier<TailCall<R>> task) {
        return new TailCall.Call<>(task);
    }

    public static <R> TailCall<R> ret(Result<R> value) {
        return new Return<>(value);
    }

    public static <R, S> TailCall<R> recIt(R acc, Iterator<S> it, BiFunction<R, S, R> f) {
        return it.hasNext()
                ? TailCall.call(() -> recIt(f.apply(acc, it.next()), it, f))
                : TailCall.ret(Result.of(acc));
    }
}
