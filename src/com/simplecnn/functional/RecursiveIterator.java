package com.simplecnn.functional;

import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * @param <E>
 * @param <T>
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class RecursiveIterator<E, T> {
    private final Function<T, T> next;
    private final BiFunction<E, T, Result<E>> f;

    private RecursiveIterator(Function<T, T> next, BiFunction<E, T, Result<E>> f) {
        this.next = next;
        this.f = f;
    }


    public TailCall<E> iterate(E acc, T it) {
        return f.apply(acc, it)
                .map(v -> TailCall.call(() -> iterate(v, next.apply(it))))
                .getOrElse(TailCall.ret(Result.of(acc)));
    }

    public E eval(E initial, T itStart) {
        return iterate(initial, itStart).eval().getOrElse(initial);
    }

    public static <E, T> RecursiveIterator<E, T> of(
            Function<T, T> next,
            BiFunction<E, T, Result<E>> f) {
        return new RecursiveIterator<>(next, f);
    }
}
