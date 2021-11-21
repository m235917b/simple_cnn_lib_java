package com.simplecnn.functional;

/**
 * A simple generic immutable pair
 *
 * @param <F> type of first element
 * @param <S> type of second element
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class Pair<F, S> {
    private final F fst;
    private final S snd;

    private Pair(F fst, S snd) {
        this.fst = fst;
        this.snd = snd;
    }

    public F getFst() {
        return fst;
    }

    public S getSnd() {
        return snd;
    }

    public static <F, S> Pair<F, S> of(F fst, S snd) {
        return new Pair<>(fst, snd);
    }
}
