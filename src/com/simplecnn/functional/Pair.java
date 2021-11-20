package com.simplecnn.functional;

/**
 * @param <F>
 * @param <S>
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public class Pair<F, S> {
    private final F fst;
    private final S snd;

    public Pair(F fst, S snd) {
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
