package com.simplecnn.functional;

import java.util.function.Function;

/**
 * @param <V>
 * @author Marvin Bergmann
 */
@SuppressWarnings("unused")
public abstract class Result<V> {
    public abstract boolean isEmpty();

    public abstract V getOrElse(V defaultValue);

    public abstract <T> Result<T> map(Function<V, T> f);

    public static class Success<V> extends Result<V> {
        private final V value;

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
        public <T> Result<T> map(Function<V, T> f) {
            return new Success<>(f.apply(value));
        }
    }

    public static class Empty<V> extends Result<V> {
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
        public <T> Result<T> map(Function<V, T> f) {
            return new Empty<>();
        }
    }

    public static <V> Result<V> of(V value) {
        return new Success<>(value);
    }

    public static <V> Result<V> empty() {
        return new Empty<>();
    }
}
