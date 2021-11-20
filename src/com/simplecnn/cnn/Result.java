package com.simplecnn.cnn;

public abstract class Result<V> {
    public abstract boolean isEmpty();

    public abstract V getOrElse(V defaultValue);

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
    }

    public static <V> Result<V> of(V value) {
        return new Success<>(value);
    }

    public static <V> Result<V> empty() {
        return new Empty<>();
    }
}
