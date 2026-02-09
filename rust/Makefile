# Makefile for building Rust extension

.PHONY: build install test clean

build:
	cd rust && cargo build --release

install: build
	cd rust && maturin develop --release

test:
	cd rust && cargo test

clean:
	cd rust && cargo clean
