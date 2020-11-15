all: preprocess compute

preprocess:
	chmod +x preprocess.py
	./preprocess.py data/airline_delay.csv --target=arr_delay --scale=minmax

compute:
	cargo build --release
	./target/release/regression