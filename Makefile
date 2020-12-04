all: preprocess compute

preprocess:
	chmod +x preprocess.py
	./preprocess.py data/flights.csv --target=DEP_DELAY --scale=minmax --ohe

compute:
	cargo run --release
