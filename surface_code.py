import argparse
from pathlib import Path

import numpy as np
import pymatching
import stim

def make_surface_circuit(distance=3, rounds=3, p=0.001, basis="z"):
    if distance < 3 or distance % 2 == 0:
        raise ValueError("Use an odd distance >= 3")
    if rounds < 1 or not 0 <= p <= 1 or basis not in ("x", "z"):
        raise ValueError("Require rounds >= 1, 0 <= p <= 1, basis x or z")
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        # These are separate locations
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
    )

def run_experiment(circuit, shots=10_000, seed=None):
    if shots < 1:
        raise ValueError("shots must be positive")
    model = circuit.detector_error_model(decompose_errors=True)
    decoder = pymatching.Matching.from_detector_error_model(model)
    events, actual = circuit.compile_detector_sampler(seed=seed).sample(shots=shots, separate_observables=True)
    # The decoder sees only detection events, never the actual logical flips
    predicted = decoder.decode_batch(events)
    failed = np.any(predicted != actual, axis=1)
    return events, actual, predicted, failed

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--basis", choices=("x", "z"), default="z")
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-circuit", type=Path)
    args = parser.parse_args()
    circuit = make_surface_circuit(args.distance, args.rounds, args.p, args.basis)
    events, actual, predicted, failed = run_experiment(circuit, args.shots, args.seed)
    if args.save_circuit:
        circuit.to_file(args.save_circuit)
    print(f"Rotated surface code: d={args.distance}, rounds={args.rounds}, " f"basis={args.basis.upper()}, noise probability per location={args.p}")
    print("First 20 shots (detectors are parity violations across space/time):")
    for i in range(min(20, args.shots)):
        print(f"{i+1:3}: fired={np.flatnonzero(events[i]).tolist()}, "
              f"actual flip={int(actual[i, 0])}, "
              f"predicted flip={int(predicted[i,0])}, logical error={bool(failed[i])}")
    count = int(failed.sum())
    rate = count / args.shots
    # Wilson binomial interval remains meaningful even with no failures
    z = 1.96
    denom = 1 + z*z/args.shots
    center = (rate + z*z/(2*args.shots))/denom
    half = z*np.sqrt(rate*(1-rate)/args.shots + z*z/(4*args.shots**2))/denom
    print(f"Logical failures: {count}/{args.shots} = {rate:.6f}")
    print(f"95% Wilson interval: [{max(0, center-half):.6f}, {min(1, center+half):.6f}]")
    print("Rate is per full memory experiment, not per round.")

if __name__ == "__main__":
    main()
