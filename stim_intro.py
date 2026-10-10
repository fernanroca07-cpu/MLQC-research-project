import numpy as np
import stim

def bell_circuit(bit_flip_probability: float = 0.0) -> stim.Circuit:
    if not 0 <= bit_flip_probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    circuit = stim.Circuit()
    circuit.append("R", [0,1])
    circuit.append("H", [0])
    circuit.append("CX", [0, 1])
    circuit.append("X_ERROR", [0], bit_flip_probability)
    circuit.append("M", [0,1])
    return circuit

def sample_bell(p: float, shots: int=10_000) -> np.ndarray:
    # Returns a Boolean array of shape
    return bell_circuit(p).compile_sampler(seed=42).sample(shots=shots)

def bell_detection_events(p: float, shots: int=10_000) -> np.ndarray:
    # Detects violations of the ideal even measurement parity
    circuit = bell_circuit(p)
    circuit.append("DETECTOR", [stim.target_rec(-2), stim.target_rec(-1)])
    return circuit.compile_detector_sampler(seed=42).sample(shots=shots)

def make_qec_dataset(shots: int=1_000):
    # Generates syndromes and logical-flip labels for decoder experiments
    circuit = stim.Circuit.generated( "surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.001)
    syndromes, logical_flips = circuit.compile_detector_sampler(seed=42).sample(shots=shots, separate_observables=True)
    return circuit, syndromes, logical_flips

def main():
    print("Stim version:", stim.__version__)
    clean = sample_bell(0)
    print("Ideal bell measurements (first 10): \n", clean[:10].astype(int))
    assert np.all(clean[:,0] == clean[:, 1])

    noisy = sample_bell(0.1)
    print("Noisy Bell mismatch fraction:", np.mean(noisy[:,0] != noisy[:,1]))
    events = bell_detection_events(0.1)
    print("Detector firing fraction: ", events.mean())
    assert not bell_detection_events(0).any()
    assert bell_detection_events(1).all()

    simulator = stim.TableauSimulator()
    simulator.h(0)
    simulator.cx(0, 1)
    print("Bell <ZZ>:", simulator.peek_observable_expectation(stim.PauliString("ZZ")))
    print("Bell <XX>:", simulator.peek_observable_expectation(stim.PauliString("XX")))

    circuit, features, labels = make_qec_dataset()
    print("Syndrome features:", features.shape)
    print("Logical-flip labels:", labels.shape)
    print("Detector error model (first 5 lines): ")
    print("\n". join(str(circuit.detector_error_model(decompose_errors=True)).splitlines()[:5]))

if __name__ == "__main__":
    main()