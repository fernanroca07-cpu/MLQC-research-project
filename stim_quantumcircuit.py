import stim

# Syndrome bits are 1 for a -1 stabilizer eigenvalue
RECOVERY = {(0,0): None, (1,0): 0, (1,1): 1, (0,1): 2}

def make_circuit(p=0.05, logical_state="0", forced_errors=None):
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    if logical_state not in ("0", "1", "+", "-"):
        raise ValueError("logical_state must be 0, 1, +, or -")
    c = stim.Circuit("R 0 1 2 3 4")
    if logical_state in ("1", "-"):
        c.append("X", [0])
    if logical_state in ("+", "-"):
        c.append("H", [0])
    c.append("CX", [0, 1, 0, 2])
    c.append("TICK")
    if forced_errors is None:
        c.append("X_ERROR", [0, 1, 2], p)
    else:
        for q in forced_errors:
            if q not in (0, 1, 2):
                raise ValueError("Errors must target data qubits 0, 1, or 2")
            c.append("X", [q])
    c.append("TICK")
    # Measure Z0 Z1 and Z1 Z2 without measuring the logical state
    c.append("CX", [0, 3, 1, 3, 1, 4, 2, 4])
    c.append("M", [3, 4])
    return c

def correct(circuit, seed=None):
    sim = stim.TableauSimulator(seed=seed)
    sim.do(circuit)
    syndrome = tuple(map(int, sim.current_measurement_record()[-2:]))
    qubit = RECOVERY[syndrome]
    executed = circuit.copy()
    if qubit is not None:
        recovery = stim.Circuit()
        recovery.append("X", [qubit])
        sim.do(recovery)
        executed += recovery
    return sim, syndrome, qubit, executed


def verify_single_errors():
    for state in ("0", "1", "+", "-"):
        for errors in ((), (0,), (1,), (2,)):
            sim, _, _, _ = correct(make_circuit(logical_state=state, forced_errors=errors))
            for stabilizer in ("ZZI", "IZZ"):
                assert sim.peek_observable_expectation(stim.PauliString(stabilizer)) == 1
            observable = "ZII" if state in ("0", "1") else "XXX"
            expected = 1 if state in ("0", "+") else -1
            assert sim.peek_observable_expectation(stim.PauliString(observable)) == expected


def main():
    verify_single_errors()
    print("Verified recovery of all single X errors for logical 0, 1, +, -")
    p = 0.1
    shots = 1_000
    circuit = make_circuit(p=p)
    print(f"\nRandom noise circuit (p={p} per data qubit): \n{circuit}")
    print("\nFirst 20 shots (logical input 0):")
    failures = 0
    for shot in range(shots):
        sim, syndrome, qubit, _ = correct(circuit)
        # After recovery, Z on data qubit 0 reads the logical Z value
        failed = sim.peek_observable_expectation(stim.PauliString("ZII")) == -1
        failures += int(failed)
        if shot < 20:
            recovery = "none" if qubit is None else f"X on {qubit}"
            print(f"{shot + 1:3}: syndrome={syndrome}, correction={recovery}, " f'logical error={failed}')
    print(f"\nObserved logical error rate: {failures}/{shots} = {failures/shots:.4f}")
    print(f"Expected logical error rate: {3*p*p*(1-p) + p**3:.4f}")
    for p in (0.001, 0.01, 0.05, 0.1):
        # Majority recovery fails when two or three data qubits flip
        print(f"{p:10.3f} {3*p*p*(1-p) + p**3:.6f}")


if __name__ == "__main__":
    main()