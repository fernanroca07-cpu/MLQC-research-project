import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim
from surface_code import make_surface_circuit

def wilson(errors, shots):
    rate = errors / shots
    z = 1.96
    den = 1 + z*z/shots
    mid = (rate + z*z/(2*shots))/den
    half = z*np.sqrt(rate*(1-rate)/shots + z*z/(4*shots**2))/den
    return max(0, mid-half), min(1, mid+half)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", nargs="+", type=int, default=[3,5,7])
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--points", type=int, default=13)
    ap.add_argument("--p-min", type=float, default=0.001)
    ap.add_argument("--p-max", type=float, default=0.02)
    ap.add_argument("--basis", choices=["x", "z"], default="z")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=Path("threshold_results"))
    args=ap.parse_args()
    distances = sorted(set(args.distances))
    if (len(distances) < 2 or any(d < 3 or d % 2 == 0 for d in distances) or args.shots < 1 or args.points < 2 or not 0 < args.p_min < args.p_max < 0.5):
        ap.error("Use >= 2 odd distances >= 3, positive shots, >= 2 points, 0<p-min<p-max<0.5")
    args.output.mkdir(parents=True, exist_ok=True)
    ps = np.geomspace(args.p_min, args.p_max, args.points)
    rows = []
    seeds = np.random.SeedSequence(args.seed).spawn(len(distances)*len(ps))
    metadata = dict(vars(args), output=str(args.output), rounds="distance", stim_verison=stim.__version__, pymatching_version=pymatching.__version__, noise="All four generator noise probabilities equal p; standard MWPM", metric="Logical failure probability per full experiment; rounds=d")
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2)) 
    with (args.output / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["distance", "rounds", "p", "shots", "errors", "rate", "low", "high"])
        writer.writeheader()
        for d in distances:
            for p in ps:
                c = make_surface_circuit(d, d, float(p), args.basis)
                decoder = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
                seed = int(seeds[len(rows)].generate_state(1, dtype=np.uint64)[0])
                sampler = c.compile_detector_sampler(seed=seed)
                errors = 0
                for start in range(0, args.shots, 5000):
                    events, actual = sampler.sample(min(5000, args.shots-start), separate_observables=True)
                    errors += int(np.count_nonzero(np.any(decoder.decode_batch(events) != actual, axis=1)))
                low, high = wilson(errors, args.shots)
                row = dict(distance=d, rounds=d, p=float(p), shots=args.shots, errors=errors, rate=errors/args.shots, low=low, high=high)
                rows.append(row)
                writer.writerow(row)
                f.flush()
                print(f"d={d} p={p:.5f}: {errors}/{args.shots} = {row['rate']:.6f}", flush=True)

    
    fig, ax = plt.subplots(figsize=(9,6))
    for d in distances:
        subset = [r for r in rows if r["distance"] == d]
        rates = np.array([r["rate"] for r in subset])
        lo = np.array([r["low"] for r in subset])
        hi = np.array([r["high"] for r in subset])
        # Linear y axis keeps zero-error observations visible
        ax.errorbar(ps, rates, yerr=[rates-lo, hi-rates], marker="o", capsize=3, label=f"d={d}, rounds={d}")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Physical noise probability per p per configured location")
    ax.set_ylabel("Logical failure probability per complete experiment")
    ax.set_title(f"Rotated surface-code memory ({args.basis.upper()})  PyMatching")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.text(0.5, 0.015, "95% Wilson intervals, Crossings are finite-distance evidence, not a precise asymptotic threshold", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(args.output / "threshold.png", dpi=180)
    fig.savefig(args.output / "threshold.pdf")
    plt.close(fig)
    lines = ["Adjacent-distance crossing brackets (point estimates only):"]
    for a, b in zip(distances[:-1], distances[1:]):
        ra = [r["rate"] for r in rows if r["distance"] == a]
        rb = [r["rate"] for r in rows if r["distance"] == b]
        diff = np.array(rb)-np.array(ra)
        brackets = [(ps[i], ps[i+1]) for i in range(len(ps)-1) if diff[i]*diff[i+1] < 0]
        lines.append(f"d={a} vs d={d}:" + (str(brackets) if brackets else "No strict sign-change bracket resolved."))
    lines.append("Refine p spacing, increase shots, and test larger distances before quoting a threshold.")
    (args.output / "crossings.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Saved plot, CSV, metadata, and crossing brackets to {args.output.resolve()}")

if __name__ == "__main__":
    main()
