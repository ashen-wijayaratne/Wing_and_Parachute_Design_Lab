#!/usr/bin/env python3
"""
Enhanced XFOIL batch runner for NACA airfoil identification
Gathers comprehensive aerodynamic data for comparison with experimental results
"""

import shutil
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
import math
import sys
import signal
import csv

# ---------- user-configurable ------------------------------------
RE = 6297342
MACH = 0.2536
ALPHA_START = -4.0
ALPHA_END = 18.0
ALPHA_STEP = 2.0
ITER = 80
TIMEOUT = 120
NUM_WORKERS = min(4, cpu_count())

XF_PATH = shutil.which("xfoil") or shutil.which("xfoil.exe")

OUTDIR = Path("xfoil_comprehensive_outputs")
POLAR_DIR = OUTDIR / "polars"
RESULTS_CSV = OUTDIR / "airfoil_data.csv"
FAILED_FILE = OUTDIR / "failed_runs.txt"

# Minimum number of data points required (changed from requiring all 12)
MIN_DATA_POINTS = 10

# Target angles from your experimental data
TARGET_ANGLES = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]

if XF_PATH is None:
    raise SystemExit("xfoil executable not found in PATH.")

M_RANGE = range(2, 7)  # 2 to 6
P_RANGE = range(2, 7)  # 2 to 6
TT_RANGE = range(12, 19)  # 12 to 18 - This ensures structural viability while avoiding the severe drag penalties of very thick airfoils 
# ------------------------------------------------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)
POLAR_DIR.mkdir(parents=True, exist_ok=True)

if FAILED_FILE.exists():
    FAILED_FILE.unlink()

def run_single(foil_code: str):
    """Run XFOIL for one foil and return comprehensive aerodynamic data at target angles."""
    polar_file = POLAR_DIR / f"{foil_code}_Re{RE}_polar.txt"
    
    cmds = [
        f"naca {foil_code}",
        "pane",
        "oper",
        f"visc {RE}",
        f"mach {MACH}",
        f"iter {ITER}",
        "pacc",
        str(polar_file),
        "",
        f"aseq {ALPHA_START} {ALPHA_END} {ALPHA_STEP}",
        "pacc",
        "",
        "quit"
    ]
    input_data = "\n".join(cmds) + "\n"

    try:
        proc = subprocess.run([XF_PATH], input=input_data, text=True, capture_output=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return (foil_code, None)

    if not polar_file.exists() or polar_file.stat().st_size == 0:
        return (foil_code, None)

    # Parse polar file and extract data at target angles
    airfoil_data = {}
    
    try:
        with open(polar_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        # Find the start of data table (after header)
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("---"):
                start_idx = i + 1
                break
        
        # Parse data lines
        for line in lines[start_idx:]:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    alpha = float(parts[0])
                    # Only store data for our target angles
                    if alpha in TARGET_ANGLES:
                        airfoil_data[alpha] = {
                            'CL': float(parts[1]),
                            'CD': float(parts[2]),
                            'CDp': float(parts[3]),
                            'CM': float(parts[4]),
                            'Top_Xtr': float(parts[5]),
                            'Bot_Xtr': float(parts[6])
                        }
                except (ValueError, IndexError):
                    continue
                    
    except Exception as e:
        print(f"Error parsing {foil_code}: {e}")
        return (foil_code, None)

    # Check if we have enough data points (changed from requiring all 12)
    if len(airfoil_data) < MIN_DATA_POINTS:
        print(f"Warning: {foil_code} only has {len(airfoil_data)}/{len(TARGET_ANGLES)} target angles (minimum {MIN_DATA_POINTS} required)")
        return (foil_code, None)
    
    # If we have at least MIN_DATA_POINTS but not all, note it
    if len(airfoil_data) < len(TARGET_ANGLES):
        print(f"Note: {foil_code} has {len(airfoil_data)}/{len(TARGET_ANGLES)} angles (partial data accepted)")

    return (foil_code, airfoil_data)

def task_from_tuple(t):
    m, p, tt = t
    foil = f"{m}{p}{tt:02d}"
    return run_single(foil)

def write_results_to_csv(results):
    """Write all airfoil data to a CSV file for analysis"""
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Airfoil', 'Alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        rows_written = 0
        for foil_code, data in results:
            if data is not None:
                for alpha in TARGET_ANGLES:
                    if alpha in data:
                        row_data = data[alpha].copy()
                        row_data['Airfoil'] = foil_code
                        row_data['Alpha'] = alpha
                        writer.writerow(row_data)
                        rows_written += 1
        
        return rows_written

def sigint_handler(signum, frame):
    raise KeyboardInterrupt

if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)
    tasks = [(m, p, tt) for m in M_RANGE for p in P_RANGE for tt in TT_RANGE]
    pool = Pool(NUM_WORKERS) if NUM_WORKERS > 1 else None
    results = []
    succ = 0
    total = len(tasks)

    print(f"Running XFOIL analysis for {total} airfoils...")
    print(f"Target angles: {TARGET_ANGLES}")
    print(f"Reynolds: {RE}, Mach: {MACH}")

    try:
        if pool:
            it = pool.imap_unordered(task_from_tuple, tasks)
            for res in it:
                if res is None:
                    continue
                foil, data = res
                if data is None:
                    with open(FAILED_FILE, "a") as f:
                        f.write(f"{foil}\n")
                    continue
                succ += 1
                print(f"✓ {foil}: Success ({len(data)} angles)")
                results.append((foil, data))
        else:
            # serial fallback
            for t in tasks:
                res = task_from_tuple(t)
                foil, data = res
                if data is None:
                    with open(FAILED_FILE, "a") as f:
                        f.write(f"{foil}\n")
                    continue
                succ += 1
                print(f"✓ {foil}: Success ({len(data)} angles)")
                results.append((foil, data))
                
    except KeyboardInterrupt:
        print("\nInterrupted by user. Terminating workers.", file=sys.stderr)
        if pool:
            pool.terminate()
            pool.join()
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        if pool:
            pool.terminate()
            pool.join()
        sys.exit(2)
    finally:
        if pool:
            pool.close()
            pool.join()

    # Write results to CSV
    if results:
        try:
            RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
            write_results_to_csv(results)
            print(f"\nResults written to: {RESULTS_CSV}")
        except PermissionError:
            print(f"\nError: Permission denied when writing to {RESULTS_CSV}")
            sys.exit(3)
        except OSError as e:
            print(f"\nError writing results to CSV: {e}")
            sys.exit(4)
    else:
        print("\nNo results to write to CSV file.")

    print(f"\nCompleted. Successful: {succ}/{total}")
    print(f"Output directory: {OUTDIR.resolve()}")