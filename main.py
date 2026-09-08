"""
Smart Crop Recommendation System - Orchestrator & CLI (main.py)
---------------------------------------------------------------
Responsibility:
- Ties the pipeline together: Data Ingestion -> Model Fitting -> Multi-Crop Inference.
- Calculates continuous suitability percentages for all 22 crop classes.
- Supports both interactive Tkinter desktop GUI (default) and terminal CLI mode (--cli).
"""

import sys
import argparse

# Ensure standard output doesn't crash on Windows terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data_prep import load_and_preprocess_data, FEATURE_NAMES, FEATURE_METADATA
from model_trainer import train_gaussian_nb, predict_crop, format_crop_name

# Practical sample test inputs (Rice/Jute conditions)
SAMPLE_INPUT = [80.0, 48.0, 40.0, 24.0, 82.0, 6.4, 236.0]


def print_banner():
    banner = """
================================================================================
               🌱 SMART CROP RECOMMENDATION SYSTEM
           AI-Powered Agricultural Suitability & Soil Analysis
================================================================================
"""
    print(banner)


def print_prediction_result(result):
    print("\n" + "=" * 65)
    print("                CROP RECOMMENDATION RESULT")
    print("=" * 65)
    print("  Input Environmental & Soil Parameters:")
    for feat, val in result["input_features"].items():
        meta = FEATURE_METADATA[feat]
        print(f"    - {meta['label']:<20}: {val:>8.2f} {meta['unit']}")
    print("-" * 65)
    disp_top = format_crop_name(result['recommended_crop'])
    print(f"  TOP RECOMMENDED CROP : {disp_top.upper()}")
    print(f"  MATCH SUITABILITY    : {result['confidence']:.1f}%")
    print("-" * 65)
    print("  Related Matchable Crops & Suitability Breakdown:")
    for idx, (crop, pct, badge) in enumerate(result["top_candidates"], 1):
        bar_len = int(pct / 2.5)  # 40 chars max
        bar = "█" * bar_len
        disp_name = format_crop_name(crop)
        print(f"    {idx}. {disp_name:<24} : {pct:>5.1f}% | {bar:<40} [{badge}]")
    print("-" * 65)
    print(f"  💡 Insight: {result['insight']}")
    print("=" * 65 + "\n")


def run_interactive_cli(model):
    print("\n--- Interactive Terminal Mode ---")
    print("Enter values for the 7 continuous parameters (press Enter to use defaults):")
    user_inputs = []
    for feat, default_val in zip(FEATURE_NAMES, SAMPLE_INPUT):
        meta = FEATURE_METADATA[feat]
        prompt = f"  {meta['label']} [{meta['min']} - {meta['max']} {meta['unit']}] (default {default_val}): "
        user_val = input(prompt).strip()
        if not user_val:
            user_inputs.append(default_val)
        else:
            try:
                user_inputs.append(float(user_val))
            except ValueError:
                print(f"    Invalid number! Using default {default_val}")
                user_inputs.append(default_val)

    res = predict_crop(model, user_inputs)
    print_prediction_result(res)


def main():
    parser = argparse.ArgumentParser(
        description="Smart Crop Recommendation System"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in terminal CLI mode without opening the Tkinter GUI",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for custom values directly in the terminal",
    )
    parser.add_argument(
        "--gui-only",
        action="store_true",
        help="Launch Tkinter GUI immediately without printing detailed evaluation",
    )
    args = parser.parse_args()

    # Step 1: Banner
    print_banner()

    # Step 2: Ingest and split data
    print("[1/3] Loading agricultural dataset & preparing train-test split...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"      Loaded 2,200 verified samples across 22 crop classes.")

    # Step 3: Train and evaluate model
    print("\n[2/3] Training Gaussian Naive Bayes Model & Profiling Crops...")
    model, accuracy, report = train_gaussian_nb(
        X_train, X_test, y_train, y_test, verbose=not args.gui_only
    )

    # Step 4: Run validation inference
    print("\n[3/3] Running sample inference & suitability analysis...")
    sample_res = predict_crop(model, SAMPLE_INPUT)
    print_prediction_result(sample_res)

    # Step 5: Interface launch
    if args.interactive:
        run_interactive_cli(model)
    elif args.cli:
        print("Pipeline execution complete. Running in CLI mode as requested.")
    else:
        # Default: Launch Tkinter GUI
        print("Launching Graphical Interface...")
        try:
            from gui_app import launch_gui
            launch_gui(model=model)
        except Exception as e:
            print(f"Notice: Tkinter interface could not open ({e}). Falling back to CLI mode.")
            run_interactive_cli(model)


if __name__ == "__main__":
    main()

