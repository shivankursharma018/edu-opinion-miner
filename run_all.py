"""
RUN_ALL.PY — Master Script
===========================
Runs the entire pipeline from scratch:
  Step 1: Create & preprocess dataset
  Step 2: Demonstrate aspect extraction
  Step 3: Train all models
  Step 4: Compare with baselines

After this, launch the dashboard with:
  streamlit run dashboard.py
"""

import os
import sys


def run_step(step_file, step_name):
    print("\n" + "█" * 60)
    print(f"  {step_name}")
    print("█" * 60)
    result = os.system(f"cd {os.path.dirname(__file__)} && python {step_file}")
    if result != 0:
        print(f"❌ {step_name} failed. Check errors above.")
        sys.exit(1)
    print(f"✅ {step_name} complete.")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    run_step("scripts/step1_prepare_data.py", "STEP 1: Data Preparation")
    run_step("scripts/step2_aspect_extraction.py", "STEP 2: Aspect Extraction")
    run_step("scripts/step3_train_models.py", "STEP 3: Model Training")
    run_step("scripts/step4_baseline_comparison.py", "STEP 4: Baseline Comparison")

    print("\n" + "=" * 60)
    print("🎉 ALL STEPS COMPLETE!")
    print("=" * 60)
    print("\nTo launch the dashboard:")
    print("  streamlit run dashboard.py")
    print("\nFiles generated:")
    print("  data/dataset.csv               → raw dataset")
    print("  models/*_model.pkl             → trained models (one per aspect)")
    print("  outputs/model_comparison.csv   → NB vs SVM vs RF results")
    print("  outputs/baseline_comparison.csv → vs TextBlob and VADER")
