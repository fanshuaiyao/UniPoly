"""
Download all 22 ADMET datasets from Therapeutics Data Commons (TDC)
and convert them to the CSV format expected by LLM-MPP.

Usage:
    pip install PyTDC
    python download_admet.py
"""
import os
os.environ["TDC_DATA_SOURCE"] = "huggingface"
import pandas as pd

# All 22 ADMET datasets: (name, tdc_module, task_type)
ADMET_DATASETS = [
    # Absorption
    ("Caco2_Wang", "ADME", "regression"),
    ("HIA_Hou", "ADME", "classification"),
    ("Pgp_Broccatelli", "ADME", "classification"),
    ("Bioavailability_Ma", "ADME", "classification"),
    ("Lipophilicity_AstraZeneca", "ADME", "regression"),
    ("Solubility_AqSolDB", "ADME", "regression"),
    # Distribution
    ("BBB_Martins", "ADME", "classification"),
    ("PPBR_AZ", "ADME", "regression"),
    ("VDss_Lombardo", "ADME", "regression"),
    # Metabolism
    ("CYP2D6_Veith", "ADME", "classification"),
    ("CYP3A4_Veith", "ADME", "classification"),
    ("CYP2C9_Veith", "ADME", "classification"),
    ("CYP2D6_Substrate_CarbonMangels", "ADME", "classification"),
    ("CYP3A4_Substrate_CarbonMangels", "ADME", "classification"),
    ("CYP2C9_Substrate_CarbonMangels", "ADME", "classification"),
    ("Half_Life_Obach", "ADME", "regression"),
    ("Clearance_Microsome_AZ", "ADME", "regression"),
    ("Clearance_Hepatocyte_AZ", "ADME", "regression"),
    # Toxicity
    ("hERG", "Tox", "classification"),
    ("AMES", "Tox", "classification"),
    ("DILI", "Tox", "classification"),
    ("LD50_Zhu", "Tox", "regression"),
]


def download_and_save(name, module, task_type, data_root):
    """Download a single TDC dataset and save as project-format CSV."""
    if module == "ADME":
        from tdc.single_pred import ADME
        data = ADME(name=name)
    elif module == "Tox":
        from tdc.single_pred import Tox
        data = Tox(name=name)
    else:
        raise ValueError(f"Unknown module: {module}")

    df = data.get_data()

    # Save raw data as-is
    dataset_dir = os.path.join(data_root, name)
    os.makedirs(dataset_dir, exist_ok=True)
    output_path = os.path.join(dataset_dir, f"{name}.csv")
    df.to_csv(output_path, index=False)

    return len(df), output_path


def main():
    # Data root is ../data/ relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, "..", "data")
    os.makedirs(data_root, exist_ok=True)

    print(f"Saving datasets to: {os.path.abspath(data_root)}")
    print(f"Total datasets to download: {len(ADMET_DATASETS)}")
    print("=" * 70)

    results = []
    for i, (name, module, task_type) in enumerate(ADMET_DATASETS, 1):
        print(f"\n[{i}/{len(ADMET_DATASETS)}] Downloading {name} ({module}, {task_type})...")
        try:
            n_samples, path = download_and_save(name, module, task_type, data_root)
            results.append((name, task_type, n_samples, "OK"))
            print(f"  -> {n_samples} samples saved to {path}")
        except Exception as e:
            results.append((name, task_type, 0, f"FAILED: {e}"))
            print(f"  -> FAILED: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<40} {'Task':<16} {'Samples':>8} {'Status'}")
    print("-" * 70)
    for name, task_type, n_samples, status in results:
        print(f"{name:<40} {task_type:<16} {n_samples:>8} {status}")

    ok_count = sum(1 for r in results if r[3] == "OK")
    print("-" * 70)
    print(f"Successfully downloaded: {ok_count}/{len(ADMET_DATASETS)}")


if __name__ == "__main__":
    main()
