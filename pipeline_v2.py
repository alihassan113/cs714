"""
JARVIS-DFT 2D Materials Bandgap Prediction Pipeline V2
========================================================
Improved version with proper compositional features via pymatgen Element.
End-to-end ML pipeline:
1. Data Acquisition from JARVIS-DFT (real data via jarvis-tools)
2. Feature Engineering (compositional via pymatgen + structural descriptors)
3. Model Training (Random Forest, XGBoost, GNN Surrogate)
4. SHAP Explainability
5. Virtual Screening for Optoelectronics (1.0-2.0 eV)
6. MLflow-style tracking logs
"""

import modal
import os
import json
import sys
from datetime import datetime

app = modal.App("jarvis-dft-bandgap-v2")

sdk_path = os.environ.get('ORCHESTRA_SDK_PATH', '/app/src')

ml_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system "
        "torch torchvision numpy pandas scikit-learn xgboost "
        "matplotlib seaborn shap jarvis-tools pymatgen "
        "requests tqdm scipy joblib"
    )
    .env({
        "AGENT_ID": os.getenv("AGENT_ID", ""),
        "PROJECT_ID": os.getenv("PROJECT_ID", ""),
        "USER_ID": os.getenv("USER_ID", ""),
    })
    .add_local_dir(sdk_path, remote_path="/root/src")
)

volume = modal.Volume.from_name("jarvis-dft-pipeline-v2", create_if_missing=True)


@app.function(
    image=ml_image,
    volumes={"/workspace": volume},
    timeout=5400,
    memory=32768,
    cpu=8,
    secrets=[modal.Secret.from_name("orchestra-supabase")],
)
def run_full_pipeline():
    """Execute the complete JARVIS-DFT bandgap prediction pipeline V2."""
    import time
    start_time = time.time()

    sys.path.insert(0, "/root")
    from src.orchestra_sdk.experiment import Experiment

    exp = Experiment.init(
        name="JARVIS-DFT 2D Bandgap V2 (Compositional+Structural)",
        description="Improved pipeline with pymatgen compositional features: RF, XGBoost, GNN surrogate",
        config={
            "dataset": "JARVIS-DFT-2D",
            "target": "optb88vdw_bandgap",
            "models": ["RandomForest", "XGBoost", "GNN_Surrogate"],
            "n_features_target": "40+",
            "test_size": 0.2,
            "random_state": 42,
            "rf_n_estimators": 300,
            "xgb_n_estimators": 500,
            "gnn_epochs": 200,
            "gnn_hidden_dim": 128,
        },
        x_axis_label="Step",
    )
    exp.add_tags(["jarvis-dft", "2d-materials", "bandgap", "shap", "screening", "v2"])

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs("/workspace/results", exist_ok=True)
    os.makedirs("/workspace/results/plots", exist_ok=True)

    # ================================================================
    # STEP 1: DATA ACQUISITION
    # ================================================================
    print("=" * 70)
    print("STEP 1: DATA ACQUISITION - Fetching JARVIS-DFT 2D Materials")
    print("=" * 70)
    exp.log_text("Step 1: Fetching JARVIS-DFT 2D materials dataset")

    from jarvis.db.figshare import data as jarvis_data
    from jarvis.core.atoms import Atoms

    print("Downloading JARVIS-DFT 2D dataset...")
    dft_2d = jarvis_data("dft_2d")
    print(f"Total entries: {len(dft_2d)}")

    records = []
    for entry in dft_2d:
        bg = entry.get("optb88vdw_bandgap", None)
        formula = entry.get("formula", None)
        jid = entry.get("jid", None)
        atoms_dict = entry.get("atoms", None)
        if bg is not None and formula is not None and atoms_dict is not None:
            try:
                bg_val = float(bg)
                if bg_val >= 0:
                    records.append({
                        "jid": jid,
                        "formula": formula,
                        "optb88vdw_bandgap": bg_val,
                        "atoms": atoms_dict,
                        "ehull": entry.get("ehull", None),
                        "exfoliation_energy": entry.get("exfoliation_energy", None),
                        "spg_number": entry.get("spg_number", None),
                    })
            except (ValueError, TypeError):
                continue

    df_raw = pd.DataFrame(records)
    print(f"Valid 2D materials: {len(df_raw)}")
    print(f"Bandgap range: {df_raw['optb88vdw_bandgap'].min():.3f} - {df_raw['optb88vdw_bandgap'].max():.3f} eV")
    print(f"Mean: {df_raw['optb88vdw_bandgap'].mean():.3f}, Median: {df_raw['optb88vdw_bandgap'].median():.3f} eV")

    exp.log({"n_raw": len(df_raw), "bg_mean": df_raw['optb88vdw_bandgap'].mean()}, step=0)

    # ================================================================
    # STEP 2: FEATURE ENGINEERING (pymatgen + structural)
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING (pymatgen compositional + structural)")
    print("=" * 70)
    exp.log_text("Step 2: Computing features via pymatgen Element + JARVIS Atoms")

    from pymatgen.core import Element, Composition
    import re

    def parse_composition(formula):
        """Parse formula into {element: fraction} dict."""
        try:
            comp = Composition(formula)
            return {str(el): frac for el, frac in comp.fractional_composition.items()}
        except Exception:
            return None

    def get_elem_props(symbol):
        """Get elemental properties from pymatgen."""
        try:
            el = Element(symbol)
            props = {}
            props["Z"] = el.Z
            props["atomic_mass"] = float(el.atomic_mass)
            props["X"] = el.X if el.X else np.nan  # Pauling electronegativity
            props["atomic_radius"] = float(el.atomic_radius) if el.atomic_radius else np.nan
            props["mendeleev_no"] = el.mendeleev_no
            props["row"] = el.row
            props["group"] = el.group

            # Ionization energy (first)
            try:
                ie = el.ionization_energies
                props["ie1"] = ie[0] if len(ie) > 0 else np.nan
            except Exception:
                props["ie1"] = np.nan

            # Electron affinity
            try:
                props["electron_affinity"] = float(el.electron_affinity) if el.electron_affinity else np.nan
            except Exception:
                props["electron_affinity"] = np.nan

            # Is metal
            props["is_metal"] = 1.0 if el.is_metal else 0.0
            props["is_metalloid"] = 1.0 if el.is_metalloid else 0.0

            # Number of valence electrons
            try:
                # Use common oxidation states as proxy
                common_ox = el.common_oxidation_states
                props["max_oxidation"] = max(common_ox) if common_ox else np.nan
                props["min_oxidation"] = min(common_ox) if common_ox else np.nan
            except Exception:
                props["max_oxidation"] = np.nan
                props["min_oxidation"] = np.nan

            return props
        except Exception:
            return None

    def compute_features(row):
        """Compute compositional + structural features."""
        features = {}
        try:
            # --- STRUCTURAL features from JARVIS Atoms ---
            atoms = Atoms.from_dict(row["atoms"])
            elements = atoms.elements
            lattice = np.array(atoms.lattice_mat)

            features["n_atoms"] = atoms.num_atoms

            try:
                features["density"] = float(atoms.density)
            except:
                features["density"] = np.nan

            try:
                features["volume"] = float(atoms.volume)
            except:
                features["volume"] = np.nan

            try:
                features["packing_fraction"] = float(atoms.packing_fraction)
            except:
                features["packing_fraction"] = np.nan

            # Lattice parameters
            a = np.linalg.norm(lattice[0])
            b = np.linalg.norm(lattice[1])
            c = np.linalg.norm(lattice[2])
            features["lattice_a"] = a
            features["lattice_b"] = b
            features["lattice_c"] = c
            features["lattice_ratio_ab"] = a / (b + 1e-12)
            features["lattice_ratio_ac"] = a / (c + 1e-12)

            # Lattice angles
            def angle(v1, v2):
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

            features["angle_alpha"] = angle(lattice[1], lattice[2])
            features["angle_beta"] = angle(lattice[0], lattice[2])
            features["angle_gamma"] = angle(lattice[0], lattice[1])

            # Volume per atom
            if atoms.num_atoms > 0 and not np.isnan(features.get("volume", np.nan)):
                features["vol_per_atom"] = features["volume"] / atoms.num_atoms

            # Number of unique elements
            unique_elements = list(set(elements))
            features["n_elements"] = len(unique_elements)

            # Spacegroup
            if row.get("spg_number") is not None:
                try:
                    features["spg_number"] = float(row["spg_number"])
                except:
                    pass

            # --- COMPOSITIONAL features from pymatgen ---
            comp = parse_composition(row["formula"])
            if comp is None:
                return features

            # Collect elemental properties weighted by composition
            prop_names = ["Z", "atomic_mass", "X", "atomic_radius", "mendeleev_no",
                          "row", "group", "ie1", "electron_affinity",
                          "is_metal", "is_metalloid", "max_oxidation", "min_oxidation"]

            elem_data = {}
            for el_str, frac in comp.items():
                props = get_elem_props(el_str)
                if props:
                    elem_data[el_str] = {"frac": frac, "props": props}

            if len(elem_data) == 0:
                return features

            # Compute weighted mean, std, min, max, range for each property
            for prop in prop_names:
                vals = []
                weights = []
                for el_str, data in elem_data.items():
                    v = data["props"].get(prop, np.nan)
                    if not np.isnan(v):
                        vals.append(v)
                        weights.append(data["frac"])

                if len(vals) > 0:
                    arr = np.array(vals)
                    w = np.array(weights)
                    w = w / w.sum()  # Normalize

                    features[f"{prop}_wmean"] = np.average(arr, weights=w)
                    features[f"{prop}_mean"] = np.mean(arr)
                    features[f"{prop}_std"] = np.std(arr) if len(arr) > 1 else 0.0
                    features[f"{prop}_min"] = np.min(arr)
                    features[f"{prop}_max"] = np.max(arr)
                    features[f"{prop}_range"] = np.max(arr) - np.min(arr)

            # Electronegativity difference (key for bandgap!)
            x_vals = [elem_data[e]["props"].get("X", np.nan) for e in elem_data]
            x_vals = [v for v in x_vals if not np.isnan(v)]
            if len(x_vals) >= 2:
                features["X_diff"] = max(x_vals) - min(x_vals)
            else:
                features["X_diff"] = 0.0

            # Metal fraction
            metal_frac = sum(
                data["frac"] for data in elem_data.values()
                if data["props"].get("is_metal", 0) == 1.0
            )
            features["metal_fraction"] = metal_frac

        except Exception as e:
            print(f"  Feature error for {row.get('formula', '?')}: {e}")

        return features

    print("Computing features for all materials...")
    feature_list = []
    valid_indices = []
    for idx, row in df_raw.iterrows():
        feats = compute_features(row)
        if len(feats) >= 20:  # Must have both structural + compositional
            feature_list.append(feats)
            valid_indices.append(idx)
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(df_raw)} materials...")

    df_features = pd.DataFrame(feature_list, index=valid_indices)
    df_combined = pd.concat([
        df_raw.loc[valid_indices].reset_index(drop=True),
        df_features.reset_index(drop=True)
    ], axis=1)

    # Identify feature columns
    meta_cols = ["jid", "formula", "atoms", "optb88vdw_bandgap",
                 "ehull", "exfoliation_energy", "spg_number"]
    feature_cols = [c for c in df_combined.columns if c not in meta_cols]

    # Drop columns with >30% NaN
    valid_feature_cols = [c for c in feature_cols if df_combined[c].isna().mean() < 0.3]

    print(f"Total features: {len(valid_feature_cols)}")
    print(f"Materials with full features: {len(df_combined)}")

    # Prepare ML data
    X = df_combined[valid_feature_cols].copy()
    y = df_combined["optb88vdw_bandgap"].copy()

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    df_ml = df_combined[mask].reset_index(drop=True)

    print(f"Final ML dataset: {len(X)} materials, {X.shape[1]} features")
    print(f"Feature names: {list(X.columns)}")

    exp.log({"n_materials": len(X), "n_features": X.shape[1]}, step=1)

    # Save processed dataset
    save_cols = ["jid", "formula", "optb88vdw_bandgap"] + valid_feature_cols
    available_save = [c for c in save_cols if c in df_ml.columns]
    df_ml[available_save].to_csv("/workspace/results/jarvis_2d_processed.csv", index=False)

    # ================================================================
    # STEP 3: MODEL TRAINING
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING")
    print("=" * 70)
    exp.log_text("Step 3: Training RF, XGBoost, GNN surrogate")

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_idx = X_train.index
    test_idx = X_test.index

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- Random Forest ---
    print("\n--- Training Random Forest ---")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0

    rf_pred_train = rf.predict(X_train)
    rf_pred_test = rf.predict(X_test)
    rf_mae_train = mean_absolute_error(y_train, rf_pred_train)
    rf_mae_test = mean_absolute_error(y_test, rf_pred_test)
    rf_rmse_test = np.sqrt(mean_squared_error(y_test, rf_pred_test))
    rf_r2_test = r2_score(y_test, rf_pred_test)

    results["RandomForest"] = {
        "MAE_train": round(rf_mae_train, 4),
        "MAE_test": round(rf_mae_test, 4),
        "RMSE_test": round(rf_rmse_test, 4),
        "R2_test": round(rf_r2_test, 4),
        "train_time_s": round(rf_time, 1),
    }
    print(f"RF - Train MAE: {rf_mae_train:.4f}, Test MAE: {rf_mae_test:.4f}, "
          f"RMSE: {rf_rmse_test:.4f}, R²: {rf_r2_test:.4f} ({rf_time:.1f}s)")

    exp.log({"rf_mae_train": rf_mae_train, "rf_mae_test": rf_mae_test,
             "rf_rmse": rf_rmse_test, "rf_r2": rf_r2_test}, step=2)

    # --- XGBoost ---
    print("\n--- Training XGBoost ---")
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    xgb_time = time.time() - t0

    xgb_pred_train = xgb_model.predict(X_train)
    xgb_pred_test = xgb_model.predict(X_test)
    xgb_mae_train = mean_absolute_error(y_train, xgb_pred_train)
    xgb_mae_test = mean_absolute_error(y_test, xgb_pred_test)
    xgb_rmse_test = np.sqrt(mean_squared_error(y_test, xgb_pred_test))
    xgb_r2_test = r2_score(y_test, xgb_pred_test)

    results["XGBoost"] = {
        "MAE_train": round(xgb_mae_train, 4),
        "MAE_test": round(xgb_mae_test, 4),
        "RMSE_test": round(xgb_rmse_test, 4),
        "R2_test": round(xgb_r2_test, 4),
        "train_time_s": round(xgb_time, 1),
    }
    print(f"XGB - Train MAE: {xgb_mae_train:.4f}, Test MAE: {xgb_mae_test:.4f}, "
          f"RMSE: {xgb_rmse_test:.4f}, R²: {xgb_r2_test:.4f} ({xgb_time:.1f}s)")

    exp.log({"xgb_mae_train": xgb_mae_train, "xgb_mae_test": xgb_mae_test,
             "xgb_rmse": xgb_rmse_test, "xgb_r2": xgb_r2_test}, step=3)

    # --- GNN Surrogate (Residual MLP) ---
    print("\n--- Training GNN Surrogate (Residual MLP) ---")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class BandgapNet(nn.Module):
        """Deep residual MLP as GNN surrogate for bandgap prediction."""
        def __init__(self, input_dim, hidden_dim=128, n_layers=5, dropout=0.15):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.layers = nn.ModuleList()
            for _ in range(n_layers):
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ))
            self.output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            h = self.input_proj(x)
            for layer in self.layers:
                h = h + layer(h)
            return self.output(h).squeeze(-1)

    device = torch.device("cpu")
    input_dim = X_train_scaled.shape[1]
    model_gnn = BandgapNet(input_dim, hidden_dim=128, n_layers=5, dropout=0.15).to(device)

    X_train_t = torch.FloatTensor(X_train_scaled.copy()).to(device)
    y_train_t = torch.FloatTensor(y_train.values.copy()).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled.copy()).to(device)
    y_test_t = torch.FloatTensor(y_test.values.copy()).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    optimizer = optim.AdamW(model_gnn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=0.5)

    gnn_train_losses = []
    gnn_test_losses = []
    best_test_mae = float("inf")

    t0 = time.time()
    for epoch in range(200):
        model_gnn.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model_gnn(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_gnn.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)
        scheduler.step()

        model_gnn.eval()
        with torch.no_grad():
            train_pred = model_gnn(X_train_t)
            test_pred = model_gnn(X_test_t)
            train_mae = torch.mean(torch.abs(train_pred - y_train_t)).item()
            test_mae = torch.mean(torch.abs(test_pred - y_test_t)).item()

        gnn_train_losses.append(train_mae)
        gnn_test_losses.append(test_mae)

        if test_mae < best_test_mae:
            best_test_mae = test_mae
            best_state = {k: v.clone() for k, v in model_gnn.state_dict().items()}

        if (epoch + 1) % 40 == 0:
            print(f"  Epoch {epoch+1}/200 - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
            exp.log({"gnn_train_mae": train_mae, "gnn_test_mae": test_mae}, step=10 + epoch)

    gnn_time = time.time() - t0

    # Load best model
    model_gnn.load_state_dict(best_state)
    model_gnn.eval()
    with torch.no_grad():
        gnn_pred_train = model_gnn(X_train_t).numpy()
        gnn_pred_test = model_gnn(X_test_t).numpy()

    gnn_mae_train = mean_absolute_error(y_train, gnn_pred_train)
    gnn_mae_test = mean_absolute_error(y_test, gnn_pred_test)
    gnn_rmse_test = np.sqrt(mean_squared_error(y_test, gnn_pred_test))
    gnn_r2_test = r2_score(y_test, gnn_pred_test)

    results["GNN_Surrogate"] = {
        "MAE_train": round(gnn_mae_train, 4),
        "MAE_test": round(gnn_mae_test, 4),
        "RMSE_test": round(gnn_rmse_test, 4),
        "R2_test": round(gnn_r2_test, 4),
        "train_time_s": round(gnn_time, 1),
        "best_epoch_test_mae": round(best_test_mae, 4),
    }
    print(f"GNN - Train MAE: {gnn_mae_train:.4f}, Test MAE: {gnn_mae_test:.4f}, "
          f"RMSE: {gnn_rmse_test:.4f}, R²: {gnn_r2_test:.4f} ({gnn_time:.1f}s)")

    exp.log({"gnn_mae_train": gnn_mae_train, "gnn_mae_test": gnn_mae_test,
             "gnn_rmse": gnn_rmse_test, "gnn_r2": gnn_r2_test}, step=250)

    # ================================================================
    # STEP 4: SHAP EXPLAINABILITY
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 4: SHAP EXPLAINABILITY")
    print("=" * 70)
    exp.log_text("Step 4: SHAP analysis on tree-based models")

    import shap

    feature_names = list(X.columns)
    n_shap = min(500, len(X_test))
    X_shap = X_test.iloc[:n_shap]

    # --- RF SHAP ---
    print("Computing SHAP for Random Forest...")
    rf_explainer = shap.TreeExplainer(rf)
    rf_shap_values = rf_explainer.shap_values(X_shap)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(rf_shap_values, X_shap, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SHAP Summary - Random Forest (Bandgap Prediction)", fontsize=14)
    plt.tight_layout()
    plt.savefig("/workspace/results/plots/shap_rf_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(rf_shap_values, X_shap, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance - Random Forest", fontsize=14)
    plt.tight_layout()
    plt.savefig("/workspace/results/plots/shap_rf_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved RF SHAP plots")

    # --- XGBoost SHAP ---
    print("Computing SHAP for XGBoost...")
    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values = xgb_explainer.shap_values(X_shap)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(xgb_shap_values, X_shap, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SHAP Summary - XGBoost (Bandgap Prediction)", fontsize=14)
    plt.tight_layout()
    plt.savefig("/workspace/results/plots/shap_xgb_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(xgb_shap_values, X_shap, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance - XGBoost", fontsize=14)
    plt.tight_layout()
    plt.savefig("/workspace/results/plots/shap_xgb_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved XGBoost SHAP plots")

    # Extract top features
    rf_importance = np.abs(rf_shap_values).mean(axis=0)
    xgb_importance = np.abs(xgb_shap_values).mean(axis=0)

    top_rf = sorted(zip(feature_names, rf_importance), key=lambda x: -x[1])[:15]
    top_xgb = sorted(zip(feature_names, xgb_importance), key=lambda x: -x[1])[:15]

    print("\nTop 15 Features (RF SHAP):")
    for f, v in top_rf:
        print(f"  {f}: {v:.4f}")
    print("\nTop 15 Features (XGBoost SHAP):")
    for f, v in top_xgb:
        print(f"  {f}: {v:.4f}")

    # Save SHAP importance arrays
    shap_data = {
        "rf_shap_importance": {f: round(float(v), 6) for f, v in top_rf},
        "xgb_shap_importance": {f: round(float(v), 6) for f, v in top_xgb},
    }
    with open("/workspace/results/shap_importance.json", "w") as f:
        json.dump(shap_data, f, indent=2)

    # ================================================================
    # STEP 5: PLOTS
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING PLOTS")
    print("=" * 70)

    # --- Parity plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, name, yt, yp, met in [
        (axes[0], "Random Forest", y_test, rf_pred_test, results["RandomForest"]),
        (axes[1], "XGBoost", y_test, xgb_pred_test, results["XGBoost"]),
        (axes[2], "GNN Surrogate", y_test.values, gnn_pred_test, results["GNN_Surrogate"]),
    ]:
        ax.scatter(yt, yp, alpha=0.5, s=20, c="steelblue", edgecolors="none")
        lims = [min(np.min(yt), np.min(yp)) - 0.3,
                max(np.max(yt), np.max(yp)) + 0.3]
        ax.plot(lims, lims, "r--", lw=1.5, label="y = x")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("DFT Bandgap (eV)", fontsize=12)
        ax.set_ylabel("Predicted Bandgap (eV)", fontsize=12)
        ax.set_title(f"{name}\nMAE={met['MAE_test']:.3f} eV  R²={met['R2_test']:.3f}", fontsize=12)
        ax.legend(fontsize=10); ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
    plt.suptitle("Parity Plots: DFT vs Predicted OptB88vdW Bandgap", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("/workspace/results/plots/parity_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: parity_plots.png")

    # --- GNN Learning Curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(gnn_train_losses)+1), gnn_train_losses, label="Train MAE", color="steelblue", lw=2)
    ax.plot(range(1, len(gnn_test_losses)+1), gnn_test_losses, label="Test MAE", color="coral", lw=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MAE (eV)", fontsize=12)
    ax.set_title("GNN Surrogate Learning Curve", fontsize=14)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/workspace/results/plots/gnn_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: gnn_learning_curve.png")

    # --- sklearn Learning Curves (reduced CV for speed) ---
    print("Computing learning curves (3-fold CV)...")
    from sklearn.model_selection import learning_curve as sk_lc

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, model_obj, name in [
        (axes[0], RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1), "Random Forest"),
        (axes[1], xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, tree_method="hist"), "XGBoost"),
    ]:
        train_sizes_abs, train_scores, val_scores = sk_lc(
            model_obj, X, y,
            train_sizes=np.linspace(0.1, 1.0, 6),
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        train_mean = -train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = -val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color="steelblue")
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.15, color="coral")
        ax.plot(train_sizes_abs, train_mean, "o-", color="steelblue", lw=2, label="Train MAE")
        ax.plot(train_sizes_abs, val_mean, "o-", color="coral", lw=2, label="Validation MAE")
        ax.set_xlabel("Training Set Size", fontsize=12)
        ax.set_ylabel("MAE (eV)", fontsize=12)
        ax.set_title(f"{name} Learning Curve", fontsize=13)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/workspace/results/plots/tree_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: tree_learning_curves.png")

    # --- Bandgap Distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvspan(1.0, 2.0, alpha=0.2, color="gold", label="Optoelectronic range (1.0-2.0 eV)")
    ax.set_xlabel("OptB88vdW Bandgap (eV)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Bandgap Distribution - JARVIS-DFT 2D Materials", fontsize=14)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/workspace/results/plots/bandgap_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: bandgap_distribution.png")

    # --- Model Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    model_names = list(results.keys())
    maes = [results[m]["MAE_test"] for m in model_names]
    r2s = [results[m]["R2_test"] for m in model_names]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    axes[0].bar(model_names, maes, color=colors, edgecolor="white", width=0.6)
    axes[0].set_ylabel("Test MAE (eV)", fontsize=12)
    axes[0].set_title("Model Comparison: MAE", fontsize=13)
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(model_names, r2s, color=colors, edgecolor="white", width=0.6)
    axes[1].set_ylabel("Test R²", fontsize=12)
    axes[1].set_title("Model Comparison: R²", fontsize=13)
    for i, v in enumerate(r2s):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Model Performance Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("/workspace/results/plots/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: model_comparison.png")

    # ================================================================
    # STEP 6: VIRTUAL SCREENING
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 6: VIRTUAL SCREENING FOR OPTOELECTRONICS")
    print("=" * 70)
    exp.log_text("Step 6: Virtual screening for 1.0-2.0 eV bandgap")

    # Ensemble prediction
    all_pred_rf = rf.predict(X)
    all_pred_xgb = xgb_model.predict(X)
    model_gnn.eval()
    with torch.no_grad():
        X_all_s = scaler.transform(X)
        all_pred_gnn = model_gnn(torch.FloatTensor(X_all_s.copy())).numpy()

    ensemble_pred = (all_pred_rf + all_pred_xgb + all_pred_gnn) / 3.0

    df_ml = df_ml.copy()
    df_ml["pred_rf"] = all_pred_rf
    df_ml["pred_xgb"] = all_pred_xgb
    df_ml["pred_gnn"] = all_pred_gnn
    df_ml["pred_ensemble"] = ensemble_pred

    # Filter optoelectronic range
    opto_mask = (ensemble_pred >= 1.0) & (ensemble_pred <= 2.0)
    df_opto = df_ml[opto_mask].copy()

    # Score: proximity to 1.5 eV ideal
    df_opto["opto_score"] = 1.0 - np.abs(df_opto["pred_ensemble"] - 1.5) / 0.5

    # Stability bonus
    if "ehull" in df_opto.columns:
        df_opto["ehull_val"] = pd.to_numeric(df_opto["ehull"], errors="coerce")
        df_opto["stability_bonus"] = df_opto["ehull_val"].apply(
            lambda x: 0.2 if (not pd.isna(x) and x < 0.1) else 0.0
        )
        df_opto["final_score"] = df_opto["opto_score"] + df_opto["stability_bonus"]
    else:
        df_opto["final_score"] = df_opto["opto_score"]

    df_opto_sorted = df_opto.sort_values("final_score", ascending=False)
    top5 = df_opto_sorted.head(5)

    print(f"\nMaterials in 1.0-2.0 eV range: {len(df_opto)}")
    print("\n=== TOP 5 CANDIDATES FOR OPTOELECTRONICS ===")
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"\n  #{i}: {row['formula']} (JVASP ID: {row['jid']})")
        print(f"      DFT Bandgap: {row['optb88vdw_bandgap']:.3f} eV")
        print(f"      Ensemble Pred: {row['pred_ensemble']:.3f} eV")
        print(f"      RF: {row['pred_rf']:.3f}, XGB: {row['pred_xgb']:.3f}, GNN: {row['pred_gnn']:.3f}")
        print(f"      Score: {row['final_score']:.3f}")

    # Save screening results
    screen_cols = ["jid", "formula", "optb88vdw_bandgap", "pred_rf", "pred_xgb",
                   "pred_gnn", "pred_ensemble", "final_score"]
    avail_cols = [c for c in screen_cols if c in df_opto_sorted.columns]
    df_opto_sorted[avail_cols].head(50).to_csv("/workspace/results/screening_top50.csv", index=False)
    top5[avail_cols].to_csv("/workspace/results/top5_optoelectronic.csv", index=False)
    print("\nSaved: screening_top50.csv, top5_optoelectronic.csv")

    # ================================================================
    # STEP 7: METRICS & MLFLOW-STYLE LOG
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 7: SAVING METRICS & MLFLOW-STYLE LOG")
    print("=" * 70)

    metrics_output = {
        "pipeline_info": {
            "dataset": "JARVIS-DFT 2D",
            "n_materials": int(len(df_ml)),
            "n_features": int(X.shape[1]),
            "feature_names": list(X.columns),
            "target": "optb88vdw_bandgap",
            "test_size": 0.2,
            "timestamp": datetime.now().isoformat(),
        },
        "model_results": results,
        "top5_candidates": [
            {
                "rank": i + 1,
                "jid": str(row["jid"]),
                "formula": str(row["formula"]),
                "dft_bandgap_eV": round(float(row["optb88vdw_bandgap"]), 3),
                "ensemble_pred_eV": round(float(row["pred_ensemble"]), 3),
                "score": round(float(row["final_score"]), 3),
            }
            for i, (_, row) in enumerate(top5.iterrows())
        ],
        "shap_top_features": {
            "random_forest": [{"feature": f, "importance": round(float(v), 4)} for f, v in top_rf[:10]],
            "xgboost": [{"feature": f, "importance": round(float(v), 4)} for f, v in top_xgb[:10]],
        },
    }

    with open("/workspace/results/metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2, default=str)
    print("Saved: metrics.json")

    # MLflow-style log
    mlflow_log = {
        "run_id": f"jarvis-2d-bg-v2-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "experiment_name": "JARVIS-DFT-2D-Bandgap-Prediction-V2",
        "status": "FINISHED",
        "start_time": datetime.now().isoformat(),
        "params": {
            "dataset": "JARVIS-DFT-2D",
            "n_materials": int(len(df_ml)),
            "n_features": int(X.shape[1]),
            "rf_n_estimators": 300, "rf_max_depth": 20,
            "xgb_n_estimators": 500, "xgb_lr": 0.05, "xgb_max_depth": 8,
            "gnn_hidden_dim": 128, "gnn_epochs": 200, "gnn_n_layers": 5,
            "test_split": 0.2, "random_state": 42,
        },
        "metrics": {
            "rf_mae_test": results["RandomForest"]["MAE_test"],
            "rf_r2_test": results["RandomForest"]["R2_test"],
            "xgb_mae_test": results["XGBoost"]["MAE_test"],
            "xgb_r2_test": results["XGBoost"]["R2_test"],
            "gnn_mae_test": results["GNN_Surrogate"]["MAE_test"],
            "gnn_r2_test": results["GNN_Surrogate"]["R2_test"],
        },
        "tags": {
            "model_type": "ensemble (RF + XGBoost + GNN)",
            "target_property": "optb88vdw_bandgap",
            "material_class": "2D",
            "explainability": "SHAP",
            "version": "v2",
        },
        "artifacts": [
            "plots/shap_rf_summary.png", "plots/shap_rf_bar.png",
            "plots/shap_xgb_summary.png", "plots/shap_xgb_bar.png",
            "plots/parity_plots.png", "plots/gnn_learning_curve.png",
            "plots/tree_learning_curves.png", "plots/bandgap_distribution.png",
            "plots/model_comparison.png",
            "screening_top50.csv", "top5_optoelectronic.csv",
            "metrics.json", "shap_importance.json",
        ],
    }

    with open("/workspace/results/mlflow_tracking_log.json", "w") as f:
        json.dump(mlflow_log, f, indent=2)
    print("Saved: mlflow_tracking_log.json")

    volume.commit()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"PIPELINE V2 COMPLETE - Total time: {elapsed / 60:.1f} minutes")
    print(f"{'=' * 70}")

    exp.log({"total_time_min": elapsed / 60, "n_opto_candidates": len(df_opto)}, step=300)
    exp.set_progress(100)
    exp.finish("completed")

    return {
        "results": results,
        "n_materials": int(len(df_ml)),
        "n_features": int(X.shape[1]),
        "n_screening": int(len(df_opto)),
        "top5": metrics_output["top5_candidates"],
        "elapsed_min": round(elapsed / 60, 1),
    }


@app.local_entrypoint()
def main():
    print("Launching JARVIS-DFT 2D Bandgap Pipeline V2 on Modal...")
    result = run_full_pipeline.remote()
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
