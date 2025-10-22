#!/usr/bin/env python3
"""
H13 Independent Dataset Validation - H08 S100 Model Transfer Learning
Agent: claude_code

Purpose: Test H08 S100→Stiffness model on external datasets WITHOUT retraining
Success Criterion: R² ≥ 0.60 on external data (training R²=0.81)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

#============================================================================
# CONFIGURATION
#============================================================================

WORKSPACE = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_13_independent_dataset_validation/claude_code")

# H08 model locations (from Iteration 03)
H08_CLAUDE_MODEL = "/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/claude_code/models/s100_stiffness_model_claude_code.pth"
H08_CODEX_MODEL = "/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/codex/models/s100_stiffness_model_codex.pth"

# S100 family proteins (20 members)
S100_GENES = [
    'S100A1', 'S100A2', 'S100A3', 'S100A4', 'S100A5', 'S100A6',
    'S100A7', 'S100A8', 'S100A9', 'S100A10', 'S100A11', 'S100A12',
    'S100A13', 'S100A14', 'S100A15', 'S100A16', 'S100B',
    'S100P', 'S100Z', 'S100G'
]

# Stiffness proxy proteins
STIFFNESS_GENES = {
    "LOX": 0.5,      # Lysyl oxidase (crosslinking)
    "TGM2": 0.3,     # Transglutaminase 2 (crosslinking)
    "COL1A1": 0.1,   # Collagen I alpha 1
    "COL3A1": -0.1   # Collagen III alpha 1 (inverse for ratio)
}

#============================================================================
# S100 MODEL ARCHITECTURE (MUST MATCH H08)
#============================================================================

class S100StiffnessModel(nn.Module):
    """
    Neural network for S100 → Stiffness prediction
    Architecture must match the model saved in H08
    """
    def __init__(self, input_dim=20, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze()

#============================================================================
# STIFFNESS PROXY CALCULATION
#============================================================================

def calculate_stiffness_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Calculate ECM stiffness proxy from protein abundances
    Proxy = 0.5*LOX + 0.3*TGM2 + 0.2*(COL1A1/COL3A1)
    """
    print("\nCalculating stiffness proxy...")

    stiffness = pd.Series(0.0, index=df.index)

    for gene, weight in STIFFNESS_GENES.items():
        if gene in df.columns:
            if weight > 0:
                stiffness += weight * df[gene]
            else:
                # Handle ratio (COL1A1/COL3A1)
                if gene == "COL3A1" and "COL1A1" in df.columns:
                    ratio = df["COL1A1"] / (df["COL3A1"] + 1e-6)  # Avoid division by zero
                    stiffness += 0.2 * ratio
        else:
            print(f"  ⚠️  {gene} not found in data")

    print(f"  Stiffness proxy calculated for {len(stiffness)} samples")
    print(f"  Mean stiffness: {stiffness.mean():.3f} ± {stiffness.std():.3f}")

    return stiffness

#============================================================================
# TRANSFER LEARNING VALIDATION
#============================================================================

def validate_s100_model_on_external(
    model_path: Path,
    external_data: pd.DataFrame,
    agent_name: str
) -> dict:
    """
    Test S100 model on external dataset WITHOUT retraining

    Args:
        model_path: Path to saved PyTorch model
        external_data: DataFrame with S100 protein z-scores
        agent_name: "claude_code" or "codex"

    Returns:
        dict with validation metrics
    """
    print("\n" + "="*80)
    print(f"H08 S100 MODEL TRANSFER LEARNING - {agent_name.upper()}")
    print("="*80)

    # Step 1: Load pre-trained model
    print("\n1. Loading pre-trained model...")
    try:
        model = S100StiffnessModel(input_dim=20)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"   ✅ Model loaded successfully from {model_path.name}")
    except FileNotFoundError:
        print(f"   ❌ Model not found: {model_path}")
        print("   ⚠️  Returning placeholder results")
        return {
            "agent": agent_name,
            "status": "MODEL_NOT_FOUND",
            "r2_external": np.nan,
            "mae_external": np.nan,
            "rmse_external": np.nan,
            "validation": "FAILED"
        }

    # Step 2: Extract S100 features from external data
    print("\n2. Extracting S100 features...")
    available_s100 = [g for g in S100_GENES if g in external_data.columns]
    print(f"   Found {len(available_s100)}/{len(S100_GENES)} S100 proteins")
    print(f"   Available: {', '.join(available_s100[:5])}...")

    if len(available_s100) < 10:
        print(f"   ❌ Insufficient S100 proteins (need ≥10, have {len(available_s100)})")
        return {
            "agent": agent_name,
            "status": "INSUFFICIENT_S100_PROTEINS",
            "r2_external": np.nan,
            "validation": "FAILED"
        }

    # Prepare input matrix (pad missing S100 proteins with zeros)
    X_external = np.zeros((len(external_data), 20))
    for i, gene in enumerate(S100_GENES):
        if gene in external_data.columns:
            X_external[:, i] = external_data[gene].fillna(0).values

    X_tensor = torch.FloatTensor(X_external)

    # Step 3: Predict stiffness
    print("\n3. Predicting stiffness with S100 model...")
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()
    print(f"   Predictions generated for {len(y_pred)} samples")
    print(f"   Mean predicted stiffness: {y_pred.mean():.3f} ± {y_pred.std():.3f}")

    # Step 4: Calculate actual stiffness proxy
    print("\n4. Calculating actual stiffness proxy...")
    y_actual = calculate_stiffness_proxy(external_data)

    # Step 5: Evaluate performance
    print("\n5. Evaluating model performance...")
    r2 = r2_score(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

    print(f"\n📊 VALIDATION RESULTS ({agent_name.upper()}):")
    print(f"   R² (external): {r2:.3f}")
    print(f"   MAE: {mae:.3f}")
    print(f"   RMSE: {rmse:.3f}")

    # Determine validation status
    if r2 >= 0.60:
        validation = "✅ STRONG (R² ≥ 0.60)"
    elif r2 >= 0.40:
        validation = "⚠️  MODERATE (0.40 ≤ R² < 0.60)"
    else:
        validation = "❌ POOR (R² < 0.40 - OVERFITTING DETECTED)"

    print(f"   Validation: {validation}")

    # Comparison to training performance
    training_r2_claude = 0.81
    training_r2_codex = 0.75
    training_r2 = training_r2_claude if agent_name == "claude_code" else training_r2_codex
    drop = training_r2 - r2

    print(f"\n   Training R²: {training_r2:.3f}")
    print(f"   Performance drop: {drop:.3f}")
    print(f"   Allowable drop: ≤ 0.15")
    print(f"   Generalization: {'✅ GOOD' if drop <= 0.15 else '⚠️  MODERATE' if drop <= 0.30 else '❌ POOR'}")

    # Save results
    results = {
        "agent": agent_name,
        "status": "SUCCESS",
        "r2_external": r2,
        "mae_external": mae,
        "rmse_external": rmse,
        "r2_training": training_r2,
        "performance_drop": drop,
        "validation": validation,
        "n_samples": len(y_actual),
        "n_s100_available": len(available_s100),
        "s100_genes_available": available_s100
    }

    return results

#============================================================================
# VISUALIZATION
#============================================================================

def plot_validation_results(
    y_actual: np.ndarray,
    y_pred_claude: np.ndarray,
    y_pred_codex: np.ndarray,
    output_path: Path
):
    """
    Create scatter plots comparing predicted vs actual stiffness
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Claude Code model
    axes[0].scatter(y_actual, y_pred_claude, alpha=0.6, s=50)
    axes[0].plot([y_actual.min(), y_actual.max()],
                 [y_actual.min(), y_actual.max()],
                 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual Stiffness Proxy')
    axes[0].set_ylabel('Predicted Stiffness (S100 Model)')
    axes[0].set_title('H08 Transfer Learning: Claude Code Model')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Codex model
    axes[1].scatter(y_actual, y_pred_codex, alpha=0.6, s=50, color='orange')
    axes[1].plot([y_actual.min(), y_actual.max()],
                 [y_actual.min(), y_actual.max()],
                 'r--', lw=2, label='Perfect prediction')
    axes[1].set_xlabel('Actual Stiffness Proxy')
    axes[1].set_ylabel('Predicted Stiffness (S100 Model)')
    axes[1].set_title('H08 Transfer Learning: Codex Model')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved scatter plot to: {output_path}")

#============================================================================
# MAIN EXECUTION
#============================================================================

def main():
    """Main execution for H08 transfer learning validation"""

    print("\n" + "="*80)
    print("H13: H08 S100 MODEL TRANSFER LEARNING VALIDATION")
    print("="*80)

    # Load external data
    external_data_file = WORKSPACE / "external_datasets" / "PXD011967" / "PXD011967_processed_zscore.csv"

    if not external_data_file.exists():
        print("\n⚠️  EXTERNAL DATA NOT YET AVAILABLE")
        print(f"   Expected: {external_data_file}")
        print("\n📋 PLACEHOLDER EXECUTION")
        print("   This script is ready to run once external data is processed")
        print("   See DATA_ACQUISITION_PLAN.md for data download instructions")
        return

    print(f"\nLoading external data: {external_data_file.name}")
    external_df = pd.read_csv(external_data_file)

    # Pivot to wide format (genes as columns)
    external_wide = external_df.pivot_table(
        index='Sample_ID',
        columns='Gene_Symbol',
        values='Z_score'
    )

    # Validate both agent models
    results = []

    # Claude Code model
    result_claude = validate_s100_model_on_external(
        Path(H08_CLAUDE_MODEL),
        external_wide,
        "claude_code"
    )
    results.append(result_claude)

    # Codex model
    result_codex = validate_s100_model_on_external(
        Path(H08_CODEX_MODEL),
        external_wide,
        "codex"
    )
    results.append(result_codex)

    # Save results
    df_results = pd.DataFrame(results)
    output_file = WORKSPACE / "h08_external_validation_claude_code.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Saved validation results to: {output_file}")

    # Summary
    print("\n" + "="*80)
    print("H08 TRANSFER LEARNING SUMMARY")
    print("="*80)
    print(f"Claude Code: R²={result_claude.get('r2_external', np.nan):.3f}")
    print(f"Codex:       R²={result_codex.get('r2_external', np.nan):.3f}")
    print(f"Target:      R² ≥ 0.60")

if __name__ == "__main__":
    main()
