# SVM to select RP and RM
# TODO add plot label

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import config
from utils.metrics import *
from models.radio_map import RadioMapNet
from models.reduced_pilot import ReducedPilotNet
from models.integrate import IntegrationNet

def main():
    parser = get_base_parser(description="Train and evaluate SVM combined RP/RM model")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter grid search for SVM.")
    parser.add_argument("--plot", action="store_true", help="Plot spatial decision map.")
    args = parser.parse_args()

    dataset = args.dataset
    loc_std = args.loc_std
    fading_ratio = args.fading_ratio
    SNR = args.snr
    eta = args.eta
    device = config.DEVICE

    # Load Arrays
    UEloc, CSI, LoS = load_dataset(dataset)
    UEloc, CSI = eliminate_block(UEloc, CSI)
    CSI_fading = add_fading(CSI, fading_ratio)
    
    x_loc = prepare_loc_features(UEloc, loc_std)
    x_pilot = prepare_pilot_features(CSI_fading, SNR, eta)
    
    train_idx, _, test_idx = split_indices(len(UEloc))
    CSI_train, CSI_test = CSI_fading[train_idx], CSI_fading[test_idx]
    LoS_test = LoS[test_idx]
    
    power = (np.linalg.norm(CSI_fading[train_idx]))**2 / np.prod(CSI_fading[train_idx].shape)
    noise = power / (10 ** (SNR / 10))

    # Load pretrained models
    model_rm = load_pretrained_model(RadioMapNet, "RM", dataset, device=device)
    model_rp = load_pretrained_model(ReducedPilotNet, "RP", dataset, eta=eta, device=device)
    model_int = load_pretrained_model(IntegrationNet, "INT", dataset, eta=eta, device=device)

    x_train_int = prepare_integration_features(model_rm, model_rp, x_loc[train_idx], x_pilot[train_idx], device)
    x_test_int = prepare_integration_features(model_rm, model_rp, x_loc[test_idx], x_pilot[test_idx], device)

    model_rm.eval(), model_int.eval()
    with torch.no_grad():
        # Radio Map
        t_loc_train = torch.as_tensor(x_loc[train_idx], dtype=torch.float32).to(device)
        t_loc_test = torch.as_tensor(x_loc[test_idx], dtype=torch.float32).to(device)

        V_rm_train = normalise_V(model_rm(t_loc_train).cpu().numpy())
        V_rm_test = normalise_V(model_rm(t_loc_test).cpu().numpy())
        
        # Integration
        t_int_train = torch.as_tensor(x_train_int, dtype=torch.float32).to(device)
        t_int_test = torch.as_tensor(x_test_int, dtype=torch.float32).to(device)
 
        V_int_train = normalise_V(model_int(t_int_train).cpu().numpy())
        V_int_test = normalise_V(model_int(t_int_test).cpu().numpy())
    
    rm_SE_train = cal_SE(CSI_train, V_rm_train, noise)
    rm_SE_test = cal_SE(CSI_test, V_rm_test, noise)
    int_SE_net_train = cal_SE(CSI_train, V_int_train, noise) * (1-0.0476)
    int_SE_net_test = cal_SE(CSI_test, V_int_test, noise) * (1-0.0476)
    
    opt_SE = cal_opt_SE(CSI_test, noise)

    train_label, _ = generate_svm_labels(rm_SE_train, int_SE_net_train)
    test_label, best_SE_test = generate_svm_labels(rm_SE_test, int_SE_net_test)

    # Check if SVM model is already computed
    config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    svm_path = config.SAVED_MODELS_DIR / f"SVM_{dataset}.joblib"

    if svm_path.exists() and not args.force_train and not args.tune:
        rel_path = svm_path.relative_to(PROJECT_ROOT) if svm_path.is_relative_to(PROJECT_ROOT) else svm_path
        print(f"Found pre-trained SVM model at: {rel_path} (use --force-train to retrain)...")
        svm = joblib.load(svm_path)
    else:
        if args.tune:
            from sklearn.model_selection import GridSearchCV
            print("Running GridSearchCV for SVM hyperparameters...")
            grid = GridSearchCV(SVC(), config.PARAM_GRID_MIN, cv=5, n_jobs=-1, verbose=1)
            grid.fit(x_loc[train_idx], train_label)
            svm = grid.best_estimator_
            print(f"Best parameters: {grid.best_params_}")
        else:
            print("Training SVM with fast default parameters (C=10, gamma='scale')...")
            svm = SVC(C=10, gamma='scale', kernel='rbf')
            svm.fit(x_loc[train_idx], train_label)

        joblib.dump(svm, svm_path)
        rel_path = svm_path.relative_to(PROJECT_ROOT) if svm_path.is_relative_to(PROJECT_ROOT) else svm_path
        print(f"Training complete: saved SVM model to {rel_path}")

    # Evaluate on test set
    pred_label = svm.predict(x_loc[test_idx])
    svm_SE_test = np.where(pred_label == 0, rm_SE_test, int_SE_net_test)

    # Benchmark ratios (% of optimal)
    svm2opt = np.mean(svm_SE_test) / np.mean(opt_SE) * 100
    best2opt = np.mean(best_SE_test) / np.mean(opt_SE) * 100
    rm2opt = np.mean(rm_SE_test) / np.mean(opt_SE) * 100
    int2opt = np.mean(int_SE_net_test) / np.mean(opt_SE) * 100

    # LoS and NLoS breakdown
    svm_LoS_SE, svm_NLoS_SE = compute(svm_SE_test, LoS_test)
    opt_LoS_SE, opt_NLoS_SE = compute(opt_SE, LoS_test)
    svm2opt_LoS = (svm_LoS_SE / opt_LoS_SE) * 100 if opt_LoS_SE > 0 else 0.0
    svm2opt_NLoS = (svm_NLoS_SE / opt_NLoS_SE) * 100 if opt_NLoS_SE > 0 else 0.0

    # Classifier statistics
    accuracy = np.mean(pred_label == test_label) * 100
    rm_pct = np.mean(pred_label == 0) * 100
    int_pct = np.mean(pred_label == 1) * 100

    print("=" * 50)
    print(f"SVM Hybrid Selector Results ({dataset.upper()} @ SNR={SNR}dB)")
    print("=" * 50)
    print(f"Classification Accuracy:      {accuracy:.2f}%")
    print(f"Decisions:                    {rm_pct:.1f}% Radio Map | {int_pct:.1f}% IntegrationNet")
    print(f"Radio Map (Net):              {rm2opt:.3f} %")
    print(f"IntegrationNet (Net):         {int2opt:.3f} %")
    print(f"SVM Hybrid (Net):             {svm2opt:.3f} %")
    print(f"Theoretical Upper (Best):     {best2opt:.3f} %") # Either RP or RM alone
    print("-" * 50)
    print(f"SVM Net LoS:                  {svm2opt_LoS:.3f} %")
    print(f"SVM Net NLoS:                 {svm2opt_NLoS:.3f} %")
    print(f"Mean SVM SE:                  {np.mean(svm_SE_test):.3f} bps/Hz")
    print(f"Optimal SE:                   {np.mean(opt_SE):.3f} bps/Hz")

    # Optional plot
    if args.plot:
        import matplotlib.pyplot as plt
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = config.OUTPUTS_DIR / f"svm_decisions_{dataset}.png"

        plt.figure(figsize=(12, 5))

        # Ground truth
        plt.subplot(1, 2, 1)
        plt.scatter(x_loc[test_idx][test_label == 0, 0], x_loc[test_idx][test_label == 0, 1], c='#1f77b4', s=10, alpha=0.6, label='Radio Map (Best)')
        plt.scatter(x_loc[test_idx][test_label == 1, 0], x_loc[test_idx][test_label == 1, 1], c='#ff7f0e', s=10, alpha=0.6, label='IntegrationNet (Best)')
        plt.title("Ground Truth Optimal Decision")
        plt.xlabel("Normalized X")
        plt.ylabel("Normalized Y")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SVM Prediction
        plt.subplot(1, 2, 2)
        plt.scatter(x_loc[test_idx][pred_label == 0, 0], x_loc[test_idx][pred_label == 0, 1], c='#1f77b4', s=10, alpha=0.6, label='Predicted Radio Map')
        plt.scatter(x_loc[test_idx][pred_label == 1, 0], x_loc[test_idx][pred_label == 1, 1], c='#ff7f0e', s=10, alpha=0.6, label='Predicted IntegrationNet')
        plt.title(f"SVM Predicted Decision (Acc: {accuracy:.1f}%)")
        plt.xlabel("Normalized X")
        plt.ylabel("Normalized Y")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        rel_plot = plot_path.relative_to(PROJECT_ROOT) if plot_path.is_relative_to(PROJECT_ROOT) else plot_path
        print(f"Saved decision map plot to: {rel_plot}")

        # Also generate bird's-eye overlay if available
        try:
            from scripts.plot_birds_eye import plot_sydney_birds_eye, plot_o1_birds_eye
            if dataset == "sydney":
                plot_sydney_birds_eye()
            elif dataset == "o1":
                plot_o1_birds_eye()
        except Exception as e:
            print(f"Note: Bird's-eye overlay generation skipped: {e}")

if __name__ == "__main__":
    main()