"""
Generate publication-quality bird's-eye overlay plots of ground truth vs SVM decisions
for both Sydney CBD and O1 scenarios.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from utils.metrics import *
from models.radio_map import RadioMapNet
from models.reduced_pilot import ReducedPilotNet
from models.integrate import IntegrationNet


def plot_sydney_birds_eye(save_path="outputs/svm_birds_eye_sydney.png"):
    dataset = "sydney"
    device = config.DEVICE

    # 1. Load data & models
    UEloc, CSI, LoS = load_dataset(dataset)
    UEloc, CSI = eliminate_block(UEloc, CSI)
    CSI_fading = add_fading(CSI, 0.0)

    x_loc = prepare_loc_features(UEloc, 1.0)
    x_pilot = prepare_pilot_features(CSI_fading, 15.0, 50)
    train_idx, _, test_idx = split_indices(len(UEloc))

    model_rm = load_pretrained_model(RadioMapNet, "RM", dataset, device=device)
    model_rp = load_pretrained_model(ReducedPilotNet, "RP", dataset, eta=50, device=device)
    model_int = load_pretrained_model(IntegrationNet, "INT", dataset, eta=50, device=device)

    x_test_int = prepare_integration_features(model_rm, model_rp, x_loc[test_idx], x_pilot[test_idx], device)

    model_rm.eval(), model_int.eval()
    with torch.no_grad():
        t_loc_test = torch.as_tensor(x_loc[test_idx], dtype=torch.float32).to(device)
        V_rm_test = normalise_V(model_rm(t_loc_test).cpu().numpy())
        t_int_test = torch.as_tensor(x_test_int, dtype=torch.float32).to(device)
        V_int_test = normalise_V(model_int(t_int_test).cpu().numpy())

    power = (np.linalg.norm(CSI_fading[train_idx]))**2 / np.prod(CSI_fading[train_idx].shape)
    noise = power / (10 ** (15.0 / 10))

    rm_SE_test = cal_SE(CSI_fading[test_idx], V_rm_test, noise)
    int_SE_net_test = cal_SE(CSI_fading[test_idx], V_int_test, noise) * (1 - 0.0476)
    opt_SE = cal_opt_SE(CSI_fading[test_idx], noise)

    test_label, best_SE_test = generate_svm_labels(rm_SE_test, int_SE_net_test)

    # 2. Load trained SVM
    svm_path = config.SAVED_MODELS_DIR / f"SVM_{dataset}.joblib"
    if not svm_path.exists():
        raise FileNotFoundError(f"SVM model not found at {svm_path}. Please run train_svm.py first.")
    svm = joblib.load(svm_path)
    pred_label = svm.predict(x_loc[test_idx])

    svm_SE_test = np.where(pred_label == 0, rm_SE_test, int_SE_net_test)
    accuracy = np.mean(pred_label == test_label) * 100
    rm_pct = np.mean(pred_label == 0) * 100
    int_pct = np.mean(pred_label == 1) * 100
    svm2opt = np.mean(svm_SE_test) / np.mean(opt_SE) * 100

    # 3. Load background image & calibrate
    bg_img_path = PROJECT_ROOT / "city_of_sydney_birds_eye.jpg"
    if not bg_img_path.exists():
        bg_img_path = PROJECT_ROOT / "data" / "sydney" / "city_of_sydney_birds_eye.jpg"
    im = Image.open(bg_img_path)
    w, h = im.size

    px_min, px_max = 631, 1257
    py_min, py_max = 56, 799
    x_min, x_max = -135.754, 135.246
    y_min, y_max = -168.861, 168.139

    sx = (x_max - x_min) / (px_max - px_min)
    sy = (y_max - y_min) / (py_max - py_min)

    X_left = x_min - px_min * sx
    X_right = x_max + (w - px_max) * sx
    Y_bottom = y_min - (h - py_max) * sy
    Y_top = y_max + py_min * sy

    bs_loc = np.load(PROJECT_ROOT / "data" / "sydney" / "BSloc.npy")
    test_ue = UEloc[test_idx]

    # 4. Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=200)

    # Color definitions
    c_rm = "#00e5ff"   # Electric Cyan for Radio Map (dominant)
    c_int = "#ff1744"  # Vivid Red for IntegrationNet (minority corners)
    s_pt = 8

    # Left: Ground Truth
    ax1.imshow(im, extent=[X_left, X_right, Y_bottom, Y_top])
    ax1.scatter(test_ue[test_label == 0, 0], test_ue[test_label == 0, 1],
                c=c_rm, s=s_pt, alpha=0.5, label=f'Radio Map (Optimal: {np.mean(test_label==0)*100:.1f}%)')
    ax1.scatter(test_ue[test_label == 1, 0], test_ue[test_label == 1, 1],
                c=c_int, s=s_pt+4, alpha=0.8, edgecolors='black', linewidths=0.3,
                label=f'IntegrationNet (Optimal: {np.mean(test_label==1)*100:.1f}%)')
    ax1.scatter(bs_loc[0, 0], bs_loc[0, 1], marker='^', c='#ffea00', s=150, edgecolors='black',
                label='Base Station', zorder=10)
    ax1.set_xlim(-155, 155)
    ax1.set_ylim(-185, 185)
    ax1.set_title("Sydney CBD: Ground Truth Optimal Policy\n(Net SE: RM=92.06%, INT=88.10%)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("East-West Distance (m)", fontsize=11)
    ax1.set_ylabel("North-South Distance (m)", fontsize=11)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3, color="white")

    # Right: SVM Prediction
    ax2.imshow(im, extent=[X_left, X_right, Y_bottom, Y_top])
    ax2.scatter(test_ue[pred_label == 0, 0], test_ue[pred_label == 0, 1],
                c=c_rm, s=s_pt, alpha=0.5, label=f'Predicted Radio Map ({rm_pct:.1f}%)')
    ax2.scatter(test_ue[pred_label == 1, 0], test_ue[pred_label == 1, 1],
                c=c_int, s=s_pt+4, alpha=0.8, edgecolors='black', linewidths=0.3,
                label=f'Predicted IntegrationNet ({int_pct:.1f}%)')
    ax2.scatter(bs_loc[0, 0], bs_loc[0, 1], marker='^', c='#ffea00', s=150, edgecolors='black',
                label='Base Station', zorder=10)
    ax2.set_xlim(-155, 155)
    ax2.set_ylim(-185, 185)
    ax2.set_title(f"Sydney CBD: SVM Spatial Discriminator Decisions\n(Accuracy: {accuracy:.1f}%, Net SE: {svm2opt:.2f}%)",
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel("East-West Distance (m)", fontsize=11)
    ax2.set_ylabel("North-South Distance (m)", fontsize=11)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3, color="white")

    plt.suptitle("Sydney CBD Ray-Tracing Scenario: Spatial Pilot Routing Overlay", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Sydney bird's-eye overlay to {out_p}")


def plot_o1_birds_eye(save_path="outputs/svm_birds_eye_o1.png"):
    dataset = "o1"
    device = config.DEVICE

    # 1. Load data & models
    UEloc, CSI, LoS = load_dataset(dataset)
    UEloc, CSI = eliminate_block(UEloc, CSI)
    CSI_fading = add_fading(CSI, 0.0)

    x_loc = prepare_loc_features(UEloc, 1.0)
    x_pilot = prepare_pilot_features(CSI_fading, 15.0, 50)
    train_idx, _, test_idx = split_indices(len(UEloc))

    model_rm = load_pretrained_model(RadioMapNet, "RM", dataset, device=device)
    model_rp = load_pretrained_model(ReducedPilotNet, "RP", dataset, eta=50, device=device)
    model_int = load_pretrained_model(IntegrationNet, "INT", dataset, eta=50, device=device)

    x_test_int = prepare_integration_features(model_rm, model_rp, x_loc[test_idx], x_pilot[test_idx], device)

    model_rm.eval(), model_int.eval()
    with torch.no_grad():
        t_loc_test = torch.as_tensor(x_loc[test_idx], dtype=torch.float32).to(device)
        V_rm_test = normalise_V(model_rm(t_loc_test).cpu().numpy())
        t_int_test = torch.as_tensor(x_test_int, dtype=torch.float32).to(device)
        V_int_test = normalise_V(model_int(t_int_test).cpu().numpy())

    power = (np.linalg.norm(CSI_fading[train_idx]))**2 / np.prod(CSI_fading[train_idx].shape)
    noise = power / (10 ** (15.0 / 10))

    rm_SE_test = cal_SE(CSI_fading[test_idx], V_rm_test, noise)
    int_SE_net_test = cal_SE(CSI_fading[test_idx], V_int_test, noise) * (1 - 0.0476)
    opt_SE = cal_opt_SE(CSI_fading[test_idx], noise)

    test_label, best_SE_test = generate_svm_labels(rm_SE_test, int_SE_net_test)

    # 2. Load trained SVM
    svm_path = config.SAVED_MODELS_DIR / f"SVM_{dataset}.joblib"
    if not svm_path.exists():
        raise FileNotFoundError(f"SVM model not found at {svm_path}. Please run train_svm.py first.")
    svm = joblib.load(svm_path)
    pred_label = svm.predict(x_loc[test_idx])

    svm_SE_test = np.where(pred_label == 0, rm_SE_test, int_SE_net_test)
    accuracy = np.mean(pred_label == test_label) * 100
    rm_pct = np.mean(pred_label == 0) * 100
    int_pct = np.mean(pred_label == 1) * 100
    svm2opt = np.mean(svm_SE_test) / np.mean(opt_SE) * 100

    # 3. Load background image & coordinate mapping
    bg_img_path = PROJECT_ROOT / "outputs" / "o1_birds_eye.jpg"
    if not bg_img_path.exists():
        raw_png = PROJECT_ROOT.parent / "reduced-pilots" / "dataset.png"
        Image.MAX_IMAGE_PIXELS = None
        im_raw = Image.open(raw_png)
        im_raw.thumbnail((1600, 800))
        im_raw.save(bg_img_path, quality=85)
    im = Image.open(bg_img_path)

    test_ue = UEloc[test_idx]
    px = 811.0 - (test_ue[:, 1] - 489.504) * 7.57
    py = 505.0 - (test_ue[:, 0] - 240.0) * 10.375

    bs_px = 811.0
    bs_py = 505.0 - (235.504 - 240.0) * 10.375

    # 4. Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), dpi=200)

    c_rm = "#00e5ff"   # Cyan for Radio Map
    c_int = "#ff1744"  # Red for IntegrationNet
    s_pt = 18

    # Top: Ground Truth
    ax1.imshow(im)
    ax1.scatter(px[test_label == 0], py[test_label == 0],
                c=c_rm, s=s_pt+6, alpha=0.9, edgecolors='black', linewidths=0.5,
                label=f'Radio Map (Optimal: {np.mean(test_label==0)*100:.1f}%)')
    ax1.scatter(px[test_label == 1], py[test_label == 1],
                c=c_int, s=s_pt, alpha=0.7,
                label=f'IntegrationNet (Optimal: {np.mean(test_label==1)*100:.1f}%)')
    ax1.scatter(bs_px, bs_py, marker='^', c='#ffea00', s=160, edgecolors='black',
                label='Base Station 3', zorder=10)
    ax1.set_title(f"O1 Scenario: Ground Truth Optimal Policy (Net SE: RM=64.24%, INT=90.77%)",
                  fontsize=13, fontweight='bold')
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax1.axis('off')

    # Bottom: SVM Prediction
    ax2.imshow(im)
    ax2.scatter(px[pred_label == 0], py[pred_label == 0],
                c=c_rm, s=s_pt+6, alpha=0.9, edgecolors='black', linewidths=0.5,
                label=f'Predicted Radio Map ({rm_pct:.1f}%)')
    ax2.scatter(px[pred_label == 1], py[pred_label == 1],
                c=c_int, s=s_pt, alpha=0.7,
                label=f'Predicted IntegrationNet ({int_pct:.1f}%)')
    ax2.scatter(bs_px, bs_py, marker='^', c='#ffea00', s=160, edgecolors='black',
                label='Base Station 3', zorder=10)
    ax2.set_title(f"O1 Scenario: SVM Spatial Discriminator Decisions (Accuracy: {accuracy:.1f}%, Net SE: {svm2opt:.2f}%)",
                  fontsize=13, fontweight='bold')
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax2.axis('off')

    plt.suptitle("DeepMIMO O1 Blockage Scenario: Spatial Pilot Routing Overlay", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved O1 bird's-eye overlay to {out_p}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="both", choices=["sydney", "o1", "both"])
    args = parser.parse_args()

    if args.dataset in ["sydney", "both"]:
        plot_sydney_birds_eye()
    if args.dataset in ["o1", "both"]:
        plot_o1_birds_eye()
