"""
config.py

Purpose: 
Centralizes all system parameters (carrier frequency fc, bandwidth B, antenna counts Nt, Nr, subcarriers Nc), file paths, simulation defaults, and training hyperparameters.

Steps for Implementation: 
- Define physical and array constants (fc = 3.5 GHz, B = 10 MHz, Nt = 32, Nr = 1, Nc = 12).
- Set default channel conditions (SNR = 15 dB, fading ratio K = 0.1, GPS error loc_std = 1.0 m, eta = 50%).
- Define project directory paths (RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR, OUTPUTS_DIR).
- Define sweep parameter arrays for SNR, GPS error, and fading ratio benchmarks.
- Configure training hyperparameters (batch size, learning rate, epochs, device).
"""

from pathlib import Path                                                        
import torch                                                                    
                                                                                
BASE_DIR           = Path(__file__).resolve().parent
DATA_DIR           = BASE_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "sydney"            # default sim
SAVED_MODELS_DIR   = BASE_DIR / "saved_models"
OUTPUTS_DIR        = BASE_DIR / "outputs"

DEFAULT_DATASET    = "sydney"

def get_dataset_dir(dataset_name=None):
    """
    Resolves the directory containing dataset arrays for 'sydney' or any extra simulation (e.g. 'o1').
    """
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET
    name = str(dataset_name).strip().lower()

    if name in ("sydney", "processed"):
        return DATA_DIR / "sydney"

    custom_dir = DATA_DIR / name
    if custom_dir.exists():
        return custom_dir

    raw_dir = RAW_DATA_DIR / name
    if raw_dir.exists():
        return raw_dir

    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found. Checked: {custom_dir} and {PROCESSED_DATA_DIR}."
    )

# Physical & MIMO Array Constants                                               
FC = 3.5e9   # 3.5 GHz carrier frequency                               
B  = 10.0e6  # 10 MHz bandwidth                                        
C  = 3.0e8   # Speed of light                                          
LAMBDA_VAL      = C / FC
ANTENNA_SPACING = LAMBDA_VAL / 2.0
N_USERS         = 27805
NT              = 32                # Transmit antennas at Base Station                       
NR              = 1                 # Single antenna user equipment                           
NC              = 12                # 12 OFDM subcarriers                                     
NS              = 14                # 14 OFDM symbols per slot / resource block                                     
                                                                                
# Simulation Defaults                                                           
DEFAULT_SNR          = 15.0    # dB                                                 
DEFAULT_FADING_RATIO = 0.1     # Rayleigh scattering K                              
DEFAULT_LOC_STD      = 1.0     # GPS standard deviation (meters)                    
DEFAULT_ETA          = 50      # Reduced pilot density (50%)                        
PILOT_OVERHEAD_RATIO = 0.0476  # 8 pilot REs / 168 total REs per RB             
                                                                                
# Benchmark Sweeps                                                              
SWEEP_SNR          = [0, 5, 10, 15, 20, 25, 30]
SWEEP_LOC_STD      = [0, 1, 2, 3, 4, 5]
SWEEP_FADING_RATIO = [0.0, 0.05, 0.1, 0.15, 0.2]
SWEEP_ETA          = [25, 50, 75, 100]

# Training Hyperparameters                                                      
TEST_SIZE     = 0.2
VAL_SIZE      = 0.125   # 1/8 (10% of 80% train+val size)
RANDOM_SEED   = 1
BATCH_SIZE    = 128
LEARNING_RATE = 1e-3
EPOCHS_RM     = 1000
EPOCHS_RP     = 100
EPOCHS_INT    = 200
DEVICE        = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps. 
is_available() else "cpu")
# search for best parameters
# Coordinates are normalized in [0, 1] (variance ~0.05), so gamma='scale' is ~10.
# Gamma values should be scaled appropriately for normalized inputs.
PARAM_GRID = {'C': [0.1, 1, 10, 50, 100, 500],
              'gamma': ['scale', 0.1, 0.5, 1, 5, 10, 20, 50],
              'kernel': ['rbf']}
PARAM_GRID_MIN = {'C': [1, 10, 50, 100],
                  'gamma': ['scale', 1.0, 5.0, 10.0, 20.0],
                  'kernel': ['rbf']}