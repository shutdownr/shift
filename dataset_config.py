class Config:
    # Dataset identifiers
    dataset_names = [
        "cif",
        "nn5",
        "tourism",
        "weather",
        "m3_m",
        "m3_q",
        "m3_y",
        "m3_o",
        "m4_h",
        "m4_w",
        "m4_y",
        "transactions",
    ]
    # Dataset identifiers for by-sample experiment
    by_sample_dataset_names = ["nn5", "tourism", "m4_h", "m4_w", "m3_o"]

    # Training set size
    train_sizes = {
        "cif": 0.6,
        "nn5": 0.6,
        "tourism": 0.6,
        "weather": 0.6,
        "m3_m": 0.6,
        "m3_q": 0.6,
        "m3_y": 0.6,
        "m3_o": 0.6,
        "m4_h": 0.6,
        "m4_w": 0.6,
        "m4_y": 0.6,
        "transactions": 0.6,
    }
    # Validation set size
    val_sizes = {
        "cif": 0.2,
        "nn5": 0.2,
        "tourism": 0.2,
        "weather": 0.2,
        "m3_m": 0.2,
        "m3_q": 0.2,
        "m3_y": 0.2,
        "m3_o": 0.2,
        "m4_h": 0.2,
        "m4_w": 0.2,
        "m4_y": 0.2,
        "transactions": 0.2,
    }
    # Stride length (sliding window step size) for generating train/val/test instances
    stride_lengths = {
        "cif": 1,
        "nn5": 5,
        "tourism": 5,
        "weather": 5,
        "m3_m": 1,
        "m3_q": 1,
        "m3_y": 1,
        "m3_o": 1,
        "m4_h": 5,
        "m4_w": 5,
        "m4_y": 5,
        "transactions": 5,
    }
    # Default length of the input window (x) for train/val/test instances
    backhorizons = {
        "cif": 15,
        "nn5": 70,
        "tourism": 30,
        "weather": 24,
        "m3_m": 18,
        "m3_q": 8,
        "m3_y": 8,
        "m3_o": 8,
        "m4_h": 60,
        "m4_w": 20,
        "m4_y": 8,
        "transactions": 14,
    }
    # Default length of the horizon (y) for train/val/test instances
    horizons = {
        "cif": 12,
        "nn5": 56,
        "tourism": 24,
        "weather": 12,
        "m3_m": 18,
        "m3_q": 8,
        "m3_y": 8,
        "m3_o": 8,
        "m4_h": 48,
        "m4_w": 13,
        "m4_y": 6,
        "transactions": 7,
    }
    # Timesnet requires different parameters to be set depending on the dataset
    # These parameters are taken as provided in the authors' codebase and adapted for the other datasets
    # https://github.com/thuml/Time-Series-Library/blob/main/scripts/short_term_forecast/TimesNet_M4.sh
    timesnet_d_model = {
        "cif": 16,
        "nn5": 16,
        "tourism": 32,
        "weather": 32,
        "m3_m": 32,
        "m3_q": 64,
        "m3_y": 64,
        "m3_o": 32,
        "m4_h": 32,
        "m4_w": 32,
        "m4_y": 64,
        "transactions": 32,
    }
    timesnet_d_ff = {
        "cif": 32,
        "nn5": 16,
        "tourism": 32,
        "weather": 32,
        "m3_m": 32,
        "m3_q": 64,
        "m3_y": 64,
        "m3_o": 32,
        "m4_h": 32,
        "m4_w": 32,
        "m4_y": 64,
        "transactions": 32,
    }
