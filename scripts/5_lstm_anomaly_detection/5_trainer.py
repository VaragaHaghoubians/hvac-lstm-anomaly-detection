"""
TRAINER - Train the LSTM Autoencoder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Purpose:
    1. Setup training callbacks (EarlyStopping, LR Scheduler, ModelCheckpoint)
    2. Train model on clean training data
    3. Validate on validation set
    4. Save best model
    5. Return training history

CRITICAL:
    - For autoencoder: X_train = y_train (input = output!)
    - Use only clean training sequences (filtered bad segments)
    - Monitor validation loss for early stopping
"""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pathlib import Path
from datetime import datetime
import json
import shutil
import numpy as np


def train_model(model, processed_data, config):
    """
    Train the LSTM autoencoder with best practices
    
    TRAINING PRINCIPLE:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Autoencoder learns to COPY the input:
    • Input: Sequence of sensor readings
    • Output: SAME sequence (reconstruction)
    • Loss: How different is reconstruction from input?
    
    The model learns:
    1. Normal patterns → Easy to reconstruct → Low loss
    2. Anomalous patterns → Hard to reconstruct → High loss
    
    Args:
        model: Compiled Keras model
        processed_data: Output from preprocess_data()
        config: ConfigParser object with TRAINING section
        
    Returns:
        history: Keras History object (contains loss curves)
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING LSTM AUTOENCODER")
    print(f"{'='*70}")
    
    # Extract data
    X_train = processed_data['X_train']
    X_val = processed_data['X_val']
    clean_mask = processed_data['clean_sequence_mask']
    val_clean_mask = processed_data.get('val_clean_sequence_mask', None)
    
    # Apply clean mask to training data
    X_train_clean = X_train[clean_mask]
    if val_clean_mask is not None and len(val_clean_mask) == len(X_val):
        X_val_clean = X_val[val_clean_mask]
    else:
        X_val_clean = X_val
    if len(X_val_clean) == 0:
        print(f"   ⚠️  WARNING: validation clean mask removed ALL {len(X_val)} validation sequences.")
        print(f"   Falling back to unfiltered validation data.")
        X_val_clean = X_val
    
    print(f"   Training samples: {len(X_train_clean):,} (filtered from {len(X_train):,})")
    print(f"   Validation samples: {len(X_val_clean):,} (filtered from {len(X_val):,})")
    
    # Get training parameters from config (lstm_autoencoder section)
    epochs = config.getint('lstm_autoencoder', 'epochs', fallback=250)
    batch_size = config.getint('lstm_autoencoder', 'batch_size', fallback=32)
    patience = config.getint('lstm_autoencoder', 'early_stopping_patience', fallback=30)
    lr_reduce_factor = config.getfloat('lstm_autoencoder', 'lr_reduce_factor', fallback=0.5)
    lr_reduce_patience = config.getint('lstm_autoencoder', 'lr_reduce_patience', fallback=5)
    lr_min = config.getfloat('lstm_autoencoder', 'lr_min', fallback=0.00001)
    
    # Get model save path — include season identifier to prevent cross-season checkpoint reuse
    base_folder  = config.get('paths', 'plots_folder', fallback='./plots')
    building_id  = config.get('global', 'building_id', fallback='C1')
    ahu_unit     = config.get('global', 'ahu_unit',     fallback='UTA1')
    season       = config.get('global', 'season',       fallback='Summer')
    year         = config.get('global', 'year',         fallback='2025')
    model_name   = f'best_autoencoder_{building_id}_{ahu_unit}_{season}{year}.keras'
    model_path   = Path(__file__).parent.parent.parent / base_folder.replace('./','') / 'lstm_autoencoder' / 'models' / model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale checkpoint before training so ModelCheckpoint always gets a
    # clean write path.  OneDrive or a previous Python process can hold a file
    # lock on the existing .keras file, causing PermissionError during fit().
    if model_path.exists():
        try:
            model_path.unlink()
            print(f"   ℹ️  Removed existing checkpoint: {model_path.name}")
        except PermissionError:
            print(f"   ⚠️  Cannot delete {model_path.name} — it may be open in another process.")
            print(f"       Close any program that has it open, then re-run.")
            raise

    print(f"\n   Training parameters:")
    print(f"      • Epochs: {epochs}")
    print(f"      • Batch size: {batch_size}")
    print(f"      • Early stopping patience: {patience}")
    print(f"      • LR reduction: factor={lr_reduce_factor}, patience={lr_reduce_patience}, min={lr_min}")
    print(f"      • Model save path: {model_path}")
    
    # ═══════════════════════════════════════════════════════════════
    # SETUP CALLBACKS
    # ═══════════════════════════════════════════════════════════════
    
    # 1. Early Stopping: Stop if validation loss doesn't improve
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    
    # 2. Model Checkpoint: Save best model based on val_loss
    checkpoint = ModelCheckpoint(
        filepath=str(model_path),
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        mode='min'
    )
    
    # 3. Learning Rate Scheduler: Reduce LR when learning plateaus
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=lr_reduce_factor,
        patience=lr_reduce_patience,
        min_lr=lr_min,
        verbose=1,
        mode='min'
    )
    
    callbacks = [early_stop, checkpoint, lr_scheduler]
    
    print(f"\n   Callbacks configured:")
    print(f"      ✓ EarlyStopping (patience={patience})")
    print(f"      ✓ ModelCheckpoint (save best)")
    print(f"      ✓ ReduceLROnPlateau (factor={lr_reduce_factor})")
    
    # ═══════════════════════════════════════════════════════════════
    # TRAIN MODEL
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"🎯 TRAINING STARTED...")
    print(f"{'='*70}\n")

    # Guard: if clean mask removed everything, fall back to unfiltered training set
    if len(X_train_clean) == 0:
        print(f"   ⚠️  WARNING: clean mask removed ALL {len(X_train)} training sequences.")
        print(f"   Falling back to unfiltered training data.")
        print(f"   → Set 'enable_clean_training_filters = false' in config.ini to suppress this.")
        X_train_clean = X_train

    # 🔑 CRITICAL: For autoencoder, X = y (input = output)
    # We train the model to reconstruct its own input!

    # ✅ SHUFFLE = TRUE (correct for LSTM-AE)
    # We shuffle the ORDER of windows, NOT the data inside each window
    # This helps SGD training without destroying temporal structure
    # Each sequence maintains its internal time order
    history = model.fit(
        X_train_clean, X_train_clean,  # Input = Output!
        validation_data=(X_val_clean, X_val_clean),  # Validation also: X = y
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,  # Show progress bar
        shuffle=True  # ✅ Shuffle SEQUENCES (not timesteps within sequences)
    )
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*70}")

    # Print final metrics
    final_train_loss = history.history['loss'][-1]
    final_val_loss   = history.history['val_loss'][-1]
    best_val_loss    = min(history.history['val_loss'])
    best_epoch       = int(np.argmin(history.history['val_loss'])) + 1
    total_epochs     = len(history.history['loss'])

    print(f"\n   TRAINING RESULTS:")
    print(f"      - Final training loss:    {final_train_loss:.6f}")
    print(f"      - Final validation loss:  {final_val_loss:.6f}")
    print(f"      - Best validation loss:   {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"      - Total epochs trained:   {total_epochs}")
    print(f"      - Model saved to:         {model_path}")

    # -----------------------------------------------------------------
    # MODEL VERSIONING
    # Saves a timestamped snapshot alongside the rolling 'best' model.
    # Use restore_version.py to list versions and roll back.
    # -----------------------------------------------------------------
    version_tag  = config.get('lstm_autoencoder', 'version_tag', fallback='run').strip()
    timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_name = f"{timestamp}_{version_tag}"
    versions_dir = model_path.parent / 'versions'
    version_dir  = versions_dir / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    # Copy best checkpoint into version folder
    version_model_path = version_dir / 'model.keras'
    shutil.copy2(model_path, version_model_path)

    # Collect key hyperparameters for the metadata snapshot
    def _cfg(key, fallback=''):
        return config.get('lstm_autoencoder', key, fallback=fallback)

    metadata = {
        'version':       version_name,
        'tag':           version_tag,
        'timestamp':     timestamp,
        'description':   '',           # filled in by user or restore_version.py
        'config': {
            'sequence_length':    config.getint('lstm_autoencoder', 'sequence_length'),
            'encoder_units':      config.getint('lstm_autoencoder', 'encoder_units',   fallback=64),
            'bottleneck_units':   config.getint('lstm_autoencoder', 'bottleneck_units', fallback=16),
            'decoder_units':      config.getint('lstm_autoencoder', 'decoder_units',   fallback=64),
            'dropout':            config.getfloat('lstm_autoencoder', 'dropout',       fallback=0.1),
            'learning_rate':      config.getfloat('lstm_autoencoder', 'learning_rate', fallback=0.001),
            'batch_size':         config.getint('lstm_autoencoder',   'batch_size',    fallback=32),
            'epochs_max':         config.getint('lstm_autoencoder',   'epochs',        fallback=300),
            'loss_function':      _cfg('loss_function', 'mae'),
            'scaler_type':        _cfg('scaler_type', 'robust'),
            'train_split':        config.getfloat('lstm_autoencoder', 'train_split',   fallback=0.70),
            'val_split':          config.getfloat('lstm_autoencoder', 'val_split',     fallback=0.15),
            'dataset_end_date':   _cfg('dataset_end_date', ''),
            'apply_ewma_smoothing': config.getboolean('lstm_autoencoder', 'apply_ewma_smoothing', fallback=False),
            'ewma_span':          config.getint('lstm_autoencoder', 'ewma_span',       fallback=3),
            'ewma_sensor_columns': _cfg('ewma_sensor_columns', ''),
            'threshold_mode':     _cfg('threshold_mode', 'ema'),
        },
        'training': {
            'best_val_loss':   round(best_val_loss,   6),
            'final_train_loss': round(final_train_loss, 6),
            'best_epoch':      best_epoch,
            'total_epochs':    total_epochs,
        },
        'eval': {}    # populated later by save_version_metrics()
    }

    meta_path = version_dir / 'metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(metadata, fh, indent=2)

    # Update the shared registry file
    registry_path = versions_dir / 'registry.json'
    registry = []
    if registry_path.exists():
        try:
            with open(registry_path, 'r', encoding='utf-8') as fh:
                registry = json.load(fh)
        except Exception:
            registry = []
    registry.append({'version': version_name, 'tag': version_tag, 'timestamp': timestamp})
    with open(registry_path, 'w', encoding='utf-8') as fh:
        json.dump(registry, fh, indent=2)

    print(f"      - Version saved:          {version_dir.name}")
    print(f"      - Metadata:               {meta_path.name}")
    print(f"{'='*70}\n")

    # Attach version_dir to history so 1_main.py can finalize eval metrics
    history.version_dir = version_dir

    return history


def save_version_metrics(version_dir, metrics):
    """
    Finalizes the version metadata.json with evaluation metrics.
    Called from 1_main.py after calculate_metrics() completes.

    Args:
        version_dir : pathlib.Path returned via history.version_dir
        metrics     : dict returned by calculate_metrics()
    """
    if version_dir is None:
        return
    meta_path = Path(version_dir) / 'metadata.json'
    if not meta_path.exists():
        return
    try:
        with open(meta_path, 'r', encoding='utf-8') as fh:
            metadata = json.load(fh)
        metadata['eval'] = {
            'r2_overall':          round(metrics.get('r2_overall', 0),    4),
            'rmse_overall_degC':   round(metrics.get('rmse_overall', 0),  4),
            'mse_overall_degC2':   round(metrics.get('mse_overall', 0),   4),
            'mae_overall_degC':    round(metrics.get('mae_overall', 0),   4),
            'anomaly_rate_pct':    round(metrics.get('anomaly_rate', 0),  2),
            'rmse_per_feature':    [round(v, 4) for v in metrics.get('rmse_per_feature', [])],
            'mae_per_feature':     [round(v, 4) for v in metrics.get('mae_per_feature',  [])],
        }
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump(metadata, fh, indent=2)
        print(f"   Version metrics saved -> {meta_path.parent.name}/metadata.json")
    except Exception as exc:
        print(f"   [versioning] Could not update metadata: {exc}")


# =============================================================================
# TEST SCRIPT - Run this file directly to test training
# =============================================================================
if __name__ == "__main__":
    import configparser
    from pathlib import Path
    import importlib
    
    data_loader = importlib.import_module('2_data_loader')
    preprocessor = importlib.import_module('3_preprocessor')
    model_module = importlib.import_module('4_model')
    
    load_and_prepare_data = data_loader.load_and_prepare_data
    preprocess_data = preprocessor.preprocess_data
    build_autoencoder = model_module.build_autoencoder
    
    print("\n🧪 TESTING TRAINER.PY (Quick sanity check)\n")
    
    # Load configuration
    config = configparser.ConfigParser()
    # Use main config.ini from project root (2 levels up)
    config_path = Path(__file__).parent.parent.parent / 'config.ini'
    config.read(config_path, encoding='utf-8-sig')
    
    try:
        # Load and preprocess data
        print("Loading data...")
        data_dict = load_and_prepare_data(config)
        
        print("Preprocessing...")
        processed = preprocess_data(data_dict, config)
        
        # Build model
        print("Building model...")
        model = build_autoencoder(processed['input_shape'], config)
        
        # Train (just 5 epochs for testing)
        print("Training for 5 epochs (test only)...")
        history = train_model(model, processed, config)
        
        print(f"\n✅ TRAINER TEST SUCCESSFUL!")
        print(f"   Loss history: {history.history['loss']}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
