"""
VISUALIZER - Create plots for LSTM Autoencoder results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Purpose:
    1. Plot training history (loss curves)
    2. Plot reconstruction error distribution
    3. Plot anomalies timeline
    4. Plot reconstruction comparison examples
    5. Save all plots to outputs folder
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import importlib
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import inverse_transform_sequences from numbered module
preprocessor = importlib.import_module('3_preprocessor')
inverse_transform_sequences = preprocessor.inverse_transform_sequences


# Setup plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.0)


def get_date_range_string(config):
    """
    Builds a consistent subtitle string for all plots without duplicating
    config reading logic across plot functions.

    Args:
        config (ConfigParser): Project configuration.

    Returns:
        str: Human-readable date range, e.g. "Periodo: 2025-07-09 - 2025-10-14".
    """
    season = config.get('global', 'season')
    year = config.get('global', 'year')
    
    # Get date range from config seasonal_presets
    season_key = f"{season.lower()}_{year}"
    try:
        start_date = config.get('seasonal_presets', f"{season.lower()}_start_{year}")
        end_date = config.get('seasonal_presets', f"{season.lower()}_end_{year}")
        return f"Periodo: {start_date} - {end_date}"
    except:
        return f"Periodo: {season} {year}"


def get_plots_dir(config):
    """
    Centralises plot output path construction so all plot functions save
    to the same season-specific directory without duplicating path logic.

    Args:
        config (ConfigParser): Project configuration.

    Returns:
        Path: Season-specific output directory (created if it does not exist).
    """
    base_folder = config.get('paths', 'plots_folder', fallback='./plots')
    building_id = config.get('global', 'building_id')
    ahu_unit = config.get('global', 'ahu_unit')
    season = config.get('global', 'season')
    year = config.get('global', 'year')
    
    folder_name = f"{building_id}_{ahu_unit}_{season}{year}"
    plots_dir = Path(__file__).parent.parent.parent / base_folder.replace('./','') / 'lstm_autoencoder' / folder_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def plot_training_history(history, config):
    """
    Verifies that training converged correctly before evaluating detection
    results — oscillating or flat loss curves indicate a training problem
    that would invalidate threshold calibration.

    Args:
        history: Keras History object returned by model.fit().
        config (ConfigParser): Project configuration.
    """
    
    plots_dir = get_plots_dir(config)
    date_range = get_date_range_string(config)
    
    # Guard: skip if history is empty (e.g. standalone test without training)
    if not history.history.get('loss'):
        print("   Skipping training history plot (no history data available).")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot loss
    epochs = range(1, len(history.history['loss']) + 1)
    ax.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    # Highlight best epoch
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = min(history.history['val_loss'])
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7, 
               label=f'Best Epoch ({best_epoch})')
    ax.scatter([best_epoch], [best_val_loss], color='green', s=100, zorder=5)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title(f'Storico Addestramento LSTM Autoencoder\n{date_range}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = plots_dir / 'training_history.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {save_path}")


def plot_reconstruction_error_distribution(results, config):
    """
    Justifies the anomaly threshold placement by showing the separation
    between normal and anomalous reconstruction errors — used to visually
    validate that the chosen percentile or EMA multiplier is meaningful.

    Args:
        results (dict): Output from detect_anomalies().
        config (ConfigParser): Project configuration.
    """
    
    plots_dir = get_plots_dir(config)
    date_range = get_date_range_string(config)
    
    train_errors = results['train_errors']
    test_errors = results['test_errors']
    threshold = results['threshold']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ─── Plot 1: Training Errors ───
    ax = axes[0]
    ax.hist(train_errors, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold = {threshold:.6f}')
    ax.set_xlabel('Errore di Ricostruzione (MSE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequenza', fontsize=12, fontweight='bold')
    ax.set_title(f'Errori di Ricostruzione - Set Training\n{date_range}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ─── Plot 2: Test Errors with Anomalies ───
    ax = axes[1]
    
    # Normal vs Anomaly
    normal_errors = test_errors[~results['filtered_anomalies']]
    anomaly_errors = test_errors[results['filtered_anomalies']]
    
    ax.hist(normal_errors, bins=50, color='lightgreen', alpha=0.7, 
            label=f'Normal ({len(normal_errors)})', edgecolor='black')
    ax.hist(anomaly_errors, bins=30, color='crimson', alpha=0.7, 
            label=f'Anomalies ({len(anomaly_errors)})', edgecolor='black')
    ax.axvline(threshold, color='orange', linestyle='--', linewidth=2, 
               label=f'Threshold = {threshold:.6f}')
    
    ax.set_xlabel('Errore di Ricostruzione (MSE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequenza', fontsize=12, fontweight='bold')
    ax.set_title(f'Errori di Ricostruzione - Set Test\n{date_range}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = plots_dir / 'reconstruction_error_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {save_path}")


def plot_anomalies_timeline(results, processed_data, data_dict, config):
    """
    Provides temporal context for detected anomalies so threshold and
    filter settings can be visually inspected and defended in the thesis.

    Args:
        results (dict): Output from detect_anomalies().
        processed_data (dict): Preprocessor output.
        data_dict (dict): Data loader output.
        config (ConfigParser): Project configuration.
    """
    
    # Get plot directory
    base_folder = config.get('paths', 'plots_folder', fallback='./plots')
    building_id = config.get('global', 'building_id')
    ahu_unit = config.get('global', 'ahu_unit')
    season = config.get('global', 'season')
    year = config.get('global', 'year')
    
    folder_name = f"{building_id}_{ahu_unit}_{season}{year}"
    plots_dir = Path(__file__).parent.parent.parent / base_folder.replace('./','') / 'lstm_autoencoder' / folder_name
    
    test_errors = results['test_errors']
    test_errors_smoothed = results.get('test_errors_smoothed', test_errors)
    threshold = results['threshold']
    anomalies = results['filtered_anomalies']
    severity = results['severity_labels']
    test_indices = processed_data['test_indices']
    
    # test_indices already stores the END timestamp of each sequence (one per error value)
    test_indices_aligned = test_indices
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    
    # ─── Plot 1: EMA-style Anomaly Detection (matches uploaded reference image) ───
    # Blue  line  : actual reconstruction error at every test window
    # Red   line  : EMA of the error  (exponentially weighted moving average)
    # Black dashed: upper bound = EMA_mean + k·EMA_std  (anomaly threshold)
    #               lower bound = EMA_mean - k·EMA_std  (symmetric, visual only)
    # Red circles : windows flagged as anomalies
    ax = axes[0]

    # Read EMA params from config
    ema_span = config.getint('lstm_autoencoder', 'ema_span', fallback=20)
    _season   = config.get('global', 'season', fallback='').lower()
    _seas_key = f'ema_sigma_multiplier_{_season}'
    if config.has_option('lstm_autoencoder', _seas_key):
        k = config.getfloat('lstm_autoencoder', _seas_key)
    else:
        k = config.getfloat('lstm_autoencoder', 'ema_sigma_multiplier', fallback=3.0)

    # Compute EMA mean for the red trend line (visual only).
    # For the threshold line in EMA mode, prefer the *actual* per-step
    # detection threshold saved by the evaluator (causal + warm-start anchor),
    # otherwise recomputing here can be off by one step at the start.
    s        = pd.Series(test_errors_smoothed)
    ema_mean = s.ewm(span=ema_span, adjust=False).mean()

    threshold_mode = config.get('lstm_autoencoder', 'threshold_mode', fallback='percentile')
    upper = None
    if threshold_mode == 'ema':
        thresh_series = results.get('threshold_info', {}).get('threshold_series', None)
        if thresh_series is not None and len(thresh_series) == len(test_errors):
            upper = np.asarray(thresh_series)

    if upper is None:
        # Fallback (static modes, or if threshold_series wasn't provided)
        ema_std  = s.ewm(span=ema_span, adjust=False).std().fillna(0)
        upper    = (ema_mean + k * ema_std).values
    # NOTE: no lower bound — reconstruction error (MAE) is always ≥ 0.
    # Only HIGH error signals an anomaly; a very low error just means normal operation.
    # A lower bound would be physically meaningless and visually misleading.

    # Break line plots at large timestamp jumps (holiday gaps, stratified blocks).
    # Use config sampling interval (fallback 30 min) so Summer/Winter behave the same.
    sampling_minutes = config.getint('lstm_autoencoder', 'sampling_interval_minutes', fallback=30)
    expected_step = pd.Timedelta(minutes=sampling_minutes) * 1.5
    ts_series = pd.Series(test_indices_aligned)
    gap_mask = (ts_series.diff() > expected_step).fillna(False).to_numpy()

    test_errors_masked = np.asarray(test_errors_smoothed, dtype=float).copy()
    ema_mean_masked = np.asarray(ema_mean.values, dtype=float).copy()
    upper_masked = np.asarray(upper, dtype=float).copy()
    test_errors_masked[gap_mask] = np.nan
    ema_mean_masked[gap_mask] = np.nan
    upper_masked[gap_mask] = np.nan

    # Plot actual values (blue)
    ax.plot(test_indices_aligned, test_errors_masked, color='steelblue', linewidth=1.2,
            alpha=0.8, label='Reconstruction Error', zorder=2)

    # Plot EMA trend (red)
    ax.plot(test_indices_aligned, ema_mean_masked, color='red', linewidth=2.0,
            label=f'EMA (span={ema_span},  α={2/(ema_span+1):.3f})', zorder=3)

    # Plot upper bound only (black dashed) — the actual anomaly threshold
    ax.plot(test_indices_aligned, upper_masked, 'k--', linewidth=1.5,
            label=f'Upper bound: EMA + {k}·σ  (anomaly threshold)', zorder=3)

    # Static 95th-percentile threshold overlay (gray dotted) — comparison baseline.
    # Shown regardless of threshold_mode so the analyst can always see what a fixed
    # quantile threshold would have done (validates EMA choice, addresses boiling-frog
    # concern: if EMA >> static, the model baseline has drifted up).
    _static_pct = config.getfloat('lstm_autoencoder', 'anomaly_threshold_percentile', fallback=95.0)
    _train_errors_for_static = results.get('train_errors', np.array([]))
    if len(_train_errors_for_static) > 0:
        _static_thr = float(np.percentile(_train_errors_for_static, _static_pct))
        ax.axhline(_static_thr, color='#888888', linestyle=':', linewidth=1.4,
                   label=f'Static {_static_pct:.0f}th-pct threshold = {_static_thr:.4f}', zorder=2)

    # Anomaly markers (red circles) — points where error > upper bound
    # Use filtered_anomalies (persistence-filtered) rather than raw crossing
    anomaly_mask  = severity == 'Anomaly'    # binary mode (enable_severity = false)
    critical_mask = severity == 'Critical'
    moderate_mask = severity == 'Moderate'
    minor_mask    = severity == 'Minor'

    if anomaly_mask.any():
        ax.scatter(test_indices_aligned[anomaly_mask], test_errors_smoothed[anomaly_mask],
                   color='red', s=120, label='Anomaly', zorder=5, marker='o',
                   edgecolors='darkred', linewidths=0.8)
    else:
        if critical_mask.any():
            ax.scatter(test_indices_aligned[critical_mask], test_errors_smoothed[critical_mask],
                       color='red', s=120, label='Critical', zorder=5, marker='o',
                       edgecolors='darkred', linewidths=0.8)
        if moderate_mask.any():
            ax.scatter(test_indices_aligned[moderate_mask], test_errors_smoothed[moderate_mask],
                       color='orange', s=90, label='Moderate', zorder=4, marker='o',
                       edgecolors='darkorange', linewidths=0.8)
        if minor_mask.any():
            ax.scatter(test_indices_aligned[minor_mask], test_errors_smoothed[minor_mask],
                       color='yellow', s=60, label='Minor', zorder=3, marker='o',
                       edgecolors='goldenrod', linewidths=0.8)

    ax.set_ylabel('Reconstruction Error (MAE)', fontsize=12, fontweight='bold')
    ax.set_title('Anomaly Detection — Exponentially Weighted Moving Average',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ─── Plot 2: Anomaly Flags (Binary) ───
    ax = axes[1]
    anomaly_binary = anomalies.astype(float)  # float so NaN sentinels work

    # Break fill_between polygons at large timestamp jumps (holiday gaps, stratified blocks)
    # so matplotlib doesn't draw a diagonal "triangle" across discontinuities.
    # (reuse expected_step and gap_mask from Plot 1)
    anomaly_binary_masked = anomaly_binary.copy()
    anomaly_binary_masked[gap_mask] = np.nan

    ax.fill_between(test_indices_aligned, 0, anomaly_binary_masked,
                    color='red', alpha=0.3, label='Anomaly Periods')
    ax.set_ylabel('Anomaly Flag', fontsize=12, fontweight='bold')
    ax.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = plots_dir / 'anomalies_timeline.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {save_path}")


def plot_reconstruction_comparison(X_test, X_reconstructed, scaler, 
                                   feature_names, sample_idx, config):
    """
    Compare original vs reconstructed for specific sequences
    
    Args:
        X_test: Test sequences (scaled)
        X_reconstructed: Reconstructed sequences (scaled)
        scaler: Scaler for inverse transform
        feature_names: List of feature names
        sample_idx: List of sample indices to plot
        config: ConfigParser object
    """
    
    # Get plot directory
    base_folder = config.get('paths', 'plots_folder', fallback='./plots')
    building_id = config.get('global', 'building_id')
    ahu_unit = config.get('global', 'ahu_unit')
    season = config.get('global', 'season')
    year = config.get('global', 'year')
    
    folder_name = f"{building_id}_{ahu_unit}_{season}{year}"
    plots_dir = Path(__file__).parent.parent.parent / base_folder.replace('./','') / 'lstm_autoencoder' / folder_name
    
    # Inverse transform to original scale
    n_features = len(feature_names)
    X_test_orig = inverse_transform_sequences(X_test, scaler, n_features)
    X_recon_orig = inverse_transform_sequences(X_reconstructed, scaler, n_features)
    
    for idx in sample_idx:
        fig, axes = plt.subplots(n_features, 1, figsize=(14, 3*n_features), sharex=True)
        
        if n_features == 1:
            axes = [axes]
        
        for i, feature_name in enumerate(feature_names):
            ax = axes[i]
            
            timesteps = range(len(X_test_orig[idx, :, i]))
            ax.plot(timesteps, X_test_orig[idx, :, i], 'b-', 
                    label='Original', linewidth=2, marker='o', markersize=4)
            ax.plot(timesteps, X_recon_orig[idx, :, i], 'r--', 
                    label='Reconstructed', linewidth=2, marker='x', markersize=4)
            
            ax.set_ylabel(feature_name, fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Timestep', fontsize=12, fontweight='bold')
        fig.suptitle(f'Reconstruction Comparison - Sample {idx}', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = plots_dir / f'reconstruction_sample_{idx}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved: {save_path}")


def plot_feature_error_metrics(X_test, X_reconstructed, scaler, feature_names, config):
    """
    Plot RMSE, MSE, MAE per feature in physical units (°C) with a summary table.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    plots_dir = get_plots_dir(config)
    date_range = get_date_range_string(config)

    # Inverse-transform to original °C scale
    n_features = len(feature_names)
    X_test_orig = inverse_transform_sequences(X_test, scaler, n_features)
    X_recon_orig = inverse_transform_sequences(X_reconstructed, scaler, n_features)

    # Compute per-feature metrics in physical units
    rmse_list, mse_list, mae_list = [], [], []
    for i in range(n_features):
        y_true = X_test_orig[:, :, i].flatten()
        y_pred = X_recon_orig[:, :, i].flatten()
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_true, y_pred)
        rmse_list.append(rmse)
        mse_list.append(mse)
        mae_list.append(mae)

    short_names = [fn.replace('Temperatura ', 'T. ') for fn in feature_names]
    x = np.arange(n_features)
    colors = ['steelblue', 'coral', 'seagreen']

    # ── Top section: 3 bar charts ────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(2, 3, height_ratios=[3, 1.2], hspace=0.55, wspace=0.35)

    for ax_idx, (data, ylabel, title) in enumerate([
        (rmse_list, 'RMSE (°C)',  'RMSE'),
        (mse_list,  'MSE', 'MSE'),
        (mae_list,  'MAE (°C)',   'MAE'),
    ]):
        ax = fig.add_subplot(gs[0, ax_idx])
        bars = ax.bar(x, data, 0.6, color=colors[ax_idx], alpha=0.75, edgecolor='black')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} per Feature\n{date_range}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() * 1.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # ── Bottom section: summary table ────────────────────────────────────────
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')

    col_labels = ['Feature', 'RMSE (°C)', 'MSE', 'MAE (°C)']
    table_data = [
        [feature_names[i], f'{rmse_list[i]:.4f}', f'{mse_list[i]:.4f}', f'{mae_list[i]:.4f}']
        for i in range(n_features)
    ]
    table_data.append([
        'Overall (mean)',
        f'{np.mean(rmse_list):.4f}',
        f'{np.mean(mse_list):.4f}',
        f'{np.mean(mae_list):.4f}',
    ])

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#1f4788')
        tbl[0, j].get_text().set_color('white')
        tbl[0, j].get_text().set_fontweight('bold')
    overall_row = len(table_data)
    for j in range(len(col_labels)):
        tbl[overall_row, j].set_facecolor('#dce6f1')
        tbl[overall_row, j].get_text().set_fontweight('bold')

    ax_table.set_title('Metriche di Ricostruzione (valori fisici)', fontsize=12,
                       fontweight='bold', pad=8)

    plt.savefig(plots_dir / 'feature_error_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved feature error metrics plot (physical units + table)")


def plot_time_series_comparison(X_test, X_reconstructed, scaler, feature_names, test_indices, config):
    """Plot full time series comparison for all features"""
    plots_dir = get_plots_dir(config)
    date_range = get_date_range_string(config)
    
    n_features = len(feature_names)
    X_test_orig = inverse_transform_sequences(X_test, scaler, n_features)
    X_recon_orig = inverse_transform_sequences(X_reconstructed, scaler, n_features)
    
    for feat_idx in range(n_features):
        fig, ax = plt.subplots(figsize=(16, 6))
        
        real_values = X_test_orig[:, -1, feat_idx]
        recon_values = X_recon_orig[:, -1, feat_idx]
        # test_indices already stores end-timestamp of each sequence (one per sample)
        aligned_indices = test_indices[:len(real_values)]
        
        ax.plot(aligned_indices, real_values, 'b-', label='Serie Temporale Reale', linewidth=1.5, alpha=0.7)
        ax.plot(aligned_indices, recon_values, 'r--', label='Serie Temporale Ricostruita', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Indice Temporale', fontsize=12, fontweight='bold')
        ax.set_ylabel(feature_names[feat_idx], fontsize=12, fontweight='bold')
        ax.set_title(f'Confronto Serie Temporale: {feature_names[feat_idx]}\\n{date_range}', 
                     fontsize=14, fontweight='bold', color='#1f4788')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_name = feature_names[feat_idx].replace(' ', '_').replace('/', '_')[:50]
        plt.savefig(plots_dir / f'time_series_comparison_{feat_idx+1:02d}_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved time series comparison for {feature_names[feat_idx]}")


def plot_results(history, results, processed_data, data_dict, config):
    """
    Orchestrates all individual plot functions in one call so the caller
    does not need to know which plots exist or in what order to run them.

    Args:
        history: Keras History object.
        results (dict): Anomaly detection results.
        processed_data (dict): Preprocessor output.
        data_dict (dict): Data loader output.
        config (ConfigParser): Project configuration.
    """
    
    print(f"\n{'='*70}")
    print(f"📊 GENERATING PLOTS")
    print(f"{'='*70}\n")
    
    # 1. Training history
    print("   Creating training history plot...")
    plot_training_history(history, config)
    
    # 2. Reconstruction error distribution
    print("   Creating reconstruction error distribution...")
    plot_reconstruction_error_distribution(results, config)
    
    # 3. Anomalies timeline
    print("   Creating anomalies timeline...")
    plot_anomalies_timeline(results, processed_data, data_dict, config)
    
    # 4. Reconstruction comparison examples
    print("   Creating reconstruction comparison examples...")
    
    # Plot 3 normal samples and 3 anomaly samples (if available)
    normal_indices = np.where(~results['filtered_anomalies'])[0][:3]
    anomaly_indices = np.where(results['filtered_anomalies'])[0][:3]
    
    sample_indices = list(normal_indices) + list(anomaly_indices)
    
    if len(sample_indices) > 0:
        plot_reconstruction_comparison(
            processed_data['X_test'],
            results['X_test_reconstructed'],
            processed_data['scaler'],
            data_dict['features'],
            sample_indices,
            config
        )
    
    # 5. Per-feature error metrics (RMSE, MSE, MAE in physical units °C)
    print("   Creating per-feature error metrics...")
    plot_feature_error_metrics(
        processed_data['X_test'],
        results['X_test_reconstructed'],
        processed_data['scaler'],
        data_dict['features'],
        config
    )
    
    # 6. Time series comparison (real vs reconstructed)
    print("   Creating time series comparisons for all features...")
    plot_time_series_comparison(
        processed_data['X_test'],
        results['X_test_reconstructed'],
        processed_data['scaler'],
        data_dict['features'],
        processed_data['test_indices'],
        config
    )
    
    print(f"\n{'='*70}")
    print(f"✅ ALL PLOTS CREATED!")
    # Get folder info for output path
    base_folder = config.get('paths', 'plots_folder', fallback='./plots')
    building_id = config.get('global', 'building_id')
    ahu_unit = config.get('global', 'ahu_unit')
    season = config.get('global', 'season')
    year = config.get('global', 'year')
    folder_name = f"{building_id}_{ahu_unit}_{season}{year}"
    output_path = Path(__file__).parent.parent.parent / base_folder.replace('./','') / 'lstm_autoencoder' / folder_name
    print(f"   Location: {output_path}")
    print(f"{'='*70}\n")


# ═════════════════════════════════════════════════════════════════════════════
# TEST SCRIPT - loads saved model, runs evaluator, generates all plots
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import configparser
    import importlib
    import sys
    from pathlib import Path
    from tensorflow.keras.models import load_model

    print("\n*** TESTING VISUALIZER.PY (standalone) ***\n")

    # ── 1. Config ─────────────────────────────────────────────────────────
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent.parent / 'config.ini'
    config.read(config_path, encoding='utf-8-sig')
    print(f"Config loaded: {config_path}")

    # ── 2. Load & preprocess data ──────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    data_loader  = importlib.import_module('2_data_loader')
    preprocessor = importlib.import_module('3_preprocessor')
    evaluator    = importlib.import_module('6_evaluator')

    data_dict      = data_loader.load_and_prepare_data(config)
    processed_data = preprocessor.preprocess_data(data_dict, config)
    print(f"Data preprocessed")
    print(f"   X_train: {processed_data['X_train'].shape}")
    print(f"   X_val:   {processed_data['X_val'].shape}")
    print(f"   X_test:  {processed_data['X_test'].shape}")

    # ── 3. Load saved model ────────────────────────────────────────────────
    base_folder  = config.get('paths', 'plots_folder', fallback='./plots')
    building_id  = config.get('global', 'building_id', fallback='C1')
    ahu_unit     = config.get('global', 'ahu_unit',     fallback='UTA1')
    season       = config.get('global', 'season',       fallback='Summer')
    year         = config.get('global', 'year',         fallback='2025')
    model_name   = f'best_autoencoder_{building_id}_{ahu_unit}_{season}{year}.keras'
    model_path   = (
        Path(__file__).parent.parent.parent
        / base_folder.replace('./', '')
        / 'lstm_autoencoder' / 'models' / model_name
    )
    if not model_path.exists():
        print(f"\nERROR - Model not found: {model_path}")
        print("   Run 1_main.py first to train and save the model.")
        sys.exit(1)

    model = load_model(str(model_path))
    print(f"\nModel loaded: {model_path}")
    model.summary()

    # ── 4. Run anomaly detection to get results ────────────────────────────
    print("\nRunning anomaly detection...")
    results = evaluator.detect_anomalies(model, processed_data, config)
    metrics = evaluator.calculate_metrics(results, processed_data)

    print(f"\nThreshold:      {results['threshold']:.6f}")
    print(f"Test anomalies: {results['filtered_anomalies'].sum()} ({metrics['anomaly_rate']:.2f}%)")
    print(f"Overall R2:     {metrics['r2_overall']:.4f}")

    # ── 5. Build minimal history object for training history plot ──────────
    # When running standalone (no training), we skip the training history plot
    # and generate all other plots that only need results + processed_data
    class _FakeHistory:
        """Minimal history stub so plot_training_history gracefully skips"""
        history = {'loss': [], 'val_loss': []}

    fake_history = _FakeHistory()

    # ── 6. Generate all plots ──────────────────────────────────────────────
    generate_plots = config.getboolean('lstm_autoencoder', 'generate_plots', fallback=True)
    if generate_plots:
        plot_results(fake_history, results, processed_data, data_dict, config)
    else:
        print("\ngenerate_plots = false in config — skipping plots.")

    print("\n*** VISUALIZER TEST COMPLETE ***\n")
