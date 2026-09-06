"""
MAIN - LSTM Autoencoder for HVAC Anomaly Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 UNSUPERVISED LSTM AUTOENCODER - "The Copycat Artist"
   Learns normal HVAC operation patterns WITHOUT target variables.
   Detects anomalies through reconstruction error.

✅ MODULAR ARCHITECTURE:
   • data_loader.py: Load and prepare CSV data
   • preprocessor.py: Scale, split, and create sequences
   • model.py: Build LSTM Autoencoder architecture
   • trainer.py: Train with callbacks and early stopping
   • evaluator.py: Detect anomalies and calculate metrics
   • visualizer.py: Create comprehensive plots
   • config.ini: All hyperparameters in one place

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📖 HOW IT WORKS (Step-by-Step)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANALOGY: The Copycat Artist
   An artist looks at a painting, memorizes it, closes their eyes,
   and tries to redraw it from memory.
   
   • Normal painting (Healthy HVAC) → Artist redraws it perfectly → Low Error
   • Weird painting (Potentially atypical HVAC) → Artist can't redraw it → High Error
   
   The difference between original and reconstruction = ANOMALY SCORE

WORKFLOW:
   1. LOAD DATA: Read feature-engineered CSV
   2. PREPROCESS: Scale features, split train/val/test, create sequences
   3. BUILD MODEL: LSTM Encoder-Decoder architecture
   4. TRAIN: Learn to reconstruct normal sequences (80% data)
   5. EVALUATE: Detect anomalies on test set (20% data)
   6. VISUALIZE: Create plots for analysis
   7. REPORT: Save results to CSV and generate summary

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔑 KEY PRINCIPLE: "Train on the Norm, Detect on the Storm"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Training Data: Learn what NORMAL looks like
   • Test Data: Detect what is WEIRD
   • You DON'T need labeled fault data, but anomaly flags still require operational/log validation.
"""

import configparser
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import importlib
import sys
import os
import random
import numpy as np

# Ensure Windows consoles can print Unicode (box-drawing, symbols) without crashing.
# This is safe on non-Windows too; it just keeps the existing encoding.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── Global reproducibility seed ──────────────────────────────────────────────
# Read seed from config before importing TensorFlow so the env-var is set first.
def _apply_global_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

_cfg_seed = configparser.ConfigParser()
_cfg_seed.read(
    Path(__file__).resolve().parents[2] / 'config.ini',
    encoding='utf-8-sig'
)
_seed_val = _cfg_seed.get('lstm_autoencoder', 'random_seed', fallback='42').strip()
if _seed_val:
    _apply_global_seed(int(_seed_val))
    print(f'[Reproducibility] Global random seed set to {_seed_val}')
else:
    print('[Reproducibility] No seed configured — results may differ between runs.')
del _cfg_seed, _seed_val
# ─────────────────────────────────────────────────────────────────────────────

# Import our modular components (using importlib for numbered modules)
data_loader = importlib.import_module('2_data_loader')
preprocessor = importlib.import_module('3_preprocessor')
model_module = importlib.import_module('4_model')
trainer = importlib.import_module('5_trainer')
evaluator = importlib.import_module('6_evaluator')
visualizer = importlib.import_module('7_visualizer')
report_generator = importlib.import_module('8_report_generator')

context_plotter = importlib.import_module('9_plot_anomaly_context')

load_and_prepare_data = data_loader.load_and_prepare_data
preprocess_data = preprocessor.preprocess_data
build_autoencoder = model_module.build_autoencoder
print_model_architecture = model_module.print_model_architecture
train_model = trainer.train_model
detect_anomalies = evaluator.detect_anomalies
calculate_metrics = evaluator.calculate_metrics
plot_results = visualizer.plot_results
generate_lstm_report = report_generator.generate_lstm_report
run_context_plots = context_plotter.run_from_dataframes


def save_results_to_csv(results, processed_data, data_dict, config, metrics=None):
    """
    Save anomaly detection results to CSV file
    
    Args:
        results: Output from detect_anomalies()
        processed_data: Preprocessor output
        data_dict: Data loader output
        config: ConfigParser object
    """
    
    print(f"\n{'='*70}")
    print(f"💾 SAVING RESULTS TO PROCESSED_DATA")
    print(f"{'='*70}\n")
    
    # Get processed_data directory from config
    base_folder = config.get('paths', 'processed_folder', fallback='./processed_data')
    building_id = config.get('global', 'building_id')
    ahu_unit = config.get('global', 'ahu_unit')
    season = config.get('global', 'season')
    year = config.get('global', 'year')
    
    folder_name = f"{building_id}_{ahu_unit}_{season}{year}"
    results_dir = Path(__file__).parent.parent.parent / base_folder.replace('./','') / 'lstm_autoencoder' / folder_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # test_indices already stores end-timestamp of each sequence (one per error value)
    test_indices = processed_data['test_indices']
    
    # Create DataFrame with results
    # For EMA mode, threshold_info contains a per-step array ('threshold_series').
    # Save that as the Threshold column so context plots show the actual moving
    # threshold that was used at each timestep — not just the training anchor.
    threshold_series = results['threshold_info'].get('threshold_series', results['threshold'])
    df_results = pd.DataFrame({
        'Timestamp'                    : test_indices[:len(results['test_errors'])],
        'Reconstruction_Error'         : results['test_errors'],
        'Reconstruction_Error_Smoothed': results['test_errors_smoothed'],
        'Threshold'                    : threshold_series,
        'Is_Anomaly_Pre_Persist'       : results['anomalies_pre_persist'],
        'Is_Candidate_Anomaly'         : results.get('candidate_anomalies', results['anomalies']),
        'Is_Anomaly'                   : results['filtered_anomalies'],
        'Severity'                     : results['severity_labels'],
        'Num_Anomalous_Features'       : results['features_anomalous']
    })
    if 'semantic_labels' in results:
        df_results['Semantic_Label'] = results['semantic_labels']
    if 'event_ids' in results:
        df_results['Event_ID'] = results['event_ids']
        event_type_map = {int(r['Event_ID']): r['Event_Type'] for r in results.get('event_rows', [])}
        df_results['Event_Type'] = df_results['Event_ID'].map(event_type_map).fillna('Normal')
    
    # Add per-feature errors
    for i, feature in enumerate(data_dict['features']):
        df_results[f'{feature}_Error'] = results['feature_errors'][:, i]
        df_results[f'{feature}_Threshold'] = results['feature_thresholds'][i]

    # Physics rule checks (Yeom-style hybrid layer)
    print(f"\n   Running physics rule checks...")
    physics_rules = evaluator.apply_physics_rule_checks(
        data_dict['df'], test_indices, config
    )
    df_results['Rule_HeatTransfer'] = physics_rules['Rule_HeatTransfer'].values
    df_results['Rule_Control']      = physics_rules['Rule_Control'].values
    df_results['Rule_Any_Fault']    = (
        physics_rules['Rule_HeatTransfer'] | physics_rules['Rule_Control']
    ).values
    df_results['LSTM_and_Rule']     = (
        df_results['Is_Anomaly'] & df_results['Rule_Any_Fault']
    )
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = results_dir / f'anomaly_results_{timestamp}.csv'
    df_results.to_csv(csv_path, index=False)
    if results.get('event_rows'):
        events_df = pd.DataFrame(results['event_rows'])
        events_csv_path = results_dir / f'anomaly_events_{timestamp}.csv'
        events_df.to_csv(events_csv_path, index=False)
        print(f"   ✓ Event table saved to: {events_csv_path}")

    # Candidate event table (screening layer): contiguous runs of Is_Candidate_Anomaly
    # with simple ranking scores to prioritize operational review.
    candidate_mask = df_results.get('Is_Candidate_Anomaly', pd.Series(False, index=df_results.index)).astype(bool).to_numpy()
    if candidate_mask.any():
        rows = []
        i = 0
        n = len(candidate_mask)
        while i < n:
            if candidate_mask[i]:
                j = i
                while j < n and candidate_mask[j]:
                    j += 1
                block = df_results.iloc[i:j]
                exceed = (block['Reconstruction_Error_Smoothed'] - block['Threshold']).clip(lower=0.0)
                duration_h = (j - i) * 0.5  # 30-min windows
                rows.append({
                    'Candidate_Event_ID'        : len(rows) + 1,
                    'Start'                     : block['Timestamp'].iloc[0],
                    'End'                       : block['Timestamp'].iloc[-1],
                    'Num_Windows'               : int(j - i),
                    'Duration_h'                : round(float(duration_h), 2),
                    'Peak_Error'                : float(block['Reconstruction_Error_Smoothed'].max()),
                    'Peak_Threshold'            : float(block.loc[block['Reconstruction_Error_Smoothed'].idxmax(), 'Threshold']),
                    'Max_Exceedance'            : float(exceed.max()),
                    'Mean_Exceedance'           : float(exceed.mean()),
                    'Mean_Anomalous_Features'   : float(block['Num_Anomalous_Features'].mean()),
                    'Start_Hour'                : int(pd.to_datetime(block['Timestamp'].iloc[0]).hour),
                    'End_Hour'                  : int(pd.to_datetime(block['Timestamp'].iloc[-1]).hour),
                })
                i = j
            else:
                i += 1

        cand_df = pd.DataFrame(rows)
        if len(cand_df) > 0:
            # Composite score: stronger exceedance + longer duration + multi-feature support.
            cand_df['Candidate_Score'] = (
                0.45 * cand_df['Max_Exceedance'] +
                0.30 * cand_df['Mean_Exceedance'] +
                0.15 * cand_df['Duration_h'] +
                0.10 * cand_df['Mean_Anomalous_Features']
            )
            cand_df = cand_df.sort_values(['Candidate_Score', 'Max_Exceedance', 'Duration_h'], ascending=False)
            candidate_csv_path = results_dir / f'candidate_events_{timestamp}.csv'
            cand_df.to_csv(candidate_csv_path, index=False)
            print(f"   ✓ Candidate event table saved to: {candidate_csv_path}")
    
    print(f"   ✓ Results saved to: {csv_path}")
    
    # Save summary statistics
    summary_path = results_dir / f'summary_{timestamp}.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("LSTM AUTOENCODER ANOMALY DETECTION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATA INFO:\n")
        f.write(f"  • Total test samples: {len(results['test_errors'])}\n")
        f.write(f"  • Date range: {test_indices[0]} to {test_indices[-1]}\n")
        f.write(f"  • Features: {', '.join(data_dict['features'])}\n\n")
        
        f.write("ANOMALY DETECTION:\n")
        f.write(f"  • Threshold: {results['threshold']:.6f}\n")
        candidate_count = int(results.get('candidate_anomalies', results['anomalies']).sum())
        candidate_rate = (candidate_count / len(results['test_errors']) * 100) if len(results['test_errors']) else 0.0
        f.write(f"  • Candidate anomalies: {candidate_count}\n")
        f.write(f"  • Candidate anomaly rate: {candidate_rate:.2f}%\n")
        f.write(f"  • Total anomalies: {results['filtered_anomalies'].sum()}\n")
        f.write(f"  • Anomaly rate: {results['filtered_anomalies'].sum()/len(results['test_errors'])*100:.2f}%\n")
        sem_stats = results.get('semantic_filter_stats', {})
        if sem_stats and sem_stats.get('enabled', False):
            f.write("  • Semantic filter (operational vs fault):\n")
            f.write(f"      - Mode: {sem_stats.get('mode', 'unknown')}\n")
            f.write(f"      - Input anomalies: {sem_stats.get('n_input_anomalies', 0)}\n")
            f.write(f"      - Potential faults: {sem_stats.get('n_potential_fault', 0)}\n")
            f.write(f"      - Periodic operational peaks: {sem_stats.get('n_periodic_operational', 0)}\n")
            f.write(f"      - Suppressed by semantic filter: {sem_stats.get('n_suppressed_by_semantic_filter', 0)}\n")
        mode_thr = results.get('mode_threshold_stats', {})
        if mode_thr and mode_thr.get('enabled', False):
            f.write("  • Mode-aware threshold:\n")
            f.write(f"      - Occupied windows: {mode_thr.get('occupied_windows', 0)}\n")
            f.write(f"      - Unoccupied windows: {mode_thr.get('unoccupied_windows', 0)}\n")
            f.write(f"      - Multipliers (occ/unocc): {mode_thr.get('occupied_multiplier', 1.0):.2f}/{mode_thr.get('unoccupied_multiplier', 1.0):.2f}\n")
        ev_stats = results.get('event_stats', {})
        if ev_stats:
            f.write("  • Event grouping:\n")
            f.write(f"      - Events total: {ev_stats.get('n_events', 0)}\n")
            f.write(f"      - Fault events: {ev_stats.get('n_fault_events', 0)}\n")
            f.write(f"      - Operational events: {ev_stats.get('n_operational_events', 0)}\n")
        if results['n_critical'] or results['n_moderate'] or results['n_minor']:
            f.write(f"  • Critical: {results['n_critical']}\n")
            f.write(f"  • Moderate: {results['n_moderate']}\n")
            f.write(f"  • Minor: {results['n_minor']}\n")
        f.write('\n')
        
        f.write("RECONSTRUCTION ERRORS:\n")
        f.write(f"  • Train mean: {results['train_errors'].mean():.6f}\n")
        f.write(f"  • Train std: {results['train_errors'].std():.6f}\n")
        f.write(f"  • Test mean (raw):      {results['test_errors'].mean():.6f}\n")
        f.write(f"  • Test std  (raw):      {results['test_errors'].std():.6f}\n")
        f.write(f"  • Test mean (smoothed): {results['test_errors_smoothed'].mean():.6f}\n")
        f.write(f"  • Test std  (smoothed): {results['test_errors_smoothed'].std():.6f}\n\n")

        if metrics is not None:
            f.write("RECONSTRUCTION QUALITY (physical units — °C):\n")
            f.write(f"  • Overall R²:   {metrics['r2_overall']:.4f}\n")
            f.write(f"  • Overall RMSE: {metrics['rmse_overall']:.4f} °C\n")
            f.write(f"  • Overall MAE:  {metrics['mae_overall']:.4f} °C\n")
            feature_names = list(data_dict.get('features', []))
            for i, feat in enumerate(feature_names):
                r2   = metrics['r2_per_feature'][i]   if i < len(metrics['r2_per_feature'])   else float('nan')
                rmse = metrics['rmse_per_feature'][i]  if i < len(metrics['rmse_per_feature'])  else float('nan')
                mae  = metrics['mae_per_feature'][i]   if i < len(metrics['mae_per_feature'])   else float('nan')
                f.write(f"  • {feat}: R²={r2:.4f}  RMSE={rmse:.4f}°C  MAE={mae:.4f}°C\n")
            f.write("\n")

        physics_r1 = int(df_results['Rule_HeatTransfer'].sum()) if 'Rule_HeatTransfer' in df_results else 0
        physics_r2 = int(df_results['Rule_Control'].sum())      if 'Rule_Control'      in df_results else 0
        physics_both = int(df_results['LSTM_and_Rule'].sum())   if 'LSTM_and_Rule'     in df_results else 0
        if physics_r1 or physics_r2:
            f.write("PHYSICS RULE CHECKS (Yeom-style hybrid layer):\n")
            f.write(f"  • Rule 1 (heat-transfer fault, dT<0.5°C & Mod>30%): {physics_r1} windows\n")
            f.write(f"  • Rule 2 (control fault, 4h-mean |SP error|>1.5°C): {physics_r2} windows\n")
            f.write(f"  • LSTM anomaly + any rule (highest-confidence):      {physics_both} windows\n\n")

        # Gap characterization ------------------------------------------------
        gap_info = processed_data.get('gap_info', {})
        if gap_info:
            f.write("GAP CHARACTERIZATION:\n")
            f.write(f"  • Gap rate:              {gap_info.get('gap_rate_pct', 0):.2f}% of total timeline\n")
            f.write(f"  • Distinct gap episodes: {gap_info.get('n_gap_episodes', 0)}\n")
            f.write(f"  • Longest gap:           {gap_info.get('max_gap_hours', 0):.1f} h\n")
            f.write(f"  • Mean gap duration:     {gap_info.get('mean_gap_hours', 0):.1f} h\n")
            f.write(f"  • Clustering ratio:      {gap_info.get('clustering_ratio', 1):.2f}\n\n")

        # Gap vs. anomaly FP diagnostic ---------------------------------------
        gd = results.get('gap_diagnostic', {})
        if gd:
            f.write("GAP vs. ANOMALY FP DIAGNOSTIC:\n")
            f.write(f"  • Total anomalies:  {gd.get('n_anomalies', 0)}\n")
            f.write(f"  • Gap-adjacent:     {gd.get('n_gap_adjacent', 0)} ({gd.get('gap_adjacent_pct', 0):.1f}%)\n")
            f.write(f"  • Genuine estimate: {gd.get('n_genuine_estimate', 0)}\n\n")

        # Episode-level statistics (Fix 3) ------------------------------------
        es = results.get('episode_stats', {})
        if es and es.get('n_episodes', 0) > 0:
            f.write("ANOMALY EPISODE STATISTICS:\n")
            f.write(f"  • Distinct episodes : {es['n_episodes']}\n")
            f.write(f"  • Mean duration     : {es['mean_duration_steps']} windows ({es['mean_duration_h']} h)\n")
            f.write(f"  • Longest episode   : {es['max_duration_steps']} windows ({es['max_duration_h']} h)\n")
            f.write("  • Episode list (start → end, windows):\n")
            for start_s, end_s, n_win in es.get('episodes', []):
                f.write(f"      {start_s}  →  {end_s}  ({n_win} windows)\n")
            f.write("\n")

        ces = results.get('candidate_episode_stats', {})
        if ces and ces.get('n_episodes', 0) > 0:
            f.write("CANDIDATE EPISODE STATISTICS:\n")
            f.write(f"  • Distinct episodes : {ces['n_episodes']}\n")
            f.write(f"  • Mean duration     : {ces['mean_duration_steps']} windows ({ces['mean_duration_h']} h)\n")
            f.write(f"  • Longest episode   : {ces['max_duration_steps']} windows ({ces['max_duration_h']} h)\n")
            f.write("  • Episode list (start → end, windows):\n")
            for start_s, end_s, n_win in ces.get('episodes', []):
                f.write(f"      {start_s}  →  {end_s}  ({n_win} windows)\n")
            f.write("\n")
    
    print(f"   ✓ Summary saved to: {summary_path}")
    print(f"{'='*70}\n")
    
    return df_results


def main():
    """
    Main execution pipeline - orchestrates all components
    """
    
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print(f"║{'':^76}║")
    print(f"║{'🎯 LSTM AUTOENCODER - HVAC ANOMALY DETECTION':^76}║")
    print(f"║{'':^76}║")
    print(f"║{'Unsupervised Learning for HVAC System Monitoring':^76}║")
    print(f"║{'':^76}║")
    print(f"{'='*80}")
    print(f"{'='*80}\n")
    
    # ══════════════════════════════════════════════════════════
    # STEP 1: Load Configuration
    # ══════════════════════════════════════════════════════════
    print("📋 STEP 1: Loading configuration...")
    # Accept accidental inline comments in values, e.g. "batch_size = 16 ; note".
    config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
    # Use main config.ini from project root (2 levels up from scripts/lstm_anomaly_detection)
    config_path = Path(__file__).parent.parent.parent / 'config.ini'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config.read(config_path, encoding='utf-8-sig')
    print(f"   ✓ Configuration loaded from: {config_path}")

    # Auto-load season-matched Optuna best-params so switching season = Summer/Winter
    # in config.ini is all that is needed — no manual architecture edits required.
    building_id = config.get('global', 'building_id')
    ahu_unit    = config.get('global', 'ahu_unit')
    season      = config.get('global', 'season')
    year        = config.get('global', 'year')
    best_params_path = Path(__file__).parent / f"optuna_best_params_{building_id}_{ahu_unit}_{season}{year}.ini"
    if not best_params_path.exists():
        raise FileNotFoundError(
            f"Missing season-specific Optuna params file: {best_params_path}\n"
            f"Run 10_optuna_tuner.py for {building_id}_{ahu_unit}_{season}{year} first."
        )

    bp = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
    bp.read(best_params_path, encoding='utf-8')
    if not bp.has_section('lstm_autoencoder_best_params'):
        raise ValueError(
            f"Invalid Optuna params file (missing [lstm_autoencoder_best_params]): {best_params_path}"
        )

    _HP_KEYS = {'encoder_units', 'bottleneck_units', 'decoder_units',
                'dropout', 'learning_rate', 'batch_size', 'sequence_length'}
    for key, val in bp.items('lstm_autoencoder_best_params'):
        if key in _HP_KEYS:
            config.set('lstm_autoencoder', key, val)
    print(f"   ✓ Optuna best params auto-loaded from: {best_params_path.name}")
    print()
    
    # ══════════════════════════════════════════════════════════
    # STEP 2: Load and Prepare Data
    # ══════════════════════════════════════════════════════════
    print("📂 STEP 2: Loading and preparing data...")
    data_dict = load_and_prepare_data(config)
    
    # ══════════════════════════════════════════════════════════
    # STEP 3: Preprocess Data (Scale, Split, Create Sequences)
    # ══════════════════════════════════════════════════════════
    print("⚙️  STEP 3: Preprocessing data...")
    processed_data = preprocess_data(data_dict, config)
    
    # ══════════════════════════════════════════════════════════
    # STEP 4: Build LSTM Autoencoder Model
    # ══════════════════════════════════════════════════════════
    print("🏗️  STEP 4: Building LSTM Autoencoder...")
    model = build_autoencoder(processed_data['input_shape'], config)
    print_model_architecture(model)
    
    # ══════════════════════════════════════════════════════════
    # STEP 5: Train the Model
    # ══════════════════════════════════════════════════════════
    print("🚀 STEP 5: Training the model...")
    # Optional dev shortcut: skip training and reuse existing best checkpoint.
    # This is useful when you only changed evaluation settings (e.g., persistence/grace period)
    # and want to regenerate anomaly tables/plots quickly.
    skip_training = os.environ.get('LSTM_SKIP_TRAINING', '').strip().lower() in {'1', 'true', 'yes', 'y'}
    if skip_training:
        try:
            # Prefer Keras' native loader (more robust across Keras versions).
            import keras
            plots_folder = config.get('paths', 'plots_folder', fallback='./plots')
            building_id = config.get('global', 'building_id')
            ahu_unit = config.get('global', 'ahu_unit')
            season = config.get('global', 'season')
            year = config.get('global', 'year')
            model_name = f"best_autoencoder_{building_id}_{ahu_unit}_{season}{year}.keras"
            model_path = Path(__file__).parent.parent.parent / plots_folder.replace('./', '') / 'lstm_autoencoder' / 'models' / model_name
            if model_path.exists():
                # Keras 3 sometimes cannot fully deserialize legacy compile state.
                # We only need the weights/graph for inference.
                model = keras.saving.load_model(str(model_path), compile=False, safe_mode=False)
                print(f"   ℹ️  Training skipped — loaded checkpoint: {model_path}")

                class _FakeHistory:
                    history = {'loss': [], 'val_loss': []}

                history = _FakeHistory()
            else:
                print(f"   ⚠️  LSTM_SKIP_TRAINING set but checkpoint not found: {model_path}")
                print("      Falling back to training.")
                history = train_model(model, processed_data, config)
        except Exception as e:
            print(f"   ⚠️  Could not skip training ({e}). Falling back to training.")
            history = train_model(model, processed_data, config)
    else:
        history = train_model(model, processed_data, config)
    
    # ══════════════════════════════════════════════════════════
    # STEP 6: Detect Anomalies and Calculate Metrics
    # ══════════════════════════════════════════════════════════
    print("🔍 STEP 6: Detecting anomalies...")
    results = detect_anomalies(model, processed_data, config)
    
    print("📊 STEP 7: Calculating evaluation metrics...")
    metrics = calculate_metrics(results, processed_data)

    # Finalize versioned metadata with eval metrics (RMSE, MAE, R2, anomaly_rate)
    trainer.save_version_metrics(getattr(history, 'version_dir', None), metrics)
    
    # ══════════════════════════════════════════════════════════
    # STEP 8: Visualize Results
    # ══════════════════════════════════════════════════════════
    print("📊 STEP 8: Generating visualizations...")
    plot_results(history, results, processed_data, data_dict, config)
    
    # ══════════════════════════════════════════════════════════
    # STEP 9: Save Results
    # ══════════════════════════════════════════════════════════
    print("💾 STEP 9: Saving results to CSV...")
    df_results = save_results_to_csv(results, processed_data, data_dict, config, metrics=metrics)
    
    # ══════════════════════════════════════════════════════════
    # STEP 10: Generate Word Report
    # ══════════════════════════════════════════════════════════
    print("📝 STEP 10: Generating Word report...")
    # Get plots directory to pass to report generator
    plots_folder = config.get('paths', 'plots_folder', fallback='./plots')
    folder_name = f"{config.get('global', 'building_id')}_{config.get('global', 'ahu_unit')}_{config.get('global', 'season')}{config.get('global', 'year')}"
    plots_dir = Path(__file__).parent.parent.parent / plots_folder.replace('./','') / 'lstm_autoencoder' / folder_name
    
    word_report_path = generate_lstm_report(results, data_dict, processed_data, config, plots_dir, metrics=metrics)
    
    # ══════════════════════════════════════════════════════════
    # STEP 11: Anomaly Context Plots
    # ══════════════════════════════════════════════════════════
    print("🔎 STEP 11: Generating anomaly context plots (±10 windows)...")
    context_out_dir = plots_dir / "anomaly_context"
    feat_df_raw = data_dict['df'].reset_index().rename(columns={data_dict['df'].index.name or 'index': 'Timestamp'})
    run_context_plots(
        df_results   = df_results,
        feat_df      = feat_df_raw,
        feature_names= data_dict['features'],
        out_dir      = context_out_dir,
        config       = config,
    )

    # ══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print(f"║{'':^76}║")
    print(f"║{'✅ ANALYSIS COMPLETE!':^76}║")
    print(f"║{'':^76}║")
    print(f"{'='*80}")
    print(f"{'='*80}\n")
    
    print("📊 FINAL RESULTS:")
    print(f"   • Test samples analyzed: {len(results['test_errors']):,}")
    print(f"   • Anomalies detected: {results['filtered_anomalies'].sum():,} ({results['filtered_anomalies'].sum()/len(results['test_errors'])*100:.2f}%)")
    print(f"   • Severity breakdown:")
    print(f"      - Critical: {results['n_critical']}")
    print(f"      - Moderate: {results['n_moderate']}")
    print(f"      - Minor: {results['n_minor']}")
    print(f"   • Reconstruction quality (R²): {metrics['r2_overall']:.4f}")
    
    # Build output paths
    plots_folder = config.get('paths', 'plots_folder', fallback='./plots')
    processed_folder = config.get('paths', 'processed_folder', fallback='./processed_data')
    building_id = config.get('global', 'building_id')
    ahu_unit = config.get('global', 'ahu_unit')
    season = config.get('global', 'season')
    year = config.get('global', 'year')
    folder_name = f"{building_id}_{ahu_unit}_{season}{year}"
    
    project_root = Path(__file__).parent.parent.parent
    plots_dir = project_root / plots_folder.replace('./','') / 'lstm_autoencoder' / folder_name
    results_dir = project_root / processed_folder.replace('./','') / 'lstm_autoencoder' / folder_name
    model_dir = project_root / plots_folder.replace('./','') / 'lstm_autoencoder' / 'models'
    
    print(f"\n   📂 Output locations:")
    print(f"      • Plots: {plots_dir}")
    print(f"      • Anomaly context plots: {plots_dir / 'anomaly_context'}")
    print(f"      • Results CSV: {results_dir}")
    print(f"      • Model: {model_dir}")
    
    print(f"\n{'='*80}")
    print(f"Done! Check your plots and results.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR OCCURRED:\n{e}\n")
        import traceback
        traceback.print_exc()
        print(f"\n💡 TIP: Check your config.ini file and data paths.")
