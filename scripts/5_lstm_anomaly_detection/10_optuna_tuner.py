"""
10_optuna_tuner.py  —  Optuna Hyperparameter Optimisation for LSTM Autoencoder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Purpose:
    Replace manual config.ini tuning with Bayesian (TPE) optimisation.
    Searches for the best combination of:
        encoder_units, bottleneck_units, dropout,
        learning_rate, batch_size, sequence_length

How to run:
    cd scripts/5_lstm_anomaly_detection
    python 10_optuna_tuner.py

Resumable:
    If interrupted, simply re-run — the SQLite study (optuna_study.db) persists
    all completed trials and continues from where it left off.

After completion:
    1. Read the printed best parameters (or open optuna_best_params.ini)
    2. Values are auto-applied into [lstm_autoencoder] in config.ini
    3. Run python 1_main.py for the final full-epoch training run

Design notes:
    • Raw data is loaded ONCE and reused across all 30 trials.
    • Preprocessed sequences are CACHED per unique sequence_length — so if
      6 trials share sequence_length=24, preprocessing runs only once.
    • MedianPruner kills trials whose val_loss is worse than the median of
      completed trials at the same epoch (after 20 epochs warm-up).
    • TFKerasPruningCallback (optuna-integration) reports per-epoch val_loss
      to the pruner for within-trial early termination.
    • bottleneck_units >= encoder_units trials are pruned immediately
      (architectural nonsense — no compression benefit).
    • decoder_units always mirrors encoder_units (symmetric autoencoder).
"""

import os
import sys
import copy
import configparser
import importlib
import warnings
from pathlib import Path

import numpy as np

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH  = PROJECT_ROOT / 'config.ini'

# Add script dir to path so relative imports (2_data_loader, etc.) work
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(SCRIPT_DIR)

# ─── Suppress TF / CUDA noise before importing TF ────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# ─── Load existing pipeline modules ──────────────────────────────────────────
data_loader_mod  = importlib.import_module('2_data_loader')
preprocessor_mod = importlib.import_module('3_preprocessor')
model_mod        = importlib.import_module('4_model')


# =============================================================================
# Config helpers
# =============================================================================

def load_config() -> configparser.ConfigParser:
    """
    Loads project config.ini with inline comment support so users can
    annotate config values with semicolon comments on the same line
    without causing parse errors.

    Returns:
        ConfigParser: Parsed project configuration.
    """
    # inline_comment_prefixes allows safe parsing even if a user accidentally
    # writes "key = value ; comment" on one line.
    config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
    config.read(CONFIG_PATH, encoding='utf-8-sig')   # utf-8-sig strips the BOM (\ufeff)
    return config


def make_trial_config(
    base_config: configparser.ConfigParser,
    overrides: dict
) -> configparser.ConfigParser:
    """
    Creates an isolated copy of the base config with trial-specific
    hyperparameter overrides so each Optuna trial has its own config
    without mutating the shared base config.

    Args:
        base_config (ConfigParser): The shared project configuration.
        overrides (dict): Hyperparameter key-value pairs to apply
                          under [lstm_autoencoder].

    Returns:
        ConfigParser: Deep-copied config with trial overrides applied.
    """
    cfg = copy.deepcopy(base_config)
    for key, val in overrides.items():
        cfg.set('lstm_autoencoder', key, str(val))
    return cfg


def parse_search_space(config: configparser.ConfigParser, key: str) -> list:
    """
    Parse 'search_X = a, b, c' from [optuna] into a typed Python list.
    Integers are returned as int, floats as float, anything else as str.
    """
    raw   = config.get('optuna', key, fallback='')
    parts = [v.strip() for v in raw.split(',') if v.strip()]
    result = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            try:
                result.append(float(p))
            except ValueError:
                result.append(p)
    return result


# =============================================================================
# Optuna Objective
# =============================================================================

class Objective:
    """
    Callable passed to study.optimize().

    Lifecycle
    ---------
    __init__  : Load raw data once; parse search spaces from config.
    __call__  : For each trial — sample params, preprocess (cached),
                build model, train, return val_loss.
    """

    def __init__(self, base_config: configparser.ConfigParser):
        """
        Parses all search spaces from config and pre-loads raw data once
        so it is not re-read on every trial, which would dominate tuning time.

        Args:
            base_config (ConfigParser): The shared project configuration.
        """
        self.base_config  = base_config
        self._data_dict   = None   # loaded once
        self._proc_cache  = {}     # sequence_length → processed_data dict

        # ── Parse search spaces ───────────────────────────────────────────
        self.encoder_choices    = parse_search_space(base_config, 'search_encoder_units')
        self.bottleneck_choices = parse_search_space(base_config, 'search_bottleneck_units')
        self.dropout_choices    = parse_search_space(base_config, 'search_dropout')
        self.lr_choices         = parse_search_space(base_config, 'search_learning_rate')
        self.batch_choices      = parse_search_space(base_config, 'search_batch_size')
        self.seq_choices        = parse_search_space(base_config, 'search_sequence_length')

        self.trial_epochs   = base_config.getint('optuna', 'trial_epochs',   fallback=60)
        self.trial_patience = base_config.getint('optuna', 'trial_patience', fallback=15)

        print("\n" + "=" * 70)
        print("🔍 SEARCH SPACES:")
        print(f"   encoder_units:    {self.encoder_choices}")
        print(f"   bottleneck_units: {self.bottleneck_choices}")
        print(f"   dropout:          {self.dropout_choices}")
        print(f"   learning_rate:    {self.lr_choices}")
        print(f"   batch_size:       {self.batch_choices}")
        print(f"   sequence_length:  {self.seq_choices}")
        print("=" * 70)

        # ── Pre-load raw data (once) ──────────────────────────────────────
        print("\n📂 Pre-loading raw data (done once for all trials) ...")
        self._data_dict = data_loader_mod.load_and_prepare_data(base_config)
        print("✅ Data pre-loaded.\n")

    # ── Internal: cached preprocessor ────────────────────────────────────────
    def _get_processed(self, seq_len: int) -> dict:
        """
        Return preprocessed sequences for a given sequence_length.
        Result is cached — if 6 trials share seq_len=24, the expensive
        scaling+split+windowing step runs only once.
        """
        if seq_len not in self._proc_cache:
            print(f"   ⚙️  Preprocessing seq_len={seq_len} "
                  f"(first time, cached for future trials)...")
            cfg = make_trial_config(self.base_config, {'sequence_length': seq_len})
            self._proc_cache[seq_len] = preprocessor_mod.preprocess_data(
                copy.deepcopy(self._data_dict), cfg
            )
        return self._proc_cache[seq_len]

    # ── Trial objective ───────────────────────────────────────────────────────
    def __call__(self, trial) -> float:
        """
        Runs one Optuna trial: samples hyperparameters, retrieves preprocessed
        sequences from cache (or runs preprocessing if sequence_length is new),
        builds and trains the model, then returns validation loss for Optuna
        to minimise.

        Args:
            trial: Optuna Trial object used to sample hyperparameter values.

        Returns:
            float: Validation loss for this trial (lower is better).
        """
        import optuna

        # Per-trial seed: ensures weight initialisation is deterministic and
        # reproducible across tuner re-runs (same trial → same random start).
        tf.random.set_seed(42 + trial.number)

        # ── Sample ───────────────────────────────────────────────────────
        encoder_units    = trial.suggest_categorical('encoder_units',    self.encoder_choices)
        bottleneck_units = trial.suggest_categorical('bottleneck_units', self.bottleneck_choices)
        dropout          = trial.suggest_categorical('dropout',          self.dropout_choices)
        learning_rate    = trial.suggest_categorical('learning_rate',    self.lr_choices)
        batch_size       = trial.suggest_categorical('batch_size',       self.batch_choices)
        seq_len          = trial.suggest_categorical('sequence_length',  self.seq_choices)

        # ── Architectural sanity guard ────────────────────────────────────
        # A bottleneck >= encoder is not a compression — prune immediately.
        if bottleneck_units >= encoder_units:
            raise optuna.exceptions.TrialPruned(
                f"Pruned: bottleneck({bottleneck_units}) >= encoder({encoder_units}) "
                f"→ no compression benefit"
            )

        print(f"\n{'─'*62}")
        print(f"  Trial {trial.number:>3} | "
              f"enc={encoder_units:>3}  bot={bottleneck_units:>3}  "
              f"do={dropout}  lr={learning_rate}  "
              f"bs={batch_size:>2}  sl={seq_len:>2}")
        print(f"{'─'*62}")

        # ── Override trial config ─────────────────────────────────────────
        trial_cfg = make_trial_config(self.base_config, {
            'encoder_units':    encoder_units,
            'bottleneck_units': bottleneck_units,
            'decoder_units':    encoder_units,   # symmetric AE: decoder mirrors encoder
            'dropout':          dropout,
            'learning_rate':    learning_rate,
            'batch_size':       batch_size,
            'sequence_length':  seq_len,
        })

        # ── Get preprocessed data (cached by seq_len) ─────────────────────
        try:
            processed = self._get_processed(seq_len)
        except Exception as exc:
            print(f"  ⚠️  Preprocessing error: {exc} — pruning trial")
            raise optuna.exceptions.TrialPruned(f"Preprocessing failed: {exc}")

        X_train    = processed['X_train']
        X_val      = processed['X_val']
        clean_mask = processed.get('clean_sequence_mask', None)

        # Apply clean training filter (same as 5_trainer.py)
        if clean_mask is not None and clean_mask.sum() > 0:
            X_train = X_train[clean_mask]

        n_features  = processed['input_shape'][1]
        input_shape = (seq_len, n_features)

        if len(X_train) == 0 or len(X_val) == 0:
            raise optuna.exceptions.TrialPruned("Empty training or validation set")

        # ── Build model ───────────────────────────────────────────────────
        tf.keras.backend.clear_session()
        model = model_mod.build_improved_autoencoder(input_shape, trial_cfg)

        # ── Callbacks ────────────────────────────────────────────────────
        from tensorflow.keras.callbacks import EarlyStopping

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.trial_patience,
                restore_best_weights=True,
                verbose=0
            )
        ]

        # Add per-epoch pruning callback if optuna-integration is installed
        try:
            from optuna.integration import TFKerasPruningCallback
            callbacks.append(TFKerasPruningCallback(trial, 'val_loss'))
        except (ImportError, AttributeError):
            pass  # MedianPruner still prunes between trials

        # ── Train ─────────────────────────────────────────────────────────
        history = model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=self.trial_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0          # suppress per-epoch output; trial summary printed below
        )

        best_val = float(min(history.history.get('val_loss', [float('inf')])))
        actual_epochs = len(history.history.get('val_loss', []))
        print(f"  → val_loss = {best_val:.6f}  (stopped at epoch {actual_epochs})")

        # Free GPU/CPU memory
        del model
        tf.keras.backend.clear_session()

        return best_val


# =============================================================================
# Results output
# =============================================================================

# Hyperparameter keys that Optuna manages (all lower-case, as configparser normalises them)
_OPTUNA_HP_KEYS = frozenset({
    'encoder_units', 'bottleneck_units', 'decoder_units',
    'dropout', 'learning_rate', 'batch_size', 'sequence_length',
})


def write_best_params(best_params: dict, out_path: Path) -> None:
    """Write the best-found parameters as a readable .ini snippet."""
    lines = [
        "; ============================================================",
        "; OPTUNA BEST HYPERPARAMETERS",
        "; These values have already been applied to config.ini",
        "; automatically. This file is kept as a reference record.",
        "; Run: python 1_main.py",
        "; ============================================================",
        "",
        "[lstm_autoencoder_best_params]",
    ]
    for k, v in best_params.items():
        lines.append(f"{k:<25} = {v}")
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"\n📝 Best params reference file written to: {out_path}")


def apply_best_params_to_config(best_params: dict, config_path: Path) -> None:
    """
    Patch [lstm_autoencoder] in config.ini with the best-found hyperparameters.

    All comments, blank lines, and every other section are preserved exactly.
    Only the value part of lines whose key matches an Optuna HP key is rewritten.
    The file is written back with utf-8-sig (BOM) to match the original encoding.
    """
    # Build a lower-case lookup: 'encoder_units' → 128
    updates = {k.lower(): v for k, v in best_params.items() if k.lower() in _OPTUNA_HP_KEYS}

    lines   = config_path.read_text(encoding='utf-8-sig').splitlines(keepends=True)
    output  = []
    in_lstm = False
    patched = {}  # key → new_value (for reporting)

    for line in lines:
        stripped = line.strip()

        # Track which section we are in
        if stripped.startswith('['):
            in_lstm = stripped.lower() == '[lstm_autoencoder]'
            output.append(line)
            continue

        if in_lstm and not stripped.startswith(';') and '=' in line:
            # Parse   key = value   (leading whitespace preserved)
            eq_pos     = line.index('=')
            raw_key    = line[:eq_pos].strip().lower()
            if raw_key in updates:
                indent    = line[: len(line) - len(line.lstrip())]
                orig_key  = line[:eq_pos].rstrip()          # keep original casing / spacing
                new_line  = f"{orig_key} = {updates[raw_key]}\n"
                output.append(new_line)
                patched[raw_key] = updates[raw_key]
                continue

        output.append(line)

    config_path.write_text(''.join(output), encoding='utf-8-sig')

    print(f"\n✅ config.ini patched automatically with best hyperparameters:")
    for k, v in patched.items():
        print(f"   [lstm_autoencoder]  {k:<25} = {v}")
    if len(patched) < len(updates):
        missing = set(updates) - set(patched)
        print(f"   ⚠️  Keys not found in [lstm_autoencoder]: {missing}")
    print(f"   File: {config_path}")


def print_results_table(study) -> None:
    """
    Gives a quick ranked overview of the best-performing trials at the
    end of the study without needing to open the SQLite database manually.

    Args:
        study: Completed Optuna Study object.
    """
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        print("   No completed trials.")
        return

    sorted_trials = sorted(completed, key=lambda t: t.value)

    print(f"\n{'='*70}")
    print(f"📊 TOP {min(5, len(sorted_trials))} TRIALS:")
    print(f"{'─'*70}")
    header = (f"  {'Trial':>5}  {'val_loss':>10}  "
              f"{'enc':>5}  {'bot':>5}  {'drop':>6}  "
              f"{'lr':>8}  {'bs':>4}  {'sl':>4}")
    print(header)
    print(f"{'─'*70}")
    for t in sorted_trials[:5]:
        p = t.params
        enc  = p.get('encoder_units',    '—')
        bot  = p.get('bottleneck_units', '—')
        drop = p.get('dropout',          '—')
        lr   = p.get('learning_rate',    '—')
        bs   = p.get('batch_size',       '—')
        sl   = p.get('sequence_length',  '—')
        print(f"  {t.number:>5}  {t.value:>10.6f}  "
              f"{enc!s:>5}  {bot!s:>5}  {drop!s:>6}  "
              f"{lr!s:>8}  {bs!s:>4}  {sl!s:>4}")
    print(f"{'='*70}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Entry point: loads config, creates or resumes the Optuna study,
    runs all trials, writes the best parameters back to config, and
    prints a summary table.
    """
    try:
        import optuna
    except ImportError:
        print("\n❌ Optuna not installed. Run:")
        print("   pip install optuna")
        print("   pip install optuna-integration   # optional, for per-epoch pruning")
        sys.exit(1)

    # Silence Optuna's own INFO messages so our output stays readable
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    config = load_config()
    n_trials = config.getint('optuna', 'n_trials', fallback=30)

    # Auto-build study_name and DB path from the 4 global variables
    # Changing building_id / ahu_unit / season / year in [global] automatically
    # creates a separate study — no manual edits needed here.
    building_id = config.get('global', 'building_id', fallback='C1')
    ahu_unit    = config.get('global', 'ahu_unit',    fallback='UTA1')
    season      = config.get('global', 'season',      fallback='Summer')
    year        = config.get('global', 'year',        fallback='2025')

    study_name = f"{building_id}_{ahu_unit}_{season}{year}"
    db_file    = SCRIPT_DIR / f"optuna_{study_name}.db"
    study_db   = f"sqlite:///{db_file}"

    print(f"\n{'='*70}")
    print(f"🚀 OPTUNA HYPERPARAMETER SEARCH  —  LSTM AUTOENCODER (HVAC)")
    print(f"{'='*70}")
    print(f"   Config:    {CONFIG_PATH}")
    print(f"   Study:     {study_name}")
    print(f"   Database:  {study_db}")
    print(f"   Trials:    {n_trials}")
    print(f"   Device:    {('GPU — ' + tf.config.list_physical_devices('GPU')[0].name) if tf.config.list_physical_devices('GPU') else 'CPU'}")
    print(f"{'='*70}")

    # ── Create or resume study ──────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials=5,    # wait for 5 complete trials before pruning
        n_warmup_steps=20,     # don't prune before epoch 20
        interval_steps=5       # check every 5 epochs
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=study_db,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True    # ← resume if interrupted
    )

    already_done = len([t for t in study.trials if t.value is not None])
    if already_done > 0:
        print(f"\n   ↩️  Resuming — {already_done} trials already in database.")
        print(f"   Current best val_loss: {study.best_value:.6f} "
              f"(trial #{study.best_trial.number})\n")
    else:
        print("\n   Starting fresh study.\n")

    # ── Run optimisation ────────────────────────────────────────────────────
    objective = Objective(config)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        gc_after_trial=True,       # release memory between trials
    )

    # ── Final results ────────────────────────────────────────────────────────
    print_results_table(study)

    best = study.best_trial
    print(f"\n🏆 BEST TRIAL: #{best.number}   val_loss = {best.value:.6f}")
    print(f"\n{'='*70}")
    print(f"📋 COPY THESE VALUES INTO [lstm_autoencoder] in config.ini:")
    print(f"{'─'*70}")

    best_params = dict(best.params)
    best_params['decoder_units'] = best_params.get('encoder_units', 64)  # symmetric AE

    for k, v in best_params.items():
        print(f"   {k:<25} = {v}")

    print(f"{'='*70}")

    out_path = SCRIPT_DIR / f'optuna_best_params_{study_name}.ini'
    write_best_params(best_params, out_path)

    # Auto-apply best params directly into config.ini — no manual copy-paste needed
    apply_best_params_to_config(best_params, CONFIG_PATH)

    print(f"\n✅ DONE. Next step:")
    print(f"   Run:   python 1_main.py")
    print()


if __name__ == '__main__':
    main()
