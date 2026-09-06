"""
IMPROVED MODEL - Stacked LSTM Autoencoder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIXES:
1. Stacked LSTM layers (64 → 32 bottleneck → 64)
2. Better capacity for 8 diverse features
3. Proper dropout placement
4. Gradient clipping to prevent exploding gradients

ARCHITECTURE:
  Encoder: LSTM(64) → LSTM(32) [bottleneck]
  Decoder: LSTM(32) → LSTM(64) → Dense(n_features)
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


def build_improved_autoencoder(input_shape, config):
    """
    Build STACKED LSTM Autoencoder with better capacity
    
    IMPROVED ARCHITECTURE:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OLD (too simple):
      Encoder: LSTM(32)
      Decoder: LSTM(32)
      
    NEW (better capacity):
      Encoder: LSTM(64, return_sequences=True) → LSTM(32, return_sequences=False)
      Decoder: LSTM(32, return_sequences=True) → LSTM(64, return_sequences=True)
    
    WHY THIS WORKS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. First encoder LSTM(64): Processes raw sequences, captures complex patterns
    2. Second encoder LSTM(32): Compresses to bottleneck (essential patterns only)
    3. First decoder LSTM(32): Begins reconstruction from compressed representation
    4. Second decoder LSTM(64): Expands to full sequence detail
    
    Rule of thumb: units ≈ 4-8× number of features
    For 8 features: 32-64 units is appropriate
    
    Args:
        input_shape: tuple (sequence_length, n_features)
        config: ConfigParser object
        
    Returns:
        compiled Keras model
    """
    
    seq_len, n_features = input_shape
    
    # Get parameters from config
    encoder_units = config.getint('lstm_autoencoder', 'encoder_units', fallback=64)
    bottleneck_units = config.getint('lstm_autoencoder', 'bottleneck_units', fallback=32)
    decoder_units = config.getint('lstm_autoencoder', 'decoder_units', fallback=64)
    dropout_rate = config.getfloat('lstm_autoencoder', 'dropout', fallback=0.2)
    learning_rate = config.getfloat('lstm_autoencoder', 'learning_rate', fallback=0.0005)
    
    print(f"\n{'='*70}")
    print(f"🏗️  BUILDING STACKED LSTM AUTOENCODER (IMPROVED)")
    print(f"{'='*70}")
    print(f"   Input shape: ({seq_len}, {n_features})")
    print(f"   Architecture:")
    print(f"      Encoder Layer 1:  LSTM({encoder_units}, return_sequences=True)")
    print(f"      Encoder Layer 2:  LSTM({bottleneck_units}, return_sequences=False) [BOTTLENECK]")
    print(f"      Decoder Layer 1:  LSTM({bottleneck_units}, return_sequences=True)")
    print(f"      Decoder Layer 2:  LSTM({decoder_units}, return_sequences=True)")
    print(f"      Output:           TimeDistributed(Dense({n_features}))")
    print(f"   Dropout: {dropout_rate}")
    print(f"   Optimizer: Adam (lr={learning_rate}, clipnorm=1.0)")
    print(f"   Loss: {config.get('lstm_autoencoder', 'loss_function', fallback='mae').upper()} (robust to near-constant features)")
    print(f"{'='*70}\n")
    
    # ═══════════════════════════════════════════════════════════════
    # ENCODER: Two-layer compression
    # ═══════════════════════════════════════════════════════════════
    inputs = Input(shape=(seq_len, n_features), name='encoder_input')
    
    # Layer 1: Process input sequence (return_sequences=True to feed to next LSTM)
    encoded = LSTM(encoder_units, return_sequences=True, name='encoder_lstm1')(inputs)
    
    if dropout_rate > 0:
        encoded = Dropout(dropout_rate, name='encoder_dropout1')(encoded)
    
    # Layer 2: Compress to bottleneck (return_sequences=False → single vector)
    encoded = LSTM(bottleneck_units, return_sequences=False, name='encoder_lstm2_bottleneck')(encoded)
    
    if dropout_rate > 0:
        encoded = Dropout(dropout_rate, name='encoder_dropout2')(encoded)
    
    # ═══════════════════════════════════════════════════════════════
    # DECODER: Two-layer reconstruction
    # ═══════════════════════════════════════════════════════════════
    
    # Repeat bottleneck vector for each timestep
    decoded = RepeatVector(seq_len, name='decoder_repeat')(encoded)
    
    # Layer 1: Begin reconstruction (return_sequences=True for next LSTM)
    decoded = LSTM(bottleneck_units, return_sequences=True, name='decoder_lstm1')(decoded)
    
    if dropout_rate > 0:
        decoded = Dropout(dropout_rate, name='decoder_dropout1')(decoded)
    
    # Layer 2: Expand reconstruction (return_sequences=True for final output)
    decoded = LSTM(decoder_units, return_sequences=True, name='decoder_lstm2')(decoded)
    
    if dropout_rate > 0:
        decoded = Dropout(dropout_rate, name='decoder_dropout2')(decoded)
    
    # Final output layer: Map to original feature space
    outputs = TimeDistributed(Dense(n_features, activation='linear'), name='decoder_output')(decoded)
    
    # ═══════════════════════════════════════════════════════════════
    # MODEL COMPILATION
    # ═══════════════════════════════════════════════════════════════
    autoencoder = Model(inputs, outputs, name='Stacked_LSTM_Autoencoder')
    
    # Use Adam with gradient clipping to prevent exploding gradients
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    
    # Compile with MAE loss
    # MAE is more robust than MSE when features include near-constant signals
    # (e.g. binary fan at 60%/80%) — MSE would square-penalise tiny reconstruction
    # wiggles and inflate the anomaly threshold unfairly.
    loss_fn = config.get('lstm_autoencoder', 'loss_function', fallback='mae')
    autoencoder.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['mse']
    )
    
    return autoencoder


def print_model_architecture(model):
    """Print detailed model summary"""
    print(f"\n{'='*70}")
    print(f"📋 MODEL ARCHITECTURE SUMMARY")
    print(f"{'='*70}\n")
    
    model.summary()
    
    print(f"\n{'='*70}")
    print(f"✅ Model built successfully!")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Trainable parameters: {sum([K.count_params(w) for w in model.trainable_weights]):,}")
    print(f"{'='*70}\n")


# ═════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY: Alias for old function name
# ═════════════════════════════════════════════════════════════════════════════
def build_autoencoder(input_shape, config):
    """
    Wrapper for build_improved_autoencoder to maintain compatibility
    """
    # Check if config has new architecture parameters
    if config.has_option('lstm_autoencoder', 'encoder_units'):
        # Use improved stacked architecture
        return build_improved_autoencoder(input_shape, config)
    else:
        # Fall back to simple architecture (backward compatibility)
        print("⚠️  Using simple architecture (config missing encoder_units/bottleneck_units)")
        print("   Add these to config for improved performance:")
        print("   encoder_units = 64")
        print("   bottleneck_units = 32")
        print("   decoder_units = 64\n")
        
        # Build simple model as fallback
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        seq_len, n_features = input_shape
        lstm_units = config.getint('lstm_autoencoder', 'lstm_units', fallback=32)
        dropout_rate = config.getfloat('lstm_autoencoder', 'dropout', fallback=0.0)
        learning_rate = config.getfloat('lstm_autoencoder', 'learning_rate', fallback=0.001)
        
        inputs = Input(shape=(seq_len, n_features))
        encoded = LSTM(lstm_units, return_sequences=False)(inputs)
        if dropout_rate > 0:
            encoded = Dropout(dropout_rate)(encoded)
        
        decoded = RepeatVector(seq_len)(encoded)
        decoded = LSTM(lstm_units, return_sequences=True)(decoded)
        if dropout_rate > 0:
            decoded = Dropout(dropout_rate)(decoded)
        
        outputs = TimeDistributed(Dense(n_features, activation='linear'))(decoded)
        
        autoencoder = Model(inputs, outputs)
        autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
        
        return autoencoder


# ═════════════════════════════════════════════════════════════════════════════
# TEST SCRIPT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import configparser
    from pathlib import Path
    
    print("\n🧪 TESTING IMPROVED MODEL.PY\n")
    
    # Load configuration
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent.parent / 'config.ini'
    config.read(config_path, encoding='utf-8-sig')
    
    try:
        # Test model building — use values from config instead of hardcoding
        seq_length = config.getint('lstm_autoencoder', 'sequence_length')
        season     = config.get('global', 'season', fallback='Winter').strip().lower()
        season_key = f"features_{season}"
        if config.has_option('lstm_autoencoder', season_key):
            feature_list = [f.strip() for f in config.get('lstm_autoencoder', season_key).split(',')]
        else:
            feature_list = [f.strip() for f in config.get('lstm_autoencoder', 'features', fallback='').split(',')]
        n_features = len(feature_list)
        print(f"   Config → seq_length={seq_length}, features ({season})={feature_list}")
        input_shape = (seq_length, n_features)
        
        # Build improved model
        model = build_improved_autoencoder(input_shape, config)
        
        # Print architecture
        print_model_architecture(model)
        
        print(f"\n✅ IMPROVED MODEL TEST SUCCESSFUL!")
        print(f"   Input shape: {input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Model ready for training!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
