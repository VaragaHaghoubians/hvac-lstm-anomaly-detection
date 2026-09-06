"""
config_utils.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shared configuration helpers for the HVAC Analysis Pipeline.

Placed at the project root (HVAC_project/) so every script can find it by
adding the project root to sys.path:

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config_utils import load_config, get_date_range, ...

Functions
---------
load_config          — Load config.ini with ExtendedInterpolation + BOM-safe encoding
get_date_range       — Read start/end dates from a config section
get_timezone         — Read the timezone string
get_columns_to_analyze — Parse a comma-separated column list from config
get_config_value     — Generic typed getter with fallback
"""

import configparser
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> configparser.ConfigParser:
    """
    Load config.ini with:
    • ExtendedInterpolation  → ${section:key} references work
    • utf-8-sig encoding     → BOM character (\\ufeff) is stripped automatically
    • inline_comment_prefixes → '; comment' after a value is ignored

    Parameters
    ----------
    config_path : str or Path
        Absolute or relative path to config.ini.

    Returns
    -------
    configparser.ConfigParser
        Fully populated config object.

    Raises
    ------
    FileNotFoundError
        If config_path does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config.ini not found: {config_path}")

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(),
        inline_comment_prefixes=(';', '#'),
    )
    config.read(config_path, encoding='utf-8-sig')

    # Attach the directory so scripts can resolve relative paths against it
    config._config_dir = config_path.parent  # type: ignore[attr-defined]

    return config


def get_date_range(section: str, config: configparser.ConfigParser):
    """
    Read start_date / end_date from *section*, falling back to [global] and
    then to None.

    Parameters
    ----------
    section : str
        Config section to look in first (e.g. 'data', 'lstm_autoencoder').
    config  : ConfigParser

    Returns
    -------
    tuple[str | None, str | None]
        (start_date, end_date) as strings, or None if not set / empty.
    """
    def _get(key: str) -> str | None:
        for sec in (section, 'data', 'global'):
            try:
                val = config.get(sec, key, fallback='').strip()
                if val:
                    return val
            except (configparser.NoSectionError, configparser.NoOptionError):
                pass
        return None

    return _get('start_date'), _get('end_date')


def get_timezone(config: configparser.ConfigParser) -> str:
    """
    Return the timezone string from [global] timezone, defaulting to
    'Europe/Rome' if not set.

    Parameters
    ----------
    config : ConfigParser

    Returns
    -------
    str
    """
    return config.get('global', 'timezone', fallback='Europe/Rome').strip()


def get_columns_to_analyze(
    section: str,
    config: configparser.ConfigParser,
    column_key: str = 'columns_to_analyze',
) -> list[str]:
    """
    Parse a comma-separated list of column names from config.

    Parameters
    ----------
    section    : str   Config section to read from.
    config     : ConfigParser
    column_key : str   Key inside *section* (default: 'columns_to_analyze').

    Returns
    -------
    list[str]
        Stripped, non-empty column names.  Empty list if key missing.
    """
    try:
        raw = config.get(section, column_key, fallback='').strip()
    except (configparser.NoSectionError, configparser.NoOptionError):
        return []

    if not raw:
        return []
    return [col.strip() for col in raw.split(',') if col.strip()]


def get_config_value(
    section: str,
    key: str,
    config: configparser.ConfigParser,
    fallback=None,
    as_type=None,
):
    """
    Generic typed getter with fallback.

    Parameters
    ----------
    section  : str
    key      : str
    config   : ConfigParser
    fallback : any     Value returned when key / section is absent.
    as_type  : type | None
        Pass ``bool``, ``int``, or ``float`` for automatic conversion.
        ``None`` (default) returns a raw string.

    Returns
    -------
    Converted value, or *fallback* on any error.
    """
    try:
        raw = config.get(section, key, fallback=None)
    except (configparser.NoSectionError, configparser.NoOptionError):
        raw = None

    if raw is None or raw == '':
        return fallback

    raw = raw.strip()

    try:
        if as_type is bool:
            return raw.lower() in ('1', 'true', 'yes', 'on')
        if as_type is int:
            return int(raw)
        if as_type is float:
            return float(raw)
    except (ValueError, TypeError):
        return fallback

    return raw
