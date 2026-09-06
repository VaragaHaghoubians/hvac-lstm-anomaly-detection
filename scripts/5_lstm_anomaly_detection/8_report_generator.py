"""
REPORT GENERATOR - Create Word report for LSTM Autoencoder results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Purpose:
    Generate comprehensive Word (.docx) report with:
    1. Executive summary
    2. Model configuration details
    3. Training results
    4. Anomaly detection summary
    5. All visualizations with descriptions
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path
from datetime import datetime


def set_cell_border(cell, **kwargs):
    """
    Set cell border
    Usage:
        set_cell_border(
            cell,
            top={"sz": 12, "val": "single", "color": "#000000"},
            bottom={"sz": 12, "val": "single", "color": "#000000"},
            left={"sz": 12, "val": "single", "color": "#000000"},
            right={"sz": 12, "val": "single", "color": "#000000"},
        )
    """
    tc = cell._element
    tcPr = tc.get_or_add_tcPr()
    
    # Create border elements
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        if edge in kwargs:
            edge_data = kwargs.get(edge)
            edge_el = OxmlElement(f'w:{edge}')
            for key in ["sz", "val", "color", "space", "shadow"]:
                if key in edge_data:
                    edge_el.set(qn(f'w:{key}'), str(edge_data[key]))
            tcBorders.append(edge_el)
    tcPr.append(tcBorders)


def add_formatted_heading(doc, text, level=1):
    """Add a formatted heading to the document"""
    heading = doc.add_heading(text, level=level)
    heading.paragraph_format.space_before = Pt(12)
    heading.paragraph_format.space_after = Pt(6)
    
    # Format text
    for run in heading.runs:
        run.font.color.rgb = RGBColor(0, 51, 102)
        run.font.bold = True
    
    return heading


def add_key_value_table(doc, data_dict, title=None):
    """Add a two-column table with key-value pairs"""
    if title:
        doc.add_paragraph(title, style='Heading 3')
    
    table = doc.add_table(rows=len(data_dict), cols=2)
    table.style = 'Light Grid Accent 1'
    
    for idx, (key, value) in enumerate(data_dict.items()):
        row = table.rows[idx]
        row.cells[0].text = str(key)
        row.cells[1].text = str(value)
        
        # Bold the key
        row.cells[0].paragraphs[0].runs[0].font.bold = True
        row.cells[0].paragraphs[0].runs[0].font.size = Pt(10)
        row.cells[1].paragraphs[0].runs[0].font.size = Pt(10)
    
    doc.add_paragraph()  # Spacing
    return table


def add_plot_with_description(doc, image_path, title, description):
    """Add a plot image with title and description in a formatted layout (Italian format)"""
    
    # Create a single-cell table for nice formatting
    table = doc.add_table(rows=3, cols=1)
    table.style = 'Light Grid Accent 1'
    
    # Row 1: Title (Centered, Bold, Dark Blue - NO DATE)
    cell_title = table.rows[0].cells[0]
    set_cell_border(cell_title, top={}, left={}, right={}, bottom={})
    cell_title.paragraphs[0].clear()
    
    title_para = cell_title.paragraphs[0]
    title_para.text = title
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_format = title_para.runs[0].font
    title_format.bold = True
    title_format.size = Pt(12)
    title_format.color.rgb = RGBColor(31, 71, 136)  # Dark blue #1f4788
    cell_title.vertical_alignment = 1  # Center vertically
    
    # Row 2: Image
    cell_image = table.rows[1].cells[0]
    set_cell_border(cell_image, top={}, left={}, right={}, bottom={})
    cell_image.paragraphs[0].clear()
    
    paragraph = cell_image.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    try:
        if Path(image_path).exists():
            run = paragraph.add_run()
            run.add_picture(str(image_path), width=Inches(6.3))
        else:
            paragraph.add_run(f"[Image not found: {image_path}]")
    except Exception as e:
        print(f"    ✗ Error adding image: {str(e)}")
        paragraph.add_run(f"[Error loading image]")
    
    # Row 3: Description
    cell_desc = table.rows[2].cells[0]
    set_cell_border(cell_desc, top={}, left={}, right={}, bottom={})
    
    # Set cell padding
    tc = cell_desc._element
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for margin_name in ['top', 'left', 'bottom', 'right']:
        node = OxmlElement(f'w:{margin_name}')
        node.set(qn('w:w'), '100')
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)
    
    desc_header = cell_desc.paragraphs[0]
    desc_header.text = "Descrizione:"
    desc_header.paragraph_format.space_after = Pt(3)
    desc_header_format = desc_header.runs[0].font
    desc_header_format.bold = True
    desc_header_format.size = Pt(11)
    
    desc_para = cell_desc.add_paragraph(description)
    desc_para.paragraph_format.line_spacing = 1.15
    desc_para_format = desc_para.runs[0].font
    desc_para_format.size = Pt(10)
    
    # Add spacing
    doc.add_paragraph()


def generate_lstm_report(results, data_dict, processed_data, config, plots_dir):
    """
    Generate comprehensive Word report for LSTM Autoencoder analysis (Italian)
    
    Args:
        results: Dictionary with anomaly detection results
        data_dict: Dictionary with data information
        processed_data: Dictionary with preprocessed data
        config: ConfigParser object
        plots_dir: Path to directory containing plots
        
    Returns:
        Path to saved Word document
    """
    
    print(f"\n{'='*70}")
    print("📝 GENERAZIONE REPORT WORD")
    print(f"{'='*70}\n")
    
    try:
        # Create document
        doc = Document()
        
        # ══════════════════════════════════════════════════════════
        # TITLE PAGE (Title without numbers, Date range at top)
        # ══════════════════════════════════════════════════════════
        title = doc.add_heading('LSTM Autoencoder Anomaly Detection', level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in title.runs:
            run.font.color.rgb = RGBColor(31, 71, 136)  # Dark blue
            run.font.size = Pt(24)
        
        # Date range from config
        season = config.get('global', 'season')
        year = config.get('global', 'year')
        try:
            start_date = config.get('seasonal_presets', f"{season.lower()}_start_{year}")
            end_date = config.get('seasonal_presets', f"{season.lower()}_end_{year}")
            period_text = f"Periodo di analisi: {start_date} - {end_date}"
        except:
            period_text = f"Periodo di analisi: {season} {year}"
        
        period_para = doc.add_paragraph(period_text)
        period_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        period_format = period_para.runs[0].font
        period_format.size = Pt(14)
        period_format.bold = True
        period_format.color.rgb = RGBColor(100, 100, 100)
        
        # Subtitle
        subtitle = doc.add_paragraph('Rilevamento Anomalie Sistema HVAC')
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle_format = subtitle.runs[0].font
        subtitle_format.size = Pt(16)
        subtitle_format.color.rgb = RGBColor(100, 100, 100)
        
        # Metadata
        building_id = config.get('global', 'building_id')
        ahu_unit = config.get('global', 'ahu_unit')
        
        metadata = doc.add_paragraph()
        metadata.alignment = WD_ALIGN_PARAGRAPH.CENTER
        metadata.add_run(f"\n\nEdificio: {building_id} | Unità: {ahu_unit}\n")
        metadata.add_run(f"Stagione: {season} {year}\n")
        metadata.add_run(f"Report Generato: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        
        for run in metadata.runs:
            run.font.size = Pt(11)
        
        doc.add_page_break()
        
        # ══════════════════════════════════════════════════════════
        # EXECUTIVE SUMMARY
        # ══════════════════════════════════════════════════════════
        add_formatted_heading(doc, 'Riepilogo Esecutivo', level=1)
        
        total_anomalies = results['filtered_anomalies'].sum()
        total_test = len(results['test_errors'])
        anomaly_rate = (total_anomalies / total_test) * 100 if total_test > 0 else 0
        
        summary_text = doc.add_paragraph()
        summary_text.add_run(
            f"Questo report presenta i risultati dell'analisi LSTM Autoencoder per il rilevamento di anomalie "
            f"nel sistema HVAC. Il modello è stato addestrato su {len(processed_data['X_train'])} sequenze "
            f"e ha identificato {total_anomalies} periodi anomali ({anomaly_rate:.2f}% dei dati di test).\n\n"
        )
        
        # Threshold label differs by mode (EMA = dynamic, others = fixed scalar)
        _thr_mode = config.get('lstm_autoencoder', 'threshold_mode', fallback='percentile')
        _thr_label = 'Soglia EMA (ancora training)' if _thr_mode == 'ema' else 'Soglia di Ricostruzione'
        summary_data = {
            'Sequenze di Addestramento': f"{len(processed_data['X_train']):,}",
            'Sequenze di Validazione': f"{len(processed_data['X_val']):,}",
            'Sequenze di Test': f"{len(processed_data['X_test']):,}",
            'Anomalie Totali Rilevate': f"{total_anomalies:,}",
            'Tasso di Anomalia': f"{anomaly_rate:.2f}%",
            _thr_label: f"{results['threshold']:.6f}",
        }
        
        # Add severity breakdown - only show if severity classification is enabled
        _severity_enabled = config.getboolean('lstm_autoencoder', 'enable_severity', fallback=True)
        if _severity_enabled:
            summary_data['Anomalie Critiche'] = f"{results['n_critical']}"
            summary_data['Anomalie Moderate'] = f"{results['n_moderate']}"
            summary_data['Anomalie Minori'] = f"{results['n_minor']}"
        else:
            summary_data['Classificazione Gravità'] = 'Disabilitata (enable_severity = false)'
        
        add_key_value_table(doc, summary_data)
        
        doc.add_page_break()
        
        # ══════════════════════════════════════════════════════════
        # MODEL CONFIGURATION
        # ══════════════════════════════════════════════════════════
        add_formatted_heading(doc, '2. Model Configuration', level=1)
        
        model_config = {
            'Architecture': 'LSTM Autoencoder (Encoder-Decoder)',
            'Encoder Units': config.getint('lstm_autoencoder', 'encoder_units', fallback=64),
            'Bottleneck Units': config.getint('lstm_autoencoder', 'bottleneck_units', fallback=16),
            'Decoder Units': config.getint('lstm_autoencoder', 'decoder_units', fallback=64),
            'Sequence Length': config.getint('lstm_autoencoder', 'sequence_length'),
            'Dropout Rate': config.getfloat('lstm_autoencoder', 'dropout', fallback=0.3),
            'Learning Rate': config.getfloat('lstm_autoencoder', 'learning_rate', fallback=0.0005),
            'Batch Size': config.getint('lstm_autoencoder', 'batch_size', fallback=32),
            'Epochs (Max)': config.getint('lstm_autoencoder', 'epochs', fallback=250),
            'Features Used': ', '.join(data_dict['features']),
            'Number of Features': data_dict['n_features'],
        }
        
        add_key_value_table(doc, model_config)
        
        # ══════════════════════════════════════════════════════════
        # ANOMALY DETECTION PARAMETERS
        # ══════════════════════════════════════════════════════════
        add_formatted_heading(doc, 'Configurazione Rilevamento Anomalie', level=1)
        
        # Build threshold config block dynamically based on active mode
        threshold_mode = config.get('lstm_autoencoder', 'threshold_mode', fallback='percentile')
        _sev_enabled_cfg = config.getboolean('lstm_autoencoder', 'enable_severity', fallback=True)
        if threshold_mode == 'ema':
            anomaly_config = {
                'Metodo Soglia': 'EMA dinamica — threshold(t) = EMA_mean + k × EMA_std',
                'EMA Span (passi)': config.getint('lstm_autoencoder', 'ema_span', fallback=20),
                'EMA Sigma Multiplier (k)': (config.getfloat('lstm_autoencoder', f'ema_sigma_multiplier_{config.get("global", "season", fallback="").lower()}', fallback=None)
                                              or config.getfloat('lstm_autoencoder', 'ema_sigma_multiplier', fallback=3.0)),
                'Features Anomale Minime': config.getint('lstm_autoencoder', 'min_anomalous_features', fallback=1),
            }
        else:
            anomaly_config = {
                'Metodo Soglia': 'Basato su percentile dei dati di addestramento',
                'Percentile Soglia': f"{config.getfloat('lstm_autoencoder', 'anomaly_threshold_percentile', fallback=99.5)}%",
                'Features Anomale Minime': config.getint('lstm_autoencoder', 'min_anomalous_features', fallback=1),
            }
        if _sev_enabled_cfg:
            anomaly_config['Moltiplicatore Gravità Critica'] = f"{config.getfloat('lstm_autoencoder', 'critical_severity_multiplier', fallback=2.5)}x"
            anomaly_config['Moltiplicatore Gravità Moderata'] = f"{config.getfloat('lstm_autoencoder', 'moderate_severity_multiplier', fallback=1.5)}x"
        
        add_key_value_table(doc, anomaly_config)
        
        # ══════════════════════════════════════════════════════════
        # THRESHOLD INTERPRETATION (NEW SECTION)
        # ══════════════════════════════════════════════════════════
        add_formatted_heading(doc, 'Interpretazione della Soglia di Anomalia', level=2)
        
        # Get threshold info from results
        threshold_info = results.get('threshold_info', {})
        expected_fp_rate = threshold_info.get('expected_fp_rate', None)

        interp_text = doc.add_paragraph()

        if threshold_mode == 'ema':
            # ── EMA dynamic threshold interpretation ──────────────────────
            span = threshold_info.get('span', config.getint('lstm_autoencoder', 'ema_span', fallback=20))
            k    = threshold_info.get('ema_sigma_multiplier',
                                      config.getfloat('lstm_autoencoder', 'ema_sigma_multiplier', fallback=3.0))
            anchor = results.get('threshold', 0.0)
            interp_text.add_run(
                f"La soglia di anomalia è calcolata in modo DINAMICO tramite media mobile esponenziale (EMA) "
                f"degli errori di ricostruzione sul set di test. Formula per ogni passo t:\n\n"
                f"    threshold(t) = EMA_media(errori, span={span}) + {k} × EMA_dev.std(errori, span={span})\n\n"
                f"L'ancora di addestramento (valore EMA al termine del training) è: {anchor:.6f}.\n\n"
            )
            interp_text.add_run(
                f"• Vantaggi dell'EMA rispetto al percentile fisso:\n"
                f"  – Si adatta ai lenti cambi di regime (es. calo delle temperature autunnali) "
                f"senza aumentare il tasso di falsi positivi.\n"
                f"  – Evidenzia picchi improvvisi sopra il baseline locale come potenziali anomalie.\n"
                f"  – Nessun parametro da ritarare per stagione o anno.\n\n"
                f"• ⚠️ Limite importante — rischio 'boiling frog':\n"
                f"  – Se una anomalia si sviluppa LENTAMENTE (es. possibile degrado motore su settimane), l'EMA\n"
                f"    sale insieme all'errore → la soglia dinamica segue il drift → possibile assenza di allarme\n"
                f"    fino a deviazioni più marcate. Per anomalie graduali usare threshold_mode = percentile.\n"
                f"  – L'EMA è più adatta per anomalie improvvise/rapide rispetto a degradazioni meccaniche lente.\n\n"
                f"• Parametri chiave:\n"
                f"  – span={span} passi ≈ {span * 0.5:.0f} ore (risoluzione 30 min) — finestra di memoria EMA.\n"
                f"  – k={k} — soglia = EMA_media + {k}×EMA_std.\n"
            )
        else:
            # ── Static percentile interpretation ──────────────────────────
            interp_text.add_run(
                f"La soglia di anomalia ({results['threshold']:.6f}) è calcolata dal "
                f"{threshold_info.get('percentile', 99.5)}° percentile "
                f"degli errori di ricostruzione sul set di addestramento (solo dati normali). "
                f"Questo approccio basato sui quantili ha implicazioni statistiche importanti "
                f"per l'interpretazione dei risultati:\n\n"
            )
            if expected_fp_rate is not None:
                interp_text.add_run(
                    f"• Tasso di falsi positivi atteso: ~{expected_fp_rate:.1f}% dei campioni normali eccedono la soglia per definizione.\n"
                    f"• Questo non è un difetto, ma una caratteristica intrinseca del metodo percentile.\n"
                    f"• La soglia cattura l'errore di ricostruzione \"tipico\" e i pattern normali rari possono eccederla.\n"
                    f"• Compromesso: soglia più stretta (percentile più alto) = maggiore sensibilità MA più falsi allarmi.\n\n"
                )
            interp_text.add_run(
                "Per tesi/pubblicazioni: \"Ci aspettiamo che circa {:.1f}% dei campioni normali eccedano la soglia "
                "per costruzione. Questo tasso controllato di falsi positivi è intrinseco all'approccio di soglia "
                "basato sui quantili e rappresenta la sensibilità del modello a pattern operativi rari ma normali.\"\n"
                .format(expected_fp_rate if expected_fp_rate else 100 - threshold_info.get('percentile', 99.5))
            )
        
        for run in interp_text.runs:
            run.font.size = Pt(10)
        
        # ══════════════════════════════════════════════════════════
        # ANOMALY RATE COMPARISON (NEW SECTION)
        # ══════════════════════════════════════════════════════════
        add_formatted_heading(doc, 'Confronto Tassi di Anomalia', level=2)
        
        comparison_text = doc.add_paragraph()
        comparison_text.add_run(
            "Il confronto dei tassi di anomalia tra i diversi set fornisce informazioni sulla generalizzazione "
            "del modello e sulla qualità operativa dei diversi periodi:\n\n"
        )
        
        for run in comparison_text.runs:
            run.font.size = Pt(10)
        
        # Create comparison table
        # Note: test_anomaly_rate is pre-persistence-filter; anomaly_rate (summary) is post-filter
        _min_consec = config.getint('lstm_autoencoder', 'min_consecutive_anomaly_windows', fallback=1)
        rate_data = {
            'Set Training (grezzo)': f"{results['train_anomaly_rate']:.2f}%",
            'Set Validazione (grezzo)': f"{results['val_anomaly_rate']:.2f}%",
            'Set Test (grezzo, pre-filtro)': f"{results['test_anomaly_rate']:.2f}%",
            f'Set Test (post-filtro ≥{_min_consec} finestre consecutive)': f"{anomaly_rate:.2f}%",
            'Anomalie Eccedenti FP Atteso': f"{max(0, results['test_anomaly_rate'] - expected_fp_rate):.2f}%" if expected_fp_rate else "N/A"
        }
        
        add_key_value_table(doc, rate_data)
        
        # Add interpretation
        interp_comparison = doc.add_paragraph()
        
        test_rate = results['test_anomaly_rate']
        train_rate = results['train_anomaly_rate']
        
        if train_rate == 0:
            if test_rate > 0:
                interp_comparison.add_run(
                    f"⚠️ Il tasso di anomalia nel test ({test_rate:.2f}%) mentre il training aveva 0%. "
                    f"Ciò suggerisce che il periodo di test potrebbe contenere condizioni anomale non viste durante l'addestramento.\n"
                )
        elif test_rate > 2 * train_rate:
            interp_comparison.add_run(
                f"⚠️ Il tasso di anomalia nel test ({test_rate:.2f}%) è {test_rate / train_rate:.1f}× più alto "
                f"rispetto al training ({train_rate:.2f}%). Ciò suggerisce che il periodo di test potrebbe contenere più "
                f"anomalie operative o condizioni fuori distribuzione rispetto al periodo di addestramento.\n"
            )
        elif test_rate < 0.5 * train_rate:
            interp_comparison.add_run(
                f"✅ Il tasso di anomalia nel test ({test_rate:.2f}%) è inferiore al training ({train_rate:.2f}%), "
                f"compatibile con una buona generalizzazione del modello e un periodo di test operativamente più stabile.\n"
            )
        else:
            interp_comparison.add_run(
                f"Il tasso di anomalia nel test ({test_rate:.2f}%) è simile al training ({train_rate:.2f}%), "
                f"compatibile con una buona generalizzazione e coerenza operativa tra i periodi.\n"
            )

        # Validation rate note — flag if notably higher than training
        val_rate = results['val_anomaly_rate']
        if val_rate > 1.5 * train_rate and train_rate > 0:
            interp_comparison.add_run(
                f"\n⚠️ Nota: il tasso di validazione ({val_rate:.2f}%) è {val_rate / train_rate:.1f}× più alto "
                f"del training ({train_rate:.2f}%). La validazione copre il periodo di spalla stagionale "
                f"(temperature in calo, regime diverso dal training) — questo gonfia l'anomaly rate della "
                f"validazione e non riflette necessariamente errori del modello.\n"
            )

        # Error magnitude note — flag distribution shift if test mean >> training mean
        _train_mean = results['train_errors'].mean()
        _test_mean  = results['test_errors'].mean()
        _err_ratio  = _test_mean / _train_mean if _train_mean > 0 else 1
        if _err_ratio > 2.5:
            interp_comparison.add_run(
                f"\n⚠️ L'errore medio di ricostruzione nel test (MAE={_test_mean:.4f}) è {_err_ratio:.1f}× "
                f"superiore al training (MAE={_train_mean:.4f}). Questo è un indicatore di shift di distribuzione: "
                f"il modello ha imparato i pattern del periodo di addestramento ma trova il periodo di test "
                f"statisticamente diverso (es. cambio stagionale). Interpretare il tasso di anomalia con cautela.\n"
            )

        for run in interp_comparison.runs:
            run.font.size = Pt(10)
        
        doc.add_page_break()
        
        # ══════════════════════════════════════════════════════════
        # ANOMALY EPISODE TABLE  (Fix 3)
        # ══════════════════════════════════════════════════════════
        es = results.get('episode_stats', {})
        if es and es.get('n_episodes', 0) > 0:
            add_formatted_heading(doc, 'Episodi Anomali Rilevati', level=2)

            ep_intro = doc.add_paragraph()
            ep_intro.add_run(
                f"Il tasso di anomalia puntuale ({anomaly_rate:.2f}%) descrive quante finestre sono state "
                f"flaggate, ma non dice quanti eventi distinti si sono verificati né quanto sono durati. "
                f"La tabella seguente raggruppa le finestre anomale consecutive in {es['n_episodes']} "
                f"episodi anomali distinti (ipotesi da verificare sui log).\n\n"
                f"Durata media: {es['mean_duration_steps']} finestre "
                f"({es['mean_duration_h']} h)  |  "
                f"Episodio più lungo: {es['max_duration_steps']} finestre "
                f"({es['max_duration_h']} h)"
            )
            for run in ep_intro.runs:
                run.font.size = Pt(10)

            # Build episode table: #  |  Inizio  |  Fine  |  Finestre  |  Durata (h)
            ep_list = es.get('episodes', [])
            # Infer step_h from the first episode if possible
            _step_h = (es['mean_duration_h'] / es['mean_duration_steps']
                       if es['mean_duration_steps'] > 0 else 0.5)

            ep_table = doc.add_table(rows=1 + len(ep_list), cols=5)
            ep_table.style = 'Light Grid Accent 1'

            # Header row
            hdr_cells = ep_table.rows[0].cells
            for col_idx, hdr_text in enumerate(
                ['#', 'Inizio', 'Fine', 'Finestre', 'Durata (h)']
            ):
                hdr_cells[col_idx].text = hdr_text
                run = hdr_cells[col_idx].paragraphs[0].runs[0]
                run.font.bold = True
                run.font.size = Pt(10)

            # Data rows
            for row_idx, (start_s, end_s, n_win) in enumerate(ep_list):
                row_cells = ep_table.rows[row_idx + 1].cells
                row_cells[0].text = str(row_idx + 1)
                row_cells[1].text = str(start_s)
                row_cells[2].text = str(end_s)
                row_cells[3].text = str(n_win)
                row_cells[4].text = f"{round(n_win * _step_h, 2)}"
                for cell in row_cells:
                    cell.paragraphs[0].runs[0].font.size = Pt(9)

            doc.add_paragraph()  # spacing after table

        # ══════════════════════════════════════════════════════════
        # PERFORMANCE METRICS
        # ══════════════════════════════════════════════════════════
        add_formatted_heading(doc, 'Metriche di Prestazione del Modello', level=1)
        
        performance_data = {
            'Errore Medio Set Training': f"{results['train_errors'].mean():.6f}",
            'Deviazione Standard Set Training': f"{results['train_errors'].std():.6f}",
            'Errore Medio Set Validazione': f"{results['val_errors'].mean():.6f}",
            'Deviazione Standard Set Validazione': f"{results['val_errors'].std():.6f}",
            'Errore Medio Set Test': f"{results['test_errors'].mean():.6f}",
            'Deviazione Standard Set Test': f"{results['test_errors'].std():.6f}",
            'Soglia Anomalia': f"{results['threshold']:.6f}",
        }
        
        add_key_value_table(doc, performance_data)
        
        doc.add_page_break()
        
        # ══════════════════════════════════════════════════════════
        # VISUALIZATIONS
        # ══════════════════════════════════════════════════════════
        add_formatted_heading(doc, 'Visualizzazioni dell\'Analisi', level=1)
        
        # Define plots with descriptions (in Italian) - ALL PLOTS INCLUDING FEATURE METRICS AND TIME SERIES
        plots_info = [
            {
                'filename': 'training_history.png',
                'title': 'Storico Addestramento',
                'description': (
                    'Curve di loss per addestramento e validazione durante le epoche. Il modello ha convergenza '
                    'con successo come indicato dai valori di loss decrescenti. L\'epoca migliore è marcata con una '
                    'linea verticale, mostrando quando il modello ha raggiunto prestazioni ottimali sul set di validazione.'
                )
            },
            {
                'filename': 'reconstruction_error_distribution.png',
                'title': 'Distribuzione Errore di Ricostruzione',
                'description': (
                    'Distribuzione degli errori di ricostruzione per i dataset di training e test. La linea tratteggiata '
                    'indica la soglia di anomalia calcolata dai dati di addestramento. Sequenze con errore di ricostruzione '
                    'sopra questa soglia sono classificate come anomalie. La separazione tra pattern normali e anomali è '
                    'generalmente osservabile, ma non costituisce da sola validazione diagnostica.'
                )
            },
            {
                'filename': 'anomalies_timeline.png',
                'title': 'Timeline delle Anomalie',
                'description': (
                    'Visualizzazione serie temporali con anomalie rilevate evidenziate. '
                    + ('Il pannello superiore mostra l\'errore di ricostruzione (blu), la soglia EMA dinamica (nero tratteggiato) '
                       'e l\'EMA degli errori (rosso). I punti rossi indicano le finestre anomale rilevate. '
                       'Il pannello inferiore evidenzia i periodi anomali (rosa) per facilitare l\'identificazione temporale.'
                       if threshold_mode == 'ema' else
                       'Il pannello superiore mostra l\'errore di ricostruzione nel tempo con soglia fissa (linea tratteggiata). '
                       'Il pannello inferiore evidenzia i periodi anomali per facilitare l\'identificazione.')
                )
            },
            {
                'filename': 'feature_error_metrics.png',
                'title': 'Metriche di Errore per Feature',
                'description': (
                    f'Metriche di errore (RMSE, MSE, MAE) nelle unita fisiche delle feature per ciascuna delle {len(data_dict["features"])} features. '
                    'RMSE penalizza errori grandi, MSE mostra la varianza, MAE è robusto agli outlier. '
                    'Valori bassi indicano buona ricostruzione, valori alti indicano difficoltà del modello.'
                )
            }
        ]
        
        # Add time series comparison for each actual feature (dynamic - matches config)
        feature_descriptions = [
            (feat, f'Confronto tra serie temporale reale (blu) e ricostruita (rosso tratteggiato) per {feat}.')
            for feat in data_dict['features']
        ]
        
        for feat_idx, (feature_name, description) in enumerate(feature_descriptions):
            safe_name = feature_name.replace(' ', '_').replace('/', '_')[:50]
            filename = f"time_series_comparison_{feat_idx+1:02d}_{safe_name}.png"
            plots_info.append({
                'filename': filename,
                'title': f'Confronto Serie Temporale: {feature_name}',
                'description': description
            })
        
        # Add each plot
        for plot_info in plots_info:
            image_path = plots_dir / plot_info['filename']
            if image_path.exists():
                add_plot_with_description(
                    doc,
                    image_path,
                    plot_info['title'],
                    plot_info['description']
                )
            else:
                print(f"⚠️  Warning: Plot not found: {plot_info['filename']}")
        
        # ══════════════════════════════════════════════════════════
        # CONCLUSIONS AND RECOMMENDATIONS
        # ══════════════════════════════════════════════════════════
        doc.add_page_break()
        add_formatted_heading(doc, 'Conclusioni e Raccomandazioni', level=1)
        
        conclusions = doc.add_paragraph()
        conclusions.add_run('Risultati Principali:\n').font.bold = True
        _severity_enabled = config.getboolean('lstm_autoencoder', 'enable_severity', fallback=True)
        # Check if test error >> training error (distribution shift indicator)
        _train_mean = results['train_errors'].mean()
        _test_mean  = results['test_errors'].mean()
        _ratio = _test_mean / _train_mean if _train_mean > 0 else 1
        severity_line = (
            f'• Le anomalie sono state classificate in tre livelli di gravità (Critica/Moderata/Minore) '
            f'per dare priorità agli interventi.\n'
            if _severity_enabled else
            '• La classificazione per gravità è disabilitata (enable_severity = false): '
            'tutte le anomalie rilevate sono equivalenti nel report.\n'
        )
        shift_line = (
            f'• ⚠️ Divario errore training/test: MAE_train={_train_mean:.4f}, MAE_test={_test_mean:.4f} '
            f'(rapporto {_ratio:.1f}×) — indica probabile shift di distribuzione (es. stagione di spalla). '
            f'I risultati del test vanno interpretati con cautela.\n'
            if _ratio > 2.5 else ''
        )
        validity_line = (
            '• La qualita della ricostruzione (MAE/RMSE) misura coerenza con i pattern appresi, '
            'ma non prova direttamente la validita diagnostica del rilevamento guasti senza conferma da log/eventi.\n'
        )
        feature_list = data_dict.get('features', [])
        n_features = len(feature_list)
        feature_summary = ', '.join(feature_list) if feature_list else 'feature interne selezionate'
        conclusions.add_run(
            f'• L\'LSTM Autoencoder ha identificato {total_anomalies} sequenze anomale '
            f'({anomaly_rate:.2f}% dei dati di test, dopo filtro di persistenza).\n'
            f'{severity_line}'
            f'{shift_line}'
            f'{validity_line}'
            f'• Il modello si basa su {n_features} segnali interni ({feature_summary}), escludendo '
            f'variabili esogene (meteo) e ridondanti (delta_t) per ridurre i falsi positivi.\n\n'
        )

        conclusions.add_run('Raccomandazioni:\n').font.bold = True
        first_rec = (
            '• Investigare per prime le anomalie critiche in quanto possono indicare deviazioni significative dall\'operazione normale.\n'
            if _severity_enabled else
            '• Rivedere i periodi anomali nel CSV dettagliato, filtrando per errore di ricostruzione più alto.\n'
        )
        conclusions.add_run(
            f'{first_rec}'
            '• Rivedere il file CSV dei risultati dettagliati per timestamp esatti e features affette.\n'
            '• Correlare le anomalie rilevate con i log di manutenzione o eventi di sistema noti.\n'
            '• Verificare le anomalie notturne/mattutine: pattern TR dominante può indicare drift termico dell\'edificio.\n'
            '• Utilizzare questi risultati per stabilire programmi di manutenzione predittiva.\n'
            '• Riaddestrare periodicamente il modello con dati aggiornati per mantenere l\'accuratezza di rilevamento.\n'
        )
        
        # ══════════════════════════════════════════════════════════
        # SAVE REPORT
        # ══════════════════════════════════════════════════════════
        folder_name = f"{building_id}_{ahu_unit}_{season}{year}"
        report_filename = f"LSTM_Autoencoder_Report_{folder_name}.docx"
        report_path = plots_dir / report_filename
        
        doc.save(str(report_path))
        
        print(f"✅ Word report saved: {report_filename}")
        print(f"   Path: {report_path}")
        print(f"{'='*70}\n")
        
        return report_path
        
    except Exception as e:
        print(f"\n❌ Error generating Word report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ═════════════════════════════════════════════════════════════════════════════
# TEST SCRIPT - loads saved model, runs evaluator, generates Word report
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import configparser
    import importlib
    import sys
    from pathlib import Path
    from tensorflow.keras.models import load_model

    print("\n*** TESTING REPORT_GENERATOR.PY (standalone) ***\n")

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
    print(f"Data preprocessed  (X_test: {processed_data['X_test'].shape})")

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
    print(f"Model loaded: {model_path}")

    # ── 4. Detect anomalies ────────────────────────────────────────────────
    results = evaluator.detect_anomalies(model, processed_data, config)

    # ── 5. Build plots_dir and generate report ─────────────────────────────
    building_id = config.get('global', 'building_id')
    ahu_unit    = config.get('global', 'ahu_unit')
    season      = config.get('global', 'season')
    year        = config.get('global', 'year')
    folder_name = f"{building_id}_{ahu_unit}_{season}{year}"
    plots_dir   = (
        Path(__file__).parent.parent.parent
        / base_folder.replace('./', '')
        / 'lstm_autoencoder' / folder_name
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"plots_dir: {plots_dir}")

    report_path = generate_lstm_report(results, data_dict, processed_data, config, plots_dir)

    if report_path:
        print(f"\n*** REPORT GENERATED SUCCESSFULLY ***")
        print(f"    {report_path}\n")
    else:
        print("\n*** REPORT GENERATION FAILED — see traceback above ***\n")
