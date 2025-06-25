import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
from pathlib import Path

def clean_data(df):
    """Pulisce il dataframe e calcola le metriche di base."""
    df.columns = df.columns.str.strip()
    if 'MR' in df.columns and df['MR'].dtype == 'object':
        df['MR_float'] = df['MR'].str.strip().str.rstrip('%').astype(float)
    else:
        df['MR_float'] = df.get('MR', pd.Series(index=df.index, dtype='float64'))
    return df

def load_data_from_directory(input_dir):
    """Carica tutti i file CSV, li combina e aggiunge una colonna 'Predictor'."""
    data_frames = []
    csv_files = list(Path(input_dir).glob('*.csv'))
    if not csv_files:
        print(f"Errore: Nessun file .csv trovato nella directory '{input_dir}'.")
        return None
    print(f"Trovati {len(csv_files)} file CSV da analizzare...")
    for file_path in csv_files:
        predictor_name = file_path.stem
        try:
            df = pd.read_csv(file_path)
            df['Predictor'] = predictor_name
            df = clean_data(df)
            data_frames.append(df)
            print(f"  - Caricato '{file_path.name}' per il predittore '{predictor_name}'")
        except Exception as e:
            print(f"Attenzione: Impossibile processare il file {file_path.name}. Errore: {e}")
    if not data_frames:
        print("Errore: Nessun file è stato caricato con successo.")
        return None
    return pd.concat(data_frames, ignore_index=True)

# --- NUOVA FUNZIONE PER CALCOLARE I KPI AVANZATI ---
def calculate_advanced_kpis(df, baseline_predictor='gshare'):
    """Calcola i KPI avanzati come IPM e Miglioramento vs. Baseline."""
    print(f"\nCalcolo dei KPI avanzati usando '{baseline_predictor}' come baseline...")
    
    # 1. Calcolo IPM (Instructions Per Misprediction)
    # Si aggiunge un valore molto piccolo al denominatore per evitare divisioni per zero
    df['IPM'] = df['Instr'] / (df['MispBr'] + 1e-9)

    # 2. Calcolo Miglioramento MPKI vs. Baseline
    baseline_df = df[df['Predictor'] == baseline_predictor][['Run', 'MPKI']].rename(columns={'MPKI': 'MPKI_baseline'})
    
    if baseline_df.empty:
        print(f"Attenzione: Impossibile trovare i dati per la baseline '{baseline_predictor}'. Il grafico del miglioramento non sarà generato.")
        df['MPKI_Improvement_%'] = np.nan
        return df

    # Unisce i dati della baseline con il dataframe principale
    df = pd.merge(df, baseline_df, on='Run', how='left')
    
    # Calcola il miglioramento percentuale
    # Si aggiunge un valore molto piccolo al denominatore per evitare divisioni per zero
    df['MPKI_Improvement_%'] = ((df['MPKI_baseline'] - df['MPKI']) / (df['MPKI_baseline'] + 1e-9)) * 100
    
    return df

# --- FUNZIONI DI PLOT ESISTENTI (INVARIATE) ---
def plot_overall_comparison(combined_df, output_dir):
    """Crea un grafico a barre che confronta le performance medie generali."""
    # ... (codice invariato)
    output_filename = os.path.join(output_dir, "comparison_overall_performance.png")
    overall_avg = combined_df.groupby('Predictor')[['IPC', 'MPKI']].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7)); fig.suptitle('Confronto Performance Medie Generali dei Predittori', fontsize=16)
    sns.barplot(ax=axes[0], x='Predictor', y='IPC', data=overall_avg.sort_values('IPC', ascending=False)); axes[0].set_title('IPC Medio (Più alto è meglio)'); axes[0].tick_params(axis='x', rotation=45)
    sns.barplot(ax=axes[1], x='Predictor', y='MPKI', data=overall_avg.sort_values('MPKI', ascending=True)); axes[1].set_title('MPKI Medio (Più basso è meglio)'); axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(output_filename); plt.close()
    print(f"Grafico di confronto generale salvato in {output_filename}")


def plot_workload_comparison(combined_df, output_dir):
    """Crea un grafico a barre raggruppato che confronta l'MPKI per workload."""
    # ... (codice invariato)
    output_filename = os.path.join(output_dir, "comparison_by_workload.png")
    plt.figure(figsize=(12, 7)); sns.barplot(x='Workload', y='MPKI', hue='Predictor', data=combined_df)
    plt.title('Confronto MPKI per Tipo di Workload'); plt.ylabel('MPKI (Più basso è meglio)'); plt.xlabel('Tipo di Workload'); plt.legend(title='Predictor'); plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(); plt.savefig(output_filename); plt.close()
    print(f"Grafico di confronto per workload salvato in {output_filename}")


# --- NUOVE FUNZIONI DI PLOT ---
def plot_ipm_comparison(df, output_dir):
    """NOVITÀ: Crea un grafico a barre per confrontare l'IPM medio."""
    output_filename = os.path.join(output_dir, "comparison_ipm.png")
    ipm_avg = df.groupby('Predictor')['IPM'].mean().reset_index().sort_values('IPM', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Predictor', y='IPM', data=ipm_avg)
    plt.title('Efficienza: Istruzioni Medie per Errore (IPM)')
    plt.ylabel('IPM (Più alto è meglio)')
    plt.xlabel('Predittore')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Grafico di confronto IPM salvato in {output_filename}")

def plot_improvement_comparison(df, output_dir, baseline_predictor='gshare'):
    """NOVITÀ: Crea un grafico a barre per il miglioramento % rispetto alla baseline."""
    output_filename = os.path.join(output_dir, "comparison_improvement_vs_baseline.png")
    # Esclude la baseline stessa dal grafico
    improvement_avg = df[df['Predictor'] != baseline_predictor].groupby('Predictor')['MPKI_Improvement_%'].mean().reset_index().sort_values('MPKI_Improvement_%', ascending=False)

    if improvement_avg.empty:
        print(f"Attenzione: Nessun dato per il grafico di miglioramento (potrebbe esserci solo la baseline).")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Predictor', y='MPKI_Improvement_%', data=improvement_avg)
    plt.title(f"Performance Relativa: Miglioramento MPKI vs. {baseline_predictor}")
    plt.ylabel(f'Miglioramento % (Più alto è meglio)')
    plt.xlabel('Predittore')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Grafico del miglioramento vs. baseline salvato in {output_filename}")


def main():
    """Funzione principale per avviare l'analisi comparativa."""
    parser = argparse.ArgumentParser(description="Confronta i risultati di più predittori da file CSV contenuti in una directory.")
    parser.add_argument("input_dir", help="La directory contenente i file CSV dei risultati (uno per predittore).")
    parser.add_argument("-o", "--output_dir", default="comparison_plots", help="La directory dove salvare i grafici di confronto (default: 'comparison_plots').")
    parser.add_argument("-b", "--baseline", default="results_gshare", help="Il nome del predittore da usare come baseline (default: 'gshare').")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    combined_df = load_data_from_directory(args.input_dir)
    
    if combined_df is not None:
        # Calcola i KPI avanzati
        combined_df = calculate_advanced_kpis(combined_df, baseline_predictor=args.baseline)
        
        # Genera i grafici
        plot_overall_comparison(combined_df, args.output_dir)
        plot_workload_comparison(combined_df, args.output_dir)
        plot_ipm_comparison(combined_df, args.output_dir)
        plot_improvement_comparison(combined_df, args.output_dir, baseline_predictor=args.baseline)
        
        print(f"\nAnalisi comparativa completata. Tutti i grafici sono stati salvati in '{args.output_dir}'.")

if __name__ == '__main__':
    main()