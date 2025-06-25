import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path # Modulo per gestire i percorsi in modo moderno

def clean_data(df):
    """Pulisce il dataframe rimuovendo spazi dai nomi delle colonne e convertendo MR in float."""
    df.columns = df.columns.str.strip()
    if 'MR' in df.columns and df['MR'].dtype == 'object':
        df['MR_float'] = df['MR'].str.strip().str.rstrip('%').astype(float)
    else:
        df['MR_float'] = df.get('MR', pd.Series(index=df.index, dtype='float64'))
    return df

def load_data_from_directory(input_dir):
    """
    Carica tutti i file CSV da una directory, li combina in un unico DataFrame
    e aggiunge una colonna 'Predictor' basata sul nome del file.
    """
    data_frames = []
    csv_files = list(Path(input_dir).glob('*.csv'))

    if not csv_files:
        print(f"Errore: Nessun file .csv trovato nella directory '{input_dir}'.")
        return None

    print(f"Trovati {len(csv_files)} file CSV da analizzare...")

    for file_path in csv_files:
        predictor_name = file_path.stem # Estrae il nome del file senza estensione
        try:
            df = pd.read_csv(file_path)
            df['Predictor'] = predictor_name
            df = clean_data(df)
            data_frames.append(df)
            print(f"  - Caricato '{file_path.name}' per il predittore '{predictor_name}'")
        except Exception as e:
            print(f"Attenzione: Impossibile leggere o processare il file {file_path.name}. Errore: {e}")

    if not data_frames:
        print("Errore: Nessun file CSV è stato caricato con successo.")
        return None

    return pd.concat(data_frames, ignore_index=True)

def plot_overall_comparison(combined_df, output_dir):
    """
    Crea un grafico a barre che confronta le performance medie generali di tutti i predittori.
    """
    output_filename = os.path.join(output_dir, "comparison_overall_performance.png")
    
    # Calcola le medie per ogni predittore
    overall_avg = combined_df.groupby('Predictor')[['IPC', 'MPKI']].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Confronto Performance Medie Generali dei Predittori', fontsize=16)

    # Grafico IPC
    sns.barplot(x='Predictor', y='IPC', data=overall_avg.sort_values('IPC', ascending=False), ax=axes[0])
    axes[0].set_title('IPC Medio (Più alto è meglio)')
    axes[0].tick_params(axis='x', rotation=45)

    # Grafico MPKI
    sns.barplot(x='Predictor', y='MPKI', data=overall_avg.sort_values('MPKI', ascending=True), ax=axes[1])
    axes[1].set_title('MPKI Medio (Più basso è meglio)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    plt.close()
    print(f"Grafico di confronto generale salvato in {output_filename}")


def plot_workload_comparison(combined_df, output_dir):
    """
    Crea un grafico a barre raggruppato che confronta l'MPKI dei predittori per ogni tipo di workload.
    """
    output_filename = os.path.join(output_dir, "comparison_by_workload.png")
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Workload', y='MPKI', hue='Predictor', data=combined_df)
    
    plt.title('Confronto MPKI per Tipo di Workload')
    plt.ylabel('MPKI (Più basso è meglio)')
    plt.xlabel('Tipo di Workload')
    plt.legend(title='Predictor')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Grafico di confronto per workload salvato in {output_filename}")


def main():
    """Funzione principale per avviare l'analisi comparativa."""
    parser = argparse.ArgumentParser(description="Confronta i risultati di più predittori da file CSV contenuti in una directory.")
    parser.add_argument("input_dir", help="La directory contenente i file CSV dei risultati (un file per predittore).")
    parser.add_argument("-o", "--output_dir", default="comparison_plots", help="La directory dove salvare i grafici di confronto (default: 'comparison_plots').")
    args = parser.parse_args()

    # Crea la directory di output se non esiste
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Carica e combina i dati
    combined_df = load_data_from_directory(args.input_dir)
    
    if combined_df is not None:
        # Genera i grafici di confronto
        plot_overall_comparison(combined_df, args.output_dir)
        plot_workload_comparison(combined_df, args.output_dir)
        print(f"\nAnalisi comparativa completata. Grafici salvati in '{args.output_dir}'.")

if __name__ == '__main__':
    main()