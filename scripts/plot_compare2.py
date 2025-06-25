import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

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

def plot_overall_comparison(combined_df, output_dir):
    """Crea un grafico a barre che confronta le performance medie generali."""
    output_filename = os.path.join(output_dir, "comparison_overall_performance.png")
    overall_avg = combined_df.groupby('Predictor')[['IPC', 'MPKI']].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Confronto Performance Medie Generali dei Predittori', fontsize=16)
    sns.barplot(x='Predictor', y='IPC', data=overall_avg.sort_values('IPC', ascending=False), ax=axes[0])
    axes[0].set_title('IPC Medio (Più alto è meglio)')
    axes[0].tick_params(axis='x', rotation=45)
    sns.barplot(x='Predictor', y='MPKI', data=overall_avg.sort_values('MPKI', ascending=True), ax=axes[1])
    axes[1].set_title('MPKI Medio (Più basso è meglio)')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    plt.close()
    print(f"Grafico di confronto generale salvato in {output_filename}")

def plot_workload_comparison(combined_df, output_dir):
    """Crea un grafico a barre raggruppato che confronta l'MPKI per workload."""
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

# --- NUOVI GRAFICI ---

def plot_mpki_distribution(combined_df, output_dir):
    """
    NOVITÀ: Crea un box plot per confrontare la distribuzione e stabilità dell'MPKI per ogni predittore.
    """
    output_filename = os.path.join(output_dir, "comparison_mpki_distribution.png")
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Predictor', y='MPKI', data=combined_df)
    plt.title('Distribuzione MPKI per Predittore (Confronto di Stabilità)')
    plt.ylabel('MPKI (Più basso è meglio, una scatola più piccola è più stabile)')
    plt.xlabel('Predittore')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Grafico a box plot della distribuzione MPKI salvato in {output_filename}")

def plot_win_loss(combined_df, output_dir):
    """
    NOVITÀ: Crea un grafico a barre che mostra quante volte ogni predittore ha ottenuto l'MPKI più basso.
    """
    output_filename = os.path.join(output_dir, "comparison_win_loss.png")
    
    # Trova la riga con l'MPKI minimo per ogni traccia unica (Run) per identificare il "vincitore"
    winners = combined_df.loc[combined_df.groupby('Run')['MPKI'].idxmin()]
    
    # Conta il numero di vittorie per ogni predittore
    win_counts = winners['Predictor'].value_counts()
    
    plt.figure(figsize=(10, 6))
    win_counts.plot(kind='bar', color=sns.color_palette())
    plt.title('Numero di "Vittorie" per Predittore (MPKI più basso per traccia)')
    plt.ylabel('Numero di Tracce Vinte')
    plt.xlabel('Predittore')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Grafico di confronto vittorie/sconfitte salvato in {output_filename}")


def main():
    """Funzione principale per avviare l'analisi comparativa."""
    parser = argparse.ArgumentParser(description="Confronta i risultati di più predittori da file CSV contenuti in una directory.")
    parser.add_argument("input_dir", help="La directory contenente i file CSV dei risultati (uno per predittore).")
    parser.add_argument("-o", "--output_dir", default="comparison_plots", help="La directory dove salvare i grafici di confronto (default: 'comparison_plots').")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    combined_df = load_data_from_directory(args.input_dir)
    
    if combined_df is not None:
        # Grafici esistenti
        plot_overall_comparison(combined_df, args.output_dir)
        plot_workload_comparison(combined_df, args.output_dir)
        
        # Chiamata ai nuovi grafici
        plot_mpki_distribution(combined_df, args.output_dir)
        plot_win_loss(combined_df, args.output_dir)
        
        print(f"\nAnalisi comparativa completata. Tutti i grafici sono stati salvati in '{args.output_dir}'.")

if __name__ == '__main__':
    main()