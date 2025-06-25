import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os # Importato il modulo 'os' per gestire percorsi e directory

def clean_data(df):
    """Pulisce il dataframe rimuovendo spazi dai nomi delle colonne e convertendo MR in float."""
    df.columns = df.columns.str.strip()
    if df['MR'].dtype == 'object':
        df['MR_float'] = df['MR'].str.strip().str.rstrip('%').astype(float)
    else:
        df['MR_float'] = df['MR']
    return df

def plot_performance_by_workload(df, output_dir):
    """Genera e salva un grafico a barre che confronta le metriche di performance per workload."""
    output_filename = os.path.join(output_dir, "workload_performance.png")
    workload_performance = df.groupby('Workload')[['IPC', 'MPKI', 'MR_float']].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Performance Medie per Tipo di Workload', fontsize=16)
    sns.barplot(x='Workload', y='IPC', data=workload_performance, ax=axes[0])
    axes[0].set_title('IPC Medio (Instructions Per Cycle)')
    axes[0].set_ylabel('IPC (Più alto è meglio)')
    sns.barplot(x='Workload', y='MPKI', data=workload_performance, ax=axes[1])
    axes[1].set_title('MPKI Medio (Mispredictions Per Kilo-Instruction)')
    axes[1].set_ylabel('MPKI (Più basso è meglio)')
    sns.barplot(x='Workload', y='MR_float', data=workload_performance, ax=axes[2])
    axes[2].set_title('Tasso di Errore Medio (%)')
    axes[2].set_ylabel('Tasso di Errore (%) (Più basso è meglio)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    plt.close()
    print(f"Grafico delle performance per workload salvato in {output_filename}")

def plot_mpki_per_run(df, output_dir):
    """Genera e salva un grafico a barre orizzontale e ordinato dell'MPKI per ogni run."""
    output_filename = os.path.join(output_dir, "mpki_per_run.png")
    df_sorted = df.sort_values('MPKI', ascending=False)
    plt.figure(figsize=(12, len(df_sorted) * 0.3))
    sns.barplot(x='MPKI', y='Run', data=df_sorted, hue='Workload', dodge=False)
    plt.title('MPKI per Singola Esecuzione (Run)')
    plt.xlabel('MPKI (Più basso è meglio)')
    plt.ylabel('Traccia (Run)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Grafico MPKI per run salvato in {output_filename}")

def plot_ipc_vs_mpki_scatter(df, output_dir):
    """Genera e salva un grafico a dispersione di IPC vs. MPKI."""
    output_filename = os.path.join(output_dir, "ipc_vs_mpki.png")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='MPKI', y='IPC', hue='Workload', data=df, s=100, alpha=0.7)
    plt.title('Correlazione IPC vs. MPKI')
    plt.xlabel('MPKI (Più basso è meglio)')
    plt.ylabel('IPC (Più alto è meglio)')
    plt.grid(True)
    plt.legend(title='Workload')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Grafico a dispersione IPC vs. MPKI salvato in {output_filename}")

def main():
    """Funzione principale per caricare i dati e generare tutti i grafici."""
    parser = argparse.ArgumentParser(description="Genera grafici di performance da un file CSV di risultati del branch predictor.")
    parser.add_argument("input_file", help="Il percorso del file CSV di input da analizzare.")
    # --- MODIFICA INIZIO ---
    parser.add_argument("-o", "--output_dir", default="plots", help="La directory dove salvare i grafici (default: 'plots').")
    # --- MODIFICA FINE ---
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir

    # Crea la directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Errore: Il file '{input_file}' non è stato trovato.")
        print("Verifica che il nome e il percorso del file siano corretti.")
        return
    except Exception as e:
        print(f"Si è verificato un errore durante la lettura del file: {e}")
        return

    df = clean_data(df)

    # Passa la directory di output alle funzioni che generano i grafici
    plot_performance_by_workload(df, output_dir)
    plot_mpki_per_run(df, output_dir)
    plot_ipc_vs_mpki_scatter(df, output_dir)

    print(f"\nTutti i grafici sono stati generati con successo nella directory '{output_dir}'.")

if __name__ == '__main__':
    main()