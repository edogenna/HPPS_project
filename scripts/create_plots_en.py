import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os # Imported the 'os' module to handle paths and directories

def clean_data(df):
    """Cleans the dataframe by removing spaces from column names and converting MR to float."""
    df.columns = df.columns.str.strip()
    if df['MR'].dtype == 'object':
        df['MR_float'] = df['MR'].str.strip().str.rstrip('%').astype(float)
    else:
        df['MR_float'] = df['MR']
    return df

def plot_performance_by_workload(df, output_dir):
    """Generates and saves a bar chart comparing performance metrics by workload."""
    output_filename = os.path.join(output_dir, "workload_performance.png")
    workload_performance = df.groupby('Workload')[['IPC', 'MPKI', 'MR_float']].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Average Performance by Workload Type', fontsize=16)
    sns.barplot(x='Workload', y='IPC', data=workload_performance, ax=axes[0])
    axes[0].set_title('Average IPC (Instructions Per Cycle)')
    axes[0].set_ylabel('IPC (Higher is better)')
    sns.barplot(x='Workload', y='MPKI', data=workload_performance, ax=axes[1])
    axes[1].set_title('Average MPKI (Mispredictions Per Kilo-Instruction)')
    axes[1].set_ylabel('MPKI (Lower is better)')
    sns.barplot(x='Workload', y='MR_float', data=workload_performance, ax=axes[2])
    axes[2].set_title('Average Miss Rate (%)')
    axes[2].set_ylabel('Miss Rate (%) (Lower is better)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    plt.close()
    print(f"Workload performance plot saved to {output_filename}")

def plot_mpki_per_run(df, output_dir):
    """Generates and saves a horizontal, sorted bar chart of MPKI for each run."""
    output_filename = os.path.join(output_dir, "mpki_per_run.png")
    df_sorted = df.sort_values('MPKI', ascending=False)
    plt.figure(figsize=(12, len(df_sorted) * 0.3))
    sns.barplot(x='MPKI', y='Run', data=df_sorted, hue='Workload', dodge=False)
    plt.title('MPKI per Single Execution (Run)')
    plt.xlabel('MPKI (Lower is better)')
    plt.ylabel('Trace (Run)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"MPKI per run plot saved to {output_filename}")

def plot_ipc_vs_mpki_scatter(df, output_dir):
    """Generates and saves an IPC vs. MPKI scatter plot."""
    output_filename = os.path.join(output_dir, "ipc_vs_mpki.png")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='MPKI', y='IPC', hue='Workload', data=df, s=100, alpha=0.7)
    plt.title('IPC vs. MPKI Correlation')
    plt.xlabel('MPKI (Lower is better)')
    plt.ylabel('IPC (Higher is better)')
    plt.grid(True)
    plt.legend(title='Workload')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"IPC vs. MPKI scatter plot saved to {output_filename}")

def main():
    """Main function to load data and generate all plots."""
    parser = argparse.ArgumentParser(description="Generates performance plots from a branch predictor results CSV file.")
    parser.add_argument("input_file", help="The path to the input CSV file to be analyzed.")
    # --- START OF CHANGE ---
    parser.add_argument("-o", "--output_dir", default="plots", help="The directory where plots will be saved (default: 'plots').")
    # --- END OF CHANGE ---
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        print("Please check that the file name and path are correct.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    df = clean_data(df)

    # Pass the output directory to the plot generation functions
    plot_performance_by_workload(df, output_dir)
    plot_mpki_per_run(df, output_dir)
    plot_ipc_vs_mpki_scatter(df, output_dir)

    print(f"\nAll plots have been successfully generated in the '{output_dir}' directory.")

if __name__ == '__main__':
    main()