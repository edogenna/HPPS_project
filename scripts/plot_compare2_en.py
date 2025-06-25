import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

def clean_data(df):
    """Cleans the dataframe by removing spaces from column names and converting MR to float."""
    df.columns = df.columns.str.strip()
    if 'MR' in df.columns and df['MR'].dtype == 'object':
        df['MR_float'] = df['MR'].str.strip().str.rstrip('%').astype(float)
    else:
        df['MR_float'] = df.get('MR', pd.Series(index=df.index, dtype='float64'))
    return df

def load_data_from_directory(input_dir):
    """
    Loads all CSV files from a directory, combines them into a single DataFrame,
    and adds a 'Predictor' column based on the filename.
    """
    data_frames = []
    csv_files = list(Path(input_dir).glob('*.csv'))
    if not csv_files:
        print(f"Error: No .csv files found in the directory '{input_dir}'.")
        return None
    print(f"Found {len(csv_files)} CSV files to analyze...")
    for file_path in csv_files:
        predictor_name = file_path.stem
        try:
            df = pd.read_csv(file_path)
            df['Predictor'] = predictor_name
            df = clean_data(df)
            data_frames.append(df)
            print(f"  - Loaded '{file_path.name}' for predictor '{predictor_name}'")
        except Exception as e:
            print(f"Warning: Could not process file {file_path.name}. Error: {e}")
    if not data_frames:
        print("Error: No files were loaded successfully.")
        return None
    return pd.concat(data_frames, ignore_index=True)

def plot_overall_comparison(combined_df, output_dir):
    """Creates a bar chart comparing overall average performance."""
    output_filename = os.path.join(output_dir, "comparison_overall_performance.png")
    overall_avg = combined_df.groupby('Predictor')[['IPC', 'MPKI']].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Overall Average Performance Comparison of Predictors', fontsize=16)
    sns.barplot(x='Predictor', y='IPC', data=overall_avg.sort_values('IPC', ascending=False), ax=axes[0])
    axes[0].set_title('Average IPC')
    axes[0].tick_params(axis='x', rotation=45)
    sns.barplot(x='Predictor', y='MPKI', data=overall_avg.sort_values('MPKI', ascending=True), ax=axes[1])
    axes[1].set_title('Average MPKI')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    plt.close()
    print(f"Overall comparison plot saved to {output_filename}")

def plot_workload_comparison(combined_df, output_dir):
    """Creates a grouped bar chart comparing MPKI by workload."""
    output_filename = os.path.join(output_dir, "comparison_by_workload.png")
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Workload', y='MPKI', hue='Predictor', data=combined_df)
    plt.title('MPKI Comparison by Workload Type')
    plt.ylabel('MPKI')
    plt.xlabel('Workload Type')
    plt.legend(title='Predictor')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Workload comparison plot saved to {output_filename}")

# --- NEW PLOTS ---

def plot_mpki_distribution(combined_df, output_dir):
    """
    NEW: Creates a box plot to compare the MPKI distribution and stability for each predictor.
    """
    output_filename = os.path.join(output_dir, "comparison_mpki_distribution.png")
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Predictor', y='MPKI', data=combined_df)
    plt.title('MPKI Distribution per Predictor (Stability Comparison)')
    plt.ylabel('MPKI')
    plt.xlabel('Predictor')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"MPKI distribution box plot saved to {output_filename}")

def plot_win_loss(combined_df, output_dir):
    """
    NEW: Creates a bar chart showing how many times each predictor achieved the lowest MPKI.
    """
    output_filename = os.path.join(output_dir, "comparison_win_loss.png")
    
    # Find the row with the minimum MPKI for each unique trace (Run) to identify the "winner"
    winners = combined_df.loc[combined_df.groupby('Run')['MPKI'].idxmin()]
    
    # Count the number of wins for each predictor
    win_counts = winners['Predictor'].value_counts()
    
    plt.figure(figsize=(10, 6))
    win_counts.plot(kind='bar', color=sns.color_palette())
    plt.title('Number of "Wins" per Predictor (Lowest MPKI per Trace)')
    plt.ylabel('Number of Traces Won')
    plt.xlabel('Predictor')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Win/loss comparison plot saved to {output_filename}")


def main():
    """Main function to start the comparative analysis."""
    parser = argparse.ArgumentParser(description="Compares the results of multiple predictors from CSV files in a directory.")
    parser.add_argument("input_dir", help="The directory containing the result CSV files (one per predictor).")
    parser.add_argument("-o", "--output_dir", default="comparison_plots", help="The directory where comparison plots will be saved (default: 'comparison_plots').")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    combined_df = load_data_from_directory(args.input_dir)
    
    if combined_df is not None:
        # Existing plots
        plot_overall_comparison(combined_df, args.output_dir)
        plot_workload_comparison(combined_df, args.output_dir)
        
        # Call the new plotting functions
        plot_mpki_distribution(combined_df, args.output_dir)
        plot_win_loss(combined_df, args.output_dir)
        
        print(f"\nComparative analysis complete. All plots have been saved in '{args.output_dir}'.")

if __name__ == '__main__':
    main()