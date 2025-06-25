import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
from pathlib import Path

def clean_data(df):
    """Cleans the dataframe and calculates basic metrics."""
    df.columns = df.columns.str.strip()
    if 'MR' in df.columns and df['MR'].dtype == 'object':
        df['MR_float'] = df['MR'].str.strip().str.rstrip('%').astype(float)
    else:
        df['MR_float'] = df.get('MR', pd.Series(index=df.index, dtype='float64'))
    return df

def load_data_from_directory(input_dir):
    """Loads all CSV files, combines them, and adds a 'Predictor' column."""
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

# --- NEW FUNCTION TO CALCULATE ADVANCED KPIs ---
def calculate_advanced_kpis(df, baseline_predictor='gshare'):
    """Calculates advanced KPIs like IPM and Improvement vs. Baseline."""
    print(f"\nCalculating advanced KPIs using '{baseline_predictor}' as a baseline...")

    # 1. Calculate IPM (Instructions Per Misprediction)
    # A very small value is added to the denominator to avoid division by zero
    df['IPM'] = df['Instr'] / (df['MispBr'] + 1e-9)

    # 2. Calculate MPKI Improvement vs. Baseline
    baseline_df = df[df['Predictor'] == baseline_predictor][['Run', 'MPKI']].rename(columns={'MPKI': 'MPKI_baseline'})
    
    if baseline_df.empty:
        print(f"Warning: Could not find data for the baseline '{baseline_predictor}'. The improvement plot will not be generated.")
        df['MPKI_Improvement_%'] = np.nan
        return df

    # Merge the baseline data with the main dataframe
    df = pd.merge(df, baseline_df, on='Run', how='left')
    
    # Calculate the percentage improvement
    # A very small value is added to the denominator to avoid division by zero
    df['MPKI_Improvement_%'] = ((df['MPKI_baseline'] - df['MPKI']) / (df['MPKI_baseline'] + 1e-9)) * 100
    
    return df

# --- EXISTING PLOTTING FUNCTIONS (UNCHANGED) ---
def plot_overall_comparison(combined_df, output_dir):
    """Creates a bar chart comparing overall average performance."""
    output_filename = os.path.join(output_dir, "comparison_overall_performance.png")
    overall_avg = combined_df.groupby('Predictor')[['IPC', 'MPKI']].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7)); fig.suptitle('Overall Average Performance Comparison of Predictors', fontsize=16)
    sns.barplot(ax=axes[0], x='Predictor', y='IPC', data=overall_avg.sort_values('IPC', ascending=False)); axes[0].set_title('Average IPC (Higher is better)'); axes[0].tick_params(axis='x', rotation=45)
    sns.barplot(ax=axes[1], x='Predictor', y='MPKI', data=overall_avg.sort_values('MPKI', ascending=True)); axes[1].set_title('Average MPKI (Lower is better)'); axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(output_filename); plt.close()
    print(f"Overall comparison plot saved to {output_filename}")


def plot_workload_comparison(combined_df, output_dir):
    """Creates a grouped bar chart comparing MPKI by workload."""
    output_filename = os.path.join(output_dir, "comparison_by_workload.png")
    plt.figure(figsize=(12, 7)); sns.barplot(x='Workload', y='MPKI', hue='Predictor', data=combined_df)
    plt.title('MPKI Comparison by Workload Type'); plt.ylabel('MPKI (Lower is better)'); plt.xlabel('Workload Type'); plt.legend(title='Predictor'); plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(); plt.savefig(output_filename); plt.close()
    print(f"Workload comparison plot saved to {output_filename}")


# --- NEW PLOTTING FUNCTIONS ---
def plot_ipm_comparison(df, output_dir):
    """NEW: Creates a bar chart to compare the average IPM."""
    output_filename = os.path.join(output_dir, "comparison_ipm.png")
    ipm_avg = df.groupby('Predictor')['IPM'].mean().reset_index().sort_values('IPM', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Predictor', y='IPM', data=ipm_avg)
    plt.title('Efficiency: Average Instructions Per Misprediction (IPM)')
    plt.ylabel('IPM (Higher is better)')
    plt.xlabel('Predictor')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"IPM comparison plot saved to {output_filename}")

def plot_improvement_comparison(df, output_dir, baseline_predictor='gshare'):
    """NEW: Creates a bar chart for the % improvement over the baseline."""
    output_filename = os.path.join(output_dir, "comparison_improvement_vs_baseline.png")
    # Excludes the baseline itself from the plot
    improvement_avg = df[df['Predictor'] != baseline_predictor].groupby('Predictor')['MPKI_Improvement_%'].mean().reset_index().sort_values('MPKI_Improvement_%', ascending=False)

    if improvement_avg.empty:
        print(f"Warning: No data for the improvement plot (only the baseline might be present).")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Predictor', y='MPKI_Improvement_%', data=improvement_avg)
    plt.title(f"Relative Performance: MPKI Improvement vs. {baseline_predictor}")
    plt.ylabel(f'Improvement % (Higher is better)')
    plt.xlabel('Predictor')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Improvement vs. baseline plot saved to {output_filename}")


def main():
    """Main function to start the comparative analysis."""
    parser = argparse.ArgumentParser(description="Compares the results of multiple predictors from CSV files in a directory.")
    parser.add_argument("input_dir", help="The directory containing the result CSV files (one per predictor).")
    parser.add_argument("-o", "--output_dir", default="comparison_plots", help="The directory where comparison plots will be saved (default: 'comparison_plots').")
    parser.add_argument("-b", "--baseline", default="gshare", help="The name of the predictor to use as a baseline (default: 'gshare').")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    combined_df = load_data_from_directory(args.input_dir)
    
    if combined_df is not None:
        # Calculate advanced KPIs
        combined_df = calculate_advanced_kpis(combined_df, baseline_predictor=args.baseline)
        
        # Generate plots
        plot_overall_comparison(combined_df, args.output_dir)
        plot_workload_comparison(combined_df, args.output_dir)
        plot_ipm_comparison(combined_df, args.output_dir)
        plot_improvement_comparison(combined_df, args.output_dir, baseline_predictor=args.baseline)
        
        print(f"\nComparative analysis complete. All plots have been saved in '{args.output_dir}'.")

if __name__ == '__main__':
    main()