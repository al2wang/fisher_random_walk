# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# def analyze_and_plot(results_dir):
#     """
#     Aggregates the simulation results across all seeds for each sample size
#     and generates a summary table and plot.
#     """
#     sample_sizes = [500, 1000, 2000, 5000]
#     summary_data = []

#     print("======================================================================")
#     print(f"{'Sample Size (n)':<16} | {'Est. Error |r_hat - r*|':<22} | {'Empirical Coverage':<18} | {'Avg CI Length'}")
#     print("======================================================================")

#     for n in sample_sizes:
#         file_path = os.path.join(results_dir, f"results_n{n}.csv")
        
#         if not os.path.exists(file_path):
#             print(f"{n:<16} | Data file not found: {file_path}")
#             continue
            
#         # Load the replication data for sample size n
#         df = pd.read_csv(file_path)
        
#         # Calculate the requested metrics
#         mean_error = df['error'].mean() 
#         coverage_pct = df['covered'].mean() * 100 # Convert binary to percentage
#         avg_ci_length = df['ci_length'].mean()
#         num_reps = len(df)
        
#         print(f"{n:<16} | {mean_error:<22.4f} | {coverage_pct:>16.1f}% | {avg_ci_length:.4f}")
        
#         summary_data.append({
#             'n': n,
#             'mean_error': mean_error,
#             'coverage_pct': coverage_pct,
#             'avg_ci_length': avg_ci_length,
#             'replications': num_reps
#         })
        
#     print("======================================================================")

#     # Save summary to CSV
#     summary_df = pd.DataFrame(summary_data)
#     if not summary_df.empty:
#         summary_path = os.path.join(results_dir, "simulation_61_summary_rerun.csv")
#         summary_df.to_csv(summary_path, index=False)
#         print(f"\nSummary metrics saved to: {summary_path}")
        
#         # Generate Visualizations
#         generate_plots(summary_df, results_dir)

# def generate_plots(df, out_dir):
#     """Generates a publication-style 1x2 plot of Error and CI Length."""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#     # Plot 1: Estimation Error
#     ax1.plot(df['n'], df['mean_error'], marker='o', linestyle='-', color='b', markersize=8)
#     ax1.set_xlabel('Sample Size (n)')
#     ax1.set_ylabel('Absolute Estimation Error')
#     ax1.set_title('Reward Estimation Error vs. Sample Size')
#     ax1.grid(True, linestyle='--', alpha=0.6)

#     # Plot 2: Average CI Length
#     ax2.plot(df['n'], df['avg_ci_length'], marker='s', linestyle='-', color='r', markersize=8)
#     ax2.set_xlabel('Sample Size (n)')
#     ax2.set_ylabel('Average CI Length')
#     ax2.set_title('Confidence Interval Length vs. Sample Size')
#     ax2.grid(True, linestyle='--', alpha=0.6)

#     plt.tight_layout()
#     plot_path = os.path.join(out_dir, "results_61_rerun.pdf")
#     plt.savefig(plot_path, dpi=300)
#     print(f"Scaling plots saved to: {plot_path}")

# if __name__ == "__main__":
#     # Ensure this matches the OUT_DIR specified in your Slurm submission script
#     RESULTS_DIRECTORY = "./results_sec_61" 
    
#     if os.path.exists(RESULTS_DIRECTORY):
#         analyze_and_plot(RESULTS_DIRECTORY)
#     else:
#         print(f"Error: Directory '{RESULTS_DIRECTORY}' does not exist.")

import os
import pandas as pd
import matplotlib.pyplot as plt

def read_and_plot(results_dir):
    """
    Reads the pre-aggregated summary CSV and generates the scaling plots.
    """
    summary_filename = "simulation_61_summary_rerun.csv"
    summary_path = os.path.join(results_dir, summary_filename)

    if not os.path.exists(summary_path):
        print(f"Error: Summary data file not found at {summary_path}")
        return

    # Load the summarized data directly
    df = pd.read_csv(summary_path)

    # Print the table to the terminal for easy viewing
    print("======================================================================")
    print(f"{'Sample Size (n)':<16} | {'Est. Error |r_hat - r*|':<22} | {'Empirical Coverage':<18} | {'Avg CI Length'}")
    print("======================================================================")

    for _, row in df.iterrows():
        n = int(row['n'])
        mean_error = row['mean_error']
        coverage_pct = row['coverage_pct']
        avg_ci_length = row['avg_ci_length']
        
        print(f"{n:<16} | {mean_error:<22.4f} | {coverage_pct:>16.1f}% | {avg_ci_length:.4f}")
        
    print("======================================================================")

    # Generate Visualizations
    generate_plots(df, results_dir)

def generate_plots(df, out_dir):
    """Generates a publication-style 1x2 plot of Error and CI Length."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Estimation Error
    ax1.plot(df['n'], df['mean_error'], marker='o', linestyle='-', color='b', markersize=8)
    ax1.set_xlabel('Sample Size (n)')
    ax1.set_ylabel('Absolute Estimation Error')
    ax1.set_title('Reward Estimation Error vs. Sample Size')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Average CI Length
    ax2.plot(df['n'], df['avg_ci_length'], marker='s', linestyle='-', color='r', markersize=8)
    ax2.set_xlabel('Sample Size (n)')
    ax2.set_ylabel('Average CI Length')
    ax2.set_title('Confidence Interval Length vs. Sample Size')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "results_61_rerun.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nScaling plots saved to: {plot_path}")

if __name__ == "__main__":
    RESULTS_DIRECTORY = "./results_sec_61" 
    
    if os.path.exists(RESULTS_DIRECTORY):
        read_and_plot(RESULTS_DIRECTORY)
    else:
        print(f"Error: Directory '{RESULTS_DIRECTORY}' does not exist.")