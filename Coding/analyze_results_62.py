import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_plot_62(results_dir):
    """Aggregates Sec 6.2 results and generates the requested Fenchel Duality plots."""
    
    all_files = glob.glob(os.path.join(results_dir, "results_62_n*_s*.csv"))
    if not all_files:
        print(f"No CSV files found in {results_dir}.")
        return

    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # We will generate one figure per sample size (n) to visualize how the 
    # regularizers behave as n changes, or we can aggregate over n. 
    # Usually, papers plot this for a fixed large n (e.g., n=5000).
    target_n = df['n'].max()
    df_n = df[df['n'] == target_n]

    # Aggregate metrics across seeds
    agg_df = df_n.groupby(['r_param', 'divergence']).agg({
        'error': 'mean',
        'covered': lambda x: x.mean() * 100, # Convert to percentage
        'ci_length': 'mean'
    }).reset_index()

    # Create the 1x3 Panel Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    styles = {
        'kl': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'chi2': {'color': 'red', 'marker': 's', 'linestyle': '--'},
        'tsallis-5': {'color': 'green', 'marker': '^', 'linestyle': '-.'},
        'tsallis-10': {'color': 'purple', 'marker': 'D', 'linestyle': ':'}
    }

    divs = agg_df['divergence'].unique()
    
    for div in divs:
        sub_df = agg_df[agg_df['divergence'] == div]
        s = styles.get(div, {'color': 'black', 'marker': 'x'})
        
        # Panel 1: Error
        axes[0].plot(sub_df['r_param'], sub_df['error'], label=div.upper(), **s)
        # Panel 2: Coverage
        axes[1].plot(sub_df['r_param'], sub_df['covered'], label=div.upper(), **s)
        # Panel 3: Length
        axes[2].plot(sub_df['r_param'], sub_df['ci_length'], label=div.upper(), **s)

    # Formatting Panel 1
    axes[0].set_title(f'Estimation Error (n={target_n})')
    axes[0].set_xlabel('Regularization Exponent (r)')
    axes[0].set_ylabel('Absolute Error |r_hat_d - r*|')
    axes[0].grid(True, alpha=0.5)

    # Formatting Panel 2
    axes[1].set_title('Empirical Coverage')
    axes[1].set_xlabel('Regularization Exponent (r)')
    axes[1].set_ylabel('Coverage (%)')
    axes[1].axhline(y=95.0, color='black', linestyle='-', alpha=0.3, label='95% Target')
    axes[1].grid(True, alpha=0.5)

    # Formatting Panel 3
    axes[2].set_title('Average CI Length')
    axes[2].set_xlabel('Regularization Exponent (r)')
    axes[2].set_ylabel('Length')
    axes[2].grid(True, alpha=0.5)

    # Global Legend
    axes[0].legend(loc='upper left')

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"policy_optimization_n{target_n}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Policy Optimization plots saved to: {plot_path}")

if __name__ == "__main__":
    RESULTS_DIRECTORY = "./results_sec_62" 
    if os.path.exists(RESULTS_DIRECTORY):
        analyze_and_plot_62(RESULTS_DIRECTORY)