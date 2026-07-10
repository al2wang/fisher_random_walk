# import argparse
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# def setup_neurips_style():
#     """Configures matplotlib for clean, academic aesthetic."""
#     sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
#     plt.rcParams.update({
#         "font.family": "serif",
#         "axes.titlesize": 16,
#         "axes.labelsize": 14,
#         "xtick.labelsize": 12,
#         "ytick.labelsize": 12,
#         "legend.fontsize": 12,
#         "figure.dpi": 300,
#         "savefig.dpi": 300,
#         "savefig.bbox": "tight"
#     })

# def plot_reversal_scatter(df, out_dir):
#     """Plots Raw vs Debiased margins, highlighting reversals."""
#     plt.figure(figsize=(8, 6))
    
#     # Define reversals: LLM thinks a1 > a2 (raw > 0), but debiased says a2 > a1 (debiased < 0)
#     reversals = df[(df['raw_delta'] > 0) & (df['debiased_delta'] < 0)]
#     agreements = df[(df['raw_delta'] > 0) & (df['debiased_delta'] > 0)]
    
#     plt.scatter(agreements['raw_delta'], agreements['debiased_delta'], 
#                 alpha=0.6, color='blue', label='Consistent Preference', edgecolor='w')
#     plt.scatter(reversals['raw_delta'], reversals['debiased_delta'], 
#                 alpha=0.8, color='crimson', marker='X', s=80, label='Chandelier Reversal', edgecolor='w')
    
#     plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    
#     plt.xlabel(r'Raw DPO Margin ($\Delta^*$)')
#     plt.ylabel(r'Debiased Margin ($\hat{\Delta}$)')
#     plt.title('Chandelier Debiasing: The Reversal Effect')
#     plt.legend(loc='upper left')
    
#     out_path = os.path.join(out_dir, "reversal_scatter.pdf")
#     plt.savefig(out_path)
#     plt.close()
#     print(f"Saved: {out_path}")

# def plot_margin_densities(df, out_dir):
#     """Plots KDE distributions of raw vs debiased margins on separate axes to fix scale issues."""
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
#     # Left subplot: Raw Margins
#     sns.kdeplot(df['raw_delta'], fill=True, color='gray', ax=axes[0])
#     axes[0].set_title('Raw DPO Margin Density')
#     axes[0].set_xlabel(r'Raw Margin ($\Delta^*$)')
#     axes[0].set_ylabel('Density')
    
#     # Right subplot: Debiased Margins
#     sns.kdeplot(df['debiased_delta'], fill=True, color='purple', ax=axes[1])
#     axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
#     axes[1].set_title('Debiased Margin Density')
#     axes[1].set_xlabel(r'Debiased Margin ($\hat{\Delta}$)')
#     axes[1].set_ylabel('')
    
#     plt.tight_layout()
#     out_path = os.path.join(out_dir, "margin_densities.pdf")
#     plt.savefig(out_path)
#     plt.close()
#     print(f"Saved: {out_path}")

# def plot_pvalue_histogram(df, out_dir):
#     """Plots the distribution of p-values, marking significance threshold."""
#     plt.figure(figsize=(8, 6))
    
#     sns.histplot(df['p_value'], bins=20, color='teal', kde=False)
#     plt.axvline(0.05, color='red', linestyle='--', linewidth=2, label='p = 0.05 Threshold')
    
#     plt.xlabel('P-Value')
#     plt.ylabel('Frequency')
#     plt.title('Statistical Significance of Top Answers')
#     plt.legend()
    
#     out_path = os.path.join(out_dir, "pvalue_distribution.pdf")
#     plt.savefig(out_path)
#     plt.close()
#     print(f"Saved: {out_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--results_csv", type=str, default="./llm_pref_data/testing_results.csv")
#     parser.add_argument("--out_dir", type=str, default="./llm_pref_data/plots")
#     args = parser.parse_args()
    
#     os.makedirs(args.out_dir, exist_ok=True)
#     setup_neurips_style()
    
#     df = pd.read_csv(args.results_csv)
    
#     plot_reversal_scatter(df, args.out_dir)
#     plot_margin_densities(df, args.out_dir)
#     plot_pvalue_histogram(df, args.out_dir)

#     print("All plots generated successfully.")


import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def setup_neurips_style():
    """Configures matplotlib for clean, academic aesthetic."""
    try:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    except AttributeError:
        sns.set(style="whitegrid", context="paper", font_scale=1.5)
        
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })

def plot_reversal_scatter(df, out_dir):
    """Plots Raw vs Debiased margins, highlighting reversals."""
    plt.figure(figsize=(8, 6))
    
    reversals = df[(df['raw_delta'] > 0) & (df['debiased_delta'] < 0)]
    agreements = df[(df['raw_delta'] > 0) & (df['debiased_delta'] > 0)]
    
    plt.scatter(agreements['raw_delta'], agreements['debiased_delta'], 
                alpha=0.6, color='blue', label='Consistent Preference', edgecolor='w')
    plt.scatter(reversals['raw_delta'], reversals['debiased_delta'], 
                alpha=0.8, color='crimson', marker='X', s=80, label='Chandelier Reversal', edgecolor='w')
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    
    plt.xlabel(r'Raw DPO Margin ($\Delta^*$)')
    plt.ylabel(r'Debiased Margin ($\hat{\Delta}$)')
    plt.title('Chandelier Debiasing: The Reversal Effect')
    plt.legend(loc='upper left')
    
    out_path = os.path.join(out_dir, "reversal_scatter.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_margin_densities(df, out_dir):
    """Plots KDE distributions of raw vs debiased margins on separate axes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left subplot: Raw Margins (using shade=True for older Seaborn compatibility)
    try:
        sns.kdeplot(df['raw_delta'], fill=True, color='gray', ax=axes[0])
    except AttributeError:
        sns.kdeplot(df['raw_delta'], shade=True, color='gray', ax=axes[0])
        
    axes[0].set_title('Raw DPO Margin Density')
    axes[0].set_xlabel(r'Raw Margin ($\Delta^*$)')
    axes[0].set_ylabel('Density')
    
    # Right subplot: Debiased Margins
    try:
        sns.kdeplot(df['debiased_delta'], fill=True, color='purple', ax=axes[1])
    except AttributeError:
        sns.kdeplot(df['debiased_delta'], shade=True, color='purple', ax=axes[1])
        
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title('Debiased Margin Density')
    axes[1].set_xlabel(r'Debiased Margin ($\hat{\Delta}$)')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "margin_densities.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_pvalue_histogram(df, out_dir):
    """Plots the distribution of p-values, using safe matplotlib primitives."""
    plt.figure(figsize=(8, 6))
    
    # Using plt.hist completely bypasses old/new Seaborn histplot versioning bugs
    plt.hist(df['p_value'], bins=20, color='teal', edgecolor='white', alpha=0.8)
    plt.axvline(0.05, color='red', linestyle='--', linewidth=2, label='p = 0.05 Threshold')
    
    plt.xlabel('P-Value')
    plt.ylabel('Frequency')
    plt.title('Statistical Significance of Top Answers')
    plt.legend()
    
    out_path = os.path.join(out_dir, "pvalue_distribution.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, default="./llm_pref_data/testing_results.csv")
    parser.add_argument("--out_dir", type=str, default="./llm_pref_data/plots")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    setup_neurips_style()
    
    df = pd.read_csv(args.results_csv)
    
    plot_reversal_scatter(df, args.out_dir)
    plot_margin_densities(df, args.out_dir)
    plot_pvalue_histogram(df, args.out_dir)

    print("All plots generated successfully.")