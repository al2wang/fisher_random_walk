import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_epoch_data(results_dir):
    """
    Scans the results directory to discover tested epoch folders 
    and aggregates preference data across all domains.
    """
    epoch_folders = glob.glob(os.path.join(results_dir, "ep_*"))
    if not epoch_folders:
        print(f"No epoch directories found in {results_dir}")
        return None, None

    all_q_data = []
    all_v_data = []

    for folder in epoch_folders:
        try:
            epoch_val = int(folder.split("_")[-1])
        except ValueError:
            continue
            
        q_files = glob.glob(os.path.join(folder, "Q_matrix_*.csv"))
        for q_file in q_files:
            domain = os.path.basename(q_file).replace("Q_matrix_", "").replace(".csv", "")
            v_file = os.path.join(folder, f"V_matrix_{domain}.csv")
            
            if os.path.exists(v_file):
                q_df = pd.read_csv(q_file, index_col=0)
                v_df = pd.read_csv(v_file, index_col=0)
                
                # Unroll matrices into tidy dataframes
                for model_a in q_df.index:
                    for model_b in q_df.columns:
                        if model_a != model_b:
                            all_q_data.append({
                                'epochs': epoch_val,
                                'domain': domain,
                                'model_A': model_a,
                                'model_B': model_b,
                                'Q_value': q_df.loc[model_a, model_b],
                                'V_value': v_df.loc[model_a, model_b]
                            })
                            
    return pd.DataFrame(all_q_data), epoch_folders

def generate_llm_dashboard(results_dir, target_epoch=4500):
    df, epoch_folders = load_epoch_data(results_dir)
    if df is None or df.empty:
        return

    # Set up a comprehensive 1x3 dashboard layout
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    sns.set_theme(style="whitegrid")

    # ------------------------------------------------------------------------
    # Panel 1: Convergence of Win Margins over Training Epochs
    # ------------------------------------------------------------------------
    # Focus on key benchmark matchups aggregated across domains
    selected_matchups = [('gpt4o', 'llama2'), ('gpt4o', 'gpt3'), ('llama2', 'alpaca')]
    
    for mA, mB in selected_matchups:
        sub_df = df[(df['model_A'] == mA) & (df['model_B'] == mB)]
        if not sub_df.empty:
            # Average across domains to see the global epoch trajectory
            conv_curve = sub_df.groupby('epochs')['Q_value'].mean().reset_index()
            axes[0].plot(conv_curve['epochs'], conv_curve['Q_value'], 
                         marker='o', linestyle='-', linewidth=2, label=f"{mA} vs {mB}")
            
    axes[0].set_title("Reward Estimation Convergence", fontsize=12)
    axes[0].set_xlabel("Training Epochs")
    axes[0].set_ylabel("Mean Debiased Win Margin ($\widehat{Q}$)")
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0].legend(loc="upper right")

    # ------------------------------------------------------------------------
    # Panel 2: Statistical Significance Bounds (95% CIs)
    # ------------------------------------------------------------------------
    # Filter for your primary targeted optimization checkpoint
    df_target = df[df['epochs'] == target_epoch]
    if df_target.empty:
        # Fallback to the closest available epoch folder if target isn't found
        available_epochs = df['epochs'].unique()
        target_epoch = available_epochs[-1]
        df_target = df[df['epochs'] == target_epoch]
        print(f"Target epoch unavailable. Defaulting to visualization of Epoch {target_epoch}")

    # Isolate comparison metrics relative to the primary baseline (gpt4o)
    baseline_vs_all = df_target[df_target['model_A'] == 'gpt4o'].copy()
    baseline_vs_all['stderr'] = np.sqrt(np.maximum(baseline_vs_all['V_value'], 0))
    baseline_vs_all['ci_half'] = 1.96 * baseline_vs_all['stderr']
    
    # Average across domains for an overarching look at baseline dominance
    ci_agg = baseline_vs_all.groupby('model_B').agg({
        'Q_value': 'mean',
        'ci_half': 'mean'
    }).reset_index()

    axes[1].errorbar(ci_agg['model_B'], ci_agg['Q_value'], yerr=ci_agg['ci_half'],
                     fmt='D', color='crimson', ecolor='black', elinewidth=2, capsize=5,
                     markersize=8, label="Debiased Margin $\pm$ 95% CI")
    
    axes[1].set_title(f"GPT-4o Margin Bounds (Epoch {target_epoch})", fontsize=12)
    axes[1].set_xlabel("Opponent Model")
    axes[1].set_ylabel("Win Margin (Positive = GPT-4o Dominance)")
    axes[1].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[1].legend(loc="lower left")

    # ------------------------------------------------------------------------
    # Panel 3: Domain-Specific Relative Performance Heatmap
    # ------------------------------------------------------------------------
    # Compute an average capability score per model per domain 
    # Defined as the mean win margin across all possible opponents in that domain
    heatmap_data = df_target.groupby(['model_A', 'domain'])['Q_value'].mean().unstack(level=1)
    
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="vlag", center=0,
                cbar_kws={'label': 'Global Win Power'}, ax=axes[2], annot_kws={"size": 10})
    
    axes[2].set_title(f"Domain Capability Matrix", fontsize=12)
    axes[2].set_xlabel("MMLU Medical Domains")
    axes[2].set_ylabel("Evaluated Model")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=30, ha='right')

    # Save finalized analytical dashboard
    plt.tight_layout()
    dashboard_path = os.path.join(results_dir, "llm_evaluation_dashboard.png")
    plt.savefig(dashboard_path, dpi=300)
    print(f"\nComprehensive diagnostic dashboard successfully saved to: {dashboard_path}")

if __name__ == "__main__":
    RESULTS_DIRECTORY = "./results_llm_eval"
    if os.path.exists(RESULTS_DIRECTORY):
        generate_llm_dashboard(RESULTS_DIRECTORY, target_epoch=4500)
    else:
        print(f"Error: Target path '{RESULTS_DIRECTORY}' cannot be resolved.")