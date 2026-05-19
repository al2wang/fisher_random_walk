import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MODELS = ['gpt3', 'alpaca', 'gpt4o', 'llama1', 'llama2']
DOMAINS = ['anatomy', 'clinical_knowledge', 'college_biology', 'college_medicine', 'medical_genetics']

def load_naive_margins(eval_dir):
    """
    Reads the raw GPT-4 JSON judgments and computes the naive win margin
    (Win Rate of A vs B) - (Win Rate of B vs A).
    """
    naive_matrices = {domain: pd.DataFrame(0.0, index=MODELS, columns=MODELS) for domain in DOMAINS}
    
    for domain in DOMAINS:
        file_path = os.path.join(eval_dir, f"{domain}_num=100.json")
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, "r") as f:
            pairs = json.load(f)
            
        # Tally wins
        win_counts = {mA: {mB: 0 for mB in MODELS} for mA in MODELS}
        pair_counts = {mA: {mB: 0 for mB in MODELS} for mA in MODELS}
        
        for pair in pairs:
            m1, m2 = pair["variable_name1"], pair["variable_name2"]
            score = pair["GPT4_output"] # Assuming 1 means m1 wins, 0 means m2 wins
            
            if m1 in MODELS and m2 in MODELS:
                win_counts[m1][m2] += score
                win_counts[m2][m1] += (1 - score)
                pair_counts[m1][m2] += 1
                pair_counts[m2][m1] += 1

        # Convert to margins
        for i in range(len(MODELS)):
            for j in range(i + 1, len(MODELS)):
                mA, mB = MODELS[i], MODELS[j]
                if pair_counts[mA][mB] > 0:
                    rate_A = win_counts[mA][mB] / pair_counts[mA][mB]
                    rate_B = win_counts[mB][mA] / pair_counts[mB][mA]
                    margin = rate_A - rate_B
                    
                    naive_matrices[domain].loc[mA, mB] = margin
                    naive_matrices[domain].loc[mB, mA] = -margin
                    
    return naive_matrices

def load_debiased_margins(results_dir, target_epoch=4500):
    """Loads the Q_matrix CSVs from the specified epoch folder."""
    epoch_dir = os.path.join(results_dir, f"ep_{target_epoch}")
    debiased_matrices = {}
    
    for domain in DOMAINS:
        q_path = os.path.join(epoch_dir, f"Q_matrix_{domain}.csv")
        if os.path.exists(q_path):
            debiased_matrices[domain] = pd.read_csv(q_path, index_col=0)
            
    return debiased_matrices

def generate_delta_dumbbell_plot(eval_dir, results_dir, target_epoch=4500):
    naive_mats = load_naive_margins(eval_dir)
    debiased_mats = load_debiased_margins(results_dir, target_epoch)
    
    if not debiased_mats:
        print("Could not find debiased matrices. Check your results directory.")
        return

    # Compile data into a single dataframe for plotting
    plot_data = []
    for domain in debiased_mats.keys():
        for i in range(len(MODELS)):
            for j in range(i + 1, len(MODELS)):
                mA, mB = MODELS[i], MODELS[j]
                
                # Exclude GPT-4o from this specific plot to focus on the weaker models 
                # where the GPT-4 judge bias is most prominent.
                if mA == 'gpt4o' or mB == 'gpt4o':
                    continue
                    
                naive_val = naive_mats[domain].loc[mA, mB]
                debiased_val = debiased_mats[domain].loc[mA, mB]
                
                plot_data.append({
                    'Matchup': f"{mA} vs {mB} ({domain[:4].upper()})",
                    'Naive': naive_val,
                    'Debiased': debiased_val,
                    'Flipped': (naive_val > 0 and debiased_val < 0) or (naive_val < 0 and debiased_val > 0)
                })
                
    df = pd.DataFrame(plot_data)
    # Sort by the magnitude of the shift
    df['Shift'] = abs(df['Debiased'] - df['Naive'])
    df = df.sort_values(by='Shift', ascending=True).tail(15) # Top 15 biggest shifts
    
    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    ax = plt.gca()
    
    # Draw the arrows
    for idx, row in df.iterrows():
        color = 'crimson' if row['Flipped'] else 'slategray'
        alpha = 0.9 if row['Flipped'] else 0.5
        linewidth = 2.5 if row['Flipped'] else 1.5
        
        ax.annotate('', xy=(row['Debiased'], row['Matchup']), xytext=(row['Naive'], row['Matchup']),
                    arrowprops=dict(arrowstyle="->", color=color, lw=linewidth, alpha=alpha))
        
    # Plot the starting points (Naive) and ending points (Debiased)
    plt.scatter(df['Naive'], df['Matchup'], color='gray', s=80, label='Naive GPT-4 Judge', zorder=3)
    plt.scatter(df['Debiased'], df['Matchup'], color='blue', s=80, label='Debiased $\hat{Q}$', zorder=3)
    
    # Formatting
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.title("The Debiasing Delta: How Fisher Random Walk Corrects Judge Bias", fontsize=14, pad=15)
    plt.xlabel("Win Margin (Positive = Left Model Wins, Negative = Right Model Wins)", fontsize=11)
    plt.ylabel("Matchup (Domain)", fontsize=11)
    
    # Highlight flips in legend
    import matplotlib.patches as mpatches
    flip_patch = mpatches.Patch(color='crimson', label='Leaderboard Flip (Winner Changed)')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(flip_patch)
    plt.legend(handles=handles, loc='lower right', frameon=True, shadow=True)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "debiasing_delta_plot.png")
    plt.savefig(out_path, dpi=300)
    print(f"Debiasing Delta plot saved to: {out_path}")

if __name__ == "__main__":
    EVAL_DIRECTORY = "./eval"
    RESULTS_DIRECTORY = "./results_llm_eval"
    generate_delta_dumbbell_plot(EVAL_DIRECTORY, RESULTS_DIRECTORY, target_epoch=4500)