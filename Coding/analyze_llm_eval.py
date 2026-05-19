import os
import pandas as pd
import numpy as np

def generate_leaderboard(results_dir):
    domains = ['anatomy', 'clinical_knowledge', 'college_biology', 'college_medicine', 'medical_genetics']
    models = ['gpt3', 'alpaca', 'gpt4o', 'llama1', 'llama2']
    
    print("\n" + "="*80)
    print("FISHER RANDOM WALK: DEBIASED LLM EVALUATION LEADERBOARD")
    print("="*80)
    
    for domain in domains:
        q_path = os.path.join(results_dir, f"Q_matrix_{domain}.csv")
        v_path = os.path.join(results_dir, f"V_matrix_{domain}.csv")
        
        if not os.path.exists(q_path):
            print(f"Waiting for {domain} results...")
            continue
            
        q_df = pd.read_csv(q_path, index_col=0)
        v_df = pd.read_csv(v_path, index_col=0)
        
        print(f"\n--- Domain: {domain.upper()} ---")
        print(f"{'Model A':<10} vs {'Model B':<10} | {'Debiased Win Margin (Q)':<25} | {'95% CI':<20} | {'Significant?'}")
        print("-" * 80)
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_a = models[i]
                model_b = models[j]
                
                # Q > 0 implies Model A is preferred to Model B
                q_val = q_df.loc[model_a, model_b]
                var_val = v_df.loc[model_a, model_b]
                std_err = np.sqrt(max(var_val, 0)) # Ensure non-negative variance
                
                ci_lower = q_val - 1.96 * std_err
                ci_upper = q_val + 1.96 * std_err
                
                # Check if 0 is excluded from the CI
                significant = "YES" if (ci_lower > 0 or ci_upper < 0) else "NO"
                
                winner = model_a if q_val > 0 else model_b
                margin = abs(q_val)
                
                print(f"{model_a:<10} vs {model_b:<10} | {winner} by {margin:.4f} \t | [{ci_lower:7.4f}, {ci_upper:7.4f}] | {significant}")

if __name__ == "__main__":
    RESULTS_DIRECTORY = "./results_llm_eval"
    generate_leaderboard(RESULTS_DIRECTORY)