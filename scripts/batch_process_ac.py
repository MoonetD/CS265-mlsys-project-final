#!/usr/bin/env python
"""
Batch processing script for activation checkpointing results.
This script runs the activation checkpointing algorithm for different models and batch sizes,
then generates visualizations of the results.
"""

import os
import re
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
import argparse
import tabulate  # Add tabulate library

# Set up Seaborn style for prettier plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def run_command(cmd):
    """Run a shell command and return the output."""
    print(f"Running: {cmd}")
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running command: {cmd}")
        print(stderr.decode('utf-8'))
        return None
    return stdout.decode('utf-8')

def parse_memory_info(log_file):
    """Parse memory information from the log file."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract memory information
    initial_peak_match = re.search(r"Initial peak memory: (\d+\.\d+) GB", content)
    final_peak_match = re.search(r"Final peak memory: (\d+\.\d+) GB", content)
    initial_exec_match = re.search(r"Initial execution time: (\d+\.\d+)s", content)
    final_exec_match = re.search(r"Final execution time: (\d+\.\d+)s", content)
    recompute_time_match = re.search(r"Recomputation time overhead: +(\d+\.\d+)s", content)
    
    # Extract summary information
    retained_count_match = re.search(r"Activations marked RETAINED: +(\d+) \((\d+\.\d+)%\)", content)
    recompute_count_match = re.search(r"Activations marked for RECOMPUTE: +(\d+) \((\d+\.\d+)%\)", content)
    retained_memory_match = re.search(r"Memory used by RETAINED activations: (\d+\.\d+) GB", content)
    saved_memory_match = re.search(r"Memory saved by RECOMPUTE decisions: (\d+\.\d+) GB", content)
    
    results = {}
    
    if initial_peak_match:
        results['initial_peak_memory'] = float(initial_peak_match.group(1))
    if final_peak_match:
        results['final_peak_memory'] = float(final_peak_match.group(1))
    if initial_exec_match:
        results['initial_exec_time'] = float(initial_exec_match.group(1))
    if final_exec_match:
        results['final_exec_time'] = float(final_exec_match.group(1))
    if recompute_time_match:
        results['recompute_time'] = float(recompute_time_match.group(1))
    
    if retained_count_match:
        results['retained_count'] = int(retained_count_match.group(1))
        results['retained_percent'] = float(retained_count_match.group(2))
    if recompute_count_match:
        results['recompute_count'] = int(recompute_count_match.group(1))
        results['recompute_percent'] = float(recompute_count_match.group(2))
    if retained_memory_match:
        results['retained_memory'] = float(retained_memory_match.group(1))
    if saved_memory_match:
        results['saved_memory'] = float(saved_memory_match.group(1))
    
    return results

def find_latest_log_file(log_dir="logs"):
    """Find the latest log file in the logs directory."""
    log_files = list(Path(log_dir).glob("activation_checkpointing_*.log"))
    if not log_files:
        print(f"No log files found in {log_dir} directory.")
        return None
    
    # Sort by creation time
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    return latest_log

def create_model_batch_configs():
    """Create configurations for models and batch sizes."""
    configs = {
        "resnet": {
            "batch_sizes": [4, 8, 16, 32, 64],
            "memory_budget": 1.5,
            "fixed_overhead": 0.8,
        },
        "transformer": {
            "batch_sizes": [2, 4, 8, 16, 32, 64, 128, 256],
            "memory_budget": 1.5,
            "fixed_overhead": 1.2,
        }
    }
    return configs

def run_activation_checkpointing(model, batch_sizes, memory_budget, fixed_overhead):
    """Run activation checkpointing for a model and batch sizes."""
    results = {}
    
    for bs in batch_sizes:
        print(f"\n{'-'*80}\nProcessing {model} with batch size {bs}\n{'-'*80}")
        
        # Build the command
        cmd = (f"conda run -n ml_env python starter_code/activation_checkpointing.py "
               f"--memory-budget {memory_budget} "
               f"--node-stats reports/profiler_stats_{model}_bs{bs}_node_stats.csv "
               f"--fixed-overhead {fixed_overhead}")
        
        # Run the command
        output = run_command(cmd)
        if output is None:
            print(f"Skipping batch size {bs} due to error")
            continue
        
        # Get the latest log file
        log_file = find_latest_log_file()
        if log_file is None:
            print(f"Could not find log file for batch size {bs}")
            continue
            
        print(f"Parsing results from {log_file}")
        
        # Parse memory information from the log file
        results[bs] = parse_memory_info(log_file)
        
        # Add batch size to results
        results[bs]['batch_size'] = bs
        
        # Wait a bit to ensure logs are written
        time.sleep(1)
    
    return results

def create_summary_dataframe(results):
    """Create a summary DataFrame from the results."""
    summary_data = []
    
    for bs, data in results.items():
        summary_data.append({
            'Batch Size': bs,
            'Initial Peak Memory (GB)': data.get('initial_peak_memory', 0),
            'Final Peak Memory (GB)': data.get('final_peak_memory', 0),
            'Memory Reduction (GB)': data.get('initial_peak_memory', 0) - data.get('final_peak_memory', 0),
            'Memory Reduction (%)': ((data.get('initial_peak_memory', 0) - data.get('final_peak_memory', 0)) / 
                                   data.get('initial_peak_memory', 1)) * 100,
            'Initial Execution Time (s)': data.get('initial_exec_time', 0),
            'Final Execution Time (s)': data.get('final_exec_time', 0),
            'Execution Time Overhead (s)': data.get('final_exec_time', 0) - data.get('initial_exec_time', 0),
            'Execution Time Overhead (%)': ((data.get('final_exec_time', 0) - data.get('initial_exec_time', 0)) / 
                                         data.get('initial_exec_time', 1)) * 100,
            'Recomputation Time (s)': data.get('recompute_time', 0),
            'Retained Activations': data.get('retained_count', 0),
            'Retained Activations (%)': data.get('retained_percent', 0),
            'Recomputed Activations': data.get('recompute_count', 0),
            'Recomputed Activations (%)': data.get('recompute_percent', 0),
            'Retained Memory (GB)': data.get('retained_memory', 0),
            'Saved Memory (GB)': data.get('saved_memory', 0),
        })
    
    df = pd.DataFrame(summary_data)
    # Sort by batch size
    df = df.sort_values('Batch Size')
    
    return df

def format_table(df, model):
    """Format the table for better display."""
    # Format the values to have fewer decimal places and appropriate formatting
    formatted_df = df.copy()
    
    # Format each column
    for col in formatted_df.columns:
        if 'Memory' in col and 'GB' in col:
            formatted_df[col] = formatted_df[col].map('{:.2f}'.format)
        elif 'Time' in col and 's' in col:
            formatted_df[col] = formatted_df[col].map('{:.4f}'.format)
        elif '%' in col:
            formatted_df[col] = formatted_df[col].map('{:.1f}%'.format)
    
    # Save to CSV
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{model}_activation_checkpointing_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary table to {csv_path}")
    
    # Generate a clean text table with tabulate
    print(f"\n{'='*100}")
    print(f"    {model.upper()} ACTIVATION CHECKPOINTING SUMMARY")
    print(f"{'='*100}\n")
    
    table_str = tabulate.tabulate(
        formatted_df, 
        headers='keys',
        tablefmt='grid',
        showindex=False,
        numalign='center',
        stralign='center'
    )
    print(table_str)
    
    # Save plain text table
    txt_path = os.path.join(output_dir, f"{model}_activation_checkpointing_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"{model.upper()} ACTIVATION CHECKPOINTING SUMMARY\n\n")
        f.write(table_str)
    print(f"Saved plain text table to {txt_path}")
    
    # Also save HTML table for easy visualization
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{model.capitalize()} Activation Checkpointing Summary</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                max-width: 1600px;
                margin: 0 auto;
                padding: 20px;
            }}
            table {{
                border-collapse: collapse;
                font-size: 12pt;
                width: 100%;
                margin: 20px 0;
            }}
            th {{
                background-color: #4472C4;
                color: white;
                font-weight: bold;
                text-align: center;
                padding: 8px;
                border: 1px solid #ddd;
            }}
            td {{
                text-align: center;
                padding: 8px;
                border: 1px solid #ddd;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            .memory-budget {{
                color: red;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h1>{model.capitalize()} Activation Checkpointing Summary</h1>
        <p>Memory Budget: <span class="memory-budget">1.5 GB</span></p>
        {tabulate.tabulate(formatted_df, headers='keys', tablefmt='html')}
    </body>
    </html>
    """
    
    html_path = os.path.join(output_dir, f"{model}_activation_checkpointing_summary.html")
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"Saved HTML table for easy visualization to {html_path}")
    
    return formatted_df

def plot_memory_comparison(df, model):
    """Plot memory comparison between with and without activation checkpointing."""
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(df))
    width = 0.35
    
    without_ac = df['Initial Peak Memory (GB)'].astype(float).values
    with_ac = df['Final Peak Memory (GB)'].astype(float).values
    
    # Convert GB to MB for clearer display
    without_ac_mb = without_ac * 1024
    with_ac_mb = with_ac * 1024
    
    # Calculate percentage reduction
    percent_reduction = ((without_ac - with_ac) / without_ac * 100)
    percent_reduction_labels = [f"{p:.1f}%" for p in percent_reduction]
    
    bars1 = plt.bar(x - width/2, without_ac_mb, width, label='Without AC', color='#1F77B4')
    bars2 = plt.bar(x + width/2, with_ac_mb, width, label='With AC', color='#FF7F0E')
    
    # Add a red line at 1.5 GB memory budget
    memory_budget_mb = 1.5 * 1024  # Convert GB to MB
    plt.axhline(y=memory_budget_mb, color='red', linestyle='-', linewidth=2, label='Memory Budget (1.5 GB)')
    
    plt.xlabel('Batch Size', fontweight='bold')
    plt.ylabel('Peak Memory (MB)', fontweight='bold')
    plt.title(f'Peak Memory Usage vs. Batch Size - {model.capitalize()}', fontweight='bold', fontsize=18)
    plt.xticks(x, df['Batch Size'], fontweight='bold')
    plt.legend(fontsize=12)
    
    # Add value labels on top of the bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 100,
                 f"{without_ac_mb[i]:.0f}", ha='center', va='bottom', fontweight='bold')
        plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 100,
                 f"{with_ac_mb[i]:.0f}", ha='center', va='bottom', fontweight='bold')
        
        # Add percentage labels
        plt.text(x[i], min(bar1.get_height(), bar2.get_height())/2,
                 percent_reduction_labels[i], ha='center', va='center', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    output_dir = "reports"
    plt.savefig(os.path.join(output_dir, f"{model}_memory_comparison.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{model}_memory_comparison.pdf"), bbox_inches='tight')
    print(f"Saved memory comparison plot to {output_dir}/{model}_memory_comparison.png")
    
def plot_time_comparison(df, model):
    """Plot execution time comparison between with and without activation checkpointing."""
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(df))
    width = 0.35
    
    without_ac = df['Initial Execution Time (s)'].astype(float).values
    with_ac = df['Final Execution Time (s)'].astype(float).values
    recompute_time = df['Recomputation Time (s)'].astype(float).values
    
    # Calculate percentage increase
    percent_increase = ((with_ac - without_ac) / without_ac * 100)
    percent_increase_labels = [f"+{p:.1f}%" for p in percent_increase]
    
    bars1 = plt.bar(x - width/2, without_ac, width, label='Without AC', color='#1F77B4')
    bars2 = plt.bar(x + width/2, with_ac, width, label='With AC', color='#FF7F0E')
    
    # Add a thin bar on top of 'With AC' to show the recomputation time portion
    plt.bar(x + width/2, recompute_time, width, bottom=with_ac-recompute_time, 
            label='Recomputation Overhead', color='#2CA02C', alpha=0.7)
    
    plt.xlabel('Batch Size', fontweight='bold')
    plt.ylabel('Execution Time (s)', fontweight='bold')
    plt.title(f'Execution Time vs. Batch Size - {model.capitalize()}', fontweight='bold', fontsize=18)
    plt.xticks(x, df['Batch Size'], fontweight='bold')
    plt.legend(fontsize=12)
    
    # Add value labels on top of the bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1,
                 f"{without_ac[i]:.2f}s", ha='center', va='bottom', fontweight='bold')
        plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.1,
                 f"{with_ac[i]:.2f}s", ha='center', va='bottom', fontweight='bold')
        
        # Add percentage labels
        plt.text(x[i], max(bar1.get_height(), bar2.get_height())/2,
                 percent_increase_labels[i], ha='center', va='center', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    output_dir = "reports"
    plt.savefig(os.path.join(output_dir, f"{model}_time_comparison.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{model}_time_comparison.pdf"), bbox_inches='tight')
    print(f"Saved time comparison plot to {output_dir}/{model}_time_comparison.png")

def process_model(model, batch_sizes, memory_budget, fixed_overhead):
    """Process a model's batch sizes and generate visualizations."""
    print(f"\n{'='*80}\nProcessing {model} with batch sizes {batch_sizes}\n{'='*80}")
    
    # Run activation checkpointing
    results = run_activation_checkpointing(model, batch_sizes, memory_budget, fixed_overhead)
    
    # Create summary DataFrame
    df = create_summary_dataframe(results)
    
    if df.empty:
        print(f"No results for {model}")
        return
    
    # Format and save table
    format_table(df, model)
    
    # Plot memory comparison
    plot_memory_comparison(df, model)
    
    # Plot time comparison
    plot_time_comparison(df, model)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Batch process activation checkpointing results.')
    parser.add_argument('--model', type=str, choices=['resnet', 'transformer', 'all'], default='all',
                        help='Model to process (resnet, transformer, or all)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("reports", exist_ok=True)
    
    # Get configurations
    configs = create_model_batch_configs()
    
    if args.model == 'all':
        # Process all models
        for model, config in configs.items():
            process_model(model, config['batch_sizes'], config['memory_budget'], config['fixed_overhead'])
    else:
        # Process only the specified model
        config = configs[args.model]
        process_model(args.model, config['batch_sizes'], config['memory_budget'], config['fixed_overhead'])
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main() 