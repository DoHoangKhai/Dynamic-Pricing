"""
Test script for the evaluation adapter with benchmark comparisons
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from .evaluation_adapter import PricingEvaluationAdapter
from .pipeline_adapter import PricingPipelineAdapter
from .benchmark_models.pricing_models import (
    CostPlusPricingModel,
    CompetitivePricingModel,
    ElasticityPricingModel,
    RuleBasedPricingModel
)

def create_sample_data(n_samples=100):
    """Create sample test data"""
    np.random.seed(42)
    
    data = {
        'product_id': [f'PROD_{i:03d}' for i in range(n_samples)],
        'product_type': np.random.choice(['Electronics', 'Clothing', 'Books'], n_samples),
        'product_group': np.random.choice(['Premium', 'Standard', 'Budget'], n_samples),
        'actual_price': np.random.uniform(10, 500, n_samples),
        'cost': None,  # Will be calculated as 60% of price
        'rating': np.random.uniform(1, 5, n_samples),
        'orders': np.random.randint(10, 1000, n_samples),
        'elasticity': np.random.uniform(0.5, 2.0, n_samples),
        'competitor_price': None,  # Will be calculated relative to actual price
        'market_share': np.random.uniform(0.05, 0.3, n_samples),
        'market_growth': np.random.uniform(-0.1, 0.2, n_samples)
    }
    
    # Calculate derived fields
    df = pd.DataFrame(data)
    df['cost'] = df['actual_price'] * 0.6
    df['competitor_price'] = df['actual_price'] * np.random.uniform(0.8, 1.2, n_samples)
    
    return df

def create_benchmark_models():
    """Create benchmark pricing models"""
    return {
        'CostPlus': CostPlusPricingModel(markup=0.5),  # 50% markup
        'Competitive': CompetitivePricingModel(discount=0.05),  # 5% below competition
        'Elasticity': ElasticityPricingModel(),
        'RuleBased': RuleBasedPricingModel()
    }

def plot_model_comparison(results_dict, output_dir):
    """Generate comparison plots for different models"""
    # Create figures directory
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Prepare data for plotting
    models = list(results_dict.keys())
    metrics = {
        'MAPE (%)': [results['metrics']['mape'] for results in results_dict.values()],
        'R2 Score': [results['metrics']['r2'] for results in results_dict.values()],
        'Profit Change (%)': [results['profit_metrics']['profit_change_pct'] for results in results_dict.values()],
        'Volume Impact': [results['profit_metrics']['volume_impact'] for results in results_dict.values()]
    }
    
    # Plot settings
    plt.style.use('seaborn')
    colors = sns.color_palette('husl', n_colors=len(metrics))
    
    # 1. Performance Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
    
    for (metric, values), ax, color in zip(metrics.items(), axes.flat, colors):
        ax.bar(models, values, color=color, alpha=0.7)
        ax.set_title(metric)
        ax.set_xticklabels(models, rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'performance_comparison.png'))
    plt.close()
    
    # 2. Price Distribution Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    data_to_plot = []
    labels = []
    
    for model_name, results in results_dict.items():
        data_to_plot.append(results['predictions_summary']['mean'])
        labels.append(f"{model_name}\nMean: ${results['predictions_summary']['mean']:.2f}")
    
    ax.bar(range(len(data_to_plot)), data_to_plot, alpha=0.7)
    ax.set_xticks(range(len(data_to_plot)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_title('Average Predicted Price by Model')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'price_distribution.png'))
    plt.close()
    
    # 3. Profit Impact Analysis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    margin_impact = [results['profit_metrics']['margin_impact'] for results in results_dict.values()]
    volume_impact = [results['profit_metrics']['volume_impact'] for results in results_dict.values()]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, margin_impact, width, label='Margin Impact', color='skyblue', alpha=0.7)
    ax.bar(x + width/2, volume_impact, width, label='Volume Impact', color='lightgreen', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.set_title('Margin vs Volume Impact by Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'profit_impact.png'))
    plt.close()

def test_evaluation_adapter():
    """Test the evaluation adapter with sample data and benchmarks"""
    # Create output directory
    output_dir = os.path.join(
        'test',
        'model_evaluation',
        'results',
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample data
    test_data = create_sample_data(100)
    train_data = create_sample_data(200)
    
    # Initialize results dictionary
    all_results = {}
    
    # 1. Evaluate RL Pipeline
    pipeline = PricingPipelineAdapter(
        use_rl=True,
        model_path=None,
        log_level=20
    )
    
    evaluator = PricingEvaluationAdapter(
        pipeline=pipeline,
        output_dir=output_dir,
        model_name='RL_Pipeline'
    )
    
    print("\nEvaluating RL Pipeline...")
    rl_results = evaluator.evaluate(test_data, train_data)
    all_results['RL_Pipeline'] = rl_results
    
    # 2. Evaluate Benchmark Models
    benchmark_models = create_benchmark_models()
    
    for model_name, model in benchmark_models.items():
        print(f"\nEvaluating {model_name}...")
        evaluator = PricingEvaluationAdapter(
            pipeline=model,
            output_dir=output_dir,
            model_name=model_name
        )
        results = evaluator.evaluate(test_data, train_data)
        all_results[model_name] = results
    
    # Generate comparison plots
    plot_model_comparison(all_results, output_dir)
    
    # Print comparative summary
    print("\n=== Model Comparison Summary ===")
    comparison_data = []
    
    for model_name, results in all_results.items():
        metrics = results['metrics']
        profit_metrics = results['profit_metrics']
        
        comparison_data.append({
            'Model': model_name,
            'MAPE (%)': metrics['mape'],
            'R2': metrics['r2'],
            'Profit Change (%)': profit_metrics['profit_change_pct'],
            'Volume Impact': profit_metrics['volume_impact'],
            'Margin Impact': profit_metrics['margin_impact']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.set_index('Model', inplace=True)
    
    print("\nMetrics Comparison:")
    print(comparison_df.round(4))
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
    
    print("\nTest completed successfully!")
    print(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    test_evaluation_adapter() 