"""
Evaluation Adapter Module

This module provides an adapter for evaluating the pricing pipeline against test data
and producing standardized evaluation metrics.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from .pipeline_adapter import PricingPipelineAdapter

class PricingEvaluationAdapter:
    """
    Adapter class for evaluating the pricing pipeline.
    
    This class runs the pipeline on test data and produces standardized evaluation metrics
    that can be compared across different models and approaches.
    """
    
    def __init__(self, pipeline=None, output_dir=None, model_name="RL_Pipeline"):
        """
        Initialize the evaluation adapter
        
        Args:
            pipeline: Optional PricingPipelineAdapter instance
            output_dir: Directory to save evaluation results
            model_name: Name of the model being evaluated
        """
        self.pipeline = pipeline or PricingPipelineAdapter(use_rl=True)
        self.output_dir = output_dir
        self.model_name = model_name
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def evaluate(self, test_data, train_data=None):
        """
        Evaluate the pipeline on test data
        
        Args:
            test_data: DataFrame with test data
            train_data: Optional DataFrame with training data for reference
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            'model_name': self.model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {},
            'segment_metrics': [],
            'profit_metrics': {}
        }
        
        # Track predictions and actuals
        predictions = []
        actuals = []
        segments = []
        
        # Process each test case
        for _, row in test_data.iterrows():
            # Prepare input data
            product_data = {
                'product_id': row.get('product_id', 'unknown'),
                'product_type': row.get('product_type', 'standard'),
                'product_group': row.get('product_group', 'general'),
                'price': row['actual_price'],
                'cost': row.get('cost', row['actual_price'] * 0.6),
                'rating': row.get('rating', 4.0),
                'number_of_orders': row.get('orders', 50),
                'elasticity': row.get('elasticity', 1.0)
            }
            
            market_info = {
                'competitor_price': row.get('competitor_price', row['actual_price']),
                'market_share': row.get('market_share', 0.1),
                'market_growth': row.get('market_growth', 0.05)
            }
            
            # Run pipeline
            pipeline_result = self.pipeline.run_pipeline(
                product_data,
                market_info,
                historical_data=train_data
            )
            
            if pipeline_result['success']:
                pred_price = pipeline_result['pricing']['recommended_price']
                actual_price = row['actual_price']
                segment = pipeline_result.get('segments', {}).get('primary_segment', 'unknown')
                
                predictions.append(pred_price)
                actuals.append(actual_price)
                segments.append(segment)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate basic metrics
        results['metrics'] = self._calculate_metrics(predictions, actuals)
        
        # Calculate segment-specific metrics
        results['segment_metrics'] = self._calculate_segment_metrics(
            predictions, actuals, segments
        )
        
        # Calculate profit metrics
        results['profit_metrics'] = self._calculate_profit_metrics(
            predictions, actuals, test_data
        )
        
        # Add summary statistics
        results['predictions_summary'] = {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions))
        }
        
        results['actuals_summary'] = {
            'mean': float(np.mean(actuals)),
            'std': float(np.std(actuals)),
            'min': float(np.min(actuals)),
            'max': float(np.max(actuals))
        }
        
        # Save results if output directory specified
        if self.output_dir:
            self._save_results(results)
        
        return results
    
    def _calculate_metrics(self, predictions, actuals):
        """Calculate standard evaluation metrics"""
        metrics = {}
        
        # Basic error metrics
        metrics['mae'] = float(np.mean(np.abs(predictions - actuals)))
        metrics['mse'] = float(np.mean((predictions - actuals) ** 2))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        
        # R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        metrics['r2'] = float(1 - (ss_res / ss_tot))
        
        # Percentage errors
        abs_perc_errors = np.abs((actuals - predictions) / actuals) * 100
        perc_errors = ((actuals - predictions) / actuals) * 100
        
        metrics['mape'] = float(np.mean(abs_perc_errors))
        metrics['mpe'] = float(np.mean(perc_errors))
        metrics['median_ape'] = float(np.median(abs_perc_errors))
        metrics['max_ape'] = float(np.max(abs_perc_errors))
        metrics['min_ape'] = float(np.min(abs_perc_errors))
        
        # Pricing specific metrics
        metrics['margin'] = float(np.mean((predictions - actuals * 0.6) / predictions * 100))
        metrics['margin_diff'] = float(metrics['margin'] - 32.63)  # Baseline margin
        metrics['margin_diff_pct'] = float(metrics['margin_diff'] / 32.63 * 100)
        
        # Price/cost ratio
        price_cost_ratio = predictions / (actuals * 0.6)
        metrics['price_cost_ratio'] = float(np.mean(price_cost_ratio))
        metrics['price_cost_ratio_diff'] = float(metrics['price_cost_ratio'] - 1.5645)
        metrics['price_cost_ratio_diff_pct'] = float(metrics['price_cost_ratio_diff'] / 1.5645 * 100)
        
        # Competitive position
        comp_prices = actuals  # Using actuals as proxy for competitor prices
        position = (predictions - comp_prices) / comp_prices * 100
        metrics['competitive_position'] = float(np.mean(position))
        metrics['competitive_position_diff'] = float(metrics['competitive_position'] - 24.5858)
        
        # Percentage above competition
        metrics['pct_above_competition'] = float(np.mean(predictions > comp_prices) * 100)
        metrics['pct_above_competition_diff'] = float(metrics['pct_above_competition'] - 67.0)
        
        # Additional metrics
        metrics['optimal_alignment'] = float(0.377)  # Placeholder
        metrics['revenue_impact'] = float(np.mean((predictions - actuals) / actuals * 100))
        metrics['profit_impact'] = float(np.mean(
            (predictions - actuals * 0.6) / (actuals - actuals * 0.6) * 100
        ))
        
        return metrics
    
    def _calculate_segment_metrics(self, predictions, actuals, segments):
        """Calculate metrics for each customer segment"""
        segment_metrics = []
        unique_segments = set(segments)
        
        for segment in unique_segments:
            mask = np.array(segments) == segment
            seg_preds = predictions[mask]
            seg_acts = actuals[mask]
            
            if len(seg_preds) > 0:
                metrics = {
                    'segment': segment,
                    'count': int(np.sum(mask)),
                    'avg_actual_price': float(np.mean(seg_acts)),
                    'avg_predicted_price': float(np.mean(seg_preds)),
                    'mae': float(np.mean(np.abs(seg_preds - seg_acts))),
                    'mape': float(np.mean(np.abs((seg_acts - seg_preds) / seg_acts)) * 100),
                    'price_diff': float(np.mean(seg_preds - seg_acts)),
                    'price_diff_pct': float(np.mean((seg_preds - seg_acts) / seg_acts * 100))
                }
                segment_metrics.append(metrics)
        
        return segment_metrics
    
    def _calculate_profit_metrics(self, predictions, actuals, test_data):
        """Calculate profit-related metrics"""
        # Assume 60% cost ratio if not provided
        costs = test_data.get('cost', actuals * 0.6)
        
        # Calculate actual and predicted profits
        actual_profits = (actuals - costs) * test_data.get('orders', 50)
        
        # Estimate demand impact using simple elasticity model
        elasticity = test_data.get('elasticity', np.ones_like(predictions))
        price_ratio = predictions / actuals
        demand_ratio = price_ratio ** (-elasticity)
        predicted_orders = test_data.get('orders', 50) * demand_ratio
        
        predicted_profits = (predictions - costs) * predicted_orders
        
        metrics = {
            'total_actual_profit': float(np.sum(actual_profits)),
            'total_predicted_profit': float(np.sum(predicted_profits)),
            'profit_change': float(np.sum(predicted_profits - actual_profits)),
            'profit_change_pct': float(np.sum(predicted_profits - actual_profits) / np.sum(actual_profits) * 100),
            'volume_impact': float(np.mean(demand_ratio)),
            'margin_impact': float(np.mean((predictions - costs) / predictions * 100) - 
                                np.mean((actuals - costs) / actuals * 100)),
            'price_effect': float(np.sum((predictions - actuals) * test_data.get('orders', 50))),
            'volume_effect': float(np.sum((predicted_orders - test_data.get('orders', 50)) * actuals)),
            'price_effect_pct': float(np.sum((predictions - actuals) * test_data.get('orders', 50)) / 
                                    np.sum(actual_profits) * 100),
            'volume_effect_pct': float(np.sum((predicted_orders - test_data.get('orders', 50)) * actuals) /
                                     np.sum(actual_profits) * 100)
        }
        
        return metrics
    
    def _save_results(self, results):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full results as JSON
        results_file = os.path.join(
            self.output_dir,
            f'evaluation_results_{timestamp}.json'
        )
        with open(results_file, 'w') as f:
            json.dump([results], f, indent=2)
        
        # Save profit comparison as CSV
        profit_file = os.path.join(
            self.output_dir,
            'profit_comparison.csv'
        )
        profit_data = {
            'Model': [results['model_name']],
            'Profit Change (%)': [results['profit_metrics']['profit_change_pct']],
            'Volume Impact': [results['profit_metrics']['volume_impact']],
            'Margin Impact': [results['profit_metrics']['margin_impact']],
            'Price Effect (%)': [results['profit_metrics']['price_effect_pct']],
            'Volume Effect (%)': [results['profit_metrics']['volume_effect_pct']]
        }
        pd.DataFrame(profit_data).to_csv(profit_file, index=False) 