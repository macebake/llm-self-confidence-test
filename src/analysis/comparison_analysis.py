#!/usr/bin/env python3
"""
Multi-Model Comparison Analysis

Compares confidence experiment results across different models to identify
performance differences and generate comparison charts.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Any
import argparse
import os
from collections import defaultdict


class ModelComparisonAnalyzer:
    """Analyzes and compares confidence experiment results across multiple models."""

    def __init__(self, puzzle_type: str = ""):
        """
        Initialize comparison analyzer.

        Args:
            puzzle_type: Description of puzzle type (e.g. "4-character", "5-character")
        """
        self.puzzle_type = puzzle_type
        self.data = None
        self.models = []
        self.conditions = ['control', 'single-shot', 'confidence-pre', 'confidence-post']

    def load_multiple_jsonl_files(self, file_paths: List[str]) -> pd.DataFrame:
        """Load multiple JSONL files and combine them."""
        all_records = []

        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping")
                continue

            records = []
            with open(file_path, 'r') as f:
                for line in f:
                    record = json.loads(line.strip())
                    records.append(record)

            print(f"Loaded {len(records)} records from {file_path}")
            all_records.extend(records)

        raw_data = pd.DataFrame(all_records)

        # Filter out invalid records based on condition requirements
        if len(raw_data) > 0:
            # For control condition: null confidence is OK, but must have proposed_solution
            control_mask = (raw_data['condition'] == 'control')
            control_valid = control_mask & (raw_data['proposed_solution'].notna())

            # For confidence conditions: must have both confidence and proposed_solution
            conf_conditions = raw_data['condition'].isin(['single-shot', 'confidence-pre', 'confidence-post'])
            conf_valid = conf_conditions & (raw_data['confidence'].notna()) & (raw_data['proposed_solution'].notna())

            # Combine valid records
            valid_mask = control_valid | conf_valid
            self.data = raw_data[valid_mask].copy()

            # Report filtering
            filtered_count = len(raw_data) - len(self.data)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} records with null solutions/confidence")
        else:
            self.data = raw_data

        self.models = sorted(self.data['model'].unique()) if 'model' in self.data.columns and len(self.data) > 0 else []

        print(f"Total loaded: {len(self.data)} records across {len(self.models)} models")
        print(f"Models: {self.models}")
        print(f"Conditions: {sorted(self.data['condition'].unique())}")
        print(f"Puzzles: {len(self.data['puzzle_id'].unique())}")

        return self.data

    def get_model_condition_stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate statistics for each model and condition combination."""
        stats_dict = defaultdict(dict)

        for model in self.models:
            model_data = self.data[self.data['model'] == model]

            for condition in self.conditions:
                condition_data = model_data[model_data['condition'] == condition]

                if len(condition_data) == 0:
                    continue

                # Convert boolean to numeric for calculations
                accuracy_values = condition_data['correct'].astype(int)
                confidence_values = condition_data['confidence'].dropna()

                stats_dict[model][condition] = {
                    'n': len(condition_data),
                    'accuracy_mean': accuracy_values.mean(),
                    'accuracy_std': accuracy_values.std(),
                    'confidence_mean': confidence_values.mean() if len(confidence_values) > 0 else None,
                    'confidence_std': confidence_values.std() if len(confidence_values) > 0 else None,
                }

        return dict(stats_dict)

    def run_model_comparison_tests(self, model_stats: Dict) -> Dict[str, Any]:
        """Run statistical tests comparing models within each condition."""
        results = {}

        for condition in self.conditions:
            # Get accuracy data for each model in this condition
            condition_data = {}
            valid_models = []

            for model in self.models:
                if condition in model_stats[model]:
                    model_condition_data = self.data[
                        (self.data['model'] == model) & (self.data['condition'] == condition)
                    ]
                    accuracy_values = model_condition_data['correct'].astype(int).values

                    if len(accuracy_values) > 0:
                        condition_data[model] = accuracy_values
                        valid_models.append(model)

            if len(valid_models) < 2:
                continue

            # Kruskal-Wallis test across models for this condition
            kruskal_stat, kruskal_p = stats.kruskal(*condition_data.values())

            # Pairwise Mann-Whitney U tests between models
            pairwise_results = {}
            for i, model1 in enumerate(valid_models):
                for j, model2 in enumerate(valid_models[i+1:], i+1):
                    mw_stat, mw_p = stats.mannwhitneyu(
                        condition_data[model1], condition_data[model2], alternative='two-sided'
                    )
                    pairwise_results[f"{model1}_vs_{model2}"] = {
                        'statistic': mw_stat, 'p_value': mw_p
                    }

            results[condition] = {
                'kruskal_wallis': {'statistic': kruskal_stat, 'p_value': kruskal_p},
                'pairwise_tests': pairwise_results,
                'valid_models': valid_models
            }

        return results

    def calculate_model_effect_sizes(self, model_stats: Dict) -> Dict[str, Dict[str, float]]:
        """Calculate Cohen's d effect sizes between models."""
        effect_sizes = defaultdict(dict)

        for condition in self.conditions:
            valid_models = [m for m in self.models if condition in model_stats[m]]

            if len(valid_models) < 2:
                continue

            # Compare each pair of models
            for i, model1 in enumerate(valid_models):
                for model2 in valid_models[i+1:]:
                    stats1 = model_stats[model1][condition]
                    stats2 = model_stats[model2][condition]

                    # Cohen's d = (mean1 - mean2) / pooled_std
                    mean_diff = stats1['accuracy_mean'] - stats2['accuracy_mean']

                    # Pooled standard deviation
                    n1, n2 = stats1['n'], stats2['n']
                    s1, s2 = stats1['accuracy_std'], stats2['accuracy_std']
                    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    effect_sizes[condition][f"{model1}_vs_{model2}"] = cohens_d

        return dict(effect_sizes)

    def create_comparison_visualizations(self, model_stats: Dict, test_results: Dict, effect_sizes: Dict):
        """Create comprehensive model comparison visualizations."""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle(f'Multi-Model Confidence Experiment Comparison ({self.puzzle_type})',
                     fontsize=20, fontweight='bold')

        # 1. Accuracy comparison by condition (grouped bar chart)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_accuracy_comparison(ax1, model_stats)

        # 2. Confidence comparison by condition
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_confidence_comparison(ax2, model_stats)

        # 3. Effect sizes heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_effect_sizes_heatmap(ax3, effect_sizes)

        # 4. Statistical significance matrix
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_significance_matrix(ax4, test_results)

        # 5. Model performance ranking
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_model_ranking(ax5, model_stats)

        # 6. Condition effectiveness across models
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_condition_effectiveness(ax6, model_stats)

        # 7. Confidence vs Accuracy scatter by model
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_confidence_accuracy_by_model(ax7)

        # 8. Performance distribution violin plot
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_performance_distribution(ax8)

        plt.tight_layout()
        chart_file = f"model_comparison_charts_{self.puzzle_type.replace('-', '_').replace(' ', '_')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Comparison charts saved to: {chart_file}")

    def _plot_accuracy_comparison(self, ax, model_stats):
        """Plot grouped bar chart of accuracy by condition and model."""
        conditions = [c for c in self.conditions if any(c in model_stats[m] for m in self.models)]
        x = np.arange(len(conditions))
        width = 0.8 / len(self.models)

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.models)))

        for i, model in enumerate(self.models):
            means = []
            stds = []
            for condition in conditions:
                if condition in model_stats[model]:
                    means.append(model_stats[model][condition]['accuracy_mean'])
                    stds.append(model_stats[model][condition]['accuracy_std'])
                else:
                    means.append(0)
                    stds.append(0)

            ax.bar(x + i * width, means, width, yerr=stds, label=model,
                   alpha=0.8, color=colors[i], capsize=3)

        ax.set_xlabel('Condition')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison by Model and Condition')
        ax.set_xticks(x + width * (len(self.models) - 1) / 2)
        ax.set_xticklabels(conditions, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1.0)

    def _plot_confidence_comparison(self, ax, model_stats):
        """Plot confidence levels by model."""
        model_confidences = []
        model_names = []

        for model in self.models:
            confidences = []
            for condition in ['single-shot', 'confidence-pre', 'confidence-post']:
                if condition in model_stats[model] and model_stats[model][condition]['confidence_mean']:
                    confidences.append(model_stats[model][condition]['confidence_mean'])

            if confidences:
                model_confidences.append(np.mean(confidences))
                model_names.append(model)

        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
        bars = ax.bar(model_names, model_confidences, color=colors, alpha=0.7)
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Confidence')
        ax.set_title('Average Confidence by Model')
        ax.tick_params(axis='x', rotation=45)

    def _plot_effect_sizes_heatmap(self, ax, effect_sizes):
        """Plot heatmap of effect sizes between models."""
        if not effect_sizes:
            ax.text(0.5, 0.5, 'No effect size data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Effect Sizes (Cohen\'s d)')
            return

        # Create matrix for heatmap
        conditions = list(effect_sizes.keys())
        if not conditions:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Effect Sizes (Cohen\'s d)')
            return

        # Get all unique model comparisons
        all_comparisons = set()
        for condition_effects in effect_sizes.values():
            all_comparisons.update(condition_effects.keys())

        comparisons = sorted(list(all_comparisons))

        if not comparisons:
            ax.text(0.5, 0.5, 'No comparisons available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Effect Sizes (Cohen\'s d)')
            return

        # Create matrix
        matrix = np.zeros((len(conditions), len(comparisons)))
        for i, condition in enumerate(conditions):
            for j, comparison in enumerate(comparisons):
                if comparison in effect_sizes[condition]:
                    matrix[i, j] = effect_sizes[condition][comparison]

        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(comparisons)))
        ax.set_xticklabels(comparisons, rotation=45, ha='right')
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions)
        ax.set_title('Effect Sizes (Cohen\'s d)')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_significance_matrix(self, ax, test_results):
        """Plot matrix showing statistical significance between models."""
        if not test_results:
            ax.text(0.5, 0.5, 'No significance data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Statistical Significance')
            return

        conditions = list(test_results.keys())
        if not conditions:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Statistical Significance')
            return

        # Get Kruskal-Wallis p-values
        p_values = []
        for condition in conditions:
            if 'kruskal_wallis' in test_results[condition]:
                p_val = test_results[condition]['kruskal_wallis']['p_value']
                p_values.append(-np.log10(p_val) if p_val > 0 else 10)
            else:
                p_values.append(0)

        colors = ['red' if p > -np.log10(0.05) else 'gray' for p in p_values]
        bars = ax.barh(conditions, p_values, color=colors, alpha=0.7)
        ax.set_xlabel('-log10(p-value)')
        ax.set_title('Statistical Significance\n(Kruskal-Wallis across models)')
        ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax.legend()

    def _plot_model_ranking(self, ax, model_stats):
        """Plot overall model performance ranking."""
        model_scores = {}

        for model in self.models:
            scores = []
            for condition in self.conditions:
                if condition in model_stats[model]:
                    scores.append(model_stats[model][condition]['accuracy_mean'])

            if scores:
                model_scores[model] = np.mean(scores)

        if not model_scores:
            ax.text(0.5, 0.5, 'No ranking data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Performance Ranking')
            return

        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        models, scores = zip(*sorted_models)

        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, scores, color=colors, alpha=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Model Performance Ranking')
        ax.tick_params(axis='x', rotation=45)

        # Add rank numbers
        for i, (model, score) in enumerate(sorted_models):
            ax.text(i, score + 0.01, f'#{i+1}', ha='center', va='bottom', fontweight='bold')

    def _plot_condition_effectiveness(self, ax, model_stats):
        """Plot how different conditions perform across models."""
        condition_improvements = defaultdict(list)

        for model in self.models:
            if 'control' not in model_stats[model]:
                continue

            control_acc = model_stats[model]['control']['accuracy_mean']

            for condition in ['single-shot', 'confidence-pre', 'confidence-post']:
                if condition in model_stats[model]:
                    improvement = model_stats[model][condition]['accuracy_mean'] - control_acc
                    condition_improvements[condition].append(improvement)

        conditions = list(condition_improvements.keys())
        if not conditions:
            ax.text(0.5, 0.5, 'No improvement data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Condition Effectiveness')
            return

        # Box plot of improvements
        improvements_data = [condition_improvements[c] for c in conditions]
        bp = ax.boxplot(improvements_data, labels=conditions, patch_artist=True)

        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Accuracy Improvement vs Control')
        ax.set_title('Condition Effectiveness Across Models')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', rotation=45)

    def _plot_confidence_accuracy_by_model(self, ax):
        """Plot confidence vs accuracy scatter by model."""
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.models)))

        for i, model in enumerate(self.models):
            model_data = self.data[self.data['model'] == model]
            conf_conditions = ['single-shot', 'confidence-pre', 'confidence-post']

            for condition in conf_conditions:
                condition_data = model_data[model_data['condition'] == condition]
                conf_vals = condition_data['confidence'].dropna()

                if len(conf_vals) > 0:
                    acc_vals = condition_data.loc[conf_vals.index, 'correct'].astype(int)
                    # Add jitter to y-axis for visibility
                    jittered_acc = acc_vals + np.random.normal(0, 0.02, len(acc_vals))
                    ax.scatter(conf_vals, jittered_acc, alpha=0.6, color=colors[i],
                              s=20, label=f'{model}' if condition == conf_conditions[0] else '')

        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('Confidence vs Accuracy by Model')
        ax.legend()
        ax.set_ylim(-0.1, 1.1)

    def _plot_performance_distribution(self, ax):
        """Plot accuracy distribution by model using violin plots."""
        model_accuracies = []
        model_names = []

        for model in self.models:
            model_data = self.data[self.data['model'] == model]
            accuracies = model_data['correct'].astype(int).values

            if len(accuracies) > 0:
                model_accuracies.append(accuracies)
                model_names.append(model)

        if not model_accuracies:
            ax.text(0.5, 0.5, 'No distribution data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Distribution')
            return

        parts = ax.violinplot(model_accuracies, positions=range(len(model_names)), showmeans=True)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45)
        ax.set_ylabel('Accuracy')
        ax.set_title('Performance Distribution by Model')

    def print_comparison_summary(self, model_stats: Dict, test_results: Dict, effect_sizes: Dict):
        """Print comprehensive comparison summary."""
        print(f"\n{'='*100}")
        print(f"MULTI-MODEL COMPARISON SUMMARY ({self.puzzle_type})")
        print(f"{'='*100}")

        # Model performance overview
        print(f"\n=, MODEL PERFORMANCE OVERVIEW:")
        overall_scores = {}
        for model in self.models:
            scores = []
            for condition in self.conditions:
                if condition in model_stats[model]:
                    scores.append(model_stats[model][condition]['accuracy_mean'])
            if scores:
                overall_scores[model] = np.mean(scores)

        ranked_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (model, score) in enumerate(ranked_models, 1):
            print(f"   #{rank}: {model} ({score:.3f} avg accuracy)")

        # Statistical significance
        print(f"\n=, STATISTICAL SIGNIFICANCE:")
        significant_conditions = []
        for condition, results in test_results.items():
            if 'kruskal_wallis' in results:
                p_val = results['kruskal_wallis']['p_value']
                if p_val < 0.05:
                    significant_conditions.append(f"{condition} (p={p_val:.3f})")

        if significant_conditions:
            print(f"   Significant model differences found in:")
            for cond in significant_conditions:
                print(f"     • {cond}")
        else:
            print(f"   No significant differences between models found")

        # Best model by condition
        print(f"\n=� BEST MODEL BY CONDITION:")
        for condition in self.conditions:
            best_model = None
            best_score = -1

            for model in self.models:
                if condition in model_stats[model]:
                    score = model_stats[model][condition]['accuracy_mean']
                    if score > best_score:
                        best_score = score
                        best_model = model

            if best_model:
                print(f"   {condition}: {best_model} ({best_score:.3f} accuracy)")

        # Effect sizes summary
        print(f"\n=� LARGEST EFFECT SIZES:")
        all_effects = []
        for condition, condition_effects in effect_sizes.items():
            for comparison, d in condition_effects.items():
                all_effects.append((condition, comparison, d))

        # Sort by absolute effect size
        all_effects.sort(key=lambda x: abs(x[2]), reverse=True)

        for condition, comparison, d in all_effects[:5]:  # Top 5
            magnitude = "large" if abs(d) > 0.8 else ("medium" if abs(d) > 0.5 else "small")
            direction = "higher" if d > 0 else "lower"
            models = comparison.split('_vs_')
            print(f"   {condition}: {models[0]} vs {models[1]} (d={d:.3f}, {magnitude} effect, {models[0]} {direction})")

    def analyze_multiple_models(self, file_paths: List[str]) -> Dict[str, Any]:
        """Run complete multi-model analysis pipeline."""
        print(f"=, Starting Multi-Model Analysis for {self.puzzle_type} puzzles...")

        # Load data from multiple files
        self.load_multiple_jsonl_files(file_paths)

        if self.data is None or len(self.data) == 0:
            print("No data loaded. Exiting.")
            return {}

        # Calculate statistics
        model_stats = self.get_model_condition_stats()
        test_results = self.run_model_comparison_tests(model_stats)
        effect_sizes = self.calculate_model_effect_sizes(model_stats)

        # Print results
        self.print_comparison_summary(model_stats, test_results, effect_sizes)

        # Create visualizations
        self.create_comparison_visualizations(model_stats, test_results, effect_sizes)

        # Save results
        summary = {
            'puzzle_type': self.puzzle_type,
            'models': self.models,
            'model_stats': model_stats,
            'test_results': test_results,
            'effect_sizes': effect_sizes
        }

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy_to_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_json(item) for item in obj]
            else:
                return obj

        output_file = f"model_comparison_results_{self.puzzle_type.replace('-', '_').replace(' ', '_')}.json"
        with open(output_file, 'w') as f:
            json_ready_summary = convert_numpy_to_json(summary)
            json.dump(json_ready_summary, f, indent=2)
        print(f"\n📁 Comparison results saved to: {output_file}")

        return {'summary': summary}


def main():
    """Command line interface for multi-model comparison."""
    parser = argparse.ArgumentParser(description="Compare confidence experiment results across models")
    parser.add_argument("files", nargs='+', help="JSONL results files to compare")
    parser.add_argument("--puzzle-type", default="", help="Description of puzzle type")

    args = parser.parse_args()

    analyzer = ModelComparisonAnalyzer(args.puzzle_type)
    results = analyzer.analyze_multiple_models(args.files)

    return results


if __name__ == "__main__":
    main()