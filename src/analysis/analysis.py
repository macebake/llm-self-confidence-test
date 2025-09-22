#!/usr/bin/env python3
"""
Confidence Experiment Analysis

Analyzes results from metacognitive accuracy experiments to identify
statistically significant differences between conditions.
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


class ConfidenceAnalyzer:
    """Analyzes confidence experiment results with statistical tests and visualizations."""

    def __init__(self, jsonl_path: str, puzzle_type: str = ""):
        """
        Initialize analyzer.

        Args:
            jsonl_path: Path to JSONL results file
            puzzle_type: Description of puzzle type (e.g. "4-character", "5-character")
        """
        self.jsonl_path = jsonl_path
        self.puzzle_type = puzzle_type
        self.data = None
        self.model_name = None
        self.conditions = ['control', 'single-shot', 'confidence-pre', 'confidence-post']

    def load_data(self) -> pd.DataFrame:
        """Load JSONL data into pandas DataFrame."""
        records = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                records.append(json.loads(line.strip()))

        raw_data = pd.DataFrame(records)

        # Filter out invalid records based on condition requirements
        valid_mask = pd.Series([True] * len(raw_data))

        # For control condition: null confidence is OK, but must have proposed_solution
        control_mask = (raw_data['condition'] == 'control')
        control_valid = control_mask & (raw_data['proposed_solution'].notna())

        # For confidence conditions: must have both confidence and proposed_solution
        conf_conditions = raw_data['condition'].isin(['single-shot', 'confidence-pre', 'confidence-post'])
        conf_valid = conf_conditions & (raw_data['confidence'].notna()) & (raw_data['proposed_solution'].notna())

        # Combine valid records
        valid_mask = control_valid | conf_valid

        self.data = raw_data[valid_mask].copy()

        # Detect model name from data
        if 'model' in self.data.columns and len(self.data) > 0:
            unique_models = self.data['model'].unique()
            if len(unique_models) == 1:
                self.model_name = unique_models[0].upper()
            else:
                self.model_name = f"Mixed-{len(unique_models)}-Models"
        else:
            self.model_name = "GPT-4o"  # Default fallback

        # Report filtering
        filtered_count = len(raw_data) - len(self.data)
        print(f"Loaded {len(raw_data)} records from {self.jsonl_path}")
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} records with null solutions/confidence")
        print(f"Valid records: {len(self.data)}")
        print(f"Model: {self.model_name}")
        print(f"Conditions: {sorted(self.data['condition'].unique())}")
        print(f"Puzzles: {len(self.data['puzzle_id'].unique())}")

        return self.data

    def get_condition_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate basic statistics for each condition."""
        stats_dict = {}

        for condition in self.conditions:
            condition_data = self.data[self.data['condition'] == condition]

            # Convert boolean to numeric for calculations
            accuracy_values = condition_data['correct'].astype(int)
            confidence_values = condition_data['confidence'].dropna()  # Remove null for control

            stats_dict[condition] = {
                'n': len(condition_data),
                'accuracy_mean': accuracy_values.mean(),
                'accuracy_std': accuracy_values.std(),
                'confidence_mean': confidence_values.mean() if len(confidence_values) > 0 else None,
                'confidence_std': confidence_values.std() if len(confidence_values) > 0 else None,
            }

        return stats_dict

    def run_statistical_tests(self, condition_stats: Dict) -> Dict[str, Any]:
        """Run all statistical tests."""
        results = {}

        # Get accuracy data for each condition
        accuracy_data = {}
        for condition in self.conditions:
            condition_data = self.data[self.data['condition'] == condition]
            accuracy_data[condition] = condition_data['correct'].astype(int).values

        # 1. Kruskal-Wallis test for all conditions
        kruskal_stat, kruskal_p = stats.kruskal(*accuracy_data.values())
        results['kruskal_wallis'] = {'statistic': kruskal_stat, 'p_value': kruskal_p}

        # 2. Mann-Whitney U tests: each confidence condition vs control
        control_accuracy = accuracy_data['control']
        mann_whitney_results = {}

        for condition in ['single-shot', 'confidence-pre', 'confidence-post']:
            condition_accuracy = accuracy_data[condition]
            mw_stat, mw_p = stats.mannwhitneyu(condition_accuracy, control_accuracy, alternative='two-sided')
            mann_whitney_results[f"{condition}_vs_control"] = {
                'statistic': mw_stat, 'p_value': mw_p
            }

        results['mann_whitney'] = mann_whitney_results

        # 3. Chi-square test for independence
        # Create contingency table: conditions x (correct/incorrect)
        contingency_data = []
        for condition in self.conditions:
            condition_data = self.data[self.data['condition'] == condition]
            correct_count = condition_data['correct'].sum()
            incorrect_count = len(condition_data) - correct_count
            contingency_data.append([correct_count, incorrect_count])

        contingency_table = np.array(contingency_data)
        chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
        results['chi_square'] = {
            'statistic': chi2_stat, 'p_value': chi2_p, 'dof': dof,
            'contingency_table': contingency_table
        }

        return results

    def calculate_effect_sizes(self, condition_stats: Dict) -> Dict[str, float]:
        """Calculate Cohen's d effect sizes vs control."""
        effect_sizes = {}
        control_stats = condition_stats['control']

        for condition in ['single-shot', 'confidence-pre', 'confidence-post']:
            condition_data = condition_stats[condition]

            # Cohen's d = (mean1 - mean2) / pooled_std
            mean_diff = condition_data['accuracy_mean'] - control_stats['accuracy_mean']

            # Pooled standard deviation
            n1, n2 = condition_data['n'], control_stats['n']
            s1, s2 = condition_data['accuracy_std'], control_stats['accuracy_std']
            pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            effect_sizes[condition] = cohens_d

        return effect_sizes

    def analyze_confidence_calibration(self) -> Dict[str, Any]:
        """Analyze how well confidence ratings predict actual performance (calibration analysis)."""
        calibration_results = {}

        # Only analyze conditions with confidence ratings
        confidence_conditions = ['single-shot', 'confidence-pre', 'confidence-post']

        for condition in confidence_conditions:
            condition_data = self.data[
                (self.data['condition'] == condition) &
                (self.data['confidence'].notna())
            ]

            if len(condition_data) == 0:
                continue

            confidences = condition_data['confidence'].values
            accuracies = condition_data['correct'].astype(int).values

            # Calculate correlation between confidence and accuracy
            if len(confidences) > 1:
                correlation, p_value = stats.pearsonr(confidences, accuracies)

                # Calculate calibration metrics
                # Bin confidences and calculate accuracy within each bin
                bins = np.linspace(0, 10, 11)  # 0-1, 1-2, ..., 9-10
                bin_indices = np.digitize(confidences, bins) - 1
                bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

                bin_accuracies = []
                bin_confidences = []
                bin_counts = []

                for i in range(len(bins) - 1):
                    mask = bin_indices == i
                    if np.sum(mask) > 0:
                        bin_acc = np.mean(accuracies[mask])
                        bin_conf = np.mean(confidences[mask]) / 10.0  # Convert to 0-1 scale
                        bin_count = np.sum(mask)

                        bin_accuracies.append(bin_acc)
                        bin_confidences.append(bin_conf)
                        bin_counts.append(bin_count)

                # Calculate Brier score (lower is better)
                brier_score = np.mean((confidences / 10.0 - accuracies) ** 2) if len(confidences) > 0 else np.nan

                # Calculate Expected Calibration Error (ECE)
                ece = 0
                total_samples = len(confidences)
                for bin_acc, bin_conf, bin_count in zip(bin_accuracies, bin_confidences, bin_counts):
                    ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

                # Calculate overconfidence/underconfidence
                mean_confidence = np.mean(confidences) / 10.0  # Convert to 0-1 scale
                mean_accuracy = np.mean(accuracies)
                calibration_bias = mean_confidence - mean_accuracy  # Positive = overconfident

                calibration_results[condition] = {
                    'correlation': correlation,
                    'correlation_p_value': p_value,
                    'brier_score': brier_score,
                    'expected_calibration_error': ece,
                    'calibration_bias': calibration_bias,
                    'mean_confidence': mean_confidence,
                    'mean_accuracy': mean_accuracy,
                    'n_samples': len(confidences),
                    'bin_data': list(zip(bin_confidences, bin_accuracies, bin_counts))
                }

        return calibration_results

    def print_calibration_analysis(self, calibration_results: Dict[str, Any]):
        """Print confidence calibration analysis."""
        print(f"\n{'='*80}")
        print(f"CONFIDENCE CALIBRATION ANALYSIS - {self.model_name} ({self.puzzle_type})")
        print(f"{'='*80}")

        print(f"\nConfidence calibration measures how well confidence ratings predict actual performance.")
        print(f"In an ideal world, post-confidence should be better calibrated than pre-confidence.")

        # Header
        header = f"{'Condition':<15} {'Correlation':<12} {'Corr_p':<8} {'Brier':<8} {'ECE':<8} {'Bias':<8} {'Interp':<15}"
        print(f"\n{header}")
        print("-" * len(header))

        # Sort by correlation (best calibrated first)
        sorted_conditions = sorted(calibration_results.items(),
                                 key=lambda x: abs(x[1]['correlation']), reverse=True)

        for condition, results in sorted_conditions:
            correlation = results['correlation']
            p_value = results['correlation_p_value']
            brier = results['brier_score']
            ece = results['expected_calibration_error']
            bias = results['calibration_bias']

            # Interpret bias
            if abs(bias) < 0.05:
                bias_interp = "well-calibrated"
            elif bias > 0:
                bias_interp = "overconfident"
            else:
                bias_interp = "underconfident"

            p_formatted = f"{p_value:.3f}" if p_value > 0.001 else "<0.001"

            print(f"{condition:<15} {correlation:<12.3f} {p_formatted:<8} {brier:<8.3f} {ece:<8.3f} {bias:<+8.3f} {bias_interp:<15}")

        # Key findings
        print(f"\n=🔍 CALIBRATION FINDINGS:")

        # Find best calibrated condition
        if calibration_results:
            best_corr_condition = max(calibration_results.items(),
                                    key=lambda x: abs(x[1]['correlation']))[0]
            best_corr_value = calibration_results[best_corr_condition]['correlation']

            lowest_ece_condition = min(calibration_results.items(),
                                     key=lambda x: x[1]['expected_calibration_error'])[0]
            lowest_ece_value = calibration_results[lowest_ece_condition]['expected_calibration_error']

            print(f"   Best correlation with performance: {best_corr_condition} (r={best_corr_value:.3f})")
            print(f"   Best calibrated (lowest ECE): {lowest_ece_condition} (ECE={lowest_ece_value:.3f})")

            # Compare pre vs post confidence
            if 'confidence-pre' in calibration_results and 'confidence-post' in calibration_results:
                pre_corr = calibration_results['confidence-pre']['correlation']
                post_corr = calibration_results['confidence-post']['correlation']
                pre_ece = calibration_results['confidence-pre']['expected_calibration_error']
                post_ece = calibration_results['confidence-post']['expected_calibration_error']

                print(f"\n   Pre-confidence correlation: {pre_corr:.3f}")
                print(f"   Post-confidence correlation: {post_corr:.3f}")

                if abs(post_corr) > abs(pre_corr):
                    print(f"   ✓ Post-confidence shows BETTER correlation (as expected)")
                else:
                    print(f"   ⚠ Pre-confidence shows better correlation (unexpected)")

                if post_ece < pre_ece:
                    print(f"   ✓ Post-confidence is BETTER calibrated (ECE: {post_ece:.3f} vs {pre_ece:.3f})")
                else:
                    print(f"   ⚠ Pre-confidence is better calibrated (ECE: {pre_ece:.3f} vs {post_ece:.3f})")

        print(f"\n   Metrics explained:")
        print(f"   • Correlation: How well confidence predicts accuracy (-1 to 1, closer to ±1 is better)")
        print(f"   • Brier Score: Prediction accuracy (0 to 1, lower is better)")
        print(f"   • ECE: Expected Calibration Error (0 to 1, lower is better)")
        print(f"   • Bias: Over/underconfidence (0 is perfect, + is overconfident, - is underconfident)")

    def format_p_value(self, p: float) -> str:
        """Format p-value with significance stars."""
        if p < 0.001:
            return f"{p:.4f}***"
        elif p < 0.01:
            return f"{p:.3f}**"
        elif p < 0.05:
            return f"{p:.3f}*"
        else:
            return f"{p:.3f}ns"

    def interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d > 0.8:
            return "large"
        elif abs_d > 0.5:
            return "medium"
        elif abs_d > 0.2:
            return "small"
        else:
            return "negligible"

    def print_summary_table(self, condition_stats: Dict, test_results: Dict, effect_sizes: Dict):
        """Print formatted summary statistics table."""
        print(f"\n{'='*80}")
        print(f"SUMMARY STATISTICS TABLE - {self.model_name} ({self.puzzle_type})")
        print(f"{'='*80}")

        # Header
        header = f"{'Condition':<15} {'Accuracy_Mean':<13} {'Accuracy_SD':<11} {'Confidence_Mean':<15} {'N':<4} {'vs_Control_p':<14} {'Effect_Size':<12}"
        print(header)
        print("-" * len(header))

        # Control row
        control_stats = condition_stats['control']
        print(f"{'control':<15} {control_stats['accuracy_mean']:<13.3f} {control_stats['accuracy_std']:<11.3f} {'N/A':<15} {control_stats['n']:<4} {'N/A':<14} {'N/A':<12}")

        # Other conditions
        for condition in ['single-shot', 'confidence-pre', 'confidence-post']:
            stats_data = condition_stats[condition]
            p_value = test_results['mann_whitney'][f"{condition}_vs_control"]['p_value']
            p_formatted = self.format_p_value(p_value)
            effect_size = effect_sizes[condition]
            conf_mean = stats_data['confidence_mean'] if stats_data['confidence_mean'] else 0.0

            print(f"{condition:<15} {stats_data['accuracy_mean']:<13.3f} {stats_data['accuracy_std']:<11.3f} {conf_mean:<15.2f} {stats_data['n']:<4} {p_formatted:<14} {effect_size:<12.3f}")

    def create_visualizations(self, condition_stats: Dict, test_results: Dict, effect_sizes: Dict, calibration_results: Dict = None):
        """Create key visualizations."""
        if calibration_results:
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Confidence Experiment Analysis - {self.model_name} ({self.puzzle_type})', fontsize=16, fontweight='bold')

        # 1. Accuracy by condition with error bars and significance stars
        conditions = list(self.conditions)
        means = [condition_stats[c]['accuracy_mean'] for c in conditions]
        stds = [condition_stats[c]['accuracy_std'] for c in conditions]

        bars = ax1.bar(conditions, means, yerr=stds, capsize=5, alpha=0.7,
                      color=['gray', 'skyblue', 'lightgreen', 'lightcoral'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy by Condition')
        ax1.set_ylim(0, 1.0)

        # Add significance stars
        max_height = max([m + s for m, s in zip(means, stds)])
        star_height = max_height + 0.05

        for i, condition in enumerate(['single-shot', 'confidence-pre', 'confidence-post']):
            p_val = test_results['mann_whitney'][f"{condition}_vs_control"]['p_value']
            if p_val < 0.05:
                stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
                ax1.text(i + 1, star_height, stars, ha='center', fontsize=12, fontweight='bold')

        # 2. Effect sizes horizontal bar chart
        effect_conditions = ['single-shot', 'confidence-pre', 'confidence-post']
        effect_values = [effect_sizes[c] for c in effect_conditions]
        colors = []
        for d in effect_values:
            abs_d = abs(d)
            if abs_d > 0.8:
                colors.append('green')
            elif abs_d > 0.5:
                colors.append('orange')
            elif abs_d > 0.2:
                colors.append('yellow')
            else:
                colors.append('gray')

        bars = ax2.barh(effect_conditions, effect_values, color=colors, alpha=0.7)
        ax2.set_xlabel("Cohen's d (Effect Size)")
        ax2.set_title('Effect Sizes vs Control')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
        ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
        ax2.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Large')
        ax2.legend()

        # 3. P-values summary
        p_values = []
        test_names = ['Kruskal-Wallis'] + [f'{c} vs Control' for c in effect_conditions]
        p_values.append(test_results['kruskal_wallis']['p_value'])
        for condition in effect_conditions:
            p_values.append(test_results['mann_whitney'][f"{condition}_vs_control"]['p_value'])

        # Color code by significance
        colors_p = ['red' if p < 0.05 else 'gray' for p in p_values]

        ax3.barh(test_names, [-np.log10(p) for p in p_values], color=colors_p, alpha=0.7)
        ax3.set_xlabel('-log10(p-value)')
        ax3.set_title('Statistical Significance')
        ax3.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax3.legend()

        # 4. Confidence vs Accuracy scatter (excluding control)
        conf_conditions = ['single-shot', 'confidence-pre', 'confidence-post']
        colors_scatter = ['skyblue', 'lightgreen', 'lightcoral']

        for i, condition in enumerate(conf_conditions):
            condition_data = self.data[self.data['condition'] == condition]
            conf_vals = condition_data['confidence'].dropna()
            acc_vals = condition_data.loc[conf_vals.index, 'correct'].astype(int)
            ax4.scatter(conf_vals, acc_vals + np.random.normal(0, 0.02, len(acc_vals)),
                       alpha=0.6, label=condition, color=colors_scatter[i])

        ax4.set_xlabel('Confidence')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Confidence vs Accuracy')
        ax4.legend()
        ax4.set_ylim(-0.1, 1.1)

        # 5. & 6. Calibration plots (if calibration results available)
        if calibration_results:
            # 5. Calibration plot (reliability diagram)
            ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

            colors = ['skyblue', 'lightgreen', 'lightcoral']
            for i, (condition, results) in enumerate(calibration_results.items()):
                bin_data = results['bin_data']
                if bin_data:
                    bin_confs, bin_accs, bin_counts = zip(*bin_data)
                    # Size points by bin count
                    sizes = [max(20, count * 3) for count in bin_counts]
                    ax5.scatter(bin_confs, bin_accs, alpha=0.7, s=sizes,
                               color=colors[i % len(colors)], label=f'{condition} (r={results["correlation"]:.2f})')

            ax5.set_xlabel('Mean Confidence')
            ax5.set_ylabel('Accuracy')
            ax5.set_title('Confidence Calibration (Reliability Diagram)')
            ax5.legend()
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
            ax5.grid(True, alpha=0.3)

            # 6. Calibration metrics comparison
            conditions = list(calibration_results.keys())
            correlations = [calibration_results[c]['correlation'] for c in conditions]
            eces = [calibration_results[c]['expected_calibration_error'] for c in conditions]

            x = np.arange(len(conditions))
            width = 0.35

            ax6_twin = ax6.twinx()
            bars1 = ax6.bar(x - width/2, [abs(c) for c in correlations], width,
                           label='|Correlation|', alpha=0.7, color='steelblue')
            bars2 = ax6_twin.bar(x + width/2, eces, width,
                                label='ECE (lower better)', alpha=0.7, color='orange')

            ax6.set_xlabel('Condition')
            ax6.set_ylabel('|Correlation|', color='steelblue')
            ax6_twin.set_ylabel('Expected Calibration Error', color='orange')
            ax6.set_title('Calibration Quality Metrics')
            ax6.set_xticks(x)
            ax6.set_xticklabels(conditions, rotation=45)

            # Add correlation values on bars
            for i, (bar, corr) in enumerate(zip(bars1, correlations)):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{corr:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        chart_file = f"analysis_charts_{self.model_name.lower().replace('-', '_')}_{self.puzzle_type.replace('-', '_').replace(' ', '_')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"📊 Charts saved to: {chart_file}")

    def print_key_findings(self, condition_stats: Dict, test_results: Dict, effect_sizes: Dict):
        """Print key findings summary."""
        print(f"\n{'='*80}")
        print(f"KEY FINDINGS - {self.model_name} ({self.puzzle_type})")
        print(f"{'='*80}")

        # Best/worst performing conditions
        accuracies = {c: condition_stats[c]['accuracy_mean'] for c in self.conditions}
        best_condition = max(accuracies, key=accuracies.get)
        worst_condition = min(accuracies, key=accuracies.get)

        print(f"=� PERFORMANCE:")
        print(f"   Best performing: {best_condition} ({accuracies[best_condition]:.3f} accuracy)")
        print(f"   Worst performing: {worst_condition} ({accuracies[worst_condition]:.3f} accuracy)")

        # Statistical significance
        kruskal_p = test_results['kruskal_wallis']['p_value']
        significant_tests = []

        for condition in ['single-shot', 'confidence-pre', 'confidence-post']:
            p_val = test_results['mann_whitney'][f"{condition}_vs_control"]['p_value']
            if p_val < 0.05:
                significant_tests.append(f"{condition} vs control (p={p_val:.3f})")

        print(f"\n=, STATISTICAL SIGNIFICANCE:")
        print(f"   Overall difference (Kruskal-Wallis): {self.format_p_value(kruskal_p)}")
        if significant_tests:
            print(f"   Significant pairwise tests:")
            for test in significant_tests:
                print(f"     • {test}")
        else:
            print(f"   No significant pairwise differences found")

        # Effect sizes
        print(f"\n=� EFFECT SIZES (vs control):")
        for condition in ['single-shot', 'confidence-pre', 'confidence-post']:
            d = effect_sizes[condition]
            interpretation = self.interpret_effect_size(d)
            direction = "higher" if d > 0 else "lower"
            print(f"   {condition}: d={d:.3f} ({interpretation}, {direction} accuracy)")

        # Overall conclusion
        print(f"\n=� CONCLUSION:")
        has_significant = len(significant_tests) > 0
        large_effects = [c for c in effect_sizes if abs(effect_sizes[c]) > 0.8]

        if has_significant and large_effects:
            print(f"   Confidence elicitation MATTERS: Found significant differences with large effects")
        elif has_significant:
            print(f"   Confidence elicitation shows SOME effect: Significant but small-medium effects")
        else:
            print(f"   Confidence elicitation appears NOT to matter: No significant differences")

    def analyze(self) -> Dict[str, Any]:
        """Run complete analysis pipeline."""
        print(f"=, Starting Analysis for {self.puzzle_type} puzzles...")

        # Load data
        self.load_data()

        print(f"=, Running Analysis for {self.model_name} {self.puzzle_type} puzzles...")

        # Calculate statistics
        condition_stats = self.get_condition_stats()
        test_results = self.run_statistical_tests(condition_stats)
        effect_sizes = self.calculate_effect_sizes(condition_stats)

        # Analyze confidence calibration
        calibration_results = self.analyze_confidence_calibration()

        # Print results
        self.print_summary_table(condition_stats, test_results, effect_sizes)
        self.print_key_findings(condition_stats, test_results, effect_sizes)
        self.print_calibration_analysis(calibration_results)

        # Create visualizations
        self.create_visualizations(condition_stats, test_results, effect_sizes, calibration_results)

        # Save results to JSON file
        summary = {
            'puzzle_type': self.puzzle_type,
            'condition_stats': condition_stats,
            'test_results': test_results,
            'effect_sizes': effect_sizes,
            'calibration_results': calibration_results,
            'best_condition': max(condition_stats, key=lambda c: condition_stats[c]['accuracy_mean']),
            'significant_differences': any(
                test_results['mann_whitney'][f"{c}_vs_control"]['p_value'] < 0.05
                for c in ['single-shot', 'confidence-pre', 'confidence-post']
            )
        }

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy_to_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_to_json(item) for item in obj]
            else:
                return obj

        # Save to file
        output_file = f"analysis_results_{self.model_name.lower().replace('-', '_')}_{self.puzzle_type.replace('-', '_').replace(' ', '_')}.json"
        with open(output_file, 'w') as f:
            import json
            json_ready_summary = convert_numpy_to_json(summary)
            json.dump(json_ready_summary, f, indent=2)
        print(f"\n📁 Results saved to: {output_file}")

        return {'summary': summary}


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Analyze confidence experiment results")
    parser.add_argument("jsonl_path", help="Path to JSONL results file")
    parser.add_argument("--puzzle-type", default="", help="Description of puzzle type")

    args = parser.parse_args()

    analyzer = ConfidenceAnalyzer(args.jsonl_path, args.puzzle_type)
    results = analyzer.analyze()

    return results


if __name__ == "__main__":
    main()