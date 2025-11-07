"""
ESG Visualization Module
Creates charts and plots for ESG analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import seaborn as sns


class ESGPlotGenerator:
    """Generate ESG visualization charts"""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """Initialize plot generator with style"""
        self.style = style
        plt.style.use('default')  # Use default style as seaborn-v0_8 may not be available
        
    def create_esg_gauge(self, esg_result: Dict) -> plt.Figure:
        """
        Create ESG score gauge chart
        
        Parameters:
        -----------
        esg_result : dict
            ESG analysis result from score_portfolio
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        overall_score = esg_result['portfolio_esg_scores']['overall']
        star_rating = esg_result['star_rating']
        star_text = esg_result['star_rating_text']
        rating_label = esg_result['rating_label']
        
        # Create gauge
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        
        # Color zones
        zones = [
            (0, 20, '#d32f2f', 'Laggard'),
            (20, 40, '#ff9800', 'Below Average'),
            (40, 60, '#ffd54f', 'Average'),
            (60, 80, '#8bc34a', 'Strong'),
            (80, 100, '#4caf50', 'Leader')
        ]
        
        # Draw zones
        for start, end, color, label in zones:
            width = end - start
            rect = mpatches.Rectangle((start, 0.3), width, 0.3, 
                                      facecolor=color, alpha=0.6, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add zone label
            ax.text(start + width/2, 0.15, label, 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Draw needle (pointer)
        needle_x = overall_score
        ax.plot([needle_x, needle_x], [0.25, 0.65], 'k-', linewidth=3)
        ax.plot(needle_x, 0.65, 'ko', markersize=10)
        
        # Add score text
        ax.text(50, 0.85, f'{overall_score:.1f}/100', 
               ha='center', va='center', fontsize=24, fontweight='bold')
        
        # Add star rating (use ASCII to avoid font warnings)
        star_ascii = '★' * star_rating  # Use filled star instead of emoji
        ax.text(50, 0.75, f'{star_ascii} ({rating_label})', 
               ha='center', va='center', fontsize=16)
        
        # Add risk multiplier
        risk_mult = esg_result.get('risk_multiplier', 1.0)
        mult_color = '#4caf50' if risk_mult < 1.0 else '#ff9800' if risk_mult > 1.0 else '#757575'
        ax.text(50, 0.05, f'Risk Multiplier: {risk_mult:.2f}x', 
               ha='center', va='center', fontsize=12, color=mult_color, fontweight='bold')
        
        ax.set_title('Portfolio ESG Score', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_esg_breakdown(self, esg_result: Dict) -> plt.Figure:
        """
        Create E-S-G component breakdown bar chart
        
        Parameters:
        -----------
        esg_result : dict
            ESG analysis result
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scores = esg_result['portfolio_esg_scores']
        
        # Data
        categories = ['Environmental', 'Social', 'Governance']
        values = [
            scores['environmental'],
            scores['social'],
            scores['governance']
        ]
        
        # Colors based on score
        colors = []
        for val in values:
            if val >= 80:
                colors.append('#4caf50')
            elif val >= 60:
                colors.append('#8bc34a')
            elif val >= 40:
                colors.append('#ffd54f')
            elif val >= 20:
                colors.append('#ff9800')
            else:
                colors.append('#d32f2f')
        
        # Create horizontal bars
        y_pos = np.arange(len(categories))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 2, i, f'{val:.1f}', va='center', fontsize=12, fontweight='bold')
        
        # Add benchmark line at 60 (good threshold)
        ax.axvline(x=60, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Good Threshold (60)')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories, fontsize=12)
        ax.set_xlabel('Score (0-100)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 105)
        ax.set_title('ESG Component Breakdown', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_sector_esg_heatmap(self, esg_result: Dict) -> plt.Figure:
        """
        Create sector ESG heatmap
        
        Parameters:
        -----------
        esg_result : dict
            ESG analysis result
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        sector_breakdown = esg_result.get('sector_breakdown', {})
        
        if not sector_breakdown:
            # Return empty figure with message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No sector data available', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Prepare data
        sectors = list(sector_breakdown.keys())
        metrics = ['Environmental', 'Social', 'Governance', 'Overall ESG']
        
        data = []
        for sector in sectors:
            sector_data = sector_breakdown[sector]
            data.append([
                sector_data['avg_env'],
                sector_data['avg_social'],
                sector_data['avg_gov'],
                sector_data['avg_esg_score']
            ])
        
        # Create DataFrame for heatmap
        df = pd.DataFrame(data, columns=metrics, index=sectors)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, len(sectors) * 0.6)))
        
        # Create heatmap
        sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', 
                   vmin=0, vmax=100, cbar_kws={'label': 'ESG Score'},
                   linewidths=1, linecolor='white', ax=ax)
        
        ax.set_title('ESG Scores by Sector', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('ESG Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sector', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_holdings_esg_table(self, esg_result: Dict) -> pd.DataFrame:
        """
        Create detailed holdings ESG table
        
        Parameters:
        -----------
        esg_result : dict
            ESG analysis result
            
        Returns:
        --------
        pd.DataFrame
        """
        holdings = esg_result.get('holdings_detail', [])
        
        if not holdings:
            return pd.DataFrame({'Message': ['No holdings data available']})
        
        # Create table data
        table_data = []
        for holding in holdings:
            table_data.append({
                'Stock': holding['stock_name'],
                'Sector': holding['sector'],
                'E': f"{holding['environmental_score']:.0f}",
                'S': f"{holding['social_score']:.0f}",
                'G': f"{holding['governance_score']:.0f}",
                'Overall': f"{holding['overall_esg_score']:.2f}",
                'Rating': holding['star_rating_text'],
                'Label': holding['rating_label'],
                'Weight': f"{holding['weight']:.1f}%",
                'Source': 'Real' if not holding['is_proxy'] else 'Proxy'
            })
        
        df = pd.DataFrame(table_data)
        
        # Sort by overall ESG score descending
        df['_sort'] = df['Overall'].astype(float)
        df = df.sort_values('_sort', ascending=False).drop('_sort', axis=1)
        
        return df
    
    def create_coverage_donut(self, esg_result: Dict) -> plt.Figure:
        """
        Create ESG data coverage donut chart
        
        Parameters:
        -----------
        esg_result : dict
            ESG analysis result
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use new data_sources breakdown (API, ML, Proxy)
        data_sources = esg_result.get('data_sources', {})
        api_data = data_sources.get('api_data', 0)
        ml_predictions = data_sources.get('ml_predictions', 0)
        sector_proxy = data_sources.get('sector_proxy', 0)
        coverage_pct = esg_result.get('coverage_percentage', 0)
        
        # Sanitize values - handle None, NaN, or invalid numbers
        import numpy as np
        api_data = 0 if api_data is None or (isinstance(api_data, float) and np.isnan(api_data)) else int(api_data)
        ml_predictions = 0 if ml_predictions is None or (isinstance(ml_predictions, float) and np.isnan(ml_predictions)) else int(ml_predictions)
        sector_proxy = 0 if sector_proxy is None or (isinstance(sector_proxy, float) and np.isnan(sector_proxy)) else int(sector_proxy)
        coverage_pct = 0 if coverage_pct is None or (isinstance(coverage_pct, float) and np.isnan(coverage_pct)) else float(coverage_pct)
        
        # Build data for pie chart (only non-zero slices)
        sizes = []
        labels = []
        colors = []
        explode_vals = []
        
        if api_data > 0:
            sizes.append(api_data)
            labels.append(f'Real API Data\n({api_data} holdings)')
            colors.append('#4caf50')  # Green
            explode_vals.append(0.05)
        
        if ml_predictions > 0:
            sizes.append(ml_predictions)
            labels.append(f'ML Predictions\n({ml_predictions} holdings)')
            colors.append('#2196f3')  # Blue
            explode_vals.append(0.05)
        
        if sector_proxy > 0:
            sizes.append(sector_proxy)
            labels.append(f'Sector Proxy\n({sector_proxy} holdings)')
            colors.append('#ff9800')  # Orange
            explode_vals.append(0.05)
        
        # If no data, create placeholder
        if not sizes:
            sizes = [1]
            labels = ['No Data\n(0 holdings)']
            colors = ['#cccccc']
            explode_vals = [0]
        
        explode = tuple(explode_vals)
        
        # Create donut
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, explode=explode, textprops={'fontsize': 12})
        
        # Make it a donut
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)
        
        # Add coverage text in center
        ax.text(0, 0, f'{coverage_pct:.1f}%\nCoverage', 
               ha='center', va='center', fontsize=20, fontweight='bold')
        
        ax.set_title('ESG Data Coverage', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def create_esg_star_distribution(self, esg_result: Dict) -> plt.Figure:
        """
        Create star rating distribution chart
        
        Parameters:
        -----------
        esg_result : dict
            ESG analysis result
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        holdings = esg_result.get('holdings_detail', [])
        
        # Count by star rating
        star_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for holding in holdings:
            stars = holding['star_rating']
            star_counts[stars] += 1
        
        # Data
        stars = list(star_counts.keys())
        counts = list(star_counts.values())
        labels = ['1★', '2★', '3★', '4★', '5★']  # Use ASCII star to avoid font warnings
        colors = ['#d32f2f', '#ff9800', '#ffd54f', '#8bc34a', '#4caf50']
        
        # Create bars
        bars = ax.bar(stars, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_xticks(stars)
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_xlabel('Star Rating', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Holdings', fontsize=12, fontweight='bold')
        ax.set_title('ESG Star Rating Distribution', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, max(counts) + 1 if max(counts) > 0 else 5)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_portfolio_shap_waterfall(self, esg_result: Dict, top_n: int = 15) -> plt.Figure:
        """
        Create portfolio-level SHAP waterfall chart showing feature contributions
        
        Parameters:
        -----------
        esg_result : dict
            ESG analysis result with portfolio_shap_analysis
        top_n : int
            Number of top features to display (default: 15)
            
        Returns:
        --------
        matplotlib.figure.Figure or None
        """
        shap_analysis = esg_result.get('portfolio_shap_analysis')
        
        if shap_analysis is None:
            print("⚠️ No SHAP analysis available in ESG result")
            return None
        
        feature_contributions = shap_analysis.get('feature_contributions', [])
        if not feature_contributions:
            print("⚠️ No feature contributions found in SHAP analysis")
            return None
        
        # Take top N features
        top_features = feature_contributions[:top_n]
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(top_features) * 0.4)))
        
        # Extract data
        features = [f['feature'] for f in top_features]
        contributions = [f['contribution'] for f in top_features]
        
        # Clean feature names for display
        display_names = []
        for f in features:
            # Simplify feature names
            name = f.replace('_', ' ').replace('Score', '').title()
            if 'Sector' in name:
                name = name.split('Sector')[1].strip() if 'Sector' in name else name
            display_names.append(name[:30])  # Truncate long names
        
        # Create colors based on contribution direction
        colors = ['#4caf50' if c > 0 else '#f44336' for c in contributions]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(display_names))
        bars = ax.barh(y_pos, contributions, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_names)
        ax.set_xlabel('SHAP Contribution to Portfolio ESG Score', fontsize=11, fontweight='bold')
        ax.set_title(
            f'Portfolio-Level SHAP Analysis\n'
            f'Top {len(top_features)} Features Driving Portfolio ESG',
            fontsize=13,
            fontweight='bold',
            pad=15
        )
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, contributions)):
            label_x = val + (max(contributions) * 0.02 if val >= 0 else min(contributions) * 0.02)
            ha = 'left' if val >= 0 else 'right'
            ax.text(
                label_x, i, f'{val:+.3f}',
                ha=ha, va='center', fontweight='bold', fontsize=9
            )
        
        # Add reference line at zero
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        
        # Add summary statistics
        base = shap_analysis.get('portfolio_base', 0)
        final = shap_analysis.get('portfolio_prediction', 0)
        coverage = shap_analysis.get('coverage_weight', 0) * 100
        
        summary_text = (
            f'Base Value: {base:.2f} | '
            f'Final Prediction: {final:.2f} | '
            f'ML Coverage: {coverage:.1f}%'
        )
        ax.text(
            0.5, -0.08, summary_text,
            ha='center', va='top',
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
        
        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Legend
        positive_patch = mpatches.Patch(color='#4caf50', label='Positive Impact')
        negative_patch = mpatches.Patch(color='#f44336', label='Negative Impact')
        ax.legend(handles=[positive_patch, negative_patch], loc='lower right', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def create_shap_grouped_bars(self, esg_result: Dict) -> plt.Figure:
        """
        Create grouped bar chart for SHAP contributions by category
        
        Parameters:
        -----------
        esg_result : dict
            ESG analysis result with portfolio_shap_analysis
            
        Returns:
        --------
        matplotlib.figure.Figure or None
        """
        shap_analysis = esg_result.get('portfolio_shap_analysis')
        
        if shap_analysis is None or 'grouped_contributions' not in shap_analysis:
            print("⚠️ No grouped SHAP analysis available")
            return None
        
        grouped = shap_analysis['grouped_contributions']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        groups = [g['group'] for g in grouped]
        contributions = [g['contribution'] for g in grouped]
        
        # Create colors
        colors = ['#4caf50' if c > 0 else '#f44336' for c in contributions]
        
        # Create bar chart
        bars = ax.bar(groups, contributions, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Customize
        ax.set_ylabel('Contribution to Portfolio ESG', fontsize=11, fontweight='bold')
        ax.set_title(
            'Portfolio ESG Drivers by Feature Group',
            fontsize=13,
            fontweight='bold',
            pad=15
        )
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        
        # Rotate x labels if needed
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, contributions):
            height = bar.get_height()
            label_y = height + (max(contributions) * 0.05 if height >= 0 else min(contributions) * 0.05)
            ax.text(
                bar.get_x() + bar.get_width() / 2, label_y,
                f'{val:+.3f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontweight='bold', fontsize=10
            )
        
        # Grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig


# Test function
if __name__ == "__main__":
    # Create sample ESG result for testing
    sample_result = {
        'portfolio_esg_scores': {
            'environmental': 75.0,
            'social': 80.0,
            'governance': 85.0,
            'overall': 79.5
        },
        'star_rating': 4,
        'star_rating_text': '⭐⭐⭐⭐',
        'rating_label': 'Strong',
        'risk_multiplier': 0.90,
        'coverage_percentage': 85.0,
        'rated_holdings': 4,
        'unrated_holdings': 1,
        'sector_breakdown': {
            'IT Services': {'avg_env': 80, 'avg_social': 85, 'avg_gov': 82, 'avg_esg_score': 82.5},
            'Banking': {'avg_env': 70, 'avg_social': 75, 'avg_gov': 88, 'avg_esg_score': 76.8}
        },
        'holdings_detail': [
            {'stock_name': 'TCS', 'sector': 'IT', 'environmental_score': 85, 'social_score': 88, 
             'governance_score': 90, 'overall_esg_score': 87.5, 'star_rating': 5, 
             'star_rating_text': '⭐⭐⭐⭐⭐', 'rating_label': 'Leader', 'weight': 30.0, 'is_proxy': False},
            {'stock_name': 'HDFC Bank', 'sector': 'Banking', 'environmental_score': 70, 'social_score': 75, 
             'governance_score': 88, 'overall_esg_score': 76.5, 'star_rating': 4, 
             'star_rating_text': '⭐⭐⭐⭐', 'rating_label': 'Strong', 'weight': 40.0, 'is_proxy': False},
        ]
    }
    
    plotter = ESGPlotGenerator()
    
    print("Testing ESG Plot Generator...")
    
    # Test gauge
    fig = plotter.create_esg_gauge(sample_result)
    print("✅ Gauge chart created")
    plt.close(fig)
    
    # Test breakdown
    fig = plotter.create_esg_breakdown(sample_result)
    print("✅ Breakdown chart created")
    plt.close(fig)
    
    # Test heatmap
    fig = plotter.create_sector_esg_heatmap(sample_result)
    print("✅ Heatmap created")
    plt.close(fig)
    
    # Test table
    df = plotter.create_holdings_esg_table(sample_result)
    print("✅ Holdings table created")
    print(df)
    
    print("\n✅ All ESG visualizations working!")
