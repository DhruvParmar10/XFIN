import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

class StressPlotGenerator:
    """Generate visualizations for portfolio stress testing results"""
    
    def __init__(self):
        # Set up matplotlib style
        plt.style.use('default')
        self.colors = {
            'low': '#2E8B57',      # Sea Green
            'medium': '#FF8C00',   # Dark Orange  
            'high': '#DC143C',     # Crimson
            'extreme': '#8B0000',  # Dark Red
            'neutral': '#4682B4'   # Steel Blue
        }
    
    def get_risk_color(self, impact_percentage):
        """Get color based on impact percentage"""
        abs_impact = abs(impact_percentage)
        if abs_impact >= 25:
            return self.colors['extreme']
        elif abs_impact >= 15:
            return self.colors['high']
        elif abs_impact >= 8:
            return self.colors['medium']
        else:
            return self.colors['low']
    
    def plot_single(self, impact_percentage, scenario_name):
        """Create a single scenario impact plot"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create horizontal bar chart
            impact_value = float(impact_percentage)
            color = self.get_risk_color(impact_value)
            
            # Create the bar
            bar = ax.barh(['Portfolio Impact'], [impact_value], 
                         color=color, alpha=0.7, height=0.4)
            
            # Customize the plot
            ax.set_xlabel('Impact (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'{scenario_name} Impact', fontsize=14, fontweight='bold', pad=20)
            
            # Set x-axis limits
            max_val = max(abs(impact_value) * 1.2, 30)
            ax.set_xlim(-max_val, 5)
            
            # Add value label on the bar
            ax.text(impact_value/2, 0, f'{impact_value:.1f}%', 
                   ha='center', va='center', fontweight='bold', fontsize=12, color='white')
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_axisbelow(True)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add zero line
            ax.axvline(x=0, color='black', linewidth=1, alpha=0.5)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            # Fallback: simple plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f'Impact: {impact_percentage:.1f}%\nScenario: {scenario_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
    
    def plot_multiple(self, comparison_df):
        """Create multiple scenario comparison plots"""
        try:
            if comparison_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'No data available for comparison', 
                       ha='center', va='center', transform=ax.transAxes)
                return fig
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Impact Comparison
            scenarios = comparison_df['scenario'].values
            impacts = comparison_df['impact_percentage'].values
            colors = [self.get_risk_color(impact) for impact in impacts]
            
            bars1 = ax1.barh(scenarios, impacts, color=colors, alpha=0.7)
            ax1.set_xlabel('Impact (%)', fontweight='bold')
            ax1.set_title('Scenario Impact Comparison', fontweight='bold', pad=20)
            ax1.grid(True, alpha=0.3, axis='x')
            ax1.set_axisbelow(True)
            
            # Add value labels
            for bar, impact in zip(bars1, impacts):
                width = bar.get_width()
                ax1.text(width/2, bar.get_y() + bar.get_height()/2, f'{impact:.1f}%',
                        ha='center', va='center', fontweight='bold', color='white')
            
            # Plot 2: Recovery Time Comparison  
            if 'recovery_months' in comparison_df.columns:
                recovery_times = comparison_df['recovery_months'].values
                bars2 = ax2.bar(scenarios, recovery_times, color=self.colors['neutral'], alpha=0.7)
                ax2.set_ylabel('Recovery Time (Months)', fontweight='bold')
                ax2.set_title('Recovery Time Comparison', fontweight='bold', pad=20)
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.set_axisbelow(True)
                
                # Rotate x-axis labels for better readability
                ax2.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, time in zip(bars2, recovery_times):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2, height/2, f'{time:.0f}',
                            ha='center', va='center', fontweight='bold', color='white')
            
            # Remove spines
            for ax in [ax1, ax2]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            # Fallback plot
            fig, ax = plt.subplots(figsize=(10, 6))
            if not comparison_df.empty:
                # Simple bar chart as fallback
                ax.bar(comparison_df['scenario'], comparison_df['impact_percentage'])
                ax.set_ylabel('Impact (%)')
                ax.set_title('Scenario Comparison')
                plt.xticks(rotation=45)
            else:
                ax.text(0.5, 0.5, 'No comparison data available', 
                       ha='center', va='center', transform=ax.transAxes)
            plt.tight_layout()
            return fig
    
    def create_risk_heatmap(self, scenarios_data):
        """Create a risk heatmap for multiple scenarios"""
        try:
            if not scenarios_data:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, 'No data for heatmap', ha='center', va='center')
                return fig
            
            # Prepare data for heatmap
            df = pd.DataFrame(scenarios_data)
            
            # Create risk matrix
            risk_levels = {'Low': 1, 'Medium': 2, 'High': 3, 'Extreme': 4}
            df['risk_numeric'] = df['risk_level'].map(risk_levels)
            
            # Create pivot table
            pivot_data = df.pivot_table(
                values='impact_percentage', 
                index='scenario', 
                columns='risk_level', 
                fill_value=0
            )
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                       center=0, ax=ax, cbar_kws={'label': 'Impact (%)'})
            
            ax.set_title('Risk Assessment Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            return fig
            
        except Exception as e:
            # Simple fallback
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Heatmap unavailable: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def create_scenario_comparison(self, scenario_results):
        """Create comprehensive scenario comparison visualization"""
        try:
            if not scenario_results:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'No scenario data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return fig
            
            # Convert to DataFrame
            df = pd.DataFrame(scenario_results)
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # Plot 1: Impact comparison
            ax1 = fig.add_subplot(gs[0, 0])
            impacts = df['impact_percentage']
            scenarios = df['scenario']
            colors = [self.get_risk_color(imp) for imp in impacts]
            
            bars = ax1.barh(scenarios, impacts, color=colors, alpha=0.7)
            ax1.set_xlabel('Impact (%)')
            ax1.set_title('Portfolio Impact by Scenario')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Recovery time
            ax2 = fig.add_subplot(gs[0, 1])
            if 'recovery_months' in df.columns:
                ax2.bar(scenarios, df['recovery_months'], color=self.colors['neutral'], alpha=0.7)
                ax2.set_ylabel('Recovery (Months)')
                ax2.set_title('Expected Recovery Time')
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot 3: Risk distribution
            ax3 = fig.add_subplot(gs[1, :])
            if 'risk_level' in df.columns:
                risk_counts = df['risk_level'].value_counts()
                wedges, texts, autotexts = ax3.pie(risk_counts.values, labels=risk_counts.index, 
                                                  autopct='%1.1f%%', startangle=90)
                ax3.set_title('Risk Level Distribution')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            # Fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Comprehensive chart unavailable: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def plot_value_at_risk(self, var_data, confidence_levels=[95, 99]):
        """Plot Value at Risk (VaR) visualization"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if isinstance(var_data, dict):
                scenarios = list(var_data.keys())
                var_values = list(var_data.values())
            else:
                # Assume it's a single value
                scenarios = ['Portfolio']
                var_values = [var_data]
            
            colors = [self.get_risk_color(var) for var in var_values]
            bars = ax.bar(scenarios, var_values, color=colors, alpha=0.7)
            
            ax.set_ylabel('Value at Risk (%)')
            ax.set_title('Portfolio Value at Risk (95% Confidence)', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, var_val in zip(bars, var_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                       f'{var_val:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'VaR plot unavailable: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig