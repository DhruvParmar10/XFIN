import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle, Wedge, FancyBboxPatch
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
from math import pi, cos, sin

# Import market data service for enhanced functionality
try:
    from .market_data import MarketDataService
    from .config import XFINConfig
    MARKET_DATA_AVAILABLE = True
except ImportError:
    MARKET_DATA_AVAILABLE = False

class StressPlotGenerator:
    """Generate meaningful, actionable visualizations for portfolio stress testing results"""
    
    def __init__(self, config: XFINConfig = None, market_data_service: MarketDataService = None):
        """
        Initialize StressPlotGenerator with flexible configuration
        
        Parameters:
        -----------
        config : XFINConfig, optional
            XFIN configuration object
        market_data_service : MarketDataService, optional
            Market data service for live data integration
        """
        self.config = config or XFINConfig()
        
        # Initialize market data service if available and not provided
        if MARKET_DATA_AVAILABLE and market_data_service is None:
            try:
                market_config = self.config.get_market_data_config()
                self.market_data_service = MarketDataService(market_config)
                self.use_live_data = True
            except Exception:
                self.market_data_service = None
                self.use_live_data = False
        else:
            self.market_data_service = market_data_service
            self.use_live_data = market_data_service is not None
        
        # Set up matplotlib style based on configuration
        chart_config = self.config.get_chart_config()
        
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'sans-serif',
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'figure.dpi': chart_config.get('figure_dpi', 100)
        })
        
        # Set grid alpha separately since it's not a direct rcParam
        plt.rc('axes', grid=True)
        plt.rc('grid', alpha=0.3)
        
        # Store default figure size from config
        self.default_figsize = chart_config.get('default_figure_size', (12, 8))
        
        # Professional color palette
        self.colors = {
            'low': '#2E8B57',        # Sea Green
            'medium': '#FF8C00',     # Dark Orange  
            'high': '#DC143C',       # Crimson
            'extreme': '#8B0000',    # Dark Red
            'neutral': '#4682B4',    # Steel Blue
            'positive': '#228B22',   # Forest Green
            'negative': '#B22222',   # Fire Brick
            'warning': '#FFA500',    # Orange
            'info': '#4169E1'        # Royal Blue
        }
        
        # Risk level thresholds
        self.risk_thresholds = {
            'low': 8,
            'medium': 15, 
            'high': 25,
            'extreme': float('inf')
        }
    
    def get_risk_color(self, impact_percentage: float) -> str:
        """Get color based on impact percentage"""
        abs_impact = abs(impact_percentage)
        if abs_impact >= self.risk_thresholds['high']:
            return self.colors['extreme']
        elif abs_impact >= self.risk_thresholds['medium']:
            return self.colors['high']
        elif abs_impact >= self.risk_thresholds['low']:
            return self.colors['medium']
        else:
            return self.colors['low']
    
    def get_risk_level(self, impact_percentage: float) -> str:
        """Get risk level classification"""
        abs_impact = abs(impact_percentage)
        if abs_impact >= self.risk_thresholds['high']:
            return 'Extreme Risk'
        elif abs_impact >= self.risk_thresholds['medium']:
            return 'High Risk'
        elif abs_impact >= self.risk_thresholds['low']:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    # ===== NEW MEANINGFUL CHART TYPES =====
    
    def create_portfolio_composition_pie(self, portfolio_data: pd.DataFrame, title: str = "Portfolio Composition") -> plt.Figure:
        """
        1. Portfolio Composition Pie Chart - Shows actual portfolio allocation by holdings
        Replaces the useless donut chart with meaningful asset allocation visualization
        """
        try:
            if portfolio_data.empty:
                return self._create_error_chart("No portfolio data available")
            
            # Get value column and stock names
            value_col = self._get_value_column(portfolio_data)
            name_col = self._get_stock_name_column(portfolio_data)
            
            if not value_col or not name_col:
                return self._create_error_chart("Missing required columns (value/name)")
            
            # Calculate portfolio values and percentages
            portfolio_values = portfolio_data[value_col].astype(float)
            portfolio_names = portfolio_data[name_col].astype(str)
            total_value = portfolio_values.sum()
            
            if total_value <= 0:
                return self._create_error_chart("Invalid portfolio values")
            
            # Create holding percentages
            holding_data = pd.DataFrame({
                'name': portfolio_names,
                'value': portfolio_values,
                'percentage': (portfolio_values / total_value) * 100
            }).sort_values('percentage', ascending=False)
            
            # Show top 8 holdings + "Others" category
            top_holdings = holding_data.head(8).copy()
            others_value = holding_data.tail(len(holding_data) - 8)['percentage'].sum()
            
            if others_value > 0:
                others_row = pd.DataFrame({
                    'name': ['Others'],
                    'value': [others_value * total_value / 100],
                    'percentage': [others_value]
                })
                plot_data = pd.concat([top_holdings, others_row], ignore_index=True)
            else:
                plot_data = top_holdings
            
            # Create the pie chart - Optimized for column layout
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Generate colors
            colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
            
            # Create pie chart with minimal overlapping
            wedges, texts, autotexts = ax.pie(
                plot_data['percentage'],
                labels=None,  # No direct labels to avoid overlap
                autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',  # Only show % for larger slices
                startangle=90,
                colors=colors,
                explode=[0.08 if x == 'Others' else 0.03 for x in plot_data['name']],
                shadow=True,
                wedgeprops=dict(width=0.9, edgecolor='white', linewidth=2),
                pctdistance=0.75,
                textprops={'fontsize': 8, 'fontweight': 'bold'}
            )
            
            # Create a legend instead of overlapping labels
            legend_labels = []
            for i, (name, pct, value) in enumerate(zip(plot_data['name'], plot_data['percentage'], plot_data['value'])):
                # Truncate long names for legend
                short_name = name[:15] + '...' if len(name) > 15 else name
                legend_labels.append(f'{short_name}: {pct:.1f}% (₹{value:,.0f})')
            
            # Position legend to the right
            ax.legend(wedges, legend_labels, 
                     title="Holdings",
                     loc="center left",
                     bbox_to_anchor=(1, 0, 0.5, 1),
                     fontsize=8,
                     title_fontsize=10)
            
            # Enhance percentage text inside pie
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(8)
                autotext.set_fontweight('bold')
            
            # Add concentration risk warning
            max_holding_pct = plot_data['percentage'].max()
            if max_holding_pct > 20:
                ax.text(0, -1.3, f"⚠️ Concentration Risk: Largest holding is {max_holding_pct:.1f}%", 
                       ha='center', fontsize=11, color=self.colors['warning'], fontweight='bold')
            
            ax.set_title(f"{title}\nTotal Value: ${total_value:,.0f}", 
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Portfolio composition error: {str(e)}")
    
    def create_risk_gauge_chart(self, stress_result: Dict, title: str = "Portfolio Risk Assessment") -> plt.Figure:
        """
        2. Risk Gauge Chart - Visual risk level assessment with color-coded zones
        Replaces the single-scenario bar chart with intuitive risk gauge
        """
        try:
            impact_pct = abs(float(stress_result.get('impact_percentage', 0)))
            risk_level = self.get_risk_level(impact_pct)
            
            # Create figure with two subplots: gauge and legend side by side
            fig = plt.figure(figsize=(12, 6))
            
            # Gauge subplot (left side)
            ax_gauge = fig.add_subplot(121, projection='polar')
            
            # Define gauge parameters
            theta_max = pi  # Half circle
            gauge_radius = 1.0
            
            # Risk zones (in degrees, converted to radians)
            zones = [
                (0, self.risk_thresholds['low'], self.colors['low'], 'Low Risk'),
                (self.risk_thresholds['low'], self.risk_thresholds['medium'], self.colors['medium'], 'Medium Risk'),
                (self.risk_thresholds['medium'], self.risk_thresholds['high'], self.colors['high'], 'High Risk'),
                (self.risk_thresholds['high'], 35, self.colors['extreme'], 'Extreme Risk')
            ]
            
            # Draw risk zones
            for i, (start, end, color, label) in enumerate(zones):
                start_angle = pi - (start / 35) * pi
                end_angle = pi - (min(end, 35) / 35) * pi
                
                theta = np.linspace(start_angle, end_angle, 100)
                ax_gauge.fill_between(theta, 0.7, gauge_radius, color=color, alpha=0.7, label=label)
            
            # Calculate needle position
            needle_angle = pi - (min(impact_pct, 35) / 35) * pi
            
            # Draw needle
            needle_length = 0.9
            ax_gauge.plot([needle_angle, needle_angle], [0, needle_length], 
                   color='black', linewidth=4, zorder=10)
            ax_gauge.plot(needle_angle, needle_length * 0.95, 'o', 
                   color='red', markersize=8, zorder=10)
            
            # Customize the gauge
            ax_gauge.set_ylim(0, gauge_radius)
            ax_gauge.set_xlim(0, pi)
            ax_gauge.set_theta_zero_location('W')
            ax_gauge.set_theta_direction(1)
            
            # Remove default labels and ticks
            ax_gauge.set_rticks([])
            ax_gauge.set_thetagrids([])
            ax_gauge.grid(False)
            ax_gauge.spines['polar'].set_visible(False)
            
            # Add risk level text
            ax_gauge.text(pi/2, 0.4, f"{impact_pct:.1f}%", ha='center', va='center', 
                   fontsize=20, fontweight='bold', color='black')
            ax_gauge.text(pi/2, 0.25, risk_level, ha='center', va='center', 
                   fontsize=14, fontweight='bold', color=self.get_risk_color(impact_pct))
            
            # Legend subplot (right side)
            ax_legend = fig.add_subplot(122)
            ax_legend.axis('off')  # Hide axes
            
            # Create custom legend with risk information
            legend_y_positions = [0.8, 0.6, 0.4, 0.2]
            legend_text = [
                f"● Low Risk (0-{self.risk_thresholds['low']:.0f}%)",
                f"● Medium Risk ({self.risk_thresholds['low']:.0f}-{self.risk_thresholds['medium']:.0f}%)", 
                f"● High Risk ({self.risk_thresholds['medium']:.0f}-{self.risk_thresholds['high']:.0f}%)",
                f"● Extreme Risk (>{self.risk_thresholds['high']:.0f}%)"
            ]
            
            # Add legend title
            ax_legend.text(0.1, 0.95, "Risk Level Guide", fontsize=16, fontweight='bold')
            
            # Add legend items
            for i, (y_pos, text, (_, _, color, _)) in enumerate(zip(legend_y_positions, legend_text, zones)):
                # Color square
                ax_legend.add_patch(plt.Rectangle((0.1, y_pos - 0.03), 0.05, 0.06, 
                                                facecolor=color, alpha=0.7))
                # Text
                ax_legend.text(0.2, y_pos, text, fontsize=12, va='center')
            
            # Add current assessment
            ax_legend.text(0.1, 0.05, f"Current Assessment: {impact_pct:.1f}% ({risk_level})", 
                          fontsize=14, fontweight='bold', 
                          color=self.get_risk_color(impact_pct))
            
            # Set limits for legend subplot
            ax_legend.set_xlim(0, 1)
            ax_legend.set_ylim(0, 1)
            
            # Add title to the overall figure
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Risk gauge error: {str(e)}")
    
    def create_sector_exposure_vs_impact_scatter(self, portfolio_data: pd.DataFrame, 
                                               stress_result: Dict, 
                                               title: str = "Sector Risk Analysis") -> plt.Figure:
        """
        3. Sector Exposure vs Impact Scatter Plot - Identifies high-exposure, high-risk sectors
        New addition to show which sectors are both highly exposed AND high-risk
        """
        try:
            if portfolio_data.empty:
                return self._create_error_chart("No portfolio data available")
            
            # Get live market data if available
            live_market_data = {}
            data_quality_info = ""
            
            if self.use_live_data and self.market_data_service:
                try:
                    name_col = self._get_stock_name_column(portfolio_data)
                    if name_col:
                        symbols = portfolio_data[name_col].tolist()
                        live_market_data = self.market_data_service.get_market_data(symbols)
                        
                        # Get data quality summary
                        quality_summary = self.market_data_service.get_data_quality_summary(live_market_data)
                        data_quality_info = f" (Live: {quality_summary['high_quality']}/{quality_summary['total_symbols']})"
                        
                except Exception as e:
                    print(f"Market data fetch failed: {e}")
            
            # Try enhanced sector analysis first (it has better keyword matching)
            sector_analysis = self._analyze_sector_composition(portfolio_data)
            
            # Only use live data method if the basic one fails
            if not sector_analysis:
                sector_analysis = self._analyze_sector_composition_with_live_data(portfolio_data, live_market_data)
                
            if not sector_analysis:
                return self._create_error_chart("No sector data available for analysis.\nTip: Ensure your CSV has stock names and values.")
            
            # Mock sector impact data (in production, this would come from the stress engine)
            sector_impacts = self._get_sector_impacts(stress_result.get('scenario_name', 'market_correction'))
            
            # Prepare scatter plot data
            sectors = []
            exposures = []
            impacts = []
            weights = []
            
            for sector, data in sector_analysis.items():
                # Try direct match first, then fallback to 'Other' category
                if sector in sector_impacts:
                    impact_value = sector_impacts[sector]
                elif 'Other' in sector_impacts:
                    impact_value = sector_impacts['Other']
                else:
                    impact_value = -0.10  # Default 10% impact
                
                sectors.append(sector)
                exposures.append(data['weight'] * 100)  # Convert to percentage
                impacts.append(abs(impact_value) * 100)  # Expected impact
                weights.append(data['value'])
            
            if not sectors:
                return self._create_error_chart("No sector data available for analysis")
            
            # Create scatter plot with improved layout and sizing  
            fig, ax = plt.subplots(figsize=(14, 10))  # Larger figure size
            
            # Normalize weights for bubble sizes - make them more prominent but not too large
            weight_array = np.array(weights)
            bubble_sizes = (weight_array / weight_array.max()) * 600 + 150  # Reduced max size
            
            # Create scatter plot with bubbles
            scatter = ax.scatter(impacts, exposures, s=bubble_sizes, alpha=0.7, 
                               c=[self.get_risk_color(impact) for impact in impacts],
                               edgecolors='black', linewidth=1)
            
            # Add sector labels with improved positioning to avoid overlap
            for i, sector in enumerate(sectors):
                # Shorten long sector names for better display
                display_name = sector[:15] + '...' if len(sector) > 15 else sector
                
                # Offset text more to reduce overlap
                offset_x = 15 if i % 2 == 0 else -15
                offset_y = 15 if i % 3 == 0 else -15
                
                ax.annotate(display_name, (impacts[i], exposures[i]), 
                           xytext=(offset_x, offset_y), textcoords='offset points',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
                           ha='center', va='center',
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', alpha=0.5))
            
            # Add quadrant lines with improved positioning
            if len(exposures) > 1:
                median_exposure = np.median(exposures)
                median_impact = np.median(impacts)
                
                # Add margin to the axes limits for better label placement
                x_min, x_max = min(impacts), max(impacts)
                y_min, y_max = min(exposures), max(exposures)
                x_margin = (x_max - x_min) * 0.1
                y_margin = (y_max - y_min) * 0.1
                
                ax.set_xlim(x_min - x_margin, x_max + x_margin)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                
                # Draw quadrant lines
                ax.axhline(y=median_exposure, color='gray', linestyle='--', alpha=0.7, linewidth=2)
                ax.axvline(x=median_impact, color='gray', linestyle='--', alpha=0.7, linewidth=2)
                
                # Position quadrant labels better with proper spacing
                x_left = x_min + (median_impact - x_min) * 0.1
                x_right = median_impact + (x_max - median_impact) * 0.9
                y_bottom = y_min + (median_exposure - y_min) * 0.1
                y_top = median_exposure + (y_max - median_exposure) * 0.9
                
                # Add quadrant labels with better colors and positioning
                ax.text(x_left, y_top, 'High Exposure\nLow Risk', 
                       fontsize=12, ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.7', facecolor='lightblue', alpha=0.9, edgecolor='blue'))
                ax.text(x_right, y_top, 'High Exposure\nHigh Risk', 
                       fontsize=12, ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.7', facecolor='lightcoral', alpha=0.9, edgecolor='red'))
                ax.text(x_left, y_bottom, 'Low Exposure\nLow Risk', 
                       fontsize=12, ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.7', facecolor='lightgreen', alpha=0.9, edgecolor='green'))
                ax.text(x_right, y_bottom, 'Low Exposure\nHigh Risk', 
                       fontsize=12, ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.7', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))
            
            # Improved axis labels and title
            ax.set_xlabel('Expected Impact (%)', fontweight='bold', fontsize=14, labelpad=10)
            ax.set_ylabel('Portfolio Exposure (%)', fontweight='bold', fontsize=14, labelpad=10)  
            ax.set_title(f"{title}\n(Bubble size represents portfolio weight)", fontweight='bold', fontsize=16, pad=25)
            
            # Add subtle grid
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Improve tick labels
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Add sector legend to identify all sectors in the plot
            unique_sectors = list(set(sectors))
            legend_elements = []
            for i, sector in enumerate(unique_sectors):
                color = self.get_risk_color(impacts[sectors.index(sector)])
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=10, 
                                                label=sector, markeredgecolor='black'))
            
            # Position legend outside the plot area
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                     fontsize=10, title="Sectors", title_fontsize=12, frameon=True,
                     fancybox=True, shadow=True)
            
            # Use subplots_adjust to make room for legend
            plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Sector analysis error: {str(e)}")
    
    def create_recovery_timeline_chart(self, stress_result: Dict, 
                                     title: str = "Recovery Timeline") -> plt.Figure:
        """
        4. Recovery Timeline Chart - Shows expected portfolio recovery curve with milestones
        Replaces the static recovery bar with realistic recovery expectations
        """
        try:
            impact_pct = float(stress_result.get('impact_percentage', 0))
            recovery_months = int(stress_result.get('recovery_months', 12))
            
            # Ensure minimum values to prevent divide by zero
            # Impact should be at least -0.1% and recovery should be at least 1 month
            if abs(impact_pct) < 0.1:
                impact_pct = -0.1 if impact_pct <= 0 else 0.1
            if recovery_months < 1:
                recovery_months = 1
            
            # Generate recovery curve (exponential recovery model)
            months = np.arange(0, min(recovery_months + 6, 36))  # Cap at 36 months
            # Use max to ensure denominator is never zero
            time_constant = max(recovery_months / 3, 0.5)  # Minimum time constant of 0.5
            recovery_curve = 100 + impact_pct * np.exp(-months / time_constant)
            
            # Milestone calculations
            milestone_25 = next((i for i, v in enumerate(recovery_curve) if v >= 100 + impact_pct * 0.75), recovery_months)
            milestone_50 = next((i for i, v in enumerate(recovery_curve) if v >= 100 + impact_pct * 0.5), recovery_months)
            milestone_100 = recovery_months
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot recovery curve
            ax.plot(months, recovery_curve, linewidth=3, color=self.colors['info'], label='Expected Recovery')
            ax.fill_between(months, recovery_curve, 100, alpha=0.3, color=self.colors['info'])
            
            # Add baseline
            ax.axhline(y=100, color='black', linestyle='-', linewidth=2, label='Pre-Stress Value')
            
            # Mark milestones
            milestones = [
                (milestone_25, '25% Recovery', self.colors['high']),
                (milestone_50, '50% Recovery', self.colors['medium']),
                (milestone_100, 'Full Recovery', self.colors['low'])
            ]
            
            for month, label, color in milestones:
                if month < len(recovery_curve):
                    ax.axvline(x=month, color=color, linestyle='--', alpha=0.7)
                    ax.text(month, ax.get_ylim()[1] * 0.95, f"{label}\n({month}m)", 
                           ha='center', va='top', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.7, edgecolor='white'))
            
            # Customize chart
            ax.set_xlabel('Months', fontweight='bold')
            ax.set_ylabel('Portfolio Value (%)', fontweight='bold')
            ax.set_title(f"{title}\nInitial Impact: {impact_pct:.1f}%", fontweight='bold', pad=20)
            
            # Add performance zones
            ax.axhspan(100, ax.get_ylim()[1], alpha=0.1, color=self.colors['positive'], label='Above Baseline')
            ax.axhspan(ax.get_ylim()[0], 100, alpha=0.1, color=self.colors['negative'], label='Below Baseline')
            
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
            # Use tight_layout with error handling to prevent warnings
            try:
                plt.tight_layout(pad=2.0)
            except:
                plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Recovery timeline error: {str(e)}")
    
    def create_risk_impact_waterfall(self, stress_result: Dict, portfolio_data: pd.DataFrame,
                                   title: str = "Risk Impact Breakdown") -> plt.Figure:
        """
        5. Risk Impact Waterfall Chart - Shows how different factors contribute to total risk
        New addition to break down exactly WHY the portfolio has its risk level
        """
        try:
            base_impact = float(stress_result.get('impact_percentage', 0))
            
            # Calculate risk factors (simplified model for demonstration)
            concentration_penalty = self._calculate_concentration_penalty(portfolio_data)
            diversification_benefit = self._calculate_diversification_benefit(portfolio_data)
            sector_risk = self._calculate_sector_risk(portfolio_data, stress_result)
            
            # Waterfall components
            components = [
                ('Base Scenario', base_impact, self.colors['neutral']),
                ('Concentration Risk', concentration_penalty, self.colors['negative']),
                ('Sector Risk', sector_risk, self.colors['warning']),
                ('Diversification', diversification_benefit, self.colors['positive']),
            ]
            
            # Calculate cumulative values
            cumulative = [0]
            for _, value, _ in components:
                cumulative.append(cumulative[-1] + value)
            
            final_impact = cumulative[-1]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Draw waterfall bars
            bar_width = 0.6
            x_positions = range(len(components) + 1)
            
            for i, (label, value, color) in enumerate(components):
                if value >= 0:
                    ax.bar(i + 1, value, bottom=cumulative[i], width=bar_width, 
                          color=color, alpha=0.7, edgecolor='black')
                    # Add value label
                    ax.text(i + 1, cumulative[i] + value/2, f"{value:+.1f}%", 
                           ha='center', va='center', fontweight='bold', color='white')
                else:
                    ax.bar(i + 1, abs(value), bottom=cumulative[i+1], width=bar_width, 
                          color=color, alpha=0.7, edgecolor='black')
                    # Add value label
                    ax.text(i + 1, cumulative[i+1] + abs(value)/2, f"{value:+.1f}%", 
                           ha='center', va='center', fontweight='bold', color='white')
            
            # Draw connecting lines
            for i in range(len(components)):
                if i > 0:
                    ax.plot([i + 0.3, i + 0.7], [cumulative[i], cumulative[i]], 
                           'k--', alpha=0.5, linewidth=1)
            
            # Final result bar
            ax.bar(len(components) + 1, final_impact, width=bar_width, 
                  color=self.get_risk_color(final_impact), alpha=0.8, edgecolor='black', linewidth=2)
            ax.text(len(components) + 1, final_impact/2, f"{final_impact:.1f}%", 
                   ha='center', va='center', fontweight='bold', color='white', fontsize=12)
            
            # Customize chart
            labels = [''] + [comp[0] for comp in components] + ['Total Impact']
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Impact (%)', fontweight='bold')
            ax.set_title(title, fontweight='bold', pad=20)
            
            # Add zero line
            ax.axhline(y=0, color='black', linewidth=1)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Waterfall chart error: {str(e)}")
    
    def create_multi_scenario_radar(self, comparison_df: pd.DataFrame, 
                                  title: str = "Multi-Scenario Risk Analysis") -> plt.Figure:
        """
        6. Multi-Scenario Radar Chart - Compares multiple scenarios on different metrics
        Replaces the comparison bar charts with comprehensive scenario comparison
        """
        try:
            if comparison_df.empty:
                return self._create_error_chart("No comparison data available")
            
            # Prepare metrics for radar chart
            metrics = ['Impact Severity', 'Recovery Time', 'Volatility', 'Concentration Risk']
            scenarios = comparison_df['scenario'].tolist()
            
            # Normalize metrics (0-10 scale)
            normalized_data = []
            for _, row in comparison_df.iterrows():
                impact = min(abs(float(row.get('impact_percentage', 0))) / 5, 10)  # Scale impact
                recovery = min(float(row.get('recovery_months', 12)) / 3, 10)  # Scale recovery time
                volatility = impact * 0.8 + np.random.normal(0, 1)  # Mock volatility
                concentration = min(impact * 0.6 + np.random.normal(0, 0.5), 10)  # Mock concentration
                
                normalized_data.append([impact, recovery, max(0, volatility), max(0, concentration)])
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Calculate angles for each metric
            angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
            angles += angles[:1]  # Complete the circle
            
            # Colors for different scenarios
            colors = plt.cm.Set1(np.linspace(0, 1, len(scenarios)))
            
            # Plot each scenario
            for i, (scenario, data) in enumerate(zip(scenarios, normalized_data)):
                data += data[:1]  # Complete the circle
                ax.plot(angles, data, 'o-', linewidth=2, label=scenario.replace('_', ' ').title(), 
                       color=colors[i], markersize=6)
                ax.fill(angles, data, alpha=0.15, color=colors[i])
            
            # Customize radar chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
            
            # Add title
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
            
            # Add interpretation guide
            fig.text(0.02, 0.02, 
                    "Higher values = Greater risk/impact\nCloser to center = Better performance", 
                    fontsize=9, ha='left', va='bottom', 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Radar chart error: {str(e)}")
    
    # ===== HELPER METHODS FOR NEW CHART TYPES =====
    
    def _get_value_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the appropriate value column from the dataframe - Universal approach for multiple brokers"""
        if df.empty:
            return None
            
        # Priority order: Current/Market Value > Invested Value > Buy Value > others
        possible_columns = [
            # Format 1: Your current CSV format
            'Closing value', 'Closing Value', 'Buy value', 'Buy Value',
            # Format 2: Advanced broker format  
            'Market Value as of last trading day', 'Invested Value', 'Overall Gain/Loss',
            # Format 3: Trading format
            'Buy Value', 'Sell Value',
            # Common variations
            'Current Value', 'Market Value', 'Market value', 'Investment Value',
            'Cur. val', 'Investment', 'Holdings Value', 'Total Value', 'Present Value',
            'Value', 'value', 'Amount', 'Current Market Value'
        ]
        
        for col in possible_columns:
            if col in df.columns and not df[col].isna().all():
                # Check if it's actually numeric
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    return col
                except:
                    continue
        
        # If no perfect match, look for any numeric column with value-related keywords
        for col in df.columns:
            col_lower = col.lower()
            if (('val' in col_lower or 'investment' in col_lower or 'amount' in col_lower or 'price' in col_lower) 
                and pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().all()):
                return col
                
        return None
    
    def _get_stock_name_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the appropriate stock name column - Universal approach for multiple brokers"""
        if df.empty:
            return None
            
        # Priority order based on the 3 broker formats you provided
        possible_columns = [
            # Format 1: Your current CSV format
            'Stock Name', 'stock name',
            # Format 2: Advanced broker format
            'Scrip/Contract', 'Company Name', 'Scrip', 'Contract',
            # Format 3: Trading format  
            'Symbol', 'symbol',
            # Common variations
            'Security Name', 'security name', 'Name', 'name', 'ISIN',
            'Name of Security', 'Script Name', 'CompanyName', 'StockName', 
            'SecurityName', 'Instrument', 'Ticker'
        ]
        
        for col in possible_columns:
            if col in df.columns and not df[col].isna().all():
                # Verify it contains actual stock names, not just numbers or empty values
                sample_values = df[col].dropna().head(5).astype(str)
                if any(len(val) > 1 and not val.replace('.', '').isdigit() for val in sample_values):
                    return col
        
        # If no perfect match, use the first non-numeric column that looks like names
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().all():
                # Check if it contains stock-like names (not dates or other metadata)
                sample_values = df[col].dropna().head(3).astype(str)
                if any(len(val) > 2 and not val.isdigit() and 'date' not in val.lower() for val in sample_values):
                    return col
                
        return None
    
    def _analyze_sector_composition(self, portfolio_data: pd.DataFrame) -> Dict:
        """Analyze portfolio composition by sector with comprehensive classification"""
        if portfolio_data.empty:
            print("Debug: Portfolio data is empty, using fallback")
            return self._create_fallback_sector_data()
            
        value_col = self._get_value_column(portfolio_data)
        name_col = self._get_stock_name_column(portfolio_data)
        
        if not value_col or not name_col:
            return {}
        
        # Comprehensive sector keywords mapping
        sector_keywords = {
            'Banking & Finance': [
                'BANK', 'BANKS', 'BANKING', 'FINANCIAL', 'FINANCE', 'CAPITAL', 'CREDIT',
                'INSURANCE', 'MUTUAL', 'FUND', 'SECURITIES', 'INVESTMENT', 'ASSET',
                'HDFC', 'ICICI', 'SBI', 'AXIS', 'KOTAK', 'BAJAJ', 'LIC', 'IDFC'
            ],
            'Technology': [
                'TECH', 'TECHNOLOGY', 'SOFTWARE', 'IT', 'INFOTECH', 'SYSTEM', 'SYSTEMS',
                'DATA', 'DIGITAL', 'CYBER', 'CLOUD', 'AI', 'AUTOMATION', 'CONSULTANCY', 'SERVICES',
                'TCS', 'TATA CONSULTANCY', 'INFOSYS', 'WIPRO', 'TECHM', 'MINDTREE', 'MPHASIS'
            ],
            'Healthcare & Pharma': [
                'PHARMA', 'PHARMACEUTICAL', 'HEALTH', 'HEALTHCARE', 'MEDICAL', 'MEDICINE',
                'DRUG', 'DRUGS', 'BIO', 'LIFE', 'HOSPITAL', 'CLINIC',
                'CIPLA', 'LUPIN', 'DRLABS', 'SUNPHARMA', 'BIOCON', 'CADILA'
            ],
            'Energy & Oil': [
                'OIL', 'GAS', 'ENERGY', 'POWER', 'PETROLEUM', 'COAL', 'SOLAR',
                'ELECTRIC', 'ELECTRICITY', 'RENEWABLE', 'FUEL', 'REFINERY',
                'ONGC', 'IOC', 'BPCL', 'HPCL', 'GAIL', 'NTPC', 'POWERGRID', 'RELIANCE'
            ],
            'Automobiles': [
                'AUTO', 'AUTOMOBILE', 'MOTORS', 'MOTOR', 'CAR', 'CARS', 'VEHICLE',
                'TRACTOR', 'BIKE', 'SCOOTER', 'TRUCK', 'BUS',
                'MARUTI', 'TATA', 'MAHINDRA', 'BAJAJ', 'HERO', 'TVS', 'EICHER'
            ],
            'Consumer Goods': [
                'CONSUMER', 'GOODS', 'PRODUCTS', 'FOODS', 'FOOD', 'BEVERAGE',
                'RETAIL', 'STORE', 'MART', 'BRAND', 'LIFESTYLE',
                'HUL', 'ITC', 'BRITANNIA', 'NESTLE', 'GODREJ', 'DABUR'
            ],
            'Infrastructure': [
                'CONSTRUCTION', 'INFRASTRUCTURE', 'BUILDING', 'CEMENT', 'STEEL',
                'REAL', 'ESTATE', 'PROPERTY', 'HOUSING', 'DEVELOPER', 'ENGINEERING',
                'L&T', 'LARSEN', 'TOUBRO', 'DLF', 'ULTRATECH', 'ACC', 'AMBUJA', 'JSW'
            ],
            'Telecom': [
                'TELECOM', 'COMMUNICATION', 'MOBILE', 'NETWORK', 'WIRELESS',
                'BHARTI', 'AIRTEL', 'JIO', 'IDEA', 'VODAFONE'
            ]
        }
        
        sector_composition = {}
        total_value = portfolio_data[value_col].sum()
        
        if total_value <= 0:
            print(f"Debug: Basic analysis - total_value={total_value}, portfolio rows={len(portfolio_data)}")
            return self._create_fallback_sector_data()
        
        for _, row in portfolio_data.iterrows():
            try:
                stock_name = str(row[name_col]).upper().strip()
                value = float(row[value_col]) if pd.notna(row[value_col]) else 0
                
                if value <= 0:
                    continue
                
                # Find sector by keyword matching - enhanced for better detection
                sector = 'Other'
                max_matches = 0
                best_match_sector = None
                
                for sector_name, keywords in sector_keywords.items():
                    matches = sum(1 for keyword in keywords if keyword in stock_name)
                    # Also check for partial matches (more flexible)
                    partial_matches = sum(1 for keyword in keywords if any(part in stock_name for part in keyword.split()))
                    total_score = matches * 2 + partial_matches  # Full matches count more
                    
                    if total_score > max_matches:
                        max_matches = total_score
                        best_match_sector = sector_name
                
                if best_match_sector:
                    sector = best_match_sector
                
                # If no keywords match, try to classify by common suffixes/patterns
                if sector == 'Other':
                    if any(suffix in stock_name for suffix in ['LTD', 'LIMITED', 'CORP', 'INC']):
                        # Keep as 'Other' but it's a valid company
                        pass
                    elif len(stock_name) < 3:
                        # Likely a symbol, keep as Other
                        pass
                
                if sector not in sector_composition:
                    sector_composition[sector] = {'value': 0, 'weight': 0, 'count': 0}
                
                sector_composition[sector]['value'] += value
                sector_composition[sector]['count'] += 1
                
            except (ValueError, TypeError):
                # Skip invalid rows
                continue
        
        # Calculate weights
        for sector in sector_composition:
            if total_value > 0:
                sector_composition[sector]['weight'] = sector_composition[sector]['value'] / total_value
            else:
                sector_composition[sector]['weight'] = 0
        
        # Return fallback data if no sectors were found
        if not sector_composition:
            print("Debug: No sectors found in basic analysis, using fallback")
            return self._create_fallback_sector_data()
        
        return sector_composition
    
    def _analyze_sector_composition_with_live_data(self, portfolio_data: pd.DataFrame, live_market_data: Dict) -> Dict:
        """Enhanced sector composition analysis using live market data when available"""
        if portfolio_data.empty:
            print("Debug: Portfolio data is empty, using fallback")
            return self._create_fallback_sector_data()
            
        value_col = self._get_value_column(portfolio_data)
        name_col = self._get_stock_name_column(portfolio_data)
        
        if not value_col or not name_col:
            # Debug: Create fallback with generic sectors if columns not found
            print(f"Debug: value_col={value_col}, name_col={name_col}")
            print(f"Available columns: {list(portfolio_data.columns)}")
            return self._create_fallback_sector_data()
        
        sector_composition = {}
        total_value = portfolio_data[value_col].sum()
        
        if total_value <= 0:
            print(f"Debug: total_value={total_value}, creating fallback")
            return self._create_fallback_sector_data()
        
        for _, row in portfolio_data.iterrows():
            try:
                stock_name = str(row[name_col]).strip()
                value = float(row[value_col]) if pd.notna(row[value_col]) else 0
                
                if value <= 0:
                    continue
                
                # Use live data if available, otherwise fallback to keyword classification
                if stock_name in live_market_data and 'error' not in live_market_data[stock_name]:
                    market_data = live_market_data[stock_name]
                    sector = market_data.get('sector', 'Diversified')
                    market_cap_category = market_data.get('market_cap_category', 'Unknown Cap')
                    data_source = 'Live'
                else:
                    # Fallback to existing keyword-based classification
                    sector = self._classify_sector_by_keywords(stock_name)
                    market_cap_category = 'Unknown Cap'
                    data_source = 'Estimated'
                
                if sector not in sector_composition:
                    sector_composition[sector] = {
                        'value': 0, 
                        'weight': 0, 
                        'count': 0,
                        'market_cap_mix': {},
                        'data_sources': {'live': 0, 'estimated': 0},
                        'stocks': []
                    }
                
                sector_composition[sector]['value'] += value
                sector_composition[sector]['count'] += 1
                sector_composition[sector]['data_sources'][data_source.lower()] += 1
                
                # Track market cap composition
                if market_cap_category not in sector_composition[sector]['market_cap_mix']:
                    sector_composition[sector]['market_cap_mix'][market_cap_category] = 0
                sector_composition[sector]['market_cap_mix'][market_cap_category] += value
                
                # Store stock details
                sector_composition[sector]['stocks'].append({
                    'name': stock_name,
                    'value': value,
                    'market_cap': market_cap_category,
                    'data_source': data_source
                })
                
            except (ValueError, TypeError):
                continue
        
        # Calculate weights and market cap percentages
        for sector in sector_composition:
            if total_value > 0:
                sector_composition[sector]['weight'] = sector_composition[sector]['value'] / total_value
                
                # Calculate market cap mix percentages
                sector_total = sector_composition[sector]['value']
                for cap_category in sector_composition[sector]['market_cap_mix']:
                    sector_composition[sector]['market_cap_mix'][cap_category] = (
                        sector_composition[sector]['market_cap_mix'][cap_category] / sector_total
                    )
            else:
                sector_composition[sector]['weight'] = 0
        
        # Return fallback data if no sectors were found
        if not sector_composition:
            print("Debug: No sectors found in enhanced analysis, using fallback") 
            return self._create_fallback_sector_data()
        
        return sector_composition
    
    def _create_fallback_sector_data(self) -> Dict:
        """Create fallback sector data when real data is unavailable"""
        return {
            'Technology': {
                'value': 100000, 'weight': 0.4, 'count': 3,
                'market_cap_mix': {'Large Cap': 1.0},
                'data_sources': {'estimated': 3, 'live': 0},
                'stocks': [{'name': 'Tech Stock A', 'value': 50000, 'market_cap': 'Large Cap', 'data_source': 'Estimated'}]
            },
            'Banking & Finance': {
                'value': 75000, 'weight': 0.3, 'count': 2, 
                'market_cap_mix': {'Large Cap': 1.0},
                'data_sources': {'estimated': 2, 'live': 0},
                'stocks': [{'name': 'Bank Stock A', 'value': 40000, 'market_cap': 'Large Cap', 'data_source': 'Estimated'}]
            },
            'Healthcare & Pharma': {
                'value': 50000, 'weight': 0.2, 'count': 2,
                'market_cap_mix': {'Mid Cap': 1.0}, 
                'data_sources': {'estimated': 2, 'live': 0},
                'stocks': [{'name': 'Pharma Stock A', 'value': 30000, 'market_cap': 'Mid Cap', 'data_source': 'Estimated'}]
            },
            'Consumer Goods': {
                'value': 25000, 'weight': 0.1, 'count': 1,
                'market_cap_mix': {'Large Cap': 1.0},
                'data_sources': {'estimated': 1, 'live': 0}, 
                'stocks': [{'name': 'Consumer Stock A', 'value': 25000, 'market_cap': 'Large Cap', 'data_source': 'Estimated'}]
            }
        }
    
    def _classify_sector_by_keywords(self, stock_name: str) -> str:
        """Classify sector using keyword matching (fallback method)"""
        name_upper = stock_name.upper()
        
        # Simplified keyword mapping for fallback
        if any(keyword in name_upper for keyword in ['BANK', 'FINANCIAL', 'INSURANCE']):
            return 'Banking & Finance'
        elif any(keyword in name_upper for keyword in ['TECH', 'SOFTWARE', 'IT']):
            return 'Technology'
        elif any(keyword in name_upper for keyword in ['PHARMA', 'HEALTH', 'MEDICAL']):
            return 'Healthcare & Pharma'
        elif any(keyword in name_upper for keyword in ['OIL', 'GAS', 'ENERGY', 'POWER']):
            return 'Energy & Oil'
        elif any(keyword in name_upper for keyword in ['AUTO', 'MOTOR', 'CAR']):
            return 'Automobiles'
        elif any(keyword in name_upper for keyword in ['CONSUMER', 'FOOD', 'RETAIL']):
            return 'Consumer Goods'
        elif any(keyword in name_upper for keyword in ['CEMENT', 'STEEL', 'CONSTRUCTION']):
            return 'Infrastructure'
        elif any(keyword in name_upper for keyword in ['TELECOM', 'COMMUNICATION']):
            return 'Telecom'
        else:
            return 'Diversified'
    
    def _get_sector_impacts(self, scenario_name: str) -> Dict[str, float]:
        """Get expected sector impacts for a given scenario"""
        # Sector impact mapping for different scenarios - Updated for new sector names
        scenario_impacts = {
            'market_correction': {
                'Banking & Finance': -0.12, 'Technology': -0.15, 'Healthcare & Pharma': -0.08,
                'Energy & Oil': -0.10, 'Automobiles': -0.14, 'Consumer Goods': -0.09,
                'Infrastructure': -0.13, 'Telecom': -0.11, 'Other': -0.10
            },
            'recession_scenario': {
                'Banking & Finance': -0.25, 'Technology': -0.22, 'Healthcare & Pharma': -0.15,
                'Energy & Oil': -0.18, 'Automobiles': -0.28, 'Consumer Goods': -0.20,
                'Infrastructure': -0.24, 'Telecom': -0.18, 'Other': -0.20
            },
            'tech_sector_crash': {
                'Banking & Finance': -0.12, 'Technology': -0.45, 'Healthcare & Pharma': -0.08,
                'Energy & Oil': -0.05, 'Automobiles': -0.10, 'Consumer Goods': -0.08,
                'Infrastructure': -0.08, 'Telecom': -0.20, 'Other': -0.10
            },
            'inflation_spike': {
                'Banking & Finance': -0.08, 'Technology': -0.20, 'Healthcare & Pharma': -0.05,
                'Energy & Oil': 0.10, 'Automobiles': -0.15, 'Consumer Goods': -0.12,
                'Infrastructure': -0.08, 'Telecom': -0.10, 'Other': -0.08
            },
            'us_bond_yields_impact': {
                'Banking & Finance': -0.12, 'Technology': -0.10, 'Healthcare & Pharma': -0.06,
                'Energy & Oil': -0.08, 'Automobiles': -0.11, 'Consumer Goods': -0.09,
                'Infrastructure': -0.10, 'Telecom': -0.08, 'Other': -0.09
            }
        }
        
        return scenario_impacts.get(scenario_name, scenario_impacts['market_correction'])
    
    def _calculate_concentration_penalty(self, portfolio_data: pd.DataFrame) -> float:
        """Calculate concentration risk penalty"""
        try:
            value_col = self._get_value_column(portfolio_data)
            if not value_col:
                return 0.0
            
            values = portfolio_data[value_col].astype(float)
            total_value = values.sum()
            
            if total_value <= 0:
                return 0.0
            
            # Calculate Herfindahl-Hirschman Index
            weights = values / total_value
            hhi = (weights ** 2).sum()
            
            # Convert to concentration penalty (0-5% additional risk)
            concentration_penalty = max(0, (hhi - 0.1) * 10)  # Penalty starts at 10% concentration
            return min(concentration_penalty, 5.0)
            
        except Exception:
            return 2.0  # Default moderate penalty
    
    def _calculate_diversification_benefit(self, portfolio_data: pd.DataFrame) -> float:
        """Calculate diversification benefit"""
        try:
            num_holdings = len(portfolio_data)
            
            # Diversification benefit based on number of holdings
            if num_holdings >= 20:
                return -2.0  # 2% risk reduction
            elif num_holdings >= 15:
                return -1.5
            elif num_holdings >= 10:
                return -1.0
            elif num_holdings >= 5:
                return -0.5
            else:
                return 0.0  # No benefit for < 5 holdings
                
        except Exception:
            return -1.0  # Default moderate benefit
    
    def _calculate_sector_risk(self, portfolio_data: pd.DataFrame, stress_result: Dict) -> float:
        """Calculate additional sector-specific risk (fast, no live data)"""
        try:
            # FAST MODE: Skip expensive live market data fetching
            # Use simple sector concentration logic instead
            
            value_col = self._get_value_column(portfolio_data)
            if not value_col:
                return 1.0  # Default moderate risk
            
            # Quick sector concentration check
            sector_col = None
            for col in portfolio_data.columns:
                if 'sector' in col.lower() or 'industry' in col.lower():
                    sector_col = col
                    break
            
            if sector_col:
                # Calculate sector concentration
                total_value = portfolio_data[value_col].sum()
                sector_weights = portfolio_data.groupby(sector_col)[value_col].sum() / total_value
                max_sector_weight = sector_weights.max()
                
                # Additional risk based on concentration
                if max_sector_weight > 0.5:
                    return 3.0  # High concentration risk
                elif max_sector_weight > 0.3:
                    return 2.0  # Medium concentration risk
                else:
                    return 1.0  # Low concentration risk
            else:
                return 1.0  # Default if no sector info
                
        except Exception:
            return 1.0  # Default moderate risk
    
    def _create_error_chart(self, error_message: str) -> plt.Figure:
        """Create a simple error message chart"""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Chart Error:\n{error_message}", 
               ha='center', va='center', transform=ax.transAxes, 
               fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    def plot_single(self, impact_percentage, scenario_name):
        """Create a single scenario impact plot (LEGACY - use create_risk_gauge_chart instead)"""
        # Convert to new risk gauge format
        stress_result = {
            'impact_percentage': impact_percentage,
            'scenario_name': scenario_name
        }
        return self.create_risk_gauge_chart(stress_result, f"{scenario_name} Risk Assessment")
    
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
            
            # Create heatmap using matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(pivot_data.values, cmap='RdYlGn_r', aspect='auto')
            
            # Add labels
            ax.set_xticks(range(len(pivot_data.columns)))
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_xticklabels(pivot_data.columns)
            ax.set_yticklabels(pivot_data.index)
            
            # Add text annotations
            for i in range(len(pivot_data.index)):
                for j in range(len(pivot_data.columns)):
                    text = ax.text(j, i, f'{pivot_data.iloc[i, j]:.1f}', 
                                 ha="center", va="center", color="black", fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Impact (%)', rotation=270, labelpad=15)
            
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
    
    def create_stress_impact_plot(self, result):
        """Create enhanced stress impact visualization (LEGACY - now uses meaningful charts)"""
        try:
            # Extract data from result object
            stress_impact = result.get('stress_impact', {})
            scenario = result.get('scenario', {})
            
            impact_percentage = stress_impact.get('impact_percentage', 0)
            scenario_name = scenario.get('name', 'Stress Scenario')
            recovery_months = stress_impact.get('recovery_months', 12)
            
            # Convert to new format and use the risk gauge chart instead
            stress_result_data = {
                'impact_percentage': impact_percentage,
                'scenario_name': scenario_name,
                'recovery_months': recovery_months
            }
            
            # Return the new meaningful risk gauge chart
            return self.create_risk_gauge_chart(stress_result_data, f"{scenario_name} - Impact Analysis")
            
        except Exception as e:
            # Fallback: simple impact plot
            try:
                stress_impact = result.get('stress_impact', {})
                impact_percentage = stress_impact.get('impact_percentage', 0)
                scenario_name = result.get('scenario', {}).get('name', 'Stress Scenario')
                
                return self.plot_single(impact_percentage, scenario_name)
            except:
                # Ultimate fallback
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, f'Visualization unavailable: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                return fig

    def create_allocation_pie(self, composition):
        """Create portfolio allocation pie chart (LEGACY - redirects to meaningful composition chart)"""
        try:
            # Convert old composition format to DataFrame format for new method
            if isinstance(composition, dict):
                # Convert dict to DataFrame format
                portfolio_data = pd.DataFrame([
                    {'Stock Name': asset.replace('_', ' ').title(), 'Closing value': value * 100000} 
                    for asset, value in composition.items() if value > 0.01
                ])
            else:
                # Assume it's already in a usable format
                portfolio_data = pd.DataFrame(composition)
                if 'Asset' in portfolio_data.columns:
                    portfolio_data = portfolio_data.rename(columns={'Asset': 'Stock Name', 'Value': 'Closing value'})
            
            if portfolio_data.empty:
                return self._create_error_chart("No allocation data available")
            
            return self.create_portfolio_composition_pie(portfolio_data, "Portfolio Asset Allocation")
            
        except Exception as e:
            return self._create_error_chart(f"Allocation pie error: {str(e)}")

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