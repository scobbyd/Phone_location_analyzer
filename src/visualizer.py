"""Visualization module for location analysis.

Creates multiple visualization types:
1. Calendar heatmap (GitHub-style)
2. Hourly presence matrix
3. Monthly summary cards
4. Interactive folium map
5. Confidence timeline
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class LocationVisualizer:
    """Creates visualizations for location analysis."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize visualizer with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def create_calendar_heatmap(
        self,
        overlaps_df: pd.DataFrame,
        output_file: Path,
        title: str = "Co-Presence Calendar"
    ) -> None:
        """Create GitHub-style calendar heatmap showing daily co-presence.

        Args:
            overlaps_df: DataFrame with overlap data
            output_file: Output file path
            title: Chart title
        """
        if overlaps_df.empty:
            print("No data to visualize")
            return

        # Get date range
        start_date = overlaps_df['timestamp'].min().date()
        end_date = overlaps_df['timestamp'].max().date()

        # Count co-locations per day
        daily_counts = overlaps_df.groupby(
            overlaps_df['timestamp'].dt.date
        ).size()

        # Create complete date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Build matrix (weeks x days)
        weeks = []
        counts_dict = daily_counts.to_dict()

        # Pad the first week from Monday
        first_dow = date_range[0].dayofweek  # 0=Mon, 6=Sun
        current_week = [0] * first_dow  # Pad with zeros before first day

        for date in date_range:
            count = counts_dict.get(date.date(), 0)
            current_week.append(count)

            if date.dayofweek == 6:  # Sunday
                weeks.append(current_week)
                current_week = []

        # Add incomplete week, pad to 7
        if current_week:
            while len(current_week) < 7:
                current_week.append(0)
            weeks.append(current_week)

        # Convert to array
        cal_array = np.array(weeks).T

        # Create figure
        fig, ax = plt.subplots(figsize=(20, 4))

        # Use green colormap (GitHub style)
        cmap = plt.cm.Greens
        im = ax.imshow(cal_array, aspect='auto', cmap=cmap, interpolation='nearest')

        # Set labels
        ax.set_yticks(range(7))
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

        # Set x-axis as week numbers
        week_labels = [f"W{i+1}" for i in range(len(weeks))]
        ax.set_xticks(range(len(weeks)))
        ax.set_xticklabels(week_labels, rotation=90)

        # Title
        ax.set_title(title, fontsize=14, pad=20)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Co-locations per day', fontsize=10)

        # Grid
        ax.set_xticks(np.arange(len(weeks)) - 0.5, minor=True)
        ax.set_yticks(np.arange(7) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Calendar heatmap saved to {output_file}")

    def create_hourly_matrix(
        self,
        presence_matrix: pd.DataFrame,
        output_file: Path,
        title: str = "Hourly Co-Presence Probability"
    ) -> None:
        """Create hourly presence matrix visualization.

        Args:
            presence_matrix: DataFrame with rows=hours, columns=dates, values=probability
            output_file: Output file path
            title: Chart title
        """
        fig, ax = plt.subplots(figsize=(20, 8))

        # Create heatmap
        sns.heatmap(
            presence_matrix,
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Co-presence probability'},
            ax=ax,
            xticklabels=False  # Too many dates
        )

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_ylabel('Hour of Day', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)

        # Highlight overnight hours
        overnight_start = self.config['interpolation']['overnight_hours']['start']
        overnight_end = self.config['interpolation']['overnight_hours']['end']

        ax.axhline(y=overnight_start, color='blue', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=overnight_end, color='blue', linestyle='--', alpha=0.3, linewidth=1)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Hourly matrix saved to {output_file}")

    def create_monthly_summary(
        self,
        overlaps_df: pd.DataFrame,
        output_file: Path
    ) -> None:
        """Create monthly summary bar charts.

        Args:
            overlaps_df: DataFrame with overlap data
            output_file: Output file path
        """
        if overlaps_df.empty:
            print("No data to visualize")
            return

        # Group by month
        grouped = overlaps_df.groupby(
            overlaps_df['timestamp'].dt.to_period('M')
        )
        monthly_total = grouped['timestamp'].size().rename('total_colocations')
        monthly_days = grouped['timestamp'].apply(lambda x: x.dt.date.nunique()).rename('unique_days')
        monthly = pd.DataFrame({'total_colocations': monthly_total, 'unique_days': monthly_days}).reset_index()
        monthly.columns = ['month', 'total_colocations', 'unique_days']

        # Count nights per month
        night_overlaps = overlaps_df[
            (overlaps_df['timestamp'].dt.hour >= 22) |
            (overlaps_df['timestamp'].dt.hour <= 6)
        ]
        monthly_nights = night_overlaps.groupby(
            night_overlaps['timestamp'].dt.to_period('M')
        )['timestamp'].apply(lambda x: x.dt.date.nunique()).rename('nights').reset_index()
        monthly_nights.columns = ['month', 'nights']

        # Merge
        monthly = monthly.merge(monthly_nights, on='month', how='left')
        monthly['nights'] = monthly['nights'].fillna(0)

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        months_str = monthly['month'].astype(str)

        # Days together
        axes[0].bar(months_str, monthly['unique_days'], color='steelblue')
        axes[0].set_title('Unique Days Together per Month', fontsize=12)
        axes[0].set_ylabel('Days', fontsize=10)
        axes[0].tick_params(axis='x', rotation=45)

        # Nights together
        axes[1].bar(months_str, monthly['nights'], color='darkgreen')
        axes[1].set_title('Nights Together per Month', fontsize=12)
        axes[1].set_ylabel('Nights', fontsize=10)
        axes[1].tick_params(axis='x', rotation=45)

        # Co-locations count
        axes[2].bar(months_str, monthly['total_colocations'], color='purple')
        axes[2].set_title('Total Co-locations per Month', fontsize=12)
        axes[2].set_ylabel('Count', fontsize=10)
        axes[2].set_xlabel('Month', fontsize=10)
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Monthly summary saved to {output_file}")

    def create_interactive_map(
        self,
        overlaps_df: pd.DataFrame,
        output_file: Path,
        privacy_round: int = 3
    ) -> None:
        """Create interactive folium map with privacy considerations.

        Args:
            overlaps_df: DataFrame with overlap data
            output_file: Output file path
            privacy_round: Decimal places to round coordinates (privacy)
        """
        if overlaps_df.empty:
            print("No overlaps to map")
            return

        # Privacy: round coordinates
        overlaps_df = overlaps_df.copy()
        overlaps_df['lat'] = overlaps_df['lat'].round(privacy_round)
        overlaps_df['lon'] = overlaps_df['lon'].round(privacy_round)

        # Center map
        center_lat = overlaps_df['lat'].mean()
        center_lon = overlaps_df['lon'].mean()

        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=self.config['visualization']['map']['default_zoom']
        )

        # Add title
        title_html = f'''
        <div style="position: fixed;
                    top: 10px; left: 50px; width: 400px; height: 90px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 10px">
        <b>Location History Overlap - IND Evidence</b><br>
        Generated: {datetime.now().strftime('%Y-%m-%d')}<br>
        Period: {overlaps_df['timestamp'].min().date()} to {overlaps_df['timestamp'].max().date()}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        # Group overlaps by location
        location_groups = overlaps_df.groupby(['lat', 'lon'])

        marker_cluster = MarkerCluster().add_to(m)

        for (lat, lon), group in location_groups:
            dates = group['timestamp'].dt.date.unique()
            count = len(group)

            # Check if overnight location
            night_count = len(group[
                (group['timestamp'].dt.hour >= 22) |
                (group['timestamp'].dt.hour <= 6)
            ])

            # Determine marker color
            if night_count > 10:
                color = 'darkgreen'
                icon = 'bed'
            elif len(dates) > 30:
                color = 'green'
                icon = 'home'
            elif len(dates) > 10:
                color = 'orange'
                icon = 'map-marker'
            else:
                color = 'blue'
                icon = 'map-marker'

            # Create popup
            popup_html = f"""
            <b>Shared Location</b><br>
            Co-locations: {count}<br>
            Unique days: {len(dates)}<br>
            First: {min(dates)}<br>
            Last: {max(dates)}<br>
            """

            if night_count > 0:
                popup_html += f"<b>Nights: {night_count}</b><br>"

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(marker_cluster)

        # Add heatmap layer
        if self.config['visualization']['map'].get('heatmap_radius', 0) > 0:
            heat_data = [[row['lat'], row['lon']] for _, row in overlaps_df.iterrows()]
            HeatMap(
                heat_data,
                radius=self.config['visualization']['map']['heatmap_radius'],
                blur=10,
                name='Heatmap'
            ).add_to(m)

        # Layer control
        folium.LayerControl().add_to(m)

        # Save
        m.save(str(output_file))
        print(f"Interactive map saved to {output_file}")

    def create_confidence_timeline(
        self,
        partner1_df: pd.DataFrame,
        partner2_df: pd.DataFrame,
        overlaps_df: pd.DataFrame,
        output_file: Path
    ) -> None:
        """Create timeline showing data density and interpolation usage.

        Args:
            partner1_df: Partner 1 data (with interpolation)
            partner2_df: Partner 2 data (with interpolation)
            overlaps_df: Overlap data
            output_file: Output file path
        """
        # Group by day
        date_range = pd.date_range(
            start=min(partner1_df['timestamp'].min(), partner2_df['timestamp'].min()),
            end=max(partner1_df['timestamp'].max(), partner2_df['timestamp'].max()),
            freq='D'
        )

        daily_stats = []

        for date in date_range:
            day_start = pd.Timestamp(date)
            day_end = day_start + timedelta(days=1)

            # Count reports per person
            p1_reports = len(partner1_df[
                (partner1_df['timestamp'] >= day_start) &
                (partner1_df['timestamp'] < day_end)
            ])

            p2_reports = len(partner2_df[
                (partner2_df['timestamp'] >= day_start) &
                (partner2_df['timestamp'] < day_end)
            ])

            # Count interpolated points
            p1_interp = len(partner1_df[
                (partner1_df['timestamp'] >= day_start) &
                (partner1_df['timestamp'] < day_end) &
                (partner1_df.get('interpolated', False))
            ]) if 'interpolated' in partner1_df.columns else 0

            p2_interp = len(partner2_df[
                (partner2_df['timestamp'] >= day_start) &
                (partner2_df['timestamp'] < day_end) &
                (partner2_df.get('interpolated', False))
            ]) if 'interpolated' in partner2_df.columns else 0

            # Count overlaps
            overlaps_count = len(overlaps_df[
                (overlaps_df['timestamp'] >= day_start) &
                (overlaps_df['timestamp'] < day_end)
            ]) if not overlaps_df.empty else 0

            daily_stats.append({
                'date': date.date(),
                'p1_reports': p1_reports,
                'p2_reports': p2_reports,
                'p1_interpolated': p1_interp,
                'p2_interpolated': p2_interp,
                'overlaps': overlaps_count
            })

        stats_df = pd.DataFrame(daily_stats)

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

        dates = stats_df['date']

        # Partner 1 reports
        axes[0].bar(dates, stats_df['p1_reports'], color='steelblue', alpha=0.7, label='Raw reports')
        axes[0].bar(dates, stats_df['p1_interpolated'], color='lightblue', alpha=0.7, label='Interpolated')
        axes[0].set_ylabel('Partner 1 Reports', fontsize=10)
        axes[0].legend(loc='upper right')
        axes[0].set_title('Data Density and Interpolation Usage', fontsize=12)

        # Partner 2 reports
        axes[1].bar(dates, stats_df['p2_reports'], color='darkgreen', alpha=0.7, label='Raw reports')
        axes[1].bar(dates, stats_df['p2_interpolated'], color='lightgreen', alpha=0.7, label='Interpolated')
        axes[1].set_ylabel('Partner 2 Reports', fontsize=10)
        axes[1].legend(loc='upper right')

        # Overlaps
        axes[2].bar(dates, stats_df['overlaps'], color='purple', alpha=0.7)
        axes[2].set_ylabel('Co-locations', fontsize=10)
        axes[2].set_xlabel('Date', fontsize=10)

        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confidence timeline saved to {output_file}")

    def create_all_visualizations(
        self,
        partner1_df: pd.DataFrame,
        partner2_df: pd.DataFrame,
        overlaps_df: pd.DataFrame,
        overlaps_interp_df: pd.DataFrame,
        presence_matrix: Optional[pd.DataFrame],
        output_dir: Path
    ) -> None:
        """Create all visualization types.

        Args:
            partner1_df: Partner 1 data (interpolated)
            partner2_df: Partner 2 data (interpolated)
            overlaps_df: Raw overlaps
            overlaps_interp_df: Overlaps with interpolation
            presence_matrix: Hourly presence matrix
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating visualizations...")

        # Calendar heatmap (raw)
        self.create_calendar_heatmap(
            overlaps_df,
            output_dir / "calendar_raw.png",
            title="Co-Presence Calendar (Raw Data)"
        )

        # Calendar heatmap (interpolated)
        self.create_calendar_heatmap(
            overlaps_interp_df,
            output_dir / "calendar_interpolated.png",
            title="Co-Presence Calendar (With Interpolation)"
        )

        # Hourly matrix
        if presence_matrix is not None and not presence_matrix.empty:
            self.create_hourly_matrix(
                presence_matrix,
                output_dir / "hourly_matrix.png"
            )

        # Monthly summary
        self.create_monthly_summary(
            overlaps_interp_df,
            output_dir / "monthly_summary.png"
        )

        # Interactive map
        self.create_interactive_map(
            overlaps_interp_df,
            output_dir / "interactive_map.html",
            privacy_round=self.config['output']['privacy']['round_coords_decimals']
        )

        # Confidence timeline
        self.create_confidence_timeline(
            partner1_df,
            partner2_df,
            overlaps_interp_df,
            output_dir / "confidence_timeline.png"
        )

        print(f"\nAll visualizations saved to {output_dir}")
