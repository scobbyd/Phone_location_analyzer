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
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
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
        title: str = "Co-Presence Calendar",
        partner1_df: Optional[pd.DataFrame] = None,
        partner2_df: Optional[pd.DataFrame] = None
    ) -> None:
        """Create GitHub-style calendar heatmap showing daily co-presence.

        Three visual states:
        - Green shades: co-located (intensity = count)
        - White: both partners have data, no co-location (apart)
        - Gray hatched: one or both partners have no data (missing)

        Args:
            overlaps_df: DataFrame with overlap data
            output_file: Output file path
            title: Chart title
            partner1_df: Partner 1 location data (for missing-data detection)
            partner2_df: Partner 2 location data (for missing-data detection)
        """
        # Determine date range - use config start_date as fixed start
        all_dfs = [df for df in [overlaps_df, partner1_df, partner2_df]
                   if df is not None and not df.empty]
        if not all_dfs:
            print("No data to visualize")
            return

        start_date = datetime.fromisoformat(self.config['start_date']).date()
        end_date = max(df['timestamp'].max().date() for df in all_dfs)

        # Count co-locations per day
        daily_counts = {}
        if not overlaps_df.empty:
            daily_counts = overlaps_df.groupby(
                overlaps_df['timestamp'].dt.date
            ).size().to_dict()

        # Build per-day data availability sets
        p1_days = set()
        p2_days = set()
        if partner1_df is not None and not partner1_df.empty:
            p1_days = set(partner1_df['timestamp'].dt.date.unique())
        if partner2_df is not None and not partner2_df.empty:
            p2_days = set(partner2_df['timestamp'].dt.date.unique())
        has_coverage_info = bool(p1_days or p2_days)

        # Create complete date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Build matrices (weeks x days): counts and missing mask
        weeks_counts = []
        weeks_missing = []
        weeks_padding = []

        first_dow = date_range[0].dayofweek  # 0=Mon, 6=Sun
        current_counts = [np.nan] * first_dow
        current_missing = [False] * first_dow
        current_padding = [True] * first_dow

        for date in date_range:
            d = date.date()
            count = daily_counts.get(d, 0)
            missing = has_coverage_info and (d not in p1_days or d not in p2_days)

            current_counts.append(count)
            current_missing.append(missing)
            current_padding.append(False)

            if date.dayofweek == 6:  # Sunday
                weeks_counts.append(current_counts)
                weeks_missing.append(current_missing)
                weeks_padding.append(current_padding)
                current_counts = []
                current_missing = []
                current_padding = []

        # Pad incomplete final week
        if current_counts:
            while len(current_counts) < 7:
                current_counts.append(np.nan)
                current_missing.append(False)
                current_padding.append(True)
            weeks_counts.append(current_counts)
            weeks_missing.append(current_missing)
            weeks_padding.append(current_padding)

        cal_array = np.array(weeks_counts).T
        missing_array = np.array(weeks_missing).T
        padding_array = np.array(weeks_padding).T

        # Create figure
        fig, ax = plt.subplots(figsize=(20, 4))

        # Use green colormap, set 0 to white
        cmap = plt.cm.Greens.copy()
        cmap.set_bad(color='none')  # NaN = transparent (padding)

        # Mask padding cells
        cal_masked = np.ma.masked_where(padding_array, cal_array)

        im = ax.imshow(cal_masked, aspect='auto', cmap=cmap, interpolation='nearest')

        # Overlay gray hatched rectangles on missing-data cells
        if has_coverage_info:
            for row in range(missing_array.shape[0]):
                for col in range(missing_array.shape[1]):
                    if missing_array[row, col] and not padding_array[row, col]:
                        rect = Rectangle(
                            (col - 0.5, row - 0.5), 1, 1,
                            linewidth=0, facecolor='#D0D0D0',
                            hatch='///', edgecolor='#999999', alpha=0.9
                        )
                        ax.add_patch(rect)

        # Set labels
        ax.set_yticks(range(7))
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

        # Set x-axis as week numbers
        week_labels = [f"W{i+1}" for i in range(len(weeks_counts))]
        ax.set_xticks(range(len(weeks_counts)))
        ax.set_xticklabels(week_labels, rotation=90)

        # Title
        ax.set_title(title, fontsize=14, pad=20)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.15)
        cbar.set_label('Co-locations per day', fontsize=10)

        # Legend
        legend_elements = [
            Patch(facecolor=plt.cm.Greens(0.6), label='Co-located'),
            Patch(facecolor='white', edgecolor='#CCCCCC', label='Apart (both tracked)'),
        ]
        if has_coverage_info:
            legend_elements.append(
                Patch(facecolor='#D0D0D0', hatch='///', edgecolor='#999999',
                      label='Missing data')
            )
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
                  framealpha=0.9, bbox_to_anchor=(0, 1.15), ncol=3)

        # Grid
        ax.set_xticks(np.arange(len(weeks_counts)) - 0.5, minor=True)
        ax.set_yticks(np.arange(7) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Calendar heatmap saved to {output_file}")

    def create_hourly_matrix(
        self,
        state_matrix: pd.DataFrame,
        output_file: Path,
        title: str = "Hourly Co-Presence",
        ensemble_days: dict = None
    ) -> None:
        """Create hourly presence matrix with 4-state rendering.

        Args:
            state_matrix: DataFrame with rows=hours, columns=dates,
                          values in {'co_located', 'apart', 'no_data'}
            output_file: Output file path
            title: Chart title
            ensemble_days: Optional dict date -> 'together'|'apart'|'ml_together'|'no_data'
                           If provided, 'apart' hours on ml_together days become ml_rescued
        """
        # Apply ensemble overlay: on ML-rescued days, upgrade 'apart' hours to 'ml_rescued'
        if ensemble_days:
            from datetime import date as date_type
            state_matrix = state_matrix.copy()
            for col in state_matrix.columns:
                try:
                    d = col if isinstance(col, date_type) else pd.Timestamp(col).date()
                except Exception:
                    continue
                if ensemble_days.get(d) == 'ml_together':
                    # On ML-rescued days, mark hours where rule-based said 'apart' as 'ml_rescued'
                    state_matrix[col] = state_matrix[col].replace({
                        'apart': 'ml_rescued',
                        'no_data': 'ml_rescued_nodata',
                    })

        # Convert states to numeric: co_located=2, ml_rescued=1, no_data=0, apart=-1
        numeric_matrix = state_matrix.replace({
            'co_located': 2,
            'ml_rescued': 1,
            'ml_rescued_nodata': 0,
            'apart': -1,
            'no_data': 0
        }).astype(float)

        fig, ax = plt.subplots(figsize=(20, 8))

        # Discrete 4-color colormap: red (apart), gray (no_data), blue (ML), green (confirmed)
        cmap = ListedColormap(['#E74C3C', '#D0D0D0', '#5DADE2', '#2ECC71'])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)

        im = ax.imshow(
            numeric_matrix.values,
            aspect='auto',
            cmap=cmap,
            norm=norm,
            interpolation='nearest'
        )

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_ylabel('Hour of Day', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)

        # Y-axis: hours 0-23
        ax.set_yticks(range(24))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(24)])

        # X-axis: date labels (show subset to avoid overcrowding)
        n_cols = len(state_matrix.columns)
        if n_cols > 30:
            step = max(n_cols // 15, 1)
            tick_positions = list(range(0, n_cols, step))
        else:
            tick_positions = list(range(n_cols))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [str(state_matrix.columns[i]) for i in tick_positions],
            rotation=45, ha='right', fontsize=8
        )

        # Highlight overnight hours
        overnight_start = self.config['interpolation']['overnight_hours']['start']
        overnight_end = self.config['interpolation']['overnight_hours']['end']

        ax.axhline(y=overnight_start - 0.5, color='blue', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=overnight_end + 0.5, color='blue', linestyle='--', alpha=0.3, linewidth=1)

        # Legend
        legend_elements = [
            Patch(facecolor='#2ECC71', label='Together (GPS confirmed)'),
            Patch(facecolor='#5DADE2', label='Together (ML predicted)'),
            Patch(facecolor='#E74C3C', label='Apart (both tracked)'),
            Patch(facecolor='#D0D0D0', label='No data'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                  framealpha=0.9, bbox_to_anchor=(1.0, 1.15), ncol=4)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Hourly matrix saved to {output_file}")

    def create_monthly_summary(
        self,
        overlaps_df: pd.DataFrame,
        output_file: Path,
        partner1_df: Optional[pd.DataFrame] = None,
        partner2_df: Optional[pd.DataFrame] = None
    ) -> None:
        """Create monthly summary bar charts with data coverage context.

        Args:
            overlaps_df: DataFrame with overlap data
            output_file: Output file path
            partner1_df: Partner 1 data (for coverage info)
            partner2_df: Partner 2 data (for coverage info)
        """
        # Build complete month range - use config start_date as fixed start
        all_dfs = [df for df in [overlaps_df, partner1_df, partner2_df]
                   if df is not None and not df.empty]
        if not all_dfs:
            print("No data to visualize")
            return

        min_date = pd.Timestamp(self.config['start_date'])
        max_date = max(df['timestamp'].max() for df in all_dfs)
        all_months = pd.period_range(start=min_date.to_period('M'),
                                     end=max_date.to_period('M'), freq='M')

        # Build monthly DataFrame with all months
        monthly = pd.DataFrame({'month': all_months})

        # Co-location stats
        if not overlaps_df.empty:
            grouped = overlaps_df.groupby(overlaps_df['timestamp'].dt.to_period('M'))
            monthly_total = grouped['timestamp'].size().rename('total_colocations')
            monthly_days = grouped['timestamp'].apply(
                lambda x: x.dt.date.nunique()).rename('unique_days')
            coloc = pd.DataFrame({
                'total_colocations': monthly_total,
                'unique_days': monthly_days
            }).reset_index()
            coloc.columns = ['month', 'total_colocations', 'unique_days']

            night_overlaps = overlaps_df[
                (overlaps_df['timestamp'].dt.hour >= 22) |
                (overlaps_df['timestamp'].dt.hour <= 6)
            ]
            if not night_overlaps.empty:
                monthly_nights = night_overlaps.groupby(
                    night_overlaps['timestamp'].dt.to_period('M')
                )['timestamp'].apply(
                    lambda x: x.dt.date.nunique()).rename('nights').reset_index()
                monthly_nights.columns = ['month', 'nights']
                coloc = coloc.merge(monthly_nights, on='month', how='left')
            else:
                coloc['nights'] = 0

            monthly = monthly.merge(coloc, on='month', how='left')
        else:
            monthly['total_colocations'] = 0
            monthly['unique_days'] = 0
            monthly['nights'] = 0

        monthly = monthly.fillna(0)

        # Data coverage: days where both partners have data
        has_coverage = partner1_df is not None and partner2_df is not None
        if has_coverage:
            p1_days = set(partner1_df['timestamp'].dt.date.unique())
            p2_days = set(partner2_df['timestamp'].dt.date.unique())

            coverage_data = []
            for period in all_months:
                month_start = period.start_time.date()
                month_end = period.end_time.date()
                days_in_month = (month_end - month_start).days + 1
                both_days = sum(1 for d in pd.date_range(month_start, month_end).date
                                if d in p1_days and d in p2_days)
                coverage_data.append({
                    'month': period,
                    'days_in_month': days_in_month,
                    'days_both_tracked': both_days
                })
            coverage_df = pd.DataFrame(coverage_data)
            monthly = monthly.merge(coverage_df, on='month', how='left')

        # Create figure
        n_plots = 4 if has_coverage else 3
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.2 * n_plots))

        months_str = monthly['month'].astype(str)
        x = np.arange(len(months_str))
        bar_width = 0.8

        # Data coverage (top, if available)
        plot_idx = 0
        if has_coverage:
            axes[plot_idx].bar(x, monthly['days_in_month'], bar_width,
                              color='#E0E0E0', label='Days in month')
            axes[plot_idx].bar(x, monthly['days_both_tracked'], bar_width,
                              color='steelblue', alpha=0.8, label='Both tracked')
            axes[plot_idx].set_title('Data Coverage (days with both partners tracked)', fontsize=12)
            axes[plot_idx].set_ylabel('Days', fontsize=10)
            axes[plot_idx].legend(loc='upper right', fontsize=9)
            axes[plot_idx].set_xticks(x)
            axes[plot_idx].set_xticklabels(months_str, rotation=45, ha='right')
            plot_idx += 1

        # Days together
        axes[plot_idx].bar(x, monthly['unique_days'], bar_width, color='steelblue')
        axes[plot_idx].set_title('Unique Days Together per Month', fontsize=12)
        axes[plot_idx].set_ylabel('Days', fontsize=10)
        axes[plot_idx].set_xticks(x)
        axes[plot_idx].set_xticklabels(months_str, rotation=45, ha='right')
        plot_idx += 1

        # Nights together
        axes[plot_idx].bar(x, monthly['nights'], bar_width, color='darkgreen')
        axes[plot_idx].set_title('Nights Together per Month', fontsize=12)
        axes[plot_idx].set_ylabel('Nights', fontsize=10)
        axes[plot_idx].set_xticks(x)
        axes[plot_idx].set_xticklabels(months_str, rotation=45, ha='right')
        plot_idx += 1

        # Co-locations count
        axes[plot_idx].bar(x, monthly['total_colocations'], bar_width, color='purple')
        axes[plot_idx].set_title('Total Co-locations per Month', fontsize=12)
        axes[plot_idx].set_ylabel('Count', fontsize=10)
        axes[plot_idx].set_xlabel('Month', fontsize=10)
        axes[plot_idx].set_xticks(x)
        axes[plot_idx].set_xticklabels(months_str, rotation=45, ha='right')

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
        # Group by day - use config start_date as fixed start
        date_range = pd.date_range(
            start=self.config['start_date'],
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

        # Find gap regions (consecutive days with 0 reports) for shading
        def find_gaps(series, min_gap_days=3):
            """Find contiguous runs of zeros, return list of (start, end) dates."""
            gaps = []
            gap_start = None
            for i, (date, val) in enumerate(zip(stats_df['date'], series)):
                if val == 0:
                    if gap_start is None:
                        gap_start = date
                else:
                    if gap_start is not None:
                        gap_end = stats_df['date'].iloc[i - 1]
                        if (gap_end - gap_start).days >= min_gap_days:
                            gaps.append((gap_start, gap_end))
                        gap_start = None
            if gap_start is not None:
                gap_end = stats_df['date'].iloc[-1]
                if (gap_end - gap_start).days >= min_gap_days:
                    gaps.append((gap_start, gap_end))
            return gaps

        p1_gaps = find_gaps(stats_df['p1_reports'])
        p2_gaps = find_gaps(stats_df['p2_reports'])

        # Partner 1 reports
        axes[0].bar(dates, stats_df['p1_reports'], color='steelblue', alpha=0.7, label='Raw reports')
        axes[0].bar(dates, stats_df['p1_interpolated'], color='lightblue', alpha=0.7, label='Interpolated')
        for gs, ge in p1_gaps:
            axes[0].axvspan(gs, ge, alpha=0.15, color='red', zorder=0)
            days = (ge - gs).days + 1
            mid = gs + (ge - gs) / 2
            axes[0].annotate(f'No data\n({days}d)', xy=(mid, axes[0].get_ylim()[1] * 0.5 if axes[0].get_ylim()[1] > 0 else 1),
                           fontsize=7, ha='center', color='red', alpha=0.8)
        axes[0].set_ylabel('Sean (reports)', fontsize=10)
        axes[0].legend(loc='upper right')
        axes[0].set_title('Data Density and Interpolation Usage', fontsize=12)

        # Partner 2 reports
        axes[1].bar(dates, stats_df['p2_reports'], color='darkgreen', alpha=0.7, label='Raw reports')
        axes[1].bar(dates, stats_df['p2_interpolated'], color='lightgreen', alpha=0.7, label='Interpolated')
        for gs, ge in p2_gaps:
            axes[1].axvspan(gs, ge, alpha=0.15, color='red', zorder=0)
            days = (ge - gs).days + 1
            mid = gs + (ge - gs) / 2
            axes[1].annotate(f'No data\n({days}d)', xy=(mid, axes[1].get_ylim()[1] * 0.5 if axes[1].get_ylim()[1] > 0 else 1),
                           fontsize=7, ha='center', color='red', alpha=0.8)
        axes[1].set_ylabel('Maia (reports)', fontsize=10)
        axes[1].legend(loc='upper right')

        # Overlaps — shade where EITHER partner has a gap
        axes[2].bar(dates, stats_df['overlaps'], color='purple', alpha=0.7)
        all_gaps = set()
        for gs, ge in p1_gaps + p2_gaps:
            axes[2].axvspan(gs, ge, alpha=0.15, color='red', zorder=0)
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

    def create_ind_calendar(
        self,
        partner1_df: pd.DataFrame,
        partner2_df: pd.DataFrame,
        overlaps_df: pd.DataFrame,
        output_file: Path,
        title: str = "Daily Cohabitation Calendar",
        ensemble_days: dict = None
    ) -> None:
        """Create IND-focused daily calendar with 4 discrete states.

        One cell per day in a GitHub-style layout (weeks x days).

        Args:
            partner1_df: Partner 1 location data
            partner2_df: Partner 2 location data
            overlaps_df: Overlap data (used to determine together days)
            output_file: Output file path
            title: Chart title
            ensemble_days: Optional dict date -> 'together'|'apart'|'ml_together'|'no_data'
        """
        # Get date range - use config start_date as fixed start
        all_dfs = [df for df in [partner1_df, partner2_df, overlaps_df]
                   if df is not None and not df.empty]
        if not all_dfs:
            print("No data to visualize for IND calendar")
            return

        start_date = datetime.fromisoformat(self.config['start_date']).date()
        end_date = max(df['timestamp'].max().date() for df in all_dfs)

        # Compute per-day sets
        p1_days = set()
        p2_days = set()
        together_days = set()
        if partner1_df is not None and not partner1_df.empty:
            p1_days = set(partner1_df['timestamp'].dt.date.unique())
        if partner2_df is not None and not partner2_df.empty:
            p2_days = set(partner2_df['timestamp'].dt.date.unique())
        if overlaps_df is not None and not overlaps_df.empty:
            together_days = set(overlaps_df['timestamp'].dt.date.unique())

        # Create complete date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Classify each day: together=2, ml_together=1, no_data=0, apart=-1
        first_dow = date_range[0].dayofweek  # 0=Mon, 6=Sun
        current_week = [np.nan] * first_dow  # padding
        current_padding = [True] * first_dow
        weeks_data = []
        weeks_padding = []

        tracked_days = 0
        together_count = 0
        ml_rescued_count = 0

        for date in date_range:
            d = date.date()
            both_tracked = d in p1_days and d in p2_days

            if ensemble_days and d in ensemble_days:
                state = ensemble_days[d]
                if state == 'together':
                    val = 2
                    tracked_days += 1
                    together_count += 1
                elif state == 'ml_together':
                    val = 1
                    tracked_days += 1
                    ml_rescued_count += 1
                elif state == 'apart':
                    val = -1
                    tracked_days += 1
                else:
                    val = 0
            else:
                is_together = d in together_days and both_tracked
                if is_together:
                    val = 2
                    tracked_days += 1
                    together_count += 1
                elif both_tracked:
                    val = -1
                    tracked_days += 1
                else:
                    val = 0

            current_week.append(val)
            current_padding.append(False)

            if date.dayofweek == 6:  # Sunday
                weeks_data.append(current_week)
                weeks_padding.append(current_padding)
                current_week = []
                current_padding = []

        # Pad incomplete final week
        if current_week:
            while len(current_week) < 7:
                current_week.append(np.nan)
                current_padding.append(True)
            weeks_data.append(current_week)
            weeks_padding.append(current_padding)

        cal_array = np.array(weeks_data).T  # 7 rows x N weeks
        padding_array = np.array(weeks_padding).T

        # Create figure
        n_weeks = cal_array.shape[1]
        fig_width = max(12, n_weeks * 0.35)
        fig, ax = plt.subplots(figsize=(fig_width, 4))

        # Discrete 4-color colormap: red, gray, blue (ML), green (GPS)
        cmap = ListedColormap(['#E74C3C', '#D0D0D0', '#5DADE2', '#2ECC71'])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)

        # Mask padding cells
        cal_masked = np.ma.masked_where(padding_array, cal_array)

        im = ax.imshow(cal_masked, aspect='auto', cmap=cmap, norm=norm,
                       interpolation='nearest')

        # Axis labels
        ax.set_yticks(range(7))
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

        # X-axis: month labels instead of week numbers
        # Find first day of each month in the date range
        month_positions = []
        month_labels_list = []
        for i, date in enumerate(date_range):
            week_idx = (first_dow + i) // 7
            if date.day == 1 or i == 0:
                month_positions.append(week_idx)
                month_labels_list.append(date.strftime('%b %Y'))

        ax.set_xticks(month_positions)
        ax.set_xticklabels(month_labels_list, rotation=45, fontsize=8, ha='right')

        ax.set_title(title, fontsize=14, pad=20)

        # Grid lines
        ax.set_xticks(np.arange(n_weeks) - 0.5, minor=True)
        ax.set_yticks(np.arange(7) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

        # Legend
        legend_elements = [
            Patch(facecolor='#2ECC71', label='Together (GPS)'),
            Patch(facecolor='#5DADE2', label='Together (ML)'),
            Patch(facecolor='#E74C3C', label='Apart'),
            Patch(facecolor='#D0D0D0', label='No data'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
                  framealpha=0.9, bbox_to_anchor=(0, 1.15), ncol=4)

        # Summary stats annotation
        total_together = together_count + ml_rescued_count
        if tracked_days > 0:
            pct = total_together / tracked_days * 100
            summary_text = (f"{total_together}/{tracked_days} tracked days together ({pct:.0f}%) "
                           f"[{together_count} GPS + {ml_rescued_count} ML]")
        else:
            summary_text = "0 tracked days"
        ax.annotate(
            summary_text,
            xy=(1.0, -0.12), xycoords='axes fraction',
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
        )

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"IND calendar saved to {output_file}")

    def create_all_visualizations(
        self,
        partner1_df: pd.DataFrame,
        partner2_df: pd.DataFrame,
        overlaps_df: pd.DataFrame,
        overlaps_interp_df: pd.DataFrame,
        presence_matrix: Optional[pd.DataFrame],
        output_dir: Path,
        state_matrix: Optional[pd.DataFrame] = None,
        ensemble_days: Optional[dict] = None
    ) -> None:
        """Create all visualization types.

        Args:
            partner1_df: Partner 1 data (interpolated)
            partner2_df: Partner 2 data (interpolated)
            overlaps_df: Raw overlaps
            overlaps_interp_df: Overlaps with interpolation
            presence_matrix: Hourly presence matrix (legacy, unused if state_matrix provided)
            output_dir: Output directory
            state_matrix: 3-state hourly matrix (co_located/apart/no_data)
            ensemble_days: Optional dict date -> 'together'|'apart'|'ml_together'|'no_data'
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating visualizations...")

        # Calendar heatmap (raw)
        self.create_calendar_heatmap(
            overlaps_df,
            output_dir / "calendar_raw.png",
            title="Co-Presence Calendar (Raw Data)",
            partner1_df=partner1_df,
            partner2_df=partner2_df
        )

        # Calendar heatmap (interpolated)
        self.create_calendar_heatmap(
            overlaps_interp_df,
            output_dir / "calendar_interpolated.png",
            title="Co-Presence Calendar (With Interpolation)",
            partner1_df=partner1_df,
            partner2_df=partner2_df
        )

        # Hourly matrix (with ensemble overlay if available)
        if state_matrix is not None and not state_matrix.empty:
            self.create_hourly_matrix(
                state_matrix,
                output_dir / "hourly_matrix.png",
                title="Hourly Co-Presence (GPS + ML Ensemble)",
                ensemble_days=ensemble_days
            )

        # Monthly summary
        self.create_monthly_summary(
            overlaps_interp_df,
            output_dir / "monthly_summary.png",
            partner1_df=partner1_df,
            partner2_df=partner2_df
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

        # IND-focused daily calendar (with ensemble)
        self.create_ind_calendar(
            partner1_df,
            partner2_df,
            overlaps_interp_df,
            output_dir / "ind_calendar.png",
            ensemble_days=ensemble_days
        )

        print(f"\nAll visualizations saved to {output_dir}")
