"""Presence interpolation for sparse location data.

Phones that aren't moving report infrequently (every 1-4 hours).
This module infers continuous presence between sparse reports.

Key innovation: Fill gaps between location reports to provide more
complete evidence of co-location, while maintaining transparency.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from geopy.distance import geodesic


class LocationInterpolator:
    """Infers continuous presence from sparse location data."""

    def __init__(
        self,
        max_gap_hours: float = 4.0,
        home_tolerance_m: float = 300.0,
        overnight_start: int = 22,
        overnight_end: int = 6
    ):
        """Initialize interpolator with configuration.

        Args:
            max_gap_hours: Maximum gap to interpolate across (hours)
            home_tolerance_m: Distance tolerance for "same location" (meters)
            overnight_start: Hour when evening starts (24h format)
            overnight_end: Hour when morning ends (24h format)
        """
        self.max_gap_hours = max_gap_hours
        self.max_gap_timedelta = timedelta(hours=max_gap_hours)
        self.home_tolerance_m = home_tolerance_m
        self.overnight_start = overnight_start
        self.overnight_end = overnight_end

    def interpolate_presence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic presence points between sparse reports.

        When two consecutive reports are at the same location (within tolerance),
        generate points at regular intervals to fill the gap. This addresses
        the core sparse-data problem: phones don't report when stationary.

        Args:
            df: DataFrame with columns ['timestamp', 'lat', 'lon']

        Returns:
            DataFrame with original + interpolated points, marked with 'interpolated' column
        """
        if df.empty:
            return df

        # Ensure sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Add marker for original data
        df['interpolated'] = False

        interpolated_rows = []

        for i in range(len(df) - 1):
            current = df.iloc[i]
            next_point = df.iloc[i + 1]

            # Calculate time gap
            time_gap = next_point['timestamp'] - current['timestamp']
            gap_hours = time_gap.total_seconds() / 3600

            # Skip very short gaps (< 30 min) or very long gaps
            if gap_hours < 0.5 or gap_hours > self.max_gap_hours:
                continue

            # Calculate distance
            dist = geodesic(
                (current['lat'], current['lon']),
                (next_point['lat'], next_point['lon'])
            ).meters

            # Only interpolate if at same location
            if dist <= self.home_tolerance_m:
                # Generate points every 30 minutes between them
                intervals = int(gap_hours * 2)  # 30-min intervals
                for interval in range(1, intervals):
                    minutes_offset = interval * 30
                    interp_time = current['timestamp'] + timedelta(minutes=minutes_offset)

                    # Use midpoint coordinates
                    alpha = interval / intervals
                    interp_lat = current['lat'] * (1 - alpha) + next_point['lat'] * alpha
                    interp_lon = current['lon'] * (1 - alpha) + next_point['lon'] * alpha

                    interpolated_rows.append({
                        'timestamp': interp_time,
                        'lat': interp_lat,
                        'lon': interp_lon,
                        'interpolated': True,
                        'accuracy': 50,
                        'source_file': f"interpolated_{current.get('source_file', 'unknown')}"
                    })

        # Combine original and interpolated
        if interpolated_rows:
            interp_df = pd.DataFrame(interpolated_rows)
            result = pd.concat([df, interp_df], ignore_index=True)
            result = result.sort_values('timestamp').reset_index(drop=True)
            return result

        return df

    def infer_overnight_stays(
        self,
        df: pd.DataFrame,
        home_location: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """Infer full overnight presence from evening + morning reports.

        If evening report (after overnight_start) is near home and next morning
        report (before overnight_end) is also near home, generate hourly points
        for the full night.

        Args:
            df: DataFrame with location data
            home_location: Optional (lat, lon) of home. If None, auto-detect.

        Returns:
            DataFrame with inferred overnight stays added
        """
        if df.empty:
            return df

        df = df.sort_values('timestamp').reset_index(drop=True)

        # Auto-detect home if not provided
        if home_location is None:
            home_location = self._detect_home_location(df)
            if home_location is None:
                return df  # Can't infer without home location

        overnight_rows = []

        for i in range(len(df) - 1):
            current = df.iloc[i]
            next_point = df.iloc[i + 1]

            # Check for overnight gap: evening report followed by morning report next day
            # Be flexible with timing - any report after 19:00 paired with next-day report before 10:00
            is_evening = current['timestamp'].hour >= 19
            is_next_morning = next_point['timestamp'].hour <= 10
            is_next_day = next_point['timestamp'].date() > current['timestamp'].date()
            gap_hours = (next_point['timestamp'] - current['timestamp']).total_seconds() / 3600

            if not (is_evening and is_next_morning and is_next_day and gap_hours <= 16):
                continue

            # Check both near home
            dist_current = geodesic(
                (current['lat'], current['lon']),
                home_location
            ).meters

            dist_next = geodesic(
                (next_point['lat'], next_point['lon']),
                home_location
            ).meters

            if dist_current <= self.home_tolerance_m and dist_next <= self.home_tolerance_m:
                # Generate hourly overnight points
                night_start = current['timestamp'].replace(
                    hour=self.overnight_start, minute=0, second=0
                )
                if night_start < current['timestamp']:
                    night_start += timedelta(days=1)

                night_end = next_point['timestamp'].replace(
                    hour=self.overnight_end, minute=0, second=0
                )

                current_hour = night_start
                while current_hour <= night_end:
                    overnight_rows.append({
                        'timestamp': current_hour,
                        'lat': home_location[0],
                        'lon': home_location[1],
                        'interpolated': True,
                        'overnight_inferred': True,
                        'source_file': 'overnight_inference'
                    })
                    current_hour += timedelta(hours=1)

        if overnight_rows:
            overnight_df = pd.DataFrame(overnight_rows)
            result = pd.concat([df, overnight_df], ignore_index=True)
            result = result.drop_duplicates(subset=['timestamp', 'lat', 'lon'])
            result = result.sort_values('timestamp').reset_index(drop=True)
            return result

        return df

    def _detect_home_location(self, df: pd.DataFrame) -> Optional[Tuple[float, float]]:
        """Auto-detect home location from overnight reports.

        Args:
            df: Location DataFrame

        Returns:
            (lat, lon) of most common overnight location, or None
        """
        overnight = df[
            (df['timestamp'].dt.hour >= self.overnight_start) |
            (df['timestamp'].dt.hour <= self.overnight_end)
        ]

        if overnight.empty:
            return None

        # Cluster by rounding coordinates
        overnight_copy = overnight.copy()
        overnight_copy['lat_rounded'] = overnight_copy['lat'].round(2)
        overnight_copy['lon_rounded'] = overnight_copy['lon'].round(2)

        # Find most common location
        location_counts = overnight_copy.groupby(['lat_rounded', 'lon_rounded']).size()
        if location_counts.empty:
            return None

        most_common = location_counts.idxmax()
        return most_common

    def calculate_presence_probability(
        self,
        partner1_df: pd.DataFrame,
        partner2_df: pd.DataFrame,
        target_time: datetime,
        time_window_hours: float = 2.0
    ) -> float:
        """Calculate probability both were at same location at target time.

        Looks at nearest reports (including interpolated) within time window
        and calculates likelihood of co-presence.

        Args:
            partner1_df: Partner 1 location data
            partner2_df: Partner 2 location data
            target_time: Time to check
            time_window_hours: Search window around target time

        Returns:
            Probability 0.0-1.0 of co-presence
        """
        window = timedelta(hours=time_window_hours)

        # Find nearest reports for each partner
        p1_nearby = partner1_df[
            (partner1_df['timestamp'] >= target_time - window) &
            (partner1_df['timestamp'] <= target_time + window)
        ]

        p2_nearby = partner2_df[
            (partner2_df['timestamp'] >= target_time - window) &
            (partner2_df['timestamp'] <= target_time + window)
        ]

        if p1_nearby.empty or p2_nearby.empty:
            return 0.0

        # Get closest reports
        p1_closest_idx = (p1_nearby['timestamp'] - target_time).abs().idxmin()
        p2_closest_idx = (p2_nearby['timestamp'] - target_time).abs().idxmin()
        p1_closest = p1_nearby.loc[p1_closest_idx]
        p2_closest = p2_nearby.loc[p2_closest_idx]

        # Calculate distance
        dist = geodesic(
            (p1_closest['lat'], p1_closest['lon']),
            (p2_closest['lat'], p2_closest['lon'])
        ).meters

        # Calculate probability based on distance and time offset
        if dist <= 150:
            base_prob = 1.0
        elif dist <= 300:
            base_prob = 0.7
        elif dist <= 500:
            base_prob = 0.4
        else:
            base_prob = 0.0

        # Reduce probability based on time offset
        p1_offset = abs((p1_closest['timestamp'] - target_time).total_seconds() / 3600)
        p2_offset = abs((p2_closest['timestamp'] - target_time).total_seconds() / 3600)
        max_offset = max(p1_offset, p2_offset)

        time_factor = max(0, 1 - (max_offset / time_window_hours))

        # Boost if both are interpolated (means they were stationary)
        if p1_closest.get('interpolated', False) and p2_closest.get('interpolated', False):
            interpolation_boost = 1.2
        else:
            interpolation_boost = 1.0

        final_prob = min(1.0, base_prob * time_factor * interpolation_boost)

        return final_prob

    def build_presence_matrix(
        self,
        partner1_df: pd.DataFrame,
        partner2_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Build 24h x days matrix of co-presence probabilities.

        Args:
            partner1_df: Partner 1 location data
            partner2_df: Partner 2 location data
            start_date: Analysis start
            end_date: Analysis end

        Returns:
            DataFrame with rows=hours (0-23), columns=dates, values=probability
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Initialize matrix
        matrix = pd.DataFrame(
            index=range(24),
            columns=date_range.date,
            dtype=float
        )

        for date in date_range:
            for hour in range(24):
                target_time = datetime.combine(date.date(), datetime.min.time())
                target_time = target_time.replace(hour=hour)

                prob = self.calculate_presence_probability(
                    partner1_df, partner2_df, target_time
                )

                matrix.loc[hour, date.date()] = prob

        return matrix
