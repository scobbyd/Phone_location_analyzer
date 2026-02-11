"""Main analysis engine for location overlap detection.

Improved version of the original analyzer with:
- Uses interpolator for gap filling
- Better country detection
- Configurable via YAML
- Type hints and clean code
- Handles multiple phones per person
"""

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import yaml

try:
    from .interpolator import LocationInterpolator
except ImportError:
    from interpolator import LocationInterpolator


class LocationAnalyzer:
    """Analyzes location data to find co-location evidence."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize analyzer with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.interpolator = LocationInterpolator(
            max_gap_hours=self.config['interpolation']['max_gap_hours'],
            home_tolerance_m=self.config['interpolation']['home_tolerance_meters'],
            overnight_start=self.config['interpolation']['overnight_hours']['start'],
            overnight_end=self.config['interpolation']['overnight_hours']['end']
        )

        # Data storage
        self.partner1_data: Optional[pd.DataFrame] = None
        self.partner2_data: Optional[pd.DataFrame] = None
        self.partner1_interpolated: Optional[pd.DataFrame] = None
        self.partner2_interpolated: Optional[pd.DataFrame] = None
        self.overlaps: Optional[pd.DataFrame] = None
        self.overlaps_with_interpolation: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

        self.start_date = datetime.fromisoformat(self.config['start_date'])

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def parse_coordinate_string(self, coord_str: str) -> Tuple[Optional[float], Optional[float]]:
        """Parse various coordinate formats from location data.

        Args:
            coord_str: Coordinate string in various formats

        Returns:
            (lat, lon) tuple or (None, None) if parsing fails
        """
        try:
            if 'geo:' in coord_str:
                # Format: "geo:-34.565782,-58.448855"
                parts = coord_str.replace('geo:', '').split(',')
                return float(parts[0]), float(parts[1])
            elif '°' in coord_str:
                # Format: "50.9877894°, 5.7707602°"
                parts = coord_str.replace('°', '').split(',')
                return float(parts[0]), float(parts[1].strip())
            else:
                return None, None
        except (ValueError, IndexError):
            return None, None

    def load_google_timeline(self, file_path: Path) -> pd.DataFrame:
        """Load Google Timeline JSON (semanticSegments format).

        Args:
            file_path: Path to JSON file

        Returns:
            DataFrame with location data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            locations = []

            if 'semanticSegments' in data:
                for segment in data['semanticSegments']:
                    # Extract from timelinePath
                    if 'timelinePath' in segment:
                        for path_point in segment['timelinePath']:
                            if 'point' in path_point and 'time' in path_point:
                                lat, lon = self.parse_coordinate_string(path_point['point'])
                                if lat and lon:
                                    timestamp = datetime.fromisoformat(
                                        path_point['time'].replace('Z', '+00:00')
                                    )
                                    timestamp = timestamp.replace(tzinfo=None)

                                    if timestamp >= self.start_date:
                                        locations.append({
                                            'lat': lat,
                                            'lon': lon,
                                            'timestamp': timestamp,
                                            'accuracy': 10,
                                            'source_file': str(file_path.name)
                                        })

                    # Extract from visit locations
                    if 'visit' in segment and 'topCandidate' in segment['visit']:
                        candidate = segment['visit']['topCandidate']
                        place_loc = candidate.get('placeLocation', {})
                        if isinstance(place_loc, dict) and 'latLng' in place_loc:
                            lat, lon = self.parse_coordinate_string(place_loc['latLng'])
                        elif isinstance(place_loc, str):
                            lat, lon = self.parse_coordinate_string(place_loc)
                        else:
                            lat, lon = None, None

                        if lat and lon and 'startTime' in segment:
                            timestamp = datetime.fromisoformat(
                                segment['startTime'].replace('Z', '+00:00')
                            )
                            timestamp = timestamp.replace(tzinfo=None)

                            if timestamp >= self.start_date:
                                locations.append({
                                    'lat': lat,
                                    'lon': lon,
                                    'timestamp': timestamp,
                                    'accuracy': 20,
                                    'place_type': candidate.get('semanticType', ''),
                                    'source_file': str(file_path.name)
                                })
                                # Also record the end time (same location)
                                if 'endTime' in segment:
                                    end_ts = datetime.fromisoformat(
                                        segment['endTime'].replace('Z', '+00:00')
                                    )
                                    end_ts = end_ts.replace(tzinfo=None)
                                    if end_ts >= self.start_date:
                                        locations.append({
                                            'lat': lat,
                                            'lon': lon,
                                            'timestamp': end_ts,
                                            'accuracy': 20,
                                            'place_type': candidate.get('semanticType', ''),
                                            'source_file': str(file_path.name)
                                        })

            return pd.DataFrame(locations)

        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return pd.DataFrame()

    def load_apple_timeline(self, file_path: Path) -> pd.DataFrame:
        """Load Apple/iPhone Timeline JSON.

        Args:
            file_path: Path to JSON file

        Returns:
            DataFrame with location data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            locations = []

            for item in data:
                # Extract from visit locations
                if 'visit' in item and 'topCandidate' in item['visit']:
                    candidate = item['visit']['topCandidate']
                    place_loc = candidate.get('placeLocation', '')
                    # placeLocation can be a string ("geo:lat,lon") or dict with latLng
                    if isinstance(place_loc, str):
                        lat, lon = self.parse_coordinate_string(place_loc)
                    elif isinstance(place_loc, dict) and 'latLng' in place_loc:
                        lat, lon = self.parse_coordinate_string(place_loc['latLng'])
                    else:
                        lat, lon = None, None

                    if lat and lon and 'startTime' in item:
                        timestamp = datetime.fromisoformat(
                            item['startTime'].replace('Z', '+00:00')
                        )
                        timestamp = timestamp.replace(tzinfo=None)

                        if timestamp >= self.start_date:
                            locations.append({
                                'lat': lat,
                                'lon': lon,
                                'timestamp': timestamp,
                                'accuracy': 20,
                                'source_file': str(file_path.name)
                            })
                            # Also add end time as a separate point (same location)
                            if 'endTime' in item:
                                end_ts = datetime.fromisoformat(
                                    item['endTime'].replace('Z', '+00:00')
                                )
                                end_ts = end_ts.replace(tzinfo=None)
                                if end_ts >= self.start_date:
                                    locations.append({
                                        'lat': lat,
                                        'lon': lon,
                                        'timestamp': end_ts,
                                        'accuracy': 20,
                                        'source_file': str(file_path.name)
                                    })

                # Extract from activity records (start/end geo points)
                if 'activity' in item and 'startTime' in item:
                    activity = item['activity']
                    for geo_key, time_key in [('start', 'startTime'), ('end', 'endTime')]:
                        geo_str = activity.get(geo_key, '')
                        if geo_str and time_key in item:
                            lat, lon = self.parse_coordinate_string(geo_str)
                            if lat and lon:
                                timestamp = datetime.fromisoformat(
                                    item[time_key].replace('Z', '+00:00')
                                )
                                timestamp = timestamp.replace(tzinfo=None)
                                if timestamp >= self.start_date:
                                    locations.append({
                                        'lat': lat,
                                        'lon': lon,
                                        'timestamp': timestamp,
                                        'accuracy': 30,
                                        'source_file': str(file_path.name)
                                    })

                # Extract from timelinePath (Google's newer format)
                if 'timelinePath' in item and 'startTime' in item:
                    base_time = datetime.fromisoformat(
                        item['startTime'].replace('Z', '+00:00')
                    )
                    base_time = base_time.replace(tzinfo=None)

                    for path_point in item['timelinePath']:
                        if 'point' in path_point:
                            lat, lon = self.parse_coordinate_string(path_point['point'])
                            if lat and lon:
                                offset_minutes = path_point.get('durationMinutesOffsetFromStartTime', 0)
                                if isinstance(offset_minutes, str):
                                    offset_minutes = int(offset_minutes)
                                timestamp = base_time + timedelta(minutes=offset_minutes)

                                if timestamp >= self.start_date:
                                    locations.append({
                                        'lat': lat,
                                        'lon': lon,
                                        'timestamp': timestamp,
                                        'accuracy': 15,
                                        'source_file': str(file_path.name)
                                    })

            return pd.DataFrame(locations)

        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return pd.DataFrame()

    def load_multiple_files(
        self,
        file_list: List[str],
        person_label: str,
        base_path: Path
    ) -> pd.DataFrame:
        """Load and combine multiple location files for one person.

        Args:
            file_list: List of filenames
            person_label: Label for this person (for metadata)
            base_path: Base directory containing files

        Returns:
            Combined DataFrame with all location data
        """
        all_data = []

        for file_name in file_list:
            file_path = base_path / file_name
            if not file_path.exists():
                print(f"  Warning: File not found: {file_path}")
                continue

            print(f"Loading {file_name} for {person_label}...")

            # Auto-detect format
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(500)

            # Try appropriate parser
            if 'semanticSegments' in sample:
                print("  Detected: Google/Android format")
                data = self.load_google_timeline(file_path)
            elif 'visit' in sample or 'activity' in sample or sample.lstrip().startswith('['):
                print("  Detected: Apple/iPhone format")
                data = self.load_apple_timeline(file_path)
            else:
                # Try both
                data = self.load_google_timeline(file_path)
                if data.empty:
                    data = self.load_apple_timeline(file_path)

            if not data.empty:
                all_data.append(data)
                print(f"  Loaded {len(data)} locations")

                # Store metadata
                self.metadata[f"{person_label}_{file_name}"] = {
                    'records': len(data),
                    'date_range': f"{data['timestamp'].min()} to {data['timestamp'].max()}",
                    'file_hash': self._calculate_file_hash(file_path)
                }
            else:
                print(f"  No data loaded from {file_name}")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            # Deduplicate
            combined = combined.drop_duplicates(subset=['timestamp', 'lat', 'lon'])
            combined = combined.sort_values('timestamp').reset_index(drop=True)
            print(f"Total unique locations for {person_label}: {len(combined)}")
            return combined

        return pd.DataFrame()

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash for file verification.

        Args:
            file_path: Path to file

        Returns:
            First 16 chars of SHA256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]

    def find_overlaps(
        self,
        partner1_df: pd.DataFrame,
        partner2_df: pd.DataFrame,
        use_interpolation: bool = False
    ) -> pd.DataFrame:
        """Find co-locations between two people.

        Args:
            partner1_df: Partner 1 location data
            partner2_df: Partner 2 location data
            use_interpolation: Whether to use interpolated data

        Returns:
            DataFrame of overlap events
        """
        overlaps = []

        distance_threshold = self.config['thresholds']['distance_meters']
        time_threshold = timedelta(minutes=self.config['thresholds']['time_minutes'])

        # Sort and index for efficiency
        df1 = partner1_df.sort_values('timestamp')
        df2 = partner2_df.sort_values('timestamp')
        df2_indexed = df2.set_index('timestamp').sort_index()

        for _, row1 in df1.iterrows():
            start_time = row1['timestamp'] - time_threshold
            end_time = row1['timestamp'] + time_threshold

            try:
                candidates = df2_indexed[start_time:end_time]

                for timestamp2, row2 in candidates.iterrows():
                    dist = geodesic(
                        (row1['lat'], row1['lon']),
                        (row2['lat'], row2['lon'])
                    ).meters

                    if dist <= distance_threshold:
                        overlaps.append({
                            'lat': (row1['lat'] + row2['lat']) / 2,
                            'lon': (row1['lon'] + row2['lon']) / 2,
                            'timestamp': row1['timestamp'],
                            'distance_meters': dist,
                            'partner1_lat': row1['lat'],
                            'partner1_lon': row1['lon'],
                            'partner2_lat': row2['lat'],
                            'partner2_lon': row2['lon'],
                            'partner1_source': row1.get('source_file', 'unknown'),
                            'partner2_source': row2.get('source_file', 'unknown'),
                            'interpolated': use_interpolation and (
                                row1.get('interpolated', False) or
                                row2.get('interpolated', False)
                            )
                        })
            except KeyError:
                continue

        return pd.DataFrame(overlaps)

    def detect_country(self, lat: float, lon: float) -> str:
        """Detect country from coordinates using configured bounds.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Country name or 'Other'
        """
        bounds = self.config['country_bounds']

        # Check South Limburg first (most specific)
        limburg = bounds['netherlands']['limburg']
        if limburg[0] <= lat <= limburg[1] and limburg[2] <= lon <= limburg[3]:
            if lat < 50.75:
                return "Belgium"
            elif lon > 6.0:
                return "Germany"
            else:
                return "Netherlands (South Limburg)"

        # Check other countries
        for country, bound in bounds.items():
            if country == 'netherlands':
                general = bound['general']
                if general[0] <= lat <= general[1] and general[2] <= lon <= general[3]:
                    return "Netherlands"
            else:
                if bound[0] <= lat <= bound[1] and bound[2] <= lon <= bound[3]:
                    return country.title()

        return "Other"

    def run_analysis(self, base_path: Path) -> Dict[str, Any]:
        """Run complete analysis pipeline.

        Args:
            base_path: Base directory containing data files

        Returns:
            Dictionary with analysis results
        """
        print("=" * 60)
        print("Location History Analysis - Immigration Evidence")
        print("=" * 60)
        print(f"Analysis start date: {self.start_date.date()}")
        print()

        # Load data
        print("Loading location data...")
        self.partner1_data = self.load_multiple_files(
            self.config['data_files']['partner1'],
            "Partner1",
            base_path
        )

        self.partner2_data = self.load_multiple_files(
            self.config['data_files']['partner2'],
            "Partner2",
            base_path
        )

        if self.partner1_data.empty or self.partner2_data.empty:
            print("\nError: Could not load location data")
            return {}

        print("\nApplying interpolation...")
        self.partner1_interpolated = self.interpolator.interpolate_presence(
            self.partner1_data
        )
        self.partner2_interpolated = self.interpolator.interpolate_presence(
            self.partner2_data
        )

        print(f"  Partner 1: {len(self.partner1_data)} -> {len(self.partner1_interpolated)} points (gap fill)")
        print(f"  Partner 2: {len(self.partner2_data)} -> {len(self.partner2_interpolated)} points (gap fill)")

        # Infer overnight stays
        print("  Inferring overnight stays...")
        self.partner1_interpolated = self.interpolator.infer_overnight_stays(
            self.partner1_interpolated
        )
        self.partner2_interpolated = self.interpolator.infer_overnight_stays(
            self.partner2_interpolated
        )

        print(f"  Partner 1: {len(self.partner1_interpolated)} points (with overnight)")
        print(f"  Partner 2: {len(self.partner2_interpolated)} points (with overnight)")

        # Find overlaps (both with and without interpolation)
        print("\nFinding co-locations (raw data)...")
        self.overlaps = self.find_overlaps(
            self.partner1_data,
            self.partner2_data,
            use_interpolation=False
        )
        print(f"  Found {len(self.overlaps)} co-locations")

        print("\nFinding co-locations (with interpolation)...")
        self.overlaps_with_interpolation = self.find_overlaps(
            self.partner1_interpolated,
            self.partner2_interpolated,
            use_interpolation=True
        )
        print(f"  Found {len(self.overlaps_with_interpolation)} co-locations")

        # Generate statistics
        stats = self._generate_statistics()

        print("\n" + "=" * 60)
        print("Analysis Complete")
        print("=" * 60)
        print(f"Days together (raw): {stats['raw']['unique_days']}")
        print(f"Days together (interpolated): {stats['interpolated']['unique_days']}")
        print(f"Nights together: {stats['raw']['nights_together']}")
        print()

        return stats

    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive statistics.

        Returns:
            Dictionary with raw and interpolated statistics
        """
        stats = {
            'raw': self._stats_for_dataframe(self.overlaps),
            'interpolated': self._stats_for_dataframe(self.overlaps_with_interpolation)
        }

        return stats

    def _stats_for_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for a single overlap DataFrame.

        Args:
            df: Overlap DataFrame

        Returns:
            Dictionary of statistics
        """
        if df.empty:
            return {}

        return {
            'total_colocations': len(df),
            'unique_days': df['timestamp'].dt.date.nunique(),
            'date_range': f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}",
            'avg_distance_m': df['distance_meters'].mean(),
            'nights_together': df[
                (df['timestamp'].dt.hour >= 22) | (df['timestamp'].dt.hour <= 6)
            ]['timestamp'].dt.date.nunique()
        }
