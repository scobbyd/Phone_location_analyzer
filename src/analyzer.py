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
        interp_cfg = self.config['interpolation']
        thresh_cfg = self.config['thresholds']
        # Home region bounding box for drive-by filtering (Benelux + Germany)
        home_region_bounds = (47.0, 55.0, 2.5, 15.0)
        self.interpolator = LocationInterpolator(
            max_gap_hours=interp_cfg['max_gap_hours'],
            home_tolerance_m=interp_cfg['home_tolerance_meters'],
            overnight_start=interp_cfg['overnight_hours']['start'],
            overnight_end=interp_cfg['overnight_hours']['end'],
            travel_max_gap_hours=interp_cfg.get('travel_max_gap_hours', 18.0),
            city_radius_m=interp_cfg.get('city_radius_meters', 10000.0),
            neighborhood_m=thresh_cfg.get('neighborhood_meters', 750.0),
            city_m=thresh_cfg.get('city_meters', 5000.0),
            home_region_bounds=home_region_bounds,
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

        Handles phantom "home" records: Apple visit/activity end-records sometimes
        snap to the home address even when timelinePath data shows the person is
        elsewhere. We collect timelinePath coverage windows and suppress
        visit/activity records whose timestamps fall inside those windows when
        the visit location is far (>5km) from the timelinePath centroid.

        Args:
            file_path: Path to JSON file

        Returns:
            DataFrame with location data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # First pass: collect timelinePath time windows + centroids
            path_windows = []  # list of (start, end, centroid_lat, centroid_lon)
            for item in data:
                if 'timelinePath' in item and 'startTime' in item:
                    base_time = datetime.fromisoformat(
                        item['startTime'].replace('Z', '+00:00')
                    ).replace(tzinfo=None)
                    lats, lons = [], []
                    max_offset = 0
                    for pp in item['timelinePath']:
                        if 'point' in pp:
                            lat, lon = self.parse_coordinate_string(pp['point'])
                            if lat and lon:
                                lats.append(lat)
                                lons.append(lon)
                                off = pp.get('durationMinutesOffsetFromStartTime', 0)
                                if isinstance(off, str):
                                    off = int(off)
                                max_offset = max(max_offset, off)
                    if lats:
                        end_time = base_time + timedelta(minutes=max_offset)
                        path_windows.append((
                            base_time, end_time,
                            sum(lats) / len(lats), sum(lons) / len(lons)
                        ))

            def _is_phantom(ts, lat, lon):
                """Check if a visit/activity record is a phantom home snap."""
                for pw_start, pw_end, pw_lat, pw_lon in path_windows:
                    if pw_start <= ts <= pw_end:
                        dist = geodesic((lat, lon), (pw_lat, pw_lon)).meters
                        if dist > 5000:
                            return True
                return False

            locations = []

            for item in data:
                # Extract from visit locations
                if 'visit' in item and 'topCandidate' in item['visit']:
                    candidate = item['visit']['topCandidate']
                    place_loc = candidate.get('placeLocation', '')
                    if isinstance(place_loc, str):
                        lat, lon = self.parse_coordinate_string(place_loc)
                    elif isinstance(place_loc, dict) and 'latLng' in place_loc:
                        lat, lon = self.parse_coordinate_string(place_loc['latLng'])
                    else:
                        lat, lon = None, None

                    if lat and lon and 'startTime' in item:
                        timestamp = datetime.fromisoformat(
                            item['startTime'].replace('Z', '+00:00')
                        ).replace(tzinfo=None)

                        if timestamp >= self.start_date and not _is_phantom(timestamp, lat, lon):
                            locations.append({
                                'lat': lat, 'lon': lon,
                                'timestamp': timestamp,
                                'accuracy': 20,
                                'source_file': str(file_path.name)
                            })
                        if 'endTime' in item:
                            end_ts = datetime.fromisoformat(
                                item['endTime'].replace('Z', '+00:00')
                            ).replace(tzinfo=None)
                            if end_ts >= self.start_date and not _is_phantom(end_ts, lat, lon):
                                locations.append({
                                    'lat': lat, 'lon': lon,
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
                                ).replace(tzinfo=None)
                                if timestamp >= self.start_date and not _is_phantom(timestamp, lat, lon):
                                    locations.append({
                                        'lat': lat, 'lon': lon,
                                        'timestamp': timestamp,
                                        'accuracy': 30,
                                        'source_file': str(file_path.name)
                                    })

                # Extract from timelinePath
                if 'timelinePath' in item and 'startTime' in item:
                    base_time = datetime.fromisoformat(
                        item['startTime'].replace('Z', '+00:00')
                    ).replace(tzinfo=None)

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
                                        'lat': lat, 'lon': lon,
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
            combined = self._deduplicate_prefer_accurate(combined)
            combined = combined.sort_values('timestamp').reset_index(drop=True)
            print(f"Total unique locations for {person_label}: {len(combined)}")
            return combined

        return pd.DataFrame()

    @staticmethod
    def _deduplicate_prefer_accurate(df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate location records, preferring higher-accuracy sources.

        Apple visit end-records often report home location even when the
        person has moved. When a timelinePath point (accuracy=15) exists
        within 2 minutes of a visit record (accuracy=20+), keep only the
        timelinePath point.
        """
        if df.empty or 'accuracy' not in df.columns:
            return df.drop_duplicates(subset=['timestamp', 'lat', 'lon'])

        df = df.sort_values(['timestamp', 'accuracy']).reset_index(drop=True)

        # Round timestamp to 2-minute bins for dedup grouping
        df['_ts_bin'] = df['timestamp'].dt.floor('2min')
        # Within each bin, keep the row with the best (lowest) accuracy
        deduped = df.loc[df.groupby('_ts_bin')['accuracy'].idxmin()]
        deduped = deduped.drop(columns=['_ts_bin'])

        # Also drop exact coordinate duplicates
        deduped = deduped.drop_duplicates(subset=['timestamp', 'lat', 'lon'])
        return deduped.reset_index(drop=True)

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

    def _compute_speeds(self, df: pd.DataFrame) -> pd.Series:
        """Compute speed in km/h for each point based on distance to next point.

        Args:
            df: Sorted DataFrame with timestamp, lat, lon

        Returns:
            Series of speeds in km/h, indexed like df
        """
        speeds = pd.Series(0.0, index=df.index)
        for i in range(len(df) - 1):
            curr = df.iloc[i]
            nxt = df.iloc[i + 1]
            dt_hours = (nxt['timestamp'] - curr['timestamp']).total_seconds() / 3600
            if dt_hours > 0 and dt_hours < 2:  # Only compute for reasonable gaps
                dist_km = geodesic(
                    (curr['lat'], curr['lon']),
                    (nxt['lat'], nxt['lon'])
                ).km
                speeds.iloc[i] = dist_km / dt_hours
        return speeds

    def find_overlaps(
        self,
        partner1_df: pd.DataFrame,
        partner2_df: pd.DataFrame,
        use_interpolation: bool = False
    ) -> pd.DataFrame:
        """Find co-locations using geofence-aware matching with drive-by filter.

        Matching strategy:
        - If both points have geofence_radius, co-located when
          distance < sum of their geofence radii.
        - Falls back to fixed tier3 (city_meters) when no geofence data.
        - Drive-by filter: for wider matches (>150m), reject if one partner
          is clearly driving (>20 km/h) while the other is stationary (<2 km/h).
          This eliminates false positives from commuting past each other.

        Tier labels (for display/stats):
        - confirmed: <= distance_meters (150m)
        - neighborhood: <= neighborhood_meters (750m)
        - city: <= city_meters (5000m)

        Args:
            partner1_df: Partner 1 location data
            partner2_df: Partner 2 location data
            use_interpolation: Whether to use interpolated data

        Returns:
            DataFrame of overlap events with 'proximity_tier' column
        """
        overlaps = []

        tier1 = self.config['thresholds']['distance_meters']
        tier2 = self.config['thresholds'].get('neighborhood_meters', 750)
        tier3 = self.config['thresholds'].get('city_meters', 5000)
        time_threshold = timedelta(minutes=self.config['thresholds']['time_minutes'])

        has_geofence = (
            'geofence_radius' in partner1_df.columns
            and 'geofence_radius' in partner2_df.columns
        )

        df1 = partner1_df.sort_values('timestamp').reset_index(drop=True)
        df2 = partner2_df.sort_values('timestamp').reset_index(drop=True)

        # Pre-compute speeds for drive-by filtering
        speeds1 = self._compute_speeds(df1)
        speeds2 = self._compute_speeds(df2)
        df1 = df1.copy()
        df2 = df2.copy()
        df1['_speed'] = speeds1.values
        df2['_speed'] = speeds2.values

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

                    # Determine match threshold from geofence radii or fixed tier
                    if has_geofence:
                        r1 = row1.get('geofence_radius', tier1)
                        r2 = row2.get('geofence_radius', tier1)
                        if pd.isna(r1):
                            r1 = tier1
                        if pd.isna(r2):
                            r2 = tier1
                        max_dist = r1 + r2
                    else:
                        max_dist = tier3

                    if dist > max_dist:
                        continue

                    # Label tier based on absolute distance
                    if dist <= tier1:
                        tier_label = 'confirmed'
                    elif dist <= tier2:
                        tier_label = 'neighborhood'
                    else:
                        tier_label = 'city'

                    # Location-aware tier restriction:
                    # In the Netherlands, only confirmed (≤150m) matches to
                    # avoid false positives from commuting past each other.
                    # In foreign countries, allow wider tiers (travel together).
                    if dist > tier1:
                        country1 = self.detect_country(row1['lat'], row1['lon'])
                        country2 = self.detect_country(row2['lat'], row2['lon'])
                        both_home = ('Netherlands' in country1 or 'Belgium' in country1 or 'Germany' in country1)
                        if both_home:
                            # In home region: also apply drive-by filter
                            s1 = row1.get('_speed', 0)
                            s2 = row2.get('_speed', 0)
                            one_driving = s1 > 20 or s2 > 20
                            one_stationary = s1 < 2 or s2 < 2
                            if one_driving and one_stationary:
                                continue  # Drive-by, skip
                            # In home region: only allow neighborhood tier
                            # if both partners are moving slowly (walking/stationary)
                            if dist > tier2:
                                continue  # No city-tier matches in home region
                            both_slow = s1 < 7 and s2 < 7
                            if not both_slow:
                                continue  # One is moving fast, likely not together

                    overlaps.append({
                        'lat': (row1['lat'] + row2['lat']) / 2,
                        'lon': (row1['lon'] + row2['lon']) / 2,
                        'timestamp': row1['timestamp'],
                        'distance_meters': dist,
                        'proximity_tier': tier_label,
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

        # Filter GPS jitter (impossible speed = noise)
        p1_before = len(self.partner1_data)
        p2_before = len(self.partner2_data)
        self.partner1_data = self.interpolator.filter_jitter(self.partner1_data)
        self.partner2_data = self.interpolator.filter_jitter(self.partner2_data)
        p1_jitter = p1_before - len(self.partner1_data)
        p2_jitter = p2_before - len(self.partner2_data)
        if p1_jitter or p2_jitter:
            print(f"  Jitter filter: removed {p1_jitter} P1 + {p2_jitter} P2 noise points")

        # Tag raw data with geofence radii based on source accuracy
        self.partner1_data = self.interpolator.tag_geofence_radius(self.partner1_data)
        self.partner2_data = self.interpolator.tag_geofence_radius(self.partner2_data)

        # Infer overnight stays on raw data (used for fine-grained matching)
        print("\n  Inferring overnight stays...")
        p1_with_overnight = self.interpolator.infer_overnight_stays(self.partner1_data)
        p2_with_overnight = self.interpolator.infer_overnight_stays(self.partner2_data)
        p1_with_overnight = self.interpolator.tag_geofence_radius(p1_with_overnight)
        p2_with_overnight = self.interpolator.tag_geofence_radius(p2_with_overnight)

        # Stored interpolation (for visualization and confidence timeline)
        print("  Applying stored interpolation...")
        self.partner1_interpolated = self.interpolator.interpolate_presence(
            p1_with_overnight
        )
        self.partner2_interpolated = self.interpolator.interpolate_presence(
            p2_with_overnight
        )
        self.partner1_interpolated = self.interpolator.tag_geofence_radius(
            self.partner1_interpolated
        )
        self.partner2_interpolated = self.interpolator.tag_geofence_radius(
            self.partner2_interpolated
        )

        print(f"  Partner 1: {len(self.partner1_data)} -> {len(self.partner1_interpolated)} points")
        print(f"  Partner 2: {len(self.partner2_data)} -> {len(self.partner2_interpolated)} points")

        # Find raw overlaps (existing point-to-point matching)
        print("\nFinding co-locations (raw data)...")
        self.overlaps = self.find_overlaps(
            self.partner1_data,
            self.partner2_data,
            use_interpolation=False
        )
        print(f"  Found {len(self.overlaps)} co-locations")

        # Fine-grained overlaps (1-min resolution, speed-based geofencing)
        print("\nFinding co-locations (fine-grained interpolation)...")
        self.overlaps_with_interpolation = self.interpolator.find_finegrained_overlaps(
            p1_with_overnight,
            p2_with_overnight,
        )
        print(f"  Found {len(self.overlaps_with_interpolation)} co-locations")

        # Generate statistics
        stats = self._generate_statistics()

        # Phase 4: IND-meaningful metrics
        self.ind_metrics = self._generate_ind_metrics()
        stats['ind_metrics'] = self.ind_metrics

        print("\n" + "=" * 60)
        print("Analysis Complete")
        print("=" * 60)
        print(f"Days together (raw): {stats['raw']['unique_days']}")
        print(f"Days together (interpolated): {stats['interpolated']['unique_days']}")
        print(f"Nights together: {stats['raw']['nights_together']}")
        print(f"Tracked days together: {self.ind_metrics['days_together']}/{self.ind_metrics['total_tracked_days']} ({self.ind_metrics['pct_days_together']}%)")
        print(f"Nights at shared address: {self.ind_metrics['nights_at_shared_address']}/{self.ind_metrics['total_tracked_nights']} ({self.ind_metrics['pct_nights_together']}%)")
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

    def _generate_ind_metrics(self) -> Dict[str, Any]:
        """Generate IND-specific cohabitation metrics based on tracked days."""
        p1 = self.partner1_interpolated
        p2 = self.partner2_interpolated
        overlaps = self.overlaps_with_interpolation

        # Days where both partners have at least one data point
        p1_days = set(p1['timestamp'].dt.date.unique())
        p2_days = set(p2['timestamp'].dt.date.unique())
        both_tracked_days = sorted(p1_days & p2_days)

        # Days with co-location
        together_days = set()
        if overlaps is not None and not overlaps.empty:
            together_days = set(overlaps['timestamp'].dt.date.unique())

        tracked_together = together_days & set(both_tracked_days)
        pct_together = (len(tracked_together) / len(both_tracked_days) * 100) if both_tracked_days else 0

        # Nights at shared address
        night_overlaps = pd.DataFrame()
        if overlaps is not None and not overlaps.empty:
            night_overlaps = overlaps[
                (overlaps['timestamp'].dt.hour >= 22) | (overlaps['timestamp'].dt.hour <= 6)
            ]
        night_dates = set(night_overlaps['timestamp'].dt.date.unique()) if not night_overlaps.empty else set()

        # Nights where both were tracked
        p1_night_dates = set(p1[
            (p1['timestamp'].dt.hour >= 22) | (p1['timestamp'].dt.hour <= 6)
        ]['timestamp'].dt.date.unique())
        p2_night_dates = set(p2[
            (p2['timestamp'].dt.hour >= 22) | (p2['timestamp'].dt.hour <= 6)
        ]['timestamp'].dt.date.unique())
        both_tracked_nights = sorted(p1_night_dates & p2_night_dates)

        pct_nights = (len(night_dates & set(both_tracked_nights)) / len(both_tracked_nights) * 100) if both_tracked_nights else 0

        return {
            'total_tracked_days': len(both_tracked_days),
            'days_together': len(tracked_together),
            'pct_days_together': round(pct_together, 1),
            'total_tracked_nights': len(both_tracked_nights),
            'nights_at_shared_address': len(night_dates & set(both_tracked_nights)),
            'pct_nights_together': round(pct_nights, 1),
            'date_range_start': min(both_tracked_days) if both_tracked_days else None,
            'date_range_end': max(both_tracked_days) if both_tracked_days else None,
            'p1_only_days': len(p1_days - p2_days),
            'p2_only_days': len(p2_days - p1_days),
            'total_gap_days': len(p1_days.symmetric_difference(p2_days)),
        }

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
