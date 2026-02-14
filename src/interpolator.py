"""Presence interpolation for sparse location data.

Phones that aren't moving report infrequently (every 1-4 hours).
This module infers continuous presence between sparse reports.

Key innovations:
- Speed-based movement classification (stationary/walking/driving/flying)
- Geofence radius based on movement speed (uncertainty modeling)
- Fine-grained temporary interpolation for dense co-location matching
- GPS jitter filtering (impossible speed = noise, not movement)
- Overnight stay inference from evening + morning reports
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from geopy.distance import geodesic

# Movement classification thresholds (km/h)
SPEED_STATIONARY = 0.5
SPEED_WALKING = 7.0
SPEED_DRIVING = 150.0
SPEED_JITTER = 1000.0  # Impossible speed for short gaps (<30 min)

# Geofence radii by movement type (meters)
GEOFENCE = {
    'stationary': 50,
    'walking': 100,
    'driving': 500,
    'jitter': 50,
}


@dataclass(slots=True)
class Segment:
    """A movement segment between two consecutive real GPS points."""

    start_time: datetime
    end_time: datetime
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    movement_type: str  # 'stationary', 'walking', 'driving', 'flying', 'jitter'
    geofence_radius: Optional[float]  # meters; None for flying
    speed_kmh: float
    dist_meters: float


class LocationInterpolator:
    """Infers continuous presence from sparse location data."""

    def __init__(
        self,
        max_gap_hours: float = 4.0,
        home_tolerance_m: float = 300.0,
        overnight_start: int = 22,
        overnight_end: int = 6,
        travel_max_gap_hours: float = 18.0,
        city_radius_m: float = 10000.0,
        neighborhood_m: float = 750.0,
        city_m: float = 5000.0,
        home_region_bounds: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.max_gap_hours = max_gap_hours
        self.max_gap_timedelta = timedelta(hours=max_gap_hours)
        self.home_tolerance_m = home_tolerance_m
        self.overnight_start = overnight_start
        self.overnight_end = overnight_end
        self.travel_max_gap_hours = travel_max_gap_hours
        self.city_radius_m = city_radius_m
        self.neighborhood_m = neighborhood_m
        self.city_m = city_m
        self.home_region_bounds = home_region_bounds

    # ------------------------------------------------------------------
    # Movement classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_movement(dist_meters: float, gap_hours: float) -> Tuple[str, float]:
        """Classify movement type from distance and time between two GPS points.

        Returns:
            (movement_type, speed_kmh)
        """
        if gap_hours <= 0:
            return 'stationary', 0.0

        speed_kmh = (dist_meters / 1000) / gap_hours

        # Jitter: impossibly fast for short intervals (<30 min)
        if speed_kmh > SPEED_JITTER and gap_hours < 0.5:
            return 'jitter', speed_kmh

        if speed_kmh <= SPEED_STATIONARY:
            return 'stationary', speed_kmh
        if speed_kmh <= SPEED_WALKING:
            return 'walking', speed_kmh
        if speed_kmh <= SPEED_DRIVING:
            return 'driving', speed_kmh
        return 'flying', speed_kmh

    @staticmethod
    def movement_geofence(movement_type: str) -> Optional[float]:
        """Map movement type to geofence radius in meters.  None = flying."""
        return GEOFENCE.get(movement_type)

    # ------------------------------------------------------------------
    # Fast approximate distance (equirectangular projection)
    # ------------------------------------------------------------------

    @staticmethod
    def _fast_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Fast approximate distance in meters.  Accurate to <0.5% at <=50 km."""
        dlat = (lat2 - lat1) * 111_320
        cos_lat = math.cos(math.radians((lat1 + lat2) * 0.5))
        dlon = (lon2 - lon1) * 111_320 * cos_lat
        return math.sqrt(dlat * dlat + dlon * dlon)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def filter_jitter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove GPS jitter: impossible speed over short intervals.

        If speed > 1000 km/h over < 30 minutes, the point is noise.
        """
        if df.empty or len(df) < 3:
            return df

        df = df.sort_values('timestamp').reset_index(drop=True)
        drop_indices = set()

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            gap_hours = (curr['timestamp'] - prev['timestamp']).total_seconds() / 3600
            if gap_hours <= 0 or gap_hours >= 0.5:
                continue

            dist = self._fast_distance_m(
                prev['lat'], prev['lon'], curr['lat'], curr['lon']
            )
            speed_kmh = (dist / 1000) / gap_hours
            if speed_kmh > SPEED_JITTER:
                drop_indices.add(i)

        if drop_indices:
            df = df.drop(index=list(drop_indices)).reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # Segment building
    # ------------------------------------------------------------------

    def build_segments(self, df: pd.DataFrame) -> List[Segment]:
        """Build movement segments from consecutive GPS points.

        Each segment represents the path between two data points, classified
        by movement type with appropriate geofence radius.
        """
        if df.empty or len(df) < 2:
            return []

        df = df.sort_values('timestamp').reset_index(drop=True)
        segments: List[Segment] = []

        for i in range(len(df) - 1):
            curr = df.iloc[i]
            nxt = df.iloc[i + 1]

            gap_secs = (nxt['timestamp'] - curr['timestamp']).total_seconds()
            gap_hours = gap_secs / 3600

            # Skip very long gaps
            if gap_hours > self.travel_max_gap_hours:
                continue

            dist = self._fast_distance_m(
                curr['lat'], curr['lon'], nxt['lat'], nxt['lon']
            )
            movement_type, speed = self.classify_movement(dist, gap_hours)
            geofence = self.movement_geofence(movement_type)

            segments.append(Segment(
                start_time=curr['timestamp'],
                end_time=nxt['timestamp'],
                start_lat=curr['lat'],
                start_lon=curr['lon'],
                end_lat=nxt['lat'],
                end_lon=nxt['lon'],
                movement_type=movement_type,
                geofence_radius=geofence,
                speed_kmh=speed,
                dist_meters=dist,
            ))

        return segments

    @staticmethod
    def interpolate_position(seg: Segment, t: datetime) -> Tuple[float, float]:
        """Get interpolated lat/lon at a specific time within a segment."""
        total = (seg.end_time - seg.start_time).total_seconds()
        if total <= 0:
            return seg.start_lat, seg.start_lon

        alpha = max(0.0, min(1.0, (t - seg.start_time).total_seconds() / total))

        lat = seg.start_lat + (seg.end_lat - seg.start_lat) * alpha
        lon = seg.start_lon + (seg.end_lon - seg.start_lon) * alpha
        return lat, lon

    # ------------------------------------------------------------------
    # Stored interpolation (for DataFrame / overnight / visualization)
    # ------------------------------------------------------------------

    def interpolate_presence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate stored presence points between sparse reports.

        Uses speed-based movement classification for geofence radii.
        Points are stored in the DataFrame for overnight inference and
        visualization.

        Fine-grained matching (1-minute resolution) is done separately in
        find_finegrained_overlaps() without storing points.
        """
        if df.empty:
            return df

        df = df.sort_values('timestamp').reset_index(drop=True)
        if 'interpolated' not in df.columns:
            df['interpolated'] = False

        segments = self.build_segments(df)
        rows: list[dict] = []

        for seg in segments:
            gap_hours = (seg.end_time - seg.start_time).total_seconds() / 3600
            if gap_hours < 0.5 or seg.geofence_radius is None:
                continue
            if seg.movement_type == 'jitter':
                continue

            # Interval based on movement type
            if seg.movement_type == 'stationary':
                interval_min = 30
            elif seg.movement_type == 'walking':
                interval_min = 15
            else:  # driving
                interval_min = 60

            n = max(1, int(gap_hours * 60 / interval_min))
            for k in range(1, n):
                t = seg.start_time + timedelta(minutes=k * interval_min)
                if t >= seg.end_time:
                    break
                lat, lon = self.interpolate_position(seg, t)
                rows.append({
                    'timestamp': t,
                    'lat': lat,
                    'lon': lon,
                    'interpolated': True,
                    'interp_type': seg.movement_type,
                    'geofence_radius': seg.geofence_radius,
                    'speed_kmh': seg.speed_kmh,
                    'accuracy': 50 if seg.movement_type == 'stationary' else 100,
                    'source_file': f"interp_{seg.movement_type}",
                })

        if rows:
            result = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            return result.sort_values('timestamp').reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Overnight stays
    # ------------------------------------------------------------------

    def infer_overnight_stays(
        self,
        df: pd.DataFrame,
        home_location: Optional[Tuple[float, float]] = None,
    ) -> pd.DataFrame:
        """Infer full overnight presence from evening + morning reports.

        If an evening report is near home and the next morning report is also
        near home, generate hourly points for the full night.
        """
        if df.empty:
            return df

        df = df.sort_values('timestamp').reset_index(drop=True)

        if home_location is None:
            home_location = self._detect_home_location(df)
            if home_location is None:
                return df

        rows: list[dict] = []

        for i in range(len(df) - 1):
            curr = df.iloc[i]
            nxt = df.iloc[i + 1]

            is_evening = curr['timestamp'].hour >= 19
            is_next_morning = nxt['timestamp'].hour <= 10
            is_next_day = nxt['timestamp'].date() > curr['timestamp'].date()
            gap_hours = (nxt['timestamp'] - curr['timestamp']).total_seconds() / 3600

            if not (is_evening and is_next_morning and is_next_day and gap_hours <= 16):
                continue

            dist_curr = self._fast_distance_m(
                curr['lat'], curr['lon'], home_location[0], home_location[1]
            )
            dist_nxt = self._fast_distance_m(
                nxt['lat'], nxt['lon'], home_location[0], home_location[1]
            )

            if dist_curr <= self.home_tolerance_m and dist_nxt <= self.home_tolerance_m:
                fill_start = curr['timestamp'].replace(
                    minute=0, second=0, microsecond=0
                ) + timedelta(hours=1)
                fill_end = nxt['timestamp'].replace(
                    minute=0, second=0, microsecond=0
                )

                t = fill_start
                while t < fill_end:
                    rows.append({
                        'timestamp': t,
                        'lat': home_location[0],
                        'lon': home_location[1],
                        'interpolated': True,
                        'interp_type': 'overnight',
                        'geofence_radius': GEOFENCE['stationary'],
                        'accuracy': 50,
                        'source_file': 'overnight_inference',
                    })
                    t += timedelta(hours=1)

        if rows:
            result = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            result = result.drop_duplicates(subset=['timestamp', 'lat', 'lon'])
            return result.sort_values('timestamp').reset_index(drop=True)
        return df

    def _detect_home_location(
        self, df: pd.DataFrame
    ) -> Optional[Tuple[float, float]]:
        """Auto-detect home location from overnight reports."""
        overnight = df[
            (df['timestamp'].dt.hour >= self.overnight_start)
            | (df['timestamp'].dt.hour <= self.overnight_end)
        ]
        if overnight.empty:
            return None

        oc = overnight.copy()
        oc['lat_r'] = oc['lat'].round(2)
        oc['lon_r'] = oc['lon'].round(2)

        counts = oc.groupby(['lat_r', 'lon_r']).size()
        if counts.empty:
            return None
        return counts.idxmax()

    # ------------------------------------------------------------------
    # Geofence tagging (for raw data)
    # ------------------------------------------------------------------

    @staticmethod
    def tag_geofence_radius(df: pd.DataFrame) -> pd.DataFrame:
        """Tag raw data points with geofence_radius based on source accuracy.

        Accuracy values from parsers:
        - 10: Google timelinePath (GPS fix)  -> 75 m
        - 15: Apple timelinePath             -> 100 m
        - 20: Visit records (place-level)    -> 150 m
        - 30: Activity start/end             -> 200 m
        - 50+: Already-interpolated points   -> keep existing

        Points that already have a geofence_radius are not overwritten.
        """
        if df.empty or 'accuracy' not in df.columns:
            return df

        df = df.copy()
        acc_map = {10: 75, 15: 100, 20: 150, 30: 200}

        if 'geofence_radius' not in df.columns:
            df['geofence_radius'] = np.nan

        mask = df['geofence_radius'].isna()
        df.loc[mask, 'geofence_radius'] = (
            df.loc[mask, 'accuracy'].map(acc_map).fillna(150)
        )
        return df

    # ------------------------------------------------------------------
    # Fine-grained co-location matching
    # ------------------------------------------------------------------

    def find_finegrained_overlaps(
        self,
        p1_df: pd.DataFrame,
        p2_df: pd.DataFrame,
        interval_seconds: int = 60,
        bucket_minutes: int = 15,
    ) -> pd.DataFrame:
        """Find co-locations using fine-grained temporary interpolation.

        Generates temporary points at *interval_seconds* resolution between
        real GPS readings.  Points are NOT stored -- only used for matching.

        Uses geofence circle overlap:  co-located when distance < r1 + r2.
        Also applies dynamic distance thresholds based on movement context:
          - both stationary/walking  -> neighborhood_m  (750 m)
          - any driving (abroad)     -> 2000 m (traveling together)

        Results are de-duplicated to one match per *bucket_minutes* window,
        keeping the closest-distance match.

        Args:
            p1_df: Partner 1 DataFrame (raw + overnight points recommended)
            p2_df: Partner 2 DataFrame
            interval_seconds: Fine-grained interpolation interval (default 60 s)
            bucket_minutes: Dedup bucket size (default 15 min)

        Returns:
            DataFrame of overlap events compatible with downstream pipeline
        """
        # Build segments from all points (real + overnight inferred)
        p1_segs = self.build_segments(p1_df)
        p2_segs = self.build_segments(p2_df)

        if not p1_segs or not p2_segs:
            return pd.DataFrame()

        # Efficient sweep-line to find overlapping segment pairs
        pairs = self._find_overlapping_segment_pairs(p1_segs, p2_segs)

        all_matches: list[dict] = []
        tier1 = 150.0  # confirmed
        travel_threshold = 2000.0  # driving together

        for s1, s2, t_start, t_end in pairs:
            # Skip flying segments
            if s1.geofence_radius is None or s2.geofence_radius is None:
                continue

            # Dynamic threshold based on movement context
            both_local = s1.movement_type in ('stationary', 'walking') and \
                         s2.movement_type in ('stationary', 'walking')
            any_driving = 'driving' in (s1.movement_type, s2.movement_type)

            # Check if segment pair is in home region
            pair_avg_lat = (s1.start_lat + s2.start_lat) * 0.5
            pair_avg_lon = (s1.start_lon + s2.start_lon) * 0.5
            pair_in_home = (self.home_region_bounds and
                            self.home_region_bounds[0] <= pair_avg_lat <= self.home_region_bounds[1] and
                            self.home_region_bounds[2] <= pair_avg_lon <= self.home_region_bounds[3])

            if both_local:
                ctx_threshold = self.neighborhood_m
            elif any_driving and not pair_in_home:
                ctx_threshold = travel_threshold  # Wide threshold only abroad
            else:
                ctx_threshold = self.neighborhood_m

            # Quick spatial pre-check (corners of both segments)
            max_geofence_sum = s1.geofence_radius + s2.geofence_radius
            # Worst-case max distance offset from segment endpoints
            max_internal = s1.dist_meters + s2.dist_meters
            cutoff = max(ctx_threshold, max_geofence_sum) + max_internal

            d_start = self._fast_distance_m(
                s1.start_lat, s1.start_lon, s2.start_lat, s2.start_lon
            )
            d_end = self._fast_distance_m(
                s1.end_lat, s1.end_lon, s2.end_lat, s2.end_lon
            )
            if min(d_start, d_end) > cutoff:
                continue

            # Pre-compute cos factor for fast distance within this pair
            avg_lat = (s1.start_lat + s2.start_lat) * 0.5
            cos_factor = math.cos(math.radians(avg_lat))

            # Fine-grained time stepping
            total_secs = (t_end - t_start).total_seconds()
            n_steps = max(1, int(total_secs / interval_seconds))

            for step in range(n_steps + 1):
                t = t_start + timedelta(seconds=step * interval_seconds)
                if t > t_end:
                    break

                lat1, lon1 = self.interpolate_position(s1, t)
                lat2, lon2 = self.interpolate_position(s2, t)

                # Inline fast distance
                dlat = (lat2 - lat1) * 111_320
                dlon = (lon2 - lon1) * 111_320 * cos_factor
                dist = math.sqrt(dlat * dlat + dlon * dlon)

                geofence_match = dist <= s1.geofence_radius + s2.geofence_radius
                threshold_match = dist <= ctx_threshold

                if not (geofence_match or threshold_match):
                    continue

                # Drive-by filter: in home region, reject driving-past-stationary
                if dist > tier1:
                    avg_lat_pos = (lat1 + lat2) * 0.5
                    avg_lon_pos = (lon1 + lon2) * 0.5
                    in_home = (self.home_region_bounds and
                               self.home_region_bounds[0] <= avg_lat_pos <= self.home_region_bounds[1] and
                               self.home_region_bounds[2] <= avg_lon_pos <= self.home_region_bounds[3])
                    if in_home:
                        one_driving = s1.movement_type == 'driving' or s2.movement_type == 'driving'
                        one_stationary = s1.movement_type == 'stationary' or s2.movement_type == 'stationary'
                        if one_driving and one_stationary:
                            continue  # Drive-by: one driving past one stationary
                        if dist > self.neighborhood_m:
                            continue  # No city-tier matches in home region
                        both_slow = (s1.movement_type in ('stationary', 'walking') and
                                     s2.movement_type in ('stationary', 'walking'))
                        if not both_slow:
                            continue  # One moving fast, likely not together

                # Tier label
                if dist <= tier1:
                    tier = 'confirmed'
                elif dist <= self.neighborhood_m:
                    tier = 'neighborhood'
                elif dist <= self.city_m:
                    tier = 'city'
                else:
                    tier = 'geofence'

                all_matches.append({
                    'timestamp': t,
                    'lat': (lat1 + lat2) * 0.5,
                    'lon': (lon1 + lon2) * 0.5,
                    'distance_meters': dist,
                    'proximity_tier': tier,
                    'partner1_lat': lat1,
                    'partner1_lon': lon1,
                    'partner2_lat': lat2,
                    'partner2_lon': lon2,
                    'partner1_source': f'seg_{s1.movement_type}',
                    'partner2_source': f'seg_{s2.movement_type}',
                    'p1_geofence': s1.geofence_radius,
                    'p2_geofence': s2.geofence_radius,
                    'p1_movement': s1.movement_type,
                    'p2_movement': s2.movement_type,
                    'geofence_match': geofence_match,
                    'interpolated': True,
                })

        if not all_matches:
            return pd.DataFrame()

        result = pd.DataFrame(all_matches)

        # De-duplicate: one match per bucket, keep closest distance
        result['_bucket'] = result['timestamp'].dt.floor(f'{bucket_minutes}min')
        result = result.loc[result.groupby('_bucket')['distance_meters'].idxmin()]
        result = result.drop(columns=['_bucket']).reset_index(drop=True)

        return result

    @staticmethod
    def _find_overlapping_segment_pairs(
        segs1: List[Segment],
        segs2: List[Segment],
    ) -> List[Tuple[Segment, Segment, datetime, datetime]]:
        """Find all (s1, s2, overlap_start, overlap_end) pairs efficiently.

        Uses a sweep-line approach: O(n + m + k) where k = output size.
        """
        pairs: list[tuple] = []
        j_start = 0

        for s1 in segs1:
            # Advance past segments that end before s1 starts
            while j_start < len(segs2) and segs2[j_start].end_time <= s1.start_time:
                j_start += 1

            j = j_start
            while j < len(segs2) and segs2[j].start_time < s1.end_time:
                s2 = segs2[j]
                ov_start = max(s1.start_time, s2.start_time)
                ov_end = min(s1.end_time, s2.end_time)
                if ov_start < ov_end:
                    pairs.append((s1, s2, ov_start, ov_end))
                j += 1

        return pairs

    # ------------------------------------------------------------------
    # Presence matrix
    # ------------------------------------------------------------------

    def calculate_presence_probability(
        self,
        partner1_df: pd.DataFrame,
        partner2_df: pd.DataFrame,
        target_time: datetime,
        time_window_hours: float = 2.0,
    ) -> float:
        """Calculate co-presence probability at *target_time*.

        Uses segment-based fine-grained interpolation: builds segments from
        nearby GPS points and checks geofence circle overlap at 1-minute
        resolution within the window.
        """
        window = timedelta(hours=time_window_hours)
        ext = timedelta(hours=time_window_hours + 1)

        p1w = partner1_df[
            (partner1_df['timestamp'] >= target_time - ext)
            & (partner1_df['timestamp'] <= target_time + ext)
        ]
        p2w = partner2_df[
            (partner2_df['timestamp'] >= target_time - ext)
            & (partner2_df['timestamp'] <= target_time + ext)
        ]

        if p1w.empty or p2w.empty:
            return np.nan

        p1_segs = self.build_segments(p1w)
        p2_segs = self.build_segments(p2w)

        if not p1_segs or not p2_segs:
            # Fallback: closest-point distance
            return self._fallback_probability(p1w, p2w, target_time, time_window_hours)

        check_start = target_time - window
        check_end = target_time + window
        best_prob = 0.0

        for s1 in p1_segs:
            for s2 in p2_segs:
                ov_start = max(s1.start_time, s2.start_time, check_start)
                ov_end = min(s1.end_time, s2.end_time, check_end)
                if ov_start >= ov_end:
                    continue
                if s1.geofence_radius is None or s2.geofence_radius is None:
                    continue

                avg_lat = (s1.start_lat + s2.start_lat) * 0.5
                cos_f = math.cos(math.radians(avg_lat))

                total_secs = (ov_end - ov_start).total_seconds()
                n_steps = max(1, int(total_secs / 60))

                for step in range(n_steps + 1):
                    t = ov_start + timedelta(seconds=step * 60)
                    if t > ov_end:
                        break

                    lat1, lon1 = self.interpolate_position(s1, t)
                    lat2, lon2 = self.interpolate_position(s2, t)

                    dlat = (lat2 - lat1) * 111_320
                    dlon = (lon2 - lon1) * 111_320 * cos_f
                    dist = math.sqrt(dlat * dlat + dlon * dlon)

                    # Base probability from distance / geofence overlap
                    r_sum = s1.geofence_radius + s2.geofence_radius
                    if dist <= r_sum:
                        base = 1.0
                    elif dist <= 150:
                        base = 0.95
                    elif dist <= self.neighborhood_m:
                        base = 0.8
                    elif dist <= self.city_m:
                        base = 0.5
                    else:
                        continue

                    # Time decay
                    offset_h = abs((t - target_time).total_seconds()) / 3600
                    time_factor = max(0.0, 1.0 - offset_h / time_window_hours)

                    prob = min(1.0, base * time_factor)
                    if prob > best_prob:
                        best_prob = prob

        return best_prob if best_prob > 0 else 0.0

    def _fallback_probability(
        self,
        p1: pd.DataFrame,
        p2: pd.DataFrame,
        target: datetime,
        window_h: float,
    ) -> float:
        """Closest-point fallback when segment building fails."""
        window = timedelta(hours=window_h)
        p1n = p1[
            (p1['timestamp'] >= target - window) & (p1['timestamp'] <= target + window)
        ]
        p2n = p2[
            (p2['timestamp'] >= target - window) & (p2['timestamp'] <= target + window)
        ]
        if p1n.empty or p2n.empty:
            return np.nan

        best = 0.0
        for _, r1 in p1n.head(10).iterrows():
            for _, r2 in p2n.head(10).iterrows():
                dist = self._fast_distance_m(r1['lat'], r1['lon'], r2['lat'], r2['lon'])
                if dist <= 150:
                    b = 1.0
                elif dist <= self.neighborhood_m:
                    b = 0.8
                elif dist <= self.city_m:
                    b = 0.5
                else:
                    continue
                off = max(
                    abs((r1['timestamp'] - target).total_seconds()),
                    abs((r2['timestamp'] - target).total_seconds()),
                ) / 3600
                tf = max(0.0, 1.0 - off / window_h)
                p = min(1.0, b * tf)
                if p > best:
                    best = p
        return best if best > 0 else 0.0

    def build_presence_matrix(
        self,
        partner1_df: pd.DataFrame,
        partner2_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        precomputed_overlaps: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build 24 h x days matrix of co-presence probabilities and states.

        When *precomputed_overlaps* is provided (from find_finegrained_overlaps),
        the matrix is built by aggregating those overlaps — much faster than
        recomputing per-hour probability.
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        prob_matrix = pd.DataFrame(
            index=range(24), columns=date_range.date, dtype=float
        )
        state_matrix = pd.DataFrame(
            index=range(24), columns=date_range.date, dtype=object
        )
        prob_matrix[:] = np.nan
        state_matrix[:] = 'no_data'

        if precomputed_overlaps is not None and not precomputed_overlaps.empty:
            return self._matrix_from_overlaps(
                precomputed_overlaps, prob_matrix, state_matrix, date_range,
                partner1_df, partner2_df
            )

        # Slow path: calculate per-hour probability
        for date in date_range:
            for hour in range(24):
                target = datetime.combine(date.date(), datetime.min.time()).replace(
                    hour=hour
                )
                prob = self.calculate_presence_probability(
                    partner1_df, partner2_df, target
                )
                prob_matrix.loc[hour, date.date()] = prob
                if np.isnan(prob):
                    state_matrix.loc[hour, date.date()] = 'no_data'
                elif prob >= 0.4:
                    state_matrix.loc[hour, date.date()] = 'co_located'
                else:
                    state_matrix.loc[hour, date.date()] = 'apart'

        return prob_matrix, state_matrix

    def _matrix_from_overlaps(
        self,
        overlaps: pd.DataFrame,
        prob_matrix: pd.DataFrame,
        state_matrix: pd.DataFrame,
        date_range: pd.DatetimeIndex,
        partner1_df: pd.DataFrame = None,
        partner2_df: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fill matrices from pre-computed fine-grained overlaps."""
        # Build per-(date, hour) sets of which partner has data
        p1_hours = set()
        p2_hours = set()
        if partner1_df is not None and not partner1_df.empty:
            p1_hours = set(zip(
                partner1_df['timestamp'].dt.date,
                partner1_df['timestamp'].dt.hour
            ))
        if partner2_df is not None and not partner2_df.empty:
            p2_hours = set(zip(
                partner2_df['timestamp'].dt.date,
                partner2_df['timestamp'].dt.hour
            ))

        # Mark hours where both partners have data as 'apart' (default before overlaps)
        for d in date_range.date:
            for h in range(24):
                if (d, h) in p1_hours and (d, h) in p2_hours:
                    state_matrix.loc[h, d] = 'apart'
                    prob_matrix.loc[h, d] = 0.0

        # Pre-index overlaps by (date, hour)
        ov = overlaps.copy()
        ov['_date'] = ov['timestamp'].dt.date
        ov['_hour'] = ov['timestamp'].dt.hour

        grouped = ov.groupby(['_date', '_hour'])

        for (d, h), grp in grouped:
            if d not in prob_matrix.columns:
                continue
            best = grp.loc[grp['distance_meters'].idxmin()]
            dist = best['distance_meters']

            r1 = best.get('p1_geofence', 50)
            r2 = best.get('p2_geofence', 50)
            if pd.isna(r1):
                r1 = 50
            if pd.isna(r2):
                r2 = 50

            if dist <= r1 + r2:
                prob = 1.0
            elif dist <= 150:
                prob = 0.95
            elif dist <= self.neighborhood_m:
                prob = 0.8
            elif dist <= self.city_m:
                prob = 0.5
            else:
                prob = 0.3

            prob_matrix.loc[h, d] = prob
            state_matrix.loc[h, d] = 'co_located' if prob >= 0.4 else 'apart'

        return prob_matrix, state_matrix
