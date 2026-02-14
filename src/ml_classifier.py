#!/usr/bin/env python3
"""ML classifier for co-location detection.

Trains a gradient-boosted classifier on day-level features derived from
the location data, using ground truth labels. Compares precision/recall
against the existing rule-based system and provides confidence-calibrated
predictions for days where the rule-based system is uncertain.

Usage:
    python3 src/ml_classifier.py          # full pipeline
    python3 src/ml_classifier.py --eval   # evaluation only (skip training)
"""

import json
import math
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

# Optional imports for ML (graceful fallback)
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.metrics import (
        classification_report, confusion_matrix, precision_recall_fscore_support
    )
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Local imports
try:
    from analyzer import LocationAnalyzer
    from interpolator import LocationInterpolator
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from analyzer import LocationAnalyzer
    from interpolator import LocationInterpolator


# ─── Ground Truth Labels ────────────────────────────────────────────────────

def build_ground_truth() -> Dict[date, str]:
    """Build day-level ground truth labels from known relationship events.

    Returns:
        dict mapping date -> 'together' | 'apart' | 'mixed'

    'mixed' days have both together and apart periods (e.g. morning together,
    afternoon apart). For binary classification we treat mixed as 'together'.
    """
    labels: Dict[date, str] = {}

    # ── Definite APART periods ──

    # All of May 2025: pre-meeting, definitely apart
    d = date(2025, 5, 1)
    while d <= date(2025, 5, 31):
        labels[d] = 'apart'
        d += timedelta(days=1)

    # Jun 1-5: before first meeting
    d = date(2025, 6, 1)
    while d <= date(2025, 6, 5):
        labels[d] = 'apart'
        d += timedelta(days=1)

    # Jun 8-29: early dating, mostly apart (no strong together evidence)
    # Skip these - uncertain period
    # But Jun 30 - Jul 3: Sean away, definitely apart
    d = date(2025, 6, 30)
    while d <= date(2025, 7, 3):
        labels[d] = 'apart'
        d += timedelta(days=1)

    # Sep 1-17: both on separate holidays, different countries
    d = date(2025, 9, 1)
    while d <= date(2025, 9, 17):
        labels[d] = 'apart'
        d += timedelta(days=1)

    # Oct 14 - Nov 23: Maia in Argentina, Sean in Netherlands
    d = date(2025, 10, 14)
    while d <= date(2025, 11, 23):
        labels[d] = 'apart'
        d += timedelta(days=1)

    # Dec 25 - Jan 14: apart (between Argentina trip and reunion)
    d = date(2025, 12, 25)
    while d <= date(2026, 1, 14):
        labels[d] = 'apart'
        d += timedelta(days=1)

    # ── Definite TOGETHER periods ──

    # Jun 6: first meeting (morning + evening)
    labels[date(2025, 6, 6)] = 'together'
    # Jun 7: together through ~16:00
    labels[date(2025, 6, 7)] = 'mixed'

    # Jul 12: Saturday outing together
    labels[date(2025, 7, 12)] = 'together'
    # Jul 13: morning together, then apart
    labels[date(2025, 7, 13)] = 'mixed'

    # Jul 17-20: Black Forest trip
    labels[date(2025, 7, 17)] = 'together'
    labels[date(2025, 7, 18)] = 'together'
    labels[date(2025, 7, 19)] = 'mixed'  # morning apart, afternoon together
    labels[date(2025, 7, 20)] = 'mixed'  # some solo, some together

    # Aug 2-22: Sean at Maia's place in Belgium
    d = date(2025, 8, 2)
    while d <= date(2025, 8, 22):
        labels[d] = 'together'
        d += timedelta(days=1)

    # Aug 7: drive together confirmed by photos
    labels[date(2025, 8, 7)] = 'together'
    # Aug 14: afternoon together in Germany
    labels[date(2025, 8, 14)] = 'together'
    # Aug 24: together in Luxembourg
    labels[date(2025, 8, 24)] = 'together'

    # After holidays (~Sep 18): Maia moved in with Sean
    # Sep 18 - Oct 7: living together (before Madrid)
    d = date(2025, 9, 18)
    while d <= date(2025, 10, 7):
        labels[d] = 'together'
        d += timedelta(days=1)

    # Oct 8-12: Madrid trip together
    d = date(2025, 10, 8)
    while d <= date(2025, 10, 12):
        labels[d] = 'together'
        d += timedelta(days=1)

    # Oct 13: Sean flew home 06:15, Maia stayed - mostly apart
    labels[date(2025, 10, 13)] = 'apart'

    # Nov 24 - Dec 24: Sean in Argentina with Maia
    d = date(2025, 11, 24)
    while d <= date(2025, 12, 24):
        labels[d] = 'together'
        d += timedelta(days=1)

    # Jan 15+: reunion (Maia back)
    d = date(2026, 1, 15)
    while d <= date(2026, 2, 1):
        labels[d] = 'together'
        d += timedelta(days=1)

    # Feb 2-6: Bulgaria trip
    d = date(2026, 2, 2)
    while d <= date(2026, 2, 6):
        labels[d] = 'together'
        d += timedelta(days=1)

    # Feb 7+: together (living together)
    d = date(2026, 2, 7)
    while d <= date(2026, 2, 13):
        labels[d] = 'together'
        d += timedelta(days=1)

    return labels


def derive_nighttime_apart_labels(
    p1_df: pd.DataFrame,
    p2_df: pd.DataFrame,
    labels: Dict[date, str],
    overlaps_df: Optional[pd.DataFrame] = None,
    start_date: date = date(2025, 6, 8),
    end_date: date = date(2025, 8, 1),
    home_radius_m: float = 500.0,
) -> Dict[date, str]:
    """Derive 'apart' labels for nights where each partner is at their own home.

    For the early dating period (Jun - before Sean moved to Maia's), detect
    each partner's home from nighttime GPS clusters, then label nights where
    Sean is at his home AND Maia is at her Belgium home as 'apart'.

    Conservative: only labels a day as 'apart' if BOTH:
    1. Both slept at their own homes that night
    2. No confirmed (<150m) overlaps exist for that day (avoids mislabeling
       days where they met during the day but went home separately)

    Only adds labels for dates not already in the labels dict.

    Args:
        p1_df: Partner 1 (Sean) location data
        p2_df: Partner 2 (Maia) location data
        labels: Existing labels dict (modified in-place)
        overlaps_df: Pre-computed overlaps for confirmed overlap check
        start_date: Start of period to check (default: day after first meeting)
        end_date: End of period (default: before Sean moved to Maia's Aug 2)
        home_radius_m: Maximum distance from home centroid to count as "at home"

    Returns:
        Modified labels dict
    """
    def _detect_home(df, period_start, period_end):
        """Find home location from nighttime GPS clusters in a period."""
        period = df[
            (df['timestamp'].dt.date >= period_start) &
            (df['timestamp'].dt.date <= period_end)
        ]
        night = period[
            (period['timestamp'].dt.hour >= 22) | (period['timestamp'].dt.hour <= 6)
        ]
        if night.empty or len(night) < 5:
            return None

        # Round to ~100m grid and find most common location
        nc = night.copy()
        nc['lat_r'] = nc['lat'].round(3)  # ~111m precision
        nc['lon_r'] = nc['lon'].round(3)
        counts = nc.groupby(['lat_r', 'lon_r']).size()
        if counts.empty:
            return None
        best = counts.idxmax()
        return (best[0], best[1])

    def _at_home(df, target_date, home_loc, radius_m):
        """Check if person has nighttime GPS near their home on a given date."""
        if home_loc is None:
            return False
        # Check evening of target_date (22:00+) and morning of next day (00:00-06:00)
        evening_start = datetime.combine(target_date, datetime.min.time().replace(hour=22))
        morning_end = datetime.combine(target_date + timedelta(days=1), datetime.min.time().replace(hour=6))

        night_pts = df[
            (df['timestamp'] >= evening_start) & (df['timestamp'] <= morning_end)
        ]
        if night_pts.empty:
            return False

        # Check if any nighttime point is near home
        for _, row in night_pts.iterrows():
            dist = _fast_distance_m(row['lat'], row['lon'], home_loc[0], home_loc[1])
            if dist <= radius_m:
                return True
        return False

    # Detect home locations for the early dating period
    sean_home = _detect_home(p1_df, start_date, end_date)
    maia_home = _detect_home(p2_df, start_date, end_date)

    if sean_home is None or maia_home is None:
        print("  Could not detect home locations for nighttime separation analysis")
        return labels

    # Check homes are sufficiently apart (should be ~50+ km for NL vs Belgium)
    home_dist = _fast_distance_m(sean_home[0], sean_home[1], maia_home[0], maia_home[1])
    print(f"  Detected homes: Sean=({sean_home[0]:.3f},{sean_home[1]:.3f}), "
          f"Maia=({maia_home[0]:.3f},{maia_home[1]:.3f}), "
          f"distance={home_dist/1000:.1f}km")

    if home_dist < 5000:
        print("  Homes too close (<5km) — skipping nighttime separation labels")
        return labels

    # Pre-compute which dates have confirmed (<150m) overlaps
    confirmed_dates: set = set()
    if overlaps_df is not None and not overlaps_df.empty:
        confirmed = overlaps_df[overlaps_df['proximity_tier'] == 'confirmed']
        if not confirmed.empty:
            confirmed_dates = set(confirmed['timestamp'].dt.date.unique())

    # Check each night in the period
    added = 0
    skipped_overlap = 0
    d = start_date
    while d <= end_date:
        if d not in labels:  # Only add if not already labeled
            # Conservative: skip if there are confirmed overlaps this day
            # (they may have met during the day but slept separately)
            if d in confirmed_dates:
                skipped_overlap += 1
                d += timedelta(days=1)
                continue

            sean_at_home = _at_home(p1_df, d, sean_home, home_radius_m)
            maia_at_home = _at_home(p2_df, d, maia_home, home_radius_m)

            if sean_at_home and maia_at_home:
                labels[d] = 'apart'
                added += 1

        d += timedelta(days=1)

    print(f"  Added {added} nighttime-separation 'apart' labels ({start_date} to {end_date})"
          f" (skipped {skipped_overlap} days with confirmed overlaps)")
    return labels


def label_to_binary(label: str) -> int:
    """Convert label to binary: together/mixed = 1, apart = 0."""
    return 0 if label == 'apart' else 1


# ─── Feature Engineering ────────────────────────────────────────────────────

def _fast_distance_m(lat1, lon1, lat2, lon2):
    """Equirectangular approximation of distance in meters."""
    dlat = (lat2 - lat1) * 111_320
    cos_lat = math.cos(math.radians((lat1 + lat2) * 0.5))
    dlon = (lon2 - lon1) * 111_320 * cos_lat
    return math.sqrt(dlat * dlat + dlon * dlon)


def extract_day_features(
    p1_df: pd.DataFrame,
    p2_df: pd.DataFrame,
    overlaps_df: pd.DataFrame,
    target_date: date,
) -> Dict[str, float]:
    """Extract feature vector for a single day.

    Features capture proximity patterns, movement patterns, temporal coverage,
    and geographic context that distinguish co-location from coincidental
    proximity (e.g. commute drive-bys).

    Args:
        p1_df: Partner 1 location data (full dataset, will be filtered)
        p2_df: Partner 2 location data
        overlaps_df: Pre-computed overlaps from fine-grained matching
        target_date: Date to extract features for

    Returns:
        Dict of feature name -> value
    """
    dt_start = datetime.combine(target_date, datetime.min.time())
    dt_end = dt_start + timedelta(days=1)

    # Filter to target day with some padding
    pad = timedelta(hours=3)
    p1_day = p1_df[(p1_df['timestamp'] >= dt_start - pad) & (p1_df['timestamp'] < dt_end + pad)]
    p2_day = p2_df[(p2_df['timestamp'] >= dt_start - pad) & (p2_df['timestamp'] < dt_end + pad)]

    # Strictly on this day
    p1_strict = p1_df[(p1_df['timestamp'] >= dt_start) & (p1_df['timestamp'] < dt_end)]
    p2_strict = p2_df[(p2_df['timestamp'] >= dt_start) & (p2_df['timestamp'] < dt_end)]

    ov_day = pd.DataFrame()
    if overlaps_df is not None and not overlaps_df.empty:
        ov_day = overlaps_df[
            (overlaps_df['timestamp'] >= dt_start) & (overlaps_df['timestamp'] < dt_end)
        ]

    features: Dict[str, float] = {}

    # ── Data availability ──
    features['p1_points'] = len(p1_strict)
    features['p2_points'] = len(p2_strict)
    features['both_have_data'] = float(len(p1_strict) > 0 and len(p2_strict) > 0)

    # Temporal coverage: how many hours of the day have data
    if not p1_strict.empty:
        features['p1_hours_covered'] = p1_strict['timestamp'].dt.hour.nunique()
        features['p1_first_hour'] = p1_strict['timestamp'].dt.hour.min()
        features['p1_last_hour'] = p1_strict['timestamp'].dt.hour.max()
    else:
        features['p1_hours_covered'] = 0
        features['p1_first_hour'] = -1
        features['p1_last_hour'] = -1

    if not p2_strict.empty:
        features['p2_hours_covered'] = p2_strict['timestamp'].dt.hour.nunique()
        features['p2_first_hour'] = p2_strict['timestamp'].dt.hour.min()
        features['p2_last_hour'] = p2_strict['timestamp'].dt.hour.max()
    else:
        features['p2_hours_covered'] = 0
        features['p2_first_hour'] = -1
        features['p2_last_hour'] = -1

    # ── Overlap features ──
    features['n_overlaps'] = len(ov_day)

    if not ov_day.empty:
        features['min_distance_m'] = ov_day['distance_meters'].min()
        features['mean_distance_m'] = ov_day['distance_meters'].mean()
        features['median_distance_m'] = ov_day['distance_meters'].median()
        features['max_distance_m'] = ov_day['distance_meters'].max()
        features['std_distance_m'] = ov_day['distance_meters'].std() if len(ov_day) > 1 else 0

        # Tier distribution
        tier_counts = ov_day['proximity_tier'].value_counts()
        features['n_confirmed'] = tier_counts.get('confirmed', 0)
        features['n_neighborhood'] = tier_counts.get('neighborhood', 0)
        features['n_city'] = tier_counts.get('city', 0)
        features['pct_confirmed'] = features['n_confirmed'] / len(ov_day) if len(ov_day) > 0 else 0

        # Temporal spread of overlaps
        overlap_hours = ov_day['timestamp'].dt.hour.unique()
        features['overlap_hours_span'] = len(overlap_hours)
        features['overlap_first_hour'] = ov_day['timestamp'].dt.hour.min()
        features['overlap_last_hour'] = ov_day['timestamp'].dt.hour.max()
        features['overlap_duration_hours'] = features['overlap_last_hour'] - features['overlap_first_hour']

        # Night overlap (22:00-06:00)
        night_mask = (ov_day['timestamp'].dt.hour >= 22) | (ov_day['timestamp'].dt.hour <= 6)
        features['n_night_overlaps'] = night_mask.sum()
        features['has_night_overlap'] = float(features['n_night_overlaps'] > 0)

        # Morning overlap (06:00-10:00)
        morning_mask = (ov_day['timestamp'].dt.hour >= 6) & (ov_day['timestamp'].dt.hour <= 10)
        features['n_morning_overlaps'] = morning_mask.sum()

        # Evening overlap (18:00-22:00)
        evening_mask = (ov_day['timestamp'].dt.hour >= 18) & (ov_day['timestamp'].dt.hour < 22)
        features['n_evening_overlaps'] = evening_mask.sum()

        # Movement context (if available)
        if 'p1_movement' in ov_day.columns:
            mv_counts = ov_day['p1_movement'].value_counts()
            features['n_stationary_overlaps'] = mv_counts.get('stationary', 0)
            features['n_walking_overlaps'] = mv_counts.get('walking', 0)
            features['n_driving_overlaps'] = mv_counts.get('driving', 0)
            features['pct_stationary'] = features['n_stationary_overlaps'] / len(ov_day)
        else:
            features['n_stationary_overlaps'] = 0
            features['n_walking_overlaps'] = 0
            features['n_driving_overlaps'] = 0
            features['pct_stationary'] = 0

    else:
        for k in ['min_distance_m', 'mean_distance_m', 'median_distance_m',
                   'max_distance_m', 'std_distance_m',
                   'n_confirmed', 'n_neighborhood', 'n_city', 'pct_confirmed',
                   'overlap_hours_span', 'overlap_first_hour', 'overlap_last_hour',
                   'overlap_duration_hours',
                   'n_night_overlaps', 'has_night_overlap',
                   'n_morning_overlaps', 'n_evening_overlaps',
                   'n_stationary_overlaps', 'n_walking_overlaps', 'n_driving_overlaps',
                   'pct_stationary']:
            features[k] = 0
        features['min_distance_m'] = 999999
        features['mean_distance_m'] = 999999
        features['median_distance_m'] = 999999
        features['max_distance_m'] = 999999

    # ── Geographic features ──
    # Are both partners in the same country/region?
    if not p1_strict.empty and not p2_strict.empty:
        p1_median_lat = p1_strict['lat'].median()
        p1_median_lon = p1_strict['lon'].median()
        p2_median_lat = p2_strict['lat'].median()
        p2_median_lon = p2_strict['lon'].median()

        features['centroid_distance_m'] = _fast_distance_m(
            p1_median_lat, p1_median_lon, p2_median_lat, p2_median_lon
        )
        features['p1_lat'] = p1_median_lat
        features['p1_lon'] = p1_median_lon
        features['p2_lat'] = p2_median_lat
        features['p2_lon'] = p2_median_lon

        # Same country heuristic (rough bounding boxes)
        features['both_in_nl_be'] = float(
            49.5 <= p1_median_lat <= 53.5 and 2.5 <= p1_median_lon <= 7.2 and
            49.5 <= p2_median_lat <= 53.5 and 2.5 <= p2_median_lon <= 7.2
        )
        features['both_abroad'] = float(
            not features['both_in_nl_be'] and
            features['centroid_distance_m'] < 50000
        )
    else:
        features['centroid_distance_m'] = 999999
        features['p1_lat'] = 0
        features['p1_lon'] = 0
        features['p2_lat'] = 0
        features['p2_lon'] = 0
        features['both_in_nl_be'] = 0
        features['both_abroad'] = 0

    # ── Movement features ──
    # P1 travel distance (spread of locations)
    if len(p1_strict) >= 2:
        p1_lats = p1_strict['lat'].values
        p1_lons = p1_strict['lon'].values
        features['p1_spread_m'] = _fast_distance_m(
            p1_lats.min(), p1_lons.min(), p1_lats.max(), p1_lons.max()
        )
    else:
        features['p1_spread_m'] = 0

    if len(p2_strict) >= 2:
        p2_lats = p2_strict['lat'].values
        p2_lons = p2_strict['lon'].values
        features['p2_spread_m'] = _fast_distance_m(
            p2_lats.min(), p2_lons.min(), p2_lats.max(), p2_lons.max()
        )
    else:
        features['p2_spread_m'] = 0

    # Day of week (weekends might have different patterns)
    features['day_of_week'] = target_date.weekday()
    features['is_weekend'] = float(target_date.weekday() >= 5)

    return features


def build_feature_matrix(
    p1_df: pd.DataFrame,
    p2_df: pd.DataFrame,
    overlaps_df: pd.DataFrame,
    dates: List[date],
) -> pd.DataFrame:
    """Build feature matrix for a list of dates.

    Args:
        p1_df: Partner 1 full location dataset
        p2_df: Partner 2 full location dataset
        overlaps_df: Pre-computed overlaps
        dates: List of dates to extract features for

    Returns:
        DataFrame with one row per date, columns = features
    """
    rows = []
    for d in dates:
        feats = extract_day_features(p1_df, p2_df, overlaps_df, d)
        feats['date'] = d
        rows.append(feats)

    df = pd.DataFrame(rows)
    df = df.set_index('date')
    return df


# ─── Month-based grouping for leave-one-group-out CV ────────────────────────

def date_to_month_group(d: date) -> str:
    """Map a date to its month string for grouped CV."""
    return f"{d.year}-{d.month:02d}"


# ─── Training and Evaluation ────────────────────────────────────────────────

def train_and_evaluate(
    feature_df: pd.DataFrame,
    labels: Dict[date, str],
) -> Tuple[object, pd.DataFrame, Dict]:
    """Train classifier with leave-one-month-out cross-validation.

    Uses leave-one-month-out to avoid temporal leakage: the model never
    trains on data from the same month it predicts.

    Returns:
        (trained_model, predictions_df, metrics_dict)
    """
    if not HAS_SKLEARN:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        return None, pd.DataFrame(), {}

    # Build training set: only dates with labels
    labeled_dates = [d for d in feature_df.index if d in labels]
    X = feature_df.loc[labeled_dates]
    y = np.array([label_to_binary(labels[d]) for d in labeled_dates])
    groups = np.array([date_to_month_group(d) for d in labeled_dates])

    print(f"\nTraining set: {len(X)} labeled days")
    print(f"  Together: {y.sum()}, Apart: {(1 - y).sum()}")
    print(f"  Months: {len(set(groups))}")

    # Feature columns (exclude non-numeric)
    feature_cols = [c for c in X.columns if X[c].dtype in [np.float64, np.int64, float, int]]
    X_feat = X[feature_cols].fillna(0).values

    # Leave-one-month-out cross-validation
    logo = LeaveOneGroupOut()

    # GradientBoosting with conservative hyperparameters (small dataset)
    base_clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )

    # Cross-validated predictions
    cv_probs = cross_val_predict(
        base_clf, X_feat, y, cv=logo, groups=groups, method='predict_proba'
    )
    cv_preds = (cv_probs[:, 1] >= 0.5).astype(int)

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, cv_preds, average='binary', pos_label=1
    )
    cm = confusion_matrix(y, cv_preds)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'n_train': len(X),
        'n_together': int(y.sum()),
        'n_apart': int((1 - y).sum()),
    }

    print(f"\nLeave-one-month-out CV Results:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0][0]}, FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}, TP={cm[1][1]}")
    print(f"\n{classification_report(y, cv_preds, target_names=['apart', 'together'])}")

    # Train final model on all labeled data
    final_clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )
    final_clf.fit(X_feat, y)

    # Build predictions DataFrame
    preds_df = pd.DataFrame({
        'date': labeled_dates,
        'true_label': [labels[d] for d in labeled_dates],
        'true_binary': y,
        'cv_pred': cv_preds,
        'cv_prob_together': cv_probs[:, 1],
    })
    preds_df = preds_df.set_index('date')

    # Feature importance
    importances = sorted(
        zip(feature_cols, final_clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\nTop 15 features:")
    for name, imp in importances[:15]:
        print(f"  {name:35s} {imp:.4f}")

    return final_clf, preds_df, metrics


# ─── Predict All Days ────────────────────────────────────────────────────────

def predict_all_days(
    clf,
    feature_df: pd.DataFrame,
    labels: Dict[date, str],
) -> pd.DataFrame:
    """Generate predictions for ALL days (labeled + unlabeled).

    Returns DataFrame with columns: date, prediction, probability,
    true_label (if available), rule_based_result.
    """
    feature_cols = [c for c in feature_df.columns
                    if feature_df[c].dtype in [np.float64, np.int64, float, int]]
    X_all = feature_df[feature_cols].fillna(0).values

    probs = clf.predict_proba(X_all)[:, 1]
    preds = (probs >= 0.5).astype(int)

    results = pd.DataFrame({
        'date': feature_df.index,
        'ml_prediction': preds,
        'ml_probability': probs,
        'rule_based': [1 if feature_df.loc[d, 'n_overlaps'] > 0 else 0
                       for d in feature_df.index],
        'true_label': [labels.get(d, 'unknown') for d in feature_df.index],
        'n_overlaps': feature_df['n_overlaps'].values,
        'min_distance': feature_df['min_distance_m'].values,
    })
    results = results.set_index('date').sort_index()

    return results


# ─── Comparison with Rule-Based ──────────────────────────────────────────────

def compare_with_rule_based(
    results_df: pd.DataFrame,
    labels: Dict[date, str],
):
    """Compare ML predictions against rule-based system on labeled data."""
    labeled = results_df[results_df['true_label'] != 'unknown'].copy()
    labeled['true_binary'] = labeled['true_label'].apply(
        lambda x: 0 if x == 'apart' else 1
    )

    print("\n" + "=" * 70)
    print("COMPARISON: ML Classifier vs Rule-Based System")
    print("=" * 70)

    # Use CV predictions for ML if available (unbiased), otherwise final model
    ml_col = 'ml_prediction_cv' if 'ml_prediction_cv' in labeled.columns and labeled['ml_prediction_cv'].notna().any() else 'ml_prediction'
    ml_prob_col = 'ml_probability_cv' if 'ml_probability_cv' in labeled.columns and labeled['ml_probability_cv'].notna().any() else 'ml_probability'

    # Filter to rows where CV predictions exist for fair comparison
    if ml_col == 'ml_prediction_cv':
        cv_labeled = labeled[labeled['ml_prediction_cv'].notna()].copy()
        cv_labeled['ml_prediction_cv'] = cv_labeled['ml_prediction_cv'].astype(int)
    else:
        cv_labeled = labeled

    for name, col in [('Rule-Based', 'rule_based'), ('ML Classifier (CV)', ml_col)]:
        eval_df = cv_labeled if 'CV' in name else labeled
        y_true = eval_df['true_binary'].values
        y_pred = eval_df[col].values

        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1
        )
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n{name}:")
        print(f"  Precision: {p:.3f}")
        print(f"  Recall:    {r:.3f}")
        print(f"  F1:        {f1:.3f}")
        print(f"  TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # Show disagreements using CV predictions
    disagree = cv_labeled[cv_labeled[ml_col] != cv_labeled['rule_based']]
    if not disagree.empty:
        print(f"\nDisagreements ({len(disagree)} days):")
        for d, row in disagree.iterrows():
            true = row['true_label']
            ml = 'together' if row[ml_col] == 1 else 'apart'
            rb = 'together' if row['rule_based'] == 1 else 'apart'
            correct = 'ML' if row[ml_col] == row['true_binary'] else 'RB'
            prob = row[ml_prob_col] if ml_prob_col in row.index else row['ml_probability']
            print(f"  {d}: ML={ml}(p={prob:.3f}), RB={rb}, TRUE={true} -> {correct} correct")

    # Show false positives from rule-based that ML fixes
    rb_fp = cv_labeled[(cv_labeled['rule_based'] == 1) & (cv_labeled['true_binary'] == 0)]
    ml_fp = cv_labeled[(cv_labeled[ml_col] == 1) & (cv_labeled['true_binary'] == 0)]
    print(f"\nRule-based false positives: {len(rb_fp)}")
    if not rb_fp.empty:
        for d, row in rb_fp.iterrows():
            prob = row[ml_prob_col] if ml_prob_col in row.index else row['ml_probability']
            ml_fixed = 'FIXED' if row[ml_col] == 0 else 'still FP'
            print(f"  {d}: {ml_fixed} (prob={prob:.3f})")

    print(f"ML false positives (CV): {len(ml_fp)}")
    if not ml_fp.empty:
        for d, row in ml_fp.iterrows():
            prob = row[ml_prob_col] if ml_prob_col in row.index else row['ml_probability']
            print(f"  {d}: prob={prob:.3f}")

    # Show false negatives (missed together days)
    rb_fn = cv_labeled[(cv_labeled['rule_based'] == 0) & (cv_labeled['true_binary'] == 1)]
    ml_fn = cv_labeled[(cv_labeled[ml_col] == 0) & (cv_labeled['true_binary'] == 1)]
    print(f"\nRule-based missed days (FN): {len(rb_fn)}")
    if not rb_fn.empty:
        for d, row in rb_fn.iterrows():
            prob = row[ml_prob_col] if ml_prob_col in row.index else row['ml_probability']
            ml_found = 'FOUND' if row[ml_col] == 1 else 'still missed'
            print(f"  {d}: {ml_found} (prob={prob:.3f})")

    print(f"ML missed days (CV FN): {len(ml_fn)}")
    if not ml_fn.empty:
        for d, row in ml_fn.iterrows():
            prob = row[ml_prob_col] if ml_prob_col in row.index else row['ml_probability']
            print(f"  {d}: prob={prob:.3f}")

    # Overall day counts
    print(f"\n--- Full Dataset (all {len(results_df)} days) ---")
    print(f"Rule-based together days: {results_df['rule_based'].sum()}")
    print(f"ML together days: {results_df['ml_prediction'].sum()}")

    # Show uncertain predictions (probability near 0.5)
    uncertain = results_df[
        (results_df['ml_probability'] >= 0.3) & (results_df['ml_probability'] <= 0.7)
    ]
    if not uncertain.empty:
        print(f"\nUncertain ML predictions (0.3 < p < 0.7): {len(uncertain)} days")
        for d, row in uncertain.iterrows():
            true = row['true_label'] if row['true_label'] != 'unknown' else '?'
            rb = 'T' if row['rule_based'] else 'A'
            print(f"  {d}: p={row['ml_probability']:.3f}, RB={rb}, true={true}")


# ─── Immich Photo Cross-Reference (stub for when API key is available) ──────

def query_immich_photos(
    person_id: str,
    api_url: str,
    api_key: str,
    start_date: str = "2025-06-01",
    end_date: str = "2026-02-14",
) -> pd.DataFrame:
    """Query Immich API for photos of a person with GPS data.

    Searches for all photos containing the person, then fetches full asset
    details to extract GPS coordinates (not included in search results).

    Returns DataFrame with columns: timestamp, lat, lon, asset_id
    """
    import ssl
    import urllib.request
    import time

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    headers = {
        "x-api-key": api_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    def _api_get(path):
        req = urllib.request.Request(f"{api_url}{path}", headers=headers)
        resp = urllib.request.urlopen(req, context=ssl_ctx, timeout=30)
        return json.loads(resp.read())

    def _api_post(path, data):
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{api_url}{path}", data=body, headers=headers, method="POST"
        )
        resp = urllib.request.urlopen(req, context=ssl_ctx, timeout=30)
        return json.loads(resp.read())

    # Search for all photos of this person (paginate with page param)
    all_items = []
    page = 1
    while True:
        try:
            result = _api_post("/search/metadata", {
                "personIds": [person_id],
                "type": "IMAGE",
                "size": 1000,
                "page": page,
            })
        except Exception as e:
            print(f"  Immich search failed (page {page}): {e}")
            break

        items = result.get("assets", {}).get("items", [])
        if not items:
            break
        all_items.extend(items)
        if len(items) < 1000:
            break
        page += 1

    if not all_items:
        return pd.DataFrame()

    # Fetch full asset details to get GPS (batch with rate limiting)
    photos = []
    for i, item in enumerate(all_items):
        asset_id = item["id"]
        taken = item.get("fileCreatedAt") or item.get("createdAt")
        if not taken:
            continue

        try:
            ts = datetime.fromisoformat(taken.replace('Z', '+00:00')).replace(tzinfo=None)
        except ValueError:
            continue

        # Filter by date range
        if ts.date() < datetime.fromisoformat(start_date).date():
            continue
        if ts.date() > datetime.fromisoformat(end_date).date():
            continue

        try:
            full = _api_get(f"/assets/{asset_id}")
            exif = full.get("exifInfo", {})
            lat = exif.get("latitude")
            lon = exif.get("longitude")

            if lat and lon:
                photos.append({
                    'timestamp': ts,
                    'lat': float(lat),
                    'lon': float(lon),
                    'asset_id': asset_id,
                    'camera_make': exif.get('make', ''),
                })
        except Exception:
            pass

        # Rate limit: ~20 req/s
        if (i + 1) % 20 == 0:
            time.sleep(1)

    return pd.DataFrame(photos)


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ML Classifier for Co-location Detection")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}")
        return 1

    # Step 1: Load data using existing pipeline
    print("\n[1/5] Loading location data via existing pipeline...")
    analyzer = LocationAnalyzer(str(config_path))
    data_path = project_root / "data"
    stats = analyzer.run_analysis(data_path)

    if not stats:
        print("ERROR: Analysis pipeline failed")
        return 1

    p1 = analyzer.partner1_interpolated
    p2 = analyzer.partner2_interpolated
    overlaps = analyzer.overlaps_with_interpolation

    # Build ground truth labels first (Immich may enrich them)
    labels = build_ground_truth()

    # Step 2: Try Immich photo cross-reference
    print("\n[2/5] Attempting Immich photo cross-reference...")
    immich_api = os.environ.get("IMMICH_API_URL", "https://immich.slaenen.casa/api")
    immich_key = os.environ.get("IMMICH_API_KEY", "")
    maia_person_id = os.environ.get("IMMICH_PERSON_MAIA", "")
    sean_person_id = os.environ.get("IMMICH_PERSON_SEAN", "")

    # Bidirectional: Maia in Sean's photos + Sean in Maia's photos
    maia_photos = query_immich_photos(maia_person_id, immich_api, immich_key)
    sean_photos = query_immich_photos(sean_person_id, immich_api, immich_key)

    # Combine verified co-location photo evidence
    all_photo_frames = []
    immich_together_dates: set = set()

    if not maia_photos.empty:
        # All photos of Maia are from Sean's library (Realme camera) = together
        print(f"  Found {len(maia_photos)} photos of Maia with GPS (all from Sean's camera)")
        all_photo_frames.append(maia_photos)
        immich_together_dates |= set(maia_photos['timestamp'].dt.date.unique())

    if not sean_photos.empty:
        # Filter to Apple-only: photos of Sean from Maia's iPhone = together
        # Photos from Sean's own Realme don't prove co-location
        if 'camera_make' in sean_photos.columns:
            iphone_sean = sean_photos[
                sean_photos['camera_make'].str.contains('Apple', case=False, na=False)
            ]
        else:
            iphone_sean = pd.DataFrame()

        print(f"  Found {len(sean_photos)} photos of Sean with GPS, "
              f"{len(iphone_sean)} from Maia's iPhone (Apple)")
        if not iphone_sean.empty:
            all_photo_frames.append(iphone_sean)
            immich_together_dates |= set(iphone_sean['timestamp'].dt.date.unique())

    combined_photos = pd.concat(all_photo_frames, ignore_index=True) if all_photo_frames else pd.DataFrame()

    if immich_together_dates:
        print(f"  Photo-confirmed co-location dates: {len(immich_together_dates)}")
        # Enrich ground truth with photo evidence
        for d in sorted(immich_together_dates):
            if d not in labels:
                labels[d] = 'together'
                print(f"    Added {d} as 'together' from photo evidence")
    else:
        print("  Immich API unavailable or no matching photos - proceeding without photo data")

    # Derive nighttime home-separation labels for early dating period
    print("\n  Deriving nighttime home-separation labels...")
    # Use raw (non-interpolated) data for home detection to avoid bias
    p1_raw = analyzer.partner1_data
    p2_raw = analyzer.partner2_data
    labels = derive_nighttime_apart_labels(
        p1_raw, p2_raw, labels, overlaps_df=overlaps
    )

    # Step 3: Build ground truth and feature matrix
    print("\n[3/5] Building training set...")

    # Get date range from data
    all_dates_p1 = set(p1['timestamp'].dt.date.unique())
    all_dates_p2 = set(p2['timestamp'].dt.date.unique())
    all_dates = sorted(all_dates_p1 | all_dates_p2)

    print(f"  Date range: {all_dates[0]} to {all_dates[-1]}")
    print(f"  Total days with data: {len(all_dates)}")
    print(f"  Days with ground truth: {len([d for d in all_dates if d in labels])}")

    feature_df = build_feature_matrix(p1, p2, overlaps, all_dates)

    # Add photo features if Immich data available
    if not combined_photos.empty:
        photo_day_counts = combined_photos.groupby(
            combined_photos['timestamp'].dt.date
        ).size()
        feature_df['n_couple_photos'] = feature_df.index.map(
            lambda d: photo_day_counts.get(d, 0)
        ).astype(float)
        feature_df['has_couple_photos'] = (feature_df['n_couple_photos'] > 0).astype(float)
        print(f"  Added photo features: {int(feature_df['has_couple_photos'].sum())} days with couple photos")
    print(f"  Feature matrix: {feature_df.shape[0]} days x {feature_df.shape[1]} features")

    # Step 4: Train and evaluate
    print("\n[4/5] Training ML classifier...")
    clf, cv_preds, metrics = train_and_evaluate(feature_df, labels)

    if clf is None:
        return 1

    # Step 5: Predict all days and compare
    print("\n[5/5] Predicting all days and comparing with rule-based...")
    results = predict_all_days(clf, feature_df, labels)

    # For labeled data: use CV predictions (unbiased) instead of final model
    for d in cv_preds.index:
        if d in results.index:
            results.loc[d, 'ml_prediction_cv'] = int(cv_preds.loc[d, 'cv_pred'])
            results.loc[d, 'ml_probability_cv'] = cv_preds.loc[d, 'cv_prob_together']

    compare_with_rule_based(results, labels)

    # Save results
    output_path = project_root / "output"
    output_path.mkdir(exist_ok=True)

    results_file = output_path / "ml_predictions.csv"
    results.to_csv(results_file)
    print(f"\nResults saved to {results_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"ML days together: {results['ml_prediction'].sum()}")
    print(f"Rule-based days together: {results['rule_based'].sum()}")
    if metrics:
        print(f"ML CV F1: {metrics['f1']:.3f} (precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f})")

    return 0


def run_ensemble(
    analyzer,
    threshold: float = 0.7,
) -> Dict[date, str]:
    """Run ML ensemble on top of an already-analyzed pipeline.

    The ensemble strategy:
      - Rule-based says 'together' → keep (high precision)
      - Rule-based says 'apart' but ML says 'together' with p >= threshold
        AND both partners have data → rescue as 'ml_together'

    Args:
        analyzer: LocationAnalyzer that has already run run_analysis()
        threshold: ML probability threshold for rescuing days (default 0.7)

    Returns:
        Dict mapping date → 'together' | 'apart' | 'no_data' | 'ml_together'
    """
    if not HAS_SKLEARN:
        print("  scikit-learn not available — skipping ML ensemble")
        return {}

    import time as _time

    p1 = analyzer.partner1_interpolated
    p2 = analyzer.partner2_interpolated
    overlaps = analyzer.overlaps_with_interpolation

    # Build ground truth
    labels = build_ground_truth()

    # Immich photo enrichment
    immich_api = os.environ.get("IMMICH_API_URL", "https://immich.slaenen.casa/api")
    immich_key = os.environ.get("IMMICH_API_KEY", "")
    maia_pid = os.environ.get("IMMICH_PERSON_MAIA", "")
    sean_pid = os.environ.get("IMMICH_PERSON_SEAN", "")

    print("  Querying Immich for photo evidence...")
    maia_photos = query_immich_photos(maia_pid, immich_api, immich_key)
    sean_photos = query_immich_photos(sean_pid, immich_api, immich_key)

    all_photo_frames = []
    immich_dates: set = set()

    if not maia_photos.empty:
        all_photo_frames.append(maia_photos)
        immich_dates |= set(maia_photos['timestamp'].dt.date.unique())

    if not sean_photos.empty and 'camera_make' in sean_photos.columns:
        iphone_sean = sean_photos[
            sean_photos['camera_make'].str.contains('Apple', case=False, na=False)
        ]
        if not iphone_sean.empty:
            all_photo_frames.append(iphone_sean)
            immich_dates |= set(iphone_sean['timestamp'].dt.date.unique())

    for d in immich_dates:
        if d not in labels:
            labels[d] = 'together'

    combined_photos = pd.concat(all_photo_frames, ignore_index=True) if all_photo_frames else pd.DataFrame()
    print(f"  Photo-confirmed dates: {len(immich_dates)}")

    # Nighttime separation labels
    labels = derive_nighttime_apart_labels(
        analyzer.partner1_data, analyzer.partner2_data, labels, overlaps_df=overlaps
    )

    # Build features for all dates
    all_dates_p1 = set(p1['timestamp'].dt.date.unique())
    all_dates_p2 = set(p2['timestamp'].dt.date.unique())
    all_dates = sorted(all_dates_p1 | all_dates_p2)

    feature_df = build_feature_matrix(p1, p2, overlaps, all_dates)

    if not combined_photos.empty:
        photo_day_counts = combined_photos.groupby(
            combined_photos['timestamp'].dt.date
        ).size()
        feature_df['n_couple_photos'] = feature_df.index.map(
            lambda d: photo_day_counts.get(d, 0)
        ).astype(float)
        feature_df['has_couple_photos'] = (feature_df['n_couple_photos'] > 0).astype(float)

    # Train model
    print("  Training ensemble classifier...")
    clf, _, metrics = train_and_evaluate(feature_df, labels)

    if clf is None:
        return {}

    # Predict all days
    results = predict_all_days(clf, feature_df, labels)

    # Build ensemble day classifications
    # Only count confirmed-tier overlaps in home region for rule-based "together"
    confirmed_overlaps = analyzer.get_confirmed_overlaps(overlaps)
    p1_days = set(p1['timestamp'].dt.date.unique())
    p2_days = set(p2['timestamp'].dt.date.unique())
    together_days = set()
    if confirmed_overlaps is not None and not confirmed_overlaps.empty:
        together_days = set(confirmed_overlaps['timestamp'].dt.date.unique())

    ensemble: Dict[date, str] = {}
    ml_rescued = 0

    for d in all_dates:
        both_tracked = d in p1_days and d in p2_days
        rb_together = d in together_days and both_tracked

        if rb_together:
            ensemble[d] = 'together'
        elif both_tracked and d in results.index:
            ml_prob = results.loc[d, 'ml_probability']
            if ml_prob >= threshold:
                ensemble[d] = 'ml_together'
                ml_rescued += 1
            else:
                ensemble[d] = 'apart'
        elif both_tracked:
            ensemble[d] = 'apart'
        else:
            ensemble[d] = 'no_data'

    # Stats: only count from first_meeting onwards
    first_meeting = None
    fm = analyzer.config.get('key_dates', {}).get('first_meeting')
    if fm:
        first_meeting = date.fromisoformat(str(fm))

    def _from_meeting(d):
        return first_meeting is None or d >= first_meeting

    rb_together_count = sum(1 for d, v in ensemble.items() if v == 'together' and _from_meeting(d))
    ml_rescued_count = sum(1 for d, v in ensemble.items() if v == 'ml_together' and _from_meeting(d))
    total_together = rb_together_count + ml_rescued_count
    total_tracked = sum(1 for d, v in ensemble.items() if v != 'no_data' and _from_meeting(d))

    print(f"\n  Ensemble Results:")
    print(f"    Rule-based together: {rb_together_count}")
    print(f"    ML-rescued (p>={threshold}): {ml_rescued_count}")
    print(f"    Total together: {total_together}/{total_tracked} tracked days")
    if metrics:
        print(f"    ML CV F1: {metrics['f1']:.3f}")

    return ensemble


if __name__ == "__main__":
    sys.exit(main())
