#!/usr/bin/env python3
"""Main entry point for immigration location analysis.

Runs the complete pipeline: GPS analysis → ML ensemble → visualizations → IND report.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datetime import date as _date

from analyzer import LocationAnalyzer
from visualizer import LocationVisualizer
from report import INDReportGenerator


def main():
    """Run complete analysis pipeline."""
    print("Immigration Location Analysis")
    print("=" * 60)
    print()

    # Initialize analyzer
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        print("Please create config.yaml first.")
        return 1

    analyzer = LocationAnalyzer(str(config_path))

    # Set data path
    data_path = Path(__file__).parent / "data"
    if not data_path.exists():
        print(f"Error: Data directory not found at {data_path}")
        print("Please create data/ directory and add JSON files.")
        return 1

    # Run analysis
    print("Starting analysis...")
    print()

    stats = analyzer.run_analysis(data_path)

    if not stats:
        print("\nAnalysis failed. Please check data files.")
        return 1

    # Create output directory
    output_path = Path(__file__).parent / analyzer.config['output']['directory']
    output_path.mkdir(parents=True, exist_ok=True)

    # ML Ensemble step
    ensemble_days = None
    try:
        from ml_classifier import run_ensemble
        print("\n" + "=" * 60)
        print("ML Ensemble Classification")
        print("=" * 60)
        ensemble_days = run_ensemble(analyzer, threshold=0.7)
        if ensemble_days:
            fm = analyzer.config.get('key_dates', {}).get('first_meeting')
            first_meeting = _date.fromisoformat(str(fm)) if fm else None
            def _fm(d): return first_meeting is None or d >= first_meeting
            rb_count = sum(1 for d, v in ensemble_days.items() if v == 'together' and _fm(d))
            ml_count = sum(1 for d, v in ensemble_days.items() if v == 'ml_together' and _fm(d))
            total = rb_count + ml_count
            tracked = sum(1 for d, v in ensemble_days.items() if v != 'no_data' and _fm(d))
            print(f"\n  Ensemble: {total}/{tracked} days together "
                  f"({total/tracked*100:.1f}%) [{rb_count} GPS + {ml_count} ML]")
    except ImportError:
        print("\nNote: ML classifier not available (scikit-learn not installed)")
        print("Running with rule-based analysis only.")
    except Exception as e:
        print(f"\nWarning: ML ensemble failed ({e}), using rule-based only.")

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = LocationVisualizer(analyzer.config)

    # Build presence matrix if we have interpolated data
    presence_matrix = None
    state_matrix = None
    if analyzer.partner1_interpolated is not None and analyzer.partner2_interpolated is not None:
        print("Building presence matrix...")
        end_date = (
            analyzer.overlaps_with_interpolation['timestamp'].max()
            if analyzer.overlaps_with_interpolation is not None
            and not analyzer.overlaps_with_interpolation.empty
            else analyzer.start_date
        )
        presence_matrix, state_matrix = analyzer.interpolator.build_presence_matrix(
            analyzer.partner1_interpolated,
            analyzer.partner2_interpolated,
            analyzer.start_date,
            end_date,
            precomputed_overlaps=analyzer.overlaps_with_interpolation,
        )

    visualizer.create_all_visualizations(
        analyzer.partner1_interpolated,
        analyzer.partner2_interpolated,
        analyzer.overlaps,
        analyzer.overlaps_with_interpolation,
        presence_matrix,
        output_path,
        state_matrix=state_matrix,
        ensemble_days=ensemble_days
    )

    # Generate report
    print("\nGenerating IND report...")
    report_gen = INDReportGenerator(analyzer.config, analyzer.metadata)

    report_gen.generate_report(
        stats['raw'],
        stats['interpolated'],
        analyzer.overlaps,
        analyzer.overlaps_with_interpolation,
        output_path / "IND_Evidence_Report.txt",
        ind_metrics=stats.get('ind_metrics')
    )

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print("Results Summary:")
    print(f"  Days together (raw): {stats['raw']['unique_days']}")
    print(f"  Days together (interpolated): {stats['interpolated']['unique_days']}")
    print(f"  Nights together: {stats['raw']['nights_together']}")
    print(f"  Total co-locations (raw): {stats['raw']['total_colocations']}")
    print(f"  Total co-locations (interpolated): {stats['interpolated']['total_colocations']}")
    if 'ind_metrics' in stats:
        ind = stats['ind_metrics']
        print()
        print("IND Evidence Metrics (rule-based):")
        print(f"  Tracked days together: {ind['days_together']}/{ind['total_tracked_days']} ({ind['pct_days_together']}%)")
        print(f"  Nights at shared address: {ind['nights_at_shared_address']}/{ind['total_tracked_nights']} ({ind['pct_nights_together']}%)")
    if ensemble_days:
        fm = analyzer.config.get('key_dates', {}).get('first_meeting')
        first_meeting = _date.fromisoformat(str(fm)) if fm else None
        def _fm(d): return first_meeting is None or d >= first_meeting
        rb = sum(1 for d, v in ensemble_days.items() if v == 'together' and _fm(d))
        ml = sum(1 for d, v in ensemble_days.items() if v == 'ml_together' and _fm(d))
        tracked = sum(1 for d, v in ensemble_days.items() if v != 'no_data' and _fm(d))
        print()
        print("Ensemble Metrics (GPS + ML):")
        print(f"  Total days together: {rb + ml}/{tracked} ({(rb+ml)/tracked*100:.1f}%)")
        print(f"    GPS-confirmed: {rb}")
        print(f"    ML-predicted:  {ml}")
    print()
    print(f"All outputs saved to: {output_path}")
    print()
    print("Files generated:")
    for file in sorted(output_path.iterdir()):
        print(f"  - {file.name}")
    print()
    print("Ready for IND submission!")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
