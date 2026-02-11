#!/usr/bin/env python3
"""Main entry point for immigration location analysis.

Simple script to run the complete analysis pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

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

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = LocationVisualizer(analyzer.config)

    # Build presence matrix if we have interpolated data
    presence_matrix = None
    if analyzer.partner1_interpolated is not None and analyzer.partner2_interpolated is not None:
        print("Building presence matrix...")
        presence_matrix = analyzer.interpolator.build_presence_matrix(
            analyzer.partner1_interpolated,
            analyzer.partner2_interpolated,
            analyzer.start_date,
            analyzer.overlaps_with_interpolation['timestamp'].max()
            if analyzer.overlaps_with_interpolation is not None and not analyzer.overlaps_with_interpolation.empty
            else analyzer.start_date
        )

    visualizer.create_all_visualizations(
        analyzer.partner1_interpolated,
        analyzer.partner2_interpolated,
        analyzer.overlaps,
        analyzer.overlaps_with_interpolation,
        presence_matrix,
        output_path
    )

    # Generate report
    print("\nGenerating IND report...")
    report_gen = INDReportGenerator(analyzer.config, analyzer.metadata)

    report_gen.generate_report(
        stats['raw'],
        stats['interpolated'],
        analyzer.overlaps,
        analyzer.overlaps_with_interpolation,
        output_path / "IND_Evidence_Report.txt"
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
