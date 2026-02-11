# Immigration Location Analysis

Analyzes location history data to provide objective evidence of cohabitation for IND (Dutch immigration) partner visa applications (TEV/MVV).

## Purpose

This tool processes location history exports from mobile phones to demonstrate:
- Consistent cohabitation at a shared residence
- Synchronized daily routines and activities
- Joint international travel
- Extended relationship duration

Location data is objective, timestamped, and difficult to fabricate, making it strong supporting evidence for genuine relationship verification.

## Key Features

### 1. Presence Interpolation (Innovation)

Mobile phones report location sparsely when stationary (every 1-4 hours). This tool intelligently fills gaps:

- If two consecutive reports are at the same location within 4 hours, assumes continuous presence
- Infers overnight stays from evening + morning reports near home
- Transparent methodology - shows both raw and interpolated results
- Never fabricates movement - only fills gaps in stationary periods

### 2. Comprehensive Analysis

- **Raw data analysis**: Conservative, using only actual GPS reports
- **Interpolated analysis**: More complete picture accounting for sparse reporting
- **Country detection**: Automatically identifies vacations abroad
- **Overnight detection**: Counts nights spent together at shared residence
- **Monthly breakdown**: Shows relationship consistency over time

### 3. Multiple Visualizations

- **Calendar heatmap**: GitHub-style daily co-presence view
- **Hourly matrix**: 24h patterns showing daily rhythms
- **Monthly summary**: Bar charts of days/nights together per month
- **Interactive map**: Folium map with privacy-conscious coordinate rounding
- **Confidence timeline**: Shows data density and interpolation usage

### 4. IND-Ready Report

- Data source verification (SHA256 hashes)
- Transparent methodology explanation
- Statistics mapped to IND criteria
- Both conservative and complete analyses
- Ready for submission

## Installation

### Requirements

- Python 3.12+
- pip or uv for package management

### Setup

```bash
# Clone or download this repository
cd immigration-location-analysis

# Install dependencies
pip install -r requirements.txt
# OR using uv (faster)
uv pip install -r requirements.txt

# Configure your analysis
cp config.yaml config.yaml.backup  # Optional: backup
nano config.yaml  # Edit with your dates and file names
```

## Usage

### 1. Export Location Data

**For Google/Android users:**
1. Go to Google Takeout (https://takeout.google.com)
2. Select "Location History" (Timeline format)
3. Download JSON files
4. Look for files like `Timeline_YYYY_MM.json` or `Records.json`

**For Apple/iPhone users:**
1. Settings > Privacy > Location Services > System Services > Significant Locations
2. Request data export from Apple (may take several days)
3. Alternatively, use third-party tools to export location history

### 2. Prepare Data

```bash
# Create data directory (already exists, symlinked)
# Copy your JSON files to data/
cp ~/Downloads/Timeline_*.json data/

# Update config.yaml with your filenames
nano config.yaml
```

### 3. Configure Analysis

Edit `config.yaml`:

```yaml
# Set your relationship start date
start_date: "2025-06-01"

# Add your key dates
key_dates:
  first_meeting: "2025-06-06"
  separation_start: "2025-10-13"
  # ... etc

# List your data files
data_files:
  partner1:
    - "Timeline_your_phone1.json"
    - "Timeline_your_phone2.json"
  partner2:
    - "Timeline_partner_iphone.json"
```

### 4. Run Analysis

```bash
python run.py
```

The script will:
1. Load and parse location data
2. Apply interpolation to fill gaps
3. Find co-locations (with and without interpolation)
4. Generate all visualizations
5. Create IND report
6. Save everything to `output/`

### 5. Review Results

Check `output/` directory:
- `IND_Evidence_Report.txt` - Main report for IND submission
- `calendar_raw.png` - Daily co-presence (raw data)
- `calendar_interpolated.png` - Daily co-presence (with interpolation)
- `hourly_matrix.png` - Hourly patterns
- `monthly_summary.png` - Monthly statistics
- `interactive_map.html` - Interactive map (open in browser)
- `confidence_timeline.png` - Data quality visualization

## Configuration Options

### Analysis Parameters

```yaml
thresholds:
  distance_meters: 150      # Co-location distance threshold
  time_minutes: 30          # Time window for matching

interpolation:
  max_gap_hours: 4          # Maximum gap to interpolate
  overnight_hours:
    start: 22               # Evening start
    end: 6                  # Morning end
  home_tolerance_meters: 300  # "Near home" tolerance
```

### Country Detection

Add custom country bounds in `config.yaml`:

```yaml
country_bounds:
  your_country: [min_lat, max_lat, min_lon, max_lon]
```

## Methodology

### How It Works

1. **Data Loading**: Parses Google/Apple JSON formats, handles multiple phones per person
2. **Interpolation**: Fills gaps between stationary reports (max 4h)
3. **Overlap Detection**: Finds times when both partners were at same place
4. **Country Detection**: Identifies vacations using coordinate bounds
5. **Statistical Analysis**: Calculates days together, nights together, etc.
6. **Visualization**: Creates multiple chart types
7. **Report Generation**: Maps evidence to IND criteria

### Why Interpolation?

Mobile phones conserve battery by reporting location infrequently when stationary:
- Moving: Every few minutes
- Stationary: Every 1-4 hours

Without interpolation, you get sparse data that underrepresents actual presence. For example:
- Phone reports location at 20:00 at home
- Next report at 08:00 next morning at home
- **Without interpolation**: Looks like 2 separate visits
- **With interpolation**: Correctly shows overnight stay

The tool transparently shows both analyses so you can see the difference.

## Privacy Considerations

### Data Privacy

- **No real data in repository**: `.gitignore` excludes all JSON files
- **Coordinate rounding**: Visualizations round to ~100m accuracy
- **Local processing**: All analysis happens on your machine
- **No external services**: No data sent anywhere

### For IND Submission

- Consider providing visualizations only (not raw JSON files)
- Maps already use privacy-conscious coordinate rounding
- Original files available for verification if requested
- SHA256 hashes prove data authenticity without exposing content

## Troubleshooting

### "No overlaps found"

Possible causes:
- Check `start_date` in config.yaml - may be filtering out all data
- Verify JSON file formats match Google/Apple expected formats
- Try increasing distance or time thresholds
- Ensure both partners' files have data in same date range

### "File not found"

- Check file paths in `config.yaml` match actual filenames
- Ensure files are in `data/` directory
- Use exact filenames (case-sensitive on Linux/Mac)

### "Empty DataFrame"

- JSON format may not be recognized
- Check first few lines of JSON file match expected formats
- Try running with just one file to isolate the issue

## Advanced Usage

### Custom Interpolation Settings

```python
# Edit src/analyzer.py
interpolator = LocationInterpolator(
    max_gap_hours=6.0,          # Increase max gap
    home_tolerance_m=500.0,     # Looser home detection
    overnight_start=21,         # Earlier evening
    overnight_end=7             # Later morning
)
```

### Additional Visualizations

```python
# In run.py, add custom visualizations
from src.visualizer import LocationVisualizer

visualizer = LocationVisualizer(analyzer.config)
# Add your custom plots here
```

### Exporting for Other Purposes

The analysis generates pandas DataFrames that can be exported:

```python
# In run.py
analyzer.overlaps.to_csv(output_path / "overlaps_raw.csv")
analyzer.overlaps_with_interpolation.to_csv(output_path / "overlaps_interpolated.csv")
```

## Technical Details

### Technology Stack

- **Python 3.12+**: Modern Python with type hints
- **pandas**: Data manipulation and analysis
- **geopy**: Geographic distance calculations
- **matplotlib/seaborn**: Static visualizations
- **plotly**: Interactive charts
- **folium**: Interactive maps
- **PyYAML**: Configuration management

### Code Quality

- Type hints throughout
- Docstrings on all public functions
- Clean imports, no star imports
- pathlib for cross-platform paths
- Graceful error handling
- Progress reporting during long operations

### Performance

- Efficient pandas operations
- Indexed DataFrame searches for overlap detection
- Vectorized operations where possible
- Typical runtime: 1-5 minutes for 6 months of data from 2 people

## Legal and Ethical Notes

### Data Authenticity

- Location data is automatically generated by devices
- Cannot be manually edited in Google/Apple exports
- SHA256 hashes provide verification
- Timestamped and GPS-verified

### Interpolation Transparency

- Methodology clearly explained in report
- Both raw and interpolated results provided
- Never fabricates location changes
- Only fills gaps in stationary periods
- IND can verify approach is sound

### Privacy Rights

- This is YOUR data, you have the right to use it
- Partner consent recommended before using their data
- Be mindful of third parties in location history
- Consider privacy when sharing visualizations

## Contributing

This is a personal project for IND evidence generation. Improvements welcome:

- Additional visualization types
- Better country detection
- Support for more location export formats
- Performance optimizations

## License

This project is provided as-is for personal use in immigration applications. No warranty provided. Use at your own discretion.

## Support

For issues or questions:
1. Check this README thoroughly
2. Review `config.yaml` comments
3. Check error messages in terminal
4. Review sample data format in existing analyzer code

## Acknowledgments

Based on earlier location analysis work, enhanced with:
- Presence interpolation methodology
- Multiple visualization types
- IND-specific evidence mapping
- Transparent dual-analysis approach

---

**Note**: This tool provides supporting evidence. IND requires multiple forms of documentation (cohabitation agreements, bank statements, photos, etc.). Location analysis should supplement, not replace, traditional evidence.
