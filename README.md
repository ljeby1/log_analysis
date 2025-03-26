# log_analysis
Review data  logs and categorize the data for patterns
Here’s the "How to Use This Script" section formatted in Markdown, suitable for inclusion in a README or documentation:

---

## How to Use This Script

Follow these steps to analyze your free-text incident log using the provided Python script:

1. **Prepare Your Data**  
   Ensure your incident log is saved as a CSV file (e.g., `incident_log.csv`) with at least a `description` column containing the free-text error reports. If you have timestamps, include them in a `timestamp` column in a recognizable format (e.g., "2025-03-25 14:30:00"). Example CSV structure:
   ```
   timestamp,description
   2025-03-25 10:15:00,"App crashed after clicking save"
   2025-03-25 11:00:00,"Network timeout during upload"
   ```

2. **Install Dependencies**  
   Before running the script, install the required Python libraries. Open your terminal and run:
   ```bash
   pip install pandas numpy nltk scikit-learn matplotlib
   ```
   Ensure you have Python 3.7+ installed.

3. **Run the Script**  
   - Save the script as `analyze_incidents.py`.
   - Update the `file_path` variable in the script to point to your CSV file (e.g., `file_path = 'path/to/your/incident_log.csv'`).
   - Execute the script from your terminal:
     ```bash
     python analyze_incidents.py
     ```

### What to Expect
- **Output**: The script will:
  - Print cluster analysis (e.g., top terms and sample incidents per cluster) and the top 10 keywords overall.
  - Generate two plots: a bar chart of cluster sizes and (if timestamps are provided) a time series of cluster frequencies.
  - Save a `clustered_incidents.csv` file with the original descriptions and assigned cluster labels.
- **Sample Output**:
  ```
  === Cluster Analysis ===
  Cluster 0 (size: 300):
  Top terms: ['network', 'timeout', 'slow', 'connection', 'error']
  Sample: network timeout during upload

  Cluster 1 (size: 200):
  Top terms: ['crash', 'app', 'froze', 'save', 'update']
  Sample: app crashed after clicking save

  Top 10 keywords across all incidents: ['network', 'crash', 'timeout', 'app', 'error', 'slow', 'connection', 'froze', 'save', 'update']
  ```

### Customization
- **Number of Clusters**: Modify `num_clusters` in the `analyze_patterns()` function (default is 5).
- **Feature Count**: Adjust `max_features` in `TfidfVectorizer` (default is 100) to include more or fewer terms.
- **Top Terms per Cluster**: Change the `-5` in `top_indices` to `-10` (or another value) to show more keywords per cluster.

If you encounter issues or need help tailoring the script to your specific data, feel free to ask!

--- 

This Markdown section provides clear, concise instructions for users of varying expertise levels. Let me know if you'd like to adjust or expand it further!
