#!/usr/bin/env python3
"""
Autonomous LFQ Dataset Processing Agent
========================================

Takes a paper folder/file as input and fully automates:
1. Reconnaissance of data files
2. Configuration setup
3. Data normalization (PHASE 1)
4. Merge to unified CSV (PHASE 2)
5. Z-score calculation (PHASE 3)

Logs every step sequentially for real-time tracking and debugging.

Usage:
    python autonomous_agent.py "data_raw/Author et al. - Year/"
    python autonomous_agent.py "data_raw/Author et al. - Year/data_file.xlsx"
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import traceback
from typing import Dict, List, Optional, Tuple
import re


class LFQProcessingAgent:
    """Autonomous agent for processing LFQ proteomics datasets."""

    def __init__(self, input_path: str, project_root: Path = None):
        """
        Initialize agent with input path.

        Parameters:
        -----------
        input_path : str
            Path to paper folder or data file
        project_root : Path
            Project root directory (auto-detected if None)
        """
        self.input_path = Path(input_path)

        # Auto-detect project root
        if project_root is None:
            current_dir = Path.cwd()
            for parent in [current_dir] + list(current_dir.parents):
                if (parent / 'references' / 'human_matrisome_v2.csv').exists():
                    self.project_root = parent
                    break
            else:
                self.project_root = Path.cwd()
        else:
            self.project_root = project_root

        # Initialize state
        self.state = {
            "phase": "initialization",
            "current_step": None,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "errors": [],
            "completed_steps": []
        }

        # Will be set during reconnaissance
        self.output_dir = None
        self.log_file = None
        self.state_file = None
        self.config = None

    def initialize_workspace(self, study_id: str):
        """Create output directory and logging files."""
        # Create output directory
        self.output_dir = self.project_root / f"XX_{study_id}_paper_to_csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize log file
        self.log_file = self.output_dir / "agent_log.md"
        self.state_file = self.output_dir / "agent_state.json"

        # Write initial log
        self._log("# Autonomous LFQ Processing Agent Log", overwrite=True)
        self._log(f"\n**Study ID:** {study_id}")
        self._log(f"**Start Time:** {self.state['start_time']}")
        self._log(f"**Input Path:** {self.input_path}")
        self._log(f"**Output Directory:** {self.output_dir.relative_to(self.project_root)}")
        self._log("\n---\n")

        # Save initial state
        self._save_state()

        self._log("✅ Workspace initialized")

    def _log(self, message: str, overwrite: bool = False):
        """Write message to log file with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.log_file:
            mode = 'w' if overwrite else 'a'
            with open(self.log_file, mode, encoding='utf-8') as f:
                if not overwrite and not message.startswith('#'):
                    f.write(f"\n[{timestamp}] {message}\n")
                else:
                    f.write(f"{message}\n")

            # Also print to console for real-time tracking
            print(f"[{timestamp}] {message}")
        else:
            # Before log file is created
            print(f"[{timestamp}] {message}")

    def _save_state(self):
        """Save current state to JSON for debugging."""
        if self.state_file:
            self.state["last_updated"] = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)

    def _update_state(self, phase: str = None, step: str = None, status: str = None):
        """Update agent state and save."""
        if phase:
            self.state["phase"] = phase
        if step:
            self.state["current_step"] = step
        if status:
            self.state["status"] = status
        self._save_state()

    def _mark_step_complete(self, step: str):
        """Mark a step as completed."""
        self.state["completed_steps"].append({
            "step": step,
            "timestamp": datetime.now().isoformat()
        })
        self._save_state()
        self._log(f"✅ Completed: {step}")

    def _log_error(self, error: str, exception: Exception = None):
        """Log error with full traceback."""
        self.state["errors"].append({
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc() if exception else None
        })
        self.state["status"] = "error"
        self._save_state()

        self._log(f"\n❌ ERROR: {error}")
        if exception:
            self._log(f"\n```\n{traceback.format_exc()}\n```")

    def run(self) -> bool:
        """
        Execute full pipeline.

        Returns:
        --------
        bool: True if successful, False if error
        """
        try:
            # PHASE 0: Reconnaissance
            self._log("\n## PHASE 0: Reconnaissance")
            self._update_state(phase="reconnaissance")

            if not self._reconnaissance():
                return False

            # PHASE 1: Data Normalization
            self._log("\n## PHASE 1: Data Normalization")
            self._update_state(phase="normalization")

            if not self._normalize_data():
                return False

            # PHASE 2: Merge to Unified CSV
            self._log("\n## PHASE 2: Merge to Unified CSV")
            self._update_state(phase="merge")

            if not self._merge_to_unified():
                return False

            # PHASE 3: Z-Score Calculation
            self._log("\n## PHASE 3: Z-Score Calculation")
            self._update_state(phase="zscore")

            if not self._calculate_zscores():
                return False

            # Final summary
            self._log("\n## PIPELINE COMPLETE")
            self._update_state(status="completed")
            self._log(f"\n**Total Steps Completed:** {len(self.state['completed_steps'])}")
            self._log(f"**Total Time:** {self._get_elapsed_time()}")
            self._log(f"\n✅ All phases completed successfully!")

            return True

        except Exception as e:
            self._log_error("Unexpected error in pipeline", e)
            return False

    def _get_elapsed_time(self) -> str:
        """Calculate elapsed time since start."""
        start = datetime.fromisoformat(self.state['start_time'])
        elapsed = datetime.now() - start
        minutes = int(elapsed.total_seconds() / 60)
        seconds = int(elapsed.total_seconds() % 60)
        return f"{minutes}m {seconds}s"

    def _reconnaissance(self) -> bool:
        """
        PHASE 0: Identify data files and generate configuration.

        Returns:
        --------
        bool: True if successful
        """
        try:
            self._log("\n### Step 0.1: Identify paper folder")

            # Determine paper folder
            if self.input_path.is_file():
                paper_folder = self.input_path.parent
                data_file = self.input_path
            else:
                paper_folder = self.input_path
                data_file = None

            self._log(f"Paper folder: {paper_folder}")

            # Extract study ID from folder name
            # Format: "Author et al. - Year"
            folder_name = paper_folder.name
            match = re.search(r'(.+?)\s+et al\.\s*-\s*(\d{4})', folder_name)
            if match:
                author = match.group(1).strip()
                year = match.group(2)
                study_id = f"{author}_{year}"
            else:
                # Fallback: use folder name
                study_id = folder_name.replace(' ', '_').replace('.', '')

            self._log(f"Detected Study ID: {study_id}")
            self._mark_step_complete("Identify paper folder")

            # Initialize workspace
            self.initialize_workspace(study_id)

            # Step 0.2: Find data files
            self._log("\n### Step 0.2: Find data files in folder")

            data_files = []
            for ext in ['.xlsx', '.xls', '.csv', '.tsv']:
                data_files.extend(list(paper_folder.glob(f"*{ext}")))

            self._log(f"Found {len(data_files)} data files:")
            for f in data_files:
                self._log(f"  - {f.name}")

            if len(data_files) == 0:
                self._log_error("No data files found in paper folder")
                return False

            # If data_file not specified, use the largest file
            if data_file is None:
                data_file = max(data_files, key=lambda f: f.stat().st_size)
                self._log(f"\n📊 Selected largest file: {data_file.name} ({data_file.stat().st_size / 1024 / 1024:.2f} MB)")

            self._mark_step_complete("Find data files")

            # Step 0.3: Inspect data file
            self._log("\n### Step 0.3: Inspect data file structure")

            if data_file.suffix == '.xlsx' or data_file.suffix == '.xls':
                # Read Excel file
                xls = pd.ExcelFile(data_file)
                sheet_names = xls.sheet_names
                self._log(f"Excel sheets found: {sheet_names}")

                # Try to identify main data sheet
                data_sheet = None
                for sheet in sheet_names:
                    if any(keyword in sheet.lower() for keyword in ['data', 'protein', 'abundance', 'matrix']):
                        data_sheet = sheet
                        break

                if data_sheet is None:
                    data_sheet = sheet_names[0]

                self._log(f"Selected data sheet: {data_sheet}")

                # Preview first few rows
                df_preview = pd.read_excel(data_file, sheet_name=data_sheet, nrows=5)
                self._log(f"\nColumns found ({len(df_preview.columns)}):")
                for col in df_preview.columns[:10]:  # Show first 10 columns
                    self._log(f"  - {col}")
                if len(df_preview.columns) > 10:
                    self._log(f"  ... and {len(df_preview.columns) - 10} more columns")

            else:
                # CSV/TSV file
                df_preview = pd.read_csv(data_file, nrows=5, sep='\t' if data_file.suffix == '.tsv' else ',')
                self._log(f"Columns found ({len(df_preview.columns)}):")
                for col in df_preview.columns[:10]:
                    self._log(f"  - {col}")

            self._mark_step_complete("Inspect data file")

            # Step 0.4: Generate configuration template
            self._log("\n### Step 0.4: Generate configuration template")

            config = {
                "study_id": study_id,
                "paper_folder": str(paper_folder.relative_to(self.project_root)),
                "data_file": str(data_file.relative_to(self.project_root)),
                "data_sheet": data_sheet if data_file.suffix in ['.xlsx', '.xls'] else None,
                "species": "Homo sapiens",  # Default, needs manual review
                "tissue": "Unknown",  # Needs manual review
                "method": "Label-free LC-MS/MS",  # Default
                "young_ages": [],  # Needs manual review
                "old_ages": [],  # Needs manual review
                "compartments": None,  # Needs manual review
                "output_dir": str(self.output_dir.relative_to(self.project_root))
            }

            # Save config
            config_file = self.output_dir / "study_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            self._log(f"✅ Configuration template saved: {config_file.name}")
            self._log("\n⚠️  **MANUAL REVIEW REQUIRED:**")
            self._log("   Please edit study_config.json to fill in:")
            self._log("   - species (Homo sapiens / Mus musculus)")
            self._log("   - tissue (organ/tissue type)")
            self._log("   - young_ages (list of ages)")
            self._log("   - old_ages (list of ages)")
            self._log("   - compartments (if applicable)")

            self.config = config
            self._mark_step_complete("Generate configuration")

            return True

        except Exception as e:
            self._log_error("Error in reconnaissance phase", e)
            return False

    def _normalize_data(self) -> bool:
        """
        PHASE 1: Data normalization pipeline.

        Returns:
        --------
        bool: True if successful
        """
        try:
            self._log("\n### Step 1.1: Load and validate configuration")

            # Check if config has required fields
            required_fields = ['study_id', 'data_file', 'species', 'young_ages', 'old_ages']
            missing_fields = [f for f in required_fields if not self.config.get(f)]

            if missing_fields:
                self._log_error(f"Missing required config fields: {missing_fields}")
                self._log("\n⚠️  Please edit study_config.json and re-run agent")
                return False

            self._log("✅ Configuration validated")
            self._mark_step_complete("Validate configuration")

            # Step 1.2: Execute normalization pipeline
            self._log("\n### Step 1.2: Execute normalization pipeline")
            self._log("⚠️  This step requires implementing full PHASE 1 logic")
            self._log("    See 01_LFQ_DATASET_NORMALIZATION_AND_MERGE.md for details")

            # TODO: Implement full PHASE 1 pipeline here
            # For now, placeholder that expects manual processing

            wide_format_file = self.output_dir / f"{self.config['study_id']}_wide_format.csv"

            if not wide_format_file.exists():
                self._log_error(f"Expected output file not found: {wide_format_file.name}")
                self._log("\n⚠️  Please run PHASE 1 processing manually using:")
                self._log(f"    01_LFQ_DATASET_NORMALIZATION_AND_MERGE.md")
                return False

            self._log(f"✅ Found wide-format file: {wide_format_file.name}")
            self._mark_step_complete("Data normalization")

            return True

        except Exception as e:
            self._log_error("Error in normalization phase", e)
            return False

    def _merge_to_unified(self) -> bool:
        """
        PHASE 2: Merge study to unified CSV.

        Returns:
        --------
        bool: True if successful
        """
        try:
            self._log("\n### Step 2.1: Prepare merge to unified CSV")

            wide_format_file = self.output_dir / f"{self.config['study_id']}_wide_format.csv"

            if not wide_format_file.exists():
                self._log_error(f"Wide-format file not found: {wide_format_file.name}")
                return False

            # Import merge function
            import sys
            sys.path.insert(0, str(self.project_root / '11_subagent_for_LFQ_ingestion'))
            from merge_to_unified import merge_study_to_unified

            self._log("✅ Loaded merge_to_unified function")

            # Step 2.2: Execute merge
            self._log("\n### Step 2.2: Execute merge")

            unified_csv_path = '08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'

            self._log(f"Merging {wide_format_file.name} to {unified_csv_path}")

            df_merged = merge_study_to_unified(
                study_csv=str(wide_format_file.relative_to(self.project_root)),
                unified_csv=unified_csv_path,
                project_root=self.project_root
            )

            self._log(f"✅ Merge complete: {len(df_merged)} total rows in unified CSV")
            self._mark_step_complete("Merge to unified CSV")

            return True

        except Exception as e:
            self._log_error("Error in merge phase", e)
            return False

    def _calculate_zscores(self) -> bool:
        """
        PHASE 3: Calculate z-scores for new study.

        Returns:
        --------
        bool: True if successful
        """
        try:
            self._log("\n### Step 3.1: Prepare z-score calculation")

            # Import z-score function
            import sys
            sys.path.insert(0, str(self.project_root / '11_subagent_for_LFQ_ingestion'))
            from universal_zscore_function import calculate_study_zscores

            self._log("✅ Loaded calculate_study_zscores function")

            # Determine groupby columns based on compartments
            if self.config.get('compartments'):
                groupby_columns = ['Tissue_Compartment']
            else:
                groupby_columns = ['Tissue']

            self._log(f"Using groupby columns: {groupby_columns}")

            # Step 3.2: Execute z-score calculation
            self._log("\n### Step 3.2: Execute z-score calculation")

            df_updated = calculate_study_zscores(
                study_id=self.config['study_id'],
                groupby_columns=groupby_columns,
                csv_path='08_merged_ecm_dataset/merged_ecm_aging_zscore.csv',
                backup=True
            )

            self._log(f"✅ Z-scores calculated for {len(df_updated)} rows")
            self._mark_step_complete("Calculate z-scores")

            return True

        except Exception as e:
            self._log_error("Error in z-score calculation phase", e)
            return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Missing required argument")
        print("\nUsage:")
        print("  python autonomous_agent.py <paper_folder_or_file>")
        print("\nExamples:")
        print("  python autonomous_agent.py 'data_raw/Randles et al. - 2021/'")
        print("  python autonomous_agent.py 'data_raw/Randles et al. - 2021/ASN.2020101442-File027.xlsx'")
        sys.exit(1)

    input_path = sys.argv[1]

    # Initialize agent
    print("\n" + "="*70)
    print("AUTONOMOUS LFQ PROCESSING AGENT")
    print("="*70)

    agent = LFQProcessingAgent(input_path)

    # Run pipeline
    success = agent.run()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
