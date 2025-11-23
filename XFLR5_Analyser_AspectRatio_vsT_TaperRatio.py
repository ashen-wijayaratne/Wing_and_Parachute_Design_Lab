import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class XFLR5BatchAnalyzer:
    """
    Efficient batch analyzer for XFLR5 CSV files with automatic AR/TR extraction
    """
    
    def __init__(self, target_cl: float = 0.34):
        self.target_cl = target_cl
        self.results = []
    
    def extract_ar_tr_from_filename(self, filename: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract Aspect Ratio and Taper Ratio from filename
        Handles formats like: AR7_andTR_point75, AR7_TR_point75, AR7.5_TR0.75, etc.
        """
        filename = filename.upper()
        
        # Pattern for AR: AR followed by numbers (with optional decimal)
        ar_match = re.search(r'AR[_\s]*(\d+\.?\d*)', filename)
        ar = float(ar_match.group(1)) if ar_match else None
        
        # Pattern for TR: TR followed by numbers, or "point" notation
        tr_match = re.search(r'TR[_\s]*(\d+\.?\d*)', filename)
        if tr_match:
            tr = float(tr_match.group(1))
        else:
            # Try "point" notation: TR_point75 -> 0.75
            point_match = re.search(r'TR[_\s]*POINT[_\s]*(\d+)', filename)
            tr = float(f"0.{point_match.group(1)}") if point_match else None
        
        return ar, tr
    
    def parse_xflr5_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Efficiently parse XFLR5 CSV file with minimal memory usage
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find data section efficiently
            data_start = next((i for i, line in enumerate(lines) 
                            if line.startswith('alpha')), -1)
            
            if data_start == -1:
                print(f"⚠️  No data header found in {file_path.name}")
                return None
            
            # Parse only data lines (skip header and metadata)
            data_lines = []
            for line in lines[data_start + 1:]:
                line = line.strip()
                if line and not any(x in line for x in ['---', 'xflr5', 'Plane name', 'Polar name']):
                    data_lines.append(line)
            
            # Parse data efficiently using list comprehensions
            parsed_data = []
            for line in data_lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 9:
                    try:
                        parsed_data.append({
                            'alpha': float(parts[0]),
                            'CL': float(parts[2]),
                            'CD': float(parts[5]),
                            'Cm': float(parts[8])
                        })
                    except (ValueError, IndexError):
                        continue
            
            return pd.DataFrame(parsed_data) if parsed_data else None
            
        except Exception as e:
            print(f"❌ Error parsing {file_path.name}: {e}")
            return None
    
    def analyze_single_file(self, file_path: Path) -> Optional[Dict]:
        """
        Analyze a single XFLR5 file and return comprehensive results
        """
        df = self.parse_xflr5_file(file_path)
        if df is None or df.empty:
            return None
        
        # Extract AR/TR from filename
        ar, tr = self.extract_ar_tr_from_filename(file_path.stem)
        
        # Calculate L/D efficiently using vectorized operations
        df['L/D'] = df['CL'] / df['CD']
        
        # Find key metrics
        max_ld_idx = df['L/D'].idxmax()
        max_ld_data = df.loc[max_ld_idx]
        
        # Find cruise point using efficient search
        cruise_data = self._find_cruise_point(df, self.target_cl)
        
        if cruise_data is None:
            print(f"⚠️  Could not find cruise point for {file_path.name}")
            return None
        
        # Compile results
        results = {
            'Filename': file_path.name,
            'Aspect_Ratio': ar,
            'Taper_Ratio': tr,
            'L_D_max': round(max_ld_data['L/D'], 2),
            'Alpha_at_L_D_max': round(max_ld_data['alpha'], 2),
            'L_D_at_cruise': round(cruise_data['L/D_cruise'], 2),
            'CL_cruise': self.target_cl,
            'Alpha_cruise': round(cruise_data['alpha_cruise'], 2),
            'CD_cruise': round(cruise_data['CD_cruise'], 5),
            'Cm_cruise': round(cruise_data['Cm_cruise'], 4),
            'CL_max': round(df['CL'].max(), 3),
            'Stall_Alpha': f">{df['alpha'].max()}°" if df['CL'].idxmax() == df['alpha'].idxmax() else f"{df.loc[df['CL'].idxmax(), 'alpha']}°",
            'Data_Points': len(df)
        }
        
        return results
    
    def _find_cruise_point(self, df: pd.DataFrame, target_cl: float) -> Optional[Dict]:
        """
        Efficiently find cruise point using interpolation
        """
        # Find bracketing points
        above_mask = df['CL'] >= target_cl
        below_mask = df['CL'] <= target_cl
        
        if not above_mask.any() or not below_mask.any():
            return None
        
        above = df[above_mask].iloc[0]
        below = df[below_mask].iloc[-1]
        
        if above['CL'] == below['CL']:
            return None
        
        # Linear interpolation
        frac = (target_cl - below['CL']) / (above['CL'] - below['CL'])
        alpha_cruise = below['alpha'] + frac * (above['alpha'] - below['alpha'])
        CD_cruise = below['CD'] + frac * (above['CD'] - below['CD'])
        Cm_cruise = below['Cm'] + frac * (above['Cm'] - below['Cm'])
        L_D_cruise = target_cl / CD_cruise
        
        return {
            'alpha_cruise': alpha_cruise,
            'CD_cruise': CD_cruise,
            'Cm_cruise': Cm_cruise,
            'L/D_cruise': L_D_cruise
        }
    
    def analyze_directory(self, directory_path: str, pattern: str = "*.csv") -> pd.DataFrame:
        """
        Analyze all CSV files in a directory matching the pattern
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            print(f"❌ Directory not found: {directory_path}")
            return pd.DataFrame()
        
        csv_files = list(dir_path.glob(pattern))
        print(f"Found {len(csv_files)} CSV files in {directory_path}")
        
        for file_path in csv_files:
            print(f"Analyzing: {file_path.name}")
            result = self.analyze_single_file(file_path)
            if result:
                self.results.append(result)
        
        return self.compile_results()
    
    def analyze_file_list(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Analyze a specific list of files
        """
        for file_path in file_paths:
            path = Path(file_path)
            if path.exists():
                print(f"Analyzing: {path.name}")
                result = self.analyze_single_file(path)
                if result:
                    self.results.append(result)
            else:
                print(f"!!File not found: {file_path}")
        
        return self.compile_results()
    
    def compile_results(self) -> pd.DataFrame:
        """
        Compile all results into a sorted DataFrame
        """
        if not self.results:
            print("!! No results to compile")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        # Sort by Aspect Ratio and Taper Ratio for better comparison
        sort_columns = []
        if 'Aspect_Ratio' in df.columns:
            sort_columns.append('Aspect_Ratio')
        if 'Taper_Ratio' in df.columns:
            sort_columns.append('Taper_Ratio')
        
        if sort_columns:
            df = df.sort_values(by=sort_columns)
        
        return df
    
    def save_results(self, output_path: str = "xflr5_analysis_results.csv"):
        """
        Save results to CSV file
        """
        if self.results:
            results_df = self.compile_results()
            results_df.to_csv(output_path, index=False)
            print(f"Done! Results saved to: {output_path}")
            return results_df
        else:
            print("!! No results to save")
            return pd.DataFrame()

def main():
    """
    Main function to demonstrate usage
    """
    # Initialize analyzer
    analyzer = XFLR5BatchAnalyzer(target_cl=0.34)
    
    analysis_dir = "Analysis_Fixed_TR_vs_AR"
    results_df = analyzer.analyze_directory(analysis_dir)
    
    # Display results
    if not results_df.empty:
        print("\n" + "="*80)
        print("XFLR5 BATCH ANALYSIS RESULTS")
        print("="*80)
        
        # Display key columns for comparison
        display_columns = ['Filename', 'Aspect_Ratio', 'Taper_Ratio', 'L_D_max', 
                          'L_D_at_cruise', 'Alpha_cruise', 'CD_cruise', 'CL_max']
        
        # Only show columns that exist in the results
        available_columns = [col for col in display_columns if col in results_df.columns]
        print(results_df[available_columns].to_string(index=False))
        
        # Save results
        analyzer.save_results()
        
        # Print best configuration
        if 'L_D_max' in results_df.columns:
            best_idx = results_df['L_D_max'].idxmax()
            best_config = results_df.loc[best_idx]
            print(f"\n BEST CONFIGURATION:")
            print(f"   File: {best_config['Filename']}")
            print(f"   AR: {best_config.get('Aspect_Ratio', 'N/A')}, TR: {best_config.get('Taper_Ratio', 'N/A')}")
            print(f"   Max L/D: {best_config['L_D_max']}")
            print(f"   Cruise L/D: {best_config['L_D_at_cruise']}")
    else:
        print("!! No results generated")

if __name__ == "__main__":
    main()