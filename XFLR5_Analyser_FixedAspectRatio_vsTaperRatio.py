import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt

class TaperRatioAnalyzer:
    """
    Specialized analyzer for taper ratio optimization at fixed AR=10
    """
    
    def __init__(self, target_cl=0.34):
        self.target_cl = target_cl
        self.results = []
        
    def analyze_taper_ratios(self, directory_path):
        """
        Analyze all taper ratio configurations in a directory
        """
        dir_path = Path(directory_path)
        csv_files = list(dir_path.glob("*.csv"))
        
        print(f"🔍 Found {len(csv_files)} CSV files for taper ratio analysis")
        
        for file_path in csv_files:
            print(f"📊 Analyzing: {file_path.name}")
            result = self.analyze_single_file(file_path)
            if result:
                self.results.append(result)
        
        return self.compile_taper_comparison()
    
    def analyze_single_file(self, file_path):
        """
        Analyze a single XFLR5 file for taper ratio optimization
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find data start
            data_lines = []
            found_header = False
            
            for line in lines:
                if line.startswith('alpha'):
                    found_header = True
                    continue
                if found_header and line.strip() and not line.startswith('---') and not 'xflr5' in line:
                    data_lines.append(line.strip())
            
            # Parse data
            alphas, cls, cds, cms, cdis = [], [], [], [], []
            
            for line in data_lines:
                parts = line.split(',')
                if len(parts) >= 6:
                    try:
                        alphas.append(float(parts[0].strip()))
                        cls.append(float(parts[2].strip()))
                        cds.append(float(parts[5].strip()))
                        cms.append(float(parts[8].strip()))
                        cdis.append(float(parts[3].strip()))  # Induced drag component
                    except:
                        continue
            
            if not alphas:
                print(f"⚠️  No data parsed from {file_path.name}")
                return None
                
            df = pd.DataFrame({
                'alpha': alphas, 'CL': cls, 'CD': cds, 
                'Cm': cms, 'CDi': cdis
            })
            
        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")
            return None
        
        # Calculate metrics
        df['L/D'] = df['CL'] / df['CD']
        df['CDp'] = df['CD'] - df['CDi']  # Profile drag
        
        # Find key points
        max_ld_idx = df['L/D'].idxmax()
        max_ld_data = df.loc[max_ld_idx]
        
        # Cruise point interpolation
        cruise_data = self._find_cruise_point(df, self.target_cl)
        if cruise_data is None:
            print(f"⚠️  Could not find cruise point for {file_path.name}")
            return None
        
        # Extract taper ratio from filename
        taper_ratio = self._extract_taper_ratio(file_path.name)
        
        # Calculate Oswald efficiency estimate
        e_estimate = self._estimate_oswald_efficiency(df, self.target_cl)
        
        results = {
            'Filename': file_path.name,
            'Taper_Ratio': taper_ratio,
            'L_D_max': round(max_ld_data['L/D'], 2),
            'Alpha_at_L_D_max': round(max_ld_data['alpha'], 2),
            'L_D_at_cruise': round(cruise_data['L/D_cruise'], 2),
            'Alpha_cruise': round(cruise_data['alpha_cruise'], 2),
            'CD_cruise': round(cruise_data['CD_cruise'], 5),
            'CDi_cruise': round(cruise_data['CDi_cruise'], 5),
            'CDp_cruise': round(cruise_data['CDp_cruise'], 5),
            'Cm_cruise': round(cruise_data['Cm_cruise'], 4),
            'CL_max': round(df['CL'].max(), 3),
            'Oswald_e_estimate': round(e_estimate, 3),
            'Induced_Drag_Ratio': round(cruise_data['CDi_cruise'] / cruise_data['CD_cruise'], 3)
        }
        
        return results
    
    def _extract_taper_ratio(self, filename):
        """Extract taper ratio from filename"""
        filename = filename.upper()
        
        # Look for point notation (point7 -> 0.7, point75 -> 0.75)
        point_match = re.search(r'POINT(\d+)', filename)
        if point_match:
            number = point_match.group(1)
            if len(number) == 1:
                return float(f"0.{number}")
            elif len(number) == 2:
                return float(f"0.{number}")
        
        # Look for decimal notation (TR0.7, TR_0.7)
        decimal_match = re.search(r'TR[_\s]*(\d+\.?\d*)', filename)
        if decimal_match:
            return float(decimal_match.group(1))
        
        return None
    
    def _find_cruise_point(self, df, target_cl):
        """Find cruise point with detailed drag breakdown"""
        try:
            # Find points that bracket the target CL
            above = df[df['CL'] >= target_cl]
            below = df[df['CL'] <= target_cl]
            
            if above.empty or below.empty:
                print(f"⚠️  No data points bracket CL={target_cl}")
                return None
            
            above = above.iloc[0]
            below = below.iloc[-1]
            
            if above['CL'] == below['CL']:
                return None
            
            frac = (target_cl - below['CL']) / (above['CL'] - below['CL'])
            
            cruise_data = {
                'alpha_cruise': below['alpha'] + frac * (above['alpha'] - below['alpha']),
                'CD_cruise': below['CD'] + frac * (above['CD'] - below['CD']),
                'CDi_cruise': below['CDi'] + frac * (above['CDi'] - below['CDi']),
                'CDp_cruise': below['CDp'] + frac * (above['CDp'] - below['CDp']),
                'Cm_cruise': below['Cm'] + frac * (above['Cm'] - below['Cm']),
                'L/D_cruise': target_cl / (below['CD'] + frac * (above['CD'] - below['CD']))
            }
            
            return cruise_data
            
        except Exception as e:
            print(f"❌ Error in cruise point interpolation: {e}")
            return None
    
    def _estimate_oswald_efficiency(self, df, cl_cruise):
        """Estimate Oswald efficiency factor from induced drag"""
        try:
            # Find data point closest to cruise CL
            idx = (df['CL'] - cl_cruise).abs().idxmin()
            cd_i = df.loc[idx, 'CDi']
            ar = 10.0  # Fixed for this analysis
            
            # Oswald efficiency: e = CL² / (π * AR * CDi)
            if cd_i > 0:
                return (cl_cruise ** 2) / (np.pi * ar * cd_i)
            return 0.9  # Default estimate
        except:
            return 0.9  # Default estimate
    
    def compile_taper_comparison(self):
        """Compile results into comparison table"""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('Taper_Ratio')
        
        # Create comprehensive comparison
        print("\n" + "="*80)
        print("📈 TAPER RATIO OPTIMIZATION RESULTS (AR=10)")
        print("="*80)
        
        # Display key comparison metrics
        comparison_cols = ['Taper_Ratio', 'L_D_max', 'L_D_at_cruise', 'Alpha_cruise', 
                          'CD_cruise', 'Oswald_e_estimate', 'Induced_Drag_Ratio', 'CL_max']
        
        available_cols = [col for col in comparison_cols if col in df.columns]
        print(df[available_cols].to_string(index=False))
        
        return df
    
    def plot_taper_comparison(self, results_df):
        """Create comparison plots for taper ratio analysis"""
        if results_df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: L/D vs Taper Ratio
        axes[0,0].plot(results_df['Taper_Ratio'], 'bo-', linewidth=2, markersize=8, label='Max L/D')
        axes[0,0].plot(results_df['Taper_Ratio'], results_df['L_D_at_cruise'], 'ro-', linewidth=2, markersize=8, label='Cruise L/D')
        axes[0,0].set_xlabel('Taper Ratio (λ)')
        axes[0,0].set_ylabel('L/D Ratio')
        axes[0,0].set_title('L/D vs Taper Ratio')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # Plot 2: Drag Components vs Taper Ratio
        axes[0,1].plot(results_df['Taper_Ratio'], results_df['CD_cruise']*1000, 'ko-', linewidth=2, markersize=8, label='Total CD × 1000')
        axes[0,1].plot(results_df['Taper_Ratio'], results_df['CDi_cruise']*1000, 'bo-', linewidth=2, markersize=8, label='Induced CD × 1000')
        axes[0,1].plot(results_df['Taper_Ratio'], results_df['CDp_cruise']*1000, 'ro-', linewidth=2, markersize=8, label='Profile CD × 1000')
        axes[0,1].set_xlabel('Taper Ratio (λ)')
        axes[0,1].set_ylabel('Drag Coefficient × 1000')
        axes[0,1].set_title('Drag Components vs Taper Ratio')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # Plot 3: Oswald Efficiency vs Taper Ratio
        axes[1,0].plot(results_df['Taper_Ratio'], results_df['Oswald_e_estimate'], 'go-', linewidth=2, markersize=8)
        axes[1,0].set_xlabel('Taper Ratio (λ)')
        axes[1,0].set_ylabel('Oswald Efficiency (e)')
        axes[1,0].set_title('Oswald Efficiency vs Taper Ratio')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Cruise Angle of Attack
        axes[1,1].plot(results_df['Taper_Ratio'], results_df['Alpha_cruise'], 'mo-', linewidth=2, markersize=8)
        axes[1,1].set_xlabel('Taper Ratio (λ)')
        axes[1,1].set_ylabel('Cruise Angle of Attack (°)')
        axes[1,1].set_title('Cruise AoA vs Taper Ratio')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('taper_ratio_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def debug_file(file_path):
    """Debug function to check file structure"""
    print(f"\n🔧 DEBUGGING: {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    print("First 15 lines:")
    for i, line in enumerate(lines[:15]):
        print(f"{i:2}: {line.strip()}")
    
    # Check data structure
    data_start = -1
    for i, line in enumerate(lines):
        if line.startswith('alpha'):
            data_start = i
            break
    
    if data_start != -1 and data_start + 1 < len(lines):
        print(f"\nFirst data line: {lines[data_start + 1].strip()}")

def main():
    """Main function for taper ratio analysis"""
    analyzer = TaperRatioAnalyzer(target_cl=0.34)
    
    # Analyze taper ratio directory
    taper_dir = "Analysis_Fixed_AR_vs_TR"
    
    # First, debug one file to check structure
    csv_files = list(Path(taper_dir).glob("*.csv"))
    if csv_files:
        debug_file(csv_files[0])
    
    results_df = analyzer.analyze_taper_ratios(taper_dir)
    
    if not results_df.empty:
        # Save results
        results_df.to_csv('taper_ratio_results.csv', index=False)
        print(f"💾 Results saved to 'taper_ratio_results.csv'")
        
        # Create plots
        analyzer.plot_taper_comparison(results_df)
        
        # Find best configuration
        best_idx = results_df['L_D_max'].idxmax()
        best_config = results_df.loc[best_idx]
        
        print(f"\n🏆 OPTIMAL TAPER RATIO:")
        print(f"   λ = {best_config['Taper_Ratio']}")
        print(f"   Max L/D: {best_config['L_D_max']}")
        print(f"   Cruise L/D: {best_config['L_D_at_cruise']}")
        print(f"   Oswald Efficiency: {best_config['Oswald_e_estimate']}")
        print(f"   Induced Drag Ratio: {best_config['Induced_Drag_Ratio']:.1%}")
    else:
        print("❌ No results generated - check file formats and data")

if __name__ == "__main__":
    main()