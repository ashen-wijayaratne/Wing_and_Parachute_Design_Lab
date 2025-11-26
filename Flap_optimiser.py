import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class ImprovedFlapAnalyzer:
    # Function to initialize the analyzer with data folder and requirements
    def __init__(self, data_folder='Total_Data'):
        self.data_folder = Path(data_folder)
        self.results = []
        
        # Flight phase requirements
        self.requirements = {
            'landing': {
                'V': 40.28,
                'CL_at_speed': 1.097,
                'CL_max_required': 1.863,
                'stall_margin': 1.3,
                'safe_alpha_min': 0.0,
                'safe_alpha_max': 10.0,
                'preferred_alpha_min': 3.0,
                'preferred_alpha_max': 8.0,
            },
            'takeoff': {
                'V': 46.67,
                'CL_at_speed': 0.818,
                'CL_max_required': 1.177,
                'stall_margin': 1.2,
                'safe_alpha_min': 0.0,
                'safe_alpha_max': 10.0,
                'preferred_alpha_min': 3.0,
                'preferred_alpha_max': 8.0,
            }
        }
        
        self.W = 19620  # N
        self.S = 18     # m²
        self.rho = 1.225  # kg/m³
    
    # Function to parse XFLR5 CSV files
    def parse_xflr5_csv(self, filepath):
        """Parse XFLR5 CSV files and extract aerodynamic data"""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Find data header line
            header_line = None
            for i, line in enumerate(lines):
                if 'alpha' in line.lower() and 'CL' in line:
                    header_line = i
                    break
            
            if header_line is None:
                return None
            
            df = pd.read_csv(filepath, skiprows=header_line)
            df.columns = [col.strip() for col in df.columns]
            
            required_cols = ['alpha', 'CL', 'CD', 'Cm']
            if not all(col in df.columns for col in required_cols):
                return None
            
            # Convert to numeric and clean data
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=required_cols)
            
            if len(df) < 5:
                return None
            
            return df
            
        except Exception as e:
            return None
    
    # Function to extract flap angle and flight phase from filename
    def extract_file_info(self, filename):
        """Extract flap angle and flight phase from filename"""
        filename_lower = filename.lower()
        
        # Detect flight phase
        if 'landing' in filename_lower:
            phase = 'landing'
        elif 'take-off' in filename_lower or 'takeoff' in filename_lower:
            phase = 'takeoff'
        else:
            phase = None
        
        # Extract flap deflection angle
        flap_patterns = [
            (30, [' 30 deg', ' 30deg', '_30_deg', '_30deg']),
            (25, [' 25 deg', ' 25deg', '_25_deg', '_25deg']),
            (20, [' 20 deg', ' 20deg', '_20_deg', '_20deg']),
            (15, [' 15 deg', ' 15deg', '_15_deg', '_15deg']),
            (10, [' 10 deg', ' 10deg', '_10_deg', '_10deg']),
            (5, [' 5 deg', ' 5deg', '_5_deg', '_5deg']),
            (0, [' 0 deg', ' 0deg', '_0_deg', '_0deg'])
        ]
        
        for angle, patterns in flap_patterns:
            for pattern in patterns:
                if pattern in filename_lower:
                    return angle, phase
        
        return 0, phase
    
    # Function to find optimal operating point within constraints
    def find_optimal_operating_point(self, df, req, phase):
        """Find optimal operating point within constraints"""
        df = df.copy()
        df['L/D'] = df['CL'] / df['CD'].replace(0, np.nan)
        
        # Filter to safe operating range
        safe_range = df[
            (df['alpha'] >= req['safe_alpha_min']) & 
            (df['alpha'] <= req['safe_alpha_max'])
        ].copy()
        
        if safe_range.empty:
            return None
        
        # Find configurations meeting CL requirement
        meets_req = safe_range[safe_range['CL'] >= req['CL_at_speed']].copy()
        
        if not meets_req.empty:
            # Prioritize preferred alpha range
            preferred_range = meets_req[
                (meets_req['alpha'] >= req['preferred_alpha_min']) &
                (meets_req['alpha'] <= req['preferred_alpha_max'])
            ].copy()
            
            if not preferred_range.empty:
                # Score based on L/D and alpha position
                preferred_range['score'] = preferred_range['L/D'] * (1 - (preferred_range['alpha'] - req['preferred_alpha_min']) / 100)
                optimal_idx = preferred_range['score'].idxmax()
            else:
                # Fallback: best L/D outside preferred range
                optimal_idx = meets_req['L/D'].idxmax()
            
            status = 'optimal'
        else:
            # No configuration meets requirement - use best available
            max_cl = safe_range['CL'].max()
            candidates = safe_range[safe_range['CL'] >= max_cl * 0.98].copy()
            optimal_idx = candidates['alpha'].idxmin()
            status = 'insufficient'
        
        optimal_point = df.loc[optimal_idx]
        
        return {
            'alpha': float(optimal_point['alpha']),
            'CL': float(optimal_point['CL']),
            'CD': float(optimal_point['CD']),
            'Cm': float(optimal_point['Cm']),
            'LD': float(optimal_point['L/D']),
            'status': status,
            'in_preferred_range': req['preferred_alpha_min'] <= optimal_point['alpha'] <= req['preferred_alpha_max']
        }
    
    # Function to analyze single flap configuration
    def analyze_configuration(self, df, flap_angle, phase):
        """Comprehensive analysis of single flap configuration"""
        if phase not in self.requirements:
            return None
        
        req = self.requirements[phase]
        df = df.copy()
        
        # Calculate key performance metrics
        CL_max = float(df['CL'].max())
        alpha_at_CL_max = float(df.loc[df['CL'].idxmax(), 'alpha'])
        
        df['L/D'] = df['CL'] / df['CD'].replace(0, np.nan)
        max_LD = float(df['L/D'].max())
        alpha_at_max_LD = float(df.loc[df['L/D'].idxmax(), 'alpha'])
        
        # Find optimal operating point
        optimal_op = self.find_optimal_operating_point(df, req, phase)
        if optimal_op is None:
            return None
        
        # Safety and performance metrics
        stall_margin_ratio = CL_max / req['CL_at_speed']
        safe_stall_margin = stall_margin_ratio >= req['stall_margin']
        meets_CL_requirement = optimal_op['CL'] >= req['CL_at_speed']
        CL_margin = optimal_op['CL'] - req['CL_at_speed']
        
        # Scoring system
        scores = {}
        
        # CL achievement scoring (35%)
        if meets_CL_requirement:
            excess_pct = (CL_margin / req['CL_at_speed']) * 100
            if 5 <= excess_pct <= 15:
                scores['CL_requirement'] = 100 + excess_pct
            elif excess_pct > 15:
                scores['CL_requirement'] = 100 + 15 - (excess_pct - 15) * 0.5
            else:
                scores['CL_requirement'] = 100 + excess_pct * 2
        else:
            scores['CL_requirement'] = 100 * (optimal_op['CL'] / req['CL_at_speed'])
        scores['CL_requirement'] = max(0, min(120, scores['CL_requirement']))
        
        # Efficiency scoring (25%)
        max_realistic_LD = 35
        scores['efficiency'] = 100 * min(optimal_op['LD'] / max_realistic_LD, 1.2)
        
        # Alpha quality scoring (20%)
        if optimal_op['in_preferred_range']:
            pref_range = req['preferred_alpha_max'] - req['preferred_alpha_min']
            alpha_position = (optimal_op['alpha'] - req['preferred_alpha_min']) / pref_range
            scores['alpha_quality'] = 100 * (1 - alpha_position * 0.3)
        elif optimal_op['alpha'] < req['preferred_alpha_min']:
            scores['alpha_quality'] = 110
        else:
            excess = optimal_op['alpha'] - req['preferred_alpha_max']
            scores['alpha_quality'] = max(30, 100 - excess * 20)
        
        # Stall margin scoring (15%)
        if safe_stall_margin:
            extra_margin = (stall_margin_ratio - req['stall_margin']) / req['stall_margin']
            scores['stall_margin'] = 100 * (1 + min(extra_margin * 0.5, 0.2))
        else:
            scores['stall_margin'] = 100 * (stall_margin_ratio / req['stall_margin'])
        scores['stall_margin'] = min(120, scores['stall_margin'])
        
        # Simplicity scoring (5%)
        scores['simplicity'] = 100 * (1 - flap_angle / 40)
        
        # Calculate total weighted score
        weights = {
            'CL_requirement': 0.35,
            'efficiency': 0.25,
            'alpha_quality': 0.20,
            'stall_margin': 0.15,
            'simplicity': 0.05
        }
        
        total_score = sum(scores[k] * weights[k] for k in scores)
        
        return {
            'phase': phase,
            'flap_angle': flap_angle,
            'optimal_alpha': optimal_op['alpha'],
            'optimal_CL': optimal_op['CL'],
            'optimal_CD': optimal_op['CD'],
            'optimal_Cm': optimal_op['Cm'],
            'optimal_LD': optimal_op['LD'],
            'optimal_status': optimal_op['status'],
            'in_preferred_range': optimal_op['in_preferred_range'],
            'CL_max': CL_max,
            'alpha_at_CL_max': alpha_at_CL_max,
            'max_LD': max_LD,
            'alpha_at_max_LD': alpha_at_max_LD,
            'CL_required': req['CL_at_speed'],
            'CL_margin': CL_margin,
            'meets_CL_requirement': meets_CL_requirement,
            'stall_margin_ratio': stall_margin_ratio,
            'safe_stall_margin': safe_stall_margin,
            'total_score': total_score
        }
    
    # Function to analyze all flap configurations in the data folder
    def process_all_files(self):
        """Process all CSV files in data folder"""
        csv_files = list(self.data_folder.glob('*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {self.data_folder}")
            return
        
        print(f"Processing {len(csv_files)} files...")
        
        for csv_file in sorted(csv_files):
            filename = csv_file.name
            
            df = self.parse_xflr5_csv(csv_file)
            if df is None:
                continue
            
            flap_angle, phase = self.extract_file_info(filename)
            if phase is None:
                continue
            
            result = self.analyze_configuration(df, flap_angle, phase)
            if result:
                self.results.append(result)
    
    # Function to generate analysis summary and recommendations
    def generate_summary(self):
        """Generate analysis summary and recommendations"""
        if not self.results:
            print("No valid results generated")
            return None
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("FLAP CONFIGURATION ANALYSIS RESULTS")
        print("="*80)
        
        for phase in ['landing', 'takeoff']:
            phase_data = df[df['phase'] == phase].sort_values('flap_angle')
            
            if phase_data.empty:
                continue
            
            req = self.requirements[phase]
            
            print(f"\n{phase.upper()} PHASE")
            print(f"Required CL: {req['CL_at_speed']:.3f}, Preferred α: {req['preferred_alpha_min']}-{req['preferred_alpha_max']}°")
            print("-" * 60)
            
            for _, row in phase_data.iterrows():
                meets_icon = "✓" if row['meets_CL_requirement'] else "✗"
                pref_icon = "✓" if row['in_preferred_range'] else "~"
                
                print(f"Flap {row['flap_angle']:2.0f}° | α={row['optimal_alpha']:4.1f}° {pref_icon} | "
                      f"CL={row['optimal_CL']:.3f} {meets_icon} | L/D={row['optimal_LD']:4.1f} | "
                      f"Score={row['total_score']:5.1f}")
            
            # Best configuration
            best_config = phase_data.loc[phase_data['total_score'].idxmax()]
            print(f"\nRecommended: {best_config['flap_angle']:.0f}° flap deflection")
            print(f"Score: {best_config['total_score']:.1f}/100, CL: {best_config['optimal_CL']:.3f}, "
                  f"α: {best_config['optimal_alpha']:.1f}°, L/D: {best_config['optimal_LD']:.1f}")
        
        # Save detailed results
        output_file = 'flap_analysis_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
        
        return df
    
    # Function to create individual plot files
    def create_individual_plots(self, df):
        """Create individual plot files in separate folder"""
        plots_dir = Path('analysis_plots')
        plots_dir.mkdir(exist_ok=True)
        
        print("Generating individual plots...")
        
        for phase in ['landing', 'takeoff']:
            phase_data = df[df['phase'] == phase].sort_values('flap_angle')
            
            if phase_data.empty:
                continue
            
            req = self.requirements[phase]
            
            # Plot 1: Optimal Alpha vs Flap Deflection
            plt.figure(figsize=(10, 6))
            colors = ['green' if x else 'orange' for x in phase_data['in_preferred_range']]
            plt.scatter(phase_data['flap_angle'], phase_data['optimal_alpha'], c=colors, s=80)
            plt.plot(phase_data['flap_angle'], phase_data['optimal_alpha'], 'k--', alpha=0.5)
            plt.axhspan(req['preferred_alpha_min'], req['preferred_alpha_max'], alpha=0.2, color='green', label='Preferred α range')
            plt.xlabel('Flap Deflection (°)')
            plt.ylabel('Optimal Operating α (°)')
            plt.title(f'{phase.title()} - Operating Angle vs Flap Deflection')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / f'{phase}_operating_angle.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 2: CL vs Flap Deflection
            plt.figure(figsize=(10, 6))
            plt.plot(phase_data['flap_angle'], phase_data['optimal_CL'], 'o-', linewidth=2, markersize=6)
            plt.axhline(req['CL_at_speed'], color='r', linestyle='--', label=f'Required CL: {req["CL_at_speed"]:.3f}')
            plt.xlabel('Flap Deflection (°)')
            plt.ylabel('Lift Coefficient (CL)')
            plt.title(f'{phase.title()} - Lift Coefficient vs Flap Deflection')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / f'{phase}_lift_coefficient.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 3: L/D vs Flap Deflection
            plt.figure(figsize=(10, 6))
            plt.plot(phase_data['flap_angle'], phase_data['optimal_LD'], 'o-', linewidth=2, markersize=6, color='green')
            plt.xlabel('Flap Deflection (°)')
            plt.ylabel('Lift-to-Drag Ratio (L/D)')
            plt.title(f'{phase.title()} - Aerodynamic Efficiency vs Flap Deflection')
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / f'{phase}_efficiency.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 4: Total Score vs Flap Deflection
            plt.figure(figsize=(10, 6))
            colors = ['green' if x and y else 'orange' 
                     for x, y in zip(phase_data['meets_CL_requirement'], phase_data['in_preferred_range'])]
            plt.bar(phase_data['flap_angle'], phase_data['total_score'], color=colors, alpha=0.7)
            plt.xlabel('Flap Deflection (°)')
            plt.ylabel('Composite Score')
            plt.title(f'{phase.title()} - Overall Performance Score')
            plt.grid(True, alpha=0.3, axis='y')
            plt.savefig(plots_dir / f'{phase}_performance_score.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to: {plots_dir}/")
    
    # Function to run the entire analysis pipeline
    def run(self):
        """Execute complete analysis pipeline"""
        print("NACA 3213 Flap Configuration Analysis")
        print("=" * 50)
        
        self.process_all_files()
        
        if not self.results:
            print("Analysis failed - no valid data processed")
            return
        
        df = self.generate_summary()
        
        if df is not None:
            self.create_individual_plots(df)
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)


if __name__ == "__main__":
    analyzer = ImprovedFlapAnalyzer('Total_Data')
    analyzer.run()