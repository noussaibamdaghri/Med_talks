import pandas as pd
import numpy as np
from typing import Dict, Any

class ExplorationMetrics:
    """AYA NAIM: Analyze and explore medical data"""
    
    @staticmethod
    def analyze_flashcards(df) -> Dict[str, Any]:
        """Analyze medical flashcards dataset"""
        metrics = {
            'total': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Text length analysis
        if 'instruction' in df.columns:
            df['instruction_length'] = df['instruction'].str.len()
            metrics['instruction_stats'] = {
                'avg_length': df['instruction_length'].mean(),
                'max_length': df['instruction_length'].max(),
                'min_length': df['instruction_length'].min()
            }
        
        if 'output' in df.columns:
            df['output_length'] = df['output'].str.len()
            metrics['output_stats'] = {
                'avg_length': df['output_length'].mean(),
                'max_length': df['output_length'].max(),
                'min_length': df['output_length'].min()
            }
        
        # Word count analysis
        if 'instruction' in df.columns:
            df['instruction_words'] = df['instruction'].str.split().str.len()
            metrics['word_stats'] = {
                'avg_words_per_instruction': df['instruction_words'].mean(),
                'total_words': df['instruction_words'].sum()
            }
        
        return metrics
    
    @staticmethod
    def create_summary_report(df, filename='data/medical_analysis_report.txt'):
        """Create a text report of data analysis"""
        import os
        os.makedirs('data', exist_ok=True)
        
        metrics = ExplorationMetrics.analyze_flashcards(df)
        
        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MEDICAL FLASHCARDS ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Dataset Overview:\n")
            f.write(f"- Total records: {metrics['total']}\n")
            f.write(f"- Columns: {', '.join(metrics['columns'])}\n\n")
            
            f.write("Text Length Analysis:\n")
            if 'instruction_stats' in metrics:
                stats = metrics['instruction_stats']
                f.write(f"- Instructions: Avg {stats['avg_length']:.1f} chars, ")
                f.write(f"Max {stats['max_length']}, Min {stats['min_length']}\n")
            
            if 'word_stats' in metrics:
                stats = metrics['word_stats']
                f.write(f"- Avg words per instruction: {stats['avg_words_per_instruction']:.1f}\n")
                f.write(f"- Total words: {stats['total_words']}\n")
        
        print(f"AYA: Analysis report saved to {filename}")
        return metrics