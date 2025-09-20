import os
import importlib.util
import pandas as pd
from pathlib import Path
import numpy as np

def load_feature_file(file_path):
    """Load a Python file containing hand landmark features."""
    try:
        spec = importlib.util.spec_from_file_location("feature_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.vetores_palavras
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def create_csv_from_features(feature_file_path, output_dir):
    """Create CSV files from a feature Python file."""
    try:
        # Load the features
        vetores_palavras = load_feature_file(feature_file_path)
        if vetores_palavras is None:
            return
        
        # Create column names for the 63 features (21 landmarks * 3 coordinates)
        column_names = []
        for i in range(21):  # 21 landmarks
            column_names.extend([
                f'landmark_{i}_x',
                f'landmark_{i}_y',
                f'landmark_{i}_z'
            ])

        # Process each word
        for palavra, repeticoes in vetores_palavras.items():
            # Create output filename for this word
            output_filename = output_dir / f"{Path(feature_file_path).stem}_{palavra}.csv"
            
            # Skip if CSV already exists
            if output_filename.exists():
                print(f"Skipping {palavra} - CSV already exists at {output_filename}")
                continue
            
            # Flatten the data for this word
            all_frames = []
            for rep in repeticoes:  # 10 repetitions
                for frame in rep:  # frames in this repetition
                    all_frames.append(frame)
            
            # Create DataFrame
            df = pd.DataFrame(all_frames, columns=column_names)
            
            # Add repetition, word and frame number columns
            df['repetition'] = np.repeat(range(10), [len(rep) for rep in repeticoes])
            df['frame'] = np.concatenate([range(len(rep)) for rep in repeticoes])
            df['word'] = palavra
            
            # Save to CSV
            df.to_csv(output_filename, index=False)
            print(f"Created CSV: {output_filename}")
        
    except Exception as e:
        print(f"Error processing {feature_file_path}: {str(e)}")

def main():
    # Base directory for feature files
    features_dir = Path("/mnt/d/dados_surdos")
    
    # Create output directory for CSVs
    output_dir = Path("/mnt/d/dados_surdos/CSVs")
    output_dir.mkdir(exist_ok=True)
    
    # Process all Python files in the directory and its subdirectories
    for feature_file in features_dir.rglob("*.py"):
        try:
            # Get the relative path from the base directory
            rel_path = feature_file.relative_to(features_dir)
            
            # Create the corresponding output directory structure
            output_subdir = output_dir / rel_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Create CSV file
            create_csv_from_features(feature_file, output_subdir)
            
        except Exception as e:
            print(f"Error processing {feature_file}: {str(e)}")

if __name__ == "__main__":
    main()
