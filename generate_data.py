# generate_data.py

import pandas as pd
import numpy as np

def generate_synthetic_well_log_data(file_path="/content/well_log_sample.csv", num_samples=20000):
    np.random.seed(42)
    depths = np.arange(1000, 1000 + num_samples)

    GR = np.random.normal(80, 5, num_samples)
    RHOB = np.random.normal(2.5, 0.05, num_samples)
    NPHI = np.random.normal(0.4, 0.03, num_samples)
    DT = np.random.normal(115, 3, num_samples)
    ILD = np.random.normal(50, 2, num_samples)
    facies_classes = ['Sandstone', 'Shale', 'Limestone']
    Facies = np.random.choice(facies_classes, num_samples)

    df = pd.DataFrame({
        'Depth': depths,
        'GR': GR,
        'RHOB': RHOB,
        'NPHI': NPHI,
        'DT': DT,
        'ILD': ILD,
        'Facies': Facies
    })

    df.to_csv(file_path, index=False)
    print(f"✅ Synthetic well log data generated at: {file_path}")

if __name__ == "__main__":
    generate_synthetic_well_log_data()
