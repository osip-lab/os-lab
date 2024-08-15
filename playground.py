import os
from local_config import DATA_PATH
import pandas as pd

specific_file_path = os.path.join(DATA_PATH, r"EMNV\Ring down\270624\ring_down_curve_parameters.xlsx")

pd.read_excel(specific_file_path)






