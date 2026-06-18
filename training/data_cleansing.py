import pandas as pd
import numpy as np

data= {
"Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
"Age": [25, 30, np.nan, 35, 28],

}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

