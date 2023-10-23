import pandas as pd
import random

# Generate dummy dataset
data = {
    'req1': ['This is a sample text' + str(i) for i in range(100)],
    'req2': ['Another sample text here' + str(i) for i in range(100)],
    'req1Product': [random.choice(['AWS', 'Azure', 'Google Cloud', 'IBM Cloud']) for _ in range(100)],
    'req2Product': [random.choice(['AWS', 'Azure', 'Google Cloud', 'IBM Cloud']) for _ in range(100)],
    'Label': [random.choice([0, 1]) for _ in range(100)]
}

df_train = pd.DataFrame(data)

# Assuming training size and selected columns are defined somewhere
training_size = 0.8
selected_columns = ['req1', 'req2', 'req1Product', 'req2Product', 'Label']
csv_path = "dummy_dataset.csv"
df_train.to_csv(csv_path, index=False)

csv_path
