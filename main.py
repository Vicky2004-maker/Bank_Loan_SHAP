import pandas as pd

# %%

prev_data = pd.read_csv(r"S:\Dataset\Bank Loan\previous_application.csv")
data = pd.read_csv(r"S:\Dataset\Bank Loan\application_data.csv")

# %%

print('=' * 50)
print(*data.columns, sep='\n')
print('=' * 50)
