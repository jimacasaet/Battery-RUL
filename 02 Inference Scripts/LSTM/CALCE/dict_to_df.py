import pandas as pd
import pickle

with open('CS2_35.pkl', 'rb') as f:
    CS2_35 = pickle.load(f)
with open('CS2_36.pkl', 'rb') as f:
    CS2_36 = pickle.load(f)
with open('CS2_37.pkl', 'rb') as f:
    CS2_37 = pickle.load(f)
with open('CS2_38.pkl', 'rb') as f:
    CS2_38 = pickle.load(f)

MSP_results = pd.DataFrame([CS2_35,
                            CS2_36,
                            CS2_37,
                            CS2_38
                            ])

# Saving results
MSP_results.to_pickle("MSP_LSTM_CALCE.pkl")
print(MSP_results)