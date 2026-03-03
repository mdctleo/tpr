# Can you show me how to parse my audit d logs
import pandas as pd

df = pd.read_csv("audit_log.csv")

start_time = df['timestamp'].min()

print("original df length: ", len(df))

# Get df that is within an hour of start time

regular_df = df[df['timestamp'] <= start_time + 3600 * 1e9]

print("regular df length: ", len(regular_df))

regular_df.to_csv("audit_log_regular.csv")

adversarial_df = df[df['timestamp'] > start_time + 3600 * 1e9]


print("adversarial df length: ", len(adversarial_df))

adversarial_df.to_csv("audit_log_adversarial.csv")