import pandas as pd
import numpy as np

# combined = pd.concat([df1, df2], axis=0) #concatenate two dataframes
# combined = pd.concat([df1, df2], axis=1) #concatenate two dataframes

# mergeq = pd.merge(df1, df2, on="column_name") #merge two dataframes
# merged = pd.merge(df1, df2, on="column_name", how="left") #merge two dataframes
# merged = pd.merge(df1, df2, on="column_name", how="right") #merge two dataframes
# merged = pd.merge(df1, df2, on="column_name", how="outer") #merge two dataframes
# merged = pd.merge(df1, df2, on="column_name", how="inner") #merge two dataframes

#Create a sample dataset
data = {
    "Name": ["Alice", "Bob", np.nan, "David"],
    "Age": [25, np.nan, 30, 35],
    "Score": [85, 90, np.nan, 88]
}

df =  pd.DataFrame(data)
print("Original Data: \n", df)

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Score"] = df["Score"].interpolate()

print("Dataset after filling missing values: \n", df)

df = df.rename(columns={"Name":"Student_Name","Score": "Exam:Score"})
print("Dataset after renaming columns: \n", df)






