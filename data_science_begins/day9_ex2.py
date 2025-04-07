import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv") 

# # print("First 5 rows: \n", df.head())
# # print("Last 5 rows: \n", df.tail())

# # print(df.describe())

# selected_columns=df[["species","sepal_length"]]

# # print("Selected Columns: \n",selected_columns)

# filtered_rows = df[(df["sepal_length"] > 5.0) & (df["species"]=="setosa")]
# # print(filtered_rows)

df= df.rename(columns={"species":"flower_type"}) #rename column
df["column_name"] = df['column_name'].astype("float") #change data type of column
df["column_name"] = pd.to_datetime(df["column_name"]) #change data type of column to datetime
df["new_column"] = df["existing_column"] * 2 #modify existing columns


df = df.dropna(axis=1) #0 for rows, columns 1 => where drop columns with missing data

df['column_name'] = df['column_name'].fillna(0)

df.fillna(method="ffill")
df.fillna(method="bfill")

