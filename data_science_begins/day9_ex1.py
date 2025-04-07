import pandas as pd

#Converts list into dictionary
s=pd.Series([10,20,30], index= ["a","b","c"])
# print(s)

data={"Name":["Alice","Bob"],"Age":[25,30]}
df = pd.DataFrame(data)
# print(df)

# print(df.head())
# print(df.tail(3)) #Last three rows

# print(df.info())
# print(df.describe())


#Selecting columns
# print(df[["Name","Age"]])

#search row where condition is through
# print(df[df["Age"] > 25])

# print(df.iloc[0])
print(df.iloc[:,1])