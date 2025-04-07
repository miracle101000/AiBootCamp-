grouped =  df.groupby("column_name")

for name, group in grouped:
    print(name)
    print(group)
    
grouped.mean()
grouped.sum()

df.groupby("column_name")['numeric_column'].mean()
df.groupby("column_name").agg({"numeric_column": ["mean","max","min"]})
pivot =df.pivot_table(index="column_name", columns="column_name", values="numeric_column", aggfunc="mean")

def range_func(x):
    return x.max() - x.min()


df.groupby("column_name")['numeric_column'].agg(range_func)
df.groupby("column_name")['numeric_column'].mean()
df.groupby("column_name")['numeric_column'].max()
df.groupby("column_name")['numeric_column'].min()

df.groupby("column_name")['numeric_column'].agg({"numeric_column": ["mean","max","min"]})