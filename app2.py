import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.read_csv('dataset1.csv')
df = df[:1000]
for i, data in enumerate(df['satisfaction']):
    if data == "satisfied":
        df.at[i, 'satisfaction'] = 1
    else:
        df.at[i, 'satisfaction'] = 0
for i, data in enumerate(df['Gender']):
    if data == "Male":
        df.at[i, 'Gender'] = 1
    else:
        df.at[i, 'Gender'] = 0
for i, data in enumerate(df['Customer Type']):
    if data == "Loyal Customer":
        df.at[i, 'Customer Type'] = 1
    else:
        df.at[i, 'Customer Type'] = 0
for i, data in enumerate(df['Type of Travel']):
    if data == "Personal Travel":
        df.at[i, 'Type of Travel'] = 1
    else:
        df.at[i, 'Type of Travel'] = 0
for i, data in enumerate(df['Class']):
    if data == "Eco":
        df.at[i, 'Class'] = 1
    else:
        df.at[i, 'Class'] = 0

df2 = pd.read_csv('dataset2.csv')
print(df2.head())

for i, data in enumerate(df2['Sex']):
    if data == "M":
        df2.at[i, 'Sex'] = 1
    else:
        df2.at[i, 'Sex'] = 0
for i, data in enumerate(df2['BP']):
    if data == "HIGH":
        df2.at[i, 'BP'] = 2
    elif data == "NORMAL":
        df2.at[i, 'BP'] = 1
    else:
        df2.at[i, 'BP'] = 0
for i, data in enumerate(df2['Cholesterol']):
    if data == "HIGH":
        df2.at[i, 'Cholesterol'] = 1
    else:
        df2.at[i, 'Cholesterol'] = 0
for i, data in enumerate(df2['Drug']):
    if data == "drugA":
        df2.at[i, 'Drug'] = 4
    elif data == "drugB":
        df2.at[i, 'Drug'] = 3
    elif data == "drugC":
        df2.at[i, 'Drug'] = 2
    elif data == "drugX":
        df2.at[i, 'Drug'] = 1
    elif data == "drugY":
        df2.at[i, 'Drug'] = 0


print(df.head())
print(df2.head())