import pandas as pd

splitStr = "\n" + "=" * 100 + "\n"

print(splitStr)
demo_df = pd.DataFrame({'Interger Feature': [0, 1, 2, 1],
						'Categorical Feature': ['socks', 'fox', 'socks', 'box']})

print(demo_df)

print(splitStr)
print(pd.get_dummies(demo_df))

demo_df['Interger Feature'] = demo_df['Interger Feature'].astype(str)
print(splitStr)
print(pd.get_dummies(demo_df))
