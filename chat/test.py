import pandas as pd

f_name = "upwork_list.xlsx"
db = pd.read_excel(f_name)

for col_name, data in db.iterrows():
    print("Num str:",col_name, "\ndata:",data["JOB TITLE"])

print(list(db.columns.values))
print(db["JOB TITLE"][:5])
