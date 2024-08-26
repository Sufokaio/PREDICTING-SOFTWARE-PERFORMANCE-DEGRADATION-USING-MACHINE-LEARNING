import mysql.connector
import os
from tqdm import tqdm

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="rootroot",
    database="code4bench"
)

cursor = conn.cursor()

query = """
    SELECT id, sourceCode, problems_id, author, time
    FROM source
    WHERE languages_id = 7 AND verdicts_id = 1
"""
cursor.execute(query)
java_programs = cursor.fetchall()

directory = "code4bench"
if not os.path.exists(directory):
    os.makedirs(directory)

pairs = {}

for program in java_programs:
    id, sourceCode, problems_id, author, time = program
    key = (problems_id, author)
    if key in pairs:
        pairs[key].append((id, time, problems_id, author, sourceCode))
    else:
        pairs[key] = [(id, time, problems_id, author, sourceCode)]


output_file = "pairs.txt"
with open(output_file, "w") as file:


    file.write(str(len(pairs)) + "\n")
    c = 0
    for key, value in tqdm(pairs.items()):
        for i in range(len(value)):
            for j in range(i + 1, len(value)):
                c += 1
                id1, time1, problems_id1, author1, sourceCode1 = value[i]

                filename1 = f"{directory}/{id1}.java"
                if not os.path.exists(filename1):
                    with open(filename1, "w") as java_file:
                        java_file.write(sourceCode1)

                id2, time2, problems_id2, author2, sourceCode2 = value[j]

                filename2 = f"{directory}/{id2}.java"
                if not os.path.exists(filename2):
                    with open(filename2, "w") as java_file:
                        java_file.write(sourceCode2)

                if time1 < time2:
                    file.write(f"{id1},{id2},1,0\n")
                else:
                    file.write(f"{id1},{id2},0,1\n")

print(c)
cursor.close()
conn.close()