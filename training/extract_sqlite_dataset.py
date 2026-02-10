import sqlite3

db_path = "sign-language-recognition.sqlite"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in database:")
for t in tables:
    print(t[0])

conn.close()
