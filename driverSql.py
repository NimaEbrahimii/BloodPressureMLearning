import pyodbc

# Connection string
conn_str = r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=C:\path\to\your\database.accdb;'

# Attempt to establish connection
try:
    conn = pyodbc.connect(conn_str)
    print("Connection successful!")
    conn.close()
except Exception as e:
    print("Connection failed:", e)


