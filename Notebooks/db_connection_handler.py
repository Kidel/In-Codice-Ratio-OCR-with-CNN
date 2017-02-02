import psycopg2

default_params = {
    'database': 'icr',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': 5432
}

def get_connection(database, user, password, host, port):
    new_params = {
      'database': database,
      'user': user,
      'password': password,
      'host': host,
      'port': port
    }
    connection = psycopg2.connect(**new_params)
    return connection

def get_default_connection():
    connection = psycopg2.connect(**default_params)
    return connection

def get_deafult_connection_cursor():
    connection = psycopg2.connect(**default_params)
    cur = connection.cursor()
    return cur

def close_connection(connection):
    connection.close()

def execute_query(query, params=None):
    records = []
    connection = get_default_connection()
    cursor = connection.cursor()
    cursor.execute(query, params)

    for record in cursor:
        records.append(record)

    connection.close()

    return records

def execute_many_query(query, params=None):
    records = []
    connection = get_default_connection()
    cursor = connection.cursor()
    cursor.executemany(query, params)

    for record in cursor:
        records.append(record)

    connection.close()

    return records
