import yaml
import sqlite3


class Catalog:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS catalog (
                id INTEGER PRIMARY KEY,
                name TEXT,
                description TEXT,
                price REAL
            )
        """
        )
        self.conn.commit()

    def add(self, name, description, price):
        self.cursor.execute(
            """
            INSERT INTO catalog (name, description, price)
            VALUES (?, ?, ?)
        """,
            (name, description, price),
        )
        self.conn.commit()

    def get(self, id):
        self.cursor.execute(
            """
            SELECT * FROM catalog WHERE id = ?
        """,
            (id,),
        )
        return self.cursor.fetchone()

    def delete(self, id):
        self.cursor.execute(
            """
            DELETE FROM catalog WHERE id = ?
        """,
            (id,),
        )
        self.conn.commit()

    def list(self):
        self.cursor.execute(
            """
            SELECT * FROM catalog
        """
        )
        return self.cursor.fetchall()

    def __del__(self):
        self.conn.close()
