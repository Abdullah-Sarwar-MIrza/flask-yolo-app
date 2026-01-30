import mysql.connector

def create_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='kingMirz@111',
        database='yolo_app'
    )

def insert_detection(conn, image_name, label, confidence, xmin, ymin, xmax, ymax):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO detections (image_name, label, confidence, xmin, ymin, xmax, ymax)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (image_name, label, confidence, xmin, ymin, xmax, ymax))
    conn.commit()
    cursor.close()