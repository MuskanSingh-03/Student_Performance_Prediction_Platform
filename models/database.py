"""
Database Module for Student Performance Prediction Platform
Provides persistent SQLite storage for user authentication, registration,
and session management.
"""

import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "platform.db")

def get_db_connection():
    """Returns a connection to the SQLite database with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema and seeds initial dummy users (Amit Yadav, Muskan Gola, etc.)."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            department TEXT DEFAULT 'Computer Science & Engineering',
            student_id TEXT DEFAULT '',
            avatar TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create Interventions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            student_name TEXT NOT NULL,
            mentor_name TEXT NOT NULL,
            action_type TEXT NOT NULL,
            notes TEXT,
            status TEXT DEFAULT 'Completed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create Student Counseling Requests table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS counseling_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            student_email TEXT NOT NULL,
            student_id TEXT NOT NULL,
            mentor_name TEXT DEFAULT 'Amit Yadav',
            topic TEXT NOT NULL,
            message TEXT,
            status TEXT DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    
    # Check if dummy users exist; if not, seed them
    cursor.execute("SELECT COUNT(*) as cnt FROM users")
    count = cursor.fetchone()["cnt"]
    
    if count == 0:
        print("[Database] Seeding initial users (Amit Yadav, Muskan Gola)...")
        initial_users = [
            {
                "name": "Amit Yadav",
                "email": "amit@niet.co.in",
                "username": "amit",
                "password": "password123",
                "role": "Faculty Mentor",
                "department": "Computer Science & Engineering",
                "student_id": "",
                "avatar": "AY"
            },
            {
                "name": "Muskan Gola",
                "email": "muskan@niet.co.in",
                "username": "muskan",
                "password": "password123",
                "role": "Student",
                "department": "BTech CSE-D",
                "student_id": "2501330100232",
                "avatar": "MG"
            },
            {
                "name": "Dean Academics",
                "email": "admin@niet.co.in",
                "username": "admin",
                "password": "admin123",
                "role": "Administrator",
                "department": "NIET Administration",
                "student_id": "",
                "avatar": "DA"
            }
        ]
        
        for u in initial_users:
            p_hash = generate_password_hash(u["password"])
            cursor.execute("""
                INSERT INTO users (name, email, username, password_hash, role, department, student_id, avatar)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (u["name"], u["email"], u["username"], p_hash, u["role"], u["department"], u["student_id"], u["avatar"]))
            
        conn.commit()
        print("[Database] Initial users successfully seeded!")
        
    conn.close()

def create_user(name, email, password, role, department="Computer Science & Engineering", student_id=""):
    """Registers a new user in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Generate initials avatar
    parts = name.strip().split()
    if len(parts) >= 2:
        avatar = (parts[0][0] + parts[-1][0]).upper()
    elif len(parts) == 1 and len(parts[0]) > 0:
        avatar = parts[0][:2].upper()
    else:
        avatar = "U"
        
    username = email.split("@")[0].lower()
    p_hash = generate_password_hash(password)
    
    try:
        cursor.execute("""
            INSERT INTO users (name, email, username, password_hash, role, department, student_id, avatar)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (name.strip(), email.strip().lower(), username, p_hash, role, department, student_id, avatar))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return {
            "id": user_id,
            "name": name.strip(),
            "email": email.strip().lower(),
            "username": username,
            "role": role,
            "department": department,
            "student_id": student_id,
            "avatar": avatar
        }, None
    except sqlite3.IntegrityError:
        conn.close()
        return None, "An account with this email or username already exists."
    except Exception as e:
        conn.close()
        return None, str(e)

def authenticate_user(login_identifier, password):
    """
    Authenticates a user by email OR username against password.
    Returns user dict on success, None on failure.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    identifier = login_identifier.strip().lower()
    cursor.execute("""
        SELECT * FROM users WHERE email = ? OR username = ?
    """, (identifier, identifier))
    
    row = cursor.fetchone()
    conn.close()
    
    if row and check_password_hash(row["password_hash"], password):
        return {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "username": row["username"],
            "role": row["role"],
            "department": row["department"],
            "student_id": row["student_id"],
            "avatar": row["avatar"]
        }
    return None

def get_user_by_id(user_id):
    """Retrieves user by primary key ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "username": row["username"],
            "role": row["role"],
            "department": row["department"],
            "student_id": row["student_id"],
            "avatar": row["avatar"]
        }
    return None

def log_intervention(student_id, student_name, mentor_name, action_type, notes=""):
    """Records a faculty mentor intervention action in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO interventions (student_id, student_name, mentor_name, action_type, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (student_id, student_name, mentor_name, action_type, notes))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id

def get_interventions(student_id=None, limit=20):
    """Retrieves logged mentor interventions, optionally filtered by student_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if student_id:
        cursor.execute("""
            SELECT * FROM interventions WHERE student_id = ? ORDER BY created_at DESC LIMIT ?
        """, (student_id, limit))
    else:
        cursor.execute("""
            SELECT * FROM interventions ORDER BY created_at DESC LIMIT ?
        """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def submit_counseling_request(student_name, student_email, student_id, mentor_name, topic, message=""):
    """Saves a student consultation request for faculty mentor review."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO counseling_requests (student_name, student_email, student_id, mentor_name, topic, message)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (student_name, student_email, student_id, mentor_name, topic, message))
    conn.commit()
    req_id = cursor.lastrowid
    conn.close()
    return req_id

def get_counseling_requests(mentor_name=None):
    """Retrieves pending and past student counseling requests."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if mentor_name:
        cursor.execute("""
            SELECT * FROM counseling_requests WHERE mentor_name LIKE ? ORDER BY created_at DESC
        """, (f"%{mentor_name}%",))
    else:
        cursor.execute("""
            SELECT * FROM counseling_requests ORDER BY created_at DESC
        """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_counseling_status(request_id, status="Resolved"):
    """Updates status of a student counseling request."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE counseling_requests SET status = ? WHERE id = ?
    """, (status, request_id))
    conn.commit()
    conn.close()

# Auto-initialize database on import
init_db()
