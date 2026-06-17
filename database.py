import pickle
import os
from datetime import datetime, date

from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, Date, ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()


class Student(Base):
    __tablename__ = "students"

    id         = Column(Integer, primary_key=True)
    name       = Column(String, nullable=False)
    grade      = Column(String, nullable=True)   # e.g. "Grade 10", "11"
    created_at = Column(DateTime, default=datetime.utcnow)

    encodings       = relationship("FaceEncoding",  back_populates="student", cascade="all, delete-orphan")
    attendance_logs = relationship("AttendanceLog", back_populates="student", cascade="all, delete-orphan")


class FaceEncoding(Base):
    __tablename__ = "face_encodings"

    id             = Column(Integer, primary_key=True)
    student_db_id  = Column(Integer, ForeignKey("students.id"), nullable=False)
    encoding_blob  = Column(LargeBinary, nullable=False)  # pickled numpy array
    created_at     = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="encodings")


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    id            = Column(Integer, primary_key=True)
    student_db_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    timestamp     = Column(DateTime, default=datetime.utcnow)
    date          = Column(Date, default=date.today)

    # Prevent duplicate check-ins for the same student on the same date
    __table_args__ = (UniqueConstraint("student_db_id", "date", name="uq_student_date"),)

    student = relationship("Student", back_populates="attendance_logs")


# --- DB setup ---
# Store outside iCloud-synced Documents folder to avoid disk I/O errors
DB_PATH = os.path.expanduser("~/attendance.db")
engine  = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


# --- Encoding helpers ---
def encoding_to_blob(encoding):
    """Serialize a numpy face encoding to bytes for DB storage."""
    return pickle.dumps(encoding)


def blob_to_encoding(blob):
    """Deserialize bytes from DB back to a numpy face encoding."""
    return pickle.loads(blob)


def load_all_encodings():
    """
    Load all face encodings from the DB.
    Returns (encodings_list, students_list) — parallel arrays.
    """
    session = Session()
    rows = session.query(FaceEncoding).all()
    encodings = [blob_to_encoding(r.encoding_blob) for r in rows]
    students  = [r.student for r in rows]
    session.close()
    return encodings, students


def save_student(name, grade, encoding):
    """Save student + face encoding to the database."""
    session = Session()
    try:
        existing = session.query(Student).filter_by(name=name, grade=grade).first()
        if existing:
            enc = FaceEncoding(student_db_id=existing.id, encoding_blob=encoding_to_blob(encoding))
            session.add(enc)
            session.commit()
            print(f"Added additional encoding for existing student: {existing.name}")
            return existing
        else:
            student = Student(name=name, grade=grade)
            session.add(student)
            session.flush()
            enc = FaceEncoding(student_db_id=student.id, encoding_blob=encoding_to_blob(encoding))
            session.add(enc)
            session.commit()
            print(f"Enrolled new student: {name} (Grade {grade})")
            return student
    except Exception as e:
        session.rollback()
        print(f"Error saving student: {e}")
        return None
    finally:
        session.close()


def log_attendance(student_db_id):
    """
    Record attendance for a student today.
    Silently skips if already logged today (UniqueConstraint).
    Returns True if logged, False if already present.
    """
    session = Session()
    try:
        log = AttendanceLog(student_db_id=student_db_id, date=date.today())
        session.add(log)
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False
    finally:
        session.close()
