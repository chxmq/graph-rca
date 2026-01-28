# Fix for ChromaDB on systems with old SQLite
import sys

# Try to use pysqlite3 if available (newer SQLite version)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
