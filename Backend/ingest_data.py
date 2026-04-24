import os
import psycopg2
import pypdf
import docx
from dotenv import load_dotenv
import logging
import re

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# Use the variable name from your .env
POSTGRES_DSN = os.getenv("POSTGRES") # Targeting 'mydb'
DATA_FOLDER = "data_source"

def init_db():
    """
    Connects to DB, DROPS the old table, and creates the NEW one.
    """
    try:
        if not POSTGRES_DSN:
            logger.error("❌ POSTGRES (mydb) is missing in .env file.")
            return None

        conn = psycopg2.connect(POSTGRES_DSN)
        cur = conn.cursor()

        # 1. Check connection
        dsn_params = conn.get_dsn_parameters()
        dbname = dsn_params.get('dbname')
        logger.info(f"✅ Connected to database: '{dbname}'")

        # 2. DROP old table to reset data (Apply 800-char Chunks)
        logger.info("♻️  Resetting table 'documents'...")
        cur.execute("DROP TABLE IF EXISTS documents;")
        
        # 3. CREATE table
        cur.execute("""
            CREATE TABLE documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                source_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        logger.info("✅ Table 'documents' created.")
        return conn
    except Exception as e:
        logger.error(f"❌ Database connection/creation failed: {e}")
        return None

def extract_text(file_path):
    """Reads content from .txt, .pdf, and .docx files."""
    ext = os.path.splitext(file_path)[1].lower()
    text_content = ""

    try:
        if ext == '.pdf':
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
        
        elif ext == '.docx':
            doc = docx.Document(file_path)
            text_content = "\n".join([para.text for para in doc.paragraphs])

        elif ext in ['.txt', '.md', '.csv', '.json']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()

        if not text_content.strip():
            logger.warning(f"⚠️  Skipped empty file: {os.path.basename(file_path)}")
            return None
            
        return text_content.strip()

    except Exception as e:
        logger.error(f"❌ Error reading {os.path.basename(file_path)}: {e}")
        return None

# Replace chunk_text function in DB.py
def chunk_text(text, chunk_size=512, overlap=100):
    """Semantic-aware chunking WITHOUT tiktoken dependency."""
    if not text: 
        return []
    
    # Clean text
    text = ' '.join(text.split())
    
    # Simple semantic chunking by sentences (more accurate than char splitting)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) < chunk_size:
            current_chunk += " " + sent
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sent
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Ensure minimum overlap
    if len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                prev_end = chunks[i-1][-overlap:]
                new_chunk = prev_end + chunk
                if len(new_chunk) > chunk_size:
                    new_chunk = new_chunk[-chunk_size:]
                overlapped.append(new_chunk)
        return overlapped
    
    return chunks


def ingest_files():
    conn = init_db()
    if not conn: return

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        logger.warning(f"📁 Created folder '{DATA_FOLDER}'. Please add files and run again.")
        return

    cur = conn.cursor()
    total_chunks = 0
    files_processed = 0
    
    logger.info(f"📂 Scanning folder: {os.path.abspath(DATA_FOLDER)}")

    for root, _, files in os.walk(DATA_FOLDER):
        for file in files:
            if file.lower().endswith(('.txt', '.md', '.csv', '.json', '.pdf', '.docx')):
                file_path = os.path.join(root, file)
                
                # 1. Extract full text
                full_text = extract_text(file_path)

                if full_text:
                    # 2. Split into 800-char chunks
                    text_chunks = chunk_text(full_text)
                    
                    # 3. Insert chunks
                    for chunk in text_chunks:
                        try:
                            cur.execute(
                                "INSERT INTO documents (content, source_file) VALUES (%s, %s)",
                                (chunk, file)
                            )
                            total_chunks += 1
                        except Exception as e:
                            logger.error(f"   ❌ Failed to insert chunk from {file}: {e}")
                            conn.rollback()
                        
                    files_processed += 1
                    logger.info(f"   📄 Processed: {file} -> {len(text_chunks)} chunks")
    
    conn.commit()
    
    # Final Stats
    cur.execute("SELECT COUNT(*) FROM documents;")
    count = cur.fetchone()[0]
    
    logger.info("-" * 40)
    logger.info(f"🏁 INGESTION COMPLETE")
    logger.info(f"   Files Processed: {files_processed}")
    logger.info(f"   Total Chunks DB: {count}")
    logger.info("-" * 40)
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    ingest_files()