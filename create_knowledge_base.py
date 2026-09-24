import os
import pickle

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# ==========================================
# 1. FILE PATHS
# ==========================================

DATA_FILE = "data/college_data.txt"

VECTOR_FOLDER = "vector_store"

INDEX_FILE = os.path.join(
    VECTOR_FOLDER,
    "college_index.faiss"
)

CHUNKS_FILE = os.path.join(
    VECTOR_FOLDER,
    "college_chunks.pkl"
)


# ==========================================
# 2. CHECK DATA FILE
# ==========================================

if not os.path.exists(DATA_FILE):

    print("College data file not found!")

    exit()


# ==========================================
# 3. READ COLLEGE DATA
# ==========================================

with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    college_text = file.read()


if not college_text.strip():

    print("College data file is empty!")

    exit()


print("College data loaded successfully!")

print(
    "Total characters:",
    len(college_text)
)


# ==========================================
# 4. SPLIT TEXT INTO CHUNKS
# ==========================================

chunk_size = 500

overlap = 50

chunks = []

start = 0

while start < len(college_text):

    end = start + chunk_size

    chunk = college_text[start:end]

    if chunk.strip():

        chunks.append(chunk.strip())

    start += chunk_size - overlap


print(
    "Total chunks created:",
    len(chunks)
)


# ==========================================
# 5. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ==========================================
# 6. GENERATE EMBEDDINGS
# ==========================================

print("Generating embeddings...")

embeddings = model.encode(
    chunks,
    convert_to_numpy=True,
    show_progress_bar=True
)


embeddings = embeddings.astype(
    "float32"
)


print(
    "Embedding shape:",
    embeddings.shape
)


# ==========================================
# 7. CREATE FAISS INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(embeddings)


print(
    "Total vectors stored:",
    index.ntotal
)


# ==========================================
# 8. CREATE VECTOR FOLDER
# ==========================================

os.makedirs(
    VECTOR_FOLDER,
    exist_ok=True
)


# ==========================================
# 9. SAVE FAISS INDEX
# ==========================================

faiss.write_index(
    index,
    INDEX_FILE
)


# ==========================================
# 10. SAVE TEXT CHUNKS
# ==========================================

with open(
    CHUNKS_FILE,
    "wb"
) as file:

    pickle.dump(
        chunks,
        file
    )


print("\nKnowledge base created successfully!")

print(
    "FAISS index saved at:",
    INDEX_FILE
)

print(
    "Chunks saved at:",
    CHUNKS_FILE
)