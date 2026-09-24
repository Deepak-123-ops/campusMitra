import streamlit as st
import requests
import pickle
import faiss
import numpy as np
import os

from sentence_transformers import SentenceTransformer


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CampusMitra",
    page_icon="🎓",
    layout="centered"
)


# ==========================================
# CHAT HISTORY INITIALIZATION
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==========================================
# TITLE
# ==========================================

st.title("🎓 CampusMitra")

st.subheader(
    "AI-Powered College Query Assistant"
)

st.write(
    "Ask questions about your college, courses, admission, "
    "facilities, and other information."
)


# ==========================================
# CLEAR CHAT BUTTON
# ==========================================

if st.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# ==========================================
# LOAD KNOWLEDGE BASE
# ==========================================

@st.cache_resource
def load_knowledge_base(index_time, chunks_time):

    index = faiss.read_index(
        "vector_store/college_index.faiss"
    )

    with open(
        "vector_store/college_chunks.pkl",
        "rb"
    ) as file:

        chunks = pickle.load(file)

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return index, chunks, model


# Get latest file modification times

index_time = os.path.getmtime(
    "vector_store/college_index.faiss"
)

chunks_time = os.path.getmtime(
    "vector_store/college_chunks.pkl"
)


index, chunks, model = load_knowledge_base(
    index_time,
    chunks_time
)


# ==========================================
# SEARCH RELEVANT COLLEGE INFORMATION
# ==========================================

def search_college_data(question):

    question_lower = question.lower()

    relevant_chunks = []


    # --------------------------------------
    # 1. KEYWORD-BASED SEARCH
    # --------------------------------------

    if (
        "department" in question_lower
        or "departments" in question_lower
    ):

        keywords = [

            "departments",

            "civil engineering",

            "computer science",

            "data science",

            "electrical",

            "electronics",

            "mechanical engineering",

            "basic sciences",

            "humanities",

            "mca",

            "mba"

        ]

        for chunk in chunks:

            chunk_lower = chunk.lower()

            if any(
                keyword in chunk_lower
                for keyword in keywords
            ):

                if chunk not in relevant_chunks:

                    relevant_chunks.append(chunk)


    # --------------------------------------
    # 2. FAISS SEMANTIC SEARCH
    # --------------------------------------

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )


    distances, indices = index.search(
    question_embedding,
    4
)


    for i in indices[0]:

        if 0 <= i < len(chunks):

            if chunks[i] not in relevant_chunks:

                relevant_chunks.append(
                    chunks[i]
                )


    # --------------------------------------
    # 3. LIMIT CONTEXT SIZE
    # --------------------------------------

    relevant_chunks = relevant_chunks[:6]


    context = "\n\n".join(
        relevant_chunks
    )


    return context


def ask_ollama(question, context):
    if not context.strip():
        return (
            "Sorry, I could not find this information "
            "in the college knowledge base."
        )

    prompt = f"""
You are CampusMitra, an AI-powered college assistant
for Einstein Academy of Technology and Management (EATM).

Your job is to answer student questions using ONLY
the provided college information.

IMPORTANT RULES:

1. Use only the provided context.
2. Do not invent or assume information.
3. Give short, clear, and organized answers.
4. Avoid repeating the same information.
5. Distinguish academic programs from departments.
6. Do not treat department names as separate courses
   unless the context clearly says they are courses.
7. If the answer is not available in the context,
   say exactly:

Sorry, I could not find this information
in the college knowledge base.

ACADEMIC PROGRAMS:

When asked about courses or programs, list only
the programs explicitly mentioned in the context.

DEPARTMENTS:

When asked about departments, list only
the departments explicitly mentioned in the context.

COLLEGE INFORMATION:
{context}

STUDENT QUESTION:
{question}

Provide an accurate and concise answer.

ANSWER:
"""

    try:
        from groq import Groq

        client = Groq(
            api_key=st.secrets["GROQ_API_KEY"]
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=150
        )

        return response.choices[0].message.content

    except Exception as error:
        return f"Error connecting to AI service: {error}"

# ==========================================
# DISPLAY PREVIOUS CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ==========================================
# CHAT INPUT
# ==========================================

question = st.chat_input(
    "Ask your college-related question..."
)


# ==========================================
# PROCESS NEW QUESTION
# ==========================================

if question:

    # --------------------------------------
    # SAVE USER QUESTION
    # --------------------------------------

    st.session_state.messages.append(

        {
            "role": "user",
            "content": question
        }

    )


    with st.chat_message("user"):

        st.write(question)


    # --------------------------------------
    # GENERATE ASSISTANT RESPONSE
    # --------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching college information..."
        ):

            try:

                context = search_college_data(
                    question
                )


                answer = ask_ollama(
                    question,
                    context
                )


                st.write(answer)


                # Save assistant answer

                st.session_state.messages.append(

                    {
                        "role": "assistant",
                        "content": answer
                    }

                )


            except Exception as error:

                error_message = (
                    f"Error: {error}"
                )


                st.error(
                    error_message
                )


                # Save error message

                st.session_state.messages.append(

                    {
                        "role": "assistant",
                        "content": error_message
                    }

                )