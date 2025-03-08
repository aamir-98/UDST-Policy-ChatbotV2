import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load Models
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # For intent classification

# Full UDST Policy Data (Detailed)
policies = {
    "Final Grade Policy": """
        The Final Grade Policy outlines assessment criteria, grading structure, and appeal processes.
        - **Passing Grade**: Minimum passing grade is **D (60%)**.
        - **Distinction**: **A (90-100%)** is considered excellent.
        - **Grade Appeal**: Students can request re-evaluation within **3 weeks** of results.
        - **Attendance Policy**: Exceeding **15% absences** results in **AF (Attendance Fail)**.
        - **Course Repetition**: Courses with **D+ or lower** can be repeated (max **2 times**).
    """,
    "Student Attendance Policy": """
        - **Mandatory Attendance**: Students must attend **85% of classes**.
        - **Excused Absences**: Only medical reasons or official university duties are valid.
        - **Excessive Absences**: More than **15% absenteeism** results in **AF grade**.
    """,
    "Graduation Policy": """
        - **Graduation Requirements**: Must complete required **credit hours & GPA minimum**.
        - **Graduation Application**: Must apply **before the deadline**.
        - **Diploma Conferral**: Diplomas awarded **4 times per year**.
        - **Withholding Diplomas**: Pending fees or unreturned items may delay graduation.
    """,
    "Student Conduct Policy": """
        - **Behavior Expectations**: Respectful conduct towards peers, faculty, and staff.
        - **Prohibited Conduct**: Harassment, violence, plagiarism, and academic fraud.
        - **Disciplinary Actions**: Warnings, suspensions, or expulsions for violations.
        - **Appeal Process**: Students can appeal disciplinary decisions.
    """,
    "Admissions Policy": """
        - **Eligibility**: Applicants must meet minimum academic requirements.
        - **Application Process**: Includes transcripts, entrance tests, and verification.
        - **Selection Criteria**: Based on merit and availability.
        - **Priority Admission**: Qatari students are given preference.
    """,
}

# Convert policy texts into numerical embeddings
policy_names = list(policies.keys())
policy_texts = list(policies.values())
policy_embeddings = np.array(embedder.encode(policy_texts, convert_to_tensor=True))

# Create FAISS Index for Fast Retrieval
embedding_dim = policy_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(policy_embeddings)

# Function to classify question intent (find closest matching policy)
def classify_policy(question):
    question_embedding = embedder.encode([question], convert_to_tensor=True)
    _, best_match = index.search(np.array(question_embedding), 1)  # Find closest policy
    return policy_names[best_match[0][0]]  # Return policy title

# Function to retrieve accurate answer from the selected policy
def get_answer(question, policy_text):
    try:
        response = qa_pipeline(question=question, context=policy_text)
        return response['answer']
    except:
        return "I couldn't find a precise answer in the policy data."

# Streamlit UI
st.title("UDST Policy Q&A Chatbot")

# User question input
user_query = st.text_input("Ask a question about UDST policies:")

# Submit button
if st.button("Get Answer"):
    if user_query.strip():
        selected_policy = classify_policy(user_query)
        answer = get_answer(user_query, policies[selected_policy])

        # Display response
        st.write(f"**Identified Policy:** {selected_policy}")
        st.write(f"### Answer: {answer}")
    else:
        st.warning("Please enter a question before submitting.")
