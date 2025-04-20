import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Sample resumes and job description data (same as before)
data = {
    'resume_id': [1, 2, 3],
    'resume_text': [
        "Experienced software engineer with expertise in Java, Python, and cloud technologies. Worked on scalable backend systems and led DevOps initiatives across multiple agile teams.",
        "Creative graphic designer with 5+ years of experience in branding, UI/UX design, and Adobe Creative Suite. Successfully delivered projects for clients in tech, retail, and entertainment sectors.",
        """Flutter Developer with 6+ years experience building production-ready apps. Expert in animations, Firebase, BLoC, and REST APIs. Worked on health, fintech, Web3, AR/VR, and social apps. Awarded 'Most Innovative Employee 2023' at UNNICON. Owner of flutter-css and flutter_geo_hash packages. Skilled in React, Python, and backend services with Django and Flask. Looking for remote opportunities or relocation for MSc Data Science and Analytics in the UK starting September 2025."""
    ]
}

job_description = """
We are looking for a skilled Flutter Developer to join our dynamic team. The ideal candidate will have experience building cross-platform mobile applications using Flutter and Dart. You will work closely with UI/UX designers, backend developers, and product managers to deliver high-quality, responsive, and maintainable mobile apps.

**Responsibilities:**
- Design and develop mobile applications using Flutter
- Collaborate with cross-functional teams to define and deliver new features
- Ensure performance, quality, and responsiveness of applications
- Maintain code integrity and organization
- Integrate third-party APIs and work with backend systems

**Requirements:**
- 3+ years of experience with Flutter and Dart
- Strong understanding of mobile app architecture and design patterns (e.g., BLoC, Provider)
- Experience with Firebase, RESTful APIs, and state management
- Familiarity with Git and CI/CD tools
- Good communication and problem-solving skills
"""

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def encode_text(text):
    """Encode the input text using BERT tokenizer and model"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embedding from the [CLS] token as the sentence embedding
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode all resumes and the job description
documents = df['resume_text'].tolist()
documents.append(job_description)

# Convert documents to BERT embeddings
document_embeddings = [encode_text(doc) for doc in documents]

# Stack all embeddings into a 2D array (n_documents, 768)
embeddings_matrix = np.vstack([embedding.reshape(1, -1) for embedding in document_embeddings])

# Split into job description and resumes
job_embedding = embeddings_matrix[-1].reshape(1, -1)  # Shape (1, 768)
resume_embeddings = embeddings_matrix[:-1]            # Shape (n_resumes, 768)

# Compute similarity scores
similarity_scores = cosine_similarity(job_embedding, resume_embeddings).flatten()

# Add scores to DataFrame and display results
df['similarity_score'] = similarity_scores
print("Resume Similarity Scores:\n", df[['resume_id', 'similarity_score']])

# Calculate average and dynamic threshold
avg_score = similarity_scores.mean()
threshold = avg_score + 0.05
print(f"\nAverage Similarity Score: {avg_score}")
print(f"Using Dynamic Threshold: {threshold}")

# Identify matching resumes
matching_resumes = df[df['similarity_score'] >= threshold]
print("\nResumes matching the job requirements:\n", matching_resumes[['resume_id', 'similarity_score']])