import time
import asyncio
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from main import RAGChatbot, FaissIndex, model, client
from DB import DocumentDB


TEST_DATASET = [
    {
        "question": "What does the fox do?",
        "ground_truth": "The quick brown fox jumps over the lazy dog."
    },
    {
        "question": "What is transforming industries?",
        "ground_truth": "Artificial Intelligence and machine learning are transforming industries."
    },
    {
        "question": "What is RAG?",
        "ground_truth": "Retrieval Augmented Generation combines retrieval systems with language models."
    }
]

#METRIC CALCULATORS

def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """Calculates semantic similarity between two texts using your existing model."""
    
    emb1 = model.encode([text1], normalize_embeddings=True)
    emb2 = model.encode([text2], normalize_embeddings=True)
    return cosine_similarity(emb1, emb2)[0][0]

async def llm_grade(prompt: str) -> float:
    """Asks GPT-4 (or 3.5) to grade an aspect from 0.0 to 1.0"""
    try:
        # FIX: Added 'await' because client is now AsyncOpenAI
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo", # Use 3.5 for faster eval
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        
        # Extract number from response
        import re
        match = re.search(r"(\d+(\.\d+)?)", content)
        if match:
            return min(1.0, max(0.0, float(match.group(1))))
        return 0.5
    except Exception as e:
        print(f"Error in grading: {e}")
        return 0.0

async def evaluate_faithfulness(answer: str, context: str) -> float:
    """Faithfulness: Is the answer derived ONLY from the context?"""
    prompt = f"""
    You are a grader. Rate the "Faithfulness" of the answer on a scale from 0.0 to 1.0.
    1.0 means the answer claims ONLY facts present in the Context.
    0.0 means the answer hallucinates facts not in the Context.
    
    Context: {context}
    Answer: {answer}
    
    Return ONLY the score.
    """
    return await llm_grade(prompt)

async def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """Answer Relevancy: Does the answer actually address the question?"""
    prompt = f"""
    You are a grader. Rate the "Relevancy" of the answer to the question on a scale from 0.0 to 1.0.
    1.0 means the answer directly and fully answers the question.
    0.0 means the answer is irrelevant or ignores the question.
    
    Question: {question}
    Answer: {answer}
    
    Return ONLY the score.
    """
    return await llm_grade(prompt)

# --- 3. MAIN EVALUATION LOOP ---

async def run_evaluation():
    print("Initializing Chatbot...")
    doc_db = DocumentDB()
    docs = doc_db.fetch_documents() 
    
    # Initialize FAISS and Chatbot
    faiss_index = FaissIndex(dim=384)
    chatbot = RAGChatbot(docs, model, faiss_index)
    
    results = {
        "latency": [],
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": []
    }

    print(f"\nStarting evaluation on {len(TEST_DATASET)} test cases...\n")

    for item in TEST_DATASET:
        q = item["question"]
        gt = item["ground_truth"]
        
        # 1. Measure Latency
        start_time = time.time()
        
        # FIX: Added 'await' because query is now async
        answer = await chatbot.query(q)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        results["latency"].append(latency_ms)

        # 2. Reconstruct Context (for grading)
        # We manually search to see what the bot 'saw'
        q_embd = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        distances, indices = chatbot.index.search(q_embd, k=2)
        
        valid_indices = [idx for idx in indices[0] if idx < len(chatbot.docs)]
        retrieved_docs = [chatbot.docs[idx] for idx in valid_indices]
        context_text = "\n".join(retrieved_docs)

        # 3. Calculate Context Precision
        if retrieved_docs:
            precision_score = calculate_cosine_similarity(retrieved_docs[0], gt)
            results["context_precision"].append(precision_score)
        else:
            results["context_precision"].append(0.0)

        # 4. Calculate AI Metrics
        faithfulness = await evaluate_faithfulness(answer, context_text)
        relevancy = await evaluate_answer_relevancy(q, answer)
        
        results["faithfulness"].append(faithfulness)
        results["answer_relevancy"].append(relevancy)
        
        print(f"Query: {q}")
        print(f" -> Latency: {latency_ms:.2f}ms | Faithfulness: {faithfulness} | Relevancy: {relevancy}")

    # --- 4. FINAL REPORT ---
    
    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    avg_latency = safe_avg(results["latency"])
    avg_faith = safe_avg(results["faithfulness"])
    avg_rel = safe_avg(results["answer_relevancy"])
    avg_prec = safe_avg(results["context_precision"])

    print("\n" + "="*40)
    print("FINAL EVALUATION RESULTS")
    print("="*40)
    print(f"{'Metric':<20} | {'Your RAG':<10} | {'Baseline':<15}")
    print("-" * 55)
    print(f"{'Context Precision':<20} | {avg_prec:.4f}     | {'0.60 - 0.75':<15}")
    print(f"{'Faithfulness':<20} | {avg_faith:.4f}     | {'0.70 - 0.85':<15}")
    print(f"{'Answer Relevancy':<20} | {avg_rel:.4f}     | {'0.60 - 0.80':<15}")
    print(f"{'Latency (ms)':<20} | {avg_latency:.2f}     | {'1500 - 3000':<15}")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(run_evaluation())