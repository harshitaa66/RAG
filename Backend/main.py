import sys
import asyncio
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import os
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import logging
from openai import AsyncOpenAI  # CHANGED: Use Async client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --- Universal Crawler Integration ---
_CRAWLER_DIR = Path(__file__).parent.parent.parent / "universal_crawler"
if str(_CRAWLER_DIR) not in sys.path:
    sys.path.insert(0, str(_CRAWLER_DIR))

try:
    from product_scraper import ProductScraper  # type: ignore[import]
    from paper_scraper import PaperScraper  # type: ignore[import]
    CRAWLER_AVAILABLE = True
    logger.info("Universal crawler loaded successfully")
except Exception as _e:
    CRAWLER_AVAILABLE = False
    ProductScraper = None
    PaperScraper = None
    logger.warning(f"Universal crawler unavailable: {_e}")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load model globally
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class VectorIndex:
    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def build(self, embeddings: np.ndarray):
        raise NotImplementedError

class FaissIndex(VectorIndex):
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product on normalized vecs = cosine similarity
        logger.info(f"Initialized FAISS cosine index with dimension: {dim}")

    def build(self, embeddings: np.ndarray):
        if embeddings.size == 0:
            logger.error("Cannot build index: embeddings array is empty")
            return
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] != self.dim:
            logger.error(f"Mismatch. Expected {self.dim}, got {embeddings.shape[1]}")
            return
        # Normalize to unit vectors so inner product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)
        self.embeddings = embeddings.astype('float32')
        self.index.reset()
        self.index.add(self.embeddings)
        logger.info(f"Built FAISS cosine index with {embeddings.shape[0]} documents")

    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (cosine_scores, indices) — scores in [-1, 1], higher is better."""
        if self.index.ntotal == 0:
            return np.array([[0.0]]), np.array([[0]])
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        norm = np.linalg.norm(query_embedding)
        query_embedding = query_embedding / max(float(norm), 1e-10)
        query_embedding = query_embedding.astype('float32')
        k = min(k, self.index.ntotal)
        return self.index.search(query_embedding, k)

class RAGChatbot:
    def __init__(self, docs: List[str], model: SentenceTransformer, index: Optional[VectorIndex] = None):
        # Filter out empty documents
        self.docs = [str(doc).strip() for doc in docs if doc and str(doc).strip()]

        if not self.docs:
            logger.error("No valid documents provided to RAGChatbot")
            # We don't raise error here to allow fallback usage

        logger.info(f"Initialized RAGChatbot with {len(self.docs)} documents")

        self.model = model
        self.index = index

        # Only build index if we have docs
        if self.docs:
            self._build_index()

    def _build_index(self):
        """Build dense (FAISS cosine) + sparse (BM25) hybrid index."""
        logger.info("Building embeddings for documents...")
        embeddings = self.model.encode(self.docs, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings.astype("float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        dim = embeddings.shape[1]
        if self.index is None:
            self.index = FaissIndex(dim)
        self.index.build(embeddings)

        # BM25 sparse index for keyword matching
        tokenized = [doc.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("Hybrid index ready (FAISS cosine + BM25)")

    def _hybrid_search(self, query: str, k: int = 5) -> Tuple[List[float], List[int]]:
        """Combine cosine similarity (FAISS) + BM25 keyword scores → ranked indices."""
        # --- Dense (cosine) ---
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        cos_scores, cos_indices = self.index.search(q_emb, k)
        cos_scores = cos_scores[0].tolist()
        cos_indices = cos_indices[0].tolist()

        # --- Sparse (BM25) over all docs ---
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_max = bm25_scores.max()
        bm25_norm = (bm25_scores / bm25_max) if bm25_max > 0 else bm25_scores

        # --- Combine: seed with FAISS candidates, pull in top BM25 extras ---
        candidates: dict = {}
        for score, idx in zip(cos_scores, cos_indices):
            if 0 <= idx < len(self.docs):
                norm_cos = (float(score) + 1.0) / 2.0   # [-1,1] → [0,1]
                candidates[idx] = 0.7 * norm_cos + 0.3 * float(bm25_norm[idx])

        for idx in np.argsort(bm25_norm)[::-1][:k]:
            if int(idx) not in candidates and 0 <= idx < len(self.docs):
                candidates[int(idx)] = 0.3 * float(bm25_norm[idx])

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in ranked[:k]]
        top_scores  = [score for _, score in ranked[:k]]
        return top_scores, top_indices

    async def call_rag(self, q: str, indices: List[int]) -> Tuple[str, List[str]]:
        """Generate response using hybrid-retrieved documents."""
        try:
            valid = [idx for idx in indices[:3] if 0 <= idx < len(self.docs)]
            retrieved_docs = [self.docs[idx] for idx in valid]

            if not retrieved_docs:
                logger.warning("No valid documents retrieved")
                return await self.fallback(q)

            context = "\n\n".join(retrieved_docs)
            logger.info(f"call_rag: {len(retrieved_docs)} docs retrieved")

            prompt = (
                f"Answer based on the context below. "
                f"If the answer is not in the context, say exactly: Not in context.\n\n"
                f"Context:\n{context}\n\nQuestion: {q}"
            )

            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )

            answer = response.choices[0].message.content.strip()

            if answer.lower().strip().rstrip('.') == "not in context":
                logger.info("RAG: 'Not in context' → escalating to crawler fallback")
                return await self.fallback(q)

            return answer, retrieved_docs

        except Exception as e:
            logger.error(f"Error in call_rag: {str(e)}")
            return await self.fallback(q)

    @staticmethod
    def _detect_domain(query: str) -> str:
        """Identify which crawler domain matches the query based on keywords."""
        q_lower = query.lower()
        words = set(q_lower.split())

        _PRODUCT_KEYWORDS = {
            'buy', 'price', 'shop', 'product', 'amazon', 'ebay', 'cheap',
            'deal', 'order', 'purchase', 'store', 'brand', 'cost', 'shipping',
            'walmart', 'available', 'stock', 'review', 'sell', 'sold', 'cart',
            'discount', 'offer', 'marketplace', 'listing',
        }
        _ACADEMIC_KEYWORDS = {
            'research', 'paper', 'study', 'journal', 'arxiv', 'citation',
            'abstract', 'published', 'author', 'doi', 'conference', 'ieee',
            'acm', 'pubmed', 'academic', 'findings', 'dataset', 'algorithm',
            'experiment', 'thesis', 'literature', 'survey', 'preprint',
        }

        if words & _PRODUCT_KEYWORDS:
            return 'products'
        if words & _ACADEMIC_KEYWORDS:
            return 'academic'
        return 'general'

    async def _crawl_for_context(self, query: str, domain: str) -> str:
        """Fetch live web context using the appropriate crawler, run in thread pool."""
        from core_crawler import CoreCrawler  # type: ignore[import]
        loop = asyncio.get_running_loop()
        # Short timeout + minimal delay so crawler doesn't block the response
        fast_crawler = CoreCrawler(timeout=5, delay_range=(0.5, 1))

        if domain == 'products':
            def _run():
                scraper = ProductScraper(crawler=fast_crawler)
                try:
                    return scraper.search(
                        query,
                        sources=['amazon', 'ebay'],
                        max_results_per_source=3,
                        use_selenium=False
                    )
                finally:
                    scraper.close()

            results = await loop.run_in_executor(None, _run)
            if not results:
                return ""
            lines = [
                f"- {p.title} | Price: {p.price or 'N/A'} | Source: {p.source}"
                for p in results[:5]
            ]
            return "Live product search results:\n" + "\n".join(lines)

        elif domain == 'academic':
            def _run():
                scraper = PaperScraper(crawler=fast_crawler)
                try:
                    return scraper.search(
                        query,
                        sources=['arxiv', 'semantic_scholar'],
                        max_results_per_source=3,
                        use_selenium=False
                    )
                finally:
                    scraper.close()

            results = await loop.run_in_executor(None, _run)
            if not results:
                return ""
            lines = []
            for p in results[:5]:
                entry = f"- {p.title}"
                if p.authors:
                    entry += f" by {', '.join(p.authors[:2])}"
                if p.abstract:
                    entry += f": {p.abstract[:200]}"
                lines.append(entry)
            return "Live academic search results:\n" + "\n".join(lines)

        else:  # general — DuckDuckGo via ddgs library (handles bot detection)
            def _run():
                from ddgs import DDGS  # type: ignore[import]
                with DDGS() as ddgs:
                    hits = list(ddgs.text(query, max_results=5))
                return [
                    f"{h['title']}: {h['body']}"
                    for h in hits if h.get('body')
                ]

            snippets = await loop.run_in_executor(None, _run)
            if not snippets:
                return ""
            return "Web search results:\n" + "\n".join(f"- {s}" for s in snippets)

    async def fallback(self, q: str) -> Tuple[str, List[str]]:
        """Generate response using live crawler context when possible, pure LLM otherwise."""
        crawler_context = ""

        if CRAWLER_AVAILABLE:
            try:
                domain = self._detect_domain(q)
                logger.info(f"Fallback: using '{domain}' crawler for: {q[:60]}")
                crawler_context = await asyncio.wait_for(
                    self._crawl_for_context(q, domain),
                    timeout=8.0
                )
            except asyncio.TimeoutError:
                logger.warning("Crawler timed out after 8s, falling back to pure LLM")
            except Exception as e:
                logger.warning(f"Crawler error in fallback, using pure LLM: {e}")

        if crawler_context:
            prompt = (
                f"Use the following live web data to answer the question accurately.\n\n"
                f"{crawler_context}\n\n"
                f"Question: {q}"
            )
        else:
            logger.info("Fallback: no crawler context, using pure LLM")
            prompt = f"Answer briefly: {q}"

        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip(), []
        except Exception as e:
            logger.error(f"Error in fallback: {str(e)}")
            return f"Error: {str(e)}", []

    async def query(self, q: str, threshold: float = 0.35, k: int = 5) -> Tuple[str, List[str]]:
        """Hybrid retrieval: cosine similarity + BM25, threshold on combined score."""
        if not q or not q.strip():
            return "Please provide a valid question.", []
        if not self.docs or not self.index:
            return await self.fallback(q)

        try:
            scores, indices = self._hybrid_search(q, k=k)
            top_score = scores[0] if scores else 0.0

            logger.info(f"Hybrid top score: {top_score:.3f} vs threshold: {threshold}")

            if top_score >= threshold:
                return await self.call_rag(q, indices)
            else:
                return await self.fallback(q)

        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            return await self.fallback(q)

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about the chatbot"""
        return {
            "num_documents": len(self.docs),
            "sample_documents": self.docs[:3] if self.docs else [],
            "index_total": self.index.index.ntotal if (self.index and hasattr(self.index, 'index')) else 0,
            "embedding_dim": self.index.dim if (self.index and hasattr(self.index, 'dim')) else None
        }
