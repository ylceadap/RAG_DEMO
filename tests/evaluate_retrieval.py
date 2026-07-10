import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, RETRIEVAL_DISTANCE_THRESHOLD
from ingest import EMBEDDING_MODEL


def main() -> int:
    cases = json.loads((Path(__file__).parent / "eval_questions.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(EMBEDDING_MODEL)
    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME)
    passed = 0
    for case in cases:
        query = model.encode([case["question"]], normalize_embeddings=True).tolist()
        result = collection.query(query_embeddings=query, n_results=4, include=["metadatas", "distances"])
        sources = {
            meta["source"]
            for meta, distance in zip(result.get("metadatas", [[]])[0], result.get("distances", [[]])[0])
            if distance <= RETRIEVAL_DISTANCE_THRESHOLD
        }
        expected = set(case["expected_sources"])
        ok = sources == expected if not expected else bool(sources & expected)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['question']} -> {sorted(sources)}")
        passed += int(ok)
    print(f"\n{passed}/{len(cases)} retrieval cases passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
