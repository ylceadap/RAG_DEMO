import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_DISTANCE_THRESHOLD


def main() -> int:
    cases = json.loads((Path(__file__).parent / "eval_questions.json").read_text(encoding="utf-8"))
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    passed = 0
    for case in cases:
        matches = vectorstore.similarity_search_with_score(case["question"], k=4)
        sources = {
            document.metadata.get("source", "unknown")
            for document, distance in matches
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
