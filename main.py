"""
Ollama Local Model Version - Train/Test CSV Format
Cost: $0.00 | Uses: train.csv, test.csv | Outputs: results.csv, training_evaluation_ollama.csv
All prompts and logic identical to Gemini version - ONLY API calls changed
"""

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except (ImportError, AttributeError):
    PATHWAY_AVAILABLE = False
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

import ollama
import json
import os
import pandas as pd
from typing import List, Dict, Tuple
import time
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

TRAIN_CSV = "train.csv"  # Format: id, content, book_name, char, label
TEST_CSV = "test.csv"    # Format: id, content, book_name, char
OUTPUT_CSV = "results.csv"

BOOK_MAPPING = {
    'In Search of the Castaways': 'In_search_of_the_castaways.txt',
    'The Count of Monte Cristo': 'The_Count_of_Monte_Cristo.txt'
}

# Track API calls
API_CALLS = 0

# =============================================================================
# IMPROVED LLM FUNCTIONS (SAME PROMPTS AS GEMINI VERSION)
# =============================================================================

def llm_extract_claims(backstory_text: str) -> List[Dict[str, str]]:
    """
    Extract structured claims with categories for better verification.
    Returns list of dicts with 'claim', 'category', and 'importance'.
    IDENTICAL PROMPT TO GEMINI VERSION
    """
    global API_CALLS
    
    prompt = f"""
You are a forensic narrative analyst. Extract ALL verifiable claims from this backstory.

Backstory:
{backstory_text}

Instructions:
1. Extract claims about: timeline/dates, locations, relationships, character traits, 
   formative events, beliefs/motivations, physical descriptions, socioeconomic status.
2. For each claim, categorize as: TEMPORAL, SPATIAL, RELATIONAL, PSYCHOLOGICAL, 
   BIOGRAPHICAL, or CAUSAL.
3. Rate importance: HIGH (core identity), MEDIUM (significant detail), LOW (minor detail).

Return ONLY valid JSON array with this structure:
[
  {{"claim": "Born in 1985", "category": "TEMPORAL", "importance": "HIGH"}},
  {{"claim": "Only child", "category": "RELATIONAL", "importance": "HIGH"}},
  {{"claim": "Afraid of water", "category": "PSYCHOLOGICAL", "importance": "MEDIUM"}}
]

No preamble, no markdown, pure JSON array only.
"""
    
    for attempt in range(3):
        try:
            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
                options={'temperature': 0, 'num_predict': 1000}
            )
            API_CALLS += 1
            
            text = response['response'].strip()
            
            # Extract JSON from response
            if '[' in text and ']' in text:
                start = text.find('[')
                end = text.rfind(']') + 1
                text = text[start:end]
            
            text = text.replace("```json", "").replace("```", "").strip()
            claims = json.loads(text)
            
            if isinstance(claims, list) and all(
                isinstance(c, dict) and 'claim' in c for c in claims
            ):
                print(f"✓ Extracted {len(claims)} structured claims")
                return claims
            else:
                print(f"Attempt {attempt+1}: Invalid claim structure")
                
        except json.JSONDecodeError as e:
            print(f"Attempt {attempt+1}: JSON parse error: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"Attempt {attempt+1}: Error: {e}")
            time.sleep(1)
    
    print("⚠ Falling back to simple extraction")
    return [{"claim": backstory_text, "category": "GENERAL", "importance": "HIGH"}]


def verify_temporal_logic(claim: str, evidence: str, context: str) -> Dict:
    """
    Dedicated function for temporal/mathematical verification.
    Uses LLM to extract dates/ages and verify arithmetic consistency.
    IDENTICAL PROMPT TO GEMINI VERSION
    """
    global API_CALLS
    
    prompt = f"""
You are a temporal logic validator. Your ONLY job is to check if dates, ages, and timelines are mathematically consistent.

CLAIM: {claim}
EVIDENCE: {evidence}
CONTEXT: {context[:300]}...

TASK:
1. Extract ALL mentions of:
   - Birth years (e.g., "born in 1988")
   - Ages at specific times (e.g., "7 years old in 1995")
   - Event dates (e.g., "graduated in 2006")
   - Time periods (e.g., "spent 10 years there")

2. Do the MATH:
   - If "born in X" and "age Y in year Z", check: X = Z - Y (allow ±1 year tolerance)
   - If "event A at age X" and "event B at age Y", check timeline makes sense
   - If durations mentioned, verify they add up

3. BE STRICT: If numbers are off by more than 1 year, it's an error.

Return ONLY this JSON:
{{
  "is_valid": true or false,
  "reason": "specific math explanation if invalid, e.g., 'Born 1988 but 7 years old in 1995 = born 1988 ✓' or 'Born 1990 but graduated college in 1985 = impossible'",
  "confidence": 0.0 to 1.0,
  "extracted_dates": {{"claim_dates": [], "evidence_dates": []}}
}}
"""
    
    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={'temperature': 0, 'num_predict': 500}
        )
        API_CALLS += 1
        
        text = response['response'].strip()
        
        if '{' in text and '}' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            text = text[start:end]
        
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result
    except Exception as e:
        print(f"    ⚠ Temporal verification error: {e}")
        return {"is_valid": True, "reason": "Could not verify", "confidence": 0.5}


def check_consistency_advanced(
    claim_obj: Dict[str, str], 
    evidence_chunks: List[Dict], 
    backstory_context: str
) -> Tuple[bool, str, float]:
    """
    Multi-pass consistency check with confidence scoring.
    Returns: (is_consistent, reason, confidence_score)
    IDENTICAL PROMPT TO GEMINI VERSION
    """
    global API_CALLS
    
    claim = claim_obj['claim']
    category = claim_obj.get('category', 'GENERAL')
    
    evidence_text = "\n---CHUNK---\n".join([
        f"[Source: Page/Section {i+1}]\n{chunk.get('text', chunk.get('chunk', ''))}" 
        for i, chunk in enumerate(evidence_chunks)
    ])
    
    if not evidence_text.strip():
        return True, "No relevant evidence found in novel", 0.5
    
    # SPECIAL HANDLING: Temporal claims get extra verification
    if category == "TEMPORAL":
        temporal_check = verify_temporal_logic(claim, evidence_text, backstory_context)
        if not temporal_check["is_valid"]:
            return False, f"TEMPORAL MATH ERROR: {temporal_check['reason']}", temporal_check["confidence"]
    
    prompt = f"""
You are a narrative consistency validator. Your job is to detect logical contradictions.

CLAIM TYPE: {category}
CLAIM: {claim}

BACKSTORY CONTEXT (for reference):
{backstory_context[:500]}...

EVIDENCE FROM NOVEL:
{evidence_text}

ANALYSIS FRAMEWORK:
1. DIRECT CONTRADICTION: Does evidence explicitly contradict the claim?
   Example: Claim="only child", Evidence="her two brothers"
   
2. CAUSAL IMPOSSIBILITY: Does the claim make later events impossible?
   Example: Claim="never learned to read", Evidence="writes novels professionally"
   
3. TEMPORAL INCONSISTENCY: Do dates/ages not align mathematically?
   Example: Claim="born 1990", Evidence="graduated college in 1985"
   
   ⚠️ PAY SPECIAL ATTENTION TO DATES, AGES, AND TIMELINES:
   - If claim says "born in X" and evidence says "age Y in year Z", verify X = Z - Y
   - If claim says "event A at age X" and "event B at age Y", check if timeline makes sense
   - If the math doesn't add up by more than 1 year, it is a CONTRADICTION
   - Account for birth year uncertainty (e.g., "born in 1988" could mean late 1987 or early 1988)
   
4. NARRATIVE CONSTRAINT VIOLATION: Does it break established world rules?
   Example: Claim="grew up in coastal town", Evidence="never saw ocean until age 30"

CRITICAL: Ignore thematically similar but logically compatible text. 
Only flag ACTUAL contradictions.

Output ONLY this JSON (no markdown):
{{
  "status": "CONSISTENT" or "CONTRADICTION",
  "reason": "specific explanation with evidence reference",
  "confidence": 0.0 to 1.0,
  "contradiction_type": "DIRECT|CAUSAL|TEMPORAL|CONSTRAINT|NONE"
}}
"""
    
    for attempt in range(2):
        try:
            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
                options={'temperature': 0, 'num_predict': 500}
            )
            API_CALLS += 1
            
            text = response['response'].strip()
            
            if '{' in text and '}' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                text = text[start:end]
            
            text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
            
            is_consistent = result.get("status") == "CONSISTENT"
            reason = result.get("reason", "No reason provided")
            confidence = float(result.get("confidence", 0.5))
            
            return is_consistent, reason, confidence
            
        except Exception as e:
            print(f"  Consistency check attempt {attempt+1} failed: {e}")
            time.sleep(0.5)
    
    return True, "Error in consistency check - defaulting to consistent", 0.3


# =============================================================================
# FALLBACK VECTOR STORE FOR WINDOWS
# =============================================================================

class SimpleVectorStore:
    """Lightweight vector store for semantic search"""
    
    def __init__(self, novel_path: str):
        print(f"  Loading novel: {novel_path}")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = self._read_and_chunk_file(novel_path)
        if self.chunks:
            print(f"  Created {len(self.chunks)} chunks")
            print(f"  Encoding chunks...")
            self.embeddings = self.model.encode(self.chunks, show_progress_bar=False)
        else:
            self.embeddings = np.array([])
    
    def _read_and_chunk_file(self, path: str, chunk_size: int = 500) -> List[str]:
        """Read file and split into overlapping chunks"""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            chunks = []
            overlap = 100
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks
        except Exception as e:
            print(f"  Error reading file {path}: {e}")
            return []

    def query(self, query: str, k: int = 5) -> List[Dict]:
        """Find top k most relevant chunks"""
        if len(self.chunks) == 0:
            return []
            
        query_vec = self.model.encode([query])
        similarities = cosine_similarity(query_vec, self.embeddings)[0]
        
        top_k = min(k, len(self.chunks))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.chunks[idx],
                'chunk': self.chunks[idx],
                'score': float(similarities[idx])
            })
        return results


def build_vector_store_for_novel(novel_filename: str):
    """Build vector store (uses Pathway on Linux, fallback on Windows)"""
    if not os.path.exists(novel_filename):
        raise FileNotFoundError(f"Novel not found: {novel_filename}")
    
    print(f"  Building vector store for: {novel_filename}")
    
    if PATHWAY_AVAILABLE:
        # Pathway implementation (same as Gemini version)
        data_source = pw.io.fs.read(
            novel_filename,
            format="binary",
            mode="static",
            with_metadata=True
        )
        documents = data_source.select(
            text=pw.this.data,
            path=pw.this.metadata["path"]
        )
        embedder = pw.xpacks.llm.embedders.SentenceTransformerEmbedder(
            model="all-MiniLM-L6-v2"
        )
        vector_store = pw.xpacks.llm.VectorStore(
            documents,
            embedder=embedder,
            parser=pw.xpacks.llm.parsers.OpenParse()
        )
        return vector_store
    else:
        print("  → Using Windows-compatible fallback")
        return SimpleVectorStore(novel_filename)


def multi_pass_retrieval(vector_store, claim: str, k_per_pass: int = 5) -> List[Dict]:
    """Enhanced retrieval: query variations + reranking (SAME AS GEMINI VERSION)"""
    all_chunks = []
    
    # Pass 1: Direct query
    chunks_1 = vector_store.query(query=claim, k=k_per_pass)
    all_chunks.extend(chunks_1)
    
    # Pass 2: Negation query (find contradictory evidence)
    negation_query = f"evidence that contradicts: {claim}"
    chunks_2 = vector_store.query(query=negation_query, k=k_per_pass)
    all_chunks.extend(chunks_2)
    
    # Pass 3: Context expansion (related events)
    context_query = f"events related to: {claim}"
    chunks_3 = vector_store.query(query=context_query, k=k_per_pass)
    all_chunks.extend(chunks_3)
    
    # Deduplicate chunks
    seen_texts = set()
    unique_chunks = []
    for chunk in all_chunks:
        text = chunk.get('text', chunk.get('chunk', ''))[:100]
        if text not in seen_texts:
            seen_texts.add(text)
            unique_chunks.append(chunk)
    
    return unique_chunks[:15]


# =============================================================================
# MAIN PIPELINE (SAME LOGIC AS GEMINI VERSION)
# =============================================================================

def process_single_backstory(
    backstory: str,
    book_name: str,
    vector_stores: Dict[str, SimpleVectorStore],
    character_name: str = ""
) -> Tuple[int, str]:
    """Process one backstory - IDENTICAL LOGIC TO GEMINI VERSION"""
    
    if book_name not in vector_stores:
        return 1, f"Unknown book: {book_name}"
    
    vector_store = vector_stores[book_name]
    
    # Extract claims
    print(f"    Extracting claims...")
    claims = llm_extract_claims(backstory)
    
    if not claims:
        return 1, "No verifiable claims found"
    
    print(f"    Found {len(claims)} claims to verify")
    
    # Verify claims
    contradictions = []
    confidence_scores = []
    
    for idx, claim_obj in enumerate(claims, 1):
        claim = claim_obj['claim']
        importance = claim_obj.get('importance', 'MEDIUM')
        
        print(f"    [{idx}/{len(claims)}] Checking: {claim[:60]}...")
        print(f"      Category: {claim_obj.get('category')} | Importance: {importance}")
        
        # Multi-pass retrieval
        evidence_chunks = multi_pass_retrieval(vector_store, claim, k_per_pass=5)
        print(f"      Retrieved {len(evidence_chunks)} evidence chunks")
        
        # Check consistency
        is_consistent, reason, confidence = check_consistency_advanced(
            claim_obj, evidence_chunks, backstory
        )
        
        confidence_scores.append(confidence)
        
        if not is_consistent:
            print(f"      ❌ CONTRADICTION (confidence: {confidence:.2f})")
            print(f"         Reason: {reason}")
            contradictions.append({
                'claim': claim,
                'reason': reason,
                'confidence': confidence,
                'importance': importance
            })
            
            # Early stopping for HIGH importance contradictions
            if importance == "HIGH" and confidence > 0.8:
                print(f"      ⚠ Critical contradiction - stopping early")
                break
        else:
            print(f"      ✓ Consistent (confidence: {confidence:.2f})")
    
    # Final decision (SAME LOGIC AS GEMINI VERSION)
    if contradictions:
        prediction = 0
        top = max(contradictions, key=lambda x: x['confidence'])
        rationale = f"{top['claim'][:40]}... - {top['reason'][:80]}"
        print(f"    ✗ PREDICTION: CONTRADICTION (0)")
    else:
        prediction = 1
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        rationale = f"All {len(claims)} claims verified consistent (avg confidence: {avg_confidence:.2f})"
        print(f"    ✓ PREDICTION: CONSISTENT (1)")
    
    return prediction, rationale


def evaluate_on_training_data(vector_stores: Dict[str, SimpleVectorStore]) -> float:
    """Evaluate on training data - OUTPUTS training_evaluation_ollama.csv"""
    
    if not os.path.exists(TRAIN_CSV):
        print(f"Training file not found: {TRAIN_CSV}")
        return 0.0
    
    print("\n" + "="*70)
    print("EVALUATING ON TRAINING DATA")
    print("="*70)
    
    train_df = pd.read_csv(TRAIN_CSV)
    
    correct = 0
    results = []
    
    for idx, row in train_df.iterrows():
        story_id = row['id']
        backstory = row['content']
        book_name = row['book_name']
        character = row['char']
        actual_label = row['label']
        
        actual = 0 if actual_label == 'contradict' else 1
        
        print(f"\n[{idx+1}/{len(train_df)}] ID={story_id} | Actual: {actual_label}")
        
        # Predict
        prediction, rationale = process_single_backstory(
            backstory, book_name, vector_stores, character
        )
        
        # Check correctness
        is_correct = (prediction == actual)
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Predicted={prediction}, Actual={actual} | {rationale[:60]}...")
        print(f"   Ollama API calls: {API_CALLS}")
        
        results.append({
            'id': story_id,
            'predicted': prediction,
            'actual': actual,
            'correct': is_correct,
            'rationale': rationale
        })
    
    accuracy = correct / len(train_df)
    
    print("\n" + "="*70)
    print(f"TRAINING ACCURACY: {accuracy:.2%} ({correct}/{len(train_df)})")
    
    # Confusion matrix
    tp = sum(1 for r in results if r['predicted'] == 1 and r['actual'] == 1)
    tn = sum(1 for r in results if r['predicted'] == 0 and r['actual'] == 0)
    fp = sum(1 for r in results if r['predicted'] == 1 and r['actual'] == 0)
    fn = sum(1 for r in results if r['predicted'] == 0 and r['actual'] == 1)
    
    print("\nConfusion Matrix:")
    print(f"  True Positives (correct consistent):  {tp}")
    print(f"  True Negatives (correct contradict):  {tn}")
    print(f"  False Positives (missed contradict):  {fp}")
    print(f"  False Negatives (wrong contradict):   {fn}")
    
    print(f"\nTotal Ollama Requests: {API_CALLS}")
    print(f"Cost: $0.00 (Local model)")
    print("="*70)
    
    # Save results
    pd.DataFrame(results).to_csv("training_evaluation_ollama.csv", index=False)
    print(f"\n✓ Detailed results saved to: training_evaluation_ollama.csv")
    
    return accuracy


def generate_test_predictions(vector_stores: Dict[str, SimpleVectorStore]):
    """Generate test predictions - OUTPUTS results.csv"""
    
    if not os.path.exists(TEST_CSV):
        print(f"Test file not found: {TEST_CSV}")
        return
    
    print("\n" + "="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70)
    
    test_df = pd.read_csv(TEST_CSV)
    results = []
    
    for idx, row in test_df.iterrows():
        story_id = row['id']
        backstory = row['content']
        book_name = row['book_name']
        character = row['char']
        
        print(f"\n[{idx+1}/{len(test_df)}] ID={story_id} | {book_name} - {character}")
        
        # Predict
        prediction, rationale = process_single_backstory(
            backstory, book_name, vector_stores, character
        )
        
        print(f"   Ollama requests: {API_CALLS}")
        
        results.append({
            'id': story_id,
            'prediction': prediction,
            'rationale': rationale
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df[['id', 'prediction']].to_csv(OUTPUT_CSV, index=False)
    results_df.to_csv("results_detailed_ollama.csv", index=False)
    
    print("\n" + "="*70)
    print(f"✓ PREDICTIONS SAVED")
    print(f"  Submission file: {OUTPUT_CSV}")
    print(f"  Detailed file: results_detailed_ollama.csv")
    print(f"  Total: {len(results)}")
    print(f"  Consistent: {sum(r['prediction'] == 1 for r in results)}")
    print(f"  Contradict: {sum(r['prediction'] == 0 for r in results)}")
    print(f"\nTotal Ollama Requests: {API_CALLS}")
    print(f"Total Cost: $0.00 (Free!)")
    print("="*70)


def check_ollama_setup():
    """Verify Ollama is working"""
    print("\nChecking Ollama setup...")
    try:
        test_response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt="Say OK",
            options={'num_predict': 5}
        )
        print(f"✓ Ollama working with model: {OLLAMA_MODEL}")
        return True
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print(f"\nSetup instructions:")
        print(f"  1. Install Ollama from https://ollama.ai/download")
        print(f"  2. Run: ollama pull {OLLAMA_MODEL}")
        print(f"  3. Verify: ollama list")
        return False


def main():
    """Main entry point"""
    
    print("="*70)
    print("HACKATHON 2026 - Ollama Version (Train/Test Format)")
    print(f"Model: {OLLAMA_MODEL} | Cost: $0.00")
    print("="*70)
    
    # Check Ollama
    if not check_ollama_setup():
        return
    
    # Build vector stores
    print("\n[STEP 1] Building vector stores...")
    vector_stores = {}
    
    for book_name, filename in BOOK_MAPPING.items():
        if os.path.exists(filename):
            print(f"\n📚 Building for: {book_name}")
            vector_stores[book_name] = build_vector_store_for_novel(filename)
        else:
            print(f"\n⚠ Novel not found: {filename}")
    
    if not vector_stores:
        print("\n❌ No novels found!")
        return
    
    # Evaluate on training data
    print("\n[STEP 2] Evaluating on training data...")
    if os.path.exists(TRAIN_CSV):
        accuracy = evaluate_on_training_data(vector_stores)
        
        if accuracy < 0.60:
            print("\n⚠ Low accuracy. Consider larger model: ollama pull llama3.1:70b")
        elif accuracy > 0.65:
            print("\n✓ Good accuracy! Proceeding to test.")
    
    # Generate test predictions
    print("\n[STEP 3] Generating test predictions...")
    generate_test_predictions(vector_stores)
    
    print("\n✓ COMPLETED!")
    print(f"Total cost: $0.00")
    print(f"Total Ollama requests: {API_CALLS}")


if __name__ == "__main__":
    main()
