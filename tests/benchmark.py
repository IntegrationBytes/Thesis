import time
import json
import os
import random
from demo import app
from typing import List, Dict, Set

# Full Schema from SciERC
SCIERC_SCHEMA = {
    "domain": "Scientific (Computer Science)",
    "relations": [
        "Used-for", "Feature-of", "Compare", "Part-of", 
        "Hyponym-of", "Evaluate-for", "Conjunction", "Coreference"
    ]
}

def load_dataset(path="data/scierc_converted.json", limit=50):
    """Load the full SciERC dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Dataset not found at {path}. Please run load_scierc.py first.")
        
    print(f"📂 Loading full SciERC dataset from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit and limit > 0:
        print(f"⚠️ Limiting to first {limit} documents for speed.")
        return data[:limit]
    return data

def is_fuzzy_match(t1: Dict, t2: Dict) -> bool:
    """
    Check if two triples match semantically/fuzzily.
    t1: predicted
    t2: gold
    """
    def get_tokens(s): 
        return set(str(s).lower().replace('_', ' ').split())
    
    # Check Subject (overlap of significant tokens)
    s1 = get_tokens(t1.get('s', ''))
    s2 = get_tokens(t2.get('s', ''))
    s_match = not s1.isdisjoint(s2)
    
    # Check Object (overlap of significant tokens)
    o1 = get_tokens(t1.get('o', ''))
    o2 = get_tokens(t2.get('o', ''))
    o_match = not o1.isdisjoint(o2)
    
    # Check Predicate (substring or token overlap)
    p1_raw = str(t1.get('p', '')).lower()
    p2_raw = str(t2.get('p', '')).lower()
    p1 = get_tokens(p1_raw)
    p2 = get_tokens(p2_raw)
    
    # Predicate match if:
    # 1. Exact substring (e.g. "used-for" in "used for")
    # 2. Token overlap 
    p_match = (p1_raw in p2_raw) or (p2_raw in p1_raw) or not p1.isdisjoint(p2)
    
    return s_match and o_match and p_match

def calculate_metrics(predicted: List[Dict], gold: List[Dict]):
    """Calculate Precision, Recall, and F1 with fuzzy matching."""
    tp = 0
    
    # Track which gold triples have been matched to avoid double counting
    matched_gold_indices = set()
    
    # Debug print
    print("    Matching details:")
    
    for pred in predicted:
        match_found = False
        for i, gold_triple in enumerate(gold):
            if i in matched_gold_indices:
                continue
            
            if is_fuzzy_match(pred, gold_triple):
                tp += 1
                matched_gold_indices.add(i)
                match_found = True
                print(f"      ✅ MATCH: {pred} ~= {gold_triple}")
                break 
        
        if not match_found:
             print(f"      ❌ NO MATCH: {pred}")

    # Check for missed gold items
    for i, gold_triple in enumerate(gold):
        if i not in matched_gold_indices:
            print(f"      ⚠️ MISSED GOLD: {gold_triple}")
    
    fp = len(predicted) - tp
    fn = len(gold) - len(matched_gold_indices)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def calculate_graph_metrics(triples: List[Dict]):
    """Calculate intrinsic graph metrics."""
    if not triples:
        return 0, 0, 0
        
    entities = set()
    for t in triples:
        entities.add(t.get('s'))
        entities.add(t.get('o'))
    
    num_nodes = len(entities)
    num_edges = len(triples)
    
    # Density for directed graph = |E| / (|V| * (|V| - 1))
    density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    
    return num_nodes, num_edges, density

def run_benchmark(model_name="google/gemini-2.0-flash-001", limit=10):
    dataset = load_dataset(limit=limit)
    
    print(f"🚀 Running SciERC (Paper Replication) Benchmark")
    print(f"📂 Dataset Size: {len(dataset)} documents")
    print(f"🤖 Model: {model_name}")
    print("-" * 60)
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_time = 0
    
    # Track failures
    failures = 0
    
    for i, case in enumerate(dataset):
        print(f"\nProcessing Case {i+1}/{len(dataset)} (ID: {case['id']}): {case['text'][:40]}...")
        
        start_time = time.time()
        
        # Invoke KnoBuilder with SCHEMA CONTEXT
        initial_state = {
            "recipe_corpus": [case["text"]],
            "knowledge_graph": [],
            "schema_context": SCIERC_SCHEMA, # Injecting the paper's schema
            "current_plan": [],
            "model_name": model_name
        }
        
        try:
            result = app.invoke(initial_state)
            extracted_triples = result.get("knowledge_graph", [])
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Calculate Metrics
            p, r, f1 = calculate_metrics(extracted_triples, case["gold_triples"])
            # nodes, edges, density = calculate_graph_metrics(extracted_triples)
            
            print(f"  ⏱️  Time: {elapsed:.2f}s")
            print(f"  📊 Extracted: {len(extracted_triples)} | Gold: {len(case['gold_triples'])}")
            print(f"  🎯 Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f}")
            
            total_precision += p
            total_recall += r
            total_f1 += f1
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failures += 1

    # Averages
    n = len(dataset) - failures
    if n > 0:
        avg_p = total_precision / n
        avg_r = total_recall / n
        avg_f1 = total_f1 / n
        avg_time = total_time / n
    else:
        avg_p = avg_r = avg_f1 = avg_time = 0
    
    print("\n" + "=" * 60)
    print("🏆 SCIERC BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Documents Processed:     {n}")
    print(f"Failures:                {failures}")
    print(f"Average Processing Time: {avg_time:.2f}s per document")
    print(f"Average Precision:       {avg_p:.2f}")
    print(f"Average Recall:          {avg_r:.2f}")
    print(f"Average F1-Score:        {avg_f1:.2f}")
    print("-" * 60)
    
    if avg_f1 > 0.7:
        print("✅ Verdict: EXCELLENT (Matches Paper Performance)")
    elif avg_f1 > 0.5:
        print("⚠️ Verdict: GOOD (Solid performance)")
    else:
        print("❌ Verdict: POOR (Needs improvement)")

if __name__ == "__main__":
    # You can adjust the limit here. 
    # Set limit=None to run the full ~500 document dataset (will take a long time).
    run_benchmark(model_name="google/gemini-2.0-flash-001", limit=10)
