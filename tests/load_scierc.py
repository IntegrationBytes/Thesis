import os
import glob
import json

def parse_ann_file(ann_path, txt_content):
    entities = {}
    relations = []
    
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if line.startswith('T'):
                # Entity: T1	Generic 26 32	system
                tid = parts[0]
                # parts[1] is "Type Start End"
                type_info = parts[1].split()
                entity_type = type_info[0]
                # Handle discontinuous spans if necessary (e.g. "10 15;20 25") - simplified here
                # Assuming simple spans for now
                text = parts[2]
                entities[tid] = text
            
            elif line.startswith('R'):
                # Relation: R1	USED-FOR Arg1:T4 Arg2:T3
                rid = parts[0]
                rel_info = parts[1].split()
                rel_type = rel_info[0]
                arg1 = rel_info[1].split(':')[1]
                arg2 = rel_info[2].split(':')[1]
                
                relations.append({
                    "s": arg1,
                    "p": rel_type,
                    "o": arg2
                })
    
    # Resolve entity IDs to text
    triples = []
    for rel in relations:
        if rel['s'] in entities and rel['o'] in entities:
            triples.append({
                "s": entities[rel['s']],
                "p": rel['p'],
                "o": entities[rel['o']]
            })
            
    return triples

def convert_scierc_to_json(raw_data_dir, output_file):
    dataset = []
    txt_files = glob.glob(os.path.join(raw_data_dir, "*.txt"))
    
    all_relations = set()
    
    for txt_path in txt_files:
        base_name = os.path.basename(txt_path).replace('.txt', '')
        ann_path = os.path.join(raw_data_dir, f"{base_name}.ann")
        
        if not os.path.exists(ann_path):
            continue
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        triples = parse_ann_file(ann_path, text)
        
        if triples:
            dataset.append({
                "id": base_name,
                "text": text,
                "gold_triples": triples
            })
            
            for t in triples:
                all_relations.add(t['p'])
                
    print(f"Converted {len(dataset)} documents.")
    print(f"Found relations: {all_relations}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    convert_scierc_to_json("data/raw_data", "scierc_converted.json")
