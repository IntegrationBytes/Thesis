from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
import ast
import os
from dotenv import load_dotenv

load_dotenv()

# KnoBuilder-inspired State [cite: 115]
class KnoBuilderState(TypedDict):
    recipe_corpus: List[str]      # The unstructured text [cite: 114]
    knowledge_graph: List[dict]   # The evolving G_t [cite: 115]
    schema_context: dict          # Cooking ontology [cite: 192]
    current_plan: List[str]       # Strategic queries [cite: 149]
    model_name: Optional[str]     # Selected LLM model name

# Extraction Module: Turns text to validated triples [cite: 191]
def extraction_node(state: KnoBuilderState):
    recipe_text = state["recipe_corpus"][0] if state["recipe_corpus"] else ""
    model_name = state.get("model_name", "llama3.2")
    
    # Determine prompt based on schema_context
    schema = state.get("schema_context", {})
    if schema:
        domain = schema.get("domain", "general knowledge")
        relations = schema.get("relations", [])
        
        if relations:
            relation_definitions = {
                "Used-for": "X is used for Y (method/tool for task)",
                "Feature-of": "X is a feature/attribute of Y",
                "Compare": "X is compared to Y",
                "Part-of": "X is a part/component of Y",
                "Hyponym-of": "X is a type/hyponym of Y",
                "Evaluate-for": "X is evaluated for Y",
                "Conjunction": "X and Y are conjoined",
                "Coreference": "X and Y refer to the same entity"
            }
            relation_str = "\n".join([f"- {r}: {relation_definitions.get(r, r)}" for r in relations])
            predicate_instruction = f"Allowed Predicates:\n{relation_str}"
        else:
            predicate_instruction = "Extract meaningful relationships between entities."

        system_prompt = (
            f"You are an expert scientific information extraction system for the {domain} domain.\n"
            f"Your task is to extract exact entity mentions and their relations from the text.\n\n"
            f"{predicate_instruction}\n\n"
            "Strategy:\n"
            "1. Identify all scientific entities (methods, tasks, metrics, materials, generic terms).\n"
            "2. For each entity, look for relations to other entities in the same or adjacent sentences.\n"
            "3. Pay special attention to 'Used-for' (methods used for tasks) and 'Feature-of' (attributes of objects).\n"
            "4. Capture Coreference: If a term refers to a previously mentioned entity (e.g., 'this method', 'it', 'the system'), link them with 'Coreference'.\n"
            "5. Look for linguistic patterns:\n"
            "   - 'such as', 'including', 'like' -> Hyponym-of\n"
            "   - 'in', 'within', 'component of' -> Part-of\n"
            "   - 'based on', 'uses', 'employing' -> Used-for\n"
            "6. Specificity: Prefer linking to specific entities over general ones (e.g., link to 'SVM' rather than 'algorithm' if 'SVM' is the focus).\n"
            "\n"
            "Rules:\n"
            "- Extract entities EXACTLY as they appear in the text (keep adjectives like 'phrase-based', 'statistical').\n"
            "- Do not infer relations that are not supported by the text.\n"
            "- Return a valid JSON list of triples.\n"
            "\n"
            "Examples:\n"
            "Text: 'We use a LSTM network for sequence modeling. It achieves high accuracy.'\n"
            "Triples: [\n"
            "  {'s': 'LSTM network', 'p': 'Used-for', 'o': 'sequence modeling'},\n"
            "  {'s': 'It', 'p': 'Coreference', 'o': 'LSTM network'},\n"
            "  {'s': 'accuracy', 'p': 'Feature-of', 'o': 'It'}\n"
            "]\n"
            "\n"
            "Text: 'The accuracy of our model is higher than the baseline SVM.'\n"
            "Triples: [\n"
            "  {'s': 'accuracy', 'p': 'Feature-of', 'o': 'model'},\n"
            "  {'s': 'model', 'p': 'Compare', 'o': 'baseline SVM'}\n"
            "]\n"
            "\n"
            "Text: 'We present a formal analysis of words in dialog, such as alternative markers.'\n"
            "Triples: [\n"
            "  {'s': 'formal analysis', 'p': 'Used-for', 'o': 'alternative markers'},\n"
            "  {'s': 'words', 'p': 'Part-of', 'o': 'dialog'},\n"
            "  {'s': 'alternative markers', 'p': 'Hyponym-of', 'o': 'words'}\n"
            "]\n"
        )
    else:
        # Default Cooking Prompt
        system_prompt = (
            "You are a cooking knowledge graph extractor. Extract triples (Subject, Predicate, Object) from the recipe text. \n"
            "Ignore navigation menus, ads, and comments. Focus ONLY on ingredients and cooking steps.\n"
            "Return ONLY a valid JSON list. Format: [{'s': 'Subject', 'p': 'Predicate', 'o': 'Object'}]. \n"
            "Do NOT use empty strings. Do NOT include explanations. Keep it brief."
        )

    try:
        # Check if using OpenRouter
        if "openrouter" in model_name or "/" in model_name:
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0
            )
        else:
            # Attempt to use local LLM (Ollama)
            # Limit context window and output tokens for speed
            llm = ChatOllama(
                model=model_name, 
                temperature=0,
                num_ctx=2048,      # Limit context window
                num_predict=512    # Limit output tokens
            )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Text:\n{recipe_text}\n\nTriples (JSON):")
        ]
        
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Clean up if markdown code blocks are used
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        try:
            triples = json.loads(content)
        except json.JSONDecodeError:
            # Fallback for single quotes or python-style dicts
            try:
                triples = ast.literal_eval(content)
            except:
                raise # Re-raise original error if both fail
        
        # Ensure it's a list of dicts
        if isinstance(triples, list):
            return {"knowledge_graph": triples}
        else:
            return {"knowledge_graph": []}
            
    except Exception as e:
        print(f"LLM extraction failed (is Ollama running?): {e}")
        # Fallback to dummy data
        return {"knowledge_graph": [{"s": "Onion", "p": "Sautéed_With", "o": "Garlic"}]}

# Consolidation Module: Merges common paths 
def consolidation_node(state: KnoBuilderState):
    # logic: Entity Resolution [cite: 209]
    # If Node_A similarity > 0.85, merge nodes [cite: 232]
    return {"knowledge_graph": state["knowledge_graph"]} # merged graph

# 3. Build the KnoBuilder Loop [cite: 576]
builder = StateGraph(KnoBuilderState)
builder.add_node("extract", extraction_node)
builder.add_node("consolidate", consolidation_node)

builder.add_edge(START, "extract")
builder.add_edge("extract", "consolidate")
builder.add_edge("consolidate", END) # In full KnoBuilder, this loops back! 

app = builder.compile()

if __name__ == "__main__":
    print("Running KnoBuilder Demo...")
    initial_state = {
        "recipe_corpus": ["Saute onion and garlic."],
        "knowledge_graph": [],
        "schema_context": {},
        "current_plan": [],
        "model_name": "google/gemini-2.0-flash-001"
    }
    result = app.invoke(initial_state)
    print("Final State:")
    print(json.dumps(result, indent=2))
