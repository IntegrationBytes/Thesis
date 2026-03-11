import streamlit as st
import graphviz
import ollama
import requests
from bs4 import BeautifulSoup
from demo import app as graph_app
from langchain_ollama import ChatOllama

st.set_page_config(page_title="KnoBuilder", layout="wide")

# Custom CSS for Linear-like styling
st.markdown("""
<style>
    /* Global Reset & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #E0E0E0;
    }
    
    /* Backgrounds */
    .stApp {
        background-color: #121212;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1A1A1A;
        border-right: 1px solid #2A2A2A;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #5E6AD2;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #4B55AA;
        box-shadow: 0 4px 12px rgba(94, 106, 210, 0.3);
    }
    
    /* Inputs */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #222222;
        color: #E0E0E0;
        border: 1px solid #333333;
        border-radius: 6px;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #5E6AD2;
        box-shadow: 0 0 0 1px #5E6AD2;
    }
    
    /* Selectbox */
    .stSelectbox > div > div > div {
        background-color: #222222;
        color: #E0E0E0;
        border: 1px solid #333333;
        border-radius: 6px;
    }

    /* JSON & Code Blocks */
    .stJson, code {
        background-color: #1E1E1E !important;
        border-radius: 6px;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("KnoBuilder")
st.markdown("Extract knowledge graphs from unstructured text using LLMs.")

def fetch_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            script.decompose()
            
        # Try to find main content
        content_element = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'entry-content', 'recipe-content']) or soup.body
        
        # Get text
        text = content_element.get_text()
        
        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text[:3000] # Limit to 3000 chars to avoid context window issues
    except Exception as e:
        return f"Error fetching URL: {e}"

# Check for Ollama connection
try:
    models_info = ollama.list()
    
    # Handle object-based response (newer ollama library)
    if hasattr(models_info, 'models'):
        # models_info.models is a list of Model objects
        # Each Model object has a 'model' attribute containing the name
        model_names = [m.model for m in models_info.models]
    # Handle dictionary response (older versions or raw API)
    elif isinstance(models_info, dict) and 'models' in models_info:
        model_names = [m['name'] for m in models_info['models']]
    else:
        model_names = []
    
    ollama_available = True
except Exception as e:
    ollama_available = False
    model_names = []
    # st.warning(f"Could not connect to Ollama: {e}. Ensure 'ollama serve' is running. Using dummy mode.")

# Add OpenRouter models
openrouter_models = [
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-r1:free"
]
model_names = openrouter_models + model_names

# Sidebar removed, moving content to main page
# with st.sidebar:
#    st.header("Configuration")
    
domain_options = {
    "Cooking (Recipes)": {"domain": "cooking", "relations": ["ingredient", "utensil", "action", "temperature", "time"]},
    "Scientific (Papers)": {"domain": "scientific", "relations": ["used_for", "part_of", "evaluated_on", "compare", "feature_of"]},
    "Biography (People)": {"domain": "biography", "relations": ["born_in", "spouse", "occupation", "educated_at", "parent", "child", "member_of", "position_held"]},
    "General Knowledge": {"domain": "general knowledge", "relations": []}
}

if model_names:
    # Use columns for configuration
    config_col1, config_col2 = st.columns([1, 2])
    with config_col1:
        selected_model = st.selectbox("Select LLM", model_names, index=0)
    with config_col2:
        selected_domain = st.selectbox("Knowledge Domain", list(domain_options.keys()), index=0)
else:
    selected_model = "Dummy (No LLM)"
    st.caption("Start Ollama or set OPENROUTER_API_KEY to use real models.")
    selected_domain = "Cooking (Recipes)"

st.markdown("### Input")
input_mode = st.radio("Input Source", ["Text", "URL"], horizontal=True, label_visibility="collapsed")

if input_mode == "Text":
    default_recipe = "Saute onion and garlic in olive oil. Add chopped tomatoes and simmer for 10 minutes."
    recipe_text = st.text_area("Recipe Corpus", value=default_recipe, height=150)
else:
    url_col, _ = st.columns([3, 1])
    with url_col:
        url = st.text_input("Enter Recipe URL", placeholder="https://www.allrecipes.com/recipe/...", label_visibility="collapsed")
    
    if url:
        with st.spinner("Fetching content..."):
            fetched_text = fetch_text_from_url(url)
            if fetched_text.startswith("Error"):
                st.error(fetched_text)
                recipe_text = ""
            else:
                st.success("Content fetched successfully!")
                with st.expander("View Fetched Content"):
                    st.text(fetched_text[:500] + "...")
                recipe_text = fetched_text
    else:
        recipe_text = ""

run_button = st.button("Build Knowledge Graph", type="primary", disabled=not recipe_text)

# Main content area
if run_button:
    # Determine if we can run
    can_run = True
    if not ollama_available and "/" not in selected_model:
         st.error("Ollama is not running and you selected a local model.")
         can_run = False
         
    if can_run:
        with st.spinner(f"Extracting {selected_domain} knowledge using {selected_model}..."):
            # Prepare initial state
            schema = domain_options[selected_domain]
            # If default cooking is selected, pass empty schema to trigger default prompt or pass specific schema?
            # Let's pass specific schema to use the new dynamic logic
            if selected_domain == "Cooking (Recipes)":
                 schema = {} # Fallback to the original hardcoded prompt for cooking to be safe, or use new one?
                 # The user liked the "original" behavior for cooking, let's keep it if schema is empty.
                 # Actually, my demo.py update handles schema={} by using the default prompt.
                 pass
            
            initial_state = {
                "recipe_corpus": [recipe_text],
                "knowledge_graph": [],
                "schema_context": schema if selected_domain != "Cooking (Recipes)" else {},
                "current_plan": [],
                "model_name": selected_model
            }
        
        try:
            result = graph_app.invoke(initial_state)
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Knowledge Graph Visualization")
                
                # Create Graphviz object
                dot = graphviz.Digraph()
                dot.attr(rankdir='LR')
                # Dark mode graph styling
                dot.attr(bgcolor='#121212')
                dot.attr('node', shape='box', style='filled', fillcolor='#1E1E1E', fontcolor='#E0E0E0', color='#333333', fontname='Inter')
                dot.attr('edge', color='#5E6AD2', fontcolor='#AAAAAA', fontname='Inter')
                
                kg = result.get("knowledge_graph", [])
                
                if not kg:
                    st.info("No knowledge extracted.")
                else:
                    # Add nodes and edges
                    for triple in kg:
                        s = triple.get("s", "").strip()
                        p = triple.get("p", "related_to").strip()
                        o = triple.get("o", "").strip()
                        
                        if s and o:
                            dot.node(str(s), str(s))
                            dot.node(str(o), str(o))
                            dot.edge(str(s), str(o), label=str(p))
                    
                    st.graphviz_chart(dot)
            
            with col2:
                st.subheader("Extracted Triples")
                st.json(result.get("knowledge_graph", []))
                
                with st.expander("Full State"):
                    st.json(result)
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    if not recipe_text and input_mode == "URL":
        st.info("Enter a URL in the sidebar to fetch content.")
    elif not recipe_text:
        st.info("Enter a recipe in the sidebar and click 'Build Knowledge Graph' to start.")
