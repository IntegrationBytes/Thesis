import ollama
try:
    models = ollama.list()
    print(f"Type: {type(models)}")
    print(f"Content: {models}")
except Exception as e:
    print(f"Error: {e}")
