from src.workflow.graph import graph
from langchain_core.messages import HumanMessage

def test_graph_compilation():
    print("Verifying graph compilation...")
    try:
        # Just check if graph object is created
        if graph:
            print("✅ Graph compiled successfully.")
        else:
            print("❌ Graph compilation failed.")
    except Exception as e:
        print(f"❌ Error during graph compilation: {e}")

if __name__ == "__main__":
    test_graph_compilation()
