import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.workflow.graph import graph

st.set_page_config(page_title="Fintilligence", page_icon="📈", layout="wide")

st.title("💰 Fintilligence")
st.markdown("### Multi-Agent Finance Intelligence Chatbot")
st.markdown("Ask about stock prices (e.g., 'Price of AAPL') or market summaries.")

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.write(f"**Session ID:** `{st.session_state.session_id}`")

if st.sidebar.button("🗑️ Clear History"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.divider()

# Workflow Visualization
st.sidebar.title("🕸️ Workflow Architecture")
with st.sidebar.expander("View Agent Logic", expanded=True):
    st.markdown("""
    This app uses **LangGraph Choreography** to orchestrate specialized experts:
    
    ```mermaid
    graph TD
        START((START)) --> G[Shield: Guardrails]
        G --> M[Analyst: Market]
        G --> N[Synthesizer: News]
        G --> E[Educator: RAG]
        
        M --> R((Router))
        N --> R
        E --> R
        
        R --> M
        R --> N
        R --> E
        R --> S[Master Synthesizer]
        S --> END((END))
        
        style G fill:#f9f,stroke:#333
        style M fill:#bbf,stroke:#333
        style N fill:#bfb,stroke:#333
        style E fill:#fdb,stroke:#333
        style S fill:#fff,stroke:#333,stroke-width:4px
    ```
    
    1. **Shield**: Decomposes query & checks safety.
    2. **Analyst**: Fetches live prices & technicals.
    3. **News**: Synthesizes latest headlines.
    4. **Educator**: RAG-based financial theory.
    5. **Master**: Deduplicates & joins all expert reports.
    """)

# Live Log Feed
st.sidebar.divider()
st.sidebar.title("📄 Live Audit Log")
if st.sidebar.button("Refresh Logs"):
    try:
        with open("fintilligence.log", "r") as f:
            lines = f.readlines()
            # Show last 15 lines
            recent_logs = "".join(lines[-15:])
            st.sidebar.code(recent_logs, language="text")
    except FileNotFoundError:
        st.sidebar.info("No logs found yet. Start a chat to generate logs!")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ask about financial markets..."):
    # Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare Config for Checkpointer
    config = {"configurable": {"thread_id": st.session_state.session_id}}
    
    # Invoke LangGraph
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        invoked_agents = []
        llm_calls = 0
        
        try:
            inputs = {"messages": [HumanMessage(content=prompt)]}
            full_response_parts = []
            
            # Get current message count to only process NEW messages from the stream
            current_state = graph.get_state(config)
            last_seen_msg_count = len(current_state.values.get("messages", []))
            
            # Use streaming to capture intermediate steps
            with st.status("Thinking...", expanded=True) as status:
                for event in graph.stream(inputs, config=config):
                    for key, value in event.items():
                        # Track invoked nodes (agents)
                        if key not in invoked_agents:
                            invoked_agents.append(key)
                            # Update status with more granular info
                            if key == "guardrails":
                                status.update(label="🛡️ Checking Safety & Routing...", state="running")
                            elif key == "market_analyst":
                                status.update(label="📊 Fetching Market Data...", state="running")
                            elif key == "news_synthesizer":
                                status.update(label="📰 Searching Financial News...", state="running")
                            elif key == "finance_qa_agent":
                                status.update(label="🎓 Consulting Finance Textbooks...", state="running")
                            elif key == "portfolio_analyst":
                                status.update(label="📊 Analyzing Portfolio...", state="running")
                            elif key == "goal_planner":
                                status.update(label="🎯 Planning Financial Goals...", state="running")
                            elif key == "response_synthesizer":
                                status.update(label="🧠 Synthesizing Final Answer...", state="running")
                        
                        # Increment LLM call for the guardrails node itself
                        if key == "guardrails":
                            llm_calls += 1
                            
                        if "messages" in value:
                            # In 'updates' stream mode (default), value["messages"] is the list of messages 
                            # the node returned as an update. We don't need to slice against full history.
                            new_messages = value["messages"]
                            
                            for msg in new_messages:
                                try:
                                    # Handle weird message types
                                    if msg is None or isinstance(msg, int):
                                        continue

                                    # Increment LLM call count for every AIMessage from agents
                                    if isinstance(msg, AIMessage) and key != "guardrails":
                                        llm_calls += 1
                                        
                                    # Detect Tool Calls
                                    if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
                                        for tool_call in msg.tool_calls:
                                            # Robust name extraction to prevent 'int' object crashes
                                            t_name = "Unknown Tool"
                                            if isinstance(tool_call, dict):
                                                t_name = tool_call.get('name', 'Unknown Tool')
                                            elif hasattr(tool_call, 'name'):
                                                t_name = tool_call.name
                                            
                                            status.write(f"⚙️ **Invoking:** `{t_name}`")
                                            t_args = {}
                                            if isinstance(tool_call, dict):
                                                t_args = tool_call.get('args', {})
                                            elif hasattr(tool_call, 'args'):
                                                t_args = tool_call.args
                                            
                                            if t_args:
                                                status.write(f"  *args:* `{t_args}`")
                                                
                                    elif isinstance(msg, ToolMessage):
                                        status.write(f"🎯 **Tool Result:**")
                                        with st.expander("View Raw Output", expanded=False):
                                            st.code(str(msg.content))

                                    # Capture AI responses for prioritization
                                    elif isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
                                        content = str(msg.content).strip()
                                        if content:
                                            is_final = msg.additional_kwargs.get("is_final_synthesis", False)
                                            full_response_parts.append({
                                                "content": content,
                                                "is_final": is_final,
                                                "node": key
                                            })
                                        
                                        # Display Token Usage
                                        try:
                                            if hasattr(msg, 'response_metadata'):
                                                usage = msg.response_metadata.get('token_usage', {}) or msg.response_metadata.get('usage_metadata', {})
                                                if usage:
                                                    input_tokens = usage.get('input_tokens', 0)
                                                    output_tokens = usage.get('output_tokens', 0)
                                                    st.caption(f"📊 **Token Usage:** {key} | Input: {input_tokens} | Output: {output_tokens}")
                                        except Exception:
                                            pass
                                except Exception as inner_e:
                                    status.write(f"⚠️ *Skipped a messages from `{key}`:* `{str(inner_e)}`")

            # --- FINAL SELECTION LOGIC ---
            # User requested sequential output from multiple agents.
            # We will join all collected AI reports with a clear visual separator.
            if full_response_parts:
                # Deduplicate by content to be safe
                unique_parts = []
                seen_content = set()
                for p in full_response_parts:
                    if p["content"] not in seen_content:
                        unique_parts.append(p["content"])
                        seen_content.add(p["content"])
                
                full_response = "\n\n---\n\n".join(unique_parts)
            else:
                # Last resort: Scan the graph state for any final AI messages
                final_state = graph.get_state(config)
                all_msgs = final_state.values.get("messages", [])
                for m in reversed(all_msgs):
                    if isinstance(m, AIMessage) and not m.tool_calls and m.content:
                        full_response = str(m.content)
                        break

            if not full_response:
                full_response = "I couldn't generate a response. Please check the logs."

            # Performance Metrics
            if invoked_agents:
                agents_str = " ➔ ".join([f"`{a}`" for a in invoked_agents])
                st.caption(f"🕵️ **Agents:** {agents_str} | 🤖 **LLM Calls:** {llm_calls}")

            # RENDER THE SEQUENTIAL RESPONSE
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
