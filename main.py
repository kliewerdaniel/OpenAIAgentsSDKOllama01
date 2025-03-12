from document_agent import create_document_agent, document_memory
from ollama_client import OllamaClient

def print_banner():
    """Print a welcome banner for the Document Analysis Agent."""
    print("\n" + "="*60)
    print("📚 Document Analysis Agent 📚".center(60))
    print("="*60)
    print("\nThis agent can analyze documents, extract information, and search for content.")
    print("It also has document memory to store and retrieve documents between sessions.")
    
    # Check for existing documents
    docs = document_memory.list_documents()
    if docs:
        print(f"\n🗃️  {len(docs)} documents already in memory:")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc['url']}")
    
    print("\nCommands:")
    print("  'exit' - Quit the program")
    print("  'list' - Show stored documents")
    print("  'help' - Show this help message")
    print("="*60 + "\n")

def main():
    print("Initializing Document Analysis Agent...")
    
    agent = create_document_agent()
    
    print_banner()
    
    # Debug: Test agent with a simple query
    try:
        print("\nDEBUG: Testing agent with 'what is war'")
        print("Processing...")
        test_response = agent.run(message="what is war")
        print(f"\nAgent (test): {test_response.message}")
        
        # If tools were used, show info about tool usage
        if test_response.tool_calls:
            print("\n🛠️  Tools Used (test):")
            for tool in test_response.tool_calls:
                # Display more info about each tool call
                inputs = getattr(tool, 'inputs', {})
                inputs_str = ', '.join(f"{k}='{v}'" for k, v in inputs.items()) if inputs else ""
                print(f"  • {tool.name}({inputs_str})")
    except Exception as e:
        import traceback
        print(f"\nDEBUG ERROR: {str(e)}")
        traceback.print_exc()
    
    # Start a conversation session
    conversation_id = None
    
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'exit':
                break
                
            if user_input.lower() == 'help':
                print_banner()
                continue
                
            if user_input.lower() == 'list':
                docs = document_memory.list_documents()
                if not docs:
                    print("\nNo documents in memory yet.")
                else:
                    print(f"\n📚 Documents in memory ({len(docs)}):")
                    for i, doc in enumerate(docs, 1):
                        metadata = doc.get('metadata', {})
                        fetched_at = metadata.get('fetched_at', 'unknown time')
                        print(f"  {i}. {doc['url']} (fetched: {fetched_at})")
                continue
            
            # Get agent response
            print("\nProcessing...")
            response = agent.run(
                message=user_input,
                conversation_id=conversation_id
            )
            
            # Store the conversation ID for continuity
            conversation_id = response.conversation_id
            
            # Print the response
            print(f"\nAgent: {response.message}")
            
            # If tools were used, show info about tool usage
            if response.tool_calls:
                print("\n🛠️  Tools Used:")
                for tool in response.tool_calls:
                    # Display more info about each tool call
                    inputs = getattr(tool, 'inputs', {})
                    inputs_str = ', '.join(f"{k}='{v}'" for k, v in inputs.items()) if inputs else ""
                    print(f"  • {tool.name}({inputs_str})")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            import traceback
            print(f"\nERROR: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
