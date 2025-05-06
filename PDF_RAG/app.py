"""
Gradio-based web UI for querying uploaded PDF documents using RAG.

This module creates a two-panel interface:
- Left: file upload, document processing, and status updates
- Right: chatbot for Q&A over processed documents

Includes event bindings to handle file ingestion, clearing documents/chat,
and querying the document corpus via a RAG pipeline.

Logging and error handling are added for robustness.
"""

import gradio as gr
from utils.pdf_utils import process_uploaded_files
from utils.rag_utils import rag_chat, clear_documents
from utils.logger import get_logger

logger = get_logger(__name__)

def create_interface():
    """
    Build and return the Gradio Blocks interface.

    UI Layout:
    - Left column:
        * File upload input for multiple PDFs
        * Buttons for processing PDFs and clearing documents
        * Status textbox showing processing results
    - Right column:
        * Chatbot display for conversation history
        * Textbox and button to send user queries
        * Button to clear chat history

    Returns:
        gr.Blocks: Configured Gradio Blocks interface.
    """
    try:
        with gr.Blocks(css="""
            /* Outer chat container scrolls once */
            .chat-container {
                height: 600px !important;
                overflow-y: auto !important;
            }
            /* Disable inner pane scroll */
            .chat-container > div {
                overflow-y: visible !important;
            }
            .document-status { min-height: 100px; }
        """) as demo:

            gr.Markdown("# 📚 PDF Document Query Application")

            with gr.Row():
                with gr.Column(scale=1):
                    upload       = gr.File(
                        label="Upload PDF Documents",
                        file_types=[".pdf"],
                        file_count="multiple"
                    )
                    btn_process  = gr.Button("Process PDFs")
                    btn_cleardoc = gr.Button("Clear Documents")
                    status       = gr.Textbox(
                        label="Document Status",
                        lines=4,
                        interactive=False,
                        elem_classes=["document-status"]
                    )

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        elem_classes=["chat-container"],
                        avatar_images=["static/user.png", "static/bot.png"],
                        type="messages"   # Markdown timestamps render here
                    )
                    msg      = gr.Textbox(label="Ask a question", lines=2)
                    btn_send = gr.Button("Send")
                    btn_clr  = gr.Button("Clear Chat")

            # === Event Bindings with error handling ===

            def safe_process(files):
                try:
                    return process_uploaded_files(files)
                except Exception as e:
                    logger.error("Error in process_uploaded_files: %s", e, exc_info=True)
                    return f"Error processing PDFs: {e}"

            def safe_clear_docs():
                try:
                    return clear_documents()
                except Exception as e:
                    logger.error("Error in clear_documents: %s", e, exc_info=True)
                    return f"Error clearing documents: {e}"

            def safe_rag_chat(message, history):
                try:
                    return rag_chat(message, history)
                except Exception as e:
                    logger.error("Error in rag_chat: %s", e, exc_info=True)
                    # Return history with an error message
                    history.append({"role":"assistant", "content":f"Error: {e}"})
                    return history

            btn_process.click(safe_process, [upload], [status])
            btn_cleardoc.click(safe_clear_docs, [], [status])
            btn_send.click(safe_rag_chat, [msg, chatbot], [chatbot]).then(lambda: "", None, [msg])
            msg.submit(safe_rag_chat, [msg, chatbot], [chatbot]).then(lambda: "", None, [msg])
            btn_clr.click(lambda: [], None, [chatbot])

            gr.Markdown("""
            **How to use**  
            1. Upload one or more PDFs  
            2. Click **Process PDFs**  
            3. Ask questions below  
            4. See answers drawn from your documents  
            
            Up to 5 exchanges (10 messages) are shown; older messages auto‑trim.
            """)

        return demo

    except Exception as e:
        logger.critical("Failed to build Gradio interface: %s", e, exc_info=True)
        raise

if __name__ == "__main__":
    try:
        demo = create_interface()
        demo.launch(share=False)
    except Exception as e:
        logger.critical("Application failed to start: %s", e, exc_info=True)
