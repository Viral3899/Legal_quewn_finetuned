# streamlit_app.py
"""Streamlit web application for the Legal LLM."""

import streamlit as st
import os
import json
import time
from datetime import datetime
from typing import List, Dict
import pandas as pd

# Import our modules
from inference_engine import LegalQuestionAnswerer
from evaluator import LegalModelEvaluator
from data_loader import DataLoader
from fine_tuner import LegalLMFineTuner
from project_structure import MODELS_DIR, LOGS_DIR, DATA_DIR

# Page configuration
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stSuccess {
        background-color: #dcfce7;
        border: 1px solid #16a34a;
    }
    .stError {
        background-color: #fef2f2;
        border: 1px solid #dc2626;
    }
    .stWarning {
        background-color: #fefce8;
        border: 1px solid #ca8a04;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #f0f9ff;
        border-left: 4px solid #0ea5e9;
    }
    .assistant-message {
        background-color: #f8fafc;
        border-left: 4px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)

class LegalAIApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        self.qa_system = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'current_model_path' not in st.session_state:
            st.session_state.current_model_path = None
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = None
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        if not os.path.exists(MODELS_DIR):
            return []
        
        models = []
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            if os.path.isdir(item_path):
                models.append(item)
        
        return models
    
    def load_model(self, model_path: str = None):
        """Load the selected model."""
        try:
            with st.spinner("Loading model... This may take a few minutes."):
                if model_path:
                    full_path = os.path.join(MODELS_DIR, model_path)
                else:
                    full_path = None
                
                self.qa_system = LegalQuestionAnswerer(full_path)
                self.qa_system.load()
                
                st.session_state.model_loaded = True
                st.session_state.current_model_path = model_path
                
                st.success("✅ Model loaded successfully!")
                return True
                
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def sidebar(self):
        """Create sidebar with model selection and controls."""
        st.sidebar.title("⚖️ Legal AI Settings")
        
        # Model selection
        st.sidebar.subheader("Model Selection")
        available_models = self.get_available_models()
        
        if available_models:
            selected_model = st.sidebar.selectbox(
                "Choose a model:",
                options=[""] + available_models,
                index=0
            )
            
            if selected_model and selected_model != st.session_state.current_model_path:
                if st.sidebar.button("Load Model"):
                    self.load_model(selected_model)
        else:
            st.sidebar.warning("No trained models found. Please train a model first.")
            if st.sidebar.button("Use Base Model (Demo)"):
                self.load_model()
        
        # Model status
        st.sidebar.subheader("Model Status")
        if st.session_state.model_loaded:
            st.sidebar.success("✅ Model is loaded")
            st.sidebar.info(f"Current model: {st.session_state.current_model_path or 'Base model'}")
        else:
            st.sidebar.error("❌ No model loaded")
        
        # Chat controls
        st.sidebar.subheader("Chat Controls")
        if st.sidebar.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        
        # Export chat
        if st.session_state.chat_history:
            chat_data = json.dumps(st.session_state.chat_history, indent=2)
            st.sidebar.download_button(
                label="📥 Download Chat History",
                data=chat_data,
                file_name=f"legal_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def chat_interface(self):
        """Main chat interface."""
        st.markdown('<h1 class="main-header">⚖️ Legal AI Assistant</h1>', unsafe_allow_html=True)
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load a model from the sidebar to start chatting.")
            return
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                    <small style="color: #6b7280; float: right;">{message["timestamp"]}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Legal AI:</strong> {message["content"]}
                    <small style="color: #6b7280; float: right;">{message["timestamp"]}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_question = st.text_area(
                    "Ask a legal question:",
                    placeholder="e.g., What is Section 302 of IPC?",
                    height=100,
                    key="user_input"
                )
            
            with col2:
                submit_button = st.form_submit_button("Send 📤", use_container_width=True)
        
        if submit_button and user_question.strip():
            # Add user message to history
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question,
                "timestamp": timestamp
            })
            
            # Generate response
            with st.spinner("🤔 Thinking..."):
                try:
                    result = self.qa_system.answer(user_question)
                    response = result["answer"]
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": timestamp
                    })
                    
                    # Show success toast
                    st.toast("✅ Response generated!", icon="✅")
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": timestamp
                    })
                    st.toast("❌ Error generating response", icon="❌")
            
            # Rerun to show the new messages
            st.rerun()
    
    def training_interface(self):
        """Interface for training new models."""
        st.title("🔧 Model Training")
        
        st.info("Train a new legal LLM model using the Indian Legal Corpus dataset.")
        
        # Training configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Configuration")
            dataset_source = st.radio(
                "Data source:",
                ["HuggingFace", "Upload CSV"],
                help="Choose whether to use the default HuggingFace dataset or upload your own CSV"
            )
            
            if dataset_source == "Upload CSV":
                uploaded_file = st.file_uploader(
                    "Upload CSV file",
                    type=['csv'],
                    help="CSV should have columns for questions and answers"
                )
        
        with col2:
            st.subheader("Training Parameters")
            num_epochs = st.slider("Number of epochs", 1, 10, 3)
            batch_size = st.slider("Batch size", 1, 16, 4)
            learning_rate = st.select_slider(
                "Learning rate",
                options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4],
                value=2e-4,
                format_func=lambda x: f"{x:.0e}"
            )
        
        # Start training
        if st.button("🚀 Start Training", type="primary"):
            if dataset_source == "Upload CSV" and uploaded_file is None:
                st.error("Please upload a CSV file first.")
                return
            
            # Show training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                with st.spinner("Initializing training..."):
                    # Initialize components
                    data_loader = DataLoader()
                    fine_tuner = LegalLMFineTuner()
                    
                    # Update training config
                    fine_tuner.training_config.num_train_epochs = num_epochs
                    fine_tuner.training_config.per_device_train_batch_size = batch_size
                    fine_tuner.training_config.learning_rate = learning_rate
                    
                    status_text.text("Loading and preparing data...")
                    progress_bar.progress(20)
                    
                    # Load data
                    if dataset_source == "HuggingFace":
                        dataset_dict = data_loader.load_and_prepare_data(source="huggingface")
                    else:
                        # Save uploaded file and load
                        csv_path = os.path.join(DATA_DIR, "uploaded_data.csv")
                        with open(csv_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        dataset_dict = data_loader.load_and_prepare_data(source="csv")
                    
                    progress_bar.progress(40)
                    status_text.text("Starting model training...")
                    
                    # Train model
                    model_path = fine_tuner.train(dataset_dict)
                    
                    progress_bar.progress(80)
                    status_text.text("Merging and saving final model...")
                    
                    # Merge and save
                    merged_path = fine_tuner.merge_and_save_model(model_path)
                    
                    progress_bar.progress(100)
                    status_text.text("Training completed!")
                    
                    st.success(f"✅ Training completed! Model saved to: {merged_path}")
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                status_text.text("Training failed.")
    
    def evaluation_interface(self):
        """Interface for model evaluation."""
        st.title("📊 Model Evaluation")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load a model first to run evaluation.")
            return
        
        st.info("Evaluate the performance of your fine-tuned model on test data.")
        
        # Evaluation parameters
        col1, col2 = st.columns(2)
        
        with col1:
            num_samples = st.slider(
                "Number of test samples",
                min_value=5,
                max_value=50,
                value=10,
                help="Number of samples to evaluate"
            )
        
        with col2:
            st.subheader("Evaluation Metrics")
            st.write("- Answer Length")
            st.write("- Legal Terms Coverage")
            st.write("- Relevance Score")
            st.write("- Success Rate")
        
        # Run evaluation
        if st.button("🔍 Run Evaluation", type="primary"):
            with st.spinner(f"Evaluating model on {num_samples} samples..."):
                try:
                    evaluator = LegalModelEvaluator(
                        os.path.join(MODELS_DIR, st.session_state.current_model_path)
                        if st.session_state.current_model_path else None
                    )
                    
                    results = evaluator.evaluate_model(num_samples)
                    st.session_state.evaluation_results = results
                    
                    st.success("✅ Evaluation completed!")
                    
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
                    return
        
        # Display results
        if st.session_state.evaluation_results:
            st.subheader("📈 Evaluation Results")
            
            summary = st.session_state.evaluation_results["summary"]
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Success Rate", f"{summary.get('success_rate', 0):.1f}%")
            with col2:
                st.metric("Legal Terms Coverage", f"{summary.get('legal_terms_coverage', 0):.1f}%")
            with col3:
                st.metric("Relevance Score", f"{summary.get('relevance_score', 0):.1f}%")
            with col4:
                st.metric("Avg Answer Length", f"{summary.get('average_answer_length', 0):.1f} words")
            
            # Sample results
            st.subheader("📝 Sample Results")
            detailed_results = st.session_state.evaluation_results["detailed_results"]
            
            for i, result in enumerate(detailed_results[:3], 1):
                with st.expander(f"Sample {i}: {result['question'][:50]}..."):
                    st.write("**Question:**", result['question'])
                    st.write("**Generated Answer:**", result['generated_answer'])
                    st.write("**Contains Legal Terms:**", "✅" if result['contains_legal_terms'] else "❌")
                    st.write("**Is Relevant:**", "✅" if result['is_relevant'] else "❌")
            
            # Download results
            results_json = json.dumps(st.session_state.evaluation_results, indent=2)
            st.download_button(
                label="📥 Download Full Results",
                data=results_json,
                file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def run(self):
        """Main application runner."""
        # Create sidebar
        self.sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔧 Training", "📊 Evaluation"])
        
        with tab1:
            self.chat_interface()
        
        with tab2:
            self.training_interface()
        
        with tab3:
            self.evaluation_interface()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #6b7280;'>"
            "Legal AI Assistant | Built with Streamlit & Qwen"
            "</div>",
            unsafe_allow_html=True
        )

def main():
    """Main function to run the Streamlit app."""
    app = LegalAIApp()
    app.run()

if __name__ == "__main__":
    main()