"""Enhanced Streamlit app with improved UI for Food Recommendation RAG System.

This version includes:
- Modern, polished UI design
- Better chat interface
- Enhanced recommendation cards
- Improved image display
- Better visual hierarchy
- Sidebar for additional features
"""

import json
import logging
from pathlib import Path
from typing import Optional

import streamlit as st
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import boto3

from config import (
    AWS_REGION,
    BEDROCK_EMBEDDING_MODEL,
    BEDROCK_LLM_MODEL,
    LLM_MODEL_KWARGS,
    FAISS_INDEX_PATH,
    DATA_DIR,
    SEARCH_K,
    ALLOWED_IMAGE_TYPES,
    MAX_IMAGE_SIZE_MB,
    LOG_LEVEL,
)
from utils_improved import (
    describe_input_image,
    enhance_search,
    clean_text,
    recommend_dishes_by_preference,
    assistant,
    encode_image_from_upload,
    format_conversation_history,
    summarize_recommendations_for_context,
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Page configuration with custom styling
st.set_page_config(
    page_title="Food Recommendation Assistant",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Recommendation card styling */
    .recommendation-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    
    /* Image container */
    .dish-image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Metadata badges */
    .metadata-badge {
        display: inline-block;
        background: #f0f0f0;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        color: #333;
    }
    
    .metadata-badge.highlight {
        background: #667eea;
        color: white;
    }
    
    /* Input area styling */
    .input-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    /* Stats styling */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Loading spinner customization */
    .stSpinner > div {
        border-top-color: #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Chat history container */
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #999;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_services():
    """Initialize AWS Bedrock and FAISS services with error handling."""
    try:
        bedrock = boto3.client('bedrock-runtime', region_name=AWS_REGION)
        
        embeddings = BedrockEmbeddings(
            client=bedrock,
            model_id=BEDROCK_EMBEDDING_MODEL
        )
        
        llm = BedrockChat(
            client=bedrock,
            model_id=BEDROCK_LLM_MODEL,
            model_kwargs=LLM_MODEL_KWARGS,
        )
        
        if not FAISS_INDEX_PATH.exists():
            st.error(f"FAISS index not found at {FAISS_INDEX_PATH}")
            return None, None, None
        
        db = FAISS.load_local(
            str(FAISS_INDEX_PATH), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        logger.info("Services initialized successfully")
        return llm, embeddings, db
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        st.error(f"Failed to initialize services: {str(e)}")
        return None, None, None


def validate_image_upload(uploaded_file) -> tuple:
    """Validate uploaded image file."""
    if uploaded_file is None:
        return True, None
    
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext not in ALLOWED_IMAGE_TYPES:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}"
    
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_IMAGE_SIZE_MB:
        return False, f"File too large. Maximum: {MAX_IMAGE_SIZE_MB}MB"
    
    return True, None


def render_recommendation_card(rec_text: str, metadata: dict, image_path: str, show_details: bool = True):
    """Render a beautiful recommendation card.
    
    Args:
        rec_text: Recommendation text/description
        metadata: Dish metadata dictionary
        image_path: Path to dish image
        show_details: Whether to show detailed metadata (nutrition, ingredients, etc.)
    """
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="recommendation-card">
            <h3 style="color: #667eea; margin-top: 0;">{rec_text}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Basic info (always shown)
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"**🍽️ Name:** {metadata.get('menu_item_name', 'N/A')}")
            st.markdown(f"**🏪 Restaurant:** {metadata.get('restaurant_name', 'N/A')}")
            st.markdown(f"**⭐ Rating:** {metadata.get('average_rating', 'N/A')}")
        
        with col_b:
            st.markdown(f"**💰 Price:** USD {metadata.get('price', 'N/A')}")
            st.markdown(f"**👥 Serves:** {metadata.get('serves', 'N/A')}")
            st.markdown(f"**🔥 Calories:** {metadata.get('calories', 'N/A')}")
        
        # Detailed metadata (only shown if show_details is True)
        if show_details:
            # Nutrition info
            nutrition = metadata.get('nutrition', 'N/A')
            if nutrition != 'N/A':
                st.markdown(f"**📊 Nutrition:** {nutrition}")
            
            # Ingredients (if available)
            ingredients = metadata.get('ingredients', '')
            if ingredients:
                with st.expander("📝 Ingredients"):
                    st.write(ingredients)
            
            # Dietary warnings (if available)
            dietary_warnings = metadata.get('dietary_warnings', '')
            if dietary_warnings:
                st.warning(f"⚠️ **Dietary Info:** {dietary_warnings}")
        
        # Dietary badge (always shown, but more detailed if show_details)
        dietary = metadata.get('vegetarian_or_nonveg', '')
        if dietary:
            badge_color = "#4caf50" if "Vegetarian" in dietary else "#ff9800"
            st.markdown(f"""
            <span style="background: {badge_color}; color: white; padding: 0.3rem 0.8rem; 
            border-radius: 20px; font-size: 0.85rem; display: inline-block; margin-top: 0.5rem;">
            {dietary}
            </span>
            """, unsafe_allow_html=True)
    
    with col2:
        image_full_path = DATA_DIR / image_path
        if image_full_path.exists():
            st.markdown('<div class="dish-image-container">', unsafe_allow_html=True)
            st.image(str(image_full_path), use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning(f"Image not found: {image_path}")


def main():
    """Main application function."""
    # Initialize session state
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'images' not in st.session_state:
        st.session_state['images'] = []
    if 'assistant_response' not in st.session_state:
        st.session_state['assistant_response'] = []
    if 'recent_recommendations' not in st.session_state:
        st.session_state['recent_recommendations'] = []
    if 'stats' not in st.session_state:
        st.session_state['stats'] = {
            'total_queries': 0,
            'recommendations_given': 0,
            'images_uploaded': 0
        }
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Statistics")
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-number">{st.session_state['stats']['total_queries']}</p>
            <p class="stat-label">Total Queries</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-number">{st.session_state['stats']['recommendations_given']}</p>
            <p class="stat-label">Recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **Food Recommendation Assistant**
        
        Ask me about:
        - Available cuisines
        - Specific dishes
        - Dietary preferences
        - Nutritional information
        
        Or upload an image to find similar dishes!
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_details = st.checkbox("Show detailed metadata", value=True)
    
    # Initialize services
    llm, embeddings, db = initialize_services()
    
    if llm is None or db is None:
        st.stop()
    
    # Enhanced header
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ Food Recommendation Assistant</h1>
        <p>Discover delicious dishes tailored to your preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input section with better styling
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Two-column layout for input
    input_col, button_col = st.columns([4, 1])
    
    with input_col:
        user_input = st.text_input(
            "💬 Ask me anything about food...",
            key="input",
            placeholder="e.g., 'What cuisines do you have?' or 'Show me Italian dishes'",
            label_visibility="collapsed"
        )
    
    with button_col:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        send_button = st.button("Send 🚀", type="primary", use_container_width=True)
    
    # Image upload with better styling
    uploaded_image = st.file_uploader(
        "📷 Upload an image to find similar dishes",
        type=ALLOWED_IMAGE_TYPES,
        help=f"Supported formats: {', '.join(ALLOWED_IMAGE_TYPES).upper()} | Max size: {MAX_IMAGE_SIZE_MB}MB"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    original_input = user_input.strip() if user_input else ""
    
    # Process input
    if send_button and (user_input or uploaded_image):
        st.session_state['stats']['total_queries'] += 1
        
        # Validate image
        if uploaded_image:
            is_valid, error_msg = validate_image_upload(uploaded_image)
            if not is_valid:
                st.error(f"❌ {error_msg}")
                st.stop()
            
            st.session_state['stats']['images_uploaded'] += 1
            
            try:
                with st.spinner("🔍 Analyzing your image..."):
                    encoded_image = encode_image_from_upload(uploaded_image)
                    st.session_state.images.append(encoded_image)
                    image_description = describe_input_image(encoded_image, llm)
                    
                    if user_input:
                        user_input = f'I am looking for this dish, recommend similar dishes: {user_input} {image_description}'
                    else:
                        user_input = f'I am looking for this dish, recommend similar dishes: {image_description}'
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.stop()
        
        if not user_input or not user_input.strip():
            st.warning("⚠️ Please enter a question or upload an image.")
            st.stop()
        
        st.session_state.past.append(user_input)
        conversation_history = format_conversation_history(
            st.session_state['past'],
            st.session_state['assistant_response']
        )
        recommendation_context = summarize_recommendations_for_context(
            st.session_state['recent_recommendations']
        )
        
        try:
            # Enhanced search
            with st.spinner("🧠 Understanding your request..."):
                enhanced_search_query = enhance_search(
                    user_input,
                    llm,
                    conversation_history=conversation_history,
                    recent_recommendations=recommendation_context
                )
                enhanced_search_query = clean_text(enhanced_search_query)
            
            # Search
            with st.spinner("🔎 Searching our menu..."):
                try:
                    results = db.similarity_search(user_input, k=SEARCH_K)
                except Exception as e:
                    st.error(f"❌ Search error: {str(e)}")
                    st.stop()
            
            if not results:
                st.warning("📭 No results found. Try a different query!")
                st.session_state.generated.append((
                    "I couldn't find any matching dishes. Please try rephrasing your question.",
                    []
                ))
                st.stop()
            
            # Compile context
            context = "\n\n".join([doc.page_content for doc in results])
            
            # Generate response
            with st.spinner("💭 Generating response..."):
                try:
                    chatbot_response = assistant(
                        context,
                        user_input,
                        llm,
                        conversation_history=conversation_history,
                        recent_recommendations=recommendation_context
                    )
                    chatbot_response_dict = json.loads(chatbot_response)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.stop()
            
            recommendation = chatbot_response_dict.get('recommendation', 'no')
            response = chatbot_response_dict.get('response', '')
            
            st.session_state.assistant_response.append(response)
            
            # Generate recommendations
            if recommendation == 'yes':
                with st.spinner("✨ Finding perfect matches..."):
                    try:
                        rec_response, relevant_images = recommend_dishes_by_preference(
                            results, 
                            original_input if original_input else user_input, 
                            llm
                        )
                        
                        if rec_response and relevant_images:
                            st.session_state['stats']['recommendations_given'] += len(rec_response)
                            st.session_state.generated.append((rec_response, relevant_images))
                            image_keys = list(relevant_images.keys())
                            recent_entries = []
                            for idx, summary in enumerate(rec_response):
                                image_path = image_keys[idx] if idx < len(image_keys) else None
                                metadata = relevant_images.get(image_path, {}) if image_path else {}
                                recent_entries.append({
                                    "name": metadata.get('menu_item_name', 'Unknown dish'),
                                    "restaurant": metadata.get('restaurant_name', 'N/A'),
                                    "rating": metadata.get('average_rating', 'N/A'),
                                    "price": metadata.get('price', 'N/A'),
                                    "serves": metadata.get('serves', 'N/A'),
                                    "calories": metadata.get('calories', 'N/A'),
                                    "diet": metadata.get('vegetarian_or_nonveg', 'N/A'),
                                    "summary": summary
                                })
                            if recent_entries:
                                st.session_state['recent_recommendations'] = (
                                    recent_entries + st.session_state['recent_recommendations']
                                )[:10]
                        else:
                            st.session_state.generated.append((response, {}))
                    except Exception as e:
                        st.session_state.generated.append((response, {}))
            else:
                st.session_state.generated.append((response, []))
                
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            st.error(f"❌ An error occurred: {str(e)}")
            st.session_state.generated.append((
                "I apologize, but I encountered an error. Please try again.",
                []
            ))
    
    # Display chat history with enhanced UI
    if st.session_state['generated']:
        st.markdown("---")
        st.markdown("### 💬 Conversation History")
        
        # Chat messages
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            # User message
            if i < len(st.session_state['past']):
                with st.chat_message("user", avatar="👤"):
                    st.write(st.session_state["past"][i])
            
            # Bot response
            response, images = st.session_state["generated"][i]
            
            if isinstance(response, list) and images:  # Recommendations
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown("### Listings:")
                    
                    for j, rec in enumerate(response):
                        if j < len(list(images.keys())):
                            image_path = list(images.keys())[j]
                            metadata = images[image_path]
                            
                            render_recommendation_card(rec, metadata, image_path, show_details)
                            
                            if j < len(response) - 1:
                                st.markdown("---")
            else:  # Text response
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(f"{response}")
        
    else:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🍽️</div>
            <h3>Welcome! Start a conversation</h3>
            <p>Ask me about cuisines, dishes, or upload an image to get started!</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

