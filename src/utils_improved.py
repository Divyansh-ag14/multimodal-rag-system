"""Improved utility functions for the Food Recommendation RAG System.

This module provides enhanced versions of utility functions with:
- Comprehensive error handling
- Type hints
- Better documentation
- Input validation
- Retry logic
"""

import base64
import json
import logging
import re
import string
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import BedrockChat

from config import MAX_RETRIES, RETRY_DELAY

# Configure logging
logger = logging.getLogger(__name__)


def encode_image_from_upload(uploaded_file) -> str:
    """Encode an uploaded file object to base64 string.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Base64 encoded string of the image
        
    Raises:
        ValueError: If uploaded_file is None or empty
    """
    if uploaded_file is None:
        raise ValueError("Uploaded file cannot be None")
    
    try:
        return base64.b64encode(uploaded_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding uploaded image: {e}")
        raise


def encode_image(image_path: str | Path) -> str:
    """Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        IOError: If file cannot be read
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except IOError as e:
        logger.error(f"Error reading image file {image_path}: {e}")
        raise


def _retry_llm_call(func, *args, **kwargs):
    """Retry wrapper for LLM API calls with exponential backoff.
    
    Args:
        func: The LLM function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the function call
        
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"LLM call failed after {MAX_RETRIES} attempts: {e}")
    
    raise last_exception


def describe_input_image(encoded_image: str, llm: BedrockChat) -> str:
    """Describe an uploaded food image using the LLM.
    
    Args:
        encoded_image: Base64 encoded image string
        llm: The language model instance
        
    Returns:
        Description of the food item in the image
        
    Raises:
        ValueError: If encoded_image is empty
        Exception: If LLM call fails after retries
    """
    if not encoded_image or not encoded_image.strip():
        raise ValueError("Encoded image cannot be empty")
    
    messages = [
        SystemMessage(content="You are an AI assistant specializing in analyzing and describing food images. Your task is to provide a concise and accurate description of the food item."),
        HumanMessage(content=[
            {
                "type": "text",
                "text": """You are an assistant tasked with providing detailed descriptions of the dish in the image. Your descriptions should focus exclusively on the food and its ingredients, without mentioning any non-food items such as plates, utensils, or decorations. Follow these guidelines to create a detailed and accurate description in a short paragraph:
                        
                        Your short and concise description should suggest what the user is looking for with key search terms. Do not include any unnecessary terms which do not help in word similarity search.
                        Identify the dish, if not sure, mention how it looks, specify the cuisine, mention the key ingredients used in the dish."""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    
    try:
        response = _retry_llm_call(llm.invoke, messages)
        return response.content
    except Exception as e:
        logger.error(f"Error describing image: {e}")
        raise Exception(f"Failed to describe image: {str(e)}")


def enhance_search(
    user_input: str,
    llm: BedrockChat,
    conversation_history: Optional[str] = None,
    recent_recommendations: Optional[str] = None,
) -> str:
    """Enhance user input into a better search query using LLM.
    
    Args:
        user_input: The original user input query
        llm: The language model instance
        
    Returns:
        Enhanced search query string
        
    Raises:
        ValueError: If user_input is empty
        Exception: If LLM call fails
    """
    if not user_input or not user_input.strip():
        raise ValueError("User input cannot be empty")
    
    history_block = conversation_history or "No prior conversation."
    rec_block = recent_recommendations or "No recent recommendations to reference."

    hyde_prompt = [
        SystemMessage(content="You are an expert culinary assistant. Your task is to produce a search query description based on user input or preference."),
        HumanMessage(content=[
            {
                "type": "text",
                "text": f'''You are an expert culinary assistant tasked with generating a search query that helps recommends a variety of menu items based on user preferences. 
                    User Input:

                    {user_input}

                    Conversation History (latest first):
                    {history_block}

                    Recent Recommendations with metadata:
                    {rec_block}

                    Generate a Response That Includes Just the Key Unique Search Terms according to the user's preference, do not include unnecessary words that don't help search.
                    The search query may or may not contain the following parameters. For example you can include similar menu items as per the user preference if mentioned, if preferences is mentioned enhance and give key search terms based on preferences.
                    The goal is to either create a detailed query using specific information provided by the user or enhance the input to find similar preferences when the information is vague.
                    
                    Menu Items:

                    List different dishes or food items that resemble the user's input.
                    Mention their respective cuisines.

                    Cuisines:

                    Include a variety of cuisines that may match or complement the user's preferences.

                    Descriptions and Ingredients:
                    Provide a very short description of each dish.
                    List key ingredients for each dish.

                    Dietary Preferences:

                    Add any dietary preferences mentioned by the user, such as vegetarian, non-vegetarian, vegan, etc.

                    Nutritional Information:

                    Add important nutritional preference mentioned by the user if any such as high protein, number of calories, etc.
                    Mention serving sizes.
                    Dietary Warnings and Suggestions:

                    Avoid any dishes or ingredients containing any allergen mentioned by the user if any suggest menu items without these, and ensure all recommended items are free from this allergen.

    '''}])
    ]
    
    try:
        response = _retry_llm_call(llm.invoke, hyde_prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error enhancing search query: {e}")
        # Fallback to original input if enhancement fails
        logger.warning("Falling back to original user input")
        return user_input


def clean_text(text: str) -> str:
    """Clean and normalize input text for search.
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned and normalized text string
        
    Raises:
        TypeError: If text is not a string
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text).__name__}")
    
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Replace newline and tab characters with a space
    text = text.replace('\n', ' ').replace('\t', ' ')

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Convert to lowercase
    text = text.lower()

    return text


def relevance_checker(context: str, preference: str, llm: BedrockChat) -> str:
    """Check if a dish is relevant to user preferences.
    
    Args:
        context: Dish description/context
        preference: User preference string
        llm: The language model instance
        
    Returns:
        "Yes" or "No" string indicating relevance
        
    Raises:
        ValueError: If context or preference is empty
    """
    if not context or not context.strip():
        raise ValueError("Context cannot be empty")
    if not preference or not preference.strip():
        raise ValueError("Preference cannot be empty")
    
    relevance_prompt = [
        SystemMessage(content="You are a restaurant assistant specializing in helping customers find the food they want."),
        HumanMessage(content=[
            {
                "type": "text",
                "text": f'''Answer the question "Is this dish relevant to the user by comparing dish details and user preference?" in one word either Yes or No, based only on the following context. Say Yes only if it is relevant otherwise say No.
                        Context:
                        {context}
                        User Preference: {preference}
                        Answer:'''}])
    ]
    
    try:
        response = _retry_llm_call(llm.invoke, relevance_prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error checking relevance: {e}")
        # Default to "No" if check fails
        return "No"


def dish_summary(dish_description: str, preference: str, llm: BedrockChat) -> str:
    """Generate a summary of a dish highlighting user preferences.
    
    Args:
        dish_description: Full description of the dish
        preference: User preference string
        llm: The language model instance
        
    Returns:
        Two-line summary of the dish
        
    Raises:
        ValueError: If dish_description or preference is empty
    """
    if not dish_description or not dish_description.strip():
        raise ValueError("Dish description cannot be empty")
    if not preference or not preference.strip():
        raise ValueError("Preference cannot be empty")
    
    summary_prompt = [
        SystemMessage(content="You are a culinary assistant designed to summarize the dish description in accordance with the user preference."),
        HumanMessage(content=[
            {
                "type": "text",
                "text": f'''
 Your task is to create a very short two lines summary of the dish in a savoury manner by highlighting the user preference. The summary should suggest why the dish is perfect for the user as per their preference.
 The summary should include dish name, origin, ingredients and any other relevant information requested by the user in a friendly way. Do not include unnecessary sentences or additional comments like here is your response. Just give the summary description.


            Dish Description:

            {dish_description} 
            
            User Preference:

            {preference}
'''}])
    ]
    
    try:
        response = _retry_llm_call(llm.invoke, summary_prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating dish summary: {e}")
        # Fallback to basic description
        return f"Recommended dish: {dish_description[:100]}..."


def recommend_dishes_by_preference(
    search_results: List[Any], 
    original_input: str,
    llm: BedrockChat
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Recommend dishes based on user preferences from search results.
    
    Args:
        search_results: List of document results from FAISS search
        original_input: Original user input/preference
        llm: The language model instance
        
    Returns:
        Tuple of (list of dish summaries, dict of relevant images with metadata)
        
    Raises:
        ValueError: If search_results is empty or original_input is empty
    """
    if not search_results:
        raise ValueError("Search results cannot be empty")
    if not original_input or not original_input.strip():
        raise ValueError("Original input cannot be empty")
    
    from config import MAX_RECOMMENDATIONS
    
    relevant_images: Dict[str, Dict[str, Any]] = {}
    responses: List[str] = []
    
    count = 0
    for doc in search_results:
        if count >= MAX_RECOMMENDATIONS:
            break
            
        try:
            relevant = relevance_checker(doc.page_content, original_input, llm)
            if relevant.lower().strip() == 'yes':
                if 'image_path' in doc.metadata:
                    relevant_images[doc.metadata['image_path']] = doc.metadata
                    summary = dish_summary(doc.page_content, original_input, llm)
                    responses.append(summary)
                    count += 1
        except Exception as e:
            logger.warning(f"Error processing search result: {e}. Skipping...")
            continue
    
    if not responses:
        logger.warning("No relevant dishes found for user preferences")
    
    return responses, relevant_images


def assistant(
    context: str,
    user_input: str,
    llm: BedrockChat,
    conversation_history: Optional[str] = None,
    recent_recommendations: Optional[str] = None,
) -> str:
    """Generate assistant response based on context and user input.
    
    Args:
        context: Retrieved context from vector search
        user_input: User's query
        llm: The language model instance
        
    Returns:
        JSON string with recommendation and response
        
    Raises:
        ValueError: If user_input is empty
        Exception: If LLM call fails or JSON parsing fails
    """
    if not user_input or not user_input.strip():
        raise ValueError("User input cannot be empty")
    
    history_block = conversation_history or "No prior conversation."
    rec_block = recent_recommendations or "No previous dishes have been shared yet."

    assistant_prompt = [
        SystemMessage(content="You are a helpful and knowledgeable assistant capable of providing food recommendations and answering general queries."),
        HumanMessage(content=[
            {
                "type": "text",
                "text": f'''
  Your task is to engage users in natural, friendly dialogue to understand their preferences, dietary restrictions, and culinary interests.
Your goal is to summarize relevant food recommendations in a single sentence based on the user's inputs and the context if the user query is indicating that they want a recommendation. 
Otherwise you can simply request user to provide preferences such which cuisine or dish they would like based on the context given. Do not answer if you don't have relevant knowledge about the query.

Remember the context given is all the dishes we have.
Conversation History (latest first):
{history_block}

Recent Recommendations (include dish name, rating, price, serves, calories, summary):
{rec_block}

When users refer to "these" or compare options, analyze the recent recommendations block and reference concrete factors like rating, price, serving size, calories, dietary tags, or nutrition info to justify your answer. Prefer ranking or shortlists over generic statements.

User Input:

{user_input}


Context:
{context}


The output should be strictly formatted in JSON, with the following structure:
"recommendation": A field indicating whether a recommendation was made ("yes" or "no").
"response": A text field containing the chatbot's conversational response to the user's input, including recommendations or additional questions if necessary.
'''
            }])
    ]
    
    try:
        response = _retry_llm_call(llm.invoke, assistant_prompt)
        response_text = response.content.strip()
        
        # Try to parse JSON to validate
        try:
            json.loads(response_text)
        except json.JSONDecodeError:
            # If not valid JSON, wrap it
            logger.warning("LLM response is not valid JSON, wrapping it")
            response_text = json.dumps({
                "recommendation": "no",
                "response": response_text
            })
        
        return response_text
    except Exception as e:
        logger.error(f"Error generating assistant response: {e}")
        # Return a safe fallback response
        return json.dumps({
            "recommendation": "no",
            "response": "I apologize, but I'm having trouble processing your request right now. Please try again or rephrase your question."
        })


def format_conversation_history(
    user_messages: List[str],
    assistant_messages: List[str],
    window: int = 4
) -> str:
    """Format recent conversation turns for context injection."""
    if not user_messages or not assistant_messages:
        return ""
    turns = min(window, len(user_messages), len(assistant_messages))
    history_lines: List[str] = []
    for idx in range(1, turns + 1):
        history_lines.append(f"User: {user_messages[-idx]}")
        history_lines.append(f"Assistant: {assistant_messages[-idx]}")
    return "\n".join(history_lines)


def summarize_recommendations_for_context(
    recommendations: List[Dict[str, Any]],
    limit: int = 5
) -> str:
    """Summarize recent recommendation metadata for follow-up reasoning."""
    if not recommendations:
        return ""
    lines: List[str] = []
    for rec in recommendations[:limit]:
        line = (
            f"Dish: {rec.get('name', 'Unknown')} | "
            f"Restaurant: {rec.get('restaurant', 'N/A')} | "
            f"Rating: {rec.get('rating', 'N/A')} | "
            f"Price: {rec.get('price', 'N/A')} | "
            f"Serves: {rec.get('serves', 'N/A')} | "
            f"Calories: {rec.get('calories', 'N/A')} | "
            f"Diet: {rec.get('diet', 'N/A')} | "
            f"Summary: {rec.get('summary', '')}"
        )
        lines.append(line)
    return "\n".join(lines)

