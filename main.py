from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
import json

# Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# ✅ SECURE: API key comes from environment variable (set in Render dashboard)
# Never hardcode API keys in code!
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Define request model - what the API expects to receive
class CommentRequest(BaseModel):
    comment: str

# Define response model - what the API promises to return
class SentimentResponse(BaseModel):
    sentiment: str  # Will be "positive", "negative", or "neutral"
    rating: int     # Will be 1, 2, 3, 4, or 5

# JSON schema for OpenAI structured output
# This forces the AI to return EXACTLY this format
response_schema = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]  # Only these three values allowed
        },
        "rating": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5  # Only 1 through 5 allowed
        }
    },
    "required": ["sentiment", "rating"],  # Both fields must be present
    "additionalProperties": False  # No extra fields allowed!
}

def analyze_sentiment(comment_text: str) -> dict:
    """
    Call OpenAI with structured output to analyze sentiment.
    This function does the actual AI work.
    """
    try:
        # Make the API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # The model specified in assignment
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant. Analyze the comment and return the sentiment and a rating."},
                {"role": "user", "content": f"Comment: {comment_text}"}
            ],
            # THIS IS THE KEY PART FOR STRUCTURED OUTPUT
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_analysis",
                    "schema": response_schema,
                    "strict": True  # Enforce the schema strictly
                }
            },
            temperature=0.1  # Low temperature = more consistent results
        )
        
        # Parse the JSON response from OpenAI
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        # If anything goes wrong with the API call, raise an exception
        raise Exception(f"OpenAI API error: {str(e)}")

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    """
    Main endpoint for sentiment analysis.
    Receives a comment and returns structured sentiment data.
    """
    # Validate input - check for empty comments
    if not request.comment or len(request.comment.strip()) == 0:
        raise HTTPException(
            status_code=400, 
            detail="Comment cannot be empty"
        )
    
    try:
        # Call the AI function
        result = analyze_sentiment(request.comment)
        
        # Return the structured result
        # FastAPI automatically converts this to JSON
        return result
        
    except Exception as e:
        # Handle errors gracefully
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint - shows API is running"""
    return {
        "message": "Sentiment Analysis API is running!",
        "endpoints": {
            "POST /comment": "Analyze sentiment of a comment",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "service": "sentiment-analysis-api"}
