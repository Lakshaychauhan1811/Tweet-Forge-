from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, validator
from fastapi.encoders import jsonable_encoder
import os
import requests
from dotenv import load_dotenv
from shutil import which
from typing import Optional, List
import logging
import time
from datetime import datetime
import secrets
import hashlib
import base64
import urllib.parse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_groq import ChatGroq
# Serper is used for web search; we call its HTTP API directly

try:
    # OpenAI via LangChain
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

try:
    # Ollama via LangChain Community
    from langchain_community.chat_models import ChatOllama
except Exception:  # pragma: no cover
    ChatOllama = None  # type: ignore


load_dotenv()



app = FastAPI(title="TweetForge - Marketing Tweet Generator", version="1.0.0")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "tweetforge_secure_session_key"),
    max_age=3600,
    same_site="lax",
    https_only=False
)


# CORS for local dev and simple deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateTweetRequest(BaseModel):
    product_name: str = Field(..., description="Product or brand name")
    audience: str = Field(..., description="Target audience description")
    tone: str = Field("engaging", description="Tone/voice e.g., playful, professional")
    key_benefits: List[str] = Field(default_factory=list, description="List of key benefits")
    call_to_action: Optional[str] = Field(None, description="CTA to include")
    hashtags_count: int = Field(2, ge=0, le=6, description="Number of hashtags to append")
    use_emojis: bool = Field(True, description="Whether to include emojis")


class GenerateTweetResponse(BaseModel):
    tweet: str


class HashtagSuggestRequest(BaseModel):
    text: str = Field(..., description="Topic, prompt, or YouTube URL context")
    max_hashtags: int = Field(5, ge=1, le=10)


class HashtagSuggestResponse(BaseModel):
    hashtags: List[str]


def get_llm():
    provider = os.getenv("MODEL_PROVIDER", "groq").lower()
    # Use a currently supported default model; override via MODEL_NAME
    model_name = os.getenv("MODEL_NAME", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in environment.")
        return ChatGroq(model=model_name, temperature=0.7, max_tokens=None)

    if provider == "ollama":
        if ChatOllama is None:
            raise RuntimeError("langchain-community not installed. Add it to requirements.")
        return ChatOllama(model=model_name, temperature=0.7)

    raise RuntimeError(f"Unsupported MODEL_PROVIDER: {provider}")


    
def _clean_env(v: str | None) -> str | None:
    if v is None:
        return None
    cleaned = v.strip().strip('"').strip("'")
    return cleaned


def _serper_search(query: str, tbs: str | None = None, num: int = 10) -> dict:
    """Minimal Serper.dev search helper.
    Expects SERPER_API_KEY in env. Returns parsed JSON dict or {}.
    """
    api_key = _clean_env(os.getenv("SERPER_API_KEY"))
    if not api_key:
        return {}
    try:
        payload = {"q": query}
        if tbs:
            payload["tbs"] = tbs  # e.g. 'qdr:d' past day, 'qdr:w' past week
        if num:
            payload["num"] = num
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        if resp.ok:
            return resp.json() or {}
        return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def fetch_research_snippets(query: str, num_results: int = 3) -> str:
    """Fetch research snippets using SerpApi first, then Groq as fallback."""
    # Use the existing logger
    import logging
    logger = logging.getLogger(__name__)
    
    # First try with Serper
    provider_used = "none"
    try:
        serper_key = _clean_env(os.getenv("SERPER_API_KEY"))
        if serper_key:
            try:
                masked = serper_key[:4] + "***" + serper_key[-4:]
                logger.info(f"SERPER key loaded (masked, len={len(serper_key)}): {masked}")
            except Exception:
                pass
            logger.info(f"🔍 Using Serper for research query: {query} (freshness=day)")
            results = _serper_search(query, tbs="qdr:d")
            if isinstance(results, dict) and results.get("error"):
                logger.warning(f"❌ Serper error: {results.get('error')}")
            logger.info(f"RAW SERPER RESPONSE: {results}")
            # Extract titles and snippets; Serper uses 'organic'
            organic = []
            if isinstance(results, dict):
                organic = (
                    results.get("organic")
                    or []
                )
            # If nothing for past day, broaden to week, then all
            if not organic:
                logger.info("No day-fresh results; retrying with past week (qdr:w)")
                results = _serper_search(query, tbs="qdr:w")
                organic = (results.get("organic") if isinstance(results, dict) else []) or []
            if not organic:
                logger.info("No week-fresh results; retrying with no freshness filter")
                results = _serper_search(query, tbs=None)
                organic = (results.get("organic") if isinstance(results, dict) else []) or []
            items = []
            for item in organic[:num_results]:
                title = item.get("title", "")
                snippet = item.get("snippet", item.get("snippet_highlighted_words", ""))
                if isinstance(snippet, list):
                    snippet = "; ".join(snippet)
                source = item.get("link", item.get("url", ""))
                if title or snippet:
                    items.append(f"- {title}: {snippet} (src: {source})")
            if items:
                logger.info(f"✅ Serper returned {len(items)} research snippets")
                provider_used = "serper"
                logger.info(f"Research provider used: {provider_used}")
                return "\n".join(items)
            else:
                logger.warning("⚠️ Serper returned no organic results")
        else:
            logger.warning("⚠️ SERPER_API_KEY not found - using Groq fallback")
    except Exception as e:
        logger.warning(f"❌ Serper error: {e} - falling back to Groq")
    
    # Fallback to Groq if SerpApi failed or returned no results
    try:
        logger.info(f"Falling back to Groq for research query: {query}")
        llm = get_llm()
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Provide 3 brief research points about: {query}. Format as bullet points.",
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"query": query})
        logger.info("Groq fallback completed successfully")
        provider_used = "groq"
        logger.info(f"Research provider used: {provider_used}")
        return result
    except Exception as e:
        logger.warning(f"Groq fallback error: {e}")
        return ""  # Return empty string if all methods fail


def build_video_context(video_url: str) -> str:
    """Try to obtain lightweight context (title/snippet) for a YouTube URL.
    Order:
      1) YouTube oEmbed (no API key) for title/author
      2) Serper (if available) for extra snippets
      3) Groq as fallback if Serper fails
    """
    import logging
    logger = logging.getLogger(__name__)
    lines: list[str] = []
    # 1) oEmbed
    try:
        oembed = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": video_url, "format": "json"}, timeout=6
        )
        if oembed.ok:
            data = oembed.json()
            title = data.get("title", "")
            author = data.get("author_name", "")
            if title or author:
                lines.append(f"- Title: {title}")
                if author:
                    lines.append(f"- Channel: {author}")
    except Exception:
        pass
    
    # 2) Serper (if available)
    try:
        serper_key = _clean_env(os.getenv("SERPER_API_KEY"))
        if serper_key:
            try:
                masked = serper_key[:4] + "***" + serper_key[-4:]
                logger.info(f"SERPER key loaded (masked, len={len(serper_key)}): {masked}")
            except Exception:
                pass
            logger.info(f"🔍 Using Serper for video context: {video_url} (freshness=day)")
            results = _serper_search(video_url, tbs="qdr:d")
            if isinstance(results, dict) and results.get("error"):
                logger.warning(f"Serper error in video context: {results.get('error')}")
            organic = []
            if isinstance(results, dict):
                organic = results.get("organic") or []
            if not organic:
                logger.info("No day-fresh results; retrying with past week (qdr:w)")
                results = _serper_search(video_url, tbs="qdr:w")
                if isinstance(results, dict):
                    organic = results.get("organic") or []
            if not organic:
                logger.info("No week-fresh results; retrying with no freshness filter")
                results = _serper_search(f"{video_url} YouTube")
                if isinstance(results, dict) and results.get("error"):
                    logger.warning(f"Serper error in fallback video context: {results.get('error')}")
                if isinstance(results, dict):
                    organic = results.get("organic") or []
            for item in organic[:3]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                if title or snippet:
                    lines.append(f"- {title}: {snippet}")
            if lines:
                logger.info(f"Video context provider: oembed+serper (lines={len(lines)})")
                return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Serper error in video context: {e}")
    
    def create_search_query(topic: str) -> str:
        """Uses an LLM to distill a topic into a clean search query."""
        logger = logging.getLogger(__name__)
        logger.info(f"Refining topic into search query: {topic}")
        try:
            llm = get_llm() # Your existing function to get the Groq LLM
            prompt = PromptTemplate(
                input_variables=["topic"],
                template=(
                    "You are an expert at creating effective search engine queries. "
                    "Distill the core keywords from the following user request. "
                    "Return ONLY the search query, with no explanation.\n\n"
                    "Request: {topic}"
                ),
            )
            chain = prompt | llm | StrOutputParser()
            search_query = chain.invoke({"topic": topic})
            logger.info(f"Refined search query: {search_query}")
            return search_query.strip()
        except Exception as e:
            logger.warning(f"Could not refine search query: {e}. Using original topic.")
            return topic # Fallback to the original topic if refinement fails

    # 3) Groq fallback
    if not lines and os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq
            
            groq_llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.5,
                max_tokens=500
            )
            
            video_prompt = PromptTemplate(
                input_variables=["url"],
                template=(
                    "Based on this YouTube video URL: {url}\n"
                    "Provide 2-3 key points about what this video might be about. "
                    "Format as bullet points. If you don't have specific information, "
                    "make educated guesses based on the URL structure."
                )
            )
            
            video_chain = video_prompt | groq_llm | StrOutputParser()
            video_info = video_chain.invoke({"url": video_url})
            
            if video_info:
                lines.append(video_info)
                logger.info(f"Video context provider: groq_fallback (lines={len(lines)})")
        except Exception as e:
            logger.warning(f"Groq API fallback error in video context: {e}")
    
    if not lines:
        logger.info("Video context provider: none")
    return "\n".join(lines)


def build_chain():
    system_prompt = (
        "You are an expert social media marketer. Craft high-converting Twitter/X posts. "
        "Constraints: 1) 280 characters max. 2) Clear hook upfront. 3) Concrete benefit. "
        "4) Natural voice (no hashtags mid-sentence). 5) End with short CTA and hashtags. "
        "6) Avoid fluff, avoid generic buzzwords, be specific."
    )

    template = (
        "{system}\n\n"
        "Context:\n"
        "- Product: {product_name}\n"
        "- Audience: {audience}\n"
        "- Tone: {tone}\n"
        "- Key benefits: {benefits}\n"
        "- CTA: {cta}\n"
        "- Hashtags wanted: {hashtags_count}\n"
        "- Emojis allowed: {use_emojis}\n"
        "- Brief research (optional):\n{research}\n\n"
        "Deliver a single tweet only (no quotes, no preface)."
    )

    prompt = PromptTemplate(
        input_variables=[
            "system",
            "product_name",
            "audience",
            "tone",
            "benefits",
            "cta",
            "hashtags_count",
            "use_emojis",
            "research",
        ],
        template=template,
    )

    llm = get_llm()
    return prompt | llm | StrOutputParser()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-tweet", response_model=GenerateTweetResponse)
async def generate_tweet_api(
    product_name: str = Form(""),
    audience: str = Form(""),
    tone: str = Form("engaging"),
    key_benefits: str = Form(""),
    call_to_action: str = Form(""),
    hashtags_count: int = Form(2),
    use_emojis: bool = Form(True),
    allow_research: bool = Form(False)
):
    try:
        # Parse key_benefits from comma-separated string
        benefits_list = [b.strip() for b in key_benefits.split(',') if b.strip()] if key_benefits else []
        benefits_joined = ", ".join(benefits_list)
        
        # Optional lightweight research (only when explicitly allowed)
        research = ""
        if allow_research:
            research_query = f"{product_name} for {audience} benefits {benefits_joined}"
            research = fetch_research_snippets(research_query, num_results=3)

        chain = build_chain()
        tweet = chain.invoke(
            {
                "system": (
                    "You are concise, persuasive, specific, and follow platform best practices. "
                    "Stay strictly grounded in the provided fields. Do not add facts that are not in the inputs "
                    "or optional research. If information is missing, write generally without inventing details."
                ),
                "product_name": product_name.strip(),
                "audience": audience.strip(),
                "tone": tone.strip(),
                "benefits": benefits_joined,
                "cta": call_to_action or "",
                "hashtags_count": str(hashtags_count),
                "use_emojis": "yes" if use_emojis else "no",
                "research": research,
            }
        )
        # Ensure 280 char guard (soft clip if model exceeds)
        if len(tweet) > 280:
            tweet = tweet[:277].rstrip() + "..."
        
        # Make sure we never return None
        if not tweet:
            tweet = "Sorry, I couldn't generate a tweet at this time."
            
        # Return a properly constructed response object that matches GenerateTweetResponse model
        return GenerateTweetResponse(tweet=tweet.strip())
    except Exception as e:
        if 'logger' in globals():
            logger.error(f"Tweet generation error: {e}")
        else:
            print(f"Tweet generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/suggest_hashtags", response_model=HashtagSuggestResponse)
def suggest_hashtags(body: HashtagSuggestRequest):
    try:
        prompt = PromptTemplate(
            input_variables=["text", "k"],
            template=(
                "Given the following marketing context or YouTube URL, suggest {k} concise, relevant hashtags. "
                "Rules: no spaces, no special chars other than #, avoid duplicates, return as a comma-separated list only.\n\n"
                "Context: {text}"
            ),
        )
        llm = get_llm()
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({"text": body.text.strip(), "k": str(body.max_hashtags)})
        # Parse comma-separated hashtags
        parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
        hashtags: List[str] = []
        for p in parts:
            tag = p
            if not tag.startswith("#"):
                tag = f"#{tag.lstrip('#').replace(' ', '')}"
            tag = tag.split()[0]
            if tag not in hashtags:
                hashtags.append(tag)
        return HashtagSuggestResponse(hashtags=hashtags[: body.max_hashtags])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve static frontend if available
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(FRONTEND_DIR):
    # Static assets under /static
    app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")
    # Use Jinja to render the provided HTML so {% %} blocks work
    templates = Jinja2Templates(directory=FRONTEND_DIR)
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if templates is None:
        return RedirectResponse(url="/index.html")
    user_logged_in = request.session.get("user_logged_in", False)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user_logged_in
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if templates is None:
        return RedirectResponse(url="/index.html")
    
    # Get user session data
    user_logged_in = request.session.get("user_logged_in", False)
    user_tier = request.session.get("user_tier", "free")
    max_chars = request.session.get("max_chars", 280)
    
    context = {
        "request": request, 
        "user": user_logged_in,
        "user_tier": user_tier,
        "max_chars": max_chars,
        "is_premium": user_tier == "premium"
    }
    return templates.TemplateResponse("dashboard.html", context)

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



app.title = "TweetForge"
app.description = "AI-Powered Professional Tweet Generator with Real-time Research"
app.version = "3.0.0"
app.docs_url = "/api/docs"
app.redoc_url = "/api/redoc"

# Add CORS middleware for better frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# OAuth 2.0 PKCE setup for X (Twitter)
X_CLIENT_ID = os.getenv("X_CLIENT_ID")
X_CLIENT_SECRET = os.getenv("X_CLIENT_SECRET")
# Use 127.0.0.1 instead of localhost to match X Developer Portal configuration
X_REDIRECT_URI = "http://127.0.0.1:8000/auth/x/callback"
X_AUTH_URL = "https://twitter.com/i/oauth2/authorize"
X_TOKEN_URL = "https://api.twitter.com/2/oauth2/token"
X_USER_INFO_URL = "https://api.twitter.com/2/users/me"
X_TWEET_URL = "https://api.twitter.com/2/tweets"

def generate_pkce_pair():
    """Generate PKCE code verifier and challenge"""
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode('utf-8')).digest()
    ).decode('utf-8').rstrip('=')
    return code_verifier, code_challenge

def check_user_tier(access_token: str) -> str:
    """Check if user has Premium (X Premium) or Free tier"""
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        # Check user info to determine tier
        response = requests.get(X_USER_INFO_URL, headers=headers, timeout=10)
        if response.ok:
            user_data = response.json()
            # X Premium users have verified status or specific features
            # This is a simplified check - you might need to adjust based on actual X API
            is_verified = user_data.get("data", {}).get("verified", False)
            # For now, let's assume all logged-in users are premium for testing
            # In production, you'd check actual X Premium status
            return "premium" if is_verified else "free"
    except Exception as e:
        logger.warning(f"Could not determine user tier: {e}")
    # For testing purposes, assume logged-in users are premium
    return "premium"  # Default to premium for logged-in users

# Mount static files only if directory exists
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Legacy templates directory support (only if not already set)
_templates_dir = "templates"
if 'templates' in globals():
    _current_templates = templates  # keep web/ templates
else:
    _current_templates = None
if _current_templates is None and os.path.isdir(_templates_dir):
    templates = Jinja2Templates(directory=_templates_dir)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Initializing server... (LangChain + Groq mode)")

# Enhanced Pydantic models with validation
class TweetRequest(BaseModel):
    topic: str
    audience: str
    tweet_type: str = "professional"
    hashtags: str = ""
    
    @validator('topic')
    def validate_topic(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Topic must be at least 3 characters long')
        if len(v) > 200:
            raise ValueError('Topic must be less than 200 characters')
        return v.strip()
    
    @validator('audience')
    def validate_audience(cls, v):
        if v and len(v) > 100:
            raise ValueError('Audience description must be less than 100 characters')
        return v.strip() if v else ""
    
    @validator('hashtags')
    def validate_hashtags(cls, v):
        if v:
            hashtags = [tag.strip() for tag in v.split(',') if tag.strip()]
            if len(hashtags) > 10:
                raise ValueError('Maximum 10 hashtags allowed')
            for tag in hashtags:
                if len(tag) > 50:
                    raise ValueError('Each hashtag must be less than 50 characters')
        return v

class User(BaseModel):
    name: str
    email: str

class TweetResponse(BaseModel):
    success: bool
    content: str
    hashtags: List[str]
    call_to_action: Optional[str]
    tone: str
    research_sources: List[str]
    tweet_type: str
    engagement_metrics: dict
    generated_at: datetime
    processing_time: float

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_code: str
    timestamp: datetime

# Mock user for testing (without OAuth)
mock_user = User(name="Neural User", email="quantum@tweetforge-nexus.io")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if templates is None:
        return RedirectResponse(url="/index.html")
    
    # Get user session data
    user_logged_in = request.session.get("user_logged_in", False)
    user_name = request.session.get("user_name", "User")
    user_username = request.session.get("user_username", "")
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "user": {"name": user_name},
        "user_logged_in": user_logged_in,
        "user_username": user_username
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if templates is None:
        return RedirectResponse(url="/index.html")
    
    # Get user session data
    user_logged_in = request.session.get("user_logged_in", False)
    
    # Force user_logged_in to be a boolean
    if user_logged_in is None or user_logged_in == "":
        user_logged_in = False
    
    user_tier = request.session.get("user_tier", "free")
    max_chars = request.session.get("max_chars", 280)
    user_name = request.session.get("user_name", "User")
    user_username = request.session.get("user_username", "")
    
    # Debug logging
    logger.info(f"Dashboard route - Session data: user_logged_in={user_logged_in}, user_name={user_name}, user_username={user_username}")
    
    context = {
        "request": request, 
        "user": {"name": user_name},
        "user_logged_in": user_logged_in,
        "user_tier": user_tier,
        "user_username": user_username,
        "max_chars": max_chars,
        "is_premium": user_tier == "premium"
    }
    return templates.TemplateResponse("dashboard.html", context)

@app.get("/login")
async def login(request: Request, redirect_url: str = None):
    # Store the redirect URL directly in the session
    if redirect_url:
        request.session["auth_redirect_url"] = redirect_url
    
    # Redirect to X login
    return RedirectResponse(url="/login/x")

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")

@app.get("/login/x")
async def login_x(request: Request, redirect_url: str = None):
    """Initiate X OAuth 2.0 PKCE flow"""
    if not X_CLIENT_ID:
        raise HTTPException(status_code=500, detail="X OAuth not configured")
    
    code_verifier, code_challenge = generate_pkce_pair()
    
    # Store code verifier in session
    request.session["code_verifier"] = code_verifier
    
    # Store the redirect URL in session if provided
    if redirect_url:
        request.session["auth_redirect_url"] = redirect_url
    
    # Build authorization URL
    auth_params = {
        "response_type": "code",
        "client_id": X_CLIENT_ID,
        "redirect_uri": X_REDIRECT_URI,
        # Request write + offline access so we can post and refresh silently
        "scope": "tweet.read tweet.write users.read offline.access",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": secrets.token_urlsafe(32)
    }
    
    # Log the authorization parameters for debugging
    logger.info(f"Authorization URL parameters: client_id={X_CLIENT_ID}, redirect_uri={X_REDIRECT_URI}")
    
    auth_url = f"{X_AUTH_URL}?" + urllib.parse.urlencode(auth_params)
    return RedirectResponse(url=auth_url)

@app.get("/auth/x/callback")
async def auth_x_callback(request: Request, code: str = None, state: str = None, error: str = None):
    """Handle X OAuth 2.0 callback"""
    if error:
        logger.error(f"OAuth error received: {error}")
        return RedirectResponse(url="/?auth_error=true")
    
    if not code:
        logger.error("Authorization code not provided")
        return RedirectResponse(url="/?auth_error=missing_code")
    
    code_verifier = request.session.get("code_verifier")
    if not code_verifier:
        logger.error("Invalid session state - no code verifier")
        return RedirectResponse(url="/?auth_error=invalid_session")
    
    try:
        # Exchange code for access token
        token_data = {
            "code": code,
            "grant_type": "authorization_code",
            "client_id": X_CLIENT_ID,
            "redirect_uri": X_REDIRECT_URI,
            "code_verifier": code_verifier
        }
        
        try:
            response = requests.post(X_TOKEN_URL, data=token_data, timeout=10)
            if not response.ok:
                logger.error(f"Failed to exchange code for token: {response.status_code} - {response.text}")
                return RedirectResponse(url="/?auth_error=token_exchange_failed")
        except Exception as e:
            logger.error(f"Exception during token exchange: {str(e)}")
            return RedirectResponse(url="/?auth_error=token_exchange_exception")
        
        token_info = response.json()
        access_token = token_info.get("access_token")
        refresh_token = token_info.get("refresh_token")
        
        if not access_token:
            raise HTTPException(status_code=400, detail="No access token received")
        
        # Get user info from X API
        headers = {"Authorization": f"Bearer {access_token}"}
        user_response = requests.get(X_USER_INFO_URL, headers=headers, timeout=10)
        if not user_response.ok:
            logger.error(f"Failed to get user info: {user_response.status_code} - {user_response.text}")
            return RedirectResponse(url="/?auth_error=user_info_failed")
            
        user_data = user_response.json()
        user_name = user_data.get("data", {}).get("name", "X User")
        user_username = user_data.get("data", {}).get("username", "")
        
        # Get user info and determine tier
        user_tier = check_user_tier(access_token)
        
        # Store user session with explicit boolean value
        request.session["user_logged_in"] = True
        request.session["access_token"] = access_token
        request.session["user_tier"] = user_tier
        request.session["max_chars"] = 25000 if user_tier == "premium" else 280
        request.session["user_name"] = user_name
        request.session["user_username"] = user_username
        if refresh_token:
            request.session["refresh_token"] = refresh_token
        
        # Force session save
        request.session.modified = True
        
        # Debug log to verify session data is set correctly
        logger.info(f"X Auth Callback - Session data set: user_logged_in=True, user_name={user_name}, user_username={user_username}")
        
        logger.info(f"User logged in with {user_tier} tier")
        logger.info(f"Session data: user_logged_in={request.session.get('user_logged_in')}, tier={user_tier}")
        
        # Get the redirect URL from session
        target_url = request.session.get("auth_redirect_url", "/dashboard")
        
        # Clear the redirect URL from session after using it
        if "auth_redirect_url" in request.session:
            del request.session["auth_redirect_url"]
            
        logger.info(f"Redirecting to: {target_url}")
            
        return RedirectResponse(url=target_url)
        
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")

@app.get("/logout/x")
async def logout_x(request: Request):
    """Logout from X"""
    request.session.clear()
    return RedirectResponse(url="/")

@app.post("/post-to-x")
async def post_to_x(request: Request, tweet: str = Form("")):
    """Post a tweet to X for the logged-in user. Requires OAuth access token in session."""
    if not tweet.strip():
        raise HTTPException(status_code=400, detail="No tweet content provided")
    access_token = request.session.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Not logged in with X")
    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        payload = {"text": tweet.strip()}
        # Privacy: do not store tweet or tokens; send directly to X and return status only
        resp = requests.post(X_TWEET_URL, headers=headers, json=payload, timeout=10)
        if not resp.ok:
            detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=f"X API error: {detail}")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-tweet-v2", response_model=TweetResponse)
async def generate_tweet_v2(
    topic: str = Form(""),
    tweet_type: str = Form("professional"), 
    max_chars: int = Form(280),    
    hashtags: str = Form(""),
    allow_research: bool = Form(False)
):
   
    """Generate tweet using LangChain + Groq and then suggest hashtags from the generated text."""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    if not topic.strip():
        raise HTTPException(status_code=400, detail="Please provide a topic or YouTube URL.")

    try:
        logger.info(f"🎯 [{request_id}] Tweet generation request received - Topic: {topic}")
        logger.info(f"🔎 [{request_id}] allow_research={allow_research}")

        # Detect YouTube URL (or any URL) in topic
        import re
        url_match = re.search(r"https?://[^\s]+", topic)
        video_url = None
        if url_match:
            url = url_match.group(0)
            if re.search(r"(youtube\.com|youtu\.be)", url, re.IGNORECASE):
                video_url = url
       # Around line 627 in your app.py
        # Remove URL from topic text for clearer prompting
        clean_topic = re.sub(r"https?://[^\s]+", "", topic).strip()
        
        needs_research = False

        # 1) Generate tweet (no hashtags inside). If video_url provided, include link at end.
        llm = get_llm()

        # 1) Generate tweet (no hashtags inside). If video_url provided, include link at end.
        llm = get_llm()
        video_context = ""
        # Use a video-specific prompt only when a YouTube URL exists; otherwise use a generic prompt
        if video_url:
            gen_prompt = PromptTemplate(
                input_variables=["topic", "video_url", "video_context", "max_chars"],
                template=(
                    "You are writing a promotional tweet for the YouTube content below. "
                    "Ground the tweet strictly in the provided video_context and topic. "
                    "Constraints: {max_chars} chars max, clear hook, one concrete detail from the video, "
                    "natural voice, short CTA. Do NOT include hashtags. End with the URL.\n\n"
                    "If the context is sparse, write a tasteful, generic promotional teaser without stating that information is missing. "
                    "Never refuse or ask for more details; do your best with what's given.\n\n"
                    "topic: {topic}\n"
                    "video_url: {video_url}\n"
                    "video_context (title/author/snippets):\n{video_context}"
                ),
            )
            gen_chain = gen_prompt | llm | StrOutputParser()
            video_context = build_video_context(video_url)
            if not video_context:
                video_context = "- YouTube video"
            tweet_text = gen_chain.invoke({
                "topic": clean_topic or "This YouTube video",
                "video_url": video_url,
                "video_context": video_context,
                "max_chars": str(max_chars),
            }).strip()
        else:
            # For non-video requests, optionally enrich with Serper if content is sparse
            # Heuristic: if there are fewer than ~12 words, or fewer than 80 characters
            content_for_research = clean_topic
            word_count = len([w for w in content_for_research.split() if w.isalpha() or any(c.isalnum() for c in w)])
            needs_research = (len(content_for_research) < 80) or (word_count < 12)
            if allow_research:
                # Force-enable research when explicitly requested
                needs_research = True
            logger.info(f"🔬 [{request_id}] research gating: allow={allow_research}, needs_research={needs_research}, len={len(content_for_research)}, words={word_count}")
            web_research = ""
            if needs_research and allow_research:
                try:
                    # Refine the topic into a better search query first
                    def create_search_query(topic: str) -> str:
                        """Uses an LLM to distill a topic into a clean search query."""
                        logger = logging.getLogger(__name__)
                        logger.info(f"Refining topic into search query: {topic}")
                        try:
                            llm = get_llm()
                            prompt = PromptTemplate(
                                input_variables=["topic"],
                                template=(
                                    "You are an expert at creating effective search engine queries. "
                                    "Distill the core keywords from the following user request. "
                                    "Return ONLY the search query, with no explanation.\n\n"
                                    "Request: {topic}"
                                ),
                            )
                            chain = prompt | llm | StrOutputParser()
                            search_query = chain.invoke({"topic": topic})
                            logger.info(f"Refined search query: {search_query}")
                            return search_query.strip()
                        except Exception as e:
                            logger.warning(f"Could not refine search query: {e}. Using original topic.")
                            return topic
                    research_query = create_search_query(content_for_research)
                    # Use the refined query for fetching research
                    logger.info(f"🔍 [{request_id}] Fetching research for query: {research_query}")
                    web_research = fetch_research_snippets(research_query, num_results=3)
                    logger.info(f"📚 [{request_id}] research chars={len(web_research)}")
                except Exception:
                    web_research = ""
            # Standard template for tweet generation
            template_text = (
                "You are writing a promotional tweet. "
                "Constraints: {max_chars} chars max, attention-grabbing hook, concrete details, natural voice, compelling CTA. "
                "Do NOT include hashtags unless specifically requested.\n\n"
                "Write strictly based on the provided topic and optional research only. "
                "Do not invent or add any facts not present in these inputs. If details are missing, keep it general.\n\n"
                "{topic}\n\n"
                "optional_research (use at most one subtle fact):\n{research}"
            )

            gen_prompt = PromptTemplate(
                    input_variables=["topic", "max_chars", "research"],
                    template=template_text,
                )
        
            gen_chain = gen_prompt | llm | StrOutputParser()
            tweet_text = gen_chain.invoke({
                "topic": clean_topic,
                "max_chars": str(max_chars),
                "research": web_research,
            }).strip()
        limit = 25000 if max_chars and max_chars > 280 else 280
        if len(tweet_text) > limit:
            tweet_text = tweet_text[: max(0, limit - 3)].rstrip() + "..."

        # Ensure video link is present if we detected it, without exceeding 280
        if video_url and video_url not in tweet_text:
            # leave room for space + url
            max_len = limit - (len(video_url) + 1)
            if len(tweet_text) > max_len:
                tweet_text = tweet_text[: max_len - 3].rstrip() + "..."
            tweet_text = f"{tweet_text} {video_url}"

        # 2) Suggest hashtags based on generated tweet
        hash_prompt = PromptTemplate(
            input_variables=["tweet"],
            template=(
                "Based on this tweet, suggest up to 5 relevant, concise hashtags. "
                "Rules: no spaces, single words, keep it brand-safe. Return as a comma-separated list only.\n\n"
                "Tweet: {tweet}"
            ),
        )
        hash_chain = hash_prompt | llm | StrOutputParser()
        raw = hash_chain.invoke({"tweet": tweet_text})
        parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
        suggested = []
        for p in parts:
            tag = p
            if not tag.startswith("#"):
                tag = f"#{tag.lstrip('#').replace(' ', '')}"
            tag = tag.split()[0]
            if tag not in suggested:
                suggested.append(tag)
        
        if hashtags:
            # merge user-provided hashtags
            user_tags = [t.strip() for t in hashtags.split(',') if t.strip()]
            user_tags = [t if t.startswith('#') else f"#{t}" for t in user_tags]
            suggested = (user_tags + [t for t in suggested if t not in user_tags])[:5]
        else:
            suggested = suggested[:5]

        processing_time = time.time() - start_time
        engagement_metrics = {
            "estimated_reach": "10K-50K",
            "viral_potential": "High" if len(tweet_text) > 200 else "Medium",
            "engagement_score": min(95, max(60, len(tweet_text) * 0.3 + len(suggested) * 5)),
            "readability_score": "High" if len(tweet_text.split()) < 20 else "Medium",
        }
        # Fill research sources section for a nicer UI when video is used
        sources: list[str] = []
        if video_url:
            sources.append(video_url)
            if video_context:
                # include title/author line
                first_line = video_context.splitlines()[0] if video_context else ""
                if first_line:
                    sources.append(first_line)

        return TweetResponse(
            success=True,
            content=tweet_text,
            hashtags=suggested,
            call_to_action=None,
            tone="promotional",
            research_sources=sources,
            tweet_type=tweet_type,
            engagement_metrics=engagement_metrics,
            generated_at=datetime.now(),
            processing_time=round(processing_time, 2),
        )
    except Exception as e:
        logger.error(f"❌ [{request_id}] Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-tweet-youtube", response_model=TweetResponse)
async def generate_tweet_youtube(
    url: str = Form(""),
    max_chars: int = Form(280),
    allow_research: bool = Form(True)
):
    """Convenience endpoint for only a YouTube URL input."""
    return await generate_tweet_v2(topic=url, tweet_type="promotional", hashtags="", max_chars=max_chars, allow_research=allow_research)


# All document processing functionality (PDF, image, OCR) has been removed

def _normalize_numeric_confusions(s: str) -> str:
    """Fix common numeric confusions in LLM outputs."""
    import re
    
    def fix(token: str) -> str:
        # Fix common percentage symbol confusions
        token = token.replace(" percent", "%").replace(" pct", "%")
        token = token.replace("percent", "%").replace("pct", "%")
        token = token.replace(" off", "%").replace("off", "%")
        
        # Fix common thousand/K confusions
        if re.search(r"\d+[kK]\b", token):
            token = re.sub(r"(\d+)[kK]\b", lambda m: m.group(1) + "000", token)
        
        # Fix common million/M confusions
        if re.search(r"\d+[mM]\b", token):
            token = re.sub(r"(\d+)[mM]\b", lambda m: m.group(1) + "000000", token)
        return token
    
    # Apply fixes to tokens that match the pattern
    return re.sub(r"[A-Za-z0-9%$€£¥₹.,:/+\-#()=xX*]+", lambda m: fix(m.group(0)), s)


def score_tweet_candidates(candidates: list[str]) -> str:
    """Score and rank tweet candidates based on promotional effectiveness."""
    import re
    
    # Normalize numeric expressions for better comparison
    normalized = [_normalize_numeric_confusions(c) for c in candidates]

    try:
        # Enhanced scoring function to better prioritize text with numbers and symbols
        def score(s: str) -> tuple[int, int, int, int, int]:
            import re
            # Count digits, symbols, and special promotional terms with higher weights
            digits = sum(ch.isdigit() for ch in s) * 3  # Higher weight for digits
            
            # Expanded symbol set with currency and percentage symbols
            symbols = len(re.findall(r"[%$€£¥₹#/:+\-*()=]", s)) * 4  # Higher weight for symbols
                
            # Look for currency-amount patterns (e.g., $50, ₹100)
            currency_amounts = len(re.findall(r"[$€£¥₹]\s*\d+|\d+\s*[$€£¥₹]|Rs\.?\s*\d+", s, re.IGNORECASE)) * 15
                
            # Look for percentage patterns (e.g., 50%, 50 percent, 50 off)
            percentages = len(re.findall(r"\d+\s*%|\d+\s*percent|\d+\s*off|save\s+\d+|discount\s+\d+", s, re.IGNORECASE)) * 15
                
            # Look for promotional codes (uppercase alphanumeric sequences)
            promo_codes = len(re.findall(r"\b[A-Z0-9]{4,}\b", s)) * 12
                
            # Expanded list of promotional terms with higher weight for important ones
            promo_terms = len(re.findall(
                r"(?i)(off|discount|save|deal|promo|code|offer|free|sale|limited|exclusive|special|only|today|now|hurry|last|chance|buy|get|use)", 
                s
            )) * 5
                
            # Look for date patterns (expiry dates are important in promotions)
            dates = len(re.findall(
                r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4}",
                s, re.IGNORECASE)) * 10
                
            text_quality = len(s.strip()) + len(s.split()) * 3
                
            # Prioritize text with a good balance of digits, symbols, and promotional terms
            numeric_symbol_score = digits + symbols + currency_amounts + percentages + promo_codes
            promo_context_score = promo_terms + dates
            
            # Extra boost for text containing both percentages/amounts AND promo codes
            combined_boost = 0
            if (currency_amounts > 0 or percentages > 0) and promo_codes > 0:
                combined_boost = 50
            
            return (numeric_symbol_score, promo_context_score, combined_boost, text_quality, len(s.strip()))
        
        best = max(normalized, key=score, default="")
        return best.strip()[:20000]
    except Exception as e:
        return ""


def _choose_amount(candidates: list[int]) -> int | None:
    """Heuristic: prefer realistic ticket-like amounts (2-3 digits) over 4+ digits.
    Return the candidate with minimum penalty: penalty = abs(len-3) then value.
    """
    if not candidates:
        return None
    scored = []
    for n in candidates:
        length_penalty = abs(len(str(n)) - 3)
        scored.append((length_penalty, n))
    scored.sort()
    return scored[0][1]





@app.get("/api/tweets")
async def get_tweets():
    """Get recent tweets (placeholder for future implementation)"""
    return {"tweets": []}

@app.get("/api/health")
async def health_check():
    """Simple health check without CrewAI."""
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    serpapi_ok = bool(os.getenv("SERPAPI_API_KEY"))
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "3.0.0",
        "services": {
            "framework": "langchain",
            "groq_api": "active" if groq_ok else "inactive",
            "serpapi_api": "active" if serpapi_ok else "inactive",
        },
    }

@app.get("/api/user-status")
async def get_user_status(request: Request):
    """Get current user login status and tier."""
    user_logged_in = request.session.get("user_logged_in", False)
    user_tier = request.session.get("user_tier", "free")
    max_chars = request.session.get("max_chars", 280)
    
    return {
        "logged_in": user_logged_in,
        "tier": user_tier,
        "max_chars": max_chars,
        "is_premium": user_tier == "premium"
    }

@app.get("/test-login")
async def test_login(request: Request):
    """Test endpoint to simulate a logged-in user."""
    request.session["user_logged_in"] = True
    request.session["user_tier"] = "premium"
    request.session["max_chars"] = 25000
    return {"message": "Test login successful", "tier": "premium", "max_chars": 25000}

@app.get("/api/stats")
async def get_stats():
    """Get application statistics"""
    return {
        "total_requests": 0,  # Placeholder
        "successful_tweets": 0,  # Placeholder
        "average_processing_time": 0,  # Placeholder
        "uptime": "active"
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    payload = ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            timestamp=datetime.now()
    )
    return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(payload))

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    payload = ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now()
    )
    return JSONResponse(status_code=500, content=jsonable_encoder(payload))

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TweetForge Nexus Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    args = parser.parse_args()
    
    print(f"🚀 Initializing TweetForge Nexus Quantum Engine on port {args.port}...")
    print("🔮 GROQ Neural Network activated successfully")
    uvicorn.run(app, host=args.host, port=args.port)
