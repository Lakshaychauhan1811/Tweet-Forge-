# TweetForge – AI-Powered Promotional Tweet Generator

TweetForge generates concise, engaging promotional tweets from a simple topic or a YouTube URL. It can optionally enrich short inputs using lightweight web research.

## 🚀 What It Does (Current)

- **Generate promotional tweets** from a text topic
- **Generate promotional tweets for YouTube links** (adds the link and uses lightweight context)
- **Optional research** via Serper when enabled; falls back to Groq if Serper is unavailable
- **Hashtag suggestions** derived from the generated tweet
- **Modern UI** with smooth interactions and copy/regenerate actions

## 📋 Prerequisites

- Python 3.8 or higher
- GROQ API key (required)
- Serper key (optional, for web research)

## 🛠️ Installation & Setup

### Option 1: Quick Setup (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd tweetforge
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```
   This will:
   - Check Python version
   - Create .env file from template
   - Install dependencies
   - Check API key configuration
   - Optionally start the server

### Option 2: Manual Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**
   ```bash
   cp env_example.txt .env
   # Edit .env file with your API keys
   ```

3. **Start the server**
   ```bash
   python start.py
   ```

## 🔑 API Keys Setup

### Required: GROQ API Key
1. Visit [console.groq.com](https://console.groq.com/)
2. Sign up for a free account
3. Create an API key
4. Add it to your `.env` file:
   ```
   GROQ_API_KEY=your_actual_api_key_here
   ```

### Optional: Serper Key (for internet research)
1. Visit [serper.dev](https://serper.dev/)
2. Sign up for a free account
3. Create an API key
4. Add it to your `.env` file:
   ```
  SERPER_API_KEY=your_actual_api_key_here
   ```

## 🎯 Usage

1. **Start the server**
  ```bash
  python start.py
  ```

2. **Open your browser**
   Navigate to `http://127.0.0.1:8000`

3. **Generate tweets**
   - Enter a short topic (e.g., “summer sale 40% off, free shipping”) and optional hashtags
   - Or paste a YouTube URL
   - Click "Generate Tweet"
   - Copy the tweet or regenerate for a new variation

## 📚 API Documentation

- **Interactive API Docs**: `http://127.0.0.1:8000/api/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/api/redoc`
- **Health Check**: `http://127.0.0.1:8000/api/health`

## 🔧 Configuration

Edit the `.env` file to customize:

```env
# Server settings
HOST=127.0.0.1
PORT=8000

# API keys
GROQ_API_KEY=your_groq_api_key_here
SERPER_API_KEY=your_serper_api_key_here

# Security
SECRET_KEY=your-super-secret-key-change-this-in-production
```

## 🐛 Troubleshooting

### Common Issues

1. **"GROQ_API_KEY not found"**
   - Make sure you've created a `.env` file
   - Verify your API key is correct
   - The app will work with basic templates without the key

2. **"Port already in use"**
   - Change the PORT in your `.env` file
   - Or stop the process using the port

3. **"Module not found" errors**
   - Run `pip install -r requirements.txt`
   - Make sure you're in the correct directory

4. **OCR features not working**
   - Install optional dependencies: `pip install easyocr pdf2image opencv-python`
   - Or uncomment them in `requirements.txt`

### Getting Help

- Check the console output for detailed error messages
- Visit the API documentation at `/api/docs`
- Ensure all required dependencies are installed

## 🏗️ Architecture

- **Backend**: FastAPI
- **AI Engine**: Groq via LangChain
- **Frontend**: HTML/CSS/JavaScript
- **Research**: Serper (optional) with Groq fallback

## 📦 Dependencies

### Core Dependencies
- FastAPI – Web framework
- Uvicorn – ASGI server
- langchain, langchain-groq – LLM access
- python-dotenv – env management

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify your API keys are correct
3. Ensure all dependencies are installed
4. Check the console output for error messages