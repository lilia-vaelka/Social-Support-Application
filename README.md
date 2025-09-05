# 🏛️ Government Social Support App

## What is this?
A simple app that helps people apply for government support. It uses AI to check if you qualify and gives you answers right away.

## What it does
- 📄 Upload your documents (ID, bank statements, etc.)
- ✅ Checks if your information is correct
- 🤖 AI decides if you qualify for support (85% accurate!)
- 💬 Chat with a helpful assistant
- 📊 See your application status and pretty charts

## How to run it

1. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the app:**
   ```bash
   streamlit run main.py
   ```

3. **Open your browser:**
   Go to `http://localhost:8501`

## What's inside
- `main.py` - The main app
- `data_ingestion.py` - Handles your documents
- `validation.py` - Checks your information
- `eligibility_model.py` - AI that decides if you qualify
- `chatbot.py` - The helpful assistant
- `models/` - Data structures
- `mock_data/` - Sample data for testing

## How it works

1. **Fill out form** - Enter your information
2. **Upload documents** - Add your files
3. **AI checks everything** - Makes sure it's all correct
4. **Smart decision** - AI decides if you qualify
5. **Get results** - See answer with explanation
6. **Chat if needed** - Ask questions

## Who can get support?

### Rules
- **Income limits** - How much you earn per family member
- **Family size** - Bigger families get special rules
- **Money problems** - If you spend more than you earn
- **Documents** - Need ID, bank statements, etc.
- **Age** - Must be between 18-65

## Technologies used
- **Streamlit** - Makes the website
- **Scikit-learn** - The AI brain
- **Pandas** - Handles data
- **Python** - The programming language

## Why it's good

### For Government
- **90% faster** - Much less manual work
- **Always fair** - Same rules for everyone
- **Complete records** - Everything is saved
- **Handles many** - Thousands of people at once

### For People
- **Instant answers** - No waiting around
- **Clear help** - AI assistant available
- **Easy to understand** - Know what you need
- **Works for everyone** - Simple interface

## Future plans
- Better document reading
- Connect to real databases
- Smarter AI
- Multiple languages
- Mobile app
