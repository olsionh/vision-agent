# Visual Product Analyzer

An AI application that identifies any product from a photo and generates a full research report including pricing, reviews, and alternatives.

Live Demo: https://huggingface.co/spaces/olsionh/vision-agent

## How It Works

1. User uploads a photo of any product
2. GPT-4o vision model identifies the exact product and model from the image
3. A LangGraph research agent autonomously searches the web multiple times
4. The agent synthesizes a structured report with pricing, reviews, pros/cons, and alternatives

## Features

- Product identification from any photo using GPT-4o multimodal vision
- Autonomous multi-step web research agent
- Live agent activity — watch each search step in real time
- Structured report with pricing, reviews, pros/cons, and alternatives
- Download report as .txt

## Tech Stack

- Vision Model: OpenAI GPT-4o (multimodal)
- Agent Framework: LangGraph
- LLM: OpenAI GPT-4o-mini
- Search: Tavily Search API
- UI: Streamlit
- Containerization: Docker
- Deployment: Hugging Face Spaces

## What Makes This Unique

Most AI apps take text as input. This app takes an image, understands what it sees, and then autonomously researches it on the live web. It combines computer vision, large language models, and agentic reasoning in a single pipeline.

## Run Locally

    git clone https://github.com/olsionh/vision-agent.git
    cd vision-agent
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

## Limitations

- Requires OpenAI and Tavily API keys
- Product identification accuracy depends on image quality
- Works best with consumer products that have an online presence

## Future Improvements

- Add price tracking over time
- Support barcode and QR code scanning
- Compare multiple products side by side
- Add shopping links directly to the report
