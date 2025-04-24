# AI-Powered PDF Catalog Splitter

An intelligent system for extracting and separating technical product sheets from PDF catalogs using AI-driven document analysis.

## Overview

This project uses AI (primarily Mistral AI models) to detect the boundaries between different technical product sheets within product catalogs. It then extracts each technical sheet into a separate PDF file, making it easier to organize, search, and distribute specific product information from large catalogs.

It was mainly tested on construction/industrial product catalogs with multiple individual product.sheets

## Technical Components

### Core Processors

**ai_processor_3/** - Latest version with OCR capabilities and better extraction

### Key Classes

- **Pipeline**: Coordinates the entire extraction process
- **OCRAnalyzer**: Performs OCR and AI analysis on document content
- **DocumentProcessor**: Extracts text and adds machine-readable page markers
- **SimpleVisualAnalyzer**: Analyzes visual elements to detect sheet boundaries
- **Evaluator**: Assesses extraction quality and provides accuracy metrics for boundary detection

## How It Works

1. **Document Enhancement**:
   - Pages are processed and enhanced with machine-readable identifiers
   - OCR is performed to extract text content

2. **Boundary Detection**:
   - AI models analyze the document to identify technical sheet boundaries
   - Multiple signals are considered: content transitions, layout changes, headers, etc.
   - Each potential boundary is assigned a confidence score

3. **Technical Sheet Extraction**:
   - Once boundaries are identified, individual PDFs are created for each sheet
   - Files are named according to detected product names and page references
   - All extracted PDFs are organized in a structured output directory

4. **Result Presentation**:
   - A web interface displays the extraction results
   - Visualizations show where the boundaries were detected
   - Users can preview and download each extracted technical sheet

## Getting Started

### Prerequisites

- Python 3.12+
- Required Python packages in `requirements.txt`
- Mistral AI API key (for AI-based analysis)

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/ai-pdf-splitting.git cd ai-pdf-splitting
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Set up your environment variables:
```
echo "MISTRAL_API_KEY=your_key_here" > .env
```

### Usage

#### Web Interface

1. Start the Streamlit web app:
```
streamlit run frontend/frontend_ocr.py
```
2. Upload a PDF catalog through the web interface
3. Click "Extract Technical Sheets"
4. View and download the extracted technical sheets

#### Command Line

Process a PDF catalog directly:
```
python ai_processor_3/pipeline.py
```
