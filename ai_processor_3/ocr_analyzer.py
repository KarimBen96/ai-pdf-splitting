from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import os
import sys
import json
import re
from mistralai import Mistral
import datauri

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_processor_3.prompt_list import prompt_technical_sheets, prompt_table_of_contents, prompt_technical_sheets_mistral
from ai_processor_3.document_processor import main


class OCRAnalyzer:
    """
    Integrates Mistral LLM for advanced boundary detection and content analysis
    of technical sheets in catalogs.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "mistral-small-latest",
        # model: str = "mistral-large-latest",
        # model: str = "pixtral-large-latest",
        prompt: str = "",
        debug: bool = False,
    ):
        """
        Initialize the OCR analyzer.

        Args:
            api_key: Mistral API key (if None, uses environment variable)
            model: Model name to use (default: "mistral-small-latest")
            prompt: System prompt for the analysis
            debug: Enable debug output
        """
        self.debug = debug
        self.model = model
        self.system_prompt = prompt

        # Use provided API key or get from environment
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key must be provided or set as MISTRAL_API_KEY environment variable"
                )

        # Initialize Mistral client
        # self.llm = ChatMistralAI(model=model, temperature=0.2)
        self.client = Mistral(api_key=api_key)

    def upload_pdf(self, pdf_filename: str) -> str:
        """
        Upload a PDF file to the Mistral API for OCR processing.

        Args:
            pdf_filename: Path to the PDF file

        Returns:
            Signed URL to access the uploaded file
        """
        uploaded_pdf = self.client.files.upload(
            file={
                "file_name": pdf_filename,
                "content": open(pdf_filename, "rb"),
            },
            purpose="ocr",
        )
        signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)
        return signed_url.url


    def save_image(self, image):
        """
        Save a base64-encoded image to disk.

        Args:
            image: Image object with image_base64 and id attributes
        """
        parsed = datauri.parse(image.image_base64)
        with open(image.id, "wb") as file:
            file.write(parsed.data)


    def create_markdown_file(self, ocr_response, output_filename="output.md"):
        """
        Create a markdown file from OCR response.

        Args:
            ocr_response: Response from OCR processing containing pages and images
            output_filename: Path to save the markdown output
        """
        with open(output_filename, "wt") as f:
            for page in ocr_response.pages:
                f.write(page.markdown)
                for image in page.images:
                    self.save_image(image)


    def process_file_old(self, pdf_filename: str) -> str:
        """
        Process a PDF file with OCR and analyze it using Mistral LLM.

        Args:
            pdf_filename: Path to the PDF file to process

        Returns:
            Processed content from the LLM
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt,
                    },
                    {
                        "type": "document_url",
                        "document_url": self.upload_pdf(pdf_filename),
                    },
                ],
            }
        ]

        chat_response = self.client.chat.complete(
            model=self.model,
            messages=messages,
        )

        return chat_response.choices[0].message.content


    def process_file(self, pdf_filename: str, document_analysis=None) -> str:
        """
        Process a PDF file with OCR and analyze it using Mistral LLM.

        Args:
            pdf_filename: Path to the PDF file to process
            document_analysis: Optional pre-analyzed document data to provide context

        Returns:
            Processed content from the LLM
        """
        # Prepare the content with document analysis if available
        content_elements = []
        
        # If document analysis is provided, add it as context
        if document_analysis:
            analysis_context = f"Document analysis information: {json.dumps(document_analysis, indent=2, ensure_ascii=False)}\n\n"
            content_elements.append({
                "type": "text",
                "text": analysis_context + self.system_prompt
            })
        else:
            content_elements.append({
                "type": "text",
                "text": self.system_prompt
            })
        
        # Add the document URL
        content_elements.append({
            "type": "document_url",
            "document_url": self.upload_pdf(pdf_filename)
        })
        
        messages = [
            {
                "role": "user",
                "content": content_elements
            }
        ]

        chat_response = self.client.chat.complete(
            model=self.model,
            messages=messages,
        )

        return chat_response.choices[0].message.content


    def chat_with_document(self, pdf_filename: str, user_query: str) -> str:
        """
        Chat with the LLM about a document that has been loaded.
        
        Args:
            pdf_filename: Path to the PDF file to process
            user_query: The user's question or prompt about the document
            
        Returns:
            The LLM's response to the user query
        """
        # First, upload the document to get the signed URL
        document_url = self.upload_pdf(pdf_filename)
        
        # Create the messages with system instructions, document reference, and user query
        messages = [
            {
                "role": "system",
                "content": "You are an assistant that helps answer questions about documents. Analyze the provided document and respond to the user's query about it."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "document_url",
                        "document_url": document_url,
                    },
                    {
                        "type": "text",
                        "text": user_query,
                    },
                ],
            }
        ]
        
        # Get response from Mistral
        chat_response = self.client.chat.complete(
            model=self.model,
            messages=messages,
        )
        
        return chat_response.choices[0].message.content


    def interactive_document_chat_old(self, pdf_filename: str):
        """
        Start an interactive chat session about a document.
        
        Args:
            pdf_filename: Path to the PDF file to discuss
        """
        print(f"Starting chat about {pdf_filename}. Type 'exit' or 'quit' to end.")
        print("Loading document...")
        
        while True:
            user_input = input("\nYour question: ")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Ending chat session.")
                break
                
            print("\nThinking...")
            response = self.chat_with_document(pdf_filename, user_input)
            print(f"\nAssistant: {response}")


    def interactive_document_chat(self, pdf_filename: str, document_analysis=None):
        """
        Start an interactive chat session about a document.
        
        Args:
            pdf_filename: Path to the PDF file to discuss
            document_analysis: Optional pre-analyzed document data to provide context
        """
        print(f"Starting chat about {pdf_filename}. Type 'exit' or 'quit' to end.")
        print("Loading document...")
        
        # If document analysis is provided, add it to the initial context
        document_context = ""
        if document_analysis:
            document_context = f"Document analysis information: {json.dumps(document_analysis, indent=2, ensure_ascii=False)}\n\n"
            print("Document analysis information has been loaded into the chat context.")
        
        # Store conversation history
        conversation = []
        
        if document_context:
            # Add document analysis as system message if available
            conversation.append({
                "role": "system",
                "content": f"The following is pre-analyzed information about the document that you can use to better answer queries: {document_context}"
            })
        
        while True:
            user_input = input("\nYour question: ")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Ending chat session.")
                break
            
            print("\nThinking...")
            
            # Include document analysis in the query if available
            messages = []
            
            # Add conversation history
            for message in conversation:
                if message["role"] != "system":  # Skip system messages for now
                    messages.append(message)
            
            # Add current user query
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "document_url",
                        "document_url": self.upload_pdf(pdf_filename),
                    },
                    {
                        "type": "text",
                        "text": user_input,
                    },
                ],
            })
            
            # Get response from Mistral
            chat_response = self.client.chat.complete(
                model=self.model,
                messages=[
                    # System message with document context if available
                    *([{"role": "system", "content": document_context}] if document_context else []),
                    # User messages
                    *messages
                ],
            )
            
            response = chat_response.choices[0].message.content
            print(f"\nAssistant: {response}")
            
            # Update conversation history
            conversation.append({"role": "user", "content": user_input})
            conversation.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    prompt_pages_description = """You are an expert document analyzer specializing in technical catalogs and specification sheets.
        Your task is to analyze document content and return a small description in French for each page.
        I inform you that the document is in French and that it contains 10 pages exactly

        You must retun a list of dictionaries with the following:
        [
            {
                "page": "page number",
                "title": "title of the page",
                "description": "description of the page content"
            }
        ]
    """

    prompt_pages_description_claude = """You are an expert document analyzer specializing in technical documentation.
        Your task is to analyze the provided document content and return a concise description for each page in French.

        The document contains exactly 10 pages.

        Return a list of dictionaries with the following structure:
        [
            {
                "page": "page number",
                "title": "main title of the page",
                "description": "brief description of the page content"
            }
        ]

        Important: Pay careful attention to page transitions, unique content identifiers, and distinct sections. Each page should be analyzed independently based on its specific content.
        
        Explain how you determine the page you're currently in
        """

    prompt_1 = """Quel format de balisage ou de structuration utilisez-vous pour identifier les différentes pages du document que je vous ai soumis? 
        Utilisez-vous:
        - Des balises XML (comme <document_content page="X">)
        - Des marqueurs textuels spécifiques
        - Des métadonnées PDF
        - Une autre méthode de délimitation des pages

        Merci de décrire précisément comment vous déterminez où commence et se termine chaque page dans le document."""


    prompt_2 = """I'm working with Mistral OCR to extract technical sheets from a catalog, but I'm experiencing issues with page handling. Please analyze the following details to suggest targeted solutions:
        Describe exactly how Mistral OCR is mishandling the pages (e.g., merging content from different pages, skipping pages, misidentifying page boundaries)
        What is the format of your input catalog? (PDF, scanned images, etc.)
        Describe the structure of your technical sheets (consistent headers/footers, table layouts, etc.)
        What specific Mistral OCR implementation or version are you using?
        Are you using any pre-processing steps before OCR?
        Have you tried adjusting any configuration parameters in Mistral OCR?
        Can you provide a simple example of the expected output vs. what you're currently getting?

        This information will help me identify the most effective solutions while working within your constraint of using only Mistral OCR."""


    prompt_3 = """Quand je te demande d'extraire des feuilles techniques d'un catalogue, tu le fais bien mais tu ne donnes pas les memes pages que celles
        du document, telles qu'elles sont présentes exactement.
        Comment faire pour avoir exactement les memes pages que celles du document ?"""


    prompt_technical_sheets = """You are an expert document analyzer specializing in technical catalogs and specification sheets.
        Your task is to analyze document content and identify boundaries between different technical 
        sheets in product catalogs.

        If you encounter a table of contents, use it to help you identify the boundaries of the technical sheets.

        You must detect the language of the document and provide the output in the same language.
        
        Important rules:
        - A technical sheet MUST describe exactly ONE product (not product categories or multiple products)
        - A single technical sheet can span one or multiple pages
        - Each boundary you identify should represent the start of a new product technical sheet
        - Subtitles can identify different technical sheets for the same family of products (same title)
        
        Do not include any other information or explanations. Avoid unnecessary details, preambles, or conclusions.
        
        
        Your output MUST be valid JSON in this exact format:
        [
            {
                "product": "product title or code",
                "confidence": "your confidence level (0.0-1.0)",
                "pages": "a list of the page numbes where the technical sheet is located eg. [1, 2, 3]",
                "reason": "explanation for boundary detection"
            }
        ]
        """



    pdf_filename_tertu_1_10 = "data/input/Catalogue-Tertu-Equipements-1-10.pdf"
    pdf_filename_tertu_1_25 = "data/input/Catalogue-Tertu-Equipements-1-25.pdf"
    pdf_filename_bordures = "data/input/CEL_Bordures_12-2020_V2.pdf"
    pdf_filename_numbered = "data/input/Catalogue-Tertu-Equipements-1-10_pagenumber.pdf"
    pdf_filename_numbered_fitz = "data/input/output_with_page_numbers.pdf"
    pdf_filename_genie_civil_1_20 = "data/input/Catalogue-general-Nos-Solutions-VRD-Genie-Civil-1-20.pdf"


    enhanced_pdf_path, analysis_path, document_analysis, tech_sheet_pages = main(pdf_filename_genie_civil_1_20)
    print(f"Document analysis: {type(document_analysis)}")

    # print(f"Enhanced PDF saved to: {enhanced_pdf_path}")
    # print(f"Document analysis saved to: {analysis_path}")
    # print(f"Likely technical sheet pages: {tech_sheet_pages}")


    # ocr_analyzer = OCRAnalyzer(prompt=prompt_technical_sheets)
    # processed_file = ocr_analyzer.process_file(enhanced_pdf_path, document_analysis)
    # print(processed_file)

    print("\n\n\n")

    ocr_analyzer = OCRAnalyzer()
    ocr_analyzer.interactive_document_chat(enhanced_pdf_path, document_analysis)
