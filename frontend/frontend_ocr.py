import os
import sys
import json
import streamlit as st
import base64
import pandas as pd
import plotly.express as px
from pathlib import Path
from pdf2image import convert_from_path
import io
import shutil
from PyPDF2 import PdfReader
import fitz

# Add parent directory to path to import the pipeline module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)  # Insert at beginning of path
from ai_processor_3.pipeline import Pipeline
from ai_processor_3.ocr_analyzer import OCRAnalyzer

# Set page configuration
st.set_page_config(page_title="PDF Technical Sheet Extractor", layout="wide")

# App title and description
st.title("PDF Technical Sheet Extractor")
st.markdown("""
Extracts technical sheets from PDF catalogs using AI analysis.
Upload a PDF catalog and the system will identify individual technical sheets and split them.
""")

# Sidebar controls
# st.sidebar.header("Settings")
# llm_model = st.sidebar.selectbox(
#     "LLM Model",
#     ["mistral-small-latest", "mistral-large-latest", "pixtral-large-latest"],
#     index=0,
# )
llm_model = "mistral-small-latest"

# Setup output directory
output_dir = "data/streamlit_output"
os.makedirs(output_dir, exist_ok=True)

# Get API key from environment
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    st.error("MISTRAL_API_KEY not found in environment variables!")
    st.stop()


# Function to convert PDF pages to images
def get_pdf_page_images(pdf_path, dpi=150):
    """Convert PDF pages to a list of images."""
    return convert_from_path(pdf_path, dpi=dpi)


# Function to display a specific page image
def display_page_image(pdf_path, page_num, dpi=150):
    """Display a specific page of a PDF as an image."""
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            st.error(f"PDF file not found: {pdf_path}")
            return None

        # Get total page count
        pdf = PdfReader(pdf_path)
        total_pages = len(pdf.pages)

        # Check if requested page is valid
        if page_num < 0 or page_num >= total_pages:
            st.warning(
                f"Page {page_num + 1} is out of range (PDF has {total_pages} pages)"
            )
            return None

        # Convert PDF page to image
        images = convert_from_path(
            pdf_path, dpi=dpi, first_page=page_num + 1, last_page=page_num + 1
        )
        if images and len(images) > 0:
            img_bytes = io.BytesIO()
            images[0].save(img_bytes, format="PNG")
            img_bytes.seek(0)
            return img_bytes
        return None
    except Exception as e:
        st.error(f"Error processing PDF page {page_num + 1}: {str(e)}")
        return None


# Function to display PDF
def display_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")


# Function to visualize technical sheets as a timeline
def visualize_technical_sheets(tech_sheets, total_pages):
    """Create a visualization of where technical sheets appear in the document."""
    # Create dataframe for visualization
    timeline_data = []

    for sheet in tech_sheets:
        product = sheet.get("product", "Unknown")
        pages = sheet.get("pages", [])

        # Convert pages from string format if needed
        if isinstance(pages, str):
            try:
                pages = json.loads(pages.replace("'", '"'))
            except:
                # Try to parse as a comma-separated list
                pages = [
                    int(p.strip()) for p in pages.strip("[]").split(",") if p.strip()
                ]

        # Add each page as a row in the dataframe
        for page in pages:
            timeline_data.append(
                {"Page": int(page), "Product": product}
            )

    # If no data was found
    if not timeline_data:
        st.warning("No technical sheet page data found in the analysis.")
        return

    # Create dataframe
    df = pd.DataFrame(timeline_data)

    # Also show a table summary
    summary_df = (
        df.groupby("Product")
        .agg(
            Pages=("Page", lambda x: sorted(list(x)))
        )
        .reset_index()
    )
    summary_df["Page Count"] = summary_df["Pages"].apply(len)
    st.dataframe(
        summary_df[["Product", "Pages"]],
        use_container_width=True,
    )


# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file to output directory
    permanent_pdf_path = os.path.join(output_dir, "current_analysis.pdf")
    with open(permanent_pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.success("File uploaded successfully!")

    # Get total page count
    pdf = PdfReader(permanent_pdf_path)
    total_pages = len(pdf.pages)
    st.info(f"Document has {total_pages} pages")


    process_button = st.button(
        "Extract Technical Sheets", use_container_width=True
    )

    if process_button:
        with st.spinner("Processing PDF..."):
            # Initialize the pipeline
            pipeline = Pipeline(
                mistral_api_key=api_key,
                llm_model=llm_model,
                output_dir=output_dir,
            )

            try:
                # Process the PDF and extract sheets
                extracted_paths = pipeline.extract_sheets_to_pdf(permanent_pdf_path)

                # Get tech sheet boundaries and analysis results
                boundaries = pipeline.extract_sheets(permanent_pdf_path)
                document_analysis = pipeline.get_document_analysis(
                    permanent_pdf_path
                )

                # Store results in session state for other tabs
                st.session_state.boundaries = boundaries
                st.session_state.document_analysis = document_analysis
                st.session_state.extracted_paths = extracted_paths
                st.session_state.original_pdf = permanent_pdf_path
                st.session_state.processed = True
                st.session_state.total_pages = total_pages

                # Show success message with count of sheets found
                st.success(
                    f"✅ Successfully extracted {len(extracted_paths)} technical sheets!"
                )

                # Show the tech sheet pages identified
                if (
                    hasattr(pipeline, "tech_sheet_pages")
                    and pipeline.tech_sheet_pages
                ):
                    st.write(
                        "Technical sheet pages identified:",
                        pipeline.tech_sheet_pages,
                    )

            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.exception(e)


    # # Create tabs for different views
    # tab_process, tab_extracted = st.tabs(
    #     ["Process", "Extracted Technical heets"]
    #     ["Extracted Technical heets"]
    # )

    # with tab_process:
    #     st.subheader("Process PDF")

    #     col1, col2 = st.columns([1, 1])

    #     with col1:
    #         process_button = st.button(
    #             "Extract Technical Sheets", use_container_width=True
    #         )

    #     if process_button:
    #         with st.spinner("Processing PDF..."):
    #             # Initialize the pipeline
    #             pipeline = Pipeline(
    #                 mistral_api_key=api_key,
    #                 llm_model=llm_model,
    #                 output_dir=output_dir,
    #             )

    #             try:
    #                 # Process the PDF and extract sheets
    #                 extracted_paths = pipeline.extract_sheets_to_pdf(permanent_pdf_path)

    #                 # Get tech sheet boundaries and analysis results
    #                 boundaries = pipeline.extract_sheets(permanent_pdf_path)
    #                 document_analysis = pipeline.get_document_analysis(
    #                     permanent_pdf_path
    #                 )

    #                 # Store results in session state for other tabs
    #                 st.session_state.boundaries = boundaries
    #                 st.session_state.document_analysis = document_analysis
    #                 st.session_state.extracted_paths = extracted_paths
    #                 st.session_state.original_pdf = permanent_pdf_path
    #                 st.session_state.processed = True
    #                 st.session_state.total_pages = total_pages

    #                 # Show success message with count of sheets found
    #                 st.success(
    #                     f"✅ Successfully extracted {len(extracted_paths)} technical sheets!"
    #                 )

    #                 # Show the tech sheet pages identified
    #                 if (
    #                     hasattr(pipeline, "tech_sheet_pages")
    #                     and pipeline.tech_sheet_pages
    #                 ):
    #                     st.write(
    #                         "Technical sheet pages identified:",
    #                         pipeline.tech_sheet_pages,
    #                     )

    #             except Exception as e:
    #                 st.error(f"Error processing PDF: {str(e)}")
    #                 st.exception(e)

    
    # st.subheader("Extracted Technical Sheets")

    if "processed" in st.session_state and st.session_state.processed:
        if (
            "boundaries" in st.session_state
            and len(st.session_state.boundaries) > 0
        ):
            # Create a list of detected technical sheets
            boundaries = st.session_state.boundaries

            # Add visualization at the top
            visualize_technical_sheets(boundaries, st.session_state.total_pages)
            
            # Display all technical sheets in full view instead of expanders
            for i, boundary in enumerate(boundaries):
                # Create a section for each technical sheet
                st.markdown(f"### {boundary.get('product', f'Technical Sheet {i + 1}')}")
                
                # Create two columns for metadata and thumbnails
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Show reason for detection
                    st.write(
                        "**Detection Reason:**",
                        boundary.get("reason", "Not provided"),
                    )
                    
                    # Add download button for this technical sheet
                    try:
                        # Get pages
                        pages = boundary.get("pages", [])
                        if isinstance(pages, str):
                            try:
                                pages = json.loads(pages.replace("'", '"'))
                            except:
                                pages = [
                                    int(p.strip())
                                    for p in pages.strip("[]").split(",")
                                    if p.strip()
                                ]
                                
                        # Find PDF path that corresponds to this technical sheet
                        pdf_path = next(
                            (
                                p
                                for p in st.session_state.extracted_paths
                                if Path(p).stem.startswith(
                                    f"sheet_{pages[0]}" if pages else ""
                                )
                            ),
                            None,
                        )

                        # if pdf_path and os.path.exists(pdf_path):
                        #     st.download_button(
                        #         label="📥 Download PDF",
                        #         data=open(pdf_path, "rb"),
                        #         file_name=f"{boundary.get('product', f'sheet_{i + 1}').replace('/', '-')}.pdf",
                        #         mime="application/pdf",
                        #         key=f"download_{i}",
                        #     )
                    except Exception as e:
                        st.error(f"Error accessing PDF file: {str(e)}")
                
                with col2:
                    # Display pages
                    pages = boundary.get("pages", [])
                    if isinstance(pages, str):
                        try:
                            pages = json.loads(pages.replace("'", '"'))
                        except:
                            pages = [
                                int(p.strip())
                                for p in pages.strip("[]").split(",")
                                if p.strip()
                            ]

                    # Create a grid of thumbnails for the pages
                    if pages:
                        st.write(f"**Pages:** {', '.join(str(p) for p in pages)}")
                        thumb_cols = st.columns(min(len(pages), 4))  # Max 4 pages per row

                        for j, page_num in enumerate(pages):
                            col_idx = j % len(thumb_cols)

                            with thumb_cols[col_idx]:
                                # Get image thumbnail
                                thumbnail = display_page_image(
                                    st.session_state.original_pdf,
                                    page_num - 1,
                                    dpi=75,
                                )
                                if thumbnail:
                                    # Add a caption that indicates clicking will zoom
                                    st.image(
                                        thumbnail,
                                        caption=f"Page {page_num} (click to zoom)",
                                        width=150,
                                        use_container_width=False,
                                    )
                                    
                                    # # Add a button to view the higher resolution version
                                    # if st.button("🔍 Zoom", key=f"zoom_{i}_{page_num}"):
                                    #     st.session_state[f"zoom_active_{i}_{page_num}"] = True
                                    
                                    # # If zoom is active, show the high-res image in a modal-like container
                                    # if st.session_state.get(f"zoom_active_{i}_{page_num}", False):
                                    #     with st.expander("High Resolution View", expanded=True):
                                    #         hi_res_img = display_page_image(
                                    #             st.session_state.original_pdf,
                                    #             page_num - 1,
                                    #             dpi=300,  # Higher DPI for zoomed view
                                    #         )
                                    #         if hi_res_img:
                                    #             st.image(
                                    #                 hi_res_img,
                                    #                 caption=f"Page {page_num} (High Resolution)",
                                    #                 use_container_width=True,
                                    #             )
                                    #             if st.button("Close", key=f"close_zoom_{i}_{page_num}"):
                                    #                 st.session_state[f"zoom_active_{i}_{page_num}"] = False
                                    #                 st.experimental_rerun()
                
                # Add a divider between technical sheets
                st.markdown("---")
        else:
            st.info("No technical sheets were detected in this document.")