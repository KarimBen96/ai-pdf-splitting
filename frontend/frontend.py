import os
import sys
import streamlit as st
import tempfile
import base64
import plotly.express as px
import pandas as pd
from pathlib import Path
from pdf2image import convert_from_path
import io
import shutil

# Add parent directory to path to import the pipeline module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)  # Insert at beginning of path
from ai_processor_2.pipeline import Pipeline

# Set page configuration
st.set_page_config(page_title="PDF Technical Sheet Extractor", layout="wide")

# App title and description
st.title("PDF Technical Sheet Extractor")
st.markdown("""
Extracts technical sheets from PDF catalogs using AI analysis.
Upload a PDF catalog and the system will identify individual technical sheets and split them.
""")

# LLM settings
use_llm = True
llm_model = "pixtral-large-latest"

# Processing settings
# confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.6, 0.05)
confidence_threshold = 0.6
# debug_mode = st.sidebar.checkbox("Debug mode", value=True)
debug_mode = True
output_dir = "data/streamlit_output"
os.makedirs(output_dir, exist_ok=True)

# Get API key
api_key = os.environ["MISTRAL_API_KEY"]


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

        # Get total page count first to avoid invalid page access
        from PyPDF2 import PdfReader

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
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Instead of using tempfile.NamedTemporaryFile which could get deleted,
    # Save the file to our output directory with a fixed name
    output_dir = "data/streamlit_output"
    os.makedirs(output_dir, exist_ok=True)

    # Use a permanent file path in our output directory
    permanent_pdf_path = os.path.join(output_dir, "current_analysis.pdf")

    # Save uploaded file to this permanent location
    with open(permanent_pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.success("File uploaded successfully!")

    # Create tabs for different views
    tab_process, tab_extracted = st.tabs(["Process", "Extracted PDFs"])

    with tab_process:
        st.subheader("Process PDF")

        if st.button("Extract Technical Sheets"):
            with st.spinner("Processing PDF..."):
                # Initialize the pipeline
                pipeline = Pipeline(
                    mistral_api_key=api_key,
                    llm_model=llm_model,
                    confidence_threshold=confidence_threshold,
                    use_llm=use_llm,
                    debug=debug_mode,
                    output_dir=output_dir,
                )

                try:
                    # Process the PDF and extract sheets
                    extracted_paths = pipeline.extract_sheets_to_pdf(permanent_pdf_path)

                    # Store results in session state for other tabs
                    st.session_state.boundaries = pipeline.extract_sheets(
                        permanent_pdf_path
                    )
                    st.session_state.extracted_paths = extracted_paths
                    st.session_state.original_pdf = permanent_pdf_path
                    st.session_state.processed = True

                    # Show success message with count of sheets found
                    st.success(
                        f"✅ Successfully extracted {len(extracted_paths)} technical sheets!"
                    )

                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    with tab_extracted:
        st.subheader("Extracted Pages")

        if "processed" in st.session_state and st.session_state.processed:
            if (
                "boundaries" in st.session_state
                and len(st.session_state.boundaries) > 0
            ):
                # Create a list of detected technical sheets
                boundaries = st.session_state.boundaries

                # First show a grid of thumbnails for all sheets
                st.subheader("Overview of All Technical Sheets")

                # Group pages by technical sheet
                for i, boundary in enumerate(boundaries):
                    st.markdown(f"### {boundary['title']}")

                    # Determine the page range for this technical sheet
                    current_page = boundary["page_number"]
                    next_boundary_page = float("inf")

                    for b in boundaries:
                        if (
                            b["page_number"] > current_page
                            and b["page_number"] < next_boundary_page
                        ):
                            next_boundary_page = b["page_number"]

                    # If it's the last sheet, show a few pages
                    if next_boundary_page == float("inf"):
                        page_range = range(
                            current_page, min(current_page + 5, 999)
                        )  # Limit to 5 pages max
                    else:
                        page_range = range(current_page, next_boundary_page)

                    # Create a row of columns for the pages in this sheet
                    page_cols = st.columns(
                        min(len(page_range), 4)
                    )  # Max 4 pages per row

                    # For each page in this technical sheet
                    for j, page_num in enumerate(page_range):
                        col_idx = j % len(page_cols)

                        with page_cols[col_idx]:
                            # Get image thumbnail with lower DPI for smaller size
                            thumbnail = display_page_image(
                                st.session_state.original_pdf, page_num, dpi=50
                            )
                            if thumbnail:
                                # Display page number and thumbnail
                                st.markdown(f"**Page {page_num + 1}**")
                                st.image(
                                    thumbnail,
                                    width=150,  # Set a fixed smaller width
                                )
                            else:
                                st.error(f"Could not load page {page_num + 1}")

                    # Add a download button for this technical sheet
                    try:
                        # Find the PDF path that corresponds to this technical sheet
                        pdf_path = next(
                            (
                                p
                                for p in st.session_state.extracted_paths
                                if Path(p).stem.startswith(
                                    f"sheet_{boundary['page_number']}"
                                )
                            ),
                            None,  # Default value if no match is found
                        )

                        if pdf_path and os.path.exists(pdf_path):
                            st.download_button(
                                label=f"📥 Download {boundary['title']} PDF",
                                data=open(pdf_path, "rb"),
                                file_name=f"{boundary['title'].replace('/', '-')}.pdf",
                                mime="application/pdf",
                                key=f"download_{i}",
                            )
                    except Exception as e:
                        st.error(f"Error accessing PDF file: {str(e)}")

                    # Add a separator between sheets
                    st.markdown("---")
            else:
                st.info("No technical sheets were detected in this document.")
        else:
            st.info("Please process a PDF file first.")

    # Clean up the temporary file when the app reruns
    if "original_pdf" in st.session_state:
        try:
            os.unlink(st.session_state.original_pdf)
        except Exception:
            pass  # Ignore errors in cleanup
