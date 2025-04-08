import os
import re
import json
import logging
import PyPDF2
from paper_extractors import PaperExtractors

# Configure logging
logger = logging.getLogger(__name__)


# Standalone cleaning functions
def clean_summary_text(text):
    """
    Clean summary text by removing emails, author affiliations, and other noise
    """
    if text is None:
        return ""

    # Remove email addresses
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)

    # Remove author affiliations with numbers
    text = re.sub(r'\d+The authors are with[^.]+\.', '', text)
    text = re.sub(r'\d+Department of[^.]+\.', '', text)
    text = re.sub(r'\d+University[^.]+\.', '', text)
    text = re.sub(r'\d+Institute[^.]+\.', '', text)

    # Remove author affiliation blocks
    text = re.sub(r'(?:The authors are|Authors are|is|are) (?:with|from|at) [^.]+\.', '', text)

    # Remove line numbers and section numbers
    text = re.sub(r'^\d+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[IVX]+\.\s+', '', text, flags=re.MULTILINE)

    # Remove citation markers
    text = re.sub(r'\[\d+\]', '', text)  # [1]
    text = re.sub(r'\(\d+\)', '', text)  # (1)
    text = re.sub(r'\[\w+\s*\d{4}\]', '', text)  # [Smith 2020]

    # Remove figure and table references
    text = re.sub(r'(?:Fig|Figure|Table)\.?\s*\d+', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove numeric superscripts (often used for affiliations)
    text = re.sub(r'(?<=[a-zA-Z])\d+', '', text)

    # Remove text inside parentheses if it looks like a citation
    text = re.sub(r'\([^)]{1,50}\d{4}[^)]{0,10}\)', '', text)

    # Remove common paper section markers
    section_headers = [
        'Abstract', 'Introduction', 'Background', 'Related Work',
        'Methodology', 'Methods', 'Experiments', 'Results',
        'Discussion', 'Conclusion', 'References'
    ]
    for header in section_headers:
        text = re.sub(f'(?:^|\n)\s*{header}\s*\n', ' ', text, flags=re.IGNORECASE)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")

    # Fix common OCR/PDF extraction issues
    text = text.replace('- ', '')  # Hyphenated words at line breaks

    # Remove trailing citations
    text = re.sub(r'\s*\[\d+(?:,\s*\d+)*\]\s*$', '', text)

    # Fix spacing after periods
    text = re.sub(r'\.(?=[A-Z])', '. ', text)

    # Final whitespace cleanup and strip
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_executive_summary(text):
    """
    Special cleaner for executive summaries to aggressively remove
    author information and other non-summary content.
    """
    if not text:
        return ""

    # Remove all author information lines
    text = re.sub(r'\d+The authors are with[^.]*\.', '', text)
    text = re.sub(r'(?:Department|University|Institute)[^.]*\.', '', text)
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '', text)

    # Remove numbered affiliations and citations
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(\d+\)', '', text)

    # Remove superscript numbers (often used for affiliations)
    text = re.sub(r'(?<=[a-zA-Z])\d+', '', text)

    # Remove lines starting with authors/locations
    text = re.sub(r'(?:^|\n)[A-Z][a-z]+ [A-Z][a-z]+ is with[^.]*\.', '', text)
    text = re.sub(r'(?:^|\n)[^,]*,[^,]*,[^.]*\.', '', text)

    # Remove various location indicators
    text = re.sub(r'(?:TX|USA|Greece|Attica)[^.]*\.', '', text)

    # Clean up section headers
    text = re.sub(r'(?:^|\n)\s*(?:Abstract|Introduction|I\.|II\.)[^\n]*\n', '', text)

    # Remove "previous work" references
    text = re.sub(r'(?:previous|prior) work[^.]*\.', '', text)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_title(title_text):
    """Extract just the title from a research paper by identifying the boundary
    between title and author list.
    """
    if not title_text:
        return "Untitled Document"

    # First normalize spacing
    title_text = re.sub(r'\s+', ' ', title_text).strip()

    # Check for common indicators of author list beginning

    # 1. Look for a single uppercase letter followed by period (initial)
    # This is a very strong indicator of author names in academic papers
    initial_pattern = r'(?<!\.[A-Z])\s+([A-Z])\.\s+'
    match = re.search(initial_pattern, title_text)
    if match and match.start() > 10:  # Ensure we're not in the middle of an acronym
        # Find the beginning of the name with this initial
        for i in range(match.start(), 0, -1):
            if (i == 0 or title_text[i - 1].isspace()) and title_text[i].isupper():
                # This looks like the start of an author name
                return title_text[:i].strip()

    # 2. Look for a common author-list pattern with multiple single initials
    # Like "K. C. Smith" or "J. K. Wu"
    multiple_initials = r'([A-Z])\.\s+([A-Z])\.'
    match = re.search(multiple_initials, title_text)
    if match and match.start() > 10:
        # Find the start of this name
        for i in range(match.start(), 0, -1):
            if (i == 0 or title_text[i - 1].isspace()) and title_text[i].isupper():
                return title_text[:i].strip()

    # 3. Look for typical comma-separated list of author names
    author_list = r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)+)(?:,\s+[A-Z][a-z]+)'
    match = re.search(author_list, title_text)
    if match and match.start() > 10:
        # Find the space before this name
        for i in range(match.start(), 0, -1):
            if title_text[i].isspace():
                return title_text[:i].strip()

    # 4. Look for transition from lowercase to uppercase at word boundaries
    # This often indicates where title (ending in lowercase) transitions to author names
    lowercase_to_capital = r'([a-z])\s+([A-Z][a-z]+)'
    match = re.search(lowercase_to_capital, title_text)
    if match and match.start() > 10:
        remaining = title_text[match.end():]
        # Check if what follows looks like author names (commas, initials, etc.)
        if (',' in remaining[:50] or
                re.search(r'[A-Z]\.', remaining[:50]) or
                ' and ' in remaining[:50].lower()):
            return title_text[:match.start() + 1].strip()

    # 5. Look for institutional affiliations which often follow author names
    institution_pattern = r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+))(?:\s*a[0-9]*|,\s*[a-z]|,\s*\*)(?:Department|University|Institute)'
    match = re.search(institution_pattern, title_text)
    if match and match.start() > 10:
        # Find the start of this section
        for i in range(match.start(), 0, -1):
            if title_text[i].isspace():
                return title_text[:i].strip()

    # 6. Look for typical department or university name patterns
    dept_patterns = [r'\s+a?Department of', r'\s+a?University', r'\s+a?Institute', r'\s+a[0-9]']
    for pattern in dept_patterns:
        match = re.search(pattern, title_text)
        if match and match.start() > 10:
            return title_text[:match.start()].strip()

    # Remove "Abstract" section if present
    if "Abstract" in title_text:
        title_text = title_text.split("Abstract")[0].strip()

    # Final cleanup
    title_text = re.sub(r'(\w+)-\s+(\w+)', r'\1-\2', title_text)  # fix hyphenated words
    title_text = re.sub(r'(?<=[a-zA-Z])\d+', '', title_text)  # remove superscripts
    title_text = re.sub(r'^\d+\s*', '', title_text)  # leading numbers
    title_text = re.sub(r'\s+', ' ', title_text).strip()  # normalize spacing

    return title_text


def format_paragraphs(text):
    """
    Format text into proper paragraphs with Markdown.
    """
    if not text:
        return ""

    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Group sentences into paragraphs (roughly 2-3 sentences per paragraph)
    paragraphs = []
    current_paragraph = []

    for sentence in sentences:
        current_paragraph.append(sentence)
        if len(current_paragraph) >= 3 or len(' '.join(current_paragraph)) > 250:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []

    # Add any remaining sentences
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))

    # Join paragraphs with double newlines for Markdown
    return '\n\n'.join(paragraphs)


def highlight_key_phrases(text, keywords):
    """
    Highlight key phrases in the text based on keywords and important phrases.
    """
    if not text or not keywords:
        return text

    # Important research paper phrases to highlight
    important_phrases = [
        "novel", "propose", "contribution", "outperform", "significant",
        "demonstrate", "improve", "results show", "we present", "approach",
        "method", "framework", "system", "algorithm", "performance",
        "experimental results", "state-of-the-art", "evaluate"
    ]

    # Combine keywords and important phrases
    all_terms = set([k.lower() for k in keywords] + important_phrases)

    # Create a copy of the text for highlighting
    highlighted_text = text

    # Highlight each term
    for term in all_terms:
        if len(term) <= 3:  # Skip very short terms
            continue

        # Case-insensitive replacement with Markdown bold
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        highlighted_text = pattern.sub(r'**\g<0>**', highlighted_text)

    return highlighted_text


class PaperProcessor:
    """Class to process research papers into structured information and summaries"""

    def __init__(self, keybert_model=None):
        """Initialize processor with optional models"""
        self.keybert_model = keybert_model

    def set_keybert_model(self, model):
        """Set the KeyBERT model to use for keyword extraction"""
        self.keybert_model = model

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file with enhanced error handling"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                logger.info(f"PDF has {len(pdf_reader.pages)} pages")

                for page_num in range(len(pdf_reader.pages)):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:  # Only add if we got actual text
                            text += page_text + "\n"
                    except Exception as e:
                        logger.error(f"Error extracting text from page {page_num}: {e}")

                # If no text was extracted, try another approach
                if not text.strip():
                    logger.warning("No text extracted using standard method, trying alternative approach")
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        # Try accessing raw content
                        if '/Contents' in page:
                            text += f"Page {page_num + 1} content exists but couldn't be extracted properly\n"

            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def extract_text_from_txt(self, txt_path):
        """Extract text from a TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(f"Successfully read text file with utf-8 encoding: {len(text)} chars")
            return text
        except UnicodeDecodeError:
            # Try another encoding if utf-8 fails
            try:
                with open(txt_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                logger.info(f"Successfully read text file with latin-1 encoding: {len(text)} chars")
                return text
            except Exception as e:
                logger.error(f"Error reading with latin-1 encoding: {e}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {e}")
            return ""

    def clean_text(self, text):
        """Clean and preprocess the extracted text"""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove non-printable characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Fix broken words with hyphens
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # Handle bullet points and numbered lists
        text = re.sub(r'•\s*', '• ', text)

        return text.strip()

    def process_paper(self, file_path):
        """Process a research paper and generate a structured summary"""
        logger.info(f"Processing paper: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": "File not found"}

        # Extract text from file
        file_extension = file_path.split('.')[-1].lower()
        text = ""

        if file_extension == 'pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_extension == 'txt':
            text = self.extract_text_from_txt(file_path)
        else:
            return {"error": "Unsupported file format"}

        # Check if text extraction succeeded
        if not text or len(text.strip()) < 100:
            logger.error(f"Failed to extract meaningful text from {file_path}")
            return {
                "error": "Could not extract sufficient text from the document. Please check if the PDF is text-based rather than scanned images."
            }

        logger.info(f"Extracted text length: {len(text)}")

        # Clean the text
        text = self.clean_text(text)
        logger.info(f"Cleaned text length: {len(text)}")

        # Create extractors instance
        extractors = PaperExtractors()

        # Extract title
        title = extractors.extract_title(text)
        # Clean the title
        title = clean_title(title)
        logger.info(f"Extracted title: {title}")

        # Extract authors
        authors = extractors.extract_authors(text, title)
        logger.info(f"Extracted authors: {authors}")

        # Extract institutions
        institutions = extractors.extract_institutions(text)
        logger.info(f"Extracted institutions: {institutions}")

        # Identify sections
        sections = extractors.identify_sections(text)
        logger.info(f"Identified sections: {list(sections.keys())}")

        # Extract keywords
        keywords = []
        if self.keybert_model:
            keywords = extractors.extract_keyphrases(text, keybert_model=self.keybert_model)
            logger.info(f"Extracted {len(keywords)} keywords")

        # Extract references
        references = extractors.extract_references(text)
        logger.info(f"Extracted {len(references)} references")

        # Initialize result dictionary with structured information
        result = {
            "title": title,
            "authors": authors,
            "institutions": institutions,
            "keywords": keywords,
            "sections": {},  # Will only contain summaries, not full text
            "references": references,
            "text_extraction_success": True,
            "processing_info": f"Processed document with {len(text)} characters"
        }

        # Key sections to summarize
        important_sections = ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']

        # Try to summarize each important section
        for section in important_sections:
            if section in sections and sections[section] and len(sections[section].strip()) > 100:
                max_length = 150 if section in ['abstract', 'conclusion'] else 250
                section_text = sections[section]

                # Create summary
                result["sections"][section] = extractors.summarize_section(section_text, max_length)
                logger.info(f"Created summary for {section}: {len(result['sections'][section])} chars")

        # If we couldn't identify proper sections, summarize the full text
        if not result["sections"] and "full_text" in sections:
            logger.warning("No sections identified, summarizing full text instead")
            summary = extractors.summarize_section(sections["full_text"], 300)
            if summary:
                result["sections"]["document_summary"] = summary

        # If still no content, create a fallback summary
        if not result["sections"]:
            logger.warning("Creating fallback summary from the beginning of the text")
            # Use the beginning of the document
            beginning = ' '.join(text.split()[:500])
            result["sections"]["document_summary"] = extractors.summarize_section(beginning, 200)

        # Create an executive summary
        if 'abstract' in result["sections"]:
            result["executive_summary"] = clean_executive_summary(clean_summary_text(result["sections"]["abstract"]))
        elif 'introduction' in result["sections"]:
            result["executive_summary"] = clean_executive_summary(
                clean_summary_text(result["sections"]["introduction"]))
        elif "document_summary" in result["sections"]:
            result["executive_summary"] = clean_executive_summary(
                clean_summary_text(result["sections"]["document_summary"]))

        # Extract problem statement from introduction or beginning of text
        if 'introduction' in result["sections"]:
            # Usually the first paragraph of the introduction contains the problem statement
            intro_text = sections.get('introduction', '')
            paragraphs = re.split(r'\n\s*\n', intro_text)
            if paragraphs:
                result["problem_statement"] = clean_summary_text(paragraphs[0][:300])  # Limit to 300 chars
            else:
                result["problem_statement"] = "Problem statement could not be extracted."
        else:
            result["problem_statement"] = "Problem statement could not be extracted from the document."

        # Extract methodology summary
        if 'methods' in result["sections"]:
            result["methodology"] = clean_summary_text(result["sections"]["methods"])
        else:
            # Try to find methodology in the text using keywords
            methodology_text = extractors.extract_methodology(text)
            if methodology_text:
                # Ensure methodology doesn't have the full text
                if len(methodology_text) > 500:
                    methodology_text = methodology_text[:497] + "..."
                result["methodology"] = clean_summary_text(methodology_text)
            else:
                result["methodology"] = "Methodology details could not be extracted from the document."

        # Extract key findings
        if 'results' in result["sections"]:
            result["findings"] = clean_summary_text(result["sections"]["results"])
        elif 'discussion' in result["sections"]:
            result["findings"] = clean_summary_text(result["sections"]["discussion"])
        else:
            result["findings"] = "Key findings could not be extracted from the document."

        # Extract conclusion
        if 'conclusion' in result["sections"]:
            result["conclusion"] = clean_summary_text(result["sections"]["conclusion"])
        else:
            result["conclusion"] = "Conclusion could not be extracted from the document."

        # IMPORTANT: Remove the raw text to prevent it from being included in the output
        # Delete full_text from sections dictionary
        if "full_text" in sections:
            del sections["full_text"]

        # CRITICAL: Verify no field has the entire document text
        max_field_length = 1000  # Maximum reasonable length for any field
        for key, value in result.items():
            if isinstance(value, str) and len(value) > max_field_length:
                # Field is too long, likely contains full text
                logger.warning(f"Field {key} contains {len(value)} chars, truncating")
                result[key] = value[:max_field_length - 3] + "..."

        # Replace the sections dict with one that only contains summaries
        summarized_sections = {}
        for section_name, section_content in result["sections"].items():
            # Make sure section content is not too long
            if len(section_content) > max_field_length:
                section_content = section_content[:max_field_length - 3] + "..."
            summarized_sections[section_name] = section_content

        result["sections"] = summarized_sections

        return result

    def create_narrative_summary(self, paper_data):
        """
        Create an enhanced narrative summary from extracted paper data

        Args:
            paper_data (dict): The extracted paper data

        Returns:
            dict: Enhanced paper data with narrative summaries
        """
        enhanced_data = paper_data.copy()

        # Create a more structured narrative of key innovations/contributions
        innovations = []

        # Check abstract and introduction for innovation statements
        abstract = paper_data.get("sections", {}).get("abstract", "")
        intro = paper_data.get("sections", {}).get("introduction", "")

        # Look for statements about contributions or proposed methods
        contribution_patterns = [
            r'we propose (.*?)(\.|\n)',
            r'our (approach|method|contribution|work) (.*?)(\.|\n)',
            r'this paper (presents|proposes|introduces|describes) (.*?)(\.|\n)',
            r'novel (.*?)(\.|\n)',
            r'key contribution[s]? (.*?)(\.|\n)'
        ]

        text_to_search = abstract + " " + intro

        for pattern in contribution_patterns:
            matches = re.findall(pattern, text_to_search, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    # Get the captured group with the actual innovation
                    innovation = match[1].strip()
                else:
                    innovation = match.strip()

                if len(innovation) > 10 and innovation not in innovations:
                    # Clean up the innovation text
                    innovation = self._clean_innovation_text(innovation)
                    innovations.append(innovation)

        # If no innovations found via patterns, use keywords
        if not innovations and paper_data.get("keywords"):
            # Use keywords to construct an innovation statement
            keywords = paper_data.get("keywords", [])[:3]
            if keywords:
                innovations.append(f"A novel approach utilizing {', '.join(keywords)}")

        # Limit to 3 most important innovations
        enhanced_data["key_innovations"] = innovations[:3]

        # Create a technical approach summary
        technical_approach = []

        # Extract from methodology
        methodology = paper_data.get("sections", {}).get("methods", "")
        if not methodology:
            methodology = paper_data.get("methodology", "")

        # Split into sentences and select the most relevant ones
        if methodology:
            sentences = re.split(r'(?<=[.!?])\s+', methodology)
            technical_sentences = []

            # Look for sentences describing the technical approach
            for sentence in sentences:
                if any(term in sentence.lower() for term in
                       ['method', 'approach', 'model', 'algorithm', 'framework', 'architecture',
                        'system', 'technique', 'process', 'procedure', 'network']):
                    technical_sentences.append(sentence)

            # If too many sentences, select a few representative ones
            if len(technical_sentences) > 3:
                technical_approach = technical_sentences[:3]
            else:
                technical_approach = technical_sentences

        enhanced_data["technical_approach"] = technical_approach

        # Create significance summary
        significance = []

        # Extract from conclusion or discussion
        conclusion = paper_data.get("sections", {}).get("conclusion", "")
        if not conclusion:
            conclusion = paper_data.get("conclusion", "")

        discussion = paper_data.get("sections", {}).get("discussion", "")

        search_text = conclusion + " " + discussion

        # Look for significance statements with improved patterns
        significance_patterns = [
            r'(important|significant|crucial|essential|valuable) (.*?)(\.|\n)',
            r'impact[s]? (.*?)(\.|\n)',
            r'improve[s]? (.*?)(\.|\n)',
            r'outperform[s]? (.*?)(\.|\n)',
            r'better than (.*?)(\.|\n)',
            r'advance[s]? (.*?)(\.|\n)',
            r'contribution[s]? (.*?)(\.|\n)'
        ]

        for pattern in significance_patterns:
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Get the captured group
                    sig = match[1].strip()
                else:
                    sig = match.strip()

                if len(sig) > 10 and sig not in significance:
                    # Clean up the significance statement
                    sig = self._clean_significance_text(sig)
                    significance.append(sig)

        # If we found too few significance statements, look in the results section
        if len(significance) < 2 and "results" in paper_data.get("sections", {}):
            results_text = paper_data["sections"]["results"]
            for pattern in significance_patterns:
                matches = re.findall(pattern, results_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        sig = match[1].strip()
                    else:
                        sig = match.strip()

                    if len(sig) > 10 and sig not in significance:
                        sig = self._clean_significance_text(sig)
                        significance.append(sig)

        enhanced_data["significance"] = significance[:3]  # Limit to 3

        return enhanced_data

    def _clean_innovation_text(self, text):
        """Clean up innovation text for better readability"""
        # Remove citations, reference marks, etc.
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d+\)', '', text)

        # Capitalize first letter
        if text and len(text) > 0:
            text = text[0].upper() + text[1:]

        # Ensure the text ends with proper punctuation
        if text and not text.endswith(('.', '!', '?')):
            text += '.'

        return text

    def _clean_significance_text(self, text):
        """Clean up significance text for better readability"""
        # Remove citations, reference marks, etc.
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d+\)', '', text)

        # Capitalize first letter
        if text and len(text) > 0:
            text = text[0].upper() + text[1:]

        # Ensure the text ends with proper punctuation
        if text and not text.endswith(('.', '!', '?')):
            text += '.'

        return text

    def create_comprehensive_summary(self, paper_data):
        """
        Create a comprehensive textual summary of the paper with enhanced formatting

        Args:
            paper_data (dict): The extracted paper data

        Returns:
            str: A well-formatted, comprehensive summary in Markdown format
        """
        # First, enhance the data with narrative summaries
        enhanced_data = self.create_narrative_summary(paper_data)

        # Start building the summary with Markdown formatting
        summary_parts = []

        # Clean and extract title, removing author information
        raw_title = enhanced_data.get("title", "Untitled Document")
        # Extract just the main title, removing author info
        title_parts = raw_title.split("Abstract")
        title = title_parts[0] if len(title_parts) > 0 else raw_title

        # Clean up the title
        title = re.sub(r'\d+', '', title)
        title = clean_summary_text(title)

        # Get authors and institutions
        authors = enhanced_data.get("authors", "Unknown authors")
        institutions = enhanced_data.get("institutions", "")

        # Clean authors and institutions
        authors = clean_summary_text(authors)
        institutions = clean_summary_text(institutions)

        # Add title with proper formatting
        summary_parts.append(f"# {title}")

        # Add authors and institutions
        summary_parts.append(f"**Authors**: {authors}")
        if institutions:
            summary_parts.append(f"**Institutions**: {institutions}")
        summary_parts.append("")

        # Extract the abstract
        abstract = ""
        # Try to get from Abstract field
        if "Abstract" in raw_title:
            abstract_part = raw_title.split("Abstract")
            if len(abstract_part) > 1:
                abstract_text = abstract_part[1]
                for header in ["Introduction", "I.", "II.", "Background", "Related"]:
                    if header in abstract_text:
                        abstract_text = abstract_text.split(header)[0]
                abstract = abstract_text.strip()

        # If we couldn't extract from title, try from sections
        if not abstract:
            abstract = enhanced_data.get("sections", {}).get("abstract", "")

        # If still no abstract, use executive summary
        if not abstract:
            abstract = enhanced_data.get("executive_summary", "")

        if abstract:
            summary_parts.append("## Abstract")
            # Clean and format the abstract
            cleaned_abstract = clean_executive_summary(clean_summary_text(abstract))
            formatted_abstract = format_paragraphs(cleaned_abstract)
            summary_parts.append(formatted_abstract)
            summary_parts.append("")

        # Key Concepts/Keywords
        keywords = enhanced_data.get("keywords", [])
        if keywords:
            summary_parts.append("## Key Concepts")
            # Clean each keyword
            cleaned_keywords = [clean_summary_text(kw) for kw in keywords]
            # Filter out empty or very short keywords
            cleaned_keywords = [kw for kw in cleaned_keywords if len(kw) > 3]
            keywords_formatted = ", ".join([f"**{kw}**" for kw in cleaned_keywords[:10]])
            summary_parts.append(keywords_formatted)
            summary_parts.append("")

        # Problem Statement
        problem = enhanced_data.get("problem_statement", "")
        if problem and problem != "Problem statement could not be extracted from the document." and len(problem) > 20:
            summary_parts.append("## Problem Statement")
            # Clean and highlight
            cleaned_problem = clean_summary_text(problem)
            highlighted_problem = highlight_key_phrases(cleaned_problem, keywords)
            summary_parts.append(highlighted_problem)
            summary_parts.append("")

        # Key Innovations with bullet points
        innovations = enhanced_data.get("key_innovations", [])
        if innovations:
            summary_parts.append("## Key Innovations")
            for innovation in innovations:
                cleaned_innovation = clean_summary_text(innovation)
                if cleaned_innovation and len(cleaned_innovation) > 5:
                    summary_parts.append(f"* {cleaned_innovation}")
            summary_parts.append("")

        # Technical Approach
        tech_approach = enhanced_data.get("technical_approach", [])
        if tech_approach:
            summary_parts.append("## Technical Approach")
            for i, approach in enumerate(tech_approach, 1):
                cleaned_approach = clean_summary_text(approach)
                if cleaned_approach and len(cleaned_approach) > 5:
                    summary_parts.append(f"{i}. {cleaned_approach}")
            summary_parts.append("")

        # Methodology
        methodology = enhanced_data.get("methodology", "")
        if methodology and methodology != "Methodology details could not be extracted from the document." and len(methodology) > 20:
            summary_parts.append("## Methodology")
            # Clean and format
            cleaned_methodology = clean_summary_text(methodology)
            formatted_methodology = format_paragraphs(cleaned_methodology)
            summary_parts.append(formatted_methodology)
            summary_parts.append("")

        # Findings/Results
        findings = enhanced_data.get("findings", "")
        if findings and findings != "Key findings could not be extracted from the document." and len(findings) > 20:
            summary_parts.append("## Key Findings")
            # Clean, format, and highlight
            cleaned_findings = clean_summary_text(findings)
            formatted_findings = format_paragraphs(cleaned_findings)
            highlighted_findings = highlight_key_phrases(formatted_findings, keywords)
            summary_parts.append(highlighted_findings)
            summary_parts.append("")

        # Significance with better organization
        significance = enhanced_data.get("significance", [])
        if significance:
            summary_parts.append("## Research Significance")
            for sig in significance:
                cleaned_sig = clean_summary_text(sig)
                if cleaned_sig and len(cleaned_sig) > 5:
                    summary_parts.append(f"* {cleaned_sig}")
            summary_parts.append("")

        # Conclusion
        conclusion = enhanced_data.get("conclusion", "")
        if conclusion and conclusion != "Conclusion could not be extracted from the document." and len(conclusion) > 20:
            summary_parts.append("## Conclusions")
            # Clean and format
            cleaned_conclusion = clean_summary_text(conclusion)
            formatted_conclusion = format_paragraphs(cleaned_conclusion)
            summary_parts.append(formatted_conclusion)
            summary_parts.append("")

        # Add References section
        references = enhanced_data.get("references", [])
        if references:
            summary_parts.append("## References")
            # Display formatted references
            for i, ref in enumerate(references):
                # Format reference properly if it doesn't already have a number
                if not ref.strip().startswith('[') and not ref.strip().startswith('('):
                    ref_text = f"{i + 1}. {ref}"
                else:
                    ref_text = ref

                summary_parts.append(f"{ref_text}")

            summary_parts.append("")

        # Join all parts with proper spacing
        comprehensive_summary = "\n".join(summary_parts)

        return comprehensive_summary

    def prepare_template_data(self, paper_data):
        """
        Prepare data for the HTML template, ensuring only summaries are included

        Args:
            paper_data (dict): The extracted paper data

        Returns:
            dict: Data ready for template rendering (without full text)
        """
        # First enhance the data with narrative summaries
        enhanced_data = self.create_narrative_summary(paper_data)

        # Create template-ready data structure
        template_data = {
            "title": clean_title(enhanced_data.get("title", "Untitled Document")),
            "authors": clean_summary_text(enhanced_data.get("authors", "Authors not specified")),
            "institutions": clean_summary_text(enhanced_data.get("institutions", "Institutions not specified")),
            "key_concepts": enhanced_data.get("keywords", [])[:10],
            "executive_summary": clean_executive_summary(
                clean_summary_text(enhanced_data.get("executive_summary", "Summary not available"))),
            "problem_statement": clean_summary_text(
                enhanced_data.get("problem_statement", "Problem statement not available")),
            "methodology": clean_summary_text(enhanced_data.get("methodology", "Methodology not available")),
            "findings": clean_summary_text(enhanced_data.get("findings", "Findings not available")),
            "conclusion": clean_summary_text(enhanced_data.get("conclusion", "Conclusion not available")),
            "references": [],
            "methodology_image": None,  # No image by default

            # Narrative sections
            "key_innovations": [clean_summary_text(i) for i in enhanced_data.get("key_innovations", [])],
            "technical_approach": [clean_summary_text(t) for t in enhanced_data.get("technical_approach", [])],
            "significance": [clean_summary_text(s) for s in enhanced_data.get("significance", [])]
        }

        # Format references for display - limit to 10
        formatted_refs = []
        for i, ref in enumerate(enhanced_data.get("references", [])[:10]):
            formatted_refs.append(f"{i + 1}. {ref}")

        template_data["references"] = formatted_refs

        # Clean up the data to ensure proper rendering
        for key, value in template_data.items():
            if isinstance(value, str):
                # Clean the string by removing excessive whitespace and fixing line breaks
                clean_value = re.sub(r'\s+', ' ', value).strip()
                clean_value = clean_value.replace('\n', ' ')
                template_data[key] = clean_value

        # IMPORTANT: Ensure we're not including any full text data
        # Remove any problematic keys that might contain full text
        keys_to_exclude = ["full_text", "sections", "text"]
        for key in keys_to_exclude:
            if key in template_data:
                del template_data[key]

        return template_data

    def process_paper_for_template(self, file_path):
        """
        Process a paper and prepare the data for template rendering

        Args:
            file_path (str): Path to the paper file

        Returns:
            dict: Template-ready data
        """
        # Process the paper with the existing method
        paper_data = self.process_paper(file_path)

        # Check if processing was successful
        if "error" in paper_data:
            return paper_data

        # Prepare the data for template rendering
        template_data = self.prepare_template_data(paper_data)

        return template_data
