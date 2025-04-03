"""
Module for processing research papers.
This handles the main workflow for processing PDF/TXT files into structured summaries.
"""

import os
import re
import json
import logging
import PyPDF2
from paper_extractors import PaperExtractors

# Configure logging
logger = logging.getLogger(__name__)


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

        # Remove page numbers and headers/footers (simplified approach)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Fix broken words with hyphens
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # Handle bullet points and numbered lists more gracefully
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

        # Extract information using the PaperExtractors class
        extractors = PaperExtractors()

        # Extract title
        title = extractors.extract_title(text)
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
            "sections": {},
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

        # Create an executive summary from the abstract or introduction
        if 'abstract' in result["sections"]:
            result["executive_summary"] = result["sections"]["abstract"]
        elif 'introduction' in result["sections"]:
            result["executive_summary"] = result["sections"]["introduction"]
        elif "document_summary" in result["sections"]:
            result["executive_summary"] = result["sections"]["document_summary"]

        # Extract problem statement from introduction or beginning of text
        if 'introduction' in result["sections"]:
            # Usually the first paragraph of the introduction contains the problem statement
            intro_text = sections.get('introduction', '')
            paragraphs = re.split(r'\n\s*\n', intro_text)
            if paragraphs:
                result["problem_statement"] = paragraphs[0]
            else:
                result["problem_statement"] = "Problem statement could not be extracted."
        else:
            result["problem_statement"] = "Problem statement could not be extracted from the document."

        # Extract methodology summary
        if 'methods' in result["sections"]:
            result["methodology"] = result["sections"]["methods"]
        else:
            # Try to find methodology in the text using keywords
            methodology_text = extractors.extract_methodology(text)
            if methodology_text:
                result["methodology"] = methodology_text
            else:
                result["methodology"] = "Methodology details could not be extracted from the document."

        # Extract key findings
        if 'results' in result["sections"]:
            result["findings"] = result["sections"]["results"]
        elif 'discussion' in result["sections"]:
            result["findings"] = result["sections"]["discussion"]
        else:
            result["findings"] = "Key findings could not be extracted from the document."

        # Extract conclusion
        if 'conclusion' in result["sections"]:
            result["conclusion"] = result["sections"]["conclusion"]
        else:
            result["conclusion"] = "Conclusion could not be extracted from the document."

        return result

    def create_narrative_summary(self, paper_data):
        """
        Create an enhanced narrative summary from extracted paper data with improved
        structure and content organization.

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

        # Look for statements about contributions, novelty or proposed methods
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
                if isinstance(match, tuple):
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

        # Create significance summary with better quality
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

    def create_comprehensive_summary(self, paper_data):
        """
        Create a comprehensive textual summary of the paper with enhanced formatting
        and better content organization.

        Args:
            paper_data (dict): The extracted paper data

        Returns:
            str: A well-formatted, comprehensive summary in Markdown format
        """
        # First, enhance the data with narrative summaries
        enhanced_data = self.create_narrative_summary(paper_data)

        # Start building the summary with Markdown formatting
        summary_parts = []

        # Title and authors with proper formatting
        title = enhanced_data.get("title", "Untitled Document")
        authors = enhanced_data.get("authors", "Unknown authors")
        institutions = enhanced_data.get("institutions", "")

        summary_parts.append(f"# {title}")
        summary_parts.append(f"**Authors**: {authors}")
        if institutions:
            summary_parts.append(f"**Institutions**: {institutions}")
        summary_parts.append("")

        # Executive Summary with improved formatting
        exec_summary = enhanced_data.get("executive_summary", "")
        if exec_summary:
            summary_parts.append("## Executive Summary")
            # Format the summary with paragraph breaks
            formatted_summary = self._format_paragraphs(exec_summary)
            summary_parts.append(formatted_summary)
            summary_parts.append("")

        # Key Concepts/Keywords
        keywords = enhanced_data.get("keywords", [])
        if keywords:
            summary_parts.append("## Key Concepts")
            keywords_formatted = ", ".join([f"**{kw}**" for kw in keywords[:10]])
            summary_parts.append(keywords_formatted)
            summary_parts.append("")

        # Problem Statement with enhanced emphasis
        problem = enhanced_data.get("problem_statement", "")
        if problem and len(problem) > 20:
            summary_parts.append("## Problem Statement")
            # Highlight key phrases in the problem statement
            highlighted_problem = self._highlight_key_phrases(problem, keywords)
            summary_parts.append(highlighted_problem)
            summary_parts.append("")

        # Key Innovations with bullet points
        innovations = enhanced_data.get("key_innovations", [])
        if innovations:
            summary_parts.append("## Key Innovations")
            for innovation in innovations:
                summary_parts.append(f"* {innovation}")
            summary_parts.append("")

        # Technical Approach with better organization
        tech_approach = enhanced_data.get("technical_approach", [])
        if tech_approach:
            summary_parts.append("## Technical Approach")
            for i, approach in enumerate(tech_approach, 1):
                summary_parts.append(f"{i}. {approach}")
            summary_parts.append("")

        # Methodology with better formatting
        methodology = enhanced_data.get("methodology", "")
        if methodology and len(methodology) > 20:
            summary_parts.append("## Methodology")
            # Format the methodology with paragraph breaks
            formatted_methodology = self._format_paragraphs(methodology)
            summary_parts.append(formatted_methodology)
            summary_parts.append("")

        # Findings/Results with enhanced formatting
        findings = enhanced_data.get("findings", "")
        if findings and len(findings) > 20:
            summary_parts.append("## Key Findings")
            # Format the findings with paragraph breaks and highlighting
            formatted_findings = self._format_paragraphs(findings)
            highlighted_findings = self._highlight_key_phrases(formatted_findings, keywords)
            summary_parts.append(highlighted_findings)
            summary_parts.append("")

        # Significance with better organization
        significance = enhanced_data.get("significance", [])
        if significance:
            summary_parts.append("## Research Significance")
            for sig in significance:
                summary_parts.append(f"* {sig}")
            summary_parts.append("")

        # Conclusion with better formatting
        conclusion = enhanced_data.get("conclusion", "")
        if conclusion and len(conclusion) > 20:
            summary_parts.append("## Conclusions")
            # Format the conclusion with paragraph breaks
            formatted_conclusion = self._format_paragraphs(conclusion)
            summary_parts.append(formatted_conclusion)
            summary_parts.append("")

        # Add a References section with proper formatting if available
        references = enhanced_data.get("references", [])
        if references:
            summary_parts.append("## References")
            for i, ref in enumerate(references[:5], 1):  # Limit to 5 key references
                summary_parts.append(f"{i}. {ref}")
            if len(references) > 5:
                summary_parts.append(f"*Plus {len(references) - 5} additional references*")
            summary_parts.append("")

        # Join all parts with proper spacing
        comprehensive_summary = "\n".join(summary_parts)

        return comprehensive_summary

    def _format_paragraphs(self, text):
        """
        Format text into proper paragraphs with Markdown.

        Args:
            text (str): The text to format

        Returns:
            str: Formatted text with proper paragraph breaks
        """
        if not text:
            return ""

        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()

        # Split into sentences (basic approach)
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

    def _highlight_key_phrases(self, text, keywords):
        """
        Highlight key phrases in the text based on keywords and important phrases.

        Args:
            text (str): The text to highlight
            keywords (list): List of keywords to look for

        Returns:
            str: Text with key phrases highlighted in Markdown
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

        # Highlight each term (simple approach)
        for term in all_terms:
            if len(term) <= 3:  # Skip very short terms to avoid over-highlighting
                continue

            # Case-insensitive replacement with Markdown bold
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            highlighted_text = pattern.sub(r'**\g<0>**', highlighted_text)

        return highlighted_text

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

    def prepare_template_data(self, paper_data):
        """
        Prepare data for the HTML template

        Args:
            paper_data (dict): The extracted paper data

        Returns:
            dict: Data ready for template rendering
        """
        # First enhance the data with narrative summaries
        enhanced_data = self.create_narrative_summary(paper_data)

        # Create template-ready data structure
        template_data = {
            "title": enhanced_data.get("title", "Untitled Document"),
            "authors": enhanced_data.get("authors", "Authors not specified"),
            "institutions": enhanced_data.get("institutions", "Institutions not specified"),
            "key_concepts": enhanced_data.get("keywords", [])[:10],
            "executive_summary": enhanced_data.get("executive_summary", "Summary not available"),
            "problem_statement": enhanced_data.get("problem_statement", "Problem statement not available"),
            "methodology": enhanced_data.get("methodology", "Methodology not available"),
            "findings": enhanced_data.get("findings", "Findings not available"),
            "conclusion": enhanced_data.get("conclusion", "Conclusion not available"),
            "references": [],
            "methodology_image": None,  # No image by default

            # New narrative sections
            "key_innovations": enhanced_data.get("key_innovations", []),
            "technical_approach": enhanced_data.get("technical_approach", []),
            "significance": enhanced_data.get("significance", [])
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