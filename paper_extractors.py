import re
import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Configure logging for this module
logger = logging.getLogger(__name__)

# Add fallback for the punkt_tab resource
try:
    import nltk
    from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

    # Create punkt_tab fallback
    if not nltk.data.find('tokenizers/punkt_tab'):
        punkt_params = PunktParameters()
        sentence_tokenizer = PunktSentenceTokenizer(punkt_params)
        nltk.data._resource_cache['tokenizers/punkt_tab'] = sentence_tokenizer
except Exception as e:
    logger.warning(f"NLTK punkt_tab initialization error (non-critical): {e}")


class PaperExtractors:
    """Collection of static methods for extracting information from research papers"""

    @staticmethod
    def extract_title(text):
        """Extract the title from a research paper with improved number filtering"""
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line:
                continue
            # remove numbers at beginning of line.
            line = re.sub(r'^\d+\s*', '', line)

            words = line.split()

            # More robust checks to filter out non-title lines
            if (len(words) < 3 or  # Skip short lines
                    re.match(r'^\d+\s*$', line) or  # Skip lines with only numbers
                    re.match(r'^\d+\.\s', line) or  # Skip lines with numbered sections
                    re.match(r'^[A-Z]+:', line) or  # Skip lines that start with all uppercase and a colon.
                    line.startswith('http')):  # Skip URLs
                continue
            # check that a certain percentage of the words are uppercase.
            uppercase_word_count = 0
            for word in words:
                if word[0].isupper():
                    uppercase_word_count += 1
            if len(words) > 0 and (uppercase_word_count / len(words)) < 0.5:
                continue

            # Check if line looks like a title
            if 3 <= len(words) <= 30:  # Adjusted title length range
                return line

        # Fallback: return first non-empty line
        for line in lines:
            if line.strip():
                return line.strip()

        return "Untitled Document"

    @staticmethod
    def extract_authors(text, title=None):
        """Extract authors from an academic paper with improved accuracy"""
        # Remove the title to avoid confusion
        if title:
            text = text.replace(title, "")

        # Look for author affiliations with superscript numbers (common in academic papers)
        # This pattern looks for names followed by superscript numbers
        author_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]\.(?:\s*[A-Z]\.)*)?(?:\s*[-\w]+)?)(?:\s*[,]?\s*)(\d+[,\d]*)'

        # First try to find authors with superscript numbers
        author_matches = re.findall(author_pattern, text[:1000])

        if author_matches:
            # Extract author names and remove duplicates
            authors = [match[0].strip() for match in author_matches if len(match[0].strip()) > 2]
            # Remove duplicates while preserving order
            seen = set()
            unique_authors = [a for a in authors if not (a in seen or seen.add(a))]
            return ", ".join(unique_authors)

        # If no authors found with superscript numbers, try alternative patterns
        # Look for an "authors" section
        author_section_pattern = r'(?:authors?[:;]?\s*)((?:[A-Z][a-zA-Z.\-]+(?:\s+[A-Z][a-zA-Z.\-]+)*(?:,\s+|\s+and\s+|\s*&\s*|\s*$)){1,})'
        author_section = re.search(author_section_pattern, text[:2000], re.IGNORECASE)

        if author_section:
            return author_section.group(1).strip()

        # Look for authors at paper start (after title and before abstract)
        # This looks for lines with only names before the abstract
        name_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][.])?(?:\s+[A-Z][a-zA-Z-]+)*)'
        text_start = text[:2000]
        abstract_pos = text_start.lower().find("abstract")

        if abstract_pos > 0:
            before_abstract = text_start[:abstract_pos]
            lines = before_abstract.split('\n')
            potential_authors = []

            for line in lines:
                line = line.strip()
                # Skip empty lines and lines that are too short or too long
                if not line or len(line) < 3 or len(line) > 150:
                    continue

                # Skip lines that look like titles or headers
                if line.isupper() or line.startswith('Fig') or line.startswith('Table'):
                    continue

                # Find all name-like patterns in the line
                names = re.findall(name_pattern, line)
                if names and all(len(name) > 2 for name in names):
                    potential_authors.extend(names)

            if potential_authors:
                return ", ".join(potential_authors[:8])  # Limit to 8 authors to avoid false positives

        # Check for a line with email addresses - often contains author names
        email_line_pattern = r'(?:[A-Za-z.\-]+@[A-Za-z.\-]+\.[a-z]{2,})'
        email_lines = re.findall(email_line_pattern, text[:3000])

        if email_lines:
            email_line = " ".join(email_lines)
            words = email_line.split()
            names = []

            for word in words:
                if re.match(r'^[A-Z][a-z]+$', word):  # Fixed pattern with proper quotes
                    names.append(word)

            if names:
                return ", ".join(names)

        # Look for "The authors are with" pattern (common in IEEE papers)
        authors_with_pattern = r'(?:The\s+authors?\s+(?:is|are)\s+with[^.]*?)((?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)+)'
        authors_with_match = re.search(authors_with_pattern, text[:3000], re.IGNORECASE)

        if authors_with_match:
            author_text = authors_with_match.group(1)
            names = re.findall(name_pattern, author_text)
            if names:
                return ", ".join(names)

        # If all else fails
        return "Not specified"

    @staticmethod
    def extract_institutions(text):
        """Extract institutions with improved accuracy"""
        # Common academic institutions keywords
        institution_keywords = [
            'University', 'Institute', 'College', 'School', 'Laboratory', 'Lab',
            'Department', 'Centre', 'Center', 'Faculty', 'Academy', 'Corporation',
            'Research', 'Institution'
        ]

        # Pattern to find institutions with superscript references (common in academic papers)
        institution_patterns = [
            r'(?:^|\n)(\d+)(?:The\s+authors?\s+(?:is|are)\s+with|[A-Za-z ,]*)(' + '|'.join(
                institution_keywords) + r'[^.,\n]{5,100})',
            r'(?:^|\n)(\d+)([^.,\n]{0,20}?' + '|'.join(institution_keywords) + r'[^.,\n]{5,100})',
            r'(?:Department|University|Institute|School)[^.,\n]{5,100}',
            r'(?:^|\n|\()(' + '|'.join(institution_keywords) + r'[^.,\n]{5,100})'
        ]

        all_institutions = []

        for pattern in institution_patterns:
            matches = re.findall(pattern, text[:5000], re.IGNORECASE)
            for match in matches:
                # If the match is a tuple, get the last item (the institution name)
                institution = match[-1] if isinstance(match, tuple) else match
                institution = institution.strip()

                # Filter out false positives
                if (len(institution) > 10 and
                        any(keyword.lower() in institution.lower() for keyword in institution_keywords) and
                        not institution.startswith("Retrieved") and
                        not "http" in institution):
                    all_institutions.append(institution)

        # Look specifically for "authors are with" pattern often used in academic papers
        with_pattern = r'(?:The\s+authors?\s+(?:is|are)\s+with\s+)([^.,\n]{5,100}' + '|'.join(
            institution_keywords) + r'[^.,\n]{5,150})'
        with_matches = re.findall(with_pattern, text[:5000], re.IGNORECASE)
        all_institutions.extend(with_matches)

        # Remove duplicates while preserving order
        unique_institutions = []
        seen = set()
        for inst in all_institutions:
            # Normalize the text for comparison
            normalized = re.sub(r'\s+', ' ', inst.lower())
            if normalized not in seen and len(inst) > 10:
                seen.add(normalized)
                unique_institutions.append(inst)

        # Limit to 3 most relevant institutions
        final_institutions = unique_institutions[:3] if unique_institutions else ["Not specified"]

        # Clean up and format nicely
        formatted_institutions = []
        for inst in final_institutions:
            # Remove numbers at the beginning and clean up
            inst = re.sub(r'^\d+\s*', '', inst).strip()
            inst = re.sub(r'\s+', ' ', inst)
            formatted_institutions.append(inst)

        return ", ".join(formatted_institutions)

    @staticmethod
    def identify_sections(text):
        """Properly identify sections in an academic paper with improved pattern recognition and fallback methods"""
        sections = {
            'full_text': text
        }

        # Try different section heading patterns

        # Pattern 1: Look for standard sections with improved regex patterns
        section_patterns = {
            'abstract': [
                r'(?:^|\n)(?:[0-9\s]*)?Abstract(?:\s|\n)+(.*?)(?=\n\s*(?:Introduction|INTRODUCTION|1\.|\d+\.\s+Introduction|I\.|II\.))',
                r'(?:^|\n)(?:[0-9\s]*)?ABSTRACT(?:\s|\n)+(.*?)(?=\n\s*(?:Introduction|INTRODUCTION|1\.|\d+\.\s+Introduction|I\.|II\.))'
            ],
            'introduction': [
                r'(?:^|\n)(?:[0-9\s]*)?(?:1\.|I\.|\d+\.\s+)?Introduction(?:\s|\n)+(.*?)(?=\n\s*(?:(?:2\.|II\.|\d+\.\s+)(?:Background|Related|Method|Methodology|METHODOLOGY|Experiment|Data|System|Approach|Results|MODEL|PROPOSED|Framework)))',
                r'(?:^|\n)(?:[0-9\s]*)?INTRODUCTION(?:\s|\n)+(.*?)(?=\n\s*(?:(?:2\.|II\.|\d+\.\s+)?(?:Background|Related|Method|Methodology|METHODOLOGY|Experiment|Data|System|Approach|Results|MODEL|PROPOSED|Framework)))'
            ],
            'related_work': [
                r'(?:^|\n)(?:[0-9\s]*)?(?:2\.|II\.|\d+\.\s+)?(?:Related\s+Work|Background|Literature\s+Review)(?:\s|\n)+(.*?)(?=\n\s*(?:(?:3\.|III\.|\d+\.\s+)?(?:Method|Methodology|METHODOLOGY|Experiment|Data|System|Approach|MODEL|PROPOSED)))',
                r'(?:^|\n)(?:[0-9\s]*)?(?:RELATED\s+WORK|BACKGROUND|LITERATURE\s+REVIEW)(?:\s|\n)+(.*?)(?=\n\s*(?:(?:3\.|III\.|\d+\.\s+)?(?:Method|Methodology|METHODOLOGY|Experiment|Data|System|Approach|MODEL|PROPOSED)))'
            ],
            'methods': [
                r'(?:^|\n)(?:[0-9\s]*)?(?:3\.|III\.|\d+\.\s+)?(?:Method(?:s|ology)?|Approach|Framework|Model|METHODOLOGY|System|Algorithm)(?:\s|\n)+(.*?)(?=\n\s*(?:(?:4\.|IV\.|\d+\.\s+)?(?:Experiment|Evaluation|Result|Discussion|Data|Implementation|Performance)))',
                r'(?:^|\n)(?:[0-9\s]*)?(?:PROPOSED\s+METHOD|METHOD(?:S|OLOGY)?|APPROACH|FRAMEWORK|MODEL|SYSTEM|ALGORITHM)(?:\s|\n)+(.*?)(?=\n\s*(?:(?:4\.|IV\.|\d+\.\s+)?(?:Experiment|Evaluation|Result|Discussion|Data|Implementation|Performance)))'
            ],
            'experimental_setup': [
                r'(?:^|\n)(?:[0-9\s]*)?(?:4\.|IV\.|\d+\.\s+)?(?:Experimental\s+Setup|Implementation|Data|Dataset)(?:\s|\n)+(.*?)(?=\n\s*(?:(?:5\.|V\.|\d+\.\s+)?(?:Result|Evaluation|Discussion|Analysis|Performance)))',
                r'(?:^|\n)(?:[0-9\s]*)?(?:EXPERIMENTAL\s+SETUP|IMPLEMENTATION|DATA|DATASET)(?:\s|\n)+(.*?)(?=\n\s*(?:(?:5\.|V\.|\d+\.\s+)?(?:Result|Evaluation|Discussion|Analysis|Performance)))'
            ],
            'results': [
                r'(?:^|\n)(?:[0-9\s]*)?(?:5\.|V\.|\d+\.\s+)?(?:Result(?:s)?|Evaluation|Analysis|Experiment(?:s)?)(?:\s|\n)+(.*?)(?=\n\s*(?:(?:6\.|VI\.|\d+\.\s+)?(?:Discussion|Conclusion|Future|Limitation)))',
                r'(?:^|\n)(?:[0-9\s]*)?(?:RESULT(?:S)?|EVALUATION|ANALYSIS|EXPERIMENT(?:S)?)(?:\s|\n)+(.*?)(?=\n\s*(?:(?:6\.|VI\.|\d+\.\s+)?(?:Discussion|Conclusion|Future|Limitation)))'
            ],
            'discussion': [
                r'(?:^|\n)(?:[0-9\s]*)?(?:6\.|VI\.|\d+\.\s+)?(?:Discussion)(?:\s|\n)+(.*?)(?=\n\s*(?:(?:7\.|VII\.|\d+\.\s+)?(?:Conclusion|Future|Summary|Limitation)))',
                r'(?:^|\n)(?:[0-9\s]*)?(?:DISCUSSION)(?:\s|\n)+(.*?)(?=\n\s*(?:(?:7\.|VII\.|\d+\.\s+)?(?:Conclusion|Future|Summary|Limitation)))'
            ],
            'conclusion': [
                r'(?:^|\n)(?:[0-9\s]*)?(?:7\.|VII\.|\d+\.\s+)?(?:Conclusion(?:s)?|Summary|Future(?:\s+Work)?)(?:\s|\n)+(.*?)(?=\n\s*(?:Acknowledgement|ACKNOWLEDGEMENT|Reference|REFERENCE|Appendix|APPENDIX|\Z))',
                r'(?:^|\n)(?:[0-9\s]*)?(?:CONCLUSION(?:S)?|SUMMARY|FUTURE(?:\s+WORK)?)(?:\s|\n)+(.*?)(?=\n\s*(?:Acknowledgement|ACKNOWLEDGEMENT|Reference|REFERENCE|Appendix|APPENDIX|\Z))'
            ]
        }

        # Try to match each section
        for section_name, patterns in section_patterns.items():
            section_text = None
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    section_text = match.group(1).strip()
                    break

            if section_text and len(section_text) > 100:  # Only add if we have meaningful content
                sections[section_name] = section_text

        # If no sections found with standard patterns, try alternative approach
        if len(sections) < 3:  # Just full_text and maybe 1-2 others
            logger.info("Few sections found with standard patterns, trying alternative approach")

            # Try to detect section headers
            header_pattern = r'(?:^|\n)(?:(?:\d+\.)|(?:[IV]+\.)|(?:[A-Z][A-Z\s]+))(?:\s*)([A-Z][A-Za-z\s]+)(?:\s|\n)'
            headers = re.findall(header_pattern, text)

            if headers:
                logger.info(f"Found potential section headers: {headers[:5]}...")

                # Try to extract content between detected headers
                for i, header in enumerate(headers):
                    header_lower = header.lower().strip()

                    # Map header to section name
                    section_name = None
                    if any(kw in header_lower for kw in ['abstract']):
                        section_name = 'abstract'
                    elif any(kw in header_lower for kw in ['introduction', 'overview']):
                        section_name = 'introduction'
                    elif any(kw in header_lower for kw in ['related', 'background', 'literature', 'review']):
                        section_name = 'related_work'
                    elif any(kw in header_lower for kw in ['method', 'approach', 'model', 'framework', 'system', 'Methodology']):
                        section_name = 'methods'
                    elif any(kw in header_lower for kw in ['experiment', 'implementation', 'setup', 'dataset']):
                        section_name = 'experimental_setup'
                    elif any(kw in header_lower for kw in ['result', 'evaluation', 'analysis', 'results']):
                        section_name = 'results'
                    elif any(kw in header_lower for kw in ['discussion']):
                        section_name = 'discussion'
                    elif any(kw in header_lower for kw in ['conclusion', 'summary', 'future']):
                        section_name = 'conclusion'

                    if section_name:
                        # Try to extract content for this section
                        header_with_num = re.escape(header)
                        if i < len(headers) - 1:
                            next_header = re.escape(headers[i + 1])
                            section_pattern = f'(?:{header_with_num})(?:\s|\n)+(.*?)(?=(?:{next_header})|(?:Reference|REFERENCE|Acknowledgement))'
                        else:
                            section_pattern = f'(?:{header_with_num})(?:\s|\n)+(.*?)(?=(?:Reference|REFERENCE|Acknowledgement|Appendix|APPENDIX|\Z))'

                        section_match = re.search(section_pattern, text, re.DOTALL)
                        if section_match:
                            section_text = section_match.group(1).strip()
                            if len(section_text) > 100:  # Only add if we have meaningful content
                                sections[section_name] = section_text

        # If still no sections found, use a simple text-splitting approach based on common section keywords
        if len(sections) < 3:
            logger.info("Fewer than 3 sections found, using simple text splitting")

            # Split the text into potential paragraphs
            paragraphs = re.split(r'\n\s*\n', text)

            # Initialize section content
            current_section = None
            section_content = ""

            # Process each paragraph
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # Check if this paragraph looks like a section header
                if len(para) < 100 and re.match(r'^(?:\d+\.|\s*[A-Z][A-Za-z\s]+\s*)$', para):
                    para_lower = para.lower()

                    # Save previous section content if we have any
                    if current_section and section_content:
                        sections[current_section] = section_content.strip()
                        section_content = ""

                    # Determine new section type
                    if 'abstract' in para_lower:
                        current_section = 'abstract'
                    elif 'introduction' in para_lower or 'overview' in para_lower:
                        current_section = 'introduction'
                    elif 'related work' in para_lower or 'background' in para_lower:
                        current_section = 'related_work'
                    elif 'method' in para_lower or 'approach' in para_lower:
                        current_section = 'methods'
                    elif 'experiment' in para_lower or 'setup' in para_lower:
                        current_section = 'experimental_setup'
                    elif 'result' in para_lower or 'evaluation' in para_lower:
                        current_section = 'results'
                    elif 'discussion' in para_lower:
                        current_section = 'discussion'
                    elif 'conclusion' in para_lower or 'summary' in para_lower:
                        current_section = 'conclusion'
                    else:
                        # Not a recognized section header, add to current section
                        if current_section:
                            section_content += "\n\n" + para
                else:
                    # Regular paragraph, add to current section
                    if current_section:
                        section_content += "\n\n" + para

            # Save the last section
            if current_section and section_content:
                sections[current_section] = section_content.strip()

        # If abstract is not found, try to extract it from the beginning of the paper
        if 'abstract' not in sections:
            # The abstract is typically one of the first paragraphs
            paragraphs = re.split(r'\n\s*\n', text[:3000])  # Look at first 3000 chars

            # Try to find a paragraph that looks like an abstract
            for i, para in enumerate(paragraphs[:5]):  # Check first 5 paragraphs
                para = para.strip()

                # Skip very short paragraphs and ones that look like titles or authors
                if len(para) < 100 or '\n' in para or re.match(r'^[A-Z\s]+$', para):
                    continue

                # If it's a decent-sized paragraph and not the title/authors, it might be the abstract
                sections['abstract'] = para
                break

        # Extract References section
        ref_match = re.search(r'(?:References|REFERENCES)\s*\n(.*?)(?:\Z)', text, re.DOTALL)
        if ref_match:
            sections['references_section'] = ref_match.group(1).strip()

        # Log what sections were found
        logger.info(f"Identified sections: {list(sections.keys())}")
        for section in sections:
            if section != 'full_text' and section != 'references_section':
                text_sample = sections[section][:100] + "..." if len(sections[section]) > 100 else sections[section]
                logger.info(f"Section '{section}', length {len(sections[section])}, sample: {text_sample}")

        return sections

    @staticmethod
    def extract_keyphrases(text, num_keywords=10, keybert_model=None):
        """Extract key phrases/keywords from the paper text with improved fallback mechanisms"""
        if not text or len(text) < 100:
            logger.warning("Text too short for keyword extraction")
            return []

        try:
            # Verify we have the KeyBERT model
            if keybert_model is None:
                logger.error("KeyBERT model not provided")
                raise ValueError("KeyBERT model required but not provided")

            # Clean text more aggressively to remove problematic characters
            # Remove references, headers, page numbers
            text = re.sub(r'\n\d+\s*$', '', text, flags=re.MULTILINE)

            # Fix the problematic character range that's causing the error
            text = text.replace('—', '-').replace('–', '-')  # Normalize all types of dashes

            # Remove references section completely
            text = re.sub(r'(?i)references.*', '', text, flags=re.DOTALL)

            # Clean text
            text = re.sub(r'\s+', ' ', text).strip()

            # Limit text length to prevent OOM errors
            text = text[:15000]

            # Extract keywords with safer parameters
            logger.info("Extracting keywords with KeyBERT")
            keywords = keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),  # Allow 1-3 word phrases
                stop_words='english',
                use_mmr=True,  # Maximize diversity
                diversity=0.7,
                top_n=num_keywords * 2  # Extract more for filtering
            )

            # Clean and filter keywords
            stop_words = set(stopwords.words('english'))
            cleaned_keywords = []
            for keyword, score in keywords:
                # Skip keywords with very low scores
                if score < 0.15:
                    continue

                # Clean keyword
                clean_keyword = re.sub(r'[^a-zA-Z\s]', '', keyword).strip()

                # Skip very short keywords or those consisting only of stopwords
                words = clean_keyword.lower().split()
                if (len(clean_keyword) > 2 and
                        1 <= len(words) <= 4 and
                        not all(word in stop_words for word in words)):
                    cleaned_keywords.append(clean_keyword)

            # Remove duplicates and limit to requested number
            unique_keywords = []
            for kw in cleaned_keywords:
                if kw.lower() not in [k.lower() for k in unique_keywords]:
                    unique_keywords.append(kw)
                    if len(unique_keywords) >= num_keywords:
                        break

            logger.info(f"Successfully extracted {len(unique_keywords)} keywords")
            return unique_keywords

        except Exception as e:
            logger.error(f"Error extracting keyphrases: {e}")

            # Fallback: manual keyword extraction using TF-IDF principles
            logger.info("Using fallback keyword extraction method")
            try:
                # Simple word frequency approach
                stop_words = set(stopwords.words('english'))

                # Tokenize and clean
                words = word_tokenize(text.lower())
                words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 3]

                # Count word frequencies
                word_freq = {}
                for word in words:
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1

                # Get top words
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:num_keywords]
                return [word for word, _ in top_words]

            except Exception as e2:
                logger.error(f"Fallback keyword extraction also failed: {e2}")
                return []

    @staticmethod
    def extract_references(text):
        """Extract structured references with improved formatting and accuracy"""
        # First, find the references section
        reference_patterns = [
            r'(?:REFERENCES|References)\s*\n(.*?)(?:\n\s*(?:Appendix|APPENDIX|\Z))',
            r'(?:REFERENCES|References)\s*\n(.*?)(?:\Z)',
            r'(?:\n\s*REFERENCES\s*\n)(.*?)(?:\Z)'
        ]

        references_text = ""
        for pattern in reference_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                references_text = match.group(1)
                break

        if not references_text:
            return []

        # Different patterns for reference formatting
        reference_item_patterns = [
            # Pattern for "[n] Author et al. Title..."
            r'\[\s*(\d+)\s*\]\s*([^\[\n]+?)(?=\[\s*\d+\s*\]|\Z)',

            # Pattern for "n. Author et al. Title..."
            r'(?:^|\n)\s*(\d+)\.\s+([^\n]+?(?:\.\s|$)(?:[^\n]+?)?)(?=\n\s*\d+\.|$)',

            # Pattern for author-year style references
            r'(?:^|\n)([A-Z][a-z]+(?:(?:,\s*|\s+and\s+|\s*&\s*)[A-Z][a-z]+)*\s*(?:\(\d{4}\)|\[\d{4}\])\.?\s+[^\n]+)(?=\n|$)'
        ]

        all_references = []

        for pattern in reference_item_patterns:
            matches = re.findall(pattern, references_text, re.MULTILINE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        # For numbered references, combine the number and text
                        ref_text = f"[{match[0]}] {match[1].strip()}"
                    else:
                        ref_text = match.strip()

                    # Clean up and format
                    ref_text = re.sub(r'\s+', ' ', ref_text)

                    # Make sure it's reasonably long to be a reference
                    if len(ref_text) > 20:
                        all_references.append(ref_text)

        # If no references found with above patterns, try line-by-line extraction
        if not all_references and references_text:
            lines = references_text.split('\n')
            current_ref = ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if this line starts a new reference
                if (re.match(r'^\[\s*\d+\s*\]', line) or
                        re.match(r'^\d+\.', line) or
                        re.match(r'^[A-Z][a-z]+', line)):

                    # Save the previous reference if it exists
                    if current_ref and len(current_ref) > 20:
                        all_references.append(current_ref)

                    # Start a new reference
                    current_ref = line
                else:
                    # Continue the current reference
                    current_ref += " " + line

            # Don't forget the last reference
            if current_ref and len(current_ref) > 20:
                all_references.append(current_ref)

        # Limit to a reasonable number and clean up
        max_refs = 20
        references = all_references[:max_refs]
        references = [re.sub(r'\s+', ' ', ref).strip() for ref in references]

        # Final check to remove duplicates
        seen = set()
        unique_references = []
        for ref in references:
            normalized = " ".join(ref.lower().split())
            if normalized not in seen:
                seen.add(normalized)
                unique_references.append(ref)

        return unique_references

    @staticmethod
    def summarize_section(text, max_length=200):
        """Extractive summarization of a text section with improved robustness"""
        if not text or len(text.strip()) < 100:
            logger.warning(f"Text too short for summarization: {len(text) if text else 0} chars")
            return text

        try:
            # Clean and prepare text
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

            # Split into sentences using a robust method
            sentences = PaperExtractors.custom_sentence_tokenize(text)
            logger.info(f"Section has {len(sentences)} sentences to summarize")

            if len(sentences) <= 3:
                # If very few sentences, just return them
                return ' '.join(sentences)

            # Calculate sentence scores based on important keywords and position
            # First, get word frequencies
            try:
                stop_words = set(stopwords.words('english'))
            except:
                # Fallback stopwords if NLTK fails
                logger.warning("NLTK stopwords failed, using hardcoded fallback")
                stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                                  "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
                                  'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
                                  'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                                  'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                                  'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
                                  'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                                  'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                                  'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
                                  'about', 'against', 'between', 'into', 'through', 'during', 'before',
                                  'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
                                  'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                                  'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                                  'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                                  'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                                  'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
                                  'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                                  'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
                                  "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
                                  'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
                                  "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                                  'won', "won't", 'wouldn', "wouldn't"])

            # Custom simple word tokenizer for robustness
            def simple_word_tokenize(text):
                return re.findall(r'\b\w+\b', text.lower())

            # Calculate word frequencies
            word_frequencies = {}
            for word in simple_word_tokenize(text):
                if word not in stop_words and len(word) > 2:
                    if word not in word_frequencies:
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1

            # Normalize word frequencies
            if word_frequencies:
                max_frequency = max(word_frequencies.values())
                for word in word_frequencies:
                    word_frequencies[word] = word_frequencies[word] / max_frequency

            # Score each sentence
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                # Position score - favor beginning and end sentences
                position_score = 0
                if i < len(sentences) * 0.2:  # First 20% of sentences
                    position_score = 0.8
                elif i > len(sentences) * 0.8:  # Last 20% of sentences
                    position_score = 0.5

                # Content score based on word frequencies
                content_score = 0
                for word in simple_word_tokenize(sentence):
                    if word in word_frequencies:
                        content_score += word_frequencies[word]

                # Sentence length penalty (avoid very short or very long sentences)
                words_in_sentence = len(simple_word_tokenize(sentence))
                length_score = 0
                if 5 <= words_in_sentence <= 25:  # Ideal length range
                    length_score = 1.0
                elif words_in_sentence < 5:  # Too short
                    length_score = 0.3
                else:  # Too long
                    length_score = 0.8

                # Combine scores
                sentence_scores[i] = (content_score * 0.6) + (position_score * 0.3) + (length_score * 0.1)

            # Determine how many sentences to include
            target_word_count = max_length
            current_word_count = 0

            # Get top scoring sentences (limited by count)
            ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

            # Select enough sentences to meet target length, preserving original order
            selected_indices = []
            for idx, _ in ranked_sentences:
                sentence_word_count = len(simple_word_tokenize(sentences[idx]))
                if current_word_count + sentence_word_count <= target_word_count:
                    selected_indices.append(idx)
                    current_word_count += sentence_word_count

                if current_word_count >= target_word_count:
                    break

            # Sort indices to maintain original order
            selected_indices.sort()

            # Create summary from selected sentences
            summary = ' '.join([sentences[i] for i in selected_indices])

            # If we still don't have enough sentences, take the first few
            if not selected_indices:
                logger.warning("Ranking algorithm failed to select sentences, using fallback")
                words = text.split()
                return ' '.join(words[:target_word_count])

            logger.info(f"Created summary with {len(simple_word_tokenize(summary))} words")
            return summary

        except Exception as e:
            logger.error(f"Error in extractive summarization: {e}")
            # Fall back to simple truncation if all else fails
            try:
                words = text.split()
                return ' '.join(words[:max_length]) + "..."
            except:
                return "Unable to summarize this section."

    @staticmethod
    def custom_sentence_tokenize(text):
        """Simple regex-based sentence tokenizer as fallback for NLTK"""
        # This is a simple regex-based tokenizer that splits on sentence-ending punctuation
        # followed by whitespace and a capital letter
        try:
            # First try NLTK tokenizer
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except:
            # Basic sentence splitting regex
            import re
            # Split on period, exclamation mark, or question mark followed by space and capital letter
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            # Further break down very long sentences
            result = []
            for sentence in sentences:
                if len(sentence) > 300:  # If sentence is very long
                    # Try to break at semicolons or colons followed by spaces
                    parts = re.split(r'(?<=[;:])\s+', sentence)
                    result.extend(parts)
                else:
                    result.append(sentence)
            return result

    @staticmethod
    def extract_methodology(text):
        """Extract methodology description from the paper"""
        try:
            # Look for methodology section using various keywords
            method_patterns = [
                r'(?i)(?:methodology|methods|approach|procedure)[\s:]*\n+(.*?)(?=\n\n|\n[A-Z]|\Z)',
                r'(?i)(?:proposed method|proposed approach|our approach|we propose)[\s:]*\n*(.*?)(?=\n\n|\Z)',
                r'(?i)(?:experimental setup|experiment[s]? setup|setup)[\s:]*\n*(.*?)(?=\n\n|\Z)'
            ]

            for pattern in method_patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match and len(match.group(1).strip()) > 100:
                    method_text = match.group(1).strip()
                    return method_text

            return ""
        except Exception as e:
            logger.error(f"Error extracting methodology: {e}")
            return ""