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
        """Extract the title with improved author separation considering commas/and"""
        lines = text.split('\n')
        potential_title_lines = []

        for line in lines[:15]:
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'^\d+\s*', '', line)

            is_potential_title_line = (
                    3 <= len(line.split()) <= 30 and
                    sum(1 for word in line.split() if word and word[0].isupper()) / len(line.split()) >= 0.5 and
                    not re.search(r'@', line) and
                    not line.lower().startswith(('abstract', 'introduction', 'keywords', 'doi', 'http'))
            )

            is_potential_author_line = (
                    len(line.split()) >= 2 and
                    sum(1 for word in line.split() if word and word[0].isupper()) / len(
                line.split()) >= 0.6 and  # Slightly relaxed uppercase ratio
                    re.search(r'([A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+)', line) and  # Look for name-like patterns
                    re.search(r'(?:,\s+|\s+and\s+|\s*&\s*)', line)  # Check for commas, "and", or "&"
            )

            is_very_likely_author_start = (
                    len(line.split()) >= 2 and
                    sum(1 for word in line.split() if word and word[0].isupper()) == len(line.split()) and
                    len(re.findall(r'[A-Z][a-z]+', line)) >= 2  # Multiple capitalized words
            )

            if is_potential_title_line and not potential_title_lines:
                potential_title_lines.append(line)
            elif potential_title_lines and is_potential_title_line and not is_potential_author_line and not is_very_likely_author_start:
                potential_title_lines.append(line)
            elif potential_title_lines and (is_potential_author_line or is_very_likely_author_start):
                # If we have a potential title and this line looks like authors, stop
                break
            elif not potential_title_lines and (is_potential_author_line or is_very_likely_author_start):
                # If the very first non-empty line looks like authors, there might be no title
                break
            elif potential_title_lines and re.match(r'^\d+\.', line):
                break
            elif potential_title_lines and line.endswith('.'):
                break

        if potential_title_lines:
            full_title = " ".join(potential_title_lines).strip()
            parts = re.split(r'[,;]', full_title)
            if len(parts) > 1:
                last_part = parts[-1].strip()
                if re.search(r'^[A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+$', last_part) or re.search(
                        r'^[A-Z]\.\s[A-Z][a-z]+', last_part):
                    full_title = ", ".join(parts[:-1]).strip()
                elif len(last_part.split()) <= 3 and sum(
                        1 for word in last_part.split() if word and word[0].isupper()) == len(last_part.split()):
                    full_title = ", ".join(parts[:-1]).strip()
            return full_title
        else:
            for line in lines:
                if line.strip():
                    return line.strip()
            return "Untitled Document"

    @staticmethod
    def extract_authors(text, title=None):
        """Extract authors (no changes here for now)"""
        if title:
            text = text.replace(title, "", 1)
        author_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]\.(?:\s*[A-Z]\.)*)?(?:\s*[-\w]+)?)(?:\s*[,]?\s*)(\d+[,\d]*)'
        author_matches = re.findall(author_pattern, text[:1000])
        if author_matches:
            authors = [match[0].strip() for match in author_matches if len(match[0].strip()) > 2]
            seen = set()
            unique_authors = [a for a in authors if not (a in seen or seen.add(a))]
            return ", ".join(unique_authors)
        author_section_pattern = r'(?:authors?[:;]?\s*)((?:[A-Z][a-zA-Z.\-]+(?:\s+[A-Z][a-zA-Z.\-]+)*(?:,\s+|\s+and\s+|\s*&\s*|\s*$)){1,})'
        author_section = re.search(author_section_pattern, text[:2000], re.IGNORECASE)
        if author_section:
            return author_section.group(1).strip()
        name_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][.])?(?:\s+[A-Z][a-zA-Z-]+)*)'
        text_start = text[:2000]
        abstract_pos = text_start.lower().find("abstract")
        if abstract_pos > 0:
            before_abstract = text_start[:abstract_pos]
            lines = before_abstract.split('\n')
            potential_authors = []
            for line in lines:
                line = line.strip()
                if not line or len(line) < 3 or len(line) > 150:
                    continue
                if line.isupper() or line.startswith('Fig') or line.startswith('Table'):
                    continue
                names = re.findall(name_pattern, line)
                if names and all(len(name) > 2 for name in names):
                    potential_authors.extend(names)
            if potential_authors:
                return ", ".join(potential_authors[:8])
        email_line_pattern = r'(?:[A-Za-z.\-]+@[A-Za-z.\-]+\.[a-z]{2,})'
        email_lines = re.findall(email_line_pattern, text[:3000])
        if email_lines:
            email_line = " ".join(email_lines)
            words = email_line.split()
            names = []
            for word in words:
                if re.match(r'^[A-Z][a-z]+$', word):
                    names.append(word)
            if names:
                return ", ".join(names)
        authors_with_pattern = r'(?:The\s+authors?\s+(?:is|are)\s+with[^.]*?)((?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)+)'
        authors_with_match = re.search(authors_with_pattern, text[:3000], re.IGNORECASE)
        if authors_with_match:
            author_text = authors_with_match.group(1)
            names = re.findall(name_pattern, author_text)
            if names:
                return ", ".join(names)
        return "Not specified"

    @staticmethod
    def extract_institutions(text):
        """
        Completely rewritten institution extraction for better accuracy and completeness
        """
        # First look for specific institution declarations in the header area (first 1500 chars)
        header_text = text[:1500]

        # Common patterns for author affiliations in academic papers
        affiliation_patterns = [
            # Superscript number pattern: Author¹, Author² followed by institution declarations
            r'(?:^|\n)(?:\d+)(?:The authors are with|Department of|University|Institute|School)([^.,\n]{5,150}[.,])',

            # Direct affiliation statements
            r'(?:^|\n)(?:The authors are with|Authors are from|From the)([^.,\n]{5,150}[.,])',

            # Affiliation statements after author list
            r'(?:^|\n)(?:[A-Z][a-z]+ [A-Z][a-z]+,? )+(?:is|are) (?:with|from|at) ([^.,\n]{5,150}[.,])',

            # Explicit affiliation sections
            r'(?:^|\n)(?:Affiliation|Institution)s?:?\s*([^.,\n]{5,150}[.,])'
        ]

        # Look for specific institutional keywords to capture
        institution_keywords = [
            'University', 'Institute', 'College', 'School', 'Laboratory', 'Lab',
            'Department', 'Centre', 'Center', 'Faculty', 'Academy', 'Corporation',
            'Research', 'Institution', 'NCSR', 'Demokritos'
        ]

        # Try to extract institutions from the header specifically for NCSR Demokritos
        ncsr_pattern = r'((?:Institute of|NCSR|National Center)[^.,]*(?:Demokritos|Informatics)[^.,\n]{0,100})'
        ncsr_matches = re.findall(ncsr_pattern, text, re.IGNORECASE)

        # Collect all institution candidates
        institution_candidates = []

        # Add NCSR matches first (highest priority)
        for match in ncsr_matches:
            if match and len(match.strip()) > 10:
                institution_candidates.append(match.strip())

        # Extract with affiliation patterns
        for pattern in affiliation_patterns:
            matches = re.findall(pattern, header_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match and len(match.strip()) > 10:
                    institution_candidates.append(match.strip())

        # If we don't have enough, try a direct keyword-based approach in the header
        if len(institution_candidates) < 3:
            for keyword in institution_keywords:
                pattern = f'({keyword}[^.,\n]{{5,150}}[.,])'
                matches = re.findall(pattern, header_text, re.IGNORECASE)
                for match in matches:
                    if match and len(match.strip()) > 10:
                        institution_candidates.append(match.strip())

        # Clean up and deduplicate
        cleaned_institutions = []
        for inst in institution_candidates:
            # Remove numbers and whitespace at the beginning
            inst = re.sub(r'^\d+\s*', '', inst)

            # Remove trailing punctuation
            inst = re.sub(r'[.,;:]+$', '', inst)

            # Clean up internal whitespace
            inst = re.sub(r'\s+', ' ', inst).strip()

            if inst and len(inst) > 10:
                # Check if the institution has a keyword
                if any(keyword.lower() in inst.lower() for keyword in institution_keywords):
                    # Don't add if very similar to existing institution
                    is_duplicate = False
                    for existing in cleaned_institutions:
                        # Check similarity
                        if existing.lower() in inst.lower() or inst.lower() in existing.lower():
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        cleaned_institutions.append(inst)

        # Look for the NCSR Demokritos pattern specifically
        found_ncsr = False
        for inst in cleaned_institutions:
            if "demokritos" in inst.lower() or "ncsr" in inst.lower():
                found_ncsr = True
                break

        # If not found, add it if it's in the text
        if not found_ncsr:
            ncsr_patterns = [
                r'(?:NCSR|National Center)[^.,\n]*Demokritos[^.,\n]{0,50}',
                r'Institute of Informatics and Telecommunications[^.,\n]{0,100}NCSR[^.,\n]{0,100}',
                r'Institute of Informatics and Telecommunications[^.,\n]{0,100}Demokritos[^.,\n]{0,100}'
            ]

            for pattern in ncsr_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    ncsr_inst = match.group(0).strip()
                    ncsr_inst = re.sub(r'\s+', ' ', ncsr_inst)  # Clean whitespace
                    if len(ncsr_inst) > 10 and ncsr_inst not in cleaned_institutions:
                        cleaned_institutions.append(ncsr_inst)
                    break

        # If we're still empty, try to extract from author affiliations with superscripts
        if not cleaned_institutions:
            superscript_pattern = r'(?:^|\n)([A-Z][a-z]+ [A-Z][a-z]+)(?:\d+).*?(?:^\d+|\n\d+)([^.,\n]{10,150})'
            matches = re.findall(superscript_pattern, text[:2000], re.MULTILINE | re.DOTALL)
            for match in matches:
                if len(match) >= 2 and len(match[1].strip()) > 10:
                    cleaned_institutions.append(match[1].strip())

        # If still no institutions, return a fallback
        if not cleaned_institutions:
            # Look for Department of... or University of... in the whole text
            fallback_patterns = [
                r'Department of[^.,\n]{5,100}',
                r'University of[^.,\n]{5,100}',
                r'Institute of[^.,\n]{5,100}'
            ]

            for pattern in fallback_patterns:
                matches = re.findall(pattern, text[:3000], re.IGNORECASE)
                for match in matches:
                    if match and len(match.strip()) > 10:
                        clean_match = re.sub(r'\s+', ' ', match).strip()
                        if clean_match not in cleaned_institutions:
                            cleaned_institutions.append(clean_match)

                            # Stop after finding 3 institutions
                            if len(cleaned_institutions) >= 3:
                                break

                if len(cleaned_institutions) >= 3:
                    break

        # Final formatting
        final_institutions = []
        for inst in cleaned_institutions[:3]:  # Limit to top 3
            # Format with capitals for major words
            words = inst.split()
            formatted_words = []
            for word in words:
                if word.lower() not in ['of', 'the', 'and', 'with', 'in', 'at', 'for', 'on', 'by']:
                    if not word.isupper():  # Don't change acronyms
                        word = word.capitalize()
                formatted_words.append(word)

            final_inst = ' '.join(formatted_words)
            final_institutions.append(final_inst)

        if final_institutions:
            return ', '.join(final_institutions)
        else:
            return "Institution not specified"

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
        """
        Extract structured references with completely rewritten logic to avoid capturing large text blocks
        """
        references = []

        # First try to find a references section
        ref_section_patterns = [
            r'(?:REFERENCES|References)\s*\n(.*?)(?:\n\s*(?:Appendix|APPENDIX|\Z))',
            r'(?:REFERENCES|References)\s*\n(.*?)(?:\Z)',
            r'(?:\n\s*REFERENCES\s*\n)(.*?)(?:\Z)'
        ]

        ref_section = ""
        for pattern in ref_section_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                ref_section = match.group(1)
                break

        if ref_section:
            # Try to extract numbered references like [1] or (1)
            numbered_ref_patterns = [
                # [1] Author et al. Title...
                r'\[\s*(\d+)\s*\]\s*([^\[\n]{10,200}?)(?=\[\s*\d+\s*\]|\n\s*\[\d+\]|\n\s*\n|\Z)',

                # 1. Author et al. Title...
                r'(?:^|\n)\s*(\d+)\.\s+([^\n]{10,200}?)(?=\n\s*\d+\.|\n\s*\n|\Z)',

                # (1) Author et al. Title...
                r'\(\s*(\d+)\s*\)\s*([^\(\n]{10,200}?)(?=\(\s*\d+\s*\)|\n\s*\(\d+\)|\n\s*\n|\Z)'
            ]

            for pattern in numbered_ref_patterns:
                matches = re.findall(pattern, ref_section, re.MULTILINE)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple) and len(match) >= 2:
                            ref_num = match[0].strip()
                            ref_text = match[1].strip()

                            # Skip if too short - likely not a real reference
                            if len(ref_text) < 20:
                                continue

                            # Clean up reference text - remove excessive whitespace
                            ref_text = re.sub(r'\s+', ' ', ref_text).strip()

                            # Final format
                            references.append(f"[{ref_num}] {ref_text}")

        # If no references found with section approach, try direct pattern matching
        if not references:
            # Look for reference patterns throughout the text
            direct_patterns = [
                # [1] Author et al. Title...
                r'\[\s*(\d+)\s*\]\s*([A-Z][^\[\n]{10,200}?)(?=\[\s*\d+\s*\]|\n\s*\[\d+\]|\n\s*\n|\Z)',

                # Direct citation of authors with year
                r'(?:^|\n)([A-Z][a-z]+(?:,\s+[A-Z]\.|\s+et\s+al\.)?(?:\s+\(\d{4}\)))\s+([^,\.\n]{10,200}?[\.|\n])'
            ]

            for pattern in direct_patterns:
                matches = re.findall(pattern, text, re.MULTILINE)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        # For [n] style, format appropriately
                        if match[0].isdigit():
                            ref_text = f"[{match[0]}] {match[1].strip()}"
                        else:
                            # For author-year style
                            ref_text = f"{match[0].strip()}, {match[1].strip()}"

                        # Clean up reference text
                        ref_text = re.sub(r'\s+', ' ', ref_text).strip()

                        # Skip if already in references or too short
                        if ref_text not in references and len(ref_text) >= 20:
                            references.append(ref_text)

        # Final cleanup - ensure no references contain full paragraphs or emails
        clean_references = []
        for ref in references:
            # Skip if suspiciously long (likely a paragraph)
            if len(ref) > 300:
                continue

            # Remove email addresses
            ref = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', ref)

            # Skip if mostly numbers (likely not a reference)
            alpha_count = sum(c.isalpha() for c in ref)
            if alpha_count < len(ref) * 0.5:
                continue

            # Clean up final format
            ref = re.sub(r'\s+', ' ', ref).strip()

            if ref and len(ref) >= 20:
                clean_references.append(ref)

        return clean_references[:20]  # Limit to 15 references

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
        """Extract methodology description from the paper with strict length limits"""
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

                    # CRITICAL: Limit length to prevent full text capture
                    # Split into sentences and take just the first few
                    sentences = re.split(r'(?<=[.!?])\s+', method_text)
                    if len(sentences) > 5:
                        # Take only first 5 sentences for a concise summary
                        method_text = ' '.join(sentences[:5])

                    # Also apply an absolute character limit
                    if len(method_text) > 500:
                        method_text = method_text[:497] + "..."

                    return method_text

            # If no specific methodology section found, look for a brief description
            # in the introduction or abstract sections
            intro_pattern = r'(?i)(?:introduction|abstract)[\s:]*\n+(.*?)(?=\n\n|\n[A-Z]|\Z)'
            match = re.search(intro_pattern, text, re.DOTALL)
            if match:
                intro_text = match.group(1).strip()
                # Look for methodology-related sentences in the introduction
                method_sentences = []
                sentences = re.split(r'(?<=[.!?])\s+', intro_text)
                for sentence in sentences:
                    if re.search(r'(?i)(method|approach|technique|algorithm|framework|system)', sentence):
                        method_sentences.append(sentence)

                if method_sentences:
                    # Return up to 3 methodology-related sentences from intro
                    return ' '.join(method_sentences[:3])

            return "Methodology details could not be extracted from the document."
        except Exception as e:
            logger.error(f"Error extracting methodology: {e}")
            return "Methodology details could not be extracted from the document."
