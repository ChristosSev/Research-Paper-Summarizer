import os
import uuid
import json
import logging
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
import torch
import warnings


# Import our custom modules
from paper_processor import PaperProcessor
from paper_extractors import PaperExtractors

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
    print(f"NLTK punkt_tab initialization error (non-critical): {e}")

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt data...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords data...")
    nltk.download('stopwords', quiet=True)

warnings.filterwarnings("ignore")



# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}
app.secret_key = 'paper_summarizer_secret_key'  # For flash messages
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize models (loading them only when needed to save memory)
summarizer = None
tokenizer = None
keybert_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize our paper processor
paper_processor = None


def load_models():
    """Load the required ML models"""
    global summarizer, tokenizer, keybert_model, paper_processor

    try:
        if summarizer is None:
            logger.info("Loading summarization model...")
            model_name = "facebook/bart-large-cnn"  # We could also use t5-small or FLAN-T5
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            summarizer = pipeline("summarization", model=model, tokenizer=tokenizer,
                                  device=0 if device == "cuda" else -1)
            logger.info("Summarization model loaded successfully")

        if keybert_model is None:
            logger.info("Loading KeyBERT model...")
            keybert_model = KeyBERT()
            logger.info("KeyBERT model loaded successfully")

        # Initialize paper processor with the keybert model
        if paper_processor is None:
            paper_processor = PaperProcessor(keybert_model=keybert_model)

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Flask routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and paper processing"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            # Generate a unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            # Save the uploaded file
            file.save(file_path)
            logger.info(f"Saved uploaded file to {file_path}")

            try:
                # Ensure models are loaded
                load_models()

                # Process the paper
                result = paper_processor.process_paper(file_path)

                # Check if processing was successful
                if "error" in result:
                    return jsonify({"error": result["error"]}), 500

                # Save the result to a JSON file
                result_filename = f"{os.path.splitext(unique_filename)[0]}.json"
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

                with open(result_path, 'w') as f:
                    json.dump(result, f)
                logger.info(f"Saved result to {result_path}")

                # Return the result ID
                return jsonify({
                    "success": True,
                    "result_id": os.path.splitext(unique_filename)[0],
                    "summary": "Processing successful"
                })

            except Exception as e:
                logger.error(f"Error processing paper: {e}")
                return jsonify({"error": f"Error processing document: {str(e)}"}), 500
            finally:
                # Clean up the original file after processing
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing file: {e}")
        except Exception as e:
            logger.error(f"Error in file upload: {e}")
            return jsonify({"error": f"Upload error: {str(e)}"}), 500

    return jsonify({"error": "Invalid file format"}), 400


@app.route('/results/<result_id>')
def view_results(result_id):
    """View the summarization results, ensuring no full text is included."""
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{result_id}.json")

    if not os.path.exists(result_path):
        logger.warning(f"Result file not found: {result_path}")
        flash("Result not found. The processed file may have expired.", "error")
        return redirect(url_for('index'))

    try:
        with open(result_path, 'r') as f:
            result = json.load(f)

        # SAFETY CHECK: Make sure no field is longer than 2000 characters
        max_field_length = 2000
        for key, value in result.items():
            if isinstance(value, str) and len(value) > max_field_length:
                # Field is suspiciously long - truncate it
                logger.warning(f"Truncating field {key} with length {len(value)}")
                result[key] = value[:max_field_length - 3] + "..."

        # Also check nested dictionaries like sections
        if "sections" in result and isinstance(result["sections"], dict):
            for section_key, section_value in result["sections"].items():
                if isinstance(section_value, str) and len(section_value) > max_field_length:
                    result["sections"][section_key] = section_value[:max_field_length - 3] + "..."

        # Generate enhanced narrative and comprehensive summaries
        narrative_summary = paper_processor.create_narrative_summary(result)
        comprehensive_summary = paper_processor.create_comprehensive_summary(result)

        # SAFETY CHECK: Make sure narrative_summary doesn't contain full text
        if isinstance(narrative_summary, dict):
            for key, value in narrative_summary.items():
                if isinstance(value, str) and len(value) > max_field_length:
                    narrative_summary[key] = value[:max_field_length - 3] + "..."
                elif isinstance(value, list):
                    limited_items = []
                    for item in value:
                        if isinstance(item, str) and len(item) > max_field_length:
                            limited_items.append(item[:max_field_length - 3] + "...")
                        else:
                            limited_items.append(item)
                    narrative_summary[key] = limited_items

        # SAFETY CHECK: Make sure comprehensive_summary doesn't contain full text
        if isinstance(comprehensive_summary, str) and len(comprehensive_summary) > 10000:
            logger.warning(f"Comprehensive summary too long: {len(comprehensive_summary)}")
            # Take the first part only - should have the important content
            comprehensive_summary = comprehensive_summary[:10000] + "..."

        # Format top 10 key concepts for display
        top_concepts = result.get("keywords", [])[:10]

        # Format references for display - limit to 10
        formatted_refs = []
        for i, ref in enumerate(result.get("references", [])[:10]):
            formatted_refs.append(f"{i + 1}. {ref}")

        # Prepare data for the template - ONLY include summarized content, not full text
        template_data = {
            "title": result.get("title", "Untitled Document"),
            #"authors": result.get("authors", "Authors not specified"),
            #"institutions": result.get("institutions", "Institutions not specified"),
            "key_concepts": top_concepts,
            #"executive_summary": result.get("executive_summary", "Summary not available"),
            #"problem_statement": result.get("problem_statement", "Problem statement not available"),
            "methodology": result.get("methodology", "Methodology not available"),
            #"findings": result.get("findings", "Findings not available"),
            "conclusion": result.get("conclusion", "Conclusion not available"),
            "references": formatted_refs,
            "narrative_summary": narrative_summary,
            #"comprehensive_summary": comprehensive_summary
        }

        # IMPORTANT: Remove any full text fields to ensure we don't send the entire corpus
        if "text_extraction_success" in result:
            template_data["text_extraction_success"] = result["text_extraction_success"]

        # CRITICAL: Remove any sections dictionary that might contain full text
        if "sections" in template_data:
            del template_data["sections"]

        # Perform one final safety check on all fields
        for key, value in template_data.items():
            if isinstance(value, str) and len(value) > max_field_length:
                template_data[key] = value[:max_field_length - 3] + "..."

        # Handle the format parameter to support both HTML and JSON responses
        format_param = request.args.get('format', 'html')
        if format_param.lower() == 'json':
            return jsonify(template_data)
        else:
            return render_template('results.html', **template_data)

    except Exception as e:
        logger.error(f"Error loading results: {e}")
        flash(f"Error loading results: {str(e)}", "error")
        return redirect(url_for('index'))


@app.route('/download/<result_id>')
def download_summary(result_id):
    """Download the paper summary as a markdown or text file"""
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{result_id}.json")

    if not os.path.exists(result_path):
        logger.warning(f"Result file not found: {result_path}")
        flash("Result not found. The processed file may have expired.", "error")
        return redirect(url_for('index'))

    try:
        with open(result_path, 'r') as f:
            result = json.load(f)

        # Create a comprehensive summary for downloading
        comprehensive_summary = paper_processor.create_comprehensive_summary(result)

        # Determine format from query param (default to markdown)
        format_type = request.args.get('format', 'md')

        if format_type == 'md':
            # Add content-disposition header for download as markdown
            from flask import Response
            response = Response(
                comprehensive_summary,
                mimetype='text/markdown',
                headers={'Content-Disposition': f'attachment;filename={result.get("title", "paper_summary").replace(" ", "_")}.md'}
            )
            return response
        else:
            # Add content-disposition header for download as text
            from flask import Response
            response = Response(
                comprehensive_summary,
                mimetype='text/plain',
                headers={'Content-Disposition': f'attachment;filename={result.get("title", "paper_summary").replace(" ", "_")}.txt'}
            )
            return response

    except Exception as e:
        logger.error(f"Error creating downloadable summary: {e}")
        flash(f"Error creating downloadable summary: {str(e)}", "error")
        return redirect(url_for('index'))


@app.route('/about')
def about():
    """Show information about the application"""
    return render_template('about.html')


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('error.html', error="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('error.html', error="Server error occurred"), 500


@app.errorhandler(413)
def request_entity_too_large(e):
    """Handle file size exceeded error"""
    return render_template('error.html', error="File size exceeded the limit (16MB)"), 413


if __name__ == '__main__':
    try:
        # Load models on startup
        load_models()
        # Run app
        app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
