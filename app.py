import os
import sys
import time
import shutil
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)  # Set working directory to project root

# Import business analyst service
from src.application.business_analyst_service import BusinessAnalystService

# Initialize Flask app
app = Flask(__name__)
app.template_folder = os.path.join(project_root, 'src', 'web', 'templates')
app.static_folder = os.path.join(project_root, 'src', 'web', 'static')

# Create static directories if they don't exist
os.makedirs(os.path.join(app.static_folder, 'visualizations'), exist_ok=True)

# Initialize the business analyst service
analyst_service = BusinessAnalystService()

# Store analysis history
analysis_history = []

@app.route('/')
def index():
    """Render the main page"""
    # Get available datasets for dropdown
    datasets = analyst_service.list_datasets()
    return render_template('index.html', history=analysis_history, datasets=datasets)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process a data analysis query"""
    try:
        # Get the query from the form
        query = request.form.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Get selected dataset
        dataset_name = request.form.get('dataset')
        
        logger.info(f"Processing query: '{query}' on dataset: '{dataset_name}'")
        
        # Execute the analysis
        start_time = time.time()
        result = analyst_service.analyze_query(query, dataset_name)
        execution_time = time.time() - start_time
        
        logger.info(f"Analysis completed in {execution_time:.2f} seconds")
        
        # Check if result contains error
        if isinstance(result, str) and (result.startswith("Error:") or result.startswith("Analysis failed:")):
            logger.warning(f"Analysis returned error: {result}")
            # Still return a 200 status code so we can display the error nicely
            return jsonify({
                'query': query,
                'dataset': dataset_name,
                'result': result,
                'execution_time': f"{execution_time:.2f} seconds",
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'visualizations': [],
                'has_error': True
            })
        
        # Create timestamp for this analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Handle visualizations
        vis_paths = process_visualizations(timestamp)
        
        # Store the analysis result
        analysis_record = {
            'query': query,
            'dataset': dataset_name,
            'result': result,
            'execution_time': f"{execution_time:.2f} seconds",
            'timestamp': timestamp,
            'visualizations': vis_paths,
            'has_error': False
        }
        
        # Add to history (keep only the latest 10 for simplicity)
        analysis_history.insert(0, analysis_record)
        if len(analysis_history) > 10:
            analysis_history.pop()
        
        return jsonify(analysis_record)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Exception in analyze route: {str(e)}")
        logger.error(error_details)
        
        return jsonify({
            'query': query if 'query' in locals() else "Unknown query",
            'dataset': dataset_name if 'dataset_name' in locals() else "Unknown dataset",
            'result': f"An unexpected error occurred: {str(e)}",
            'execution_time': "Error",
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'visualizations': [],
            'has_error': True,
            'error_details': error_details
        }), 500

def process_visualizations(timestamp):
    """Process visualization files generated by the analysis"""
    vis_paths = []
    viz_files = [
        'sales_by_category.png', 
        'sales_by_month.png', 
        'sales_by_region.png', 
        'sales_by_year.png'
    ]
    
    static_viz_dir = os.path.join(app.static_folder, 'visualizations')
    
    for viz_file in viz_files:
        source_path = os.path.join(project_root, viz_file)
        if os.path.exists(source_path):
            logger.info(f"Found visualization: {viz_file}")
            # Create a timestamped filename to avoid overwriting
            file_name, file_ext = os.path.splitext(viz_file)
            new_filename = f"{file_name}_{timestamp}{file_ext}"
            dest_path = os.path.join(static_viz_dir, new_filename)
            
            # Copy the file to static directory
            shutil.copy2(source_path, dest_path)
            
            # Add to visualization paths (using relative path for templates)
            vis_paths.append(f"/static/visualizations/{new_filename}")
    
    return vis_paths

@app.route('/history')
def history():
    """Return analysis history"""
    return jsonify(analysis_history)

@app.route('/datasets')
def list_datasets():
    """Return available datasets"""
    datasets = analyst_service.list_datasets()
    return jsonify(datasets)

@app.route('/test')
def test():
    """Simple test endpoint to verify app is running"""
    return jsonify({
        'status': 'ok',
        'message': 'Flask app is running correctly'
    })

if __name__ == '__main__':
    logger.info(f"Starting Flask app on port 8080")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Template folder: {app.template_folder}")
    logger.info(f"Static folder: {app.static_folder}")
    
    # Run the app
    app.run(host='0.0.0.0', port=8080, debug=True)