import threading
import time
from datetime import datetime
from funda.outlier_engine import run_outlier_detection, STRATEGIES
import logging

# Global variables for progress tracking
refresh_status = {
    'is_running': False,
    'progress': 0,
    'current_strategy': '',
    'message': 'Ready to refresh data',
    'start_time': None,
    'estimated_completion': None
}

def refresh_outlier_data():
    """Run outlier detection for all strategies with progress tracking"""
    global refresh_status
    
    refresh_status['is_running'] = True
    refresh_status['progress'] = 0
    refresh_status['start_time'] = datetime.now()
    refresh_status['message'] = 'Starting outlier detection...'
    
    total_strategies = len(STRATEGIES)
    
    try:
        for i, strategy in enumerate(STRATEGIES):
            refresh_status['current_strategy'] = strategy
            refresh_status['progress'] = int((i / total_strategies) * 100)
            refresh_status['message'] = f'Processing {strategy} strategy...'
            
            logging.info(f"Running outlier detection ({strategy})")
            run_outlier_detection(strategy)
            
            # Estimate remaining time
            elapsed = (datetime.now() - refresh_status['start_time']).total_seconds()
            if i > 0:  # After first strategy
                avg_time_per_strategy = elapsed / (i + 1)
                remaining_strategies = total_strategies - (i + 1)
                estimated_remaining = avg_time_per_strategy * remaining_strategies
                refresh_status['estimated_completion'] = datetime.now().timestamp() + estimated_remaining
        
        refresh_status['progress'] = 100
        refresh_status['message'] = 'Data refresh completed successfully!'
        refresh_status['is_running'] = False
        
    except Exception as e:
        refresh_status['message'] = f'Error: {str(e)}'
        refresh_status['is_running'] = False
        logging.error(f"Outlier detection failed: {e}")

def start_refresh_thread():
    """Start outlier detection in a separate thread"""
    if not refresh_status['is_running']:
        thread = threading.Thread(target=refresh_outlier_data)
        thread.daemon = True
        thread.start()
        return True
    return False

def get_refresh_status():
    """Get current refresh status"""
    return refresh_status.copy()
