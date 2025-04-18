import os
import nltk
import ssl

def get_nltk_data_dir():
    """Returns the directory where NLTK data should be stored"""
    # Use project-specific directory for NLTK data
    nltk_data_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__)
                )
            )
        ), 'nltk_data'
    )
    os.makedirs(nltk_data_dir, exist_ok=True)
    return nltk_data_dir

def download_nltk_resources():
    """Download all required NLTK resources"""
    nltk_data_dir = get_nltk_data_dir()
    
    # Handle SSL certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download necessary NLTK resources
    resources = ['punkt', 'punkt_tab', 'words', 'stopwords']
    for resource in resources:
        try:
            nltk.download(resource, download_dir=nltk_data_dir, quiet=False)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")

if __name__ == "__main__":
    download_nltk_resources()