# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r ..\requirements_task1.txt

# Install NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

Write-Host "Environment setup complete. Activate it with: .\venv\Scripts\Activate"
