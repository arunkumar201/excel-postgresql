# Excel to PostgreSQL Data Importer

This Python script efficiently imports data from Excel files into a PostgreSQL database using chunked processing and multithreading.

## Features
- Supports .xlsx and .xls file formats
- Automatically determines SQL column types based on data
- Processes large files in chunks to manage memory usage

## Prerequisites
- Python 3.7+
- PostgreSQL database

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/cr-arunkumar/excel-to-postgres.git
   cd excel-to-postgres
   ```

2. Set up a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
5. Create `.env` from .env.example file inside the python folder. 


## RUN 
```bash
python index.py
```