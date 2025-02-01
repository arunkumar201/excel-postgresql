import time
import psycopg2
import io
import os
from threading import Thread
from queue import Queue
from tqdm import tqdm
from dotenv import load_dotenv
import python_calamine
import pandas as pd
import itertools

load_dotenv()

conn_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
file_path = os.getenv('EXCEL_FILE_PATH')
CHUNK_SIZE = int(os.getenv('CHUNKS_SIZE', CHUNKS_SIZE=25000))
NUMBER_OF_THREADS = max(os.cpu_count()-1, 1)

def get_sql_type(value):
    if isinstance(value, int):
        return 'INTEGER'
    elif isinstance(value, float):
        return 'DOUBLE PRECISION'
    elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return 'TIMESTAMP'
    else:
        return 'TEXT'

def process_chunk(chunk, table_name):
    with psycopg2.connect(conn_string) as pg_conn:
        with pg_conn.cursor() as cur:
            csv_buffer = io.StringIO()
            pd.DataFrame(chunk).to_csv(csv_buffer, index=False, header=False)
            csv_buffer.seek(0)
            cur.copy_from(csv_buffer, table_name, sep=',', null='')
            pg_conn.commit()

def worker(queue, table_name, progress_queue):
    while True:
        chunk = queue.get()
        if chunk is None:
            queue.task_done()
            break
        process_chunk(chunk, table_name)
        progress_queue.put(len(chunk))
        queue.task_done()

def read_excel_and_enqueue(data_iterator, queue):
    total_rows = 0
    chunk = []
    for row in data_iterator:
        chunk.append(row)
        if len(chunk) >= CHUNK_SIZE:
            queue.put(chunk)
            total_rows += len(chunk)
            chunk = []
    if chunk:
        queue.put(chunk)
        total_rows += len(chunk)
    return total_rows

def main():
    start_time = time.time()
    try:
        table_name = 'copy_test'

        # Single-pass Excel reading
        with open(file_path, 'rb') as file:
            start_time=time.time()
            workbook = python_calamine.CalamineWorkbook.from_filelike(file)
            sheet = workbook.get_sheet_by_index(0)
            rows = iter(sheet.to_python())
            headers = list(map(str, next(rows)))  # Get headers
            print(f"Excel file read duration: {time.time() - start_time:.2f} seconds")
            # Get first data row for schema inference
            first_data_row = next(rows, None)
            if first_data_row is None:
                print("No data rows found in Excel file")
                return

            # Create data iterator including first data row
            data_rows = itertools.chain([first_data_row], rows)
            data_iterator = (dict(zip(headers, row)) for row in data_rows)

            # Infer schema from first data row
            column_defs = [f'"{col}" {get_sql_type(val)}' 
                         for col, val in zip(headers, first_data_row)]
            column_defs_str = ',\n    '.join(column_defs)

        # Create table
        with psycopg2.connect(conn_string) as pg_conn:
            with pg_conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS {table_name}')
                cur.execute(f'''
                    CREATE TABLE {table_name} (
                        {column_defs_str}
                    )
                ''')
            pg_conn.commit()

        # Setup queues and workers
        queue = Queue(maxsize=NUMBER_OF_THREADS * 2)
        progress_queue = Queue()
        workers = [Thread(target=worker, args=(queue, table_name, progress_queue)) 
                 for _ in range(NUMBER_OF_THREADS)]
        for t in workers:
            t.start()

        # Start data reading and chunking
        reader_thread = Thread(target=read_excel_and_enqueue, args=(data_iterator, queue))
        reader_thread.start()

        # Progress tracking
        total_rows = 0
        with tqdm(desc="Inserting rows", unit="row") as pbar:
            while reader_thread.is_alive() or not queue.empty():
                while not progress_queue.empty():
                    total_rows += progress_queue.get()
                    pbar.update(total_rows - pbar.n)
                time.sleep(0.1)

        # Cleanup
        for _ in range(NUMBER_OF_THREADS):
            queue.put(None)
        for t in workers:
            t.join()
        reader_thread.join()

        print(f"\nTotal rows inserted: {total_rows}")
        print(f"Total duration: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
