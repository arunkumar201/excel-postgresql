import * as XLSX from 'xlsx';
import postgres from 'postgres';
import fs from 'fs';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import os from 'os';
import path from 'path';
import dotenv from 'dotenv';
import { Readable } from 'stream';

dotenv.config();

const CHUNK_SIZE = 50000; 
const NUM_WORKERS = Math.max(1, os.cpus().length - 1);

const createDbConnection = () => postgres({
    host: process.env.DB_HOST || 'localhost',
    port: Number(process.env.DB_PORT) || 5432,
    database: process.env.DB_NAME || 'excel_postgres',
    username: process.env.DB_USER || 'prisma',
    password: process.env.DB_PASSWORD || 'topsecret',
    max: 20,
    idle_timeout: 10,
    prepare: false,
});

async function createTable(sql: postgres.Sql, tableName: string, columns: string[]): Promise<void> {
    const columnDefs = columns.map(col => `"${col}" TEXT`).join(', ');
    const createTableSQL = `
        CREATE TABLE IF NOT EXISTS "${tableName}" (
            id SERIAL PRIMARY KEY,
            ${columnDefs},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    `;
    await sql.unsafe(createTableSQL);
    console.log(`Table "${tableName}" created successfully`);
}

if (!isMainThread) {
    const sql = createDbConnection();
    parentPort?.on('message', async (message) => {
        if (message.type === 'data') {
            await copyBatchToPostgres(sql, workerData.tableName, workerData.columns, message.batch);
            parentPort?.postMessage('done');
        } else if (message.type === 'end') {
            await sql.end();
            process.exit(0);
        }
    });
} else {
    async function processExcelStream(filePath: string, tableName: string): Promise<void> {
        const sql = createDbConnection();
        console.time('Reading Excel file...');
        const workbook = XLSX.readFile(filePath, { cellDates: true, raw: true });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        
        if (!worksheet) {
            throw new Error('Worksheet not found');
        }
        
        const headers = XLSX.utils.sheet_to_json(worksheet, { header: 1 })[0] as string[];
        console.timeEnd('Reading Excel file...');
        await createTable(sql, tableName, headers);
        
        console.time('Import Duration');
        const workers = Array(NUM_WORKERS).fill(null).map(() => new Worker(__filename, { workerData: { tableName, columns: headers } }));
        let workerIndex = 0;
        let batch: any[] = [];
        
        const range = XLSX.utils.decode_range(worksheet['!ref'] || 'A1');
        for (let rowNum = 1; rowNum <= range.e.r; rowNum++) {
            const rowData: any = {};
            headers.forEach((header, colIndex) => {
                const cellRef = XLSX.utils.encode_cell({ r: rowNum, c: colIndex });
                rowData[header] = worksheet[cellRef]?.v || null;
            });
            
            batch.push(rowData);
            if (batch.length >= CHUNK_SIZE) {
                const worker = workers[workerIndex];
                worker.postMessage({ type: 'data', batch });
                batch = [];
                workerIndex = (workerIndex + 1) % workers.length;
            }
        }

        if (batch.length > 0) {
            const worker = workers[workerIndex];
            worker.postMessage({ type: 'data', batch });
        }

        await Promise.all(workers.map(worker => 
            new Promise<void>((resolve) => {
                worker.postMessage({ type: 'end' });
                worker.on('exit', () => resolve());
            })
        ));
        console.timeEnd('Import Duration');
        console.log('Import completed successfully');
        await sql.end();
    }

    const filePath = path.resolve("./fakeData.xlsx");
    const tableName = "excel_data";

    if (!fs.existsSync(filePath)) {
        console.error('Excel file not found:', filePath);
        process.exit(1);
    }

    processExcelStream(filePath, tableName).catch(error => {
        console.error('Failed to import data:', error);
        process.exit(1);
    });
}

async function copyBatchToPostgres(sql: postgres.Sql, tableName: string, columns: string[], batch: any[]): Promise<void> {
    if (batch.length === 0) return;

    const columnList = columns.map(col => `"${col}"`).join(', ');
    const values = batch.map(row =>
        `(${columns.map(col => {
            const val = row[col];
            if (val === null || val === undefined) return 'NULL';
            if (typeof val === 'string') return `'${val.replace(/'/g, "''")}'`;
            return val;
        }).join(', ')})`
    ).join(',\n');

    const query = `INSERT INTO "${tableName}" (${columnList}) VALUES ${values}`;
    await sql.unsafe(query);
}