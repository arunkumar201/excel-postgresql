
import { faker } from '@faker-js/faker';
import ExcelJS from 'exceljs';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import os from 'os';
import fs from 'fs';

// Constants
const NUM_COLUMNS = 100;
const NUM_ROWS = 1_000_000;
const FILE_NAME = 'fakeData.xlsx';
const NUM_WORKERS = Math.max(1, os.cpus().length - 1);
const ROWS_PER_WORKER = Math.ceil(NUM_ROWS / NUM_WORKERS);

if (!isMainThread) {
    const { startRow, endRow, columns } = workerData;
    const rows = [];
    for (let i = startRow; i < endRow; i++) {
        const rowData: { [key: string]: any } = {};
        columns.forEach((col: { key: string | number; }, index: number) => {
            rowData[col.key] = index === 0 ? faker.lorem.sentence() : faker.finance.accountName();
        });
        rows.push(rowData);
    }
    parentPort?.postMessage(rows);
} else {
    async function generateExcel() {
        try {
            const workbook = new ExcelJS.stream.xlsx.WorkbookWriter({ filename: FILE_NAME, useStyles: true });
            const worksheet = workbook.addWorksheet('Fake Data');

            const columns = Array.from({ length: NUM_COLUMNS }, (_, i) => ({
                header: `Column ${i + 1}`,
                key: `column${i + 1}`,
                width: 15
            }));
            worksheet.columns = columns;

            const workerPromises = Array.from({ length: NUM_WORKERS }, (_, i) => {
                const startRow = i * ROWS_PER_WORKER;
                const endRow = Math.min(startRow + ROWS_PER_WORKER, NUM_ROWS);
                return new Promise<any[]>((resolve, reject) => {
                    const worker = new Worker(__filename, {
                        workerData: { startRow, endRow, columns }
                    });
                    worker.on('message', resolve);
                    worker.on('error', reject);
                });
            });

            const results = await Promise.all(workerPromises);
            results.flat().forEach(rowData => worksheet.addRow(rowData).commit());

            await workbook.commit();

            const stats = fs.statSync(FILE_NAME);
            const fileSizeInBytes = stats.size;
            const fileSizeInMB = fileSizeInBytes / (1024 * 1024);
            const fileSizeInGB = fileSizeInMB / 1024;

            console.log(`Excel file generated: ${FILE_NAME}`);
            console.log(`Total number of rows: ${worksheet.rowCount}`);
            console.log(`File size: ${fileSizeInMB.toFixed(2)} MB (${fileSizeInGB.toFixed(2)} GB)`);
        } catch (error) {
            console.error("An error occurred during Excel file generation:", error);
        }
    }

    generateExcel().catch(console.error);
}
