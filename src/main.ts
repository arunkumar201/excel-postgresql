import * as XLSX from "xlsx";
import postgres from "postgres";
import path from "path";
import dotenv from "dotenv";
import { Transform } from "stream";
import * as fs from "fs";
import copyFrom from "pg-copy-streams";

dotenv.config();

const sql = postgres({
  host: process.env.DB_HOST || "localhost",
  port: Number(process.env.DB_PORT) || 5432,
  database: process.env.DB_NAME || "excel-postgres",
  username: process.env.DB_USER || "prisma",
  password: process.env.DB_PASSWORD || "topsecret",
  max: 20,
  idle_timeout: 200,
});

interface ColumnInfo {
  name: string;
  type: string;
}

async function inferColumnTypes(data: any[]): Promise<ColumnInfo[]> {
  const columns: ColumnInfo[] = [];
  if (data.length === 0) return columns;

  const firstRow = data[0];
  for (const key of Object.keys(firstRow)) {
    const values = data.slice(0, 100).map((row) => row[key]);
    const columnType = inferType(values);
    columns.push({
      name: key.replace(/[^a-zA-Z0-9_]/g, "_").toLowerCase(),
      type: columnType,
    });
  }

  return columns;
}

function inferType(values: any[]): string {
  const nonNullValues = values.filter((v) => v != null);
  if (nonNullValues.length === 0) return "TEXT";

  const allNumbers = nonNullValues.every((v) => !isNaN(Number(v)));
  const allIntegers =
    allNumbers && nonNullValues.every((v) => Number.isInteger(Number(v)));
  const allDates = nonNullValues.every((v) => !isNaN(Date.parse(String(v))));

  if (allIntegers) return "INTEGER";
  if (allNumbers) return "NUMERIC";
  if (allDates) return "TIMESTAMP";
  return "TEXT";
}

async function createTable(
  tableName: string,
  columns: ColumnInfo[]
): Promise<void> {
  const columnDefinitions = columns
    .map((col) => `"${col.name}" ${col.type}`)
    .join(",\n    ");

  const createTableSQL = `
        CREATE TABLE IF NOT EXISTS "${tableName}" (
            id SERIAL PRIMARY KEY,
            ${columnDefinitions},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    `;

  await sql.unsafe(createTableSQL);
  console.log("Table created successfully");
}

async function importExcelToPostgres(
  filePath: string,
  tableName: string
): Promise<void> {
  console.time("Import Duration");

  try {
    console.log("Reading Excel file...");
    const workbook = XLSX.readFile(filePath, {
      cellStyles: false,
      cellNF: false,
      sheetStubs: false,
    });
    const worksheet = workbook.Sheets[workbook.SheetNames[0]];

    // Sample first 100 rows for column type inference
    const dataSample = XLSX.utils.sheet_to_json(worksheet, { range: 0 });
    const columns = await inferColumnTypes(dataSample);
    await createTable(tableName, columns);

    const columnNames = columns.map((col) => col.name);

    // Create transform stream to convert Excel rows to CSV format
    const transformer = new Transform({
      objectMode: true,
      transform(row, encoding, callback) {
        const csvRow = columnNames
          .map((col) => {
            const value = row[col];
            // Handle date formatting
            if (
              columns.find((c) => c.name === col)?.type === "TIMESTAMP" &&
              value
            ) {
              return new Date(value).toISOString();
            }
            // Escape CSV values
            if (typeof value === "string") {
              return `"${value.replace(/"/g, '""')}"`;
            }
            return value !== undefined ? value : null;
          })
          .join(",");
        this.push(csvRow + "\n");
        callback();
      },
    });

    // Setup COPY command stream
    const copyStream = sql.connection.stream(
      copyFrom.from(
        `COPY "${tableName}" (${columnNames
          .map((name) => `"${name}"`)
          .join(", ")}) FROM STDIN WITH (FORMAT CSV)`
      )
    );

    // Pipe Excel data through transformer to COPY stream
    const excelStream = XLSX.stream.to_json(worksheet, {
      raw: true,
      rawNumbers: true,
    });
    excelStream.pipe(transformer).pipe(copyStream);

    // Wait for completion
    await new Promise((resolve, reject) => {
      copyStream.on("end", resolve);
      copyStream.on("error", reject);
      transformer.on("error", reject);
      excelStream.on("error", reject);
    });

    console.log("Import completed successfully");
    console.timeEnd("Import Duration");
  } catch (error) {
    console.error("Error during import:", error);
    throw error;
  } finally {
    await sql.end();
  }
}

// Execute the import
const filePath = path.resolve("../fakeData.xlsx");
const tableName = "excel_data";
if (!fs.existsSync(filePath)) {
  console.error("Excel file not found:", filePath);
  process.exit(1);
}

importExcelToPostgres(filePath, tableName)
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Failed to import data:", error);
    process.exit(1);
  });
