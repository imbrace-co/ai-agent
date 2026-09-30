/**
 * Data Board Service
 * Handles parsing sample data files (Excel/CSV) and using AI to suggest
 * appropriate board field types based on the board model schema.
 */

import * as XLSX from "xlsx";
import { generateObject, generateText } from "ai";
import { z } from "zod";
import { createOllama } from "ollama-ai-provider-v2";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createConfiguredOpenAI } from "@/utils/openaiClient";
import config from "@/config";
import logger from "@/lib/logger";
import {
  fetchUrlAsBase64,
  pdfBase64ToPngPages,
  parseJsonWithRepair,
} from "@/core/agents/processor/document-ai-utils";
import { resolveImbraceModel } from "@/providers/imbraceModels";
import { credentialHeader } from "@/utils/credential";

// ── Types ──────────────────────────────────────────────────────────────────────

export interface SubField {
  field_name: string;
  type: string;
}

export interface FieldTypeSuggestion {
  field_name: string;
  type: string;
  description: string;
  sample_data: string;
  sub_fields: SubField[];
}

export interface ParsedFileData {
  fileName: string;
  columns: string[];
  /** Up to 5 sample rows keyed by column name */
  sampleRows: Record<string, string>[];
}

/** Options to select a specific vision model for OCR instead of the default suggestion model. */
export interface OcrModelOptions {
  modelName: string;
  providerId: string;
  xAccessToken: string;
  organizationId: string;
}

// ── File Parsing ───────────────────────────────────────────────────────────────

/**
 * Fetch and parse an Excel (.xlsx/.xls) or CSV file from a URL.
 * Returns column names and up to 5 sample rows.
 *
 * @param fileUrl - Public URL of the file
 * @returns Parsed file data with columns and sample rows
 */
export async function parseSampleDataFromUrl(
  fileUrl: string,
): Promise<ParsedFileData> {
  logger.info("Fetching sample data file", { url: fileUrl });

  const response = await fetch(fileUrl, { redirect: "follow" });
  if (!response.ok) {
    throw new Error(
      `Failed to fetch file: HTTP ${response.status} from ${fileUrl}`,
    );
  }

  const arrayBuffer = await response.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);

  // Detect CSV vs binary Excel from URL extension or content-type
  const contentType = response.headers.get("content-type") ?? "";
  const isCSV =
    fileUrl.toLowerCase().endsWith(".csv") ||
    contentType.includes("text/csv") ||
    contentType.includes("text/plain");

  let workbook: XLSX.WorkBook;
  if (isCSV) {
    workbook = XLSX.read(buffer.toString("utf8"), { type: "string" });
  } else {
    workbook = XLSX.read(buffer, { type: "buffer" });
  }

  const sheetName = workbook.SheetNames[0];
  if (!sheetName) {
    throw new Error(`No sheets found in file: ${fileUrl}`);
  }

  const sheet = workbook.Sheets[sheetName];
  if (!sheet) {
    throw new Error(`Sheet "${sheetName}" not found in file: ${fileUrl}`);
  }
  // header: 1 returns a 2D array, first row = headers
  const rows = XLSX.utils.sheet_to_json<any[]>(sheet, { header: 1 });

  if (rows.length === 0) {
    return {
      fileName: extractFileName(fileUrl),
      columns: [],
      sampleRows: [],
    };
  }

  const headers = (rows[0] as any[])
    .map((h: any) => String(h ?? "").trim())
    .filter(Boolean);

  const sampleRows = rows.slice(1, 6).map((row: any[]) => {
    const record: Record<string, string> = {};
    headers.forEach((col, i) => {
      const val = row[i];
      record[col] = val !== undefined && val !== null ? String(val) : "";
    });
    return record;
  });

  logger.info("File parsed successfully", {
    fileName: extractFileName(fileUrl),
    columnCount: headers.length,
    rowCount: sampleRows.length,
  });

  return {
    fileName: extractFileName(fileUrl),
    columns: headers,
    sampleRows,
  };
}

/** Extract the filename from a URL path. */
function extractFileName(url: string): string {
  try {
    return new URL(url).pathname.split("/").pop() || url;
  } catch {
    return url.split("/").pop() || url;
  }
}

// ── Board Schema ───────────────────────────────────────────────────────────────

/**
 * Fetch the board model schema from the given URL using the access token.
 *
 * @param boardSchemaUrl - Full URL of the board model schema endpoint
 * @param xAccessToken - User access token forwarded to the board API
 * @returns Raw schema data from the board API
 */
export async function fetchBoardModelSchema(
  boardSchemaUrl: string,
  xAccessToken: string,
): Promise<unknown> {
  logger.info("Fetching board model schema", { url: boardSchemaUrl });

  const response = await fetch(boardSchemaUrl, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      ...credentialHeader(xAccessToken),
    },
  });

  if (!response.ok) {
    throw new Error(
      `Failed to fetch board model schema: HTTP ${response.status} from ${boardSchemaUrl}`,
    );
  }

  const data = await response.json();
  logger.info("Board model schema fetched successfully");
  return data;
}

// ── Model Resolution ───────────────────────────────────────────────────────────

/**
 * Resolve the configured suggestion AI model.
 * Uses SUGGESTION_MODEL_* env vars if set; falls back to gpt-4o-mini.
 */
function resolveSuggestionModel(): Parameters<
  typeof generateObject
>[0]["model"] {
  const {
    modelId: suggestionModelId,
    providerUrl: suggestionProviderUrl,
    providerType,
  } = config.suggestion;

  if (suggestionModelId && suggestionProviderUrl) {
    if (providerType === "ollama") {
      const ollama = createOllama({
        baseURL: suggestionProviderUrl.endsWith("/api")
          ? suggestionProviderUrl
          : `${suggestionProviderUrl}/api`,
      });
      logger.info("Using suggestion model (ollama)", {
        modelId: suggestionModelId,
      });
      return ollama(suggestionModelId);
    }
    const provider = createOpenAICompatible({
      name: "openai-compatible",
      baseURL: suggestionProviderUrl,
      apiKey: config.openai.apiKey,
    });
    logger.info("Using suggestion model", {
      modelId: suggestionModelId,
      providerType,
    });
    return provider(suggestionModelId) as any;
  }

  const openai = createConfiguredOpenAI();
  logger.info("Using default OpenAI model: gpt-4o-mini");
  return openai("gpt-4o-mini");
}

// ── OCR for PDF / Image files ──────────────────────────────────────────────────

const STRUCTURED_EXTENSIONS = new Set([".csv", ".xlsx", ".xls"]);
const UNSTRUCTURED_EXTENSIONS = new Set([
  ".pdf",
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".gif",
]);

/** Return "structured", "unstructured", or "unknown" based on URL extension only. */
function detectFileType(
  fileUrl: string,
): "structured" | "unstructured" | "unknown" {
  try {
    const pathname = new URL(fileUrl).pathname.toLowerCase();
    const ext = pathname.includes(".") ? "." + pathname.split(".").pop()! : "";
    if (STRUCTURED_EXTENSIONS.has(ext)) return "structured";
    if (UNSTRUCTURED_EXTENSIONS.has(ext)) return "unstructured";
  } catch {
    // Relative or malformed URL — fall through
  }
  return "unknown";
}

/**
 * Use a vision-capable AI model to OCR a PDF or image file and extract
 * field names + sample values, returned as ParsedFileData.
 *
 * @param fileUrl - Public URL of the PDF or image
 * @param model - Vision-capable AI model instance
 * @returns Parsed file data with columns and one sample row
 */
async function ocrFileWithAI(
  fileUrl: string,
  model: Parameters<typeof generateObject>[0]["model"],
): Promise<ParsedFileData> {
  const fileName = extractFileName(fileUrl);
  logger.info("Starting AI OCR for unstructured file", {
    fileName,
    url: fileUrl,
  });

  const { extension, base64 } = await fetchUrlAsBase64(fileUrl);

  // Build image parts — PDFs are rendered to PNG pages first (max 3 pages)
  let imageParts: Array<{
    type: "image";
    image: Buffer;
    mimeType: string;
  }>;

  if (extension === "pdf") {
    const pages = await pdfBase64ToPngPages(base64);
    imageParts = pages.slice(0, 3).map((pageBase64) => ({
      type: "image" as const,
      image: Buffer.from(pageBase64, "base64"),
      mimeType: "image/png",
    }));
    logger.info("PDF converted to images for OCR", {
      totalPages: pages.length,
      processingPages: imageParts.length,
    });
  } else {
    const mimeType = extension === "jpeg" ? "image/jpeg" : `image/${extension}`;
    imageParts = [
      {
        type: "image" as const,
        image: Buffer.from(base64, "base64"),
        mimeType,
      },
    ];
  }

  const { text } = await generateText({
    model: model as Parameters<typeof generateText>[0]["model"],
    system:
      "You are a document analysis assistant. Extract field names and sample values from the document. Return ONLY valid JSON, no markdown fences.",
    messages: [
      {
        role: "user",
        content: [
          ...imageParts,
          {
            type: "text" as const,
            text: `Analyze this document and extract all field names, column headers, or data labels.
Also extract a representative sample value for each field.
- If this is a form: treat field labels as column names, filled values as sample data
- If this is a table: use column headers and a row of data
- If this is an invoice/receipt: extract line item fields and values

Return ONLY a JSON object in this exact shape:
{
  "columns": ["field1", "field2", ...],
  "sample_values": [
    { "field_name": "field1", "value": "example value" },
    ...
  ]
}`,
          },
        ],
      },
    ],
  });

  const parsed = parseJsonWithRepair(text as string) as {
    columns?: unknown[];
    sample_values?: { field_name?: string; value?: string }[];
  };

  const columns = Array.isArray(parsed?.columns)
    ? (parsed.columns as string[]).filter((c) => typeof c === "string")
    : [];

  // Convert flat sample_values list into a single sampleRow record
  const sampleRow: Record<string, string> = {};
  if (Array.isArray(parsed?.sample_values)) {
    parsed.sample_values.forEach(({ field_name, value }) => {
      if (field_name) sampleRow[field_name] = value ?? "";
    });
  }

  logger.info("AI OCR extraction complete", {
    fileName,
    columnCount: columns.length,
  });

  return {
    fileName,
    columns,
    sampleRows: Object.keys(sampleRow).length > 0 ? [sampleRow] : [],
  };
}

/**
 * Parse a file from URL, auto-routing based on file type:
 * - Excel / CSV → XLSX parser
 * - PDF / image → AI vision OCR
 *
 * @param fileUrl - Public URL of the file
 * @param ocrModelOptions - Optional model override for OCR (uses suggestion model when omitted)
 * @returns Parsed file data with columns and sample rows
 */
export async function parseFileFromUrl(
  fileUrl: string,
  ocrModelOptions?: OcrModelOptions,
): Promise<ParsedFileData> {
  let fileType = detectFileType(fileUrl);

  // If extension is ambiguous, probe content-type via a HEAD request
  if (fileType === "unknown") {
    try {
      const headRes = await fetch(fileUrl, {
        method: "HEAD",
        redirect: "follow",
      });
      const ct = headRes.headers.get("content-type") ?? "";
      if (ct.includes("application/pdf") || ct.includes("image/")) {
        fileType = "unstructured";
      } else {
        fileType = "structured";
      }
    } catch {
      fileType = "structured"; // safe default
    }
  }

  if (fileType === "unstructured") {
    let model: Parameters<typeof generateObject>[0]["model"];
    if (ocrModelOptions) {
      const resolved = await resolveImbraceModel(
        ocrModelOptions.organizationId,
        ocrModelOptions.xAccessToken,
        ocrModelOptions.modelName,
        ocrModelOptions.providerId,
      );
      model = resolved.model as Parameters<typeof generateObject>[0]["model"];
    } else {
      model = resolveSuggestionModel();
    }
    return ocrFileWithAI(fileUrl, model);
  }

  return parseSampleDataFromUrl(fileUrl);
}

// ── AI Suggestion ──────────────────────────────────────────────────────────────

const FieldSuggestionSchema = z.object({
  suggestions: z.array(
    z.object({
      field_name: z
        .string()
        .describe("The column/field name from the sample data"),
      type: z
        .string()
        .describe(
          "The board field type that best matches this column (must be from the schema)",
        ),
      description: z
        .string()
        .describe("Brief description of what this field contains"),
      sample_data: z
        .string()
        .describe("One representative example value from the data"),
      sub_fields: z
        .array(
          z.object({
            field_name: z.string().describe("Sub-field column name"),
            type: z.string().describe("Board field type for this sub-field"),
          }),
        )
        .describe(
          "Sub-fields for TableInTable type; empty array for all other types",
        ),
    }),
  ),
});

/**
 * Use AI to suggest board field types for each column in the sample data files.
 *
 * @param parsedFiles - Parsed sample data (columns + rows per file)
 * @param boardSchema - Board model schema with available field types
 * @param ocrModelOptions - Optional model override (same model the caller
 *   selected for OCR). When present, the suggestion step reuses that model
 *   (e.g. a Bedrock model) instead of the env-configured SUGGESTION_MODEL_* /
 *   OpenAI default — so one model choice drives the whole pipeline. This is
 *   what previously forced every extract through OpenAI (with an empty API key)
 *   even when a Bedrock model was picked in the UI.
 * @returns Array of field type suggestions
 */
export async function suggestFieldTypesWithAI(
  parsedFiles: ParsedFileData[],
  boardSchema: unknown,
  ocrModelOptions?: OcrModelOptions,
): Promise<FieldTypeSuggestion[]> {
  let model: Parameters<typeof generateObject>[0]["model"];
  let suggestionModelId: string;
  if (ocrModelOptions) {
    const resolved = await resolveImbraceModel(
      ocrModelOptions.organizationId,
      ocrModelOptions.xAccessToken,
      ocrModelOptions.modelName,
      ocrModelOptions.providerId,
    );
    model = resolved.model as Parameters<typeof generateObject>[0]["model"];
    suggestionModelId = ocrModelOptions.modelName;
  } else {
    model = resolveSuggestionModel();
    suggestionModelId = config.suggestion.modelId ?? "gpt-4o-mini";
  }

  const systemPrompt = `You are a data schema analyst. You will receive:
1. A board model schema listing available field types with their definitions
2. Sample data files with column names and example rows

Your task is to suggest the most appropriate board field type for each unique column
based on the data content and available types in the schema.

Guidelines:
- Only use field types that exist in the provided board schema
- ShortText: for brief strings (names, codes, short labels)
- LongText: for longer free-form text (descriptions, notes, addresses)
- SingleSelection: for columns with a limited set of discrete values (status, category, type)
- Number: for numeric data (prices, quantities, percentages, IDs)
- TableInTable: for nested/relational data; identify meaningful sub-fields
- Match data patterns carefully before assigning a type`;

  const dataDescription = parsedFiles
    .map(
      (file) => `
File: ${file.fileName}
Columns: ${file.columns.join(", ")}
Sample rows (up to 3):
${JSON.stringify(file.sampleRows.slice(0, 3), null, 2)}`,
    )
    .join("\n\n---\n");

  const userPrompt = `Board Model Schema:
${JSON.stringify(boardSchema, null, 2)}

Sample Data:
${dataDescription}

Return a "suggestions" array with one entry per unique column across all files.
Each entry must include field_name, type (from schema), description, sample_data, and sub_fields.`;

  logger.info("Calling AI for field type suggestions", {
    fileCount: parsedFiles.length,
    totalColumns: parsedFiles.reduce((sum, f) => sum + f.columns.length, 0),
    modelId: suggestionModelId ?? "gpt-4o-mini",
  });

  const result = await generateObject({
    model,
    schema: FieldSuggestionSchema,
    system: systemPrompt,
    prompt: userPrompt,
  });

  logger.info("AI field type suggestions generated", {
    suggestionCount: result.object.suggestions.length,
  });

  return result.object.suggestions;
}
