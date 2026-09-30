/**
 * Data Board Controller
 * Handles requests for board field type suggestions based on uploaded sample data.
 */

import { Request, Response } from "express";
import axios from "axios";
import { z } from "zod";
import logger from "@/lib/logger";
import { ApiResponse } from "@/types/api";
import config from "@/config";
import {
  parseFileFromUrl,
  fetchBoardModelSchema,
  suggestFieldTypesWithAI,
  type FieldTypeSuggestion,
  type OcrModelOptions,
} from "@/services/dataBoardService";

// ── Request validation schema ──────────────────────────────────────────────────

const suggestFieldTypesSchema = z.object({
  file_urls: z
    .array(z.string().url("Each file_url must be a valid URL"))
    .min(1, "At least one file URL is required"),
  board_schema_url: z.string().url().optional(),
  organization_id: z.string().optional(),
  model_name: z.string().optional(),
  provider_id: z.string().optional(),
});

const suggestFieldTypesInternalSchema = z.object({
  file_urls: z
    .array(z.string().url("Each file_url must be a valid URL"))
    .min(1, "At least one file URL is required"),
  organization_id: z.string().optional(),
  model_name: z.string().optional(),
  provider_id: z.string().optional(),
});

/**
 * Static board model schema for the internal endpoint. Mirrors the
 * `BoardFieldType` enum in the data_board service. Used in place of a live
 * fetch from the legacy backend so we don't need a user access token on
 * server-to-server callers (e.g. data_board's POST /api/schemas/_extract-attributes).
 *
 * Keep in sync with `src/domain/shared/board.types.ts` BoardFieldType in data_board.
 */
const INTERNAL_BOARD_MODEL_SCHEMA = {
  available_field_types: [
    {
      name: "ShortText",
      description: "Brief text (names, codes, short labels)",
    },
    {
      name: "LongText",
      description: "Longer free-form text (descriptions, notes, addresses)",
    },
    {
      name: "Number",
      description: "Numeric data (quantities, IDs, percentages)",
    },
    { name: "Email", description: "Email addresses" },
    { name: "Phone", description: "Phone numbers" },
    { name: "Date", description: "Calendar dates without time" },
    { name: "Datetime", description: "Dates with time component" },
    {
      name: "SingleSelection",
      description: "One value from a limited set (status, category)",
    },
    {
      name: "MultipleSelection",
      description: "Multiple values from a limited set (tags)",
    },
    { name: "Assignee", description: "Single user assigned" },
    { name: "MultipleAssignee", description: "Multiple users assigned" },
    { name: "Priority", description: "Priority levels (Low/Medium/High etc.)" },
    { name: "Currency", description: "Monetary amounts" },
    { name: "Link", description: "URLs / hyperlinks" },
    { name: "Attachment", description: "File attachments" },
    { name: "Notes", description: "Rich notes" },
    { name: "Country", description: "Country codes or names" },
    { name: "Origin", description: "Source / origin marker" },
    {
      name: "MapToBoard",
      description: "Reference to a record in another board",
    },
    {
      name: "TableInTable",
      description: "Nested/relational sub-table; supply sub_fields",
    },
    { name: "Checkbox", description: "Boolean true/false" },
    { name: "Formula", description: "Computed value" },
    { name: "Rating", description: "Numeric rating (e.g. 1-5)" },
  ],
};

/**
 * @param req - Express request
 * @param res - Express response with FieldTypeSuggestion[]
 */
export async function suggestFieldTypes(
  req: Request,
  res: Response,
): Promise<void> {
  try {
    // Extract access token from userContext (set by parseUserContext middleware)
    const userContext = (req as any).userContext;
    const xAccessToken = userContext?.x_access_token as string | undefined;

    if (!xAccessToken) {
      res.status(401).json({
        success: false,
        error: "Missing credential (x-access-token or x-api-key)",
        timestamp: new Date().toISOString(),
      } satisfies ApiResponse);
      return;
    }

    // Validate request body
    const parseResult = suggestFieldTypesSchema.safeParse(req.body);
    if (!parseResult.success) {
      res.status(400).json({
        success: false,
        error: "Invalid request body",
        data: parseResult.error.flatten().fieldErrors,
        timestamp: new Date().toISOString(),
      } satisfies ApiResponse);
      return;
    }

    const {
      file_urls,
      model_name,
      provider_id,
      organization_id,
      board_schema_url,
    } = parseResult.data;

    logger.info("Field type suggestion request received", {
      fileCount: file_urls.length,
      boardSchemaUrl: board_schema_url ?? "(using INTERNAL_BOARD_MODEL_SCHEMA)",
    });

    // Build OCR model options when the caller supplies a specific vision model
    const ocrModelOptions: OcrModelOptions | undefined =
      model_name && provider_id
        ? {
            modelName: model_name,
            providerId: provider_id,
            xAccessToken,
            organizationId: organization_id ?? "",
          }
        : undefined;

    // Step 1: Parse all files in parallel (Excel/CSV → XLSX parser; PDF/image → AI OCR)
    const parsedFiles = await Promise.all(
      file_urls.map((url) => parseFileFromUrl(url, ocrModelOptions)),
    );

    logger.info("Sample files parsed", {
      files: parsedFiles.map((f) => ({
        name: f.fileName,
        columns: f.columns.length,
      })),
    });

    // Step 2: Resolve board model schema — fetch only when the caller supplied
    // an explicit URL; otherwise use the in-process constant (parity with the
    // internal variant; avoids a fragile cross-service call to a legacy
    // endpoint that no longer exists in the microservice architecture).
    const boardSchema = board_schema_url
      ? await fetchBoardModelSchema(board_schema_url, xAccessToken)
      : INTERNAL_BOARD_MODEL_SCHEMA;

    // Step 3: Ask AI to suggest field types
    // Reuse the caller-selected model (ocrModelOptions) so a Bedrock choice
    // drives this step too instead of falling back to OpenAI gpt-4o-mini.
    const suggestions = await suggestFieldTypesWithAI(
      parsedFiles,
      boardSchema,
      ocrModelOptions,
    );

    const response: ApiResponse<FieldTypeSuggestion[]> = {
      success: true,
      data: suggestions,
      timestamp: new Date().toISOString(),
    };

    res.status(200).json(response);
  } catch (error) {
    logger.error("Field type suggestion error", {
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
    });

    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Internal server error",
      timestamp: new Date().toISOString(),
    } satisfies ApiResponse);
  }
}

/**
 * POST /api/data-board/suggest-field-types-internal
 *
 * Server-to-server variant of `suggestFieldTypes` for callers (e.g. the
 * data_board service) that authenticate via gateway-injected
 * `x-user-id` + `x-organization-id` headers instead of an end-user
 * `x-access-token`. Mounted with `requireUserAndOrgContext` instead of
 * `authorize`.
 */
export async function suggestFieldTypesInternal(
  req: Request,
  res: Response,
): Promise<void> {
  try {
    const parseResult = suggestFieldTypesInternalSchema.safeParse(req.body);
    if (!parseResult.success) {
      res.status(400).json({
        success: false,
        error: "Invalid request body",
        data: parseResult.error.flatten().fieldErrors,
        timestamp: new Date().toISOString(),
      } satisfies ApiResponse);
      return;
    }

    const { file_urls, model_name, provider_id, organization_id } =
      parseResult.data;

    // Resolve identity from gateway-injected headers (set by parseUserContext).
    // The internal endpoint authenticates via x-user-id + x-organization-id, but
    // model resolution still needs a credential + org id to call the provider.
    const userContext = (req as any).userContext;
    const xAccessToken = userContext?.x_access_token as string | undefined;
    const orgId =
      organization_id ?? (userContext?.x_org_id as string | undefined) ?? "";

    logger.info("Internal field type suggestion request", {
      fileCount: file_urls.length,
      ...(model_name && provider_id ? { model_name, provider_id } : {}),
    });

    // Build OCR model options when the caller supplies a specific vision model
    // and a credential is available for provider resolution.
    const ocrModelOptions: OcrModelOptions | undefined =
      model_name && provider_id && xAccessToken
        ? {
            modelName: model_name,
            providerId: provider_id,
            xAccessToken,
            organizationId: orgId,
          }
        : undefined;

    const parsedFiles = await Promise.all(
      file_urls.map((url) => parseFileFromUrl(url, ocrModelOptions)),
    );

    logger.info("Sample files parsed (internal)", {
      files: parsedFiles.map((f) => ({
        name: f.fileName,
        columns: f.columns.length,
      })),
    });

    const suggestions = await suggestFieldTypesWithAI(
      parsedFiles,
      INTERNAL_BOARD_MODEL_SCHEMA,
      ocrModelOptions,
    );

    const response: ApiResponse<FieldTypeSuggestion[]> = {
      success: true,
      data: suggestions,
      timestamp: new Date().toISOString(),
    };

    res.status(200).json(response);
  } catch (error) {
    logger.error("Internal field type suggestion error", {
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
    });

    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Internal server error",
      timestamp: new Date().toISOString(),
    } satisfies ApiResponse);
  }
}

/**
 * GET /api/databoards?type=<General|System|KnowledgeHub>
 *
 * Lists boards filtered by board `type` for the FE board switcher (used
 * when the chat is scoped to a databoard via URL — the user can pick
 * a sibling board of the same type without leaving the chat).
 *
 * Auth: `authorize` + `requireOrganizationId`.
 */
export async function handleListDataboards(
  req: Request,
  res: Response,
): Promise<void> {
  const userContext = (req as any).userContext;
  const organizationId = userContext?.x_org_id as string | undefined;
  const typeRaw = req.query["type"];
  const type = typeof typeRaw === "string" ? typeRaw.trim() : "";

  const baseUrl = config.imbracePrivateApi?.url;
  if (!baseUrl) {
    res.status(500).json({ error: "Imbrace private API is not configured" });
    return;
  }

  // Build URL — the upstream API parses `types[]=` (unencoded brackets)
  // as an array. URLSearchParams would encode the brackets, which the
  // Imbrace private API treats as a literal key and doesn't filter by.
  // Match the format used in tool/databoard/get-board.ts.
  const trimmedBase = baseUrl.replace(/\/$/, "");
  const upstreamUrl = type
    ? `${trimmedBase}/v1/board?types[]=${encodeURIComponent(type)}`
    : `${trimmedBase}/v1/board`;

  try {
    const response = await axios.request({
      method: "GET",
      url: upstreamUrl,
      headers: {
        "Content-Type": "application/json",
        ...(organizationId ? { "x-organization-id": organizationId } : {}),
      },
    });

    const raw = response.data;
    const boards = Array.isArray(raw)
      ? raw
      : Array.isArray(raw?.data)
        ? raw.data
        : [];

    // Reason: only echo a small projection — selector UI needs id/name/type;
    // full schema (`fields`, etc.) is heavy and unused on the FE listing.
    const projection = (boards as Array<Record<string, unknown>>).map((b) => ({
      id:
        (b["_id"] as string | undefined) ??
        (b["id"] as string | undefined) ??
        "",
      name: (b["name"] as string | undefined) ?? null,
      type: (b["type"] as string | undefined) ?? null,
      description: (b["description"] as string | undefined) ?? null,
    }));

    res.status(200).json({ success: true, data: projection });
  } catch (error: any) {
    const status = (error?.response?.status as number | undefined) ?? 500;
    const apiMessage =
      error?.response?.data?.message ||
      error?.response?.data?.error ||
      error?.message ||
      "Failed to fetch boards";
    logger.error("handleListDataboards failed", {
      type,
      status,
      error: apiMessage,
    });
    res.status(status === 404 ? 404 : 500).json({
      success: false,
      error: apiMessage,
      timestamp: new Date().toISOString(),
    } satisfies ApiResponse);
  }
}

/**
 * GET /api/databoard/:id
 *
 * Returns a small projection of board metadata (id, name, description, type)
 * so the FE can render board-aware greetings when chat is launched with
 * `?databoardId=…`. Auth: `authorize` + `requireOrganizationId`.
 */
export async function handleGetDataboardById(
  req: Request,
  res: Response,
): Promise<void> {
  const id = req.params["id"];
  const userContext = (req as any).userContext;
  const organizationId = userContext?.x_org_id as string | undefined;

  if (!id) {
    res.status(400).json({ error: "Board id is required" });
    return;
  }

  const baseUrl = config.imbracePrivateApi?.url;
  if (!baseUrl) {
    res.status(500).json({ error: "Imbrace private API is not configured" });
    return;
  }

  try {
    const response = await axios.request({
      method: "GET",
      url: `${baseUrl.replace(/\/$/, "")}/v1/board/${id}`,
      headers: {
        "Content-Type": "application/json",
        ...(organizationId ? { "x-organization-id": organizationId } : {}),
      },
    });

    const data = (response.data ?? {}) as Record<string, unknown>;
    const projection = {
      id:
        (data["_id"] as string | undefined) ??
        (data["id"] as string | undefined) ??
        id,
      name: (data["name"] as string | undefined) ?? null,
      description: (data["description"] as string | undefined) ?? null,
      type: (data["type"] as string | undefined) ?? null,
      organization_id: (data["organization_id"] as string | undefined) ?? null,
    };

    res.status(200).json({ success: true, data: projection });
  } catch (error: any) {
    const status = (error?.response?.status as number | undefined) ?? 500;
    const apiMessage =
      error?.response?.data?.message ||
      error?.response?.data?.error ||
      error?.message ||
      "Failed to fetch board";
    logger.error("handleGetDataboardById failed", {
      id,
      status,
      error: apiMessage,
    });
    res.status(status === 404 ? 404 : 500).json({
      success: false,
      error: apiMessage,
      timestamp: new Date().toISOString(),
    } satisfies ApiResponse);
  }
}
