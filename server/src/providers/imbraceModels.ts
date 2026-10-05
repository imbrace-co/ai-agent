import axios from "axios";
import config from "@/config";
import { credentialHeader } from "@/utils/credential";
import { wrapLanguageModel, type LanguageModelMiddleware } from "ai";
import { jsonrepair } from "jsonrepair";
import { repairToolInput } from "@/utils/repair-tool-input";
import { createAmazonBedrock } from "@ai-sdk/amazon-bedrock";
import { createOpenAI } from "@ai-sdk/openai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOllama } from "ollama-ai-provider-v2";

export interface ImbraceModel {
  name: string;
  is_toolCall_available?: boolean;
  is_vision_available?: boolean;
  provider: string;
  [key: string]: any;
}

export interface ProviderConfig {
  apiKey?: string;
  baseURL?: string;
  region?: string;
  access_key?: string;
  secret_key?: string;
  host?: string;
}

export async function getSystemConfig(
  organizationId: string,
  xAccessToken: string,
) {
  const url = `${config.webApp.url}/api/v3/ai/providers`;
  const { data } = await axios.get(url, {
    headers: {
      "x-organization-id": organizationId,
      ...credentialHeader(xAccessToken),
      "content-type": "application/json",
    },
  });

  const list = Array.isArray(data?.data)
    ? data.data
    : Array.isArray(data)
      ? data
      : [];

  return Array.isArray(list) ? list : [];
}

export async function fetchImbraceModels(
  organizationId: string,
  xAccessToken: string,
): Promise<ImbraceModel[]> {
  const url = `${config.webApp.url}/api/v3/ai/workflow-agent/models`;
  const { data } = await axios.get(url, {
    headers: {
      "x-organization-id": organizationId,
      ...credentialHeader(xAccessToken),
      "content-type": "application/json",
    },
  });

  const list = Array.isArray(data?.data)
    ? data.data
    : Array.isArray(data)
      ? data
      : [];

  return Array.isArray(list) ? list : [];
}

export async function getImbraceModelDetails(
  organizationId: string,
  modelId: string,
): Promise<ImbraceModel | null> {
  const url = `${config.webApp.url}/api/v2/ai/workflow-agent/models/${modelId}`;
  const { data } = await axios.get(url, {
    headers: {
      "x-organization-id": organizationId,
      "content-type": "application/json",
    },
  });

  const details = (data && data.data) || data || null;
  return (details as ImbraceModel) ?? null;
}

async function fetchCustomProviderConfig(
  providerId: string,
  organizationId: string,
  xAccessToken: string,
): Promise<{
  providerType: string;
  config: ProviderConfig;
  modelId?: string;
} | null> {
  // Reason: read the provider over chat-ai's internal surface
  // (AI_SERVICE_V2_INTERNAL_URL → /api/v1/providers/:id). chat-ai returns
  // secrets UNMASKED only when x-internal-service-key matches its
  // INTERNAL_SERVICE_KEY (set it to AI_SERVICE_V2_API_KEY); otherwise they come
  // back masked with bullets and every non-vLLM branch below throws for want
  // of a key. It also avoids the gateway 404: ai-agent used to call
  // `${webApp.url}/api/v3/ai/...`, but the gateway mounts the AI router at
  // `/v3/ai` (no `/api` prefix), so that path never matched. Falls back to
  // the public path when the internal URL is unset (masked but still tells
  // us WHERE to send vLLM/Ollama traffic).
  const internalBase = config.aiv2Internal.url;
  const url = internalBase
    ? `${internalBase}/api/v1/providers/${providerId}`
    : `${config.webApp.url}/api/v3/ai/providers/${providerId}`;
  const headers: Record<string, string> = {
    "x-organization-id": organizationId,
    "content-type": "application/json",
  };
  if (internalBase) {
    headers["x-api-key"] = config.aiv2Internal.apiKey;
    // chat-ai returns unmasked credentials only when this matches its INTERNAL_SERVICE_KEY.
    headers["x-internal-service-key"] = config.aiv2Internal.apiKey;
  } else {
    Object.assign(headers, credentialHeader(xAccessToken));
  }
  try {
    const { data } = await axios.get(url, { headers });

    if (!data) return null;
    const providerData = data.data || data;
    const providerConfig = providerData.config || {};

    // Check for specific provider configs
    if (providerConfig.openApi) {
      return {
        providerType: "openai", // Custom assumes OpenAI compatible usually
        config: {
          apiKey: providerConfig.openApi.api_key,
          baseURL: providerConfig.openApi.base_url,
        },
      };
    }

    if (providerConfig.custom) {
      return {
        providerType: "custom",
        config: {
          apiKey: providerConfig.custom.api_key,
          baseURL: providerConfig.custom.base_url,
        },
      };
    }

    if (providerConfig.google) {
      return {
        providerType: "google",
        config: {
          apiKey: providerConfig.google.api_key,
          baseURL: providerConfig.google.base_url,
        },
        modelId: providerConfig.google.model, // Google config might specify the model
      };
    }

    if (providerConfig.bedrock) {
      return {
        providerType: "bedrock",
        config: {
          access_key: providerConfig.bedrock.access_key,
          secret_key: providerConfig.bedrock.secret_key,
          region: providerConfig.bedrock.region,
        },
      };
    }

    if (providerConfig.ollama) {
      return {
        providerType: "ollama",
        config: {
          host: providerConfig.ollama.host,
        },
      };
    }

    if (providerConfig.vllm) {
      return {
        providerType: "vllm",
        config: {
          baseURL: providerConfig.vllm.host,
          apiKey: providerConfig.vllm.api_key,
        },
      };
    }

    // Fallback or generic handling if needed
    return null;
  } catch (error) {
    console.error(
      `Error fetching custom provider config for ID ${providerId}:`,
      error,
    );
    return null;
  }
}

export function selectImbraceModelId(
  models: ImbraceModel[],
  modelId: string,
): string {
  if (!Array.isArray(models) || models.length === 0) return modelId;

  const provider = models.find((m) => m.name === modelId)?.provider;
  if (!provider) return modelId;
  return provider.toLowerCase();
}

/**
 * Middleware that sanitises tool-call inputs in the prompt before they reach
 * the AI provider API.
 *
 * Reason: When a previous tool call was marked invalid (e.g. the SDK could
 * not parse the JSON produced by Qwen / Nova), the `input` field is stored
 * as a raw string. Providers like Bedrock and vLLM require `input` to be a
 * JSON object — sending a raw string causes a 400 error (e.g. Bedrock's
 * ValidationException or vLLM's "Can only get item pairs from a mapping").
 * This middleware ensures every `tool-call` part has an object `input`.
 */
const toolInputRepairMiddleware: LanguageModelMiddleware = {
  specificationVersion: "v3" as const,
  transformParams: async ({ params }) => {
    return {
      ...params,
      prompt: (params as any).prompt.map((msg: any) => {
        if (msg.role !== "assistant" || !Array.isArray(msg.content)) return msg;

        let changed = false;
        const content = msg.content.map((part: any) => {
          if (part.type !== "tool-call") return part;
          if (typeof part.input === "string") {
            changed = true;
            try {
              return { ...part, input: JSON.parse(part.input) };
            } catch {
              try {
                // Reason: jsonrepair handles truncated/malformed JSON produced
                // by Qwen/Nova (e.g. input cut off mid-string or mid-array).
                return { ...part, input: JSON.parse(jsonrepair(part.input)) };
              } catch {
                // Last resort: empty object prevents Bedrock 400 ValidationException
                return { ...part, input: {} };
              }
            }
          }
          return part;
        });

        return changed ? { ...msg, content } : msg;
      }),
    };
  },
};

function buildBedrockModel(modelId: string, providerConfig?: ProviderConfig) {
  if (providerConfig) {
    if (!providerConfig.access_key || !providerConfig.secret_key) {
      throw new Error("Missing credentials for custom Bedrock provider");
    }
  }

  const bedrock = createAmazonBedrock({
    ...(providerConfig?.region
      ? { region: providerConfig.region }
      : { region: config.aws.region_ai }),
    ...(providerConfig?.access_key
      ? { accessKeyId: providerConfig.access_key }
      : { accessKeyId: config.aws.accessKeyId }),
    ...(providerConfig?.secret_key
      ? { secretAccessKey: providerConfig.secret_key }
      : { secretAccessKey: config.aws.secretAccessKey }),
  });

  // Reason: Wrap with middleware to fix invalid tool-call inputs before
  // they are sent back to Bedrock on subsequent turns.
  return wrapLanguageModel({
    model: bedrock(modelId) as any,
    middleware: toolInputRepairMiddleware,
  });
}

function buildOpenAIModel(modelId: string, providerConfig?: ProviderConfig) {
  if (providerConfig && !providerConfig.apiKey) {
    throw new Error("Missing API Key for custom OpenAI provider");
  }

  const openai = createOpenAICompatible({
    name: "custom-openai",
    baseURL: providerConfig?.baseURL || config.openai.openAIProxyURL,
    ...(providerConfig?.apiKey
      ? { apiKey: providerConfig.apiKey }
      : { apiKey: config.openai.apiKey }),
  });
  return openai(modelId);
}

function buildGeminiModel(modelId: string, providerConfig?: ProviderConfig) {
  if (providerConfig && !providerConfig.apiKey) {
    throw new Error("Missing API Key for custom Google provider");
  }

  // Reason: @ai-sdk/google defaults to v1beta internally.
  // Respect custom provider baseURL > env proxy URL > SDK default (omit to let SDK use its own default).
  const baseURL =
    providerConfig?.baseURL || config.google.googleProxyURL || undefined;

  const gemini = createGoogleGenerativeAI({
    apiKey: providerConfig?.apiKey || config.google.apiKey,
    ...(baseURL ? { baseURL } : {}),
  });
  return gemini(modelId);
}

function buildOllamaModel(modelId: string, providerConfig?: ProviderConfig) {
  if (providerConfig && !providerConfig.host) {
    throw new Error("Missing host for custom Ollama provider");
  }

  const ollama = createOllama({
    baseURL: (providerConfig?.host || config.ollama.url) + "/api",
    ...(providerConfig?.apiKey ? { apiKey: providerConfig.apiKey } : {}),
  });
  return ollama(modelId);
}

/**
 * Builds a custom OpenAI-compatible model instance (DeepSeek, Groq, Together,
 * Fireworks, OpenRouter, etc.) using `createOpenAICompatible`. Unlike the
 * official `@ai-sdk/openai` factory — which targets the Responses API
 * (`/responses`) — this one targets Chat Completions (`/chat/completions`),
 * the endpoint every OpenAI-compatible third party actually implements.
 * @param modelId - The model identifier to use
 * @param providerConfig - Provider configuration with baseURL and apiKey
 * @returns An AI SDK model instance configured for the custom provider
 */
function buildCustomOpenAICompatibleModel(
  modelId: string,
  providerConfig?: ProviderConfig,
) {
  if (!providerConfig?.baseURL) {
    throw new Error("Missing base URL for custom OpenAI-compatible provider");
  }
  if (!providerConfig.apiKey) {
    throw new Error("Missing API Key for custom OpenAI-compatible provider");
  }

  const openai = createOpenAICompatible({
    name: "custom",
    baseURL: providerConfig.baseURL,
    apiKey: providerConfig.apiKey,
  });
  return openai(modelId);
}

/**
 * Builds a vLLM model instance using the OpenAI-compatible API.
 * @param modelId - The model identifier to use
 * @param providerConfig - Optional provider configuration with baseURL and optional apiKey
 * @returns An AI SDK model instance configured for vLLM
 */
function buildVllmModel(modelId: string, providerConfig?: ProviderConfig) {
  const baseURL = providerConfig?.baseURL || "http://localhost:8000"; // Default vLLM API URL
  if (!baseURL) {
    throw new Error("Missing base URL for vLLM provider");
  }

  const openai = createOpenAICompatible({
    name: "vLLM",
    baseURL,
    // Reason: vLLM may not require an API key, but createOpenAI requires
    // a non-empty apiKey string. Using "dummy" allows key-less deployments.
    apiKey: providerConfig?.apiKey || "dummy",
  });
  return openai(modelId);
}

/** Supported model families for provider-specific prompt injection. */
export type ModelFamily = "claude" | "gpt" | "gemini" | "ollama" | "generic";

/**
 * Detects the model family from the model ID and provider type.
 * Used to select the appropriate model-specific system prompt.
 * @param modelId - The model identifier (e.g. "anthropic.claude-3-5-sonnet", "gpt-4o")
 * @param providerType - The resolved provider type (e.g. "bedrock", "openai", "google")
 * @returns The detected model family
 */
export function detectModelFamily(
  modelId: string,
  providerType: string,
): ModelFamily {
  const id = modelId.toLowerCase();

  // Reason: Bedrock Claude models contain "anthropic" or "claude" in the model ID
  if (id.includes("claude") || id.includes("anthropic")) return "claude";
  if (
    id.includes("gpt") ||
    id.includes("o1") ||
    id.includes("o3") ||
    id.includes("o4")
  )
    return "gpt";
  if (id.includes("gemini")) return "gemini";
  if (providerType.includes("ollama")) return "ollama";

  return "generic";
}

export interface ResolvedModel {
  model: any;
  isToolCallAvailable: boolean;
  modelFamily: ModelFamily;
}

/**
 * Resolves the AI SDK model instance and tool-call capability for a given model/provider pair.
 * @param organizationId - The organization identifier
 * @param xAccessToken - Access token for API calls
 * @param modelId - The model identifier
 * @param provider_id - The provider identifier ("system" or a custom provider ID)
 * @returns The resolved model and whether it supports tool calling
 */
export async function resolveImbraceModel(
  organizationId: string,
  xAccessToken: string,
  modelId: string,
  provider_id: string,
): Promise<ResolvedModel> {
  try {
    // defaults
    let finalProviderType = "openai";
    let finalProviderConfig: ProviderConfig | undefined = undefined;
    let finalModelId = modelId;
    let isToolCallAvailable = true;

    if (provider_id === "system") {
      if (modelId === "Default") {
        // Reason: the system default must follow the deployment's own env
        // config (AI_PROVIDER), never a hardcoded OpenAI gpt-4o — OSS has no
        // OpenAI key, so that path failed with "You didn't provide an API key".
        // AI_PROVIDER=vllm → the env-configured vLLM model; otherwise the
        // env Bedrock model (aws.modelId).
        if (config.ai.provider === "vllm") {
          const vllmModelId = config.ai.vllm.modelId;
          console.log(
            `using default vLLM model (system): ${vllmModelId} @ ${config.ai.vllm.baseURL}`,
          );
          return {
            model: buildVllmModel(vllmModelId, {
              baseURL: config.ai.vllm.baseURL,
              apiKey: config.ai.vllm.apiKey,
            }),
            isToolCallAvailable: true,
            modelFamily: detectModelFamily(vllmModelId, "vllm"),
          };
        }
        const bedrockModelId = config.aws.modelId;
        console.log(`using default bedrock model (system): ${bedrockModelId}`);
        return {
          model: buildBedrockModel(bedrockModelId),
          isToolCallAvailable: true,
          modelFamily: detectModelFamily(bedrockModelId, "bedrock"),
        };
      }

      const models = await fetchImbraceModels(organizationId, xAccessToken);
      finalProviderType = selectImbraceModelId(models, modelId);

      // Reason: Use the model metadata to determine tool-call support.
      // Defaults to true for models that don't explicitly set this flag.
      const matchedModel = models.find((m) => m.name === modelId);
      if (matchedModel?.is_toolCall_available === false) {
        isToolCallAvailable = false;
      }

      // System providers use global config, so finalProviderConfig stays undefined
    } else {
      // Custom Provider
      const customProvider = await fetchCustomProviderConfig(
        provider_id,
        organizationId,
        xAccessToken,
      );
      if (customProvider) {
        finalProviderType = customProvider.providerType;
        finalProviderConfig = customProvider.config;
        if (customProvider.modelId) {
          finalModelId = customProvider.modelId;
        }
      } else {
        console.warn(
          `Could not fetch custom provider config for ${provider_id}, falling back to system defaults.`,
        );
      }
    }

    // Build the model based on type
    let model: any;
    if (finalProviderType.includes("bedrock")) {
      console.log(`using bedrock model (${provider_id})`);
      model = buildBedrockModel(finalModelId, finalProviderConfig);
    } else if (finalProviderType.includes("google")) {
      console.log(`using google model (${provider_id})`);
      model = buildGeminiModel(finalModelId, finalProviderConfig);
    } else if (finalProviderType.includes("openai")) {
      console.log(`using openai model (${provider_id})`);
      model = buildOpenAIModel(finalModelId, finalProviderConfig);
    } else if (finalProviderType.includes("custom")) {
      console.log(`using custom OpenAI-compatible model (${provider_id})`);
      model = buildCustomOpenAICompatibleModel(
        finalModelId,
        finalProviderConfig,
      );
    } else if (finalProviderType.includes("ollama")) {
      console.log(`using ollama model (${provider_id})`);
      model = buildOllamaModel(finalModelId, finalProviderConfig);
    } else if (finalProviderType.includes("vllm")) {
      console.log(`using vLLM model (${provider_id})`);
      model = buildVllmModel(finalModelId, finalProviderConfig);
    } else {
      console.log("using fallback bedrock model");
      model = buildBedrockModel(finalModelId, finalProviderConfig);
    }

    const modelFamily = detectModelFamily(finalModelId, finalProviderType);
    console.log(
      `Tool calling available: ${isToolCallAvailable}, model family: ${modelFamily}`,
    );
    return { model, isToolCallAvailable, modelFamily };
  } catch (err) {
    console.error("Error resolving model:", err);
    console.log("using error fallback bedrock model");
    return {
      model: buildBedrockModel(modelId),
      isToolCallAvailable: true,
      modelFamily: detectModelFamily(modelId, "bedrock"),
    };
  }
}
