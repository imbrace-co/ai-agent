/**
 * Chat Processor
 * Handles execution of the main chat agent.
 * Follows the same pattern as subtask-processor for consistency.
 *
 * Utility functions are split into:
 * - chat-message-utils.ts      — message conversion & file hints
 * - chat-stream-bus-handler.ts  — bus event handling during streaming
 * - chat-prompt-builder.ts      — system prompt construction
 * - chat-agent-tools.ts         — tool set creation
 */

import {
  createUIMessageStream,
  createUIMessageStreamResponse,
  convertToModelMessages,
  generateText,
  stepCountIs,
  ToolLoopAgent,
  InvalidToolInputError,
  UIMessage as SDKUIMessage,
} from "ai";
import { repairToolInput } from "@/utils/repair-tool-input";
import { repeatedFailureStop } from "@/utils/stop-conditions";
import logger from "@/lib/logger";
import { resolveImbraceModel } from "@/providers/imbraceModels";
import { getAssistantSettings } from "@/utils/agent";
import { SessionBus } from "@/session/bus";
import { SessionManager } from "@/session/session";
import { runWithToolContext } from "@/core/agents/tool/toolContext";
import type { ToolContext } from "@/core/agents/tool/toolContext";
import {
  ChatContext,
  ChatInput,
  ChatOptions,
  ChatResult,
  UIMessage,
  DEFAULT_CHAT_OPTIONS,
} from "../types/chat";

import {
  toModelMessages,
  appendFilesAsToolHints,
  formatChatOutput,
} from "./chat-message-utils";
import { createBusStreamHandler } from "./chat-stream-bus-handler";
import {
  buildChatAgentPrompt,
  buildFinalPrompt,
  fetchFolderSummary,
} from "./chat-prompt-builder";
import { createChatAgentTools } from "./chat-agent-tools";

/**
 * Detect whether an error is a provider context-window overflow.
 * Matches common error messages from OpenAI, Anthropic, Google, and Bedrock.
 */
function isContextOverflowError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const msg = error.message.toLowerCase();
  return (
    (msg.includes("context") &&
      (msg.includes("overflow") || msg.includes("too long"))) ||
    msg.includes("maximum context length") ||
    msg.includes("max_tokens") ||
    msg.includes("token limit") ||
    msg.includes("input length") ||
    msg.includes("context_length_exceeded")
  );
}

/**
 * Generate a summary response when agent completes without final text.
 * @param model - AI model instance
 * @param messages - Conversation messages
 * @param stepCount - Number of tool steps executed
 * @param maxSteps - Maximum allowed steps
 * @param temperature - Model temperature
 * @param writer - Stream writer
 * @param chatId - Chat session ID
 */
async function generateSummaryResponse(
  model: unknown,
  messages: UIMessage[],
  stepCount: number,
  maxSteps: number,
  temperature: number,
  writer: { write: (data: unknown) => void },
  chatId: string,
): Promise<void> {
  logger.warn("No final response after tool execution, generating summary", {
    stepCount,
    maxSteps,
    hitLimit: stepCount >= maxSteps,
    chatId,
  });

  // Reason: Always use convertToModelMessages() directly instead of
  // the heuristic toModelMessages() which only checks the first message.
  // DB-loaded messages have `parts` (UIMessage format) and must be
  // explicitly converted to ModelMessage[] for generateText().
  const modelMessages = convertToModelMessages(
    messages as unknown as SDKUIMessage[],
  );

  const summaryResult = await generateText({
    model: model as Parameters<typeof generateText>[0]["model"],
    system: `You are a helpful assistant. Based on the ${stepCount} tool operation(s) completed and the context gathered, provide a clear and concise final summary response to the user.
Explain what was done and present the key findings or results.
${
  stepCount >= maxSteps
    ? "Note: The maximum number of tool calls was reached. If the task is incomplete, explain what was accomplished and what remains to be done."
    : ""
}
Be helpful and provide actionable information.`,
    messages: [
      ...modelMessages,
      {
        role: "assistant" as const,
        content: `I've completed ${stepCount} tool operations. Let me summarize the results for you.`,
      },
    ],
    temperature,
  });

  if (summaryResult.text) {
    const textPartId = `text-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    writer.write({ type: "text-start", id: textPartId });
    writer.write({
      type: "text-delta",
      id: textPartId,
      delta: summaryResult.text,
    });
    writer.write({ type: "text-end", id: textPartId });

    logger.info("Summary response generated after tool execution", {
      summaryLength: summaryResult.text.length,
      chatId,
    });
  }
}

/**
 * Process a chat stream using the main agent.
 * Similar to processSubtask but for the primary chat agent.
 * @param input - Chat input parameters
 * @param context - Chat context
 * @param options - Chat options
 * @returns Response with SSE stream
 */
export async function processChatStream(
  input: ChatInput,
  context: ChatContext,
  options: ChatOptions = {},
): Promise<Response> {
  const startTime = Date.now();
  const { messages, assistant_id } = input;
  const { organizationId, xAccessToken, sessionID } = context;

  const mergedOptions = { ...DEFAULT_CHAT_OPTIONS, ...options };
  const {
    maxSteps,
    temperature: defaultTemperature,
    showThinkingProcess,
  } = mergedOptions;

  logger.info(`Starting chat processing: ${sessionID}`, {
    messageCount: messages.length,
    assistantId: assistant_id,
  });

  // Reuse pre-fetched assistant from context when available, otherwise fetch
  const assistant =
    (context.assistant as Record<string, unknown> | undefined) ??
    (await getAssistantSettings(assistant_id, xAccessToken, organizationId));

  // Validate assistant configuration
  if (!assistant) {
    logger.warn("Assistant not found", { assistant_id, sessionID });
    return Response.json(
      { error: `Assistant not found: ${assistant_id}` },
      { status: 404 },
    );
  }

  if (!assistant.model_id || !assistant.provider_id) {
    logger.warn("Assistant missing model/provider configuration", {
      assistant_id,
      sessionID,
    });
    return Response.json(
      { error: "Assistant is missing model_id/provider_id configuration" },
      { status: 400 },
    );
  }

  // Short-circuit: when streaming is disabled, run the agent in non-streaming
  // mode but still emit the result as a single AI SDK UI message stream so
  // the frontend (which always expects SSE) can parse it uniformly.
  if ((assistant["streaming"] as boolean | undefined) === false) {
    const result = await processChat(
      input,
      { ...context, assistant },
      options,
    );
    const textPartId = `text-${Date.now()}`;
    if (!result.success && options.errorCapture && !options.errorCapture.errorText) {
      options.errorCapture.errorText = result.error || result.text || "Chat failed";
    }
    const stream = createUIMessageStream({
      // Persist the answer like the streaming path does; the chat UI reloads the
      // conversation from the server after the stream ends.
      ...(options.onStreamFinish ? { onFinish: options.onStreamFinish } : {}),
      execute: ({ writer }) => {
        writer.write({ type: "start" });
        if (result.success) {
          writer.write({ type: "text-start", id: textPartId });
          writer.write({ type: "text-delta", id: textPartId, delta: result.text });
          writer.write({ type: "text-end", id: textPartId });
        } else {
          writer.write({ type: "error", errorText: result.error || result.text || "Chat failed" });
        }
        writer.write({ type: "finish" });
      },
    });
    return createUIMessageStreamResponse({ stream, status: result.success ? 200 : 500 });
  }

  // Resolve AI model
  const { model, modelFamily } = await resolveImbraceModel(
    organizationId,
    xAccessToken,
    assistant.model_id as string,
    assistant.provider_id as string,
  );

  // Wrap entire stream creation in request-scoped tool context
  // Reason: AsyncLocalStorage isolates this context from concurrent requests
  const toolCtx: ToolContext = {
    xAccessToken,
    assistant_id,
    assistantData: assistant,
    organization_id: organizationId,
    thread_id: sessionID,
    user_id: context.userId,
  };

  // Create per-request session bus and manager for multi-agent orchestration
  const bus = new SessionBus();
  const sessionManager = new SessionManager(bus);
  // Reason: AbortController propagates cancellation to sub-agents when the
  // parent stream is aborted (e.g. client disconnect). Without this,
  // sub-agents keep running as ghost processes after the parent closes.
  const abortController = new AbortController();

  // Reason: createUIMessageStream uses originalMessages.at(-1) (when its role
  // is "assistant") to seed processUIMessageStream's state.message. Without
  // this seed, when the agent emits tool-output-available for a tool call
  // from the *previous* turn (HITL approval flow), the server-side state
  // tracker can't find the matching tool-call part and throws
  // "no tool invocation found for tool call <id>".
  const originalMessages = (
    Array.isArray(messages) && messages.length > 0
      ? (messages as unknown as SDKUIMessage[])
      : undefined
  );

  return runWithToolContext(toolCtx, () => {
    const stream = createUIMessageStream({
      ...(mergedOptions.onStreamFinish
        ? { onFinish: mergedOptions.onStreamFinish }
        : {}),
      ...(originalMessages ? { originalMessages } : {}),
      // Reason: AI SDK default onError masks errors as "An error occurred."
      // We forward the actual message so the FE can render the real cause
      // (e.g. provider cert expired, rate limit, validation failure).
      // Also capture into errorCapture so onStreamFinish persists the error
      // as the assistant message body (otherwise an empty row gets saved).
      onError: (error) => {
        const message =
          error instanceof Error ? error.message : String(error);
        logger.error("UI message stream error", { sessionID, error: message });
        if (mergedOptions.errorCapture) {
          // Reason: keep the first error — subsequent "No output generated."
          // is a downstream symptom, not the real cause.
          if (!mergedOptions.errorCapture.errorText) {
            mergedOptions.errorCapture.errorText = message;
          }
        }
        return message;
      },
      // Reason: SDK default `generateId` produces short non-UUID ids
      // (e.g. "abc123"). Our DB layer normalises non-UUIDs into a fresh
      // UUID when creating a message — which decouples the FE's message
      // id from the DB id and breaks Message.update() on HITL approval
      // continuation. Use crypto.randomUUID so server, FE, and DB share
      // the same UUID for each assistant message.
      generateId: () => crypto.randomUUID(),
      execute: async ({ writer }) => {
        const untypedWriter = writer as any;
        const subAgentAccumulator = mergedOptions.subAgentAccumulator;

        // Subscribe to session bus events and forward sub-agent events to client stream
        const unsubscribeBus = bus.subscribeAll(
          createBusStreamHandler(untypedWriter, subAgentAccumulator),
        );

        const { allTools, cleanup } = await createChatAgentTools(
          assistant,
          context,
          writer,
          model,
        );

        logger.info("Total tools available", {
          count: Object.keys(allTools).length,
          tools: Object.keys(allTools),
        });

        logger.info("Starting agent stream creation", { sessionID });

        // Pre-fetch folder summaries for prompt injection (non-blocking)
        const folderIds = assistant["folder_ids"] as string[] | undefined;
        const folderSummaries =
          Array.isArray(folderIds) && folderIds.length > 0
            ? await fetchFolderSummary(folderIds)
            : undefined;

        // Normalize messages and build instructions
        const normalizedMessages = appendFilesAsToolHints(
          messages as unknown as UIMessage[],
        );
        const agentPrompt = buildChatAgentPrompt(
          assistant,
          modelFamily,
          folderSummaries,
        );
        const finalPrompt = buildFinalPrompt(
          agentPrompt,
          "task" in allTools,
          context.subAgentSummary,
          context.boardId,
        );

        let stepCount = 0;
        // Reason: Tracks whether the agent paused for HITL approval. When the
        // model emits a tool call that requires `needsApproval`, the SDK
        // suspends execution and includes a `tool-approval-request` part on
        // the step. Without this flag we'd fall into the "no final text →
        // generate summary" branch below and the summary model would
        // hallucinate that the tool succeeded.
        let hasPendingApproval = false;
        const agentTemperature =
          (assistant.temperature as number) ?? defaultTemperature;

        // Reason: Track tool call patterns across steps to detect and warn
        // about repeated identical tool calls (e.g. RAG called 10+ times
        // with the same title). This is observability-only — actual dedup
        // is handled by the tool's internal cache.
        const toolCallTracker = new Map<string, number>();

        // Create and configure agent
        const agent = new ToolLoopAgent({
          model,
          tools: allTools,
          instructions: finalPrompt,
          temperature: agentTemperature,
          activeTools: Object.keys(allTools) as (keyof typeof allTools)[],
          // Reason: Circuit-breaker — stop the loop if the same tool+input fails
          // 3× in a row. Without this, the agent has been observed to retry the
          // same failing tool call 40+ times until maxSteps, burning ~1.3M tokens.
          stopWhen: [stepCountIs(maxSteps), repeatedFailureStop(3)],
          // Reason: Some Bedrock models (e.g. Amazon Nova, Qwen) occasionally
          // return tool call inputs in a format the SDK cannot parse — either
          // as a pre-stringified JSON string (double-encoded) or with encoding
          // quirks that trip the SDK's internal parser. When left unrepaired,
          // the invalid input gets rejected by Bedrock on the next turn.
          // This repair normalises the input by re-parsing and re-stringifying.
          experimental_repairToolCall: async ({ toolCall, error }) => {
            if (error instanceof InvalidToolInputError) {
              try {
                // Handle case where input is already an object (not stringified)
                if (
                  typeof toolCall.input === "object" &&
                  toolCall.input !== null
                ) {
                  logger.warn("Repaired object-typed tool call input", {
                    toolName: toolCall.toolName,
                    toolCallId: toolCall.toolCallId,
                  });
                  return { ...toolCall, input: JSON.stringify(toolCall.input) };
                }

                let parsed: unknown = toolCall.input;
                // Unwrap string-encoding layers (handles double/triple encoding)
                while (typeof parsed === "string") {
                  parsed = JSON.parse(parsed);
                }

                if (typeof parsed === "object" && parsed !== null) {
                  logger.warn("Repaired tool call input", {
                    toolName: toolCall.toolName,
                    toolCallId: toolCall.toolCallId,
                  });
                  return { ...toolCall, input: JSON.stringify(parsed) };
                }
              } catch {
                // Reason: repairToolInput first fixes XML-contaminated keys
                // (value leaked into key name with </parameter> tags), then
                // applies jsonrepair for truncated JSON from Qwen/Nova.
                try {
                  if (typeof toolCall.input === "string") {
                    const repaired = JSON.parse(
                      repairToolInput(toolCall.input),
                    );
                    if (typeof repaired === "object" && repaired !== null) {
                      logger.warn(
                        "Repaired tool call input via repairToolInput",
                        {
                          toolName: toolCall.toolName,
                          toolCallId: toolCall.toolCallId,
                        },
                      );
                      return { ...toolCall, input: JSON.stringify(repaired) };
                    }
                  }
                } catch {
                  // repairToolInput also failed — fall through
                }
              }
            }
            logger.warn("Could not repair tool call", {
              toolName: toolCall.toolName,
              toolCallId: toolCall.toolCallId,
              error: error.message,
            });
            return null;
          },
          onStepFinish: (step) => {
            stepCount++;

            // Reason: detect HITL pause — when a tool requires approval, the
            // SDK suspends execution and emits a `tool-approval-request` part
            // on the step content. Skip the "no final text" summary fallback
            // below so we don't hallucinate that the tool already ran.
            const stepContent = (step as { content?: Array<{ type?: string }> })
              .content;
            if (
              Array.isArray(stepContent) &&
              stepContent.some((p) => p?.type === "tool-approval-request")
            ) {
              hasPendingApproval = true;
              logger.info("Tool approval requested — pausing agent", {
                sessionID,
                stepCount,
              });
            }

            if (step.toolCalls?.length > 0) {
              logger.info("Tool calls:", { toolCalls: step.toolCalls });

              // Track tool call patterns for duplicate detection
              for (const tc of step.toolCalls) {
                const input = tc.input as Record<string, unknown> | undefined;
                const trackKey = `${tc.toolName}::${input?.["tool_title"] || ""}`;
                const count = (toolCallTracker.get(trackKey) || 0) + 1;
                toolCallTracker.set(trackKey, count);
                if (count >= 3) {
                  logger.warn("Repeated tool call detected", {
                    toolName: tc.toolName,
                    toolTitle: input?.["tool_title"],
                    repeatCount: count,
                    sessionID,
                  });
                }
              }
            }
            if (step.toolResults?.length > 0) {
              logger.info("Tool results:", { toolResults: step.toolResults });
            }
            if (step.text) {
              logger.info("Content:", { content: step.text });
            }

            if (stepCount >= maxSteps - 2) {
              logger.warn("Approaching tool step limit", {
                currentStep: stepCount,
                maxSteps,
                sessionID,
              });
            }
          },
          experimental_telemetry: {
            isEnabled: true,
            metadata: {
              chatId: sessionID,
              organizationId,
              assistantId: assistant_id,
              agentPrompt: finalPrompt,
            },
          },
        });

        try {
          const modelMessagesForAgent = toModelMessages(normalizedMessages);
          logger.info("Messages handed to agent.stream", {
            sessionID,
            count: modelMessagesForAgent.length,
            roles: modelMessagesForAgent.map((m) => m.role),
            preview: modelMessagesForAgent.map((m) => {
              const c = (m as { content: unknown }).content;
              if (typeof c === "string") return { role: m.role, text: c.slice(0, 200) };
              if (Array.isArray(c)) {
                return {
                  role: m.role,
                  parts: c.map((p: any) => ({
                    type: p?.type,
                    text: typeof p?.text === "string" ? p.text.slice(0, 200) : undefined,
                    toolName: p?.toolName,
                  })),
                };
              }
              return { role: m.role, content: c };
            }),
          });
          const agentStream = await agent.stream({
            messages: modelMessagesForAgent,
          });

          try {
            const uiStream = agentStream.toUIMessageStream({
              sendReasoning: showThinkingProcess,
            });
            await writer.merge(uiStream);
          } catch (streamErr) {
            if (isContextOverflowError(streamErr)) {
              logger.warn(
                "Context overflow detected during stream — marking for compaction",
                {
                  sessionID,
                  error:
                    streamErr instanceof Error
                      ? streamErr.message
                      : String(streamErr),
                },
              );
              if (mergedOptions.usageCapture) {
                mergedOptions.usageCapture.needsCompaction = true;
              }
              const noticeId = `compact-notice-${Date.now()}`;
              untypedWriter.write({ type: "text-start", id: noticeId });
              untypedWriter.write({
                type: "text-delta",
                id: noticeId,
                delta:
                  "_Context window exceeded — conversation will be compacted on your next message._",
              });
              untypedWriter.write({ type: "text-end", id: noticeId });
            } else {
              throw streamErr;
            }
          }

          const finalText = await agentStream.text;

          // Capture token usage for context compaction tracking
          try {
            const usage = await agentStream.totalUsage;
            if (usage && mergedOptions.usageCapture) {
              if (usage.inputTokens != null) {
                mergedOptions.usageCapture.inputTokens = usage.inputTokens;
              }
              if (usage.outputTokens != null) {
                mergedOptions.usageCapture.outputTokens = usage.outputTokens;
              }
              if (usage.totalTokens != null) {
                mergedOptions.usageCapture.totalTokens = usage.totalTokens;
              }
            }
            logger.info("Agent stream usage captured", {
              sessionID,
              inputTokens: usage?.inputTokens,
              outputTokens: usage?.outputTokens,
              totalTokens: usage?.totalTokens,
            });
          } catch (usageErr) {
            logger.warn("Failed to capture agent stream usage", {
              sessionID,
              error:
                usageErr instanceof Error
                  ? usageErr.message
                  : String(usageErr),
            });
          }

          const endTime = Date.now();
          logger.info("Agent stream completed", {
            stepCount,
            maxSteps,
            hasFinalText: Boolean(finalText?.trim()),
            sessionID,
            durationMs: endTime - startTime,
          });

          // Generate summary if needed
          const hasToolCalls = stepCount > 0;
          // Reason: skip the summary when the agent is paused for HITL —
          // there's no real result yet, so the summary model would invent
          // one (e.g. "Board created successfully") that contradicts the
          // pending approval state.
          const needsSummary =
            hasToolCalls && !finalText?.trim() && !hasPendingApproval;

          if (hasPendingApproval) {
            logger.info(
              "Agent paused for tool approval — skipping summary fallback",
              { sessionID, stepCount },
            );
          }

          if (needsSummary) {
            try {
              await generateSummaryResponse(
                model,
                normalizedMessages,
                stepCount,
                maxSteps,
                agentTemperature,
                writer as { write: (data: unknown) => void },
                sessionID,
              );
            } catch (summaryErr) {
              logger.warn("Failed to generate summary response", {
                sessionID,
                stepCount,
                error:
                  summaryErr instanceof Error
                    ? summaryErr.message
                    : String(summaryErr),
              });
            }
          }
        } finally {
          // Abort any running sub-agents before cleanup
          abortController.abort();
          // Cleanup: unsubscribe bus, dispose session resources, cleanup tools
          unsubscribeBus();
          sessionManager.dispose();
          bus.dispose();
          await cleanup();
        }
      },
    });

    return createUIMessageStreamResponse({ stream });
  });
}

/**
 * Process a chat request and return a non-streaming result.
 * Similar to processSubtask but returns ChatResult.
 * @param input - Chat input parameters
 * @param context - Chat context
 * @param options - Chat options
 * @returns Chat result
 */
export async function processChat(
  input: ChatInput,
  context: ChatContext,
  options: ChatOptions = {},
): Promise<ChatResult> {
  const startTime = Date.now();
  const { messages, assistant_id } = input;
  const { organizationId, xAccessToken, sessionID } = context;

  const mergedOptions = { ...DEFAULT_CHAT_OPTIONS, ...options };
  const { maxSteps, temperature: defaultTemperature } = mergedOptions;

  logger.info(`Starting non-streaming chat: ${sessionID}`, {
    messageCount: messages.length,
    assistantId: assistant_id,
  });

  // Reuse pre-fetched assistant from context when available, otherwise fetch
  const assistant =
    (context.assistant as Record<string, unknown> | undefined) ??
    (await getAssistantSettings(assistant_id, xAccessToken, organizationId));

  if (!assistant) {
    return {
      text: "",
      metadata: { error: true },
      sessionId: sessionID,
      success: false,
      error: `Assistant not found: ${assistant_id}`,
    };
  }

  if (!assistant.model_id || !assistant.provider_id) {
    return {
      text: "",
      metadata: { error: true },
      sessionId: sessionID,
      success: false,
      error: "Assistant is missing model_id/provider_id configuration",
    };
  }

  // Resolve AI model
  const { model, modelFamily } = await resolveImbraceModel(
    organizationId,
    xAccessToken,
    assistant.model_id as string,
    assistant.provider_id as string,
  );

  // Wrap in request-scoped tool context for concurrency isolation
  const toolCtx: ToolContext = {
    xAccessToken,
    assistant_id,
    assistantData: assistant,
    organization_id: organizationId,
    thread_id: sessionID,
    user_id: context.userId,
  };

  // Create per-request session bus and manager (non-streaming: events are discarded)
  const bus = new SessionBus();
  const sessionManager = new SessionManager(bus);
  const abortController = new AbortController();

  return runWithToolContext(toolCtx, async () => {
    // Create tools (writer is not needed for non-streaming)
    const mockWriter = { write: () => {}, merge: async () => {} };
    const { allTools, cleanup } = await createChatAgentTools(
      assistant,
      context,
      mockWriter,
      model,
    );

    // Pre-fetch folder summaries for prompt injection
    const folderIdsNonStream = assistant["folder_ids"] as string[] | undefined;
    const folderSummariesNonStream =
      Array.isArray(folderIdsNonStream) && folderIdsNonStream.length > 0
        ? await fetchFolderSummary(folderIdsNonStream)
        : undefined;

    const normalizedMessages = appendFilesAsToolHints(
      messages as unknown as UIMessage[],
    );
    const agentPrompt = buildChatAgentPrompt(
      assistant,
      modelFamily,
      folderSummariesNonStream,
    );
    const finalPrompt = buildFinalPrompt(
      agentPrompt,
      "task" in allTools,
      undefined,
      context.boardId,
    );
    const agentTemperature =
      (assistant.temperature as number) ?? defaultTemperature;

    let stepCount = 0;

    const agent = new ToolLoopAgent({
      model,
      tools: allTools,
      instructions: finalPrompt,
      temperature: agentTemperature,
      activeTools: Object.keys(allTools) as (keyof typeof allTools)[],
      stopWhen: [stepCountIs(maxSteps), repeatedFailureStop(3)],
      experimental_repairToolCall: async ({ toolCall, error }) => {
        if (error instanceof InvalidToolInputError) {
          try {
            if (typeof toolCall.input === "object" && toolCall.input !== null) {
              logger.warn("Repaired object-typed tool call input", {
                toolName: toolCall.toolName,
                toolCallId: toolCall.toolCallId,
              });
              return { ...toolCall, input: JSON.stringify(toolCall.input) };
            }

            let parsed: unknown = toolCall.input;
            while (typeof parsed === "string") {
              parsed = JSON.parse(parsed);
            }

            if (typeof parsed === "object" && parsed !== null) {
              logger.warn("Repaired tool call input", {
                toolName: toolCall.toolName,
                toolCallId: toolCall.toolCallId,
              });
              return { ...toolCall, input: JSON.stringify(parsed) };
            }
          } catch {
            // Reason: repairToolInput first fixes XML-contaminated keys
            // (value leaked into key name with </parameter> tags), then
            // applies jsonrepair for truncated JSON from Qwen/Nova.
            try {
              if (typeof toolCall.input === "string") {
                const repaired = JSON.parse(repairToolInput(toolCall.input));
                if (typeof repaired === "object" && repaired !== null) {
                  logger.warn("Repaired tool call input via repairToolInput", {
                    toolName: toolCall.toolName,
                    toolCallId: toolCall.toolCallId,
                  });
                  return { ...toolCall, input: JSON.stringify(repaired) };
                }
              }
            } catch {
              // repairToolInput also failed — fall through
            }
          }
        }
        return null;
      },
      onStepFinish: () => {
        stepCount++;
      },
    });

    try {
      const result = await agent.generate({
        messages: toModelMessages(normalizedMessages),
      });

      const endTime = Date.now();
      const rawOutputText = result.text || "";
      const outputText = input.fromConnector
        ? formatChatOutput(sessionID, rawOutputText)
        : rawOutputText;

      logger.info("Chat completed (non-streaming)", {
        sessionID,
        stepCount,
        durationMs: endTime - startTime,
      });

      // Extract usage info safely
      const usageInfo = result.usage
        ? {
            promptTokens:
              (result.usage as Record<string, number>)["promptTokens"] ?? 0,
            completionTokens:
              (result.usage as Record<string, number>)["completionTokens"] ?? 0,
            totalTokens:
              (result.usage as Record<string, number>)["totalTokens"] ?? 0,
          }
        : undefined;

      return {
        text: outputText,
        metadata: {
          model: {
            modelID: assistant.model_id,
            providerID: assistant.provider_id,
          },
          duration: endTime - startTime,
          steps: stepCount,
          usage: result.usage,
        },
        sessionId: sessionID,
        success: true,
        usage: usageInfo,
      };
    } catch (error) {
      const endTime = Date.now();
      logger.error("Chat failed", {
        sessionID,
        error,
        durationMs: endTime - startTime,
      });

      return {
        text: "",
        metadata: {
          error: true,
          errorMessage: error instanceof Error ? error.message : String(error),
          duration: endTime - startTime,
        },
        sessionId: sessionID,
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    } finally {
      abortController.abort();
      sessionManager.dispose();
      bus.dispose();
      await cleanup();
    }
  });
}
