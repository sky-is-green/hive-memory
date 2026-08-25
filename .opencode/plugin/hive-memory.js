// HiveMemory plugin for opencode — native curation seam (Mode C).
//
// On every user turn this plugin:
//   1. observes the previous assistant reply back into the hive store
//   2. calls the studio to assemble curated context for the new query
//   3. merges that context into the leading system message
//
// The caller's own provider generates the reply — the plugin only curates,
// so it works with ANY provider/model opencode is configured for (local or
// hosted). Every call fails open: if the studio is unreachable the session
// continues untouched, never broken.
//
// Configuration (env):
//   HIVE_STUDIO_URL       studio base URL   (default http://127.0.0.1:8765)
//   HIVE_CONVERSATION_ID  conversation key  (default: opencode-<folder name>)
//
// NOTE: mutually exclusive with routing opencode through the studio's
// /v1/openai endpoint — that would curate twice.

const STUDIO_URL = process.env.HIVE_STUDIO_URL || "http://127.0.0.1:8765";

async function hiveFetch(path, body) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 8000);
  try {
    const res = await fetch(STUDIO_URL + path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null; // studio down -> session proceeds uncurated
  } finally {
    clearTimeout(timer);
  }
}

function lastOf(messages, role) {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m && m.role === role && typeof m.content === "string" && m.content.trim()) {
      return m.content;
    }
  }
  return "";
}

export const HiveMemoryPlugin = async ({ directory }) => {
  const folder = String(directory || "default")
    .split(/[\\/]/)
    .filter(Boolean)
    .pop();
  const conversationId =
    process.env.HIVE_CONVERSATION_ID ||
    "opencode-" +
      String(folder || "default")
        .toLowerCase()
        .replace(/[^a-z0-9\-_]/g, "-");

  return {
    "experimental.chat.messages.transform": async (_input, output) => {
      try {
        const messages =
          (output && Array.isArray(output.messages) && output.messages) ||
          (output && Array.isArray(output) && output) ||
          [];
        if (!messages.length) return;

        // 1. observe the previous assistant reply into the store
        const prevReply = lastOf(messages, "assistant");
        if (prevReply) {
          await hiveFetch("/v1/hive/observe", {
            conversation_id: conversationId,
            reply: prevReply,
          });
        }

        // 2. curate the incoming query
        const query = lastOf(messages, "user");
        if (!query) return;
        const curated = await hiveFetch("/v1/hive/curate", {
          query,
          conversation_id: conversationId,
        });
        const content = curated && curated.assembled_content;
        if (!content || !content.trim()) return;

        // 3. merge curated context into the leading system message
        if (messages[0] && messages[0].role === "system") {
          messages[0] = { ...messages[0], content: content + "\n\n" + messages[0].content };
        } else {
          messages.unshift({ role: "system", content });
        }
      } catch {
        // never break the agent session over curation
      }
    },
  };
};

export default HiveMemoryPlugin;