import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

type HookOutput = {
	hookSpecificOutput?: { additionalContext?: string };
};

export default function (pi: ExtensionAPI) {
	let instructions = "";

	pi.on("session_start", async (_event, ctx) => {
		instructions = "";
		try {
			const result = await pi.exec("colgrep", ["--session-hook"], {
				cwd: ctx.cwd,
				timeout: 10_000,
			});
			if (result.code === 0) {
				instructions = (JSON.parse(result.stdout) as HookOutput).hookSpecificOutput?.additionalContext ?? "";
			}
		} catch {
			// Keep Pi usable if colgrep is missing or returns malformed output.
		}
	});

	pi.on("before_agent_start", (event) => {
		if (instructions) {
			return { systemPrompt: `${event.systemPrompt}\n\n${instructions}` };
		}
	});
}
