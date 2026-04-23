import { expect, it } from "@effect/vitest";
import { Effect } from "effect";

import { selectOutputTensor } from "./encoder-session-engine.js";

it.effect("requires the curated ONNX output tensor by default", () =>
  Effect.sync(() => {
    expect(() =>
      selectOutputTensor({
        logits: { dims: [1, 2, 3] },
      } as never)
    ).toThrow(/required output tensor/);
  }),
);

it.effect("allows first-output fallback only when explicitly requested", () =>
  Effect.sync(() => {
    const tensor = { dims: [1, 2, 3] };
    expect(
      selectOutputTensor(
        {
          logits: tensor,
        } as never,
        { allowFallback: true },
      ),
    ).toBe(tensor);
  }),
);
