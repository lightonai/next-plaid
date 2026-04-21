import { expect, it } from "@effect/vitest";
import { Effect, Fiber } from "effect";
import { TestClock } from "effect/testing";

import { measureDurationMs } from "./effect-timing.js";

it.effect("measures durations through the Effect clock", () =>
  Effect.gen(function*() {
    const measuredFiber = yield* measureDurationMs(
      Effect.sleep("3 seconds").pipe(Effect.as("done")),
    ).pipe(
      Effect.forkChild({ startImmediately: true }),
    );

    yield* Effect.yieldNow;
    yield* TestClock.adjust("3 seconds");

    const [value, durationMs] = yield* Fiber.join(measuredFiber);
    expect(value).toBe("done");
    expect(durationMs).toBe(3000);
  }),
);
