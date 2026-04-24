import { Clock, Effect } from "effect";

function nanosToMs(nanos: bigint): number {
  return Number(nanos) / 1_000_000;
}

export function measureDurationMs<A, E, R>(
  effect: Effect.Effect<A, E, R>,
): Effect.Effect<readonly [A, number], E, R> {
  return Effect.gen(function*() {
    const startedAt = yield* Clock.currentTimeNanos;
    const value = yield* effect;
    const finishedAt = yield* Clock.currentTimeNanos;
    return [value, nanosToMs(finishedAt - startedAt)] as const;
  });
}
