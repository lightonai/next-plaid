#!/usr/bin/env node

import { writeFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const topics = [
  "Alan_Turing",
  "Vincent_van_Gogh",
  "Photosynthesis",
  "Black_hole",
  "Great_Pyramid_of_Giza",
  "Jazz",
  "DNA",
  "Mount_Everest",
  "Marie_Curie",
  "Amazon_rainforest",
  "Byzantine_Empire",
  "Quantum_mechanics",
  "Silk_Road",
  "Penicillin",
  "Great_Wall_of_China",
  "Ada_Lovelace",
  "Volcano",
  "Apollo_11",
  "Mona_Lisa",
  "Tea",
  "Renaissance",
  "Coral_reef",
  "Printing_press",
  "Isaac_Newton",
  "Hibiscus",
  "Origami",
  "Hurricane",
  "Oldowan",
  "Fibonacci_number",
  "Nikola_Tesla",
];

const endpoint = "https://simple.wikipedia.org/api/rest_v1/page/summary/";

function normalizeExtract(text) {
  return text
    .replace(/\s+/g, " ")
    .replace(/\s+([.,;:!?])/g, "$1")
    .trim();
}

async function fetchTopic(title) {
  const response = await fetch(`${endpoint}${encodeURIComponent(title)}`, {
    headers: {
      "user-agent": "next-plaid-browser-demo/0.1 (https://github.com/lightonai)",
      "accept": "application/json",
    },
  });
  if (!response.ok) {
    throw new Error(`wiki fetch failed for ${title}: ${response.status} ${response.statusText}`);
  }
  const body = await response.json();
  const extract = normalizeExtract(String(body.extract ?? ""));
  if (extract.length < 120) {
    throw new Error(`wiki extract too short for ${title}: ${extract.length} chars`);
  }
  return {
    document_id: title.toLowerCase().replaceAll("_", "-"),
    semantic_text: extract,
    metadata: {
      slug: title.toLowerCase(),
      title: String(body.titles?.normalized ?? body.title ?? title.replaceAll("_", " ")),
      source: "Simple English Wikipedia",
      url: String(body.content_urls?.desktop?.page ?? `https://simple.wikipedia.org/wiki/${title}`),
      description: body.description ? String(body.description) : null,
    },
    source_span: {
      source_id: title.toLowerCase(),
      source_uri: String(body.content_urls?.desktop?.page ?? `https://simple.wikipedia.org/wiki/${title}`),
      title: String(body.titles?.normalized ?? body.title ?? title.replaceAll("_", " ")),
      excerpt: extract,
      locator: {
        type: "section",
        path: ["Simple English Wikipedia", String(body.titles?.normalized ?? body.title ?? title.replaceAll("_", " "))],
        anchor: title.toLowerCase(),
      },
    },
  };
}

async function main() {
  const docs = [];
  for (const title of topics) {
    try {
      const doc = await fetchTopic(title);
      docs.push(doc);
      process.stdout.write(`ok ${title} (${doc.semantic_text.length} chars)\n`);
    } catch (error) {
      process.stderr.write(`skip ${title}: ${error instanceof Error ? error.message : String(error)}\n`);
    }
  }

  const outPath = resolve(
    new URL(".", import.meta.url).pathname,
    "..",
    "playwright-harness",
    "fixtures",
    "wiki-corpus.json",
  );
  await mkdir(dirname(outPath), { recursive: true });
  const payload = {
    id: "simple-wiki-demo-v1",
    attribution: "Text excerpts from Simple English Wikipedia, CC BY-SA 3.0.",
    documents: docs,
    queries: [
      { id: "enigma-codebreaker", text: "Who cracked the Enigma code during the war?", expectedSlug: "alan_turing" },
      { id: "painter-cut-ear", text: "The painter who famously cut off his own ear", expectedSlug: "vincent_van_gogh" },
      { id: "gravity-apple", text: "The scientist who figured out gravity by watching fruit fall", expectedSlug: "isaac_newton" },
      { id: "oldest-living-coral", text: "Underwater ecosystems built by tiny animals over centuries", expectedSlug: "coral_reef" },
      { id: "mold-that-saves-lives", text: "A mold that turned out to kill bacteria and save lives", expectedSlug: "penicillin" },
      { id: "stone-wonder-egypt", text: "An ancient wonder made of huge stone blocks in the desert", expectedSlug: "great_pyramid_of_giza" },
      { id: "first-moon-landing", text: "When humans first walked on the surface of the moon", expectedSlug: "apollo_11" },
      { id: "folded-paper-art", text: "The Japanese art of folding paper into shapes", expectedSlug: "origami" },
      { id: "spiral-in-nature", text: "A number sequence that shows up in sunflowers and pinecones", expectedSlug: "fibonacci_number" },
      { id: "green-energy-of-plants", text: "How plants turn sunlight into food", expectedSlug: "photosynthesis" },
    ],
  };
  await writeFile(outPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  process.stdout.write(`\nwrote ${docs.length} docs -> ${outPath}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
