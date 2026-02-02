//! Tests for Svelte code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_component() {
    let source = r#"<script>
    let count = 0;

    function increment() {
        count += 1;
    }
</script>

<button on:click={increment}>
    Clicked {count} times
</button>
"#;
    let units = parse(source, Language::Svelte, "test.svelte");
    assert_eq!(units.len(), 3);

    let text = build_embedding_text(&units[0]);
    assert_eq!(
        text,
        "Constant: count
Signature: let count = 0;
Code:
    let count = 0;
File: test test.svelte"
    );

    let text = build_embedding_text(&units[1]);
    assert_eq!(
        text,
        "Function: increment
Signature: function increment() {
Code:
    function increment() {
        count += 1;
    }
File: test test.svelte"
    );

    let text = build_embedding_text(&units[2]);
    assert_eq!(
        text,
        "Code block: template
Signature: <button on:click={increment}>
Code:
<button on:click={increment}>
    Clicked {count} times
</button>
File: test test.svelte"
    );
}

#[test]
fn test_reactive_declarations() {
    let source = r#"<script>
    let count = 0;
    $: doubled = count * 2;
    $: quadrupled = doubled * 2;

    function increment() {
        count += 1;
    }
</script>
"#;
    let units = parse(source, Language::Svelte, "test.svelte");
    assert_eq!(units.len(), 2);

    let text = build_embedding_text(&units[0]);
    assert_eq!(
        text,
        "Constant: count
Signature: let count = 0;
Code:
    let count = 0;
File: test test.svelte"
    );

    let text = build_embedding_text(&units[1]);
    assert_eq!(
        text,
        "Function: increment
Signature: function increment() {
Code:
    function increment() {
        count += 1;
    }
File: test test.svelte"
    );
}

#[test]
fn test_props() {
    let source = r#"<script>
    export let name = 'World';
    export let greeting = 'Hello';

    function greet() {
        return `${greeting}, ${name}!`;
    }
</script>"#;
    let units = parse(source, Language::Svelte, "test.svelte");
    assert_eq!(units.len(), 3);

    let name_unit = get_unit_by_name(&units, "name").unwrap();
    let name_text = build_embedding_text(name_unit);
    let expected_name = r#"Constant: name
Signature: export let name = 'World';
Code:
    export let name = 'World';
File: test test.svelte"#;
    assert_eq!(name_text, expected_name);

    let greet_unit = get_unit_by_name(&units, "greet").unwrap();
    let greet_text = build_embedding_text(greet_unit);
    let expected_greet = r#"Function: greet
Signature: function greet() {
Code:
    function greet() {
        return `${greeting}, ${name}!`;
    }
File: test test.svelte"#;
    assert_eq!(greet_text, expected_greet);
}

#[test]
fn test_typescript_support() {
    let source = r#"
<script lang="ts">
    interface User {
        name: string;
        age: number;
    }

    export let user: User;

    function getDisplayName(user: User): string {
        return `${user.name} (${user.age})`;
    }
</script>
"#;
    let units = parse(source, Language::Svelte, "test.svelte");

    assert!(!units.is_empty(), "Should extract TypeScript component");
}

#[test]
fn test_stores() {
    let source = r#"
<script>
    import { writable } from 'svelte/store';

    const count = writable(0);

    function increment() {
        count.update(n => n + 1);
    }

    function reset() {
        count.set(0);
    }
</script>
"#;
    let units = parse(source, Language::Svelte, "test.svelte");

    assert!(!units.is_empty(), "Should extract stores");
}

#[test]
fn test_lifecycle_functions() {
    let source = r#"
<script>
    import { onMount, onDestroy } from 'svelte';

    let data = null;

    onMount(async () => {
        const response = await fetch('/api/data');
        data = await response.json();
    });

    onDestroy(() => {
        console.log('Component destroyed');
    });
</script>
"#;
    let units = parse(source, Language::Svelte, "test.svelte");

    assert!(!units.is_empty(), "Should extract lifecycle functions");
}

#[test]
fn test_event_handlers() {
    let source = r#"
<script>
    function handleClick(event) {
        console.log('Clicked!', event);
    }

    function handleInput(event) {
        console.log('Input:', event.target.value);
    }
</script>
"#;
    let units = parse(source, Language::Svelte, "test.svelte");

    assert!(!units.is_empty(), "Should extract event handlers");
}

#[test]
fn test_context_api() {
    let source = r#"
<script>
    import { setContext, getContext } from 'svelte';

    const key = Symbol();

    function setTheme(theme) {
        setContext(key, theme);
    }

    function getTheme() {
        return getContext(key);
    }
</script>
"#;
    let units = parse(source, Language::Svelte, "test.svelte");

    assert!(!units.is_empty(), "Should extract context API");
}

#[test]
fn test_slots_and_props() {
    let source = r#"
<script>
    export let title = 'Default Title';
    export let subtitle = '';

    function formatTitle(title) {
        return title.toUpperCase();
    }
</script>

<div>
    <h1>{formatTitle(title)}</h1>
    {#if subtitle}
        <h2>{subtitle}</h2>
    {/if}
    <slot />
</div>
"#;
    let units = parse(source, Language::Svelte, "test.svelte");

    assert!(!units.is_empty(), "Should extract slots and props");
}

#[test]
fn test_async_await() {
    let source = r#"
<script>
    let promise = null;

    async function fetchData(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error('Network error');
        }
        return response.json();
    }

    function load() {
        promise = fetchData('/api/data');
    }
</script>
"#;
    let units = parse(source, Language::Svelte, "test.svelte");

    assert!(!units.is_empty(), "Should extract async functions");
}

#[test]
fn test_function_with_imports() {
    let source = r#"<script>
import axios from 'axios';

async function fetchUsers() {
    const response = await axios.get('/api/users');
    return response.data;
}
</script>
"#;
    let units = parse(source, Language::Svelte, "test.svelte");
    let func = get_unit_by_name(&units, "fetchUsers").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: fetchUsers
Signature: async function fetchUsers() {
Calls: get
Variables: const, response
Uses: axios
Code:
async function fetchUsers() {
    const response = await axios.get('/api/users');
    return response.data;
}
File: test test.svelte"#;
    assert_eq!(text, expected);
}
