//! Tests for Vue code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_options_api_component() {
    let source = r#"<script>
export default {
    name: 'MyComponent',
    data() {
        return { count: 0 }
    },
    methods: {
        increment() {
            this.count++
        }
    }
}
</script>

<template>
    <button @click="increment">{{ count }}</button>
</template>
"#;
    let units = parse(source, Language::Vue, "test.vue");

    assert_eq!(units.len(), 3);

    let text = build_embedding_text(&units[0]);
    assert_eq!(text, "Function: data\nSignature: data() {\nCode:\n    data() {\n        return { count: 0 }\n    },\nFile: test test.vue");

    let text = build_embedding_text(&units[1]);
    assert_eq!(text, "Function: increment\nSignature: increment() {\nCode:\n        increment() {\n            this.count++\n        }\nFile: test test.vue");

    let text = build_embedding_text(&units[2]);
    assert_eq!(text, "Code block: template\nSignature: <button @click=\"increment\">{{ count }}</button>\nCode:\n    <button @click=\"increment\">{{ count }}</button>\nFile: test test.vue");
}

#[test]
fn test_composition_api_setup() {
    let source = r#"<script setup>
import { ref } from 'vue'

const count = ref(0)

function increment() {
    count.value++
}
</script>

<template>
    <button @click="increment">{{ count }}</button>
</template>
"#;
    let units = parse(source, Language::Vue, "test.vue");

    assert_eq!(units.len(), 3);

    let text = build_embedding_text(&units[0]);
    assert_eq!(text, "Constant: count\nSignature: const count = ref(0)\nCode:\nconst count = ref(0)\nFile: test test.vue");

    let text = build_embedding_text(&units[1]);
    assert_eq!(text, "Function: increment\nSignature: function increment() {\nCode:\nfunction increment() {\n    count.value++\n}\nFile: test test.vue");

    let text = build_embedding_text(&units[2]);
    assert_eq!(text, "Code block: template\nSignature: <button @click=\"increment\">{{ count }}</button>\nCode:\n    <button @click=\"increment\">{{ count }}</button>\nFile: test test.vue");
}

#[test]
fn test_script_with_typescript() {
    let source = r#"<script lang="ts">
import { defineComponent, ref } from 'vue'

interface User {
    name: string
    age: number
}

export default defineComponent({
    setup() {
        const user = ref<User>({ name: 'John', age: 30 })
        return { user }
    }
})
</script>
"#;
    let units = parse(source, Language::Vue, "test.vue");

    assert_eq!(units.len(), 3);

    let text = build_embedding_text(&units[0]);
    assert_eq!(text, "Class: User\nSignature: interface User {\nCode:\ninterface User {\n    name: string\n    age: number\n}\nFile: test test.vue");

    let text = build_embedding_text(&units[1]);
    assert_eq!(text, "Function: setup\nSignature: setup() {\nCalls: ref\nVariables: const, user\nCode:\n    setup() {\n        const user = ref<User>({ name: 'John', age: 30 })\n        return { user }\n    }\nFile: test test.vue");

    let text = build_embedding_text(&units[2]);
    assert_eq!(text, "Constant: user\nSignature: const user = ref<User>({ name: 'John', age: 30 })\nCode:\n        const user = ref<User>({ name: 'John', age: 30 })\nFile: test test.vue");
}

#[test]
fn test_computed_properties() {
    let source = r#"
<script>
export default {
    data() {
        return {
            firstName: 'John',
            lastName: 'Doe'
        }
    },
    computed: {
        fullName() {
            return `${this.firstName} ${this.lastName}`
        }
    }
}
</script>
"#;
    let units = parse(source, Language::Vue, "test.vue");

    assert!(!units.is_empty(), "Should extract computed properties");
}

#[test]
fn test_lifecycle_hooks() {
    let source = r#"
<script>
export default {
    mounted() {
        console.log('Component mounted')
    },
    beforeUnmount() {
        console.log('Component will unmount')
    }
}
</script>
"#;
    let units = parse(source, Language::Vue, "test.vue");

    assert!(!units.is_empty(), "Should extract lifecycle hooks");
}

#[test]
fn test_props_definition() {
    let source = r#"
<script>
export default {
    props: {
        title: {
            type: String,
            required: true
        },
        count: {
            type: Number,
            default: 0
        }
    },
    methods: {
        handleClick() {
            this.$emit('clicked', this.count)
        }
    }
}
</script>
"#;
    let units = parse(source, Language::Vue, "test.vue");

    assert!(!units.is_empty(), "Should extract props");
}

#[test]
fn test_async_setup() {
    let source = r#"
<script setup>
import { ref } from 'vue'

const data = ref(null)

async function fetchData() {
    const response = await fetch('/api/data')
    data.value = await response.json()
}

fetchData()
</script>
"#;
    let units = parse(source, Language::Vue, "test.vue");

    assert!(!units.is_empty(), "Should extract async setup");
}

#[test]
fn test_watchers() {
    let source = r#"
<script>
export default {
    data() {
        return { searchQuery: '' }
    },
    watch: {
        searchQuery(newValue, oldValue) {
            this.performSearch(newValue)
        }
    },
    methods: {
        performSearch(query) {
            console.log('Searching:', query)
        }
    }
}
</script>
"#;
    let units = parse(source, Language::Vue, "test.vue");

    assert!(!units.is_empty(), "Should extract watchers");
}

#[test]
fn test_composables() {
    let source = r#"
<script setup>
import { ref, onMounted } from 'vue'

function useCounter() {
    const count = ref(0)
    const increment = () => count.value++
    return { count, increment }
}

const { count, increment } = useCounter()
</script>
"#;
    let units = parse(source, Language::Vue, "test.vue");

    assert!(!units.is_empty(), "Should extract composables");
}

#[test]
fn test_function_with_imports() {
    let source = r#"<script setup>
import { ref, computed } from 'vue'
import axios from 'axios'

const data = ref(null)

async function fetchData(url) {
    const response = await axios.get(url)
    data.value = response.data
}
</script>
"#;
    let units = parse(source, Language::Vue, "test.vue");
    let func = get_unit_by_name(&units, "fetchData").unwrap();
    let text = build_embedding_text(func);

    // Vue inherits Uses support from JavaScript/TypeScript handling
    let expected = r#"Function: fetchData
Signature: async function fetchData(url) {
Parameters: url
Calls: get
Variables: const, response
Uses: axios
Code:
async function fetchData(url) {
    const response = await axios.get(url)
    data.value = response.data
}
File: test test.vue"#;
    assert_eq!(text, expected);
}
