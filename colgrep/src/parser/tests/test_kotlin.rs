//! Tests for Kotlin code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"fun greet(name: String): String {
    return "Hello, $name!"
}"#;
    let units = parse(source, Language::Kotlin, "test.kt");

    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: greet
Signature: fun greet(name: String): String {
Parameters: name
Code:
fun greet(name: String): String {
    return "Hello, $name!"
}
File: test test.kt"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_kdoc() {
    let source = r#"/**
 * Calculates the sum of two numbers.
 * @param a First number
 * @param b Second number
 * @return Sum of a and b
 */
fun add(a: Int, b: Int): Int {
    return a + b
}"#;
    let units = parse(source, Language::Kotlin, "test.kt");

    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: add
Signature: fun add(a: Int, b: Int): Int {
Description: Calculates the sum of two numbers. @param a First number @param b Second number @return Sum of a and b /
Parameters: a, b
Code:
fun add(a: Int, b: Int): Int {
    return a + b
}
File: test test.kt"#;
    assert_eq!(text, expected);
}

#[test]
fn test_class_definition() {
    let source = r#"class Person(val name: String, var age: Int) {
    fun greet(): String {
        return "Hello, I'm $name"
    }
}"#;
    let units = parse(source, Language::Kotlin, "test.kt");

    let unit = get_unit_by_name(&units, "Person").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Class: Person
Signature: class Person(val name: String, var age: Int) {
Code:
class Person(val name: String, var age: Int) {
    fun greet(): String {
        return "Hello, I'm $name"
    }
}
File: test test.kt"#;
    assert_eq!(text, expected);
}

#[test]
fn test_extension_function() {
    let source = r#"fun String.addExclamation(): String {
    return this + "!"
}"#;
    let units = parse(source, Language::Kotlin, "test.kt");

    let unit = get_unit_by_name(&units, "addExclamation").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: addExclamation
Signature: fun String.addExclamation(): String {
Code:
fun String.addExclamation(): String {
    return this + "!"
}
File: test test.kt"#;
    assert_eq!(text, expected);
}

#[test]
fn test_suspend_function() {
    let source = r#"suspend fun fetchData(url: String): String {
    return withContext(Dispatchers.IO) {
        URL(url).readText()
    }
}"#;
    let units = parse(source, Language::Kotlin, "test.kt");

    let func = get_unit_by_name(&units, "fetchData").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: fetchData
Signature: suspend fun fetchData(url: String): String {
Parameters: url
Calls: Dispatchers, IO), URL, URL(url), readText, withContext
Code:
suspend fun fetchData(url: String): String {
    return withContext(Dispatchers.IO) {
        URL(url).readText()
    }
}
File: test test.kt"#;
    assert_eq!(text, expected);
}

#[test]
fn test_data_class() {
    let source = r#"data class User(val id: Int, val name: String, val email: String)"#;
    let units = parse(source, Language::Kotlin, "test.kt");

    let unit = get_unit_by_name(&units, "User").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Class: User
Signature: data class User(val id: Int, val name: String, val email: String)
Code:
data class User(val id: Int, val name: String, val email: String)
File: test test.kt"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_generics() {
    let source = r#"fun <T> identity(value: T): T {
    return value
}

fun <K, V> createPair(key: K, value: V): Pair<K, V> {
    return Pair(key, value)
}"#;
    let units = parse(source, Language::Kotlin, "test.kt");

    let unit = get_unit_by_name(&units, "identity").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: identity
Signature: fun <T> identity(value: T): T {
Parameters: value
Code:
fun <T> identity(value: T): T {
    return value
}
File: test test.kt"#;
    assert_eq!(text, expected);

    let unit = get_unit_by_name(&units, "createPair").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: createPair
Signature: fun <K, V> createPair(key: K, value: V): Pair<K, V> {
Parameters: key, value
Calls: Pair
Code:
fun <K, V> createPair(key: K, value: V): Pair<K, V> {
    return Pair(key, value)
}
File: test test.kt"#;
    assert_eq!(text, expected);
}

#[test]
fn test_object_declaration() {
    let source = r#"object Singleton {
    fun doSomething(): String {
        return "singleton"
    }
}"#;
    let units = parse(source, Language::Kotlin, "test.kt");

    let unit = get_unit_by_name(&units, "Singleton").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Class: Singleton
Signature: object Singleton {
Code:
object Singleton {
    fun doSomething(): String {
        return "singleton"
    }
}
File: test test.kt"#;
    assert_eq!(text, expected);
}

#[test]
fn test_companion_object() {
    let source = r#"class Factory {
    companion object {
        fun create(): Factory {
            return Factory()
        }
    }
}"#;
    let units = parse(source, Language::Kotlin, "test.kt");

    let unit = get_unit_by_name(&units, "Factory").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Class: Factory
Signature: class Factory {
Calls: Factory
Code:
class Factory {
    companion object {
        fun create(): Factory {
            return Factory()
        }
    }
}
File: test test.kt"#;
    assert_eq!(text, expected);
}

#[test]
fn test_sealed_class() {
    let source = r#"sealed class Result<out T> {
    data class Success<T>(val data: T) : Result<T>()
    data class Error(val message: String) : Result<Nothing>()
}"#;
    let units = parse(source, Language::Kotlin, "test.kt");

    let unit = get_unit_by_name(&units, "Result").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Class: Result
Signature: sealed class Result<out T> {
Code:
sealed class Result<out T> {
    data class Success<T>(val data: T) : Result<T>()
    data class Error(val message: String) : Result<Nothing>()
}
File: test test.kt"#;
    assert_eq!(text, expected);
}

#[test]
fn test_inline_function() {
    let source = r#"inline fun <reified T> parseJson(json: String): T {
    return Gson().fromJson(json, T::class.java)
}"#;
    let units = parse(source, Language::Kotlin, "test.kt");

    let unit = get_unit_by_name(&units, "parseJson").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: parseJson
Signature: inline fun <reified T> parseJson(json: String): T {
Parameters: json
Calls: Gson, Gson(), T, class, fromJson
Code:
inline fun <reified T> parseJson(json: String): T {
    return Gson().fromJson(json, T::class.java)
}
File: test test.kt"#;
    assert_eq!(text, expected);
}

#[test]
fn test_class_inheritance() {
    let source = r#"open class Animal {
    open fun speak(): String {
        return "..."
    }
}

class Dog : Animal() {
    override fun speak(): String {
        return "Woof!"
    }
}"#;
    let units = parse(source, Language::Kotlin, "test.kt");

    let animal = get_unit_by_name(&units, "Animal").unwrap();
    let animal_text = build_embedding_text(animal);
    // Animal has no parent
    assert!(!animal_text.contains("Extends:"));

    let dog = get_unit_by_name(&units, "Dog").unwrap();
    let dog_text = build_embedding_text(dog);
    let expected_dog = r#"Class: Dog
Signature: class Dog : Animal() {
Extends: Animal
Code:
class Dog : Animal() {
    override fun speak(): String {
        return "Woof!"
    }
}
File: test test.kt"#;
    assert_eq!(dog_text, expected_dog);
}

#[test]
fn test_function_with_imports() {
    let source = r#"import java.util.Arrays

fun sortArray(arr: IntArray) {
    Arrays.sort(arr)
}"#;
    let units = parse(source, Language::Kotlin, "test.kt");
    let func = get_unit_by_name(&units, "sortArray").unwrap();
    let text = build_embedding_text(func);
    let expected = r#"Function: sortArray
Signature: fun sortArray(arr: IntArray) {
Parameters: arr
Calls: Arrays, sort
Uses: Arrays
Code:
fun sortArray(arr: IntArray) {
    Arrays.sort(arr)
}
File: test test.kt"#;
    assert_eq!(text, expected);
}
