//! Tests for Java code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_method() {
    let source = r#"public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}"#;
    let units = parse(source, Language::Java, "Calculator.java");

    let class_unit = get_unit_by_name(&units, "Calculator").unwrap();
    let class_text = build_embedding_text(class_unit);
    let class_expected = r#"Class: Calculator
Signature: public class Calculator {
Code:
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
File: calculator Calculator.java"#;
    assert_eq!(class_text, class_expected);

    let method = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(method);

    let expected = "Method: add
Signature: public int add(int a, int b) {
Parameters: a, b
Returns: int
Code:
    public int add(int a, int b) {
        return a + b;
    }
File: calculator Calculator.java";
    assert_eq!(text, expected);
}

#[test]
fn test_method_with_javadoc() {
    let source = r#"public class Math {
    /**
     * Calculates the sum of two numbers.
     * @param a First number
     * @param b Second number
     * @return Sum of a and b
     */
    public int add(int a, int b) {
        return a + b;
    }
}"#;
    let units = parse(source, Language::Java, "Math.java");

    let method = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(method);

    let expected = "Method: add
Signature: public int add(int a, int b) {
Description: Calculates the sum of two numbers. @param a First number @param b Second number @return Sum of a and b /
Parameters: a, b
Returns: int
Code:
    public int add(int a, int b) {
        return a + b;
    }
File: math Math.java";
    assert_eq!(text, expected);
}

#[test]
fn test_static_method() {
    let source = r#"public class Utils {
    public static String format(String template, Object... args) {
        return String.format(template, args);
    }
}"#;
    let units = parse(source, Language::Java, "Utils.java");

    let method = get_unit_by_name(&units, "format").unwrap();
    let text = build_embedding_text(method);

    let expected = "Method: format
Signature: public static String format(String template, Object... args) {
Parameters: template
Returns: String
Calls: format
Variables: args
Code:
    public static String format(String template, Object... args) {
        return String.format(template, args);
    }
File: utils Utils.java";
    assert_eq!(text, expected);
}

#[test]
fn test_method_with_generics() {
    let source = r#"public class Container<T> {
    private T value;

    public T getValue() {
        return value;
    }

    public void setValue(T value) {
        this.value = value;
    }
}"#;
    let units = parse(source, Language::Java, "Container.java");

    let class_unit = get_unit_by_name(&units, "Container").unwrap();
    let class_text = build_embedding_text(class_unit);
    let class_expected = r#"Class: Container
Signature: public class Container<T> {
Code:
public class Container<T> {
    private T value;

    public T getValue() {
        return value;
    }

    public void setValue(T value) {
        this.value = value;
    }
}
File: container Container.java"#;
    assert_eq!(class_text, class_expected);

    let get_unit = get_unit_by_name(&units, "getValue").unwrap();
    let get_text = build_embedding_text(get_unit);
    let get_expected = r#"Method: getValue
Signature: public T getValue() {
Returns: T
Code:
    public T getValue() {
        return value;
    }
File: container Container.java"#;
    assert_eq!(get_text, get_expected);

    let set_unit = get_unit_by_name(&units, "setValue").unwrap();
    let set_text = build_embedding_text(set_unit);
    let set_expected = r#"Method: setValue
Signature: public void setValue(T value) {
Parameters: value
Returns: void
Code:
    public void setValue(T value) {
        this.value = value;
    }
File: container Container.java"#;
    assert_eq!(set_text, set_expected);
}

#[test]
fn test_constructor() {
    let source = r#"public class Person {
    private String name;
    private int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
}"#;
    let units = parse(source, Language::Java, "Person.java");

    let class_unit = get_unit_by_name(&units, "Person").unwrap();
    let class_text = build_embedding_text(class_unit);
    let class_expected = r#"Class: Person
Signature: public class Person {
Code:
public class Person {
    private String name;
    private int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
}
File: person Person.java"#;
    assert_eq!(class_text, class_expected);
}

#[test]
fn test_interface() {
    let source = r#"public interface Drawable {
    void draw();
    Rectangle getBounds();
}"#;
    let units = parse(source, Language::Java, "Drawable.java");

    let unit = get_unit_by_name(&units, "Drawable").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Class: Drawable
Signature: public interface Drawable {
Code:
public interface Drawable {
    void draw();
    Rectangle getBounds();
}
File: drawable Drawable.java"#;
    assert_eq!(text, expected);
}

#[test]
fn test_method_throws() {
    let source = r#"public class FileReader {
    public String read(String path) throws IOException {
        return Files.readString(Path.of(path));
    }
}"#;
    let units = parse(source, Language::Java, "FileReader.java");

    let method = get_unit_by_name(&units, "read").unwrap();
    let text = build_embedding_text(method);

    let expected = "Method: read
Signature: public String read(String path) throws IOException {
Parameters: path
Returns: String
Calls: of, readString
Code:
    public String read(String path) throws IOException {
        return Files.readString(Path.of(path));
    }
File: file reader FileReader.java";
    assert_eq!(text, expected);
}

#[test]
fn test_enum() {
    let source = r#"public enum Status {
    ACTIVE("active"),
    INACTIVE("inactive"),
    PENDING("pending");

    private final String value;

    Status(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }
}"#;
    let units = parse(source, Language::Java, "Status.java");

    let unit = get_unit_by_name(&units, "Status").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Class: Status
Signature: public enum Status {
Code:
public enum Status {
    ACTIVE("active"),
    INACTIVE("inactive"),
    PENDING("pending");

    private final String value;

    Status(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }
}
File: status Status.java"#;
    assert_eq!(text, expected);
}

#[test]
fn test_abstract_class() {
    let source = r#"public abstract class Shape {
    public abstract double area();

    public void describe() {
        System.out.println("I am a shape");
    }
}"#;
    let units = parse(source, Language::Java, "Shape.java");

    let class_unit = get_unit_by_name(&units, "Shape").unwrap();
    let class_text = build_embedding_text(class_unit);
    let class_expected = r#"Class: Shape
Signature: public abstract class Shape {
Code:
public abstract class Shape {
    public abstract double area();

    public void describe() {
        System.out.println("I am a shape");
    }
}
File: shape Shape.java"#;
    assert_eq!(class_text, class_expected);

    let method_unit = get_unit_by_name(&units, "describe").unwrap();
    let method_text = build_embedding_text(method_unit);
    let method_expected = r#"Method: describe
Signature: public void describe() {
Returns: void
Calls: println
Code:
    public void describe() {
        System.out.println("I am a shape");
    }
File: shape Shape.java"#;
    assert_eq!(method_text, method_expected);
}

#[test]
fn test_annotations() {
    let source = r#"public class Service {
    @Override
    @Deprecated
    public String toString() {
        return "Service";
    }

    @PostConstruct
    public void init() {
        // Initialize
    }
}"#;
    let units = parse(source, Language::Java, "Service.java");

    let method = get_unit_by_name(&units, "toString").unwrap();
    let text = build_embedding_text(method);
    let expected = r#"Method: toString
Signature: @Override
Returns: String
Code:
    @Override
    @Deprecated
    public String toString() {
        return "Service";
    }
File: service Service.java"#;
    assert_eq!(text, expected);
}

#[test]
fn test_lambda_expression() {
    let source = r#"public class StreamExample {
    public List<String> filter(List<String> items) {
        return items.stream()
            .filter(s -> s.startsWith("a"))
            .collect(Collectors.toList());
    }
}"#;
    let units = parse(source, Language::Java, "StreamExample.java");

    let unit = get_unit_by_name(&units, "filter").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Method: filter
Signature: public List<String> filter(List<String> items) {
Parameters: items
Returns: List<String>
Calls: collect, filter, startsWith, stream, toList
Code:
    public List<String> filter(List<String> items) {
        return items.stream()
            .filter(s -> s.startsWith("a"))
            .collect(Collectors.toList());
    }
File: stream example StreamExample.java"#;
    assert_eq!(text, expected);
}
