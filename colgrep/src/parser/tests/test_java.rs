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

    // Class should be extracted as a single chunk containing all methods
    // Methods are NOT extracted separately - they remain inside the class chunk
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

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "add").is_none(),
        "Methods should not be extracted separately from classes"
    );
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

    // Class is extracted as a single chunk with javadoc and method inside
    let class_unit = get_unit_by_name(&units, "Math").unwrap();
    let class_text = build_embedding_text(class_unit);
    let class_expected = r#"Class: Math
Signature: public class Math {
Code:
public class Math {
    /**
     * Calculates the sum of two numbers.
     * @param a First number
     * @param b Second number
     * @return Sum of a and b
     */
    public int add(int a, int b) {
        return a + b;
    }
}
File: math Math.java"#;
    assert_eq!(class_text, class_expected);

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "add").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_static_method() {
    let source = r#"public class Utils {
    public static String format(String template, Object... args) {
        return String.format(template, args);
    }
}"#;
    let units = parse(source, Language::Java, "Utils.java");

    // Class is extracted as a single chunk with static method inside
    let class_unit = get_unit_by_name(&units, "Utils").unwrap();
    let class_text = build_embedding_text(class_unit);
    let class_expected = r#"Class: Utils
Signature: public class Utils {
Calls: format
Variables: args
Code:
public class Utils {
    public static String format(String template, Object... args) {
        return String.format(template, args);
    }
}
File: utils Utils.java"#;
    assert_eq!(class_text, class_expected);

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "format").is_none(),
        "Methods should not be extracted separately from classes"
    );
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

    // Class is extracted as a single chunk with all methods inside
    let class_unit = get_unit_by_name(&units, "Container").unwrap();
    let class_text = build_embedding_text(class_unit);
    let class_expected = r#"Class: Container
Signature: public class Container<T> {
Parameters: T
Variables: value
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

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "getValue").is_none(),
        "Methods should not be extracted separately from classes"
    );
    assert!(
        get_unit_by_name(&units, "setValue").is_none(),
        "Methods should not be extracted separately from classes"
    );
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
Variables: age, name
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

    // Class is extracted as a single chunk with method inside
    let class_unit = get_unit_by_name(&units, "FileReader").unwrap();
    let class_text = build_embedding_text(class_unit);
    let class_expected = r#"Class: FileReader
Signature: public class FileReader {
Calls: of, readString
Code:
public class FileReader {
    public String read(String path) throws IOException {
        return Files.readString(Path.of(path));
    }
}
File: file reader FileReader.java"#;
    assert_eq!(class_text, class_expected);

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "read").is_none(),
        "Methods should not be extracted separately from classes"
    );
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
Variables: value
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

    // Abstract class is extracted as a single chunk with all methods inside
    let class_unit = get_unit_by_name(&units, "Shape").unwrap();
    let class_text = build_embedding_text(class_unit);
    let class_expected = r#"Class: Shape
Signature: public abstract class Shape {
Calls: println
Code:
public abstract class Shape {
    public abstract double area();

    public void describe() {
        System.out.println("I am a shape");
    }
}
File: shape Shape.java"#;
    assert_eq!(class_text, class_expected);

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "describe").is_none(),
        "Methods should not be extracted separately from classes"
    );
    assert!(
        get_unit_by_name(&units, "area").is_none(),
        "Abstract methods should not be extracted separately from classes"
    );
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

    // Class is extracted as a single chunk with annotated methods inside
    let class_unit = get_unit_by_name(&units, "Service").unwrap();
    let class_text = build_embedding_text(class_unit);
    let class_expected = r#"Class: Service
Signature: public class Service {
Code:
public class Service {
    @Override
    @Deprecated
    public String toString() {
        return "Service";
    }

    @PostConstruct
    public void init() {
        // Initialize
    }
}
File: service Service.java"#;
    assert_eq!(class_text, class_expected);

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "toString").is_none(),
        "Methods should not be extracted separately from classes"
    );
    assert!(
        get_unit_by_name(&units, "init").is_none(),
        "Methods should not be extracted separately from classes"
    );
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

    // Class is extracted as a single chunk with lambda-containing method inside
    let class_unit = get_unit_by_name(&units, "StreamExample").unwrap();
    let class_text = build_embedding_text(class_unit);
    let class_expected = r#"Class: StreamExample
Signature: public class StreamExample {
Calls: collect, filter, startsWith, stream, toList
Code:
public class StreamExample {
    public List<String> filter(List<String> items) {
        return items.stream()
            .filter(s -> s.startsWith("a"))
            .collect(Collectors.toList());
    }
}
File: stream example StreamExample.java"#;
    assert_eq!(class_text, class_expected);

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "filter").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_class_inheritance() {
    let source = r#"public class Animal {
    public void speak() {
        System.out.println("...");
    }
}

public class Dog extends Animal {
    @Override
    public void speak() {
        System.out.println("Woof!");
    }
}"#;
    let units = parse(source, Language::Java, "Animals.java");

    let animal = get_unit_by_name(&units, "Animal").unwrap();
    let animal_text = build_embedding_text(animal);
    // Animal has no parent
    assert!(!animal_text.contains("Extends:"));

    let dog = get_unit_by_name(&units, "Dog").unwrap();
    let dog_text = build_embedding_text(dog);
    let expected_dog = r#"Class: Dog
Signature: public class Dog extends Animal {
Extends: Animal
Calls: println
Code:
public class Dog extends Animal {
    @Override
    public void speak() {
        System.out.println("Woof!");
    }
}
File: animals Animals.java"#;
    assert_eq!(dog_text, expected_dog);
}

#[test]
fn test_function_with_imports() {
    let source = r#"import java.util.List;
import java.util.ArrayList;

public class ListUtils {
    public List<String> createList() {
        return new ArrayList<String>();
    }
}"#;
    let units = parse(source, Language::Java, "ListUtils.java");

    // Class is extracted as a single chunk with method inside
    let class_unit = get_unit_by_name(&units, "ListUtils").unwrap();
    let class_text = build_embedding_text(class_unit);
    let class_expected = r#"Class: ListUtils
Signature: public class ListUtils {
Calls: new
Uses: ArrayList
Code:
public class ListUtils {
    public List<String> createList() {
        return new ArrayList<String>();
    }
}
File: list utils ListUtils.java"#;
    assert_eq!(class_text, class_expected);

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "createList").is_none(),
        "Methods should not be extracted separately from classes"
    );
}
