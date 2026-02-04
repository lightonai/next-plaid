//! Tests for C# code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_method() {
    let source = "public class Calculator
{
    public int Add(int a, int b)
    {
        return a + b;
    }
}";
    let units = parse(source, Language::CSharp, "Calculator.cs");

    // Class is extracted as a single chunk with method inside
    let class_unit = get_unit_by_name(&units, "Calculator").unwrap();
    let class_text = build_embedding_text(class_unit);
    assert_eq!(
        class_text,
        "Class: Calculator
Signature: public class Calculator
Code:
public class Calculator
{
    public int Add(int a, int b)
    {
        return a + b;
    }
}
File: calculator Calculator.cs"
    );

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "Add").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_method_with_xml_doc() {
    let source = r#"public class Math
{
    /// <summary>
    /// Calculates the sum of two numbers.
    /// </summary>
    /// <param name="a">First number</param>
    /// <param name="b">Second number</param>
    /// <returns>Sum of a and b</returns>
    public int Add(int a, int b)
    {
        return a + b;
    }
}"#;
    let units = parse(source, Language::CSharp, "Math.cs");

    // Class is extracted as a single chunk with XML doc and method inside
    let class_unit = get_unit_by_name(&units, "Math").unwrap();
    let class_text = build_embedding_text(class_unit);
    assert_eq!(
        class_text,
        r#"Class: Math
Signature: public class Math
Code:
public class Math
{
    /// <summary>
    /// Calculates the sum of two numbers.
    /// </summary>
    /// <param name="a">First number</param>
    /// <param name="b">Second number</param>
    /// <returns>Sum of a and b</returns>
    public int Add(int a, int b)
    {
        return a + b;
    }
}
File: math Math.cs"#
    );

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "Add").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_class_definition() {
    let source = r#"public class Person
{
    public string Name { get; set; }
    public int Age { get; set; }

    public Person(string name, int age)
    {
        Name = name;
        Age = age;
    }

    public string Greet()
    {
        return $"Hello, I'm {Name}";
    }
}"#;
    let units = parse(source, Language::CSharp, "Person.cs");

    // Class is extracted as a single chunk with all members inside
    let class_unit = get_unit_by_name(&units, "Person").unwrap();
    let class_text = build_embedding_text(class_unit);
    assert_eq!(
        class_text,
        r#"Class: Person
Signature: public class Person
Code:
public class Person
{
    public string Name { get; set; }
    public int Age { get; set; }

    public Person(string name, int age)
    {
        Name = name;
        Age = age;
    }

    public string Greet()
    {
        return $"Hello, I'm {Name}";
    }
}
File: person Person.cs"#
    );

    // Verify NO separate method/constructor units exist
    assert!(
        get_unit_by_name(&units, "Greet").is_none(),
        "Methods should not be extracted separately from classes"
    );
    // Constructor has same name as class, so we check there's only 1 unit with name "Person"
    let person_units: Vec<_> = units.iter().filter(|u| u.name == "Person").collect();
    assert_eq!(
        person_units.len(),
        1,
        "Should only have 1 Person unit (the class), not separate constructor"
    );
}

#[test]
fn test_async_method() {
    let source = "public class DataService
{
    public async Task<string> FetchDataAsync(string url)
    {
        using var client = new HttpClient();
        return await client.GetStringAsync(url);
    }
}";
    let units = parse(source, Language::CSharp, "DataService.cs");

    // Class is extracted as a single chunk with async method inside
    let class_unit = get_unit_by_name(&units, "DataService").unwrap();
    let class_text = build_embedding_text(class_unit);
    assert_eq!(
        class_text,
        "Class: DataService
Signature: public class DataService
Calls: new
Variables: client
Code:
public class DataService
{
    public async Task<string> FetchDataAsync(string url)
    {
        using var client = new HttpClient();
        return await client.GetStringAsync(url);
    }
}
File: data service DataService.cs"
    );

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "FetchDataAsync").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_static_method() {
    let source = "public static class Utils
{
    public static string Format(string template, params object[] args)
    {
        return string.Format(template, args);
    }
}";
    let units = parse(source, Language::CSharp, "Utils.cs");

    // Static class is extracted as a single chunk with static method inside
    let class_unit = get_unit_by_name(&units, "Utils").unwrap();
    let class_text = build_embedding_text(class_unit);
    assert_eq!(
        class_text,
        "Class: Utils
Signature: public static class Utils
Code:
public static class Utils
{
    public static string Format(string template, params object[] args)
    {
        return string.Format(template, args);
    }
}
File: utils Utils.cs"
    );

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "Format").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_interface() {
    let source = "public interface IDrawable
{
    void Draw();
    Rectangle GetBounds();
}";
    let units = parse(source, Language::CSharp, "IDrawable.cs");

    // Interface is extracted as a single chunk with all method signatures inside
    let interface_unit = get_unit_by_name(&units, "IDrawable").unwrap();
    let interface_text = build_embedding_text(interface_unit);
    assert_eq!(
        interface_text,
        "Class: IDrawable
Signature: public interface IDrawable
Code:
public interface IDrawable
{
    void Draw();
    Rectangle GetBounds();
}
File: idrawable IDrawable.cs"
    );

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "Draw").is_none(),
        "Interface methods should not be extracted separately"
    );
    assert!(
        get_unit_by_name(&units, "GetBounds").is_none(),
        "Interface methods should not be extracted separately"
    );
}

#[test]
fn test_generic_method() {
    let source = "public class Container<T>
{
    private T _value;

    public T GetValue()
    {
        return _value;
    }

    public void SetValue(T value)
    {
        _value = value;
    }
}";
    let units = parse(source, Language::CSharp, "Container.cs");

    // Generic class is extracted as a single chunk with all methods inside
    let class_unit = get_unit_by_name(&units, "Container").unwrap();
    let class_text = build_embedding_text(class_unit);
    assert_eq!(
        class_text,
        "Class: Container
Signature: public class Container<T>
Variables: _value
Code:
public class Container<T>
{
    private T _value;

    public T GetValue()
    {
        return _value;
    }

    public void SetValue(T value)
    {
        _value = value;
    }
}
File: container Container.cs"
    );

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "GetValue").is_none(),
        "Methods should not be extracted separately from classes"
    );
    assert!(
        get_unit_by_name(&units, "SetValue").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_extension_method() {
    let source = r#"public static class StringExtensions
{
    public static string AddExclamation(this string str)
    {
        return str + "!";
    }
}"#;
    let units = parse(source, Language::CSharp, "StringExtensions.cs");

    // Extension class is extracted as a single chunk with extension method inside
    let class_unit = get_unit_by_name(&units, "StringExtensions").unwrap();
    let class_text = build_embedding_text(class_unit);
    assert_eq!(
        class_text,
        r#"Class: StringExtensions
Signature: public static class StringExtensions
Code:
public static class StringExtensions
{
    public static string AddExclamation(this string str)
    {
        return str + "!";
    }
}
File: string extensions StringExtensions.cs"#
    );

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "AddExclamation").is_none(),
        "Extension methods should not be extracted separately from classes"
    );
}

#[test]
fn test_property_accessor() {
    let source = "public class Circle
{
    private double _radius;

    public double Radius
    {
        get => _radius;
        set => _radius = value > 0 ? value : 0;
    }

    public double Area => Math.PI * _radius * _radius;
}";
    let units = parse(source, Language::CSharp, "Circle.cs");

    // Class with properties is extracted as a single chunk
    let class_unit = get_unit_by_name(&units, "Circle").unwrap();
    let class_text = build_embedding_text(class_unit);
    assert_eq!(
        class_text,
        "Class: Circle
Signature: public class Circle
Variables: _radius
Code:
public class Circle
{
    private double _radius;

    public double Radius
    {
        get => _radius;
        set => _radius = value > 0 ? value : 0;
    }

    public double Area => Math.PI * _radius * _radius;
}
File: circle Circle.cs"
    );
}

#[test]
fn test_record() {
    let source = "public record Person(string Name, int Age);

public record Point
{
    public double X { get; init; }
    public double Y { get; init; }
}";
    let units = parse(source, Language::CSharp, "Records.cs");

    // Records are extracted as RawCode blocks
    assert_eq!(units.len(), 1);
    let unit = &units[0];
    let text = build_embedding_text(unit);
    assert_eq!(
        text,
        "public record Person(string Name, int Age);

public record Point
{
    public double X { get; init; }
    public double Y { get; init; }
}"
    );
}

#[test]
fn test_linq_expression() {
    let source = r#"public class QueryService
{
    public IEnumerable<string> FilterNames(IEnumerable<string> names)
    {
        return names.Where(n => n.StartsWith("A"))
                    .OrderBy(n => n)
                    .ToList();
    }
}"#;
    let units = parse(source, Language::CSharp, "QueryService.cs");

    // Class with LINQ method is extracted as a single chunk
    let class_unit = get_unit_by_name(&units, "QueryService").unwrap();
    let class_text = build_embedding_text(class_unit);
    assert_eq!(
        class_text,
        r#"Class: QueryService
Signature: public class QueryService
Code:
public class QueryService
{
    public IEnumerable<string> FilterNames(IEnumerable<string> names)
    {
        return names.Where(n => n.StartsWith("A"))
                    .OrderBy(n => n)
                    .ToList();
    }
}
File: query service QueryService.cs"#
    );

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "FilterNames").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_class_inheritance() {
    let source = r#"public class Animal
{
    public virtual void Speak()
    {
        Console.WriteLine("...");
    }
}

public class Dog : Animal
{
    public override void Speak()
    {
        Console.WriteLine("Woof!");
    }
}"#;
    let units = parse(source, Language::CSharp, "Animals.cs");

    // Animal class is extracted as a single chunk
    let animal = get_unit_by_name(&units, "Animal").unwrap();
    let animal_text = build_embedding_text(animal);
    assert_eq!(
        animal_text,
        r#"Class: Animal
Signature: public class Animal
Code:
public class Animal
{
    public virtual void Speak()
    {
        Console.WriteLine("...");
    }
}
File: animals Animals.cs"#
    );
    // Animal has no parent
    assert!(!animal_text.contains("Extends:"));

    // Dog class is extracted as a single chunk with inheritance info
    let dog = get_unit_by_name(&units, "Dog").unwrap();
    let dog_text = build_embedding_text(dog);
    assert_eq!(
        dog_text,
        r#"Class: Dog
Signature: public class Dog : Animal
Extends: Animal
Code:
public class Dog : Animal
{
    public override void Speak()
    {
        Console.WriteLine("Woof!");
    }
}
File: animals Animals.cs"#
    );

    // Verify NO separate method units exist
    assert!(
        get_unit_by_name(&units, "Speak").is_none(),
        "Methods should not be extracted separately from classes"
    );
}

#[test]
fn test_function_with_imports() {
    // Note: C# uses namespace imports (using System.Collections.Generic;), not class imports.
    // This means the Uses field won't be populated for types from imported namespaces
    // since the import extracts "Generic" (the namespace) but the code uses "List" (the class).
    // However, object creation is still tracked - the class name is extracted from `new ClassName()`.
    let source = r#"using MyApp.Models;

public class Factory
{
    public User CreateUser()
    {
        return new User();
    }
}"#;
    let units = parse(source, Language::CSharp, "Factory.cs");

    // Class is extracted as a single chunk
    let class_unit = get_unit_by_name(&units, "Factory").unwrap();
    let class_text = build_embedding_text(class_unit);
    assert_eq!(
        class_text,
        r#"Class: Factory
Signature: public class Factory
Calls: new
Code:
public class Factory
{
    public User CreateUser()
    {
        return new User();
    }
}
File: factory Factory.cs"#
    );

    // Verify NO separate method unit exists
    assert!(
        get_unit_by_name(&units, "CreateUser").is_none(),
        "Methods should not be extracted separately from classes"
    );
}
