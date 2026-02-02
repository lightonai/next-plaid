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

    let method_unit = get_unit_by_name(&units, "Add").unwrap();
    let method_text = build_embedding_text(method_unit);
    assert_eq!(
        method_text,
        "Method: Add
Signature: public int Add(int a, int b)
Parameters: a, b
Code:
    public int Add(int a, int b)
    {
        return a + b;
    }
File: calculator Calculator.cs"
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

    let method_unit = get_unit_by_name(&units, "Add").unwrap();
    let method_text = build_embedding_text(method_unit);
    assert_eq!(
        method_text,
        "Method: Add
Signature: public int Add(int a, int b)
Parameters: a, b
Code:
    public int Add(int a, int b)
    {
        return a + b;
    }
File: math Math.cs"
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

    let constructor_unit = get_unit_by_name(&units, "Person")
        .into_iter()
        .chain(units.iter())
        .find(|u| u.name == "Person" && u.unit_type == crate::parser::UnitType::Method)
        .unwrap();
    let constructor_text = build_embedding_text(constructor_unit);
    assert_eq!(
        constructor_text,
        "Method: Person
Signature: public Person(string name, int age)
Parameters: name, age
Code:
    public Person(string name, int age)
    {
        Name = name;
        Age = age;
    }
File: person Person.cs"
    );

    let method_unit = get_unit_by_name(&units, "Greet").unwrap();
    let method_text = build_embedding_text(method_unit);
    assert_eq!(
        method_text,
        r#"Method: Greet
Signature: public string Greet()
Code:
    public string Greet()
    {
        return $"Hello, I'm {Name}";
    }
File: person Person.cs"#
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

    let class_unit = get_unit_by_name(&units, "DataService").unwrap();
    let class_text = build_embedding_text(class_unit);
    assert_eq!(
        class_text,
        "Class: DataService
Signature: public class DataService
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

    let method_unit = get_unit_by_name(&units, "FetchDataAsync").unwrap();
    let method_text = build_embedding_text(method_unit);
    assert_eq!(
        method_text,
        "Method: FetchDataAsync
Signature: public async Task<string> FetchDataAsync(string url)
Parameters: url
Calls: new
Variables: client
Code:
    public async Task<string> FetchDataAsync(string url)
    {
        using var client = new HttpClient();
        return await client.GetStringAsync(url);
    }
File: data service DataService.cs"
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

    let method_unit = get_unit_by_name(&units, "Format").unwrap();
    let method_text = build_embedding_text(method_unit);
    assert_eq!(
        method_text,
        "Method: Format
Signature: public static string Format(string template, params object[] args)
Parameters: template, args
Code:
    public static string Format(string template, params object[] args)
    {
        return string.Format(template, args);
    }
File: utils Utils.cs"
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

    let draw_unit = get_unit_by_name(&units, "Draw").unwrap();
    let draw_text = build_embedding_text(draw_unit);
    assert_eq!(
        draw_text,
        "Method: Draw
Signature: void Draw();
Code:
    void Draw();
File: idrawable IDrawable.cs"
    );

    let get_bounds_unit = get_unit_by_name(&units, "GetBounds").unwrap();
    let get_bounds_text = build_embedding_text(get_bounds_unit);
    assert_eq!(
        get_bounds_text,
        "Method: GetBounds
Signature: Rectangle GetBounds();
Code:
    Rectangle GetBounds();
File: idrawable IDrawable.cs"
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

    let class_unit = get_unit_by_name(&units, "Container").unwrap();
    let class_text = build_embedding_text(class_unit);
    assert_eq!(
        class_text,
        "Class: Container
Signature: public class Container<T>
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

    let get_value_unit = get_unit_by_name(&units, "GetValue").unwrap();
    let get_value_text = build_embedding_text(get_value_unit);
    assert_eq!(
        get_value_text,
        "Method: GetValue
Signature: public T GetValue()
Code:
    public T GetValue()
    {
        return _value;
    }
File: container Container.cs"
    );

    let set_value_unit = get_unit_by_name(&units, "SetValue").unwrap();
    let set_value_text = build_embedding_text(set_value_unit);
    assert_eq!(
        set_value_text,
        "Method: SetValue
Signature: public void SetValue(T value)
Parameters: value
Code:
    public void SetValue(T value)
    {
        _value = value;
    }
File: container Container.cs"
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

    let method_unit = get_unit_by_name(&units, "AddExclamation").unwrap();
    let method_text = build_embedding_text(method_unit);
    assert_eq!(
        method_text,
        r#"Method: AddExclamation
Signature: public static string AddExclamation(this string str)
Parameters: str
Code:
    public static string AddExclamation(this string str)
    {
        return str + "!";
    }
File: string extensions StringExtensions.cs"#
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

    let class_unit = get_unit_by_name(&units, "Circle").unwrap();
    let class_text = build_embedding_text(class_unit);
    assert_eq!(
        class_text,
        "Class: Circle
Signature: public class Circle
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
        "Code block: raw_code_1
Signature: public record Person(string Name, int Age);
Code:
public record Person(string Name, int Age);

public record Point
{
    public double X { get; init; }
    public double Y { get; init; }
}
File: records Records.cs"
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

    let method_unit = get_unit_by_name(&units, "FilterNames").unwrap();
    let method_text = build_embedding_text(method_unit);
    assert_eq!(
        method_text,
        r#"Method: FilterNames
Signature: public IEnumerable<string> FilterNames(IEnumerable<string> names)
Parameters: names
Code:
    public IEnumerable<string> FilterNames(IEnumerable<string> names)
    {
        return names.Where(n => n.StartsWith("A"))
                    .OrderBy(n => n)
                    .ToList();
    }
File: query service QueryService.cs"#
    );
}
