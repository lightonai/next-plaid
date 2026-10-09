//! Tests for XML extraction (size-bounded element units named by tag path).

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

#[test]
fn test_small_file_is_one_unit() {
    let source = r#"<?xml version="1.0" encoding="utf-8"?>
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.1" />
  </ItemGroup>
</Project>
"#;
    let units = assert_extractor_invariants(source, Language::Xml, "App.csproj");
    assert_eq!(units.len(), 1);
    let expected = r#"Section: Project
Signature: <Project Sdk="Microsoft.NET.Sdk">
File: app App.csproj
Code:
<?xml version="1.0" encoding="utf-8"?>
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.1" />
  </ItemGroup>
</Project>"#;
    assert_eq!(build_embedding_text(&units[0]), expected);
}

/// A large root is split into its children, each named by its tag path and
/// identifying attribute; the ranges tile the file with no one-line stragglers.
#[test]
fn test_large_project_split_by_children() {
    let mut source =
        String::from("<Project>\n  <PropertyGroup>\n    <A>1</A>\n  </PropertyGroup>\n");
    for t in ["Build", "Test", "Pack"] {
        source.push_str(&format!("  <Target Name=\"{t}\">\n"));
        for i in 0..30 {
            source.push_str(&format!("    <Exec Command=\"step {i}\" />\n"));
        }
        source.push_str("  </Target>\n");
    }
    source.push_str("</Project>\n");
    let units = assert_extractor_invariants(&source, Language::Xml, "build.proj");
    let names: Vec<&str> = units.iter().map(|u| u.name.as_str()).collect();
    assert!(
        names.contains(&"Project > Target Name=\"Test\""),
        "{names:?}"
    );
    // A run mixing tags lists them, with the members' identities.
    let build = get_unit_by_name(&units, "Project > PropertyGroup, Target (Build)").unwrap();
    // The first run starts at the root's opening tag.
    assert_eq!(build.line, 1);
    assert!(build.code.contains("<PropertyGroup>"));
    let pack = get_unit_by_name(&units, "Project > Target Name=\"Pack\"").unwrap();
    // The root's closing tag joins the last run.
    assert!(pack.code.trim_end().ends_with("</Project>"));
    assert!(units.iter().all(|u| u.unit_type != UnitType::RawCode));
}

#[test]
fn test_android_manifest_names() {
    let mut source = String::from(
        "<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\">\n  <application android:label=\"@string/app_name\">\n",
    );
    for a in ["MainActivity", "SettingsActivity"] {
        source.push_str(&format!("    <activity android:name=\".{a}\">\n"));
        for i in 0..30 {
            source.push_str(&format!(
                "      <meta-data android:name=\"k{i}\" android:value=\"v\" />\n"
            ));
        }
        source.push_str("    </activity>\n");
    }
    source.push_str("  </application>\n</manifest>\n");
    let units = assert_extractor_invariants(&source, Language::Xml, "AndroidManifest.xml");
    assert!(
        get_unit_by_name(
            &units,
            "manifest > application > activity android:name=\".SettingsActivity\""
        )
        .is_some(),
        "{:?}",
        units.iter().map(|u| &u.name).collect::<Vec<_>>()
    );
}

#[test]
fn test_maven_dependency_named_by_artifact_id() {
    let mut source = String::from("<project>\n  <dependencies>\n");
    for a in ["spring-core", "junit", "guava"] {
        source.push_str(&format!(
            "    <dependency>\n      <groupId>org.x</groupId>\n      <artifactId>{a}</artifactId>\n      <version>1.0</version>\n    </dependency>\n"
        ));
    }
    source.push_str("  </dependencies>\n");
    for i in 0..50 {
        source.push_str(&format!("  <!-- filler {i} -->\n"));
    }
    source.push_str("</project>\n");
    let units = assert_extractor_invariants(&source, Language::Xml, "pom.xml");
    let deps = units
        .iter()
        .find(|u| u.name.contains("dependencies"))
        .unwrap();
    assert_eq!(deps.name, "project > dependencies");
    // A run of several dependencies names the first and last.
    let mut big = String::from("<project>\n  <dependencies>\n");
    for i in 0..30 {
        big.push_str(&format!(
            "    <dependency>\n      <artifactId>lib{i}</artifactId>\n    </dependency>\n"
        ));
    }
    big.push_str("  </dependencies>\n</project>\n");
    let units = assert_extractor_invariants(&big, Language::Xml, "pom.xml");
    assert!(
        units.iter().any(|u| u
            .name
            .starts_with("project > dependencies > dependency (lib0 … ")),
        "{:?}",
        units.iter().map(|u| &u.name).collect::<Vec<_>>()
    );
    for u in &units {
        assert!(u.end_line + 1 - u.line <= 50, "unit too long: {}", u.name);
    }
}

#[test]
fn test_xslt_templates_are_functions() {
    let mut source = String::from(
        "<xsl:stylesheet version=\"1.0\" xmlns:xsl=\"http://www.w3.org/1999/XSL/Transform\">\n",
    );
    source.push_str("  <xsl:template match=\"/\">\n    <html><xsl:apply-templates/></html>\n  </xsl:template>\n");
    source.push_str("  <xsl:template name=\"format-date\">\n    <xsl:param name=\"date\"/>\n");
    for _ in 0..30 {
        source.push_str("    <xsl:value-of select=\"$date\"/>\n");
    }
    source.push_str("  </xsl:template>\n</xsl:stylesheet>\n");
    let units = assert_extractor_invariants(&source, Language::Xml, "page.xsl");
    let fmt = units
        .iter()
        .find(|u| u.name.ends_with("xsl:template name=\"format-date\""))
        .unwrap();
    assert_eq!(fmt.unit_type, UnitType::Function);
    let root = units
        .iter()
        .find(|u| u.name.ends_with("xsl:template match=\"/\""))
        .unwrap();
    assert_eq!(root.unit_type, UnitType::Function);
}

/// A multi-megabyte generated file must stay fast and come out as many
/// bounded units, not one absurd unit.
#[test]
fn test_huge_generated_file() {
    let mut source = String::from("<?xml version=\"1.0\"?>\n<resources>\n");
    for i in 0..60_000 {
        source.push_str(&format!(
            "  <string name=\"key_{i}\">Value number {i}</string>\n"
        ));
    }
    source.push_str("</resources>\n");
    assert!(source.len() > 3_000_000);
    let t = std::time::Instant::now();
    let units = assert_extractor_invariants(&source, Language::Xml, "strings.xml");
    assert!(
        t.elapsed().as_secs_f64() < 20.0,
        "too slow: {:?}",
        t.elapsed()
    );
    assert!(units.len() > 1000);
    for u in &units {
        assert!(
            u.end_line + 1 - u.line <= 52,
            "unit too long: {} lines",
            u.end_line + 1 - u.line
        );
    }
    assert!(units[0].name.starts_with("resources > string (key_0 … "));
}

#[test]
fn test_minified_single_line() {
    let source = format!("<a>{}</a>", "<b x=\"1\"/>".repeat(1000));
    let units = assert_extractor_invariants(&source, Language::Xml, "min.xml");
    assert_eq!(units.len(), 1);
}

/// Malformed XML (or a `.props` file that is really a Java properties file)
/// has no root element: it is chunked by size instead of dropped.
#[test]
fn test_not_xml_falls_back_to_chunks() {
    let mut source = String::new();
    for i in 0..120 {
        source.push_str(&format!("key.{i}=value {i}\n"));
    }
    let units = assert_extractor_invariants(&source, Language::Xml, "en-us.props");
    assert!(units.len() >= 2);
    assert!(units.iter().all(|u| u.end_line + 1 - u.line <= 50));
}

#[test]
fn test_plist_and_xaml() {
    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key>
  <string>com.example.app</string>
</dict>
</plist>
"#;
    let units = assert_extractor_invariants(plist, Language::Xml, "Info.plist");
    assert_eq!(units[0].name, "plist");
    assert!(units[0].code.contains("CFBundleIdentifier"));

    let xaml = r#"<Window x:Class="App.MainWindow" xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml">
  <Grid x:Name="LayoutRoot">
    <Button x:Name="OkButton" Content="OK" Click="OnOk" />
  </Grid>
</Window>
"#;
    let units = assert_extractor_invariants(xaml, Language::Xml, "MainWindow.xaml");
    assert_eq!(units[0].name, "Window");
}

/// An XML fragment (several top-level elements, as in an XSLT module of
/// bare templates) splits into its top-level elements.
#[test]
fn test_fragment_with_several_top_level_elements() {
    let mut source = String::from("<!-- module -->\n");
    for t in ["first", "second", "third"] {
        source.push_str(&format!("<xsl:template name=\"{t}\">\n"));
        for _ in 0..30 {
            source.push_str("  <xsl:apply-templates/>\n");
        }
        source.push_str("</xsl:template>\n");
    }
    let units = assert_extractor_invariants(&source, Language::Xml, "module.xsl");
    for t in ["first", "second", "third"] {
        let u = get_unit_by_name(&units, &format!("xsl:template name=\"{t}\"")).unwrap();
        assert_eq!(u.unit_type, UnitType::Function);
        assert!(u.end_line + 1 - u.line <= 33);
    }
}

#[test]
fn test_empty_file() {
    assert!(parse("", Language::Xml, "empty.xml").is_empty());
    assert!(parse("  \n", Language::Xml, "empty.xml").is_empty());
}
