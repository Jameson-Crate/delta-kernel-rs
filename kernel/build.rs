use std::path::PathBuf;
use std::{env, fs};

use rustc_version::{version_meta, Channel};

fn main() {
    println!("cargo::rustc-check-cfg=cfg(NIGHTLY_CHANNEL)");
    // note if we're on the nightly channel so we can enable doc_auto_cfg if so
    if let Channel::Nightly = version_meta().unwrap().channel {
        println!("cargo:rustc-cfg=NIGHTLY_CHANNEL");
    }

    generate_delta_error_conditions();

    // Generate prost bindings for the declarative-plans proto schema only when the feature is
    // enabled. Off-by-default consumers don't pay the protoc / codegen cost and don't pull in
    // prost-build.
    #[cfg(feature = "declarative-plans")]
    compile_proto_definitions();
}

struct ErrorClass {
    name: String,
    variant: String,
    message_template: String,
    sql_state: Option<String>,
    parameter_names: Vec<String>,
}

fn generate_delta_error_conditions() {
    let catalog_path = "error-catalog/delta-error-classes.json";
    println!("cargo:rerun-if-changed={catalog_path}");

    let catalog = fs::read_to_string(catalog_path).expect("read Delta error catalog");
    let catalog: serde_json::Value =
        serde_json::from_str(&catalog).expect("parse Delta error catalog");
    let entries = catalog.as_object().expect("Delta error catalog object");
    let mut classes: Vec<_> = entries
        .iter()
        .map(|(name, entry)| parse_error_class(name, entry))
        .collect();
    classes.sort_by(|left, right| left.name.cmp(&right.name));

    let generated = render_error_conditions(&classes);
    let output_path = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"))
        .join("delta_error_conditions.rs");
    fs::write(output_path, generated).expect("write generated Delta error conditions");
}

fn parse_error_class(name: &str, entry: &serde_json::Value) -> ErrorClass {
    let messages = entry["message"]
        .as_array()
        .expect("Delta error class message array");
    let message_template = messages
        .iter()
        .map(|message| message.as_str().expect("Delta error message string"))
        .collect::<Vec<_>>()
        .join("\n");
    ErrorClass {
        name: name.to_string(),
        variant: condition_variant(name),
        parameter_names: extract_parameter_names(&message_template),
        message_template,
        sql_state: entry["sqlState"].as_str().map(str::to_string),
    }
}

fn condition_variant(condition: &str) -> String {
    condition
        .split('_')
        .map(|word| {
            let mut characters = word.chars();
            match characters.next() {
                Some(first) => {
                    first.to_ascii_uppercase().to_string()
                        + &characters.as_str().to_ascii_lowercase()
                }
                None => String::new(),
            }
        })
        .collect()
}

fn extract_parameter_names(template: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut remaining = template;
    while let Some(open) = remaining.find('<') {
        let placeholder = &remaining[open + 1..];
        let Some(close) = placeholder.find('>') else {
            break;
        };
        let name = &placeholder[..close];
        if !names.iter().any(|existing| existing == name) {
            names.push(name.to_string());
        }
        remaining = &placeholder[close + 1..];
    }
    names
}

fn render_error_conditions(classes: &[ErrorClass]) -> String {
    let variants = classes
        .iter()
        .map(|class| {
            format!(
                "    /// Delta error condition `{}`.\n    {},\n",
                class.name, class.variant
            )
        })
        .collect::<String>();
    let names = render_match_arms(classes, |class| format!("{:?}", class.name));
    let sql_states = render_match_arms(classes, |class| match &class.sql_state {
        Some(sql_state) => format!("Some({sql_state:?})"),
        None => "None".to_string(),
    });
    let parameter_names = render_match_arms(classes, |class| {
        let names = class
            .parameter_names
            .iter()
            .map(|name| format!("{name:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("&[{names}]")
    });
    let templates = render_match_arms(classes, |class| format!("{:?}", class.message_template));

    format!(
        "// @generated from error-catalog/delta-error-classes.json.\n\
         \n\
         /// Stable, string-identified Delta error conditions.\n\
         ///\n\
         /// Enum layout and discriminants are unspecified. Persist or transmit [`Self::name`]\n\
         /// rather than casting a condition to an integer.\n\
         #[non_exhaustive]\n\
         #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]\n\
         pub enum DeltaErrorCondition {{\n\
         {variants}\
         }}\n\
         \n\
         impl DeltaErrorCondition {{\n\
             /// Returns the stable string identity of this condition.\n\
             pub const fn name(self) -> &'static str {{\n\
                 match self {{\n\
         {names}\
                 }}\n\
             }}\n\
         \n\
             /// Returns the SQLSTATE associated with this condition, when defined.\n\
             pub const fn sql_state(self) -> Option<&'static str> {{\n\
                 match self {{\n\
         {sql_states}\
                 }}\n\
             }}\n\
         \n\
             /// Returns the ordered, deduplicated names of this condition's message parameters.\n\
             pub const fn parameter_names(self) -> &'static [&'static str] {{\n\
                 match self {{\n\
         {parameter_names}\
                 }}\n\
             }}\n\
         \n\
             /// Returns the diagnostic message template for this condition.\n\
             pub const fn message_template(self) -> &'static str {{\n\
                 match self {{\n\
         {templates}\
                 }}\n\
             }}\n\
         }}\n"
    )
}

fn render_match_arms(classes: &[ErrorClass], value: impl Fn(&ErrorClass) -> String) -> String {
    classes
        .iter()
        .map(|class| format!("            Self::{} => {},\n", class.variant, value(class)))
        .collect()
}

#[cfg(feature = "declarative-plans")]
fn compile_proto_definitions() {
    let proto_dir = "proto";
    let proto_files = [
        "schema.proto",
        "expressions.proto",
        "plan.proto",
        "operation.proto",
    ];

    for file in &proto_files {
        println!("cargo:rerun-if-changed={proto_dir}/{file}");
    }

    let files: Vec<String> = proto_files
        .iter()
        .map(|f| format!("{proto_dir}/{f}"))
        .collect();

    #[cfg(feature = "vendored-protoc")]
    set_vendored_protoc();

    // Don't propagate `.proto` comments into the generated code as doc comments: they contain
    // angle-bracket generics (`Vec<...>`, `Option<...>`) that rustdoc would parse as unclosed
    // HTML tags. The `.proto` files stay the canonical reference for the wire format.
    prost_build::Config::new()
        .disable_comments(["."])
        .compile_protos(&files, &[proto_dir])
        .expect("failed to compile .proto files");
}

#[cfg(all(feature = "declarative-plans", feature = "vendored-protoc"))]
fn set_vendored_protoc() {
    // Leave a caller-supplied `protoc` alone, so a pinned toolchain wins over the vendored binary.
    if std::env::var_os("PROTOC").is_none() {
        let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc binary");
        std::env::set_var("PROTOC", protoc);
    }
}
