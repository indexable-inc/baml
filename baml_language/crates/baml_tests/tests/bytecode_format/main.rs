//! Tests for bytecode display formats (textual, expanded, expanded unoptimized).

use baml_tests::engine::{OptLevel, compile_source_with_opt};
use bex_vm::debug::{BytecodeFormat, HeadNames, display_program};
use bex_vm_types::{Function, Object};

/// The functions to dump, plus the pool's head names — a compiled `Program` has
/// no heap, so its heads cannot name themselves.
fn compile_display_functions(source: &str, opt: OptLevel) -> (Vec<(String, Function)>, HeadNames) {
    let program = compile_source_with_opt(source, opt);
    let heads = bex_vm::debug::HeadNames::of(&program);
    let mut functions: Vec<(String, Function)> = program
        .function_indices
        .iter()
        .filter(|(name, _)| !name.starts_with("baml."))
        .filter_map(|(name, idx)| match program.objects.get(*idx) {
            Some(Object::Function(f)) => Some((name.clone(), (**f).clone())),
            _ => None,
        })
        .collect();
    functions.sort_by(|(a, _), (b, _)| a.cmp(b));
    (functions, heads)
}

#[test]
fn bytecode_display_formats() {
    // Normalize CRLF → LF so line numbers are consistent across platforms.
    let source = include_str!("bytecode_display.baml").replace("\r\n", "\n");

    // 1. Textual format (optimized)
    let (o1, o1_heads) = compile_display_functions(&source, OptLevel::One);
    let o1_refs: Vec<(String, &Function)> = o1.iter().map(|(n, f)| (n.clone(), f)).collect();
    let textual = display_program(&o1_refs, BytecodeFormat::Textual, &o1_heads);

    // 2. Expanded format (optimized)
    let expanded = display_program(&o1_refs, BytecodeFormat::Expanded, &o1_heads);

    // 3. Expanded format (unoptimized)
    let (o0, o0_heads) = compile_display_functions(&source, OptLevel::Zero);
    let o0_refs: Vec<(String, &Function)> = o0.iter().map(|(n, f)| (n.clone(), f)).collect();
    let expanded_unopt = display_program(&o0_refs, BytecodeFormat::Expanded, &o0_heads);

    insta::with_settings!({omit_expression => true, snapshot_path => "snapshots"}, {
        insta::assert_snapshot!("bytecode_display_textual", textual);
        insta::assert_snapshot!("bytecode_display_expanded", expanded);
        insta::assert_snapshot!("bytecode_display_expanded_unoptimized", expanded_unopt);
    });
}
