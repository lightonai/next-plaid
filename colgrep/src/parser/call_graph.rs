//! Call graph construction for code units.

use super::types::CodeUnit;
use rayon::prelude::*;
use std::collections::HashMap;

/// Build call graph and populate called_by for all units.
pub fn build_call_graph(units: &mut [CodeUnit]) {
    // Build index: function_name -> indices of units with that name
    let mut name_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, unit) in units.iter().enumerate() {
        name_to_indices
            .entry(unit.name.clone())
            .or_default()
            .push(i);
    }

    // Collect all calls first to avoid borrow issues
    let calls_map: Vec<(usize, Vec<String>)> = units
        .iter()
        .enumerate()
        .map(|(i, u)| (i, u.calls.clone()))
        .collect();

    let caller_names: Vec<String> = units.iter().map(|unit| unit.name.clone()).collect();

    let name_to_indices_ref = &name_to_indices;
    let edges: Vec<(usize, String)> = calls_map
        .par_iter()
        .map(|(caller_idx, calls)| {
            let caller_name = caller_names[*caller_idx].clone();
            let mut local_edges = Vec::new();
            for callee_name in calls {
                if let Some(indices) = name_to_indices_ref.get(callee_name) {
                    local_edges.extend(
                        indices
                            .iter()
                            .copied()
                            .map(|callee_idx| (callee_idx, caller_name.clone())),
                    );
                }
            }
            local_edges
        })
        .flatten()
        .collect();

    let mut called_by_map: HashMap<usize, Vec<String>> = HashMap::new();
    for (callee_idx, caller_name) in edges {
        called_by_map
            .entry(callee_idx)
            .or_default()
            .push(caller_name);
    }

    for (callee_idx, mut callers) in called_by_map {
        callers.sort_unstable();
        callers.dedup();
        units[callee_idx].called_by = callers;
    }
}
