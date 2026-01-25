//! Call graph construction for code units.

use super::types::CodeUnit;
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

    // For each unit, find what calls it
    for (caller_idx, calls) in calls_map {
        let caller_name = units[caller_idx].name.clone();
        for callee_name in calls {
            if let Some(indices) = name_to_indices.get(&callee_name) {
                for &callee_idx in indices {
                    if !units[callee_idx].called_by.contains(&caller_name) {
                        units[callee_idx].called_by.push(caller_name.clone());
                    }
                }
            }
        }
    }
}
