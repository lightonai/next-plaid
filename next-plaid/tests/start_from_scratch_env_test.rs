use next_plaid::{IndexConfig, UpdateConfig};

#[test]
fn defaults_use_index_default_start_from_scratch_env() {
    unsafe {
        std::env::set_var("INDEX_DEFAULT_START_FROM_SCRATCH", "159");
    }
    assert_eq!(
        std::env::var("INDEX_DEFAULT_START_FROM_SCRATCH")
            .ok()
            .as_deref(),
        Some("159")
    );
    assert_eq!(next_plaid::default_start_from_scratch(), 159);
    assert_eq!(IndexConfig::default().start_from_scratch, 159);
    assert_eq!(UpdateConfig::default().start_from_scratch, 159);
}
