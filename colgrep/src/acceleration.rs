use anyhow::{bail, Result};

const FORCE_CPU_ENV_VARS: &[&str] = &["FORCE_CPU", "COLGREP_FORCE_CPU", "NEXT_PLAID_FORCE_CPU"];
const FORCE_GPU_ENV_VARS: &[&str] = &["FORCE_GPU", "COLGREP_FORCE_GPU", "NEXT_PLAID_FORCE_GPU"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccelerationMode {
    #[default]
    Auto,
    ForceCpu,
    ForceGpu,
}

fn env_var_is_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn env_acceleration_mode() -> Result<AccelerationMode> {
    let force_cpu = FORCE_CPU_ENV_VARS
        .iter()
        .any(|name| env_var_is_truthy(name));
    let force_gpu = FORCE_GPU_ENV_VARS
        .iter()
        .any(|name| env_var_is_truthy(name));

    if force_cpu && force_gpu {
        bail!("FORCE_CPU and FORCE_GPU are both set; choose only one");
    }

    Ok(match (force_cpu, force_gpu) {
        (true, false) => AccelerationMode::ForceCpu,
        (false, true) => AccelerationMode::ForceGpu,
        _ => AccelerationMode::Auto,
    })
}

pub fn env_acceleration_mode_lossy() -> AccelerationMode {
    env_acceleration_mode().unwrap_or(AccelerationMode::Auto)
}

pub fn apply_acceleration_mode(mode: AccelerationMode) {
    match mode {
        AccelerationMode::Auto => {
            for name in FORCE_CPU_ENV_VARS.iter().chain(FORCE_GPU_ENV_VARS.iter()) {
                std::env::remove_var(name);
            }
        }
        AccelerationMode::ForceCpu => {
            for name in FORCE_GPU_ENV_VARS {
                std::env::remove_var(name);
            }
            for name in FORCE_CPU_ENV_VARS {
                std::env::set_var(name, "1");
            }
        }
        AccelerationMode::ForceGpu => {
            for name in FORCE_CPU_ENV_VARS {
                std::env::remove_var(name);
            }
            for name in FORCE_GPU_ENV_VARS {
                std::env::set_var(name, "1");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn force_gpu_wins_over_none() {
        let _guard = env_lock().lock().unwrap();
        for name in FORCE_CPU_ENV_VARS.iter().chain(FORCE_GPU_ENV_VARS.iter()) {
            std::env::remove_var(name);
        }

        std::env::set_var("FORCE_GPU", "1");
        assert_eq!(env_acceleration_mode().unwrap(), AccelerationMode::ForceGpu);
        std::env::remove_var("FORCE_GPU");
    }

    #[test]
    fn conflicting_env_is_rejected() {
        let _guard = env_lock().lock().unwrap();
        for name in FORCE_CPU_ENV_VARS.iter().chain(FORCE_GPU_ENV_VARS.iter()) {
            std::env::remove_var(name);
        }

        std::env::set_var("FORCE_CPU", "1");
        std::env::set_var("FORCE_GPU", "1");
        assert!(env_acceleration_mode().is_err());
        std::env::remove_var("FORCE_CPU");
        std::env::remove_var("FORCE_GPU");
    }
}
