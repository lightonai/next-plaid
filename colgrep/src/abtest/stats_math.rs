//! Statistical helpers for the A/B report.
//!
//! Rank-based tests and bootstrap intervals only: session costs are
//! heavy-tailed, so means and t-tests would be dominated by outliers. The
//! normal approximation (with tie and continuity corrections) is accurate at
//! the minimum sample size enforced below, which keeps colgrep free of
//! external stats crates.
//!
//! Everything here is deterministic: the bootstrap is seeded from the data
//! itself, so re-running `colgrep ab` on unchanged samples prints identical
//! numbers. A report whose interval jitters between runs invites re-rolling
//! until it looks good.

/// Minimum per-group size for the Mann-Whitney normal approximation.
pub const MANN_WHITNEY_MIN_N: usize = 5;

/// Resamples drawn for a bootstrap interval.
const BOOTSTRAP_RESAMPLES: usize = 4000;

/// Abramowitz & Stegun 7.1.26 approximation of erf (max abs error 1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    sign * (1.0 - poly * (-x * x).exp())
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn two_sided_p(z: f64) -> f64 {
    (2.0 * (1.0 - normal_cdf(z.abs()))).clamp(0.0, 1.0)
}

/// Continuity correction: shrink the deviation from the mean by 0.5 without
/// crossing zero.
fn continuity_corrected(d: f64) -> f64 {
    if d == 0.0 {
        0.0
    } else {
        d - 0.5 * d.signum()
    }
}

/// Average ranks (1-based) with ties sharing their mean rank.
/// Returns the ranks aligned with `values` plus the tie term Σ(t³ − t).
fn ranks_with_ties(values: &[f64]) -> (Vec<f64>, f64) {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks = vec![0.0; n];
    let mut tie_term = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[idx[j + 1]] == values[idx[i]] {
            j += 1;
        }
        let avg_rank = (i + 1 + j + 1) as f64 / 2.0;
        let tied = (j - i + 1) as f64;
        if tied > 1.0 {
            tie_term += tied * tied * tied - tied;
        }
        for k in i..=j {
            ranks[idx[k]] = avg_rank;
        }
        i = j + 1;
    }
    (ranks, tie_term)
}

/// Two-sided Mann-Whitney U test for two independent samples.
///
/// Returns `None` when either group is below [`MANN_WHITNEY_MIN_N`] — saying
/// "not enough samples" beats printing a p-value the data cannot support.
pub fn mann_whitney_u_p(a: &[f64], b: &[f64]) -> Option<f64> {
    let n1 = a.len();
    let n2 = b.len();
    if n1 < MANN_WHITNEY_MIN_N || n2 < MANN_WHITNEY_MIN_N {
        return None;
    }

    let combined: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
    let (ranks, tie_term) = ranks_with_ties(&combined);
    let r1: f64 = ranks[..n1].iter().sum();

    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let nf = n1f + n2f;
    let u1 = r1 - n1f * (n1f + 1.0) / 2.0;
    let mean = n1f * n2f / 2.0;
    let var = n1f * n2f / 12.0 * ((nf + 1.0) - tie_term / (nf * (nf - 1.0)));
    if var <= 0.0 {
        return None; // every observation identical
    }
    let z = continuity_corrected(u1 - mean) / var.sqrt();
    Some(two_sided_p(z))
}

/// xorshift64* — a tiny deterministic PRNG for the bootstrap.
struct Rng(u64);

impl Rng {
    fn seeded(values: &[(f64, f64)]) -> Self {
        // Seed from the data so the interval is reproducible for a given
        // sample set but still varies between different data.
        let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
        for (a, b) in values {
            h ^= a.to_bits().rotate_left(17) ^ b.to_bits();
            h = h.wrapping_mul(0x2545_f491_4f6c_dd1d).rotate_left(29);
        }
        Self(h | 1)
    }

    fn next_below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 33) as usize % n.max(1)
    }
}

/// Percentile bootstrap interval for the ratio of two *ratio estimators*:
/// `(Σ numerator_a / Σ denominator_a) / (Σ numerator_b / Σ denominator_b)`.
///
/// Used for "cost per location found, treatment vs control": summing before
/// dividing keeps sessions that found nothing from producing an undefined
/// per-session ratio. Returns `None` when either arm is empty or a resampled
/// denominator collapses to zero too often to interpolate.
pub fn bootstrap_ratio_ci(
    a: &[(f64, f64)],
    b: &[(f64, f64)],
    confidence: f64,
) -> Option<(f64, f64)> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let ratio_of = |pairs: &[(f64, f64)], rng: &mut Rng| -> Option<f64> {
        let mut num = 0.0;
        let mut den = 0.0;
        for _ in 0..pairs.len() {
            let (n, d) = pairs[rng.next_below(pairs.len())];
            num += n;
            den += d;
        }
        (den > 0.0).then(|| num / den)
    };

    let mut seed_input: Vec<(f64, f64)> = a.to_vec();
    seed_input.extend_from_slice(b);
    let mut rng = Rng::seeded(&seed_input);

    let mut reps: Vec<f64> = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for _ in 0..BOOTSTRAP_RESAMPLES {
        if let (Some(ra), Some(rb)) = (ratio_of(a, &mut rng), ratio_of(b, &mut rng)) {
            if rb > 0.0 {
                reps.push(ra / rb);
            }
        }
    }
    if reps.len() < BOOTSTRAP_RESAMPLES / 2 {
        return None;
    }
    reps.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let alpha = (1.0 - confidence) / 2.0;
    let lo = reps[((alpha * reps.len() as f64) as usize).min(reps.len() - 1)];
    let hi = reps[(((1.0 - alpha) * reps.len() as f64) as usize).min(reps.len() - 1)];
    Some((lo, hi))
}

/// Median of a sample (0.0 for an empty slice).
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Format an integer with thousands separators: 1234567 → "1,234,567".
pub fn fmt_thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A percentage change rendered the way a person would say it: "18% cheaper"
/// / "7% more expensive" / "about the same".
pub fn fmt_change(ratio: f64) -> String {
    if !ratio.is_finite() || ratio <= 0.0 {
        return "n/a".to_string();
    }
    let pct = (ratio - 1.0) * 100.0;
    if pct.abs() < 2.0 {
        "about the same".to_string()
    } else if pct < 0.0 {
        format!("{:.0}% cheaper", -pct)
    } else {
        format!("{:.0}% more expensive", pct)
    }
}

/// Compact column annotation: "59% less" / "36% more" / "same".
pub fn fmt_delta_short(ratio: f64) -> String {
    if !ratio.is_finite() || ratio <= 0.0 {
        return "—".to_string();
    }
    let pct = (ratio - 1.0) * 100.0;
    if pct.abs() < 2.0 {
        "same".to_string()
    } else if pct < 0.0 {
        format!("{:.0}% less", -pct)
    } else {
        format!("{:.0}% more", pct)
    }
}

/// A confidence interval said out loud: "between 9% and 91% less",
/// "between 2% and 122% more", or "from 63% less to 34% more" when the
/// interval straddles no-change. `lo`/`hi` are treatment÷control ratios.
pub fn fmt_range(lo: f64, hi: f64) -> String {
    if !lo.is_finite() || !hi.is_finite() {
        return "unknown".to_string();
    }
    let pct = |r: f64| ((r - 1.0) * 100.0).abs().round() as i64;
    // Both ends point the same way: name the direction once, at the end.
    // An end that rounds to 0% becomes "no change" — "between 0% and 122%
    // more" reads like a typo.
    let same_direction = |near: f64, far: f64, word: &str| {
        if pct(near) == 0 {
            format!("between no change and {}% {}", pct(far), word)
        } else {
            format!("between {}% and {}% {}", pct(near), pct(far), word)
        }
    };
    if hi < 1.0 {
        // Both ends are savings; the smaller saving reads first.
        same_direction(hi, lo, "less")
    } else if lo > 1.0 {
        same_direction(lo, hi, "more")
    } else if pct(lo) == 0 {
        // Interval touches parity at the low end.
        format!("up to {}% more", pct(hi))
    } else if pct(hi) == 0 {
        format!("up to {}% less", pct(lo))
    } else {
        format!("from {}% less to {}% more", pct(lo), pct(hi))
    }
}

/// "2.4x more" / "3x fewer" / "about the same" for a plain count comparison.
pub fn fmt_multiple(a: f64, b: f64) -> String {
    if a <= 0.0 || b <= 0.0 {
        return "—".to_string();
    }
    let (bigger, smaller, word) = if a >= b {
        (a, b, "more")
    } else {
        (b, a, "fewer")
    };
    let x = bigger / smaller;
    if x < 1.15 {
        "about the same".to_string()
    } else {
        format!("{:.1}x {}", x, word)
    }
}

/// Human p-value: "<0.0001" below display precision.
pub fn fmt_p(p: f64) -> String {
    if p < 0.0001 {
        "<0.0001".to_string()
    } else {
        format!("{:.4}", p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_cdf_matches_known_quantiles() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((normal_cdf(1.959964) - 0.975).abs() < 1e-4);
        assert!((normal_cdf(-1.959964) - 0.025).abs() < 1e-4);
    }

    #[test]
    fn mann_whitney_detects_separated_groups() {
        let a: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let b: Vec<f64> = (11..=20).map(|i| i as f64).collect();
        let p = mann_whitney_u_p(&a, &b).unwrap();
        assert!(p < 0.05, "expected significance, got p={p}");
    }

    #[test]
    fn mann_whitney_interleaved_groups_are_not_significant() {
        let a = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
        let b = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let p = mann_whitney_u_p(&a, &b).unwrap();
        assert!(p > 0.5, "interleaved groups should not differ, p={p}");
    }

    #[test]
    fn mann_whitney_requires_min_per_group() {
        assert!(mann_whitney_u_p(&[1.0; 4], &[2.0; 10]).is_none());
        assert!(mann_whitney_u_p(&[1.0; 10], &[2.0; 4]).is_none());
    }

    #[test]
    fn mann_whitney_all_identical_returns_none() {
        assert!(mann_whitney_u_p(&[3.0; 6], &[3.0; 6]).is_none());
    }

    #[test]
    fn median_odd_even_and_empty() {
        assert_eq!(median(&[]), 0.0);
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
    }

    #[test]
    fn bootstrap_ci_brackets_a_clear_ratio() {
        // Treatment: 100 tokens per location. Control: 200 per location.
        let a: Vec<(f64, f64)> = (0..30).map(|i| (100.0 + i as f64, 1.0)).collect();
        let b: Vec<(f64, f64)> = (0..30).map(|i| (200.0 + i as f64, 1.0)).collect();
        let (lo, hi) = bootstrap_ratio_ci(&a, &b, 0.95).unwrap();
        assert!(lo < 0.6 && hi > 0.5, "expected ~0.55 inside [{lo}, {hi}]");
        assert!(hi < 1.0, "a clear win should exclude 1.0, got hi={hi}");
    }

    #[test]
    fn bootstrap_ci_is_deterministic_for_the_same_data() {
        let a: Vec<(f64, f64)> = (0..12).map(|i| (100.0 + i as f64, 2.0)).collect();
        let b: Vec<(f64, f64)> = (0..12).map(|i| (130.0 + i as f64, 2.0)).collect();
        let first = bootstrap_ratio_ci(&a, &b, 0.95).unwrap();
        let second = bootstrap_ratio_ci(&a, &b, 0.95).unwrap();
        assert_eq!(first, second, "report numbers must not jitter between runs");
    }

    #[test]
    fn bootstrap_ci_handles_zero_denominators_and_empty_input() {
        // Every session found nothing: no ratio is definable.
        let a = vec![(100.0, 0.0); 10];
        let b = vec![(100.0, 0.0); 10];
        assert!(bootstrap_ratio_ci(&a, &b, 0.95).is_none());
        assert!(bootstrap_ratio_ci(&[], &[(1.0, 1.0)], 0.95).is_none());
    }

    #[test]
    fn short_delta_and_multiple_formatting() {
        assert_eq!(fmt_delta_short(0.41), "59% less");
        assert_eq!(fmt_delta_short(1.36), "36% more");
        assert_eq!(fmt_delta_short(1.01), "same");
        assert_eq!(fmt_delta_short(0.0), "—");

        assert_eq!(fmt_multiple(62.0, 26.0), "2.4x more");
        assert_eq!(fmt_multiple(26.0, 62.0), "2.4x fewer");
        assert_eq!(fmt_multiple(10.0, 10.0), "about the same");
        assert_eq!(fmt_multiple(5.0, 0.0), "—");
    }

    #[test]
    fn range_formatting_reads_as_a_sentence() {
        // Both ends savings: smaller saving first.
        assert_eq!(fmt_range(0.09, 0.91), "between 9% and 91% less");
        // Both ends increases.
        assert_eq!(fmt_range(1.02, 2.22), "between 2% and 122% more");
        // Straddling no-change.
        assert_eq!(fmt_range(0.37, 1.34), "from 63% less to 34% more");
        // An interval touching parity says so instead of printing "0%".
        assert_eq!(fmt_range(1.0, 2.22), "up to 122% more");
        assert_eq!(fmt_range(0.55, 1.0), "up to 45% less");
        assert_eq!(fmt_range(1.004, 2.22), "between no change and 122% more");
        assert_eq!(fmt_range(f64::NAN, 1.0), "unknown");
    }

    #[test]
    fn change_formatting_reads_like_speech() {
        assert_eq!(fmt_change(0.82), "18% cheaper");
        assert_eq!(fmt_change(1.33), "33% more expensive");
        assert_eq!(fmt_change(1.005), "about the same");
        assert_eq!(fmt_change(f64::NAN), "n/a");
    }

    #[test]
    fn thousands_formatting() {
        assert_eq!(fmt_thousands(0), "0");
        assert_eq!(fmt_thousands(999), "999");
        assert_eq!(fmt_thousands(1000), "1,000");
        assert_eq!(fmt_thousands(1234567), "1,234,567");
    }

    #[test]
    fn p_value_formatting() {
        assert_eq!(fmt_p(0.5), "0.5000");
        assert_eq!(fmt_p(0.00005), "<0.0001");
    }
}
