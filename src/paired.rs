use crate::{Tier, classify, route_input, route_score};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
};

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum DataKind {
    Fixture,
    Measured,
}

#[derive(Deserialize, Serialize)]
struct ModelRates {
    model: String,
    input_usd_per_million_tokens: f64,
    output_usd_per_million_tokens: f64,
}

impl ModelRates {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.model.trim().is_empty()
            || [self.input_usd_per_million_tokens, self.output_usd_per_million_tokens]
                .into_iter()
                .any(|rate| !rate.is_finite() || rate < 0.0)
        {
            return Err("Model rates need a model ID and finite, nonnegative token prices".into());
        }
        Ok(())
    }

    fn cost(&self, run: &ModelRun) -> Result<f64, String> {
        if run.model != self.model {
            return Err(format!("expected model {}, got {}", self.model, run.model));
        }
        if !run.quality.is_finite() || !(0.0..=1.0).contains(&run.quality) {
            return Err("quality must be in [0, 1]".into());
        }
        if run.input_tokens == 0 {
            return Err("input_tokens must be positive".into());
        }
        let cost = run.input_tokens as f64 / 1_000_000.0 * self.input_usd_per_million_tokens
            + run.output_tokens as f64 / 1_000_000.0 * self.output_usd_per_million_tokens;
        if !cost.is_finite() {
            return Err("token charges overflowed".into());
        }
        Ok(cost)
    }
}

#[derive(Deserialize, Serialize)]
struct RateCard {
    data_kind: DataKind,
    small: ModelRates,
    large: ModelRates,
}

#[derive(Deserialize)]
struct ModelRun {
    model: String,
    input_tokens: u32,
    output_tokens: u32,
    quality: f64,
}

#[derive(Deserialize)]
struct Record {
    id: String,
    request: Value,
    small: ModelRun,
    large: ModelRun,
}

#[derive(Serialize)]
struct RegressedRequest {
    id: String,
    score: f64,
    quality_regret: f64,
    cost_saved_usd: f64,
}

#[derive(Default, Serialize)]
struct Usage {
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: f64,
    #[serde(skip)]
    quality: f64,
}

impl Usage {
    fn add(&mut self, run: &ModelRun, cost: f64) {
        self.input_tokens += u64::from(run.input_tokens);
        self.output_tokens += u64::from(run.output_tokens);
        self.cost_usd += cost;
        self.quality += run.quality;
    }
}

#[derive(Default)]
struct Policy {
    threshold: f64,
    small_requests: usize,
    degraded_requests: usize,
    improved_requests: usize,
    usage: Usage,
    regret: f64,
    worst: Vec<RegressedRequest>,
}

impl Policy {
    fn observe(
        &mut self,
        record: &Record,
        score: f64,
        force_large: bool,
        small_cost: f64,
        large_cost: f64,
    ) {
        if route_score(score, self.threshold, force_large) == Tier::Large {
            self.usage.add(&record.large, large_cost);
            return;
        }
        self.usage.add(&record.small, small_cost);
        self.small_requests += 1;
        let loss = record.large.quality - record.small.quality;
        self.improved_requests += usize::from(loss < 0.0);
        if loss > 0.0 {
            self.degraded_requests += 1;
            self.regret += loss;
            self.worst.push(RegressedRequest {
                id: record.id.clone(),
                score,
                quality_regret: loss,
                cost_saved_usd: large_cost - small_cost,
            });
            self.worst.sort_by(|a, b| {
                b.quality_regret
                    .total_cmp(&a.quality_regret)
                    .then_with(|| a.id.cmp(&b.id))
            });
            self.worst.truncate(10);
        }
    }
}

pub(crate) fn evaluate(
    records_path: &str,
    rates_path: &str,
    thresholds: &[&str],
    default_threshold: f64,
) -> Result<(), Box<dyn Error>> {
    if thresholds.len() > 64 {
        return Err("Supply at most 64 thresholds".into());
    }
    let mut thresholds = if thresholds.is_empty() {
        vec![default_threshold]
    } else {
        thresholds
            .iter()
            .map(|value| value.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?
    };
    if thresholds.iter().any(|&value| !value.is_finite() || value <= 0.0) {
        return Err("Thresholds must be finite and positive".into());
    }
    thresholds.sort_by(f64::total_cmp);
    thresholds.dedup();

    let rates: RateCard = serde_json::from_reader(BufReader::new(File::open(rates_path)?))?;
    rates.small.validate()?;
    rates.large.validate()?;
    let mut policies: Vec<_> = thresholds
        .into_iter()
        .map(|threshold| Policy { threshold, ..Policy::default() })
        .collect();
    let mut baseline = Usage::default();
    let mut forced_large_requests = 0;
    let mut seen = HashSet::new();
    for (index, line) in BufReader::new(File::open(records_path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let at_line = |error: &dyn std::fmt::Display| format!("{records_path}:{}: {error}", index + 1);
        let record: Record = serde_json::from_str(&line).map_err(|error| at_line(&error))?;
        if record.id.trim().is_empty() || record.id.len() > 128 || !seen.insert(record.id.clone()) {
            return Err(at_line(&"IDs must be unique and contain 1-128 bytes of nonblank text").into());
        }
        let (context, force_large) = route_input(&record.request).map_err(|error| at_line(&error))?;
        let small_cost = rates.small.cost(&record.small).map_err(|error| at_line(&error))?;
        let large_cost = rates.large.cost(&record.large).map_err(|error| at_line(&error))?;
        let score = classify(&context, default_threshold, force_large).score;
        forced_large_requests += usize::from(force_large);
        baseline.add(&record.large, large_cost);
        for policy in &mut policies {
            policy.observe(&record, score, force_large, small_cost, large_cost);
        }
    }
    let samples = seen.len();
    if samples == 0 {
        return Err("No paired records".into());
    }
    if !baseline.cost_usd.is_finite() || policies.iter().any(|policy| !policy.usage.cost_usd.is_finite()) {
        return Err("Aggregate token charges overflowed".into());
    }

    let samples_f64 = samples as f64;
    let reports: Vec<_> = policies.iter().map(|policy| {
        let saved = baseline.cost_usd - policy.usage.cost_usd;
        let savings_fraction = (baseline.cost_usd > 0.0)
            .then(|| saved / baseline.cost_usd)
            .filter(|fraction| fraction.is_finite());
        json!({
            "threshold": policy.threshold,
            "small_requests": policy.small_requests,
            "usage": policy.usage,
            "saved_usd": saved,
            "savings_fraction": savings_fraction,
            "mean_quality": policy.usage.quality / samples_f64,
            "mean_quality_delta_vs_large": (policy.usage.quality - baseline.quality) / samples_f64,
            "mean_quality_regret": policy.regret / samples_f64,
            "quality_regressions": policy.degraded_requests,
            "quality_improvements": policy.improved_requests,
            "worst_regressions": policy.worst,
        })
    }).collect();
    println!("{}", serde_json::to_string_pretty(&json!({
        "rates": rates,
        "samples": samples,
        "forced_large_requests": forced_large_requests,
        "always_large": {
            "usage": baseline,
            "mean_quality": baseline.quality / samples_f64,
        },
        "policies": reports,
    }))?);
    Ok(())
}
