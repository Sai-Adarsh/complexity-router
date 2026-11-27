use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

struct Corpus(PathBuf);

impl Corpus {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "complexity-router-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn write(&self, name: &str, content: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, content).unwrap();
        path
    }
}

impl Drop for Corpus {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_complexity-router"))
        .env_remove("ROUTE_THRESHOLD")
        .args(args)
        .output()
        .unwrap()
}

fn successful(output: Output) -> String {
    assert!(output.status.success(), "{:?}", output);
    String::from_utf8(output.stdout).unwrap()
}

fn fails(output: Output, reason: &str) {
    assert!(!output.status.success(), "{:?}", output);
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains(reason), "{error:?} lacks {reason:?}");
}

fn report(records: &Path, rates: &Path, options: &[&str]) -> Output {
    let mut args = vec![
        "evaluate-paired",
        records.to_str().unwrap(),
        rates.to_str().unwrap(),
    ];
    args.extend_from_slice(options);
    cli(&args)
}

fn fixture_rates() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/prices.json")
}

fn paired_record() -> Value {
    serde_json::from_str(
        include_str!("../fixtures/paired.jsonl")
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn compares_usage_and_positive_regret_on_the_same_chat_requests() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/paired.jsonl");
    let value: Value = serde_json::from_str(&successful(report(
        &fixture,
        &fixture_rates(),
        &["8", "1.8", "1.8", "3.2"],
    )))
    .unwrap();
    assert_eq!(value["rates"]["data_kind"], "fixture");
    assert_eq!(value["samples"], 6);
    assert_eq!(value["forced_large_requests"], 2);
    assert_eq!(
        value["always_large"]["usage"],
        json!({
            "input_tokens": 721, "output_tokens": 561, "cost_usd": 0.004808
        })
    );
    let policies = value["policies"].as_array().unwrap();
    assert_eq!(policies.len(), 3);
    assert_eq!(policies[0]["threshold"], 1.8);
    assert_eq!(policies[0]["small_requests"], 1);
    assert_eq!(policies[0]["mean_quality_regret"], 0.0);
    assert_eq!(policies[1]["quality_improvements"], 1);
    assert_eq!(policies[1]["quality_regressions"], 1);
    assert!(
        policies[1]["mean_quality_delta_vs_large"].as_f64().unwrap()
            > -policies[1]["mean_quality_regret"].as_f64().unwrap()
    );
    assert_eq!(policies[2]["small_requests"], 4);
    assert_eq!(policies[2]["quality_regressions"], 2);
    assert_eq!(policies[2]["worst_regressions"][0]["id"], "arithmetic-trap");
    assert_eq!(policies[2]["worst_regressions"][1]["id"], "tradeoff");
    assert!((policies[2]["savings_fraction"].as_f64().unwrap() - 0.6043).abs() < 0.0001);
}

#[test]
fn reports_zero_baseline_without_dividing_by_zero() {
    let corpus = Corpus::new();
    let records = corpus.write("input.jsonl", &format!("\n{}\n", paired_record()));
    let mut rates: Value = serde_json::from_str(include_str!("../fixtures/prices.json")).unwrap();
    rates["data_kind"] = json!("measured");
    for model in ["small", "large"] {
        rates[model]["input_usd_per_million_tokens"] = json!(0);
        rates[model]["output_usd_per_million_tokens"] = json!(0);
    }
    let rates = corpus.write("rates.json", &rates.to_string());
    let value: Value = serde_json::from_str(&successful(report(&records, &rates, &[]))).unwrap();
    let policy = &value["policies"][0];
    assert_eq!(value["rates"]["data_kind"], "measured");
    assert_eq!(policy["threshold"], 3.2);
    assert_eq!(policy["saved_usd"], 0.0);
    assert!(policy["savings_fraction"].is_null());
}

#[test]
fn keeps_only_ten_worst_regressions_with_stable_order_for_equal_grades() {
    let corpus = Corpus::new();
    let mut rows = String::new();
    for index in (0..12).rev() {
        let mut record = paired_record();
        record["id"] = json!(format!("sample-{index:02}"));
        record["small"]["quality"] = json!(0.0);
        rows.push_str(&format!("{record}\n"));
    }
    let records = corpus.write("regressions.jsonl", &rows);
    let value: Value =
        serde_json::from_str(&successful(report(&records, &fixture_rates(), &[]))).unwrap();
    let policy = &value["policies"][0];
    assert_eq!(policy["quality_regressions"], 12);
    let worst = policy["worst_regressions"].as_array().unwrap();
    assert_eq!(worst.len(), 10);
    assert_eq!(worst[0]["id"], "sample-00");
    assert_eq!(worst[9]["id"], "sample-09");
}

#[test]
fn rejects_invalid_paired_records_with_source_line() {
    let corpus = Corpus::new();
    let good = paired_record();
    let rates = fixture_rates();
    let invalid = [
        ("/id", json!(" "), "IDs must be unique"),
        (
            "/small/model",
            json!("incorrect"),
            "expected model fixture-8b",
        ),
        ("/large/quality", json!(1.01), "quality must be in [0, 1]"),
        (
            "/small/input_tokens",
            json!(0),
            "input_tokens must be positive",
        ),
        ("/request/messages", json!([]), "Include a user message"),
    ];
    for (path, value, reason) in invalid {
        let mut record = good.clone();
        *record.pointer_mut(path).unwrap() = value;
        let records = corpus.write("invalid.jsonl", &format!("\n{record}\n"));
        let output = report(&records, &rates, &[]);
        let error = String::from_utf8(output.stderr.clone()).unwrap();
        assert!(error.contains("invalid.jsonl:2:"));
        fails(output, reason);
    }
    let duplicate = corpus.write("dup.jsonl", &format!("{good}\n{good}\n"));
    fails(report(&duplicate, &rates, &[]), "IDs must be unique");
    let invalid_json = corpus.write("broken.jsonl", "{");
    fails(report(&invalid_json, &rates, &[]), "broken.jsonl:1:");
    let empty = corpus.write("empty.jsonl", "\n");
    fails(report(&empty, &rates, &[]), "No paired records");
}

#[test]
fn rejects_invalid_prices_thresholds_and_token_charge_overflow() {
    let corpus = Corpus::new();
    let mut record = paired_record();
    let records = corpus.write("input.jsonl", &record.to_string());
    let valid_rates = fixture_rates();
    for threshold in ["0", "NaN", "-1"] {
        fails(
            report(&records, &valid_rates, &[threshold]),
            "finite and positive",
        );
    }
    fails(
        report(&records, &valid_rates, &["1"; 65]),
        "at most 64 thresholds",
    );
    let mut rates: Value = serde_json::from_str(include_str!("../fixtures/prices.json")).unwrap();
    rates["small"]["input_usd_per_million_tokens"] = json!(-1);
    let invalid = corpus.write("bad-rates.json", &rates.to_string());
    fails(report(&records, &invalid, &[]), "nonnegative token prices");

    rates["small"]["input_usd_per_million_tokens"] = json!(1.7e308);
    record["small"]["input_tokens"] = json!(2_000_000);
    let rates = corpus.write("overflow-rates.json", &rates.to_string());
    let overflow = corpus.write("overflow.jsonl", &record.to_string());
    fails(report(&overflow, &rates, &[]), "token charges overflowed");

    record["small"]["input_tokens"] = json!(1_000_000);
    let first = record.clone();
    record["id"] = json!("second");
    let overflow = corpus.write("aggregate.jsonl", &format!("{first}\n{record}\n"));
    fails(
        report(&overflow, &rates, &[]),
        "Aggregate token charges overflowed",
    );
}

#[test]
fn exposes_demo_and_legacy_labels() {
    let demo = successful(cli(&["demo"]));
    let decisions: Vec<Value> = demo
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(decisions[0]["decision"]["route"], "small");
    assert_eq!(decisions[1]["decision"]["route"], "large");
    let corpus = Corpus::new();
    let labels = corpus.write(
        "labels.jsonl",
        "\n{\"prompt\":\"hello\",\"route\":\"small\"}\n",
    );
    let output: Value =
        serde_json::from_str(&successful(cli(&["evaluate", labels.to_str().unwrap()]))).unwrap();
    assert_eq!(output["label_accuracy"], 1.0);
    let invalid = corpus.write("broken.jsonl", "{}");
    fails(
        cli(&["evaluate", invalid.to_str().unwrap()]),
        "broken.jsonl:1:",
    );
    let empty = corpus.write("empty.jsonl", "\n");
    fails(
        cli(&["evaluate", empty.to_str().unwrap()]),
        "No labeled prompts",
    );
    fails(cli(&["unknown"]), "usage:");
    let output = Command::new(env!("CARGO_BIN_EXE_complexity-router"))
        .env("ROUTE_THRESHOLD", "inf")
        .arg("demo")
        .output()
        .unwrap();
    fails(output, "ROUTE_THRESHOLD must be finite and positive");
}
