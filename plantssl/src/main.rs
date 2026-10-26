//! Command-line entry point. Run `plantssl --help` for the commands.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Serialize;
use serde_json::json;
use tracing::info;

use plantssl::data::{Dataset, Split, SplitConfig};
use plantssl::eval::{self, Report};
use plantssl::model::{ModelMeta, PlantCnn};
use plantssl::predict::Predictor;
use plantssl::ssl::{self, SslConfig};
use plantssl::train::{self, TrainConfig};

#[derive(Parser)]
#[command(version, about = "Semi-supervised plant disease classification")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Split a dataset into labeled, stream, validation and test pools.
    Split(SplitArgs),
    /// Train a supervised model on the labeled pool.
    Train(TrainArgs),
    /// Improve a trained model with pseudo-labels from the stream pool.
    Ssl(SslArgs),
    /// Score a model on the test or validation pool.
    Eval(EvalArgs),
    /// Classify image files.
    Predict(PredictArgs),
}

#[derive(Args, Serialize)]
struct SplitArgs {
    /// Dataset root, with class folders directly inside or inside train/ and valid/.
    #[arg(long, default_value = "data/plantvillage")]
    data: PathBuf,
    #[arg(long, default_value = "runs/split.json")]
    out: PathBuf,
    /// Share of all images that keeps its labels.
    #[arg(long, default_value_t = 0.2)]
    labeled: f64,
    #[arg(long, default_value_t = 0.1)]
    validation: f64,
    #[arg(long, default_value_t = 0.1)]
    test: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Args, Serialize)]
struct Common {
    #[arg(long, default_value = "runs/split.json")]
    split: PathBuf,
    /// cuda, wgpu or cpu.
    #[arg(long, default_value = plantssl::default_device_name())]
    device: String,
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Args, Serialize)]
struct TrainArgs {
    #[command(flatten)]
    common: Common,
    #[arg(long, default_value = "runs/supervised")]
    out: PathBuf,
    #[arg(long, default_value_t = 30)]
    epochs: usize,
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,
    #[arg(long, default_value_t = 128)]
    image_size: u32,
    /// Channels in the first convolution stage.
    #[arg(long, default_value_t = 32)]
    width: usize,
    /// Also train on the stream pool with its true labels: the fully supervised upper bound.
    #[arg(long)]
    with_stream_labels: bool,
}

#[derive(Args, Serialize)]
struct SslArgs {
    #[command(flatten)]
    common: Common,
    /// Model to start from, usually the output of `train`.
    #[arg(long, default_value = "runs/supervised")]
    init: PathBuf,
    #[arg(long, default_value = "runs/ssl")]
    out: PathBuf,
    /// The stream is revealed in this many equal slices.
    #[arg(long, default_value_t = 10)]
    rounds: usize,
    /// Minimum confidence for a pseudo-label.
    #[arg(long, default_value_t = 0.95)]
    threshold: f32,
    #[arg(long, default_value_t = 3)]
    epochs_per_round: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
}

#[derive(Clone, Copy, ValueEnum)]
enum Pool {
    Test,
    Validation,
}

#[derive(Args)]
struct EvalArgs {
    #[command(flatten)]
    common: Common,
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum, default_value = "test")]
    pool: Pool,
}

#[derive(Args)]
struct PredictArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(required = true)]
    images: Vec<PathBuf>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            // Burn's kernel compiler logs its IR at info level, so only our own logs default on
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,plantssl=info".into()),
        )
        .with_target(false)
        .init();

    match Cli::parse().command {
        Command::Split(args) => split(args),
        Command::Train(args) => train(args),
        Command::Ssl(args) => ssl(args),
        Command::Eval(args) => eval(args),
        Command::Predict(args) => predict(args),
    }
}

fn split(args: SplitArgs) -> Result<()> {
    let dataset = Dataset::scan(&args.data)?;
    let split = Split::new(
        &dataset,
        SplitConfig {
            labeled: args.labeled,
            validation: args.validation,
            test: args.test,
            seed: args.seed,
        },
    )?;
    split.save(&args.out)?;
    info!(
        "{} images in {} classes: {} labeled, {} stream, {} validation, {} test. Saved to {}",
        dataset.samples.len(),
        split.classes.len(),
        split.labeled.len(),
        split.stream.len(),
        split.validation.len(),
        split.test.len(),
        args.out.display()
    );
    Ok(())
}

fn train(args: TrainArgs) -> Result<()> {
    let c = &args.common;
    let split = Split::load(&c.split)?;
    let device = plantssl::device(&c.device)?;
    let mut train_set = split.resolve(&split.labeled);
    if args.with_stream_labels {
        train_set.extend(split.resolve(&split.stream));
    }
    info!("training on {} labeled images", train_set.len());

    let meta = ModelMeta {
        classes: split.classes.clone(),
        image_size: args.image_size,
        width: args.width,
    };
    let config = TrainConfig {
        epochs: args.epochs,
        batch_size: c.batch_size,
        learning_rate: args.lr,
        image_size: args.image_size,
        seed: c.seed,
        ..TrainConfig::default()
    };
    let fit = train::fit(
        PlantCnn::new(&meta, &device),
        &train_set,
        &split.resolve(&split.validation),
        &config,
        &device,
    )?;
    fit.model.save(&args.out, &meta)?;
    plantssl::write_json(
        &args.out.join("train.json"),
        &json!({ "args": args, "best_epoch": fit.best_epoch, "epochs": fit.history }),
    )?;
    info!(
        "best validation accuracy {:.2}% at epoch {}. Saved to {}",
        fit.best_val_accuracy * 100.0,
        fit.best_epoch,
        args.out.display()
    );
    Ok(())
}

fn ssl(args: SslArgs) -> Result<()> {
    let c = &args.common;
    let split = Split::load(&c.split)?;
    let device = plantssl::device(&c.device)?;
    let (model, meta) = PlantCnn::load(&args.init, &device)?;
    let config = SslConfig {
        rounds: args.rounds,
        threshold: args.threshold,
        train: TrainConfig {
            epochs: args.epochs_per_round,
            batch_size: c.batch_size,
            learning_rate: args.lr,
            warmup_epochs: 0,
            image_size: meta.image_size,
            seed: c.seed,
            ..TrainConfig::default()
        },
    };
    let run = ssl::run(model, &split, &config, &device)?;
    run.model.save(&args.out, &meta)?;
    plantssl::write_json(
        &args.out.join("ssl.json"),
        &json!({ "args": args, "best_round": run.best_round, "rounds": run.rounds }),
    )?;
    info!(
        "kept round {}. Saved to {}",
        run.best_round,
        args.out.display()
    );
    Ok(())
}

fn eval(args: EvalArgs) -> Result<()> {
    let c = &args.common;
    let split = Split::load(&c.split)?;
    let device = plantssl::device(&c.device)?;
    let (model, meta) = PlantCnn::load(&args.model, &device)?;
    let (name, pool) = match args.pool {
        Pool::Test => ("test", &split.test),
        Pool::Validation => ("validation", &split.validation),
    };
    let samples = split.resolve(pool);
    let predictions = eval::predict(&model, &samples, meta.image_size, c.batch_size * 2, &device)?;
    let truth: Vec<usize> = samples.iter().map(|(_, label)| *label).collect();
    let predicted: Vec<usize> = predictions.iter().map(|p| p.label).collect();
    let report = Report::new(&split.classes, &truth, &predicted);

    let out = args.model.join(format!("eval-{name}.json"));
    plantssl::write_json(&out, &report)?;
    info!(
        "{name}: {} images, accuracy {:.2}%, macro F1 {:.2}%",
        report.images,
        report.accuracy * 100.0,
        report.macro_f1 * 100.0
    );
    let mut worst = report.classes.clone();
    worst.sort_by(|a, b| a.f1.total_cmp(&b.f1));
    for class in worst.iter().take(5) {
        info!("  weakest: {:<50} F1 {:.2}%", class.class, class.f1 * 100.0);
    }
    info!("full report in {}", out.display());
    Ok(())
}

fn predict(args: PredictArgs) -> Result<()> {
    let predictor = Predictor::load(&args.model, plantssl::device(&args.device)?)?;
    for path in &args.images {
        let image = image::open(path)?;
        let scores = predictor.classify(&image);
        println!("{}", path.display());
        for score in scores.iter().take(3) {
            println!("  {:>6.2}%  {}", score.probability * 100.0, score.class);
        }
    }
    Ok(())
}
