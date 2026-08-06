use std::env;

use bullet_lib::{
    game::{inputs::Chess768, outputs::OutputBuckets},
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader::DirectSequentialDataLoader, ValueTrainerBuilder},
};
use bulletformat::ChessBoard;

const HIDDEN: usize = 512;
const OUTPUT_BUCKETS: usize = 8;
const QA: i16 = 255;
const QB: i16 = 64;

/// NARC's output factorisation. Buckets 0..2 are queenless positions with
/// zero, one, or at least two rooks. Buckets 3..4 contain one queen, and
/// buckets 5..7 contain two or more queens split by rook count.
#[derive(Clone, Copy, Default)]
struct NarcMajorBuckets;

impl OutputBuckets<ChessBoard> for NarcMajorBuckets {
    const BUCKETS: usize = OUTPUT_BUCKETS;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let mut queens = 0u8;
        let mut rooks = 0u8;
        for (piece, _) in *pos {
            match piece & 7 {
                3 => rooks += 1,
                4 => queens += 1,
                _ => {}
            }
        }
        if queens == 0 {
            rooks.min(2)
        } else if queens == 1 {
            3 + rooks.min(1)
        } else {
            5 + if rooks == 0 { 0 } else if rooks <= 2 { 1 } else { 2 }
        }
    }
}

fn arg_usize(name: &str, default: usize) -> usize {
    env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn main() {
    let data_files = env::var("NARC_BULLET_FILES").expect("NARC_BULLET_FILES is required");
    let paths: Vec<String> = data_files
        .split('\n')
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect();
    assert!(!paths.is_empty(), "no bulletformat files were provided");
    let path_refs: Vec<&str> = paths.iter().map(String::as_str).collect();

    let superbatches = arg_usize("NARC_SUPERBATCHES", 24);
    let batches_per_superbatch = arg_usize("NARC_BATCHES_PER_SUPERBATCH", 1024);
    let batch_size = arg_usize("NARC_BATCH_SIZE", 16_384);
    let output_dir = env::var("NARC_CHECKPOINT_DIR").unwrap_or_else(|_| "/tmp/narc-checkpoints".to_string());
    let initial_lr = env::var("NARC_INITIAL_LR").ok().and_then(|value| value.parse().ok()).unwrap_or(0.001);
    let final_lr = env::var("NARC_FINAL_LR").ok().and_then(|value| value.parse().ok()).unwrap_or(0.00005);
    let wdl_value = env::var("NARC_WDL").ok().and_then(|value| value.parse().ok()).unwrap_or(0.30);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        .output_buckets(NarcMajorBuckets)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").round().quantise::<i32>(i32::from(QA) * i32::from(QB)),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", 768, HIDDEN);
            let l1 = builder.new_affine("l1", 2 * HIDDEN, OUTPUT_BUCKETS);
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            l1.forward(stm_hidden.concat(ntm_hidden)).select(output_buckets)
        });

    if let Ok(path) = env::var("NARC_INITIAL_WEIGHTS") {
        trainer.optimiser.load_weights_from_file(&path).expect("failed to load expanded champion weights");
        println!("Loaded expanded champion weights from {path}");
    }

    let schedule = TrainingSchedule {
        net_id: "narc-h512-major-wdl30".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_value },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr,
            final_lr,
            final_superbatch: superbatches,
        },
        save_rate: superbatches,
    };
    let settings = LocalSettings {
        threads: 6,
        test_set: None,
        output_directory: &output_dir,
        batch_queue_size: 64,
    };
    let dataloader = DirectSequentialDataLoader::new(&path_refs);
    trainer.run(&schedule, &settings, &dataloader);
}
