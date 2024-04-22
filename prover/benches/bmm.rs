#![cfg(feature = "python")]

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hcs_common::{
    python::*, quantise_f32_u8_nne, test_sponge, BMMNode, Ligero, Model, Node, NodeCommitment,
    NodeCommitmentState, Poly, QArray, RequantiseBMMNode,
};
use hcs_prover::ProveModel;
use hcs_verifier::VerifyModel;
use pyo3::Python;

const SAMPLE_SIZE: usize = 10;

pub const S_INPUT: f32 = 0.003921568859368563;
pub const Z_INPUT: u8 = 0;

const S_I: f32 = 0.003921568859368563;
const Z_I: i8 = -128;
const S_W: f32 = 0.012436429969966412;
const Z_W: i8 = 0;
const S_O: f32 = 0.1573459506034851;
const Z_O: i8 = 47;

macro_rules! PATH {
    () => {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/benches/parameters/fully_connected_layer/{}"
        )
    };
}

fn build_fully_connected_layer_mnist<F, S, PCS>(resize_factor: usize) -> Model<i8, i32>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let w_array: QArray<i8> = QArray::read(&format!(PATH!(), "weights.json"));
    let b_array: QArray<i32> = QArray::read(&format!(PATH!(), "bias.json"));

    let bmm: BMMNode<i8, i32> = BMMNode::new(w_array, b_array, Z_I);

    let req_bmm: RequantiseBMMNode<i8> = RequantiseBMMNode::new(10, S_I, Z_I, S_W, Z_W, S_O, Z_O);

    Model::new(
        vec![resize_factor as usize * 28 * 28],
        vec![Node::BMM(bmm), Node::RequantiseBMM(req_bmm)],
    )
}

fn quantise_input(raw_input: &QArray<f32>) -> QArray<i8> {
    let quantised_input: QArray<u8> = QArray::new(
        quantise_f32_u8_nne(raw_input.values(), S_INPUT, Z_INPUT),
        raw_input.shape().clone(),
    );

    (quantised_input.cast::<i32>() - 128).cast::<i8>()
}

fn bench_fully_connected_layer(c: &mut Criterion) {
    for resize_factor in (1..=16).step_by(3) {
        let resize_factor_str = resize_factor.to_string();
        let args: Vec<(&str, &str)> = vec![
            ("resize_factor", &resize_factor_str),
            ("overwrite_cache", "True"),
        ];
        bench_tf_inference(c, resize_factor, args.clone());

        let fc_model = build_fully_connected_layer_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
            resize_factor * resize_factor,
        );
        let raw_input = Python::with_gil(|py| {
            get_model_input::<Vec<f32>>(
                py,
                &get_model(py, "QFullyConnectedLayer", Some(args[..1].to_vec())),
                None,
            )
        });

        bench_verifiaml_inference(c, &fc_model, &raw_input, resize_factor);

        let mut sponge: PoseidonSponge<Fr> = test_sponge();
        let mut rng = test_rng();
        let (ck, vk) = fc_model
            .setup_keys::<Fr, PoseidonSponge<Fr>, Ligero<Fr>, _>(&mut rng)
            .unwrap();

        let (node_coms, node_coms_states) =
            bench_verifiaml_proof(c, &fc_model, &raw_input, &ck, &mut sponge, resize_factor);

        bench_verifiaml_verification::<Ligero<Fr>, PoseidonSponge<Fr>>(
            c,
            &fc_model,
            &ck,
            &vk,
            &node_coms,
            &node_coms_states,
            &raw_input,
            &mut sponge,
            resize_factor,
        );
    }
}

fn bench_tf_inference(c: &mut Criterion, resize_factor: usize, args: Vec<(&str, &str)>) {
    let mut group = c.benchmark_group("TensorFlow");
    group.sample_size(SAMPLE_SIZE);

    Python::with_gil(|py| {
        let model = get_model(py, "QFullyConnectedLayer", Some(args.clone()));
        save_model_parameters_as_qarray(py, &model, &format!(PATH!(), ""));
        group.bench_function(
            BenchmarkId::new(
                "inference",
                format!("{} params", resize_factor * resize_factor * 28 * 28 * 10),
            ),
            |b| b.iter(|| get_model_output(py, &model, None)),
        );
    });
}

fn bench_verifiaml_inference(
    c: &mut Criterion,
    model: &Model<i8, i32>,
    raw_input: &QArray<f32>,
    resize_factor: usize,
) {
    let mut group = c.benchmark_group("verifiaml");
    group.sample_size(SAMPLE_SIZE);

    // Quantisation happens in the tf inference benchmark, so we benchmark it here
    // too in order to make the comparison as fair as possible
    group.bench_function(
        BenchmarkId::new(
            "inference",
            format!("{} params", resize_factor * resize_factor * 28 * 28 * 10),
        ),
        |b| b.iter(|| model.evaluate(quantise_input(&raw_input))),
    );
}

fn bench_verifiaml_proof<PCS, S>(
    c: &mut Criterion,
    model: &Model<i8, i32>,
    raw_input: &QArray<f32>,
    ck: &PCS::CommitterKey,
    sponge: &mut S,
    resize_factor: usize,
) -> (
    Vec<NodeCommitment<Fr, S, PCS>>,
    Vec<NodeCommitmentState<Fr, S, PCS>>,
)
where
    S: CryptographicSponge,
    PCS: PolynomialCommitment<Fr, Poly<Fr>, S>,
{
    let mut group = c.benchmark_group("verifiaml");
    group.sample_size(SAMPLE_SIZE);

    let (node_coms, node_com_states): (
        Vec<NodeCommitment<Fr, S, PCS>>,
        Vec<NodeCommitmentState<Fr, S, PCS>>,
    ) = model.commit(ck, None).into_iter().unzip();

    let mut rng = test_rng();

    group.bench_function(
        BenchmarkId::new(
            "proof",
            format!("{} params", resize_factor * resize_factor * 28 * 28 * 10),
        ),
        |b| {
            b.iter(|| {
                // Quantisation happens in the tf inference benchmark, so we benchmark it here
                // too in order to make the comparison as fair as possible
                model.prove_inference(
                    ck,
                    Some(&mut rng),
                    sponge,
                    &node_coms,
                    &node_com_states,
                    quantise_input(&raw_input),
                );
            })
        },
    );

    (node_coms, node_com_states)
}

fn bench_verifiaml_verification<PCS, S>(
    c: &mut Criterion,
    model: &Model<i8, i32>,
    ck: &PCS::CommitterKey,
    vk: &PCS::VerifierKey,
    node_coms: &Vec<NodeCommitment<Fr, S, PCS>>,
    node_com_states: &Vec<NodeCommitmentState<Fr, S, PCS>>,
    raw_input: &QArray<f32>,
    sponge: &mut S,
    resize_factor: usize,
) where
    S: CryptographicSponge,
    PCS: PolynomialCommitment<Fr, Poly<Fr>, S>,
{
    let mut group = c.benchmark_group("verifiaml");
    group.sample_size(SAMPLE_SIZE);

    let mut rng = test_rng();

    group.bench_function(
        BenchmarkId::new(
            "verification",
            format!("{} params", resize_factor * resize_factor * 28 * 28 * 10),
        ),
        |b| {
            b.iter_batched(
                || {
                    model.prove_inference(
                        ck,
                        Some(&mut rng),
                        &mut sponge.clone(),
                        node_coms,
                        node_com_states,
                        quantise_input(&raw_input),
                    )
                },
                |proof| {
                    model.verify_inference(vk, &mut sponge.clone(), node_coms, proof);
                },
                criterion::BatchSize::SmallInput,
            )
        },
    );
}

criterion_group!(benches, bench_fully_connected_layer);
criterion_main!(benches);
