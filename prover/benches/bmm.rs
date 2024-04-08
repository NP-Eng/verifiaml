#![cfg(feature = "python")]
use ark_bn254::Fr;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hcs_common::{
    compatibility::example_models::fully_connected_layer::{
        build_fully_connected_layer_mnist,
        parameters::{S_INPUT, Z_INPUT},
    },
    get_model, get_model_output, quantise_f32_u8_nne, run_python, test_sponge, Ligero,
    NodeCommitment, NodeCommitmentState, QArray,
};
use hcs_prover::ProveModel;
use hcs_verifier::VerifyModel;

fn quantise_input(raw_input: &QArray<f32>) -> QArray<i8> {
    let quantised_input: QArray<u8> = QArray::new(
        quantise_f32_u8_nne(raw_input.values(), S_INPUT, Z_INPUT),
        raw_input.shape().clone(),
    );

    (quantised_input.cast::<i32>() - 128).cast::<i8>()
}

fn tf_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("tf_inference");

    run_python(|py| {
        let model = get_model(py, "QFullyConnectedLayer");
        group.bench_function(BenchmarkId::new("fully connected layer", "mnist"), |b| {
            b.iter(|| get_model_output(py, &model, None))
        });
    });
}

fn verifiaml_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("verifiaml_inference");
    let fc_model = build_fully_connected_layer_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();
    let raw_input = QArray::read("benches/data/input_test_150.json");

    // Quantisation happens in the tf inference benchmark, so we benchmark it here
    // too in order to make the comparison as fair as possible
    group.bench_function(BenchmarkId::new("fully connected layer", "mnist"), |b| {
        b.iter(|| {
            fc_model.evaluate(quantise_input(&raw_input));
        })
    });
}

fn verifiaml_proof(c: &mut Criterion) {
    let mut group = c.benchmark_group("verifiaml_proof");
    let fc_model = build_fully_connected_layer_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();
    let raw_input = QArray::read("benches/data/input_test_150.json");

    let mut sponge: PoseidonSponge<Fr> = test_sponge();
    let mut rng = test_rng();
    let (ck, _) = fc_model
        .setup_keys::<Fr, PoseidonSponge<Fr>, Ligero<Fr>, _>(&mut rng)
        .unwrap();

    let (node_coms, node_com_states): (
        Vec<NodeCommitment<Fr, PoseidonSponge<Fr>, Ligero<Fr>>>,
        Vec<NodeCommitmentState<Fr, PoseidonSponge<Fr>, Ligero<Fr>>>,
    ) = fc_model.commit(&ck, None).into_iter().unzip();

    group.bench_function(BenchmarkId::new("fully connected layer", "mnist"), |b| {
        b.iter(|| {
            // Quantisation happens in the tf inference benchmark, so we benchmark it here
            // too in order to make the comparison as fair as possible
            fc_model.prove_inference(
                &ck,
                Some(&mut rng),
                &mut sponge,
                &node_coms,
                &node_com_states,
                quantise_input(&raw_input),
            );
        })
    });
}

fn verifiaml_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("verifiaml_verification");
    let fc_model = build_fully_connected_layer_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();
    let raw_input = QArray::read("benches/data/input_test_150.json");

    let mut sponge: PoseidonSponge<Fr> = test_sponge();
    let mut rng = test_rng();
    let (ck, vk) = fc_model
        .setup_keys::<Fr, PoseidonSponge<Fr>, Ligero<Fr>, _>(&mut rng)
        .unwrap();

    let (node_coms, node_com_states): (
        Vec<NodeCommitment<Fr, PoseidonSponge<Fr>, Ligero<Fr>>>,
        Vec<NodeCommitmentState<Fr, PoseidonSponge<Fr>, Ligero<Fr>>>,
    ) = fc_model.commit(&ck, None).into_iter().unzip();

    let proof = fc_model.prove_inference(
        &ck,
        Some(&mut rng),
        &mut sponge,
        &node_coms,
        &node_com_states,
        quantise_input(&raw_input),
    );

    group.bench_function(BenchmarkId::new("fully connected layer", "mnist"), |b| {
        b.iter(|| {
            fc_model.verify_inference(&vk, &mut sponge, &node_coms, &proof);
        })
    });
}

criterion_group!(
    benches,
    tf_inference,
    verifiaml_inference,
    verifiaml_proof,
    verifiaml_verification
);
criterion_main!(benches);
