use hcs_common::{
    quantise_f32_u8_nne, test_sponge, BMMNode, Ligero, Model, Node, Poly, QArray, QSmallType,
    QTypeArray, RequantiseBMMNode, ReshapeNode,
};
use hcs_prover::ProveModel;
use hcs_verifier::VerifyModel;

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::test_rng;

mod parameters;

use parameters::*;

const INPUT_DIMS: &[usize] = &[28, 28];
const OUTPUT_DIM: usize = 10;

// This is the cleaner way to format a fixed string with various data due to
// the time at which Rust expands macros
macro_rules! PATH {
    () => {
        "prover/examples/simple_perceptron_mnist/{}"
    };
}

// TODO this is incorrect now that we have switched to logs
fn build_simple_perceptron_mnist<F, S, PCS>() -> Model<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let flat_dim = INPUT_DIMS.iter().product();

    let reshape: ReshapeNode<F, S, PCS> = ReshapeNode::new(INPUT_DIMS.to_vec(), vec![flat_dim]);

    let w_array: QArray<i8> = QArray::read(&format!(PATH!(), "parameters/weights.json"));
    let b_array: QArray<i32> = QArray::read(&format!(PATH!(), "parameters/bias.json"));

    let bmm: BMMNode<F, S, PCS> = BMMNode::new(w_array, b_array, (flat_dim, OUTPUT_DIM), Z_I);

    let req_bmm: RequantiseBMMNode<F, S, PCS> =
        RequantiseBMMNode::new(OUTPUT_DIM, S_I, Z_I, S_W, Z_W, S_O, Z_O);

    Model::new(
        INPUT_DIMS.to_vec(),
        vec![
            Node::Reshape(reshape),
            Node::BMM(bmm),
            Node::RequantiseBMM(req_bmm),
        ],
    )
}

// Auxiliary function
fn unpadded_inference(
    raw_input: QArray<f32>,
    perceptron: &Model<Fr, PoseidonSponge<Fr>, Ligero<Fr>>,
) -> QArray<u8> {
    let quantised_input: QArray<u8> = QArray::new(
        quantise_f32_u8_nne(raw_input.values(), S_INPUT, Z_INPUT),
        raw_input.shape().clone(),
    );

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<QSmallType>();

    let output_i8 = perceptron.evaluate(input_i8);

    (output_i8.cast::<i32>() + 128).cast()
}

// Auxiliary function
fn padded_inference(
    raw_input: QArray<f32>,
    perceptron: &Model<Fr, PoseidonSponge<Fr>, Ligero<Fr>>,
) -> QArray<u8> {
    let quantised_input: QArray<u8> = QArray::new(
        quantise_f32_u8_nne(raw_input.values(), S_INPUT, Z_INPUT),
        raw_input.shape().clone(),
    );

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<QSmallType>();

    let output_i8 = perceptron.evaluate(input_i8);

    (output_i8.cast::<i32>() + 128).cast()
}

fn run_unpadded_simple_perceptron_mnist() {
    let raw_input: QArray<f32> = QArray::read(&format!(PATH!(), "data/input_test_150.json"));
    let expected_output: QArray<u8> = QArray::read(&format!(PATH!(), "data/output_test_150.json"));

    let perceptron = build_simple_perceptron_mnist();
    let output_u8 = unpadded_inference(raw_input, &perceptron);

    println!("Output: {:?}", output_u8);
    assert_eq!(output_u8, expected_output);
}

fn run_padded_simple_perceptron_mnist() {
    let raw_input: QArray<f32> = QArray::read(&format!(PATH!(), "data/input_test_150.json"));
    let expected_output: QArray<u8> = QArray::read(&format!(PATH!(), "data/output_test_150.json"));

    let perceptron = build_simple_perceptron_mnist();
    let output_u8 = padded_inference(raw_input, &perceptron);

    println!("Output: {:?}", output_u8);
    assert_eq!(output_u8, expected_output);
}

fn multi_run_unpadded_simple_perceptron_mnist() {
    let perceptron = build_simple_perceptron_mnist();

    // Mnist test samples with index
    // 6393, 1894, 5978, 6120, 817, 3843, 7626, 9272, 498, 4622
    let raw_inputs: Vec<QArray<f32>> =
        QArray::read_list(&format!(PATH!(), "data/10_test_inputs.json"));
    let expected_outputs: Vec<QArray<u8>> =
        QArray::read_list(&format!(PATH!(), "data/10_test_outputs.json"));

    for (raw_input, expected_output) in raw_inputs.into_iter().zip(expected_outputs.into_iter()) {
        assert_eq!(unpadded_inference(raw_input, &perceptron), expected_output);
    }

    println!("Unpadded compatibility test successful");
}

fn multi_run_padded_simple_perceptron_mnist() {
    let perceptron = build_simple_perceptron_mnist();

    // Mnist test samples with index
    // 6393, 1894, 5978, 6120, 817, 3843, 7626, 9272, 498, 4622
    let raw_inputs: Vec<QArray<f32>> =
        QArray::read_list(&format!(PATH!(), "data/10_test_inputs.json"));
    let expected_outputs: Vec<QArray<u8>> =
        QArray::read_list(&format!(PATH!(), "data/10_test_outputs.json"));

    for (raw_input, expected_output) in raw_inputs.into_iter().zip(expected_outputs.into_iter()) {
        assert_eq!(padded_inference(raw_input, &perceptron), expected_output);
    }

    println!("Padded compatibility test successful");
}

fn prove_inference_simple_perceptron_mnist() {
    let input: QArray<f32> = QArray::read(&format!(PATH!(), "data/input_test_150.json"));
    let expected_output: QArray<u8> = QArray::read(&format!(PATH!(), "data/output_test_150.json"));

    let perceptron = build_simple_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

    let quantised_input: QArray<u8> = QArray::new(
        quantise_f32_u8_nne(input.values(), S_INPUT, Z_INPUT),
        input.shape().clone(),
    );

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<QSmallType>();

    let mut rng = test_rng();
    let (ck, _) = perceptron.setup_keys(&mut rng).unwrap();

    let mut sponge: PoseidonSponge<Fr> = test_sponge();

    let (node_coms, node_com_states): (Vec<_>, Vec<_>) =
        perceptron.commit(&ck, None).into_iter().unzip();

    let inference_proof = perceptron.prove_inference(
        &ck,
        Some(&mut rng),
        &mut sponge,
        &node_coms,
        &node_com_states,
        input_i8,
    );

    let output_qtypearray = inference_proof.inputs_outputs[1].clone();

    let output_i8 = match output_qtypearray {
        QTypeArray::S(o) => o,
        _ => panic!("Expected QTypeArray::S"),
    };

    let output_u8: QArray<u8> = (output_i8.cast::<i32>() + 128).cast();

    println!("Padded output: {:?}", output_u8.values());
    assert_eq!(
        output_u8.compact_resize(vec![OUTPUT_DIM], 0),
        expected_output
    );
}

fn verify_inference_simple_perceptron_mnist() {
    let input: QArray<f32> = QArray::read(&format!(PATH!(), "data/input_test_150.json"));
    let expected_output: QArray<u8> = QArray::read(&format!(PATH!(), "data/output_test_150.json"));

    let perceptron = build_simple_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

    let quantised_input: QArray<u8> = QArray::new(
        quantise_f32_u8_nne(input.values(), S_INPUT, Z_INPUT),
        input.shape().clone(),
    );

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<QSmallType>();

    let mut rng = test_rng();
    let (ck, vk) = perceptron.setup_keys(&mut rng).unwrap();

    let mut sponge: PoseidonSponge<Fr> = test_sponge();

    let (node_coms, node_com_states): (Vec<_>, Vec<_>) =
        perceptron.commit(&ck, None).into_iter().unzip();

    let inference_proof = perceptron.prove_inference(
        &ck,
        Some(&mut rng),
        &mut sponge,
        &node_coms,
        &node_com_states,
        input_i8,
    );

    let output_qtypearray = inference_proof.inputs_outputs[1].clone();

    let mut sponge: PoseidonSponge<Fr> = test_sponge();

    assert!(perceptron.verify_inference(&vk, &mut sponge, &node_coms, inference_proof));

    let output_i8 = match output_qtypearray {
        QTypeArray::S(o) => o,
        _ => panic!("Expected QTypeArray::S"),
    };

    let output_u8: QArray<u8> = (output_i8.cast::<i32>() + 128).cast();

    println!("Padded output: {:?}", output_u8.values());
    assert_eq!(
        output_u8.compact_resize(vec![OUTPUT_DIM], 0),
        expected_output
    );
}

fn main() {
    run_unpadded_simple_perceptron_mnist();
    run_padded_simple_perceptron_mnist();
    multi_run_unpadded_simple_perceptron_mnist();
    multi_run_padded_simple_perceptron_mnist();
    prove_inference_simple_perceptron_mnist();
    verify_inference_simple_perceptron_mnist();
}
