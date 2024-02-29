use hcs_common::{
    quantise_f32_u8_nne, test_sponge, BMMNode, Ligero, Model, Node, Poly, QArray, QSmallType,
    QTypeArray, ReLUNode, RequantiseBMMNode, ReshapeNode,
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
const INTER_DIM: usize = 28;
const OUTPUT_DIM: usize = 10;

macro_rules! PATH {
    () => {
        "prover/examples/two_layer_perceptron_mnist/{}.json"
    };
}

fn build_two_layer_perceptron_mnist<F, S, PCS>() -> Model<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let flat_dim = INPUT_DIMS.iter().product();

    let reshape: ReshapeNode<F, S, PCS> = ReshapeNode::new(INPUT_DIMS.to_vec(), vec![flat_dim]);

    let w1_array: QArray<i8> = QArray::read(&format!(PATH!(), "parameters/weights_1"));
    let b1_array: QArray<i32> = QArray::read(&format!(PATH!(), "parameters/bias_1"));
    let w2_array: QArray<i8> = QArray::read(&format!(PATH!(), "parameters/weights_2"));
    let b2_array: QArray<i32> = QArray::read(&format!(PATH!(), "parameters/bias_2"));

    let bmm_1: BMMNode<F, S, PCS> = BMMNode::new(
        w1_array.move_values(),
        b1_array.move_values(),
        (flat_dim, INTER_DIM),
        Z_1_I,
    );

    let req_bmm_1: RequantiseBMMNode<F, S, PCS> =
        RequantiseBMMNode::new(INTER_DIM, S_1_I, Z_1_I, S_1_W, Z_1_W, S_1_O, Z_1_O);

    let relu: ReLUNode<F, S, PCS> = ReLUNode::new(28, Z_1_O);

    let bmm_2: BMMNode<F, S, PCS> = BMMNode::new(
        w2_array.move_values(),
        b2_array.move_values(),
        (INTER_DIM, OUTPUT_DIM),
        Z_2_I,
    );

    let req_bmm_2: RequantiseBMMNode<F, S, PCS> =
        RequantiseBMMNode::new(OUTPUT_DIM, S_2_I, Z_2_I, S_2_W, Z_2_W, S_2_O, Z_2_O);

    Model::new(
        INPUT_DIMS.to_vec(),
        vec![
            Node::Reshape(reshape),
            Node::BMM(bmm_1),
            Node::RequantiseBMM(req_bmm_1),
            Node::ReLU(relu),
            Node::BMM(bmm_2),
            Node::RequantiseBMM(req_bmm_2),
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

    let output_i8 = perceptron.padded_evaluate(input_i8);

    (output_i8.cast::<i32>() + 128).cast()
}

fn run_unpadded_two_layer_perceptron_mnist() {
    let raw_input: QArray<f32> = QArray::read(&format!(PATH!(), "data/input_test_150"));
    let expected_output: QArray<u8> = QArray::read(&format!(PATH!(), "data/output_test_150"));

    let perceptron = build_two_layer_perceptron_mnist();
    let output_u8 = unpadded_inference(raw_input, &perceptron);

    println!("Output: {:?}", output_u8);
    assert_eq!(output_u8, expected_output);
}

fn run_padded_two_layer_perceptron_mnist() {
    let raw_input: QArray<f32> = QArray::read(&format!(PATH!(), "data/input_test_150"));
    let expected_output: QArray<u8> = QArray::read(&format!(PATH!(), "data/output_test_150"));

    let perceptron = build_two_layer_perceptron_mnist();
    let output_u8 = padded_inference(raw_input, &perceptron);

    println!("Output: {:?}", output_u8);
    assert_eq!(output_u8, expected_output);
}

fn prove_inference_two_layer_perceptron_mnist() {
    let input: QArray<f32> = QArray::read(&format!(PATH!(), "data/input_test_150"));
    let expected_output: QArray<u8> = QArray::read(&format!(PATH!(), "data/output_test_150"));

    let perceptron = build_two_layer_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

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

    let output_u8 = (output_i8.cast::<i32>() + 128).cast::<u8>();

    println!("Padded output: {:?}", output_u8.values());
    assert_eq!(
        output_u8.compact_resize(vec![OUTPUT_DIM], 0),
        expected_output
    );
}

fn verify_inference_two_layer_perceptron_mnist() {
    let input: QArray<f32> = QArray::read(&format!(PATH!(), "data/input_test_150"));
    let expected_output: QArray<u8> = QArray::read(&format!(PATH!(), "data/output_test_150"));

    let perceptron = build_two_layer_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

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

    let output_u8 = (output_i8.cast::<i32>() + 128).cast::<u8>();

    println!("Padded output: {:?}", output_u8.values());
    assert_eq!(
        output_u8.compact_resize(vec![OUTPUT_DIM], 0),
        expected_output
    );
}

fn main() {
    run_unpadded_two_layer_perceptron_mnist();
    run_padded_two_layer_perceptron_mnist();
    prove_inference_two_layer_perceptron_mnist();
    verify_inference_two_layer_perceptron_mnist();
}
