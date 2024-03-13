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

mod input;
mod parameters;

use pyo3::prelude::*;
use ark_std::test_rng;
use input::*;
use parameters::*;

const INPUT_DIMS: &[usize] = &[28, 28];
const INTER_DIM: usize = 28;
const OUTPUT_DIM: usize = 10;

fn generate_test_params() -> (Vec<Vec<Fr>>, Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>) {
    Python::with_gil(
        |py| -> PyResult<(Vec<Vec<Fr>>, Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>)> {
            Pymodule::from_code(
                py,
                include_str!("../two_layer_perceptron_mnist.py"),
                "two_layer_perceptron_mnist.py",
                "two_layer_perceptron_mnist",
            )?;
        }
    )
}
// TODO this is incorrect now that we have switched to logs
fn build_two_layer_perceptron_mnist<F, S, PCS>() -> Model<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let flat_dim = INPUT_DIMS.iter().product();

    let reshape: ReshapeNode<F, S, PCS> = ReshapeNode::new(INPUT_DIMS.to_vec(), vec![flat_dim]);

    let bmm_1: BMMNode<F, S, PCS> = BMMNode::new(
        WEIGHTS_1.to_vec(),
        BIAS_1.to_vec(),
        (flat_dim, INTER_DIM),
        Z_1_I,
    );

    let req_bmm_1: RequantiseBMMNode<F, S, PCS> =
        RequantiseBMMNode::new(INTER_DIM, S_1_I, Z_1_I, S_1_W, Z_1_W, S_1_O, Z_1_O);

    let relu: ReLUNode<F, S, PCS> = ReLUNode::new(28, Z_1_O);

    let bmm_2: BMMNode<F, S, PCS> = BMMNode::new(
        WEIGHTS_2.to_vec(),
        BIAS_2.to_vec(),
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

fn run_native_two_layer_perceptron_mnist() {
    /**** Change here ****/
    let input = NORMALISED_INPUT_TEST_150;
    let expected_output: Vec<u8> = vec![138, 106, 149, 160, 174, 152, 141, 146, 169, 207];
    /**********************/

    let perceptron = build_two_layer_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

    let quantised_input: QArray<u8> = input
        .iter()
        .map(|r| quantise_f32_u8_nne(r, S_INPUT, Z_INPUT))
        .collect::<Vec<Vec<u8>>>()
        .into();

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<QSmallType>();

    let output_i8 = perceptron.evaluate(input_i8);

    let output_u8 = (output_i8.cast::<i32>() + 128).cast::<u8>();

    println!("Output: {:?}", output_u8.values());
    assert_eq!(output_u8.move_values(), expected_output);
}

fn run_padded_two_layer_perceptron_mnist() {
    /**** Change here ****/
    let input = NORMALISED_INPUT_TEST_150;
    let expected_output: Vec<u8> = vec![138, 106, 149, 160, 174, 152, 141, 146, 169, 207];
    /**********************/

    let perceptron = build_two_layer_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

    let quantised_input: QArray<u8> = input
        .iter()
        .map(|r| quantise_f32_u8_nne(r, S_INPUT, Z_INPUT))
        .collect::<Vec<Vec<u8>>>()
        .into();

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<QSmallType>();

    let output_i8 = perceptron.padded_evaluate(input_i8);

    let output_u8 = (output_i8.cast::<i32>() + 128).cast::<u8>();

    println!("Output: {:?}", output_u8.values());
    assert_eq!(output_u8.move_values(), expected_output);
}

fn prove_inference_two_layer_perceptron_mnist() {
    /**** Change here ****/
    let input = NORMALISED_INPUT_TEST_150;
    let expected_output: Vec<u8> = vec![138, 106, 149, 160, 174, 152, 141, 146, 169, 207];
    /**********************/

    let perceptron = build_two_layer_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

    let quantised_input: QArray<u8> = input
        .iter()
        .map(|r| quantise_f32_u8_nne(r, S_INPUT, Z_INPUT))
        .collect::<Vec<Vec<u8>>>()
        .into();

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
    assert_eq!(output_u8.move_values()[0..OUTPUT_DIM], expected_output);
}

fn verify_inference_two_layer_perceptron_mnist() {
    /**** Change here ****/
    let input = NORMALISED_INPUT_TEST_150;
    let expected_output: Vec<u8> = vec![138, 106, 149, 160, 174, 152, 141, 146, 169, 207];
    /**********************/
    let perceptron = build_two_layer_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

    let quantised_input: QArray<u8> = input
        .iter()
        .map(|r| quantise_f32_u8_nne(r, S_INPUT, Z_INPUT))
        .collect::<Vec<Vec<u8>>>()
        .into();

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
    assert_eq!(output_u8.move_values()[0..OUTPUT_DIM], expected_output);
}

fn main() {
    run_native_two_layer_perceptron_mnist();
    run_padded_two_layer_perceptron_mnist();
    prove_inference_two_layer_perceptron_mnist();
    verify_inference_two_layer_perceptron_mnist();
}
